use std::{
    collections::HashMap,
    env, io,
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
        mpsc::TryRecvError,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const PREDICTION_RETURN_RECONNECT_WAIT: Duration = Duration::from_secs(5);

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{
        StageReply, StageReplyEnvelope, StageReplyIdentity, StageStateHeader, StageWireMessage,
        WireActivationDType, WireMessageKind, WireReplyKind, read_stage_message, recv_ready,
        recv_reply_envelope, send_ready, send_reply_envelope, state_flags, write_stage_message,
    },
};

use super::{
    DirectReturnDelay,
    socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint},
};
use super::{consume_optional_client_ready_hello, send_client_ready_hello_if_enabled};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PredictionReturnKey {
    request_id: u64,
    session_id: u64,
}

impl PredictionReturnKey {
    pub(crate) fn new(request_id: u64, session_id: u64) -> Self {
        Self {
            request_id,
            session_id,
        }
    }
}

pub struct PredictionReturnHub {
    waiters: Mutex<HashMap<PredictionReturnKey, Arc<PredictionReturnWaiter>>>,
}

#[derive(Default)]
pub(crate) struct PredictionReturnSinks {
    streams: Mutex<HashMap<PredictionReturnKey, TcpStream>>,
}

struct PredictionReturnWaiter {
    sender: mpsc::Sender<PredictionReturnEvent>,
    stream_generation: AtomicU64,
    reopener: Mutex<Option<PredictionReturnReopener>>,
}

impl PredictionReturnWaiter {
    fn new(sender: mpsc::Sender<PredictionReturnEvent>) -> Self {
        Self {
            sender,
            stream_generation: AtomicU64::new(0),
            reopener: Mutex::new(None),
        }
    }

    fn begin_stream_generation(&self) -> u64 {
        self.stream_generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn is_current_generation(&self, generation: u64) -> bool {
        self.stream_generation.load(Ordering::SeqCst) == generation
    }

    fn send_if_current(&self, generation: u64, event: PredictionReturnEvent) -> bool {
        if !self.is_current_generation(generation) {
            return true;
        }
        self.sender.send(event).is_ok()
    }

    fn set_reopener(&self, reopener: PredictionReturnReopener) -> Result<()> {
        *self
            .reopener
            .lock()
            .map_err(|_| anyhow!("prediction return reopener lock poisoned"))? = Some(reopener);
        Ok(())
    }

    fn reopen_stream(&self) -> Result<bool> {
        let reopener = self
            .reopener
            .lock()
            .map_err(|_| anyhow!("prediction return reopener lock poisoned"))?
            .clone();
        let Some(reopener) = reopener else {
            return Ok(false);
        };
        eprintln!(
            "direct prediction return receiver reopening: request_id={} session_id={}",
            reopener.key.request_id, reopener.key.session_id
        );
        let stream = reopener.open_stream()?;
        eprintln!(
            "direct prediction return receiver reopened: request_id={} session_id={}",
            reopener.key.request_id, reopener.key.session_id
        );
        let key = reopener.key;
        let waiter = reopener.waiter.clone();
        thread::spawn(move || {
            if let Err(error) = handle_return_stream(key, waiter, stream) {
                eprintln!("direct prediction return reader failed after reconnect: {error:#}");
            }
        });
        Ok(true)
    }
}

enum PredictionReturnEvent {
    Reply(Box<StageReplyEnvelope>),
    StreamEnded(String),
}

#[derive(Clone)]
struct PredictionReturnReopener {
    key: PredictionReturnKey,
    waiter: Arc<PredictionReturnWaiter>,
    config: StageConfig,
    wire_dtype: WireActivationDType,
}

impl PredictionReturnReopener {
    fn open_stream(&self) -> Result<TcpStream> {
        open_downstream_prediction_return_stream(
            &self.config,
            self.key.request_id,
            self.key.session_id,
            self.wire_dtype,
        )
        .context("reopen downstream direct prediction return stream")
    }
}

impl Default for PredictionReturnHub {
    fn default() -> Self {
        Self {
            waiters: Mutex::new(HashMap::new()),
        }
    }
}

pub struct PredictionReturnListener {
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    hub: Arc<PredictionReturnHub>,
}

impl PredictionReturnListener {
    pub fn start(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind direct prediction return listener {bind_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("set direct prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let hub = Arc::new(PredictionReturnHub::default());
        let thread_hub = hub.clone();
        let thread = thread::spawn(move || {
            while !thread_shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(error) = stream.set_nonblocking(false) {
                            eprintln!(
                                "direct prediction return connection failed: set blocking: {error}"
                            );
                            continue;
                        }
                        let hub = thread_hub.clone();
                        thread::spawn(move || {
                            if let Err(error) = handle_prediction_return_connection(hub, stream) {
                                eprintln!("direct prediction return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        eprintln!("direct prediction return listener failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            shutdown,
            thread: Some(thread),
            hub,
        })
    }

    pub fn hub(&self) -> Arc<PredictionReturnHub> {
        self.hub.clone()
    }
}

impl Drop for PredictionReturnListener {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle_prediction_return_connection(
    hub: Arc<PredictionReturnHub>,
    mut stream: TcpStream,
) -> Result<()> {
    consume_optional_client_ready_hello(&mut stream)
        .context("consume optional direct prediction return client ready hello")?;
    send_ready(&mut stream).context("send direct prediction return ready")?;
    let open = read_stage_message(&mut stream, 0).context("read direct prediction return open")?;
    hub.handle_return_connection(open, stream)
}

impl PredictionReturnHub {
    pub(crate) fn register(
        self: &Arc<Self>,
        request_id: u64,
        session_id: u64,
    ) -> Result<PredictionReturnReceiver> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let (sender, receiver) = mpsc::channel();
        let waiter = Arc::new(PredictionReturnWaiter::new(sender));
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .insert(key, waiter.clone());
        Ok(PredictionReturnReceiver {
            key,
            waiter,
            hub: self.clone(),
            receiver,
        })
    }

    fn unregister(&self, key: PredictionReturnKey, waiter: &Arc<PredictionReturnWaiter>) {
        if let Ok(mut waiters) = self.waiters.lock() {
            let remove = waiters
                .get(&key)
                .is_some_and(|registered| Arc::ptr_eq(registered, waiter));
            if remove {
                waiters.remove(&key);
            }
        }
    }

    pub(crate) fn handle_return_connection(
        &self,
        open: StageWireMessage,
        stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        let waiter = self.lookup_waiter(key)?;
        handle_return_stream(key, waiter, stream)
    }

    fn lookup_waiter(&self, key: PredictionReturnKey) -> Result<Arc<PredictionReturnWaiter>> {
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("no prediction return waiter for request {}", key.request_id))
    }
}

pub(crate) struct PredictionReturnReceiver {
    key: PredictionReturnKey,
    waiter: Arc<PredictionReturnWaiter>,
    hub: Arc<PredictionReturnHub>,
    receiver: mpsc::Receiver<PredictionReturnEvent>,
}

impl PredictionReturnReceiver {
    pub(crate) fn enable_downstream_reconnect(
        &self,
        config: &StageConfig,
        wire_dtype: WireActivationDType,
    ) -> Result<()> {
        self.waiter.set_reopener(PredictionReturnReopener {
            key: self.key,
            waiter: self.waiter.clone(),
            config: config.clone(),
            wire_dtype,
        })
    }

    pub(crate) fn attach_opened_stream(&self, stream: TcpStream) {
        let key = self.key;
        let waiter = self.waiter.clone();
        thread::spawn(move || {
            if let Err(error) = handle_return_stream(key, waiter, stream) {
                eprintln!("direct prediction return reader failed: {error:#}");
            }
        });
    }

    pub(crate) fn try_recv_expected(
        &self,
        expected: WireReplyKind,
    ) -> Result<Option<StageReplyEnvelope>> {
        let Some(envelope) = self.try_recv()? else {
            return Ok(None);
        };
        if envelope.reply.kind != expected {
            bail!(
                "expected {expected:?} direct prediction return, got {:?}",
                envelope.reply.kind
            );
        }
        Ok(Some(envelope))
    }

    fn try_recv(&self) -> Result<Option<StageReplyEnvelope>> {
        match self.receiver.try_recv() {
            Ok(PredictionReturnEvent::Reply(reply)) => Ok(Some(*reply)),
            Ok(PredictionReturnEvent::StreamEnded(error)) => {
                if self.waiter.reopen_stream()? {
                    return Ok(None);
                }
                Err(anyhow!(error))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(anyhow!("prediction return channel disconnected"))
            }
        }
    }
}

impl Drop for PredictionReturnReceiver {
    fn drop(&mut self) {
        self.hub.unregister(self.key, &self.waiter);
    }
}

fn handle_return_stream(
    key: PredictionReturnKey,
    waiter: Arc<PredictionReturnWaiter>,
    mut stream: TcpStream,
) -> Result<()> {
    let generation = waiter.begin_stream_generation();
    loop {
        match recv_reply_envelope(&mut stream) {
            Ok(envelope) => {
                if envelope.reply.kind == WireReplyKind::PredictionReturnReconnect {
                    let _ = waiter.send_if_current(
                        generation,
                        PredictionReturnEvent::StreamEnded(
                            "direct prediction return reconnect requested".to_string(),
                        ),
                    );
                    return Ok(());
                }
                if !waiter
                    .send_if_current(generation, PredictionReturnEvent::Reply(Box::new(envelope)))
                {
                    return Ok(());
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                let _ = waiter.send_if_current(
                    generation,
                    PredictionReturnEvent::StreamEnded(
                        "direct prediction return stream closed".to_string(),
                    ),
                );
                return Ok(());
            }
            Err(error) => {
                let _ = waiter.send_if_current(
                    generation,
                    PredictionReturnEvent::StreamEnded(error.to_string()),
                );
                return Err(error).with_context(|| {
                    format!(
                        "read direct prediction return for request {} session {}",
                        key.request_id, key.session_id
                    )
                });
            }
        }
    }
}

pub(crate) struct DirectPredictionReturnWriter {
    sender: mpsc::Sender<QueuedPredictionReturn>,
}

struct QueuedPredictionReturn {
    ready_at: Instant,
    envelope: StageReplyEnvelope,
}

#[derive(Clone)]
struct PredictionReturnReconnect {
    key: PredictionReturnKey,
    sinks: Arc<PredictionReturnSinks>,
    wait_timeout: Duration,
}

#[derive(Clone, Copy, Default)]
struct DirectReturnReconnectFault {
    every_successful_sends: Option<usize>,
}

impl DirectReturnReconnectFault {
    fn from_env() -> Self {
        Self {
            every_successful_sends: env::var("SKIPPY_SPEC_RETURN_RECONNECT_EVERY")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0),
        }
    }

    fn should_reconnect_before_send(self, successful_sends: usize) -> bool {
        self.every_successful_sends
            .is_some_and(|every| successful_sends > 0 && successful_sends.is_multiple_of(every))
    }
}

impl DirectPredictionReturnWriter {
    pub(crate) fn new(stream: TcpStream) -> Self {
        Self::new_inner(stream, None, DirectReturnReconnectFault::from_env())
    }

    pub(crate) fn new_with_sink_reconnect(
        stream: TcpStream,
        request_id: u64,
        session_id: u64,
        sinks: Arc<PredictionReturnSinks>,
    ) -> Self {
        Self::new_inner(
            stream,
            Some(PredictionReturnReconnect {
                key: PredictionReturnKey::new(request_id, session_id),
                sinks,
                wait_timeout: PREDICTION_RETURN_RECONNECT_WAIT,
            }),
            DirectReturnReconnectFault::from_env(),
        )
    }

    fn new_inner(
        stream: TcpStream,
        reconnect: Option<PredictionReturnReconnect>,
        reconnect_fault: DirectReturnReconnectFault,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            write_queued_prediction_returns(stream, receiver, reconnect, reconnect_fault);
        });
        Self { sender }
    }

    #[cfg(test)]
    fn new_with_sink_reconnect_for_test(
        stream: TcpStream,
        request_id: u64,
        session_id: u64,
        sinks: Arc<PredictionReturnSinks>,
        reconnect_fault: DirectReturnReconnectFault,
        wait_timeout: Duration,
    ) -> Self {
        Self::new_inner(
            stream,
            Some(PredictionReturnReconnect {
                key: PredictionReturnKey::new(request_id, session_id),
                sinks,
                wait_timeout,
            }),
            reconnect_fault,
        )
    }

    pub(crate) fn send(&self, message: &StageWireMessage, reply: StageReply) -> Result<()> {
        self.sender
            .send(QueuedPredictionReturn {
                ready_at: direct_prediction_return_ready_at(message),
                envelope: reply_envelope_for_message(message, reply),
            })
            .map_err(|_| anyhow!("direct prediction return writer closed"))
    }
}

fn direct_prediction_return_ready_at(message: &StageWireMessage) -> Instant {
    let now = Instant::now();
    let delay = DirectReturnDelay::from_env();
    if !delay.should_delay(message) {
        return now;
    }
    eprintln!(
        "skippy direct return validation delay: request_id={} session_id={} decode_step={} delay_ms={}",
        message.request_id,
        message.session_id,
        message.state.decode_step,
        delay.delay.as_millis()
    );
    now + delay.delay
}

pub(crate) fn reply_envelope_for_message(
    message: &StageWireMessage,
    reply: StageReply,
) -> StageReplyEnvelope {
    if message_requests_identified_reply(message)
        && matches!(
            reply.kind,
            WireReplyKind::PredictedToken | WireReplyKind::PredictedTokens
        )
    {
        StageReplyEnvelope::identified(reply, StageReplyIdentity::from_message(message))
    } else {
        StageReplyEnvelope::plain(reply)
    }
}

pub(crate) fn message_requests_identified_reply(message: &StageWireMessage) -> bool {
    (message.state.flags & state_flags::IDENTIFIED_REPLY) != 0
}

fn write_queued_prediction_returns(
    mut stream: TcpStream,
    receiver: mpsc::Receiver<QueuedPredictionReturn>,
    reconnect: Option<PredictionReturnReconnect>,
    reconnect_fault: DirectReturnReconnectFault,
) {
    let mut successful_sends = 0usize;
    for queued in receiver {
        let now = Instant::now();
        if queued.ready_at > now {
            thread::sleep(queued.ready_at.duration_since(now));
        }
        let mut force_reconnect =
            reconnect.is_some() && reconnect_fault.should_reconnect_before_send(successful_sends);
        loop {
            if force_reconnect {
                eprintln!(
                    "direct prediction return writer forcing reconnect after {successful_sends} replies"
                );
                force_reconnect = false;
                if let Err(error) = send_prediction_return_reconnect(&mut stream) {
                    eprintln!(
                        "direct prediction return writer failed to request reconnect: {error:#}"
                    );
                }
                let _ = stream.shutdown(Shutdown::Both);
                drop(stream);
                let Some(next_stream) = reconnect_prediction_return_stream(reconnect.as_ref())
                else {
                    return;
                };
                stream = next_stream;
                continue;
            }
            match send_direct_prediction_return_envelope(&mut stream, queued.envelope.clone()) {
                Ok(()) => {
                    successful_sends += 1;
                    break;
                }
                Err(error) => {
                    eprintln!("direct prediction return writer failed: {error:#}");
                    let Some(next_stream) = reconnect_prediction_return_stream(reconnect.as_ref())
                    else {
                        return;
                    };
                    stream = next_stream;
                }
            }
        }
    }
}

fn reconnect_prediction_return_stream(
    reconnect: Option<&PredictionReturnReconnect>,
) -> Option<TcpStream> {
    let reconnect = reconnect?;
    match reconnect.sinks.take_wait(
        reconnect.key.request_id,
        reconnect.key.session_id,
        reconnect.wait_timeout,
    ) {
        Ok(Some(stream)) => {
            eprintln!(
                "direct prediction return writer reconnected: request_id={} session_id={}",
                reconnect.key.request_id, reconnect.key.session_id
            );
            Some(stream)
        }
        Ok(None) => {
            eprintln!(
                "direct prediction return writer reconnect timed out: request_id={} session_id={}",
                reconnect.key.request_id, reconnect.key.session_id
            );
            None
        }
        Err(error) => {
            eprintln!("direct prediction return writer reconnect failed: {error:#}");
            None
        }
    }
}

fn send_prediction_return_reconnect(stream: &mut TcpStream) -> Result<()> {
    send_direct_prediction_return_envelope(
        stream,
        StageReplyEnvelope::plain(StageReply {
            kind: WireReplyKind::PredictionReturnReconnect,
            predicted: 0,
            predicted_tokens: Vec::new(),
            stats: Default::default(),
        }),
    )
    .context("send direct prediction return reconnect request")
}

impl PredictionReturnSinks {
    pub(crate) fn insert_opened_sink(
        &self,
        open: StageWireMessage,
        stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        self.streams
            .lock()
            .map_err(|_| anyhow!("prediction return sinks lock poisoned"))?
            .insert(key, stream);
        Ok(())
    }

    pub(crate) fn take_wait(
        &self,
        request_id: u64,
        session_id: u64,
        timeout: Duration,
    ) -> Result<Option<TcpStream>> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let started = std::time::Instant::now();
        loop {
            if let Some(stream) = self
                .streams
                .lock()
                .map_err(|_| anyhow!("prediction return sinks lock poisoned"))?
                .remove(&key)
            {
                return Ok(Some(stream));
            }
            if started.elapsed() >= timeout {
                return Ok(None);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }
}

pub(crate) fn open_prediction_return_stream(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
    _timeout_secs: u64,
) -> Result<TcpStream> {
    let endpoint = driver_stage_endpoint(config, topology)?;
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    let attempts = 1;
    let mut last_error = None;
    for _ in 0..attempts {
        match connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2)) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                send_client_ready_hello_if_enabled(&mut stream)
                    .context("send prediction return client ready hello")?;
                recv_ready(&mut stream).context("prediction return sink did not become ready")?;
                write_stage_message(
                    &mut stream,
                    &prediction_return_open_message(request_id, session_id),
                    wire_dtype,
                )
                .context("open direct prediction return stream")?;
                return Ok(stream);
            }
            Err(error) => {
                last_error = Some(anyhow!(error));
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out"))
        .context(format!(
            "connect direct prediction return sink at {endpoint}"
        )))
}

pub(crate) fn open_downstream_prediction_return_stream(
    config: &StageConfig,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
) -> Result<TcpStream> {
    let downstream = config
        .downstream
        .as_ref()
        .ok_or_else(|| anyhow!("direct prediction return requires downstream stage"))?;
    let endpoint = strip_tcp_prefix(&downstream.endpoint);
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    let mut stream = connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2))
        .with_context(|| format!("connect downstream prediction return sink at {endpoint}"))?;
    stream.set_nodelay(true).ok();
    send_client_ready_hello_if_enabled(&mut stream)
        .context("send downstream prediction return client ready hello")?;
    recv_ready(&mut stream).context("downstream prediction return sink did not become ready")?;
    write_stage_message(
        &mut stream,
        &prediction_return_open_message(request_id, session_id),
        wire_dtype,
    )
    .context("open downstream prediction return stream")?;
    Ok(stream)
}

#[cfg(test)]
fn send_direct_prediction_return(stream: &mut TcpStream, reply: StageReply) -> Result<()> {
    send_direct_prediction_return_envelope(stream, StageReplyEnvelope::plain(reply))
}

pub(crate) fn send_direct_prediction_return_for_message(
    stream: &mut TcpStream,
    message: &StageWireMessage,
    reply: StageReply,
) -> Result<()> {
    send_direct_prediction_return_envelope(stream, reply_envelope_for_message(message, reply))
}

fn send_direct_prediction_return_envelope(
    stream: &mut TcpStream,
    envelope: StageReplyEnvelope,
) -> Result<()> {
    send_reply_envelope(stream, envelope).context("send direct prediction return")
}

fn driver_stage_endpoint<'a>(
    config: &'a StageConfig,
    topology: Option<&'a StageTopology>,
) -> Result<&'a str> {
    if let Some(topology) = topology {
        return driver_stage_endpoint_from_topology(topology);
    }
    if let Some(upstream) = config
        .upstream
        .as_ref()
        .filter(|upstream| upstream.stage_index == 0)
    {
        return Ok(strip_tcp_prefix(&upstream.endpoint));
    }
    Err(anyhow!("direct prediction return requires topology"))
}

fn driver_stage_endpoint_from_topology(topology: &StageTopology) -> Result<&str> {
    topology
        .stages
        .iter()
        .find(|stage| stage.stage_index == 0)
        .map(|stage| strip_tcp_prefix(&stage.endpoint))
        .ok_or_else(|| anyhow!("topology does not contain driver-facing stage 0"))
}

fn strip_tcp_prefix(endpoint: &str) -> &str {
    endpoint.strip_prefix("tcp://").unwrap_or(endpoint)
}

fn prediction_return_open_message(request_id: u64, session_id: u64) -> StageWireMessage {
    StageWireMessage {
        kind: WireMessageKind::PredictionReturnOpen,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(
            WireMessageKind::PredictionReturnOpen,
            WireActivationDType::F32,
        ),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{
        recv_reply, recv_reply_envelope, send_reply_predicted_with_stats,
    };
    use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig};
    use std::io::Write;

    #[test]
    fn handle_return_connection_delivers_reply_to_registered_waiter() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let open = prediction_return_open_message(request_id, session_id);
        let handle = {
            let hub = hub.clone();
            thread::spawn(move || hub.handle_return_connection(open, server))
        };

        send_reply_predicted_with_stats(&mut client, 42, Default::default()).unwrap();

        let reply = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        drop(client);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn direct_prediction_return_preserves_predicted_token_sideband() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let reply = StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: 42,
            predicted_tokens: vec![42, 43, 123],
            stats: Default::default(),
        };
        send_direct_prediction_return(&mut server, reply).unwrap();

        let received = recv_reply(&mut client).unwrap();
        assert_eq!(received.kind, WireReplyKind::PredictedToken);
        assert_eq!(received.predicted, 42);
        assert_eq!(received.predicted_tokens, vec![42, 43, 123]);
    }

    #[test]
    fn queued_prediction_returns_use_absolute_ready_times() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            write_queued_prediction_returns(
                server,
                receiver,
                None,
                DirectReturnReconnectFault::default(),
            );
        });
        let ready_at = Instant::now() + Duration::from_millis(250);

        sender
            .send(QueuedPredictionReturn {
                ready_at,
                envelope: StageReplyEnvelope::plain(predicted_token_reply(11)),
            })
            .unwrap();
        sender
            .send(QueuedPredictionReturn {
                ready_at,
                envelope: StageReplyEnvelope::plain(predicted_token_reply(12)),
            })
            .unwrap();

        let started = Instant::now();
        let first = recv_reply(&mut client).unwrap();
        let first_elapsed = started.elapsed();
        let second = recv_reply(&mut client).unwrap();
        let second_gap = started.elapsed().saturating_sub(first_elapsed);

        assert_eq!(first.predicted, 11);
        assert_eq!(second.predicted, 12);
        assert!(first_elapsed >= Duration::from_millis(200));
        assert!(second_gap < Duration::from_millis(200));
        drop(sender);
        handle.join().unwrap();
    }

    #[test]
    fn queued_prediction_returns_can_reconnect_to_replacement_sink() {
        let request_id = 31;
        let session_id = 37;
        let sinks = Arc::new(PredictionReturnSinks::default());
        let (mut stale_client, stale_server) = connected_stream_pair();
        let (mut current_client, current_server) = connected_stream_pair();
        stale_client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        current_client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut message = prediction_return_open_message(request_id, session_id);
        message.kind = WireMessageKind::VerifySpan;

        let writer = DirectPredictionReturnWriter::new_with_sink_reconnect_for_test(
            stale_server,
            request_id,
            session_id,
            sinks.clone(),
            DirectReturnReconnectFault {
                every_successful_sends: Some(1),
            },
            Duration::from_millis(50),
        );
        writer.send(&message, predicted_token_reply(11)).unwrap();
        assert_eq!(recv_reply(&mut stale_client).unwrap().predicted, 11);

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                current_server,
            )
            .unwrap();
        writer.send(&message, predicted_token_reply(42)).unwrap();

        assert_eq!(recv_reply(&mut current_client).unwrap().predicted, 42);
    }

    #[test]
    fn forced_reconnect_round_trips_through_receiver_reopen_without_loss() {
        let request_id = 31;
        let session_id = 37;
        let sinks = Arc::new(PredictionReturnSinks::default());
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let reopen_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let reopen_addr = reopen_listener.local_addr().unwrap();
        let config = stage_config_with_downstream(reopen_addr);
        receiver
            .enable_downstream_reconnect(&config, WireActivationDType::F32)
            .unwrap();
        let sink_acceptor =
            spawn_reopened_sink_acceptor(reopen_listener, sinks.clone(), request_id, session_id);

        let (coordinator_stream, writer_stream) = connected_stream_pair();
        receiver.attach_opened_stream(coordinator_stream);
        wait_for_stream_generation(&receiver, 1);
        let mut message = prediction_return_open_message(request_id, session_id);
        message.kind = WireMessageKind::VerifySpan;
        let writer = DirectPredictionReturnWriter::new_with_sink_reconnect_for_test(
            writer_stream,
            request_id,
            session_id,
            sinks,
            DirectReturnReconnectFault {
                every_successful_sends: Some(1),
            },
            Duration::from_secs(2),
        );

        writer.send(&message, predicted_token_reply(11)).unwrap();
        writer.send(&message, predicted_token_reply(42)).unwrap();

        let first = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        let second = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        assert_eq!(first.predicted, 11);
        assert_eq!(second.predicted, 42);
        sink_acceptor.join().unwrap();
        assert!(
            receiver
                .try_recv_expected(WireReplyKind::PredictedToken)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn direct_prediction_return_for_message_echoes_window_identity() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut message = prediction_return_open_message(17, 23);
        message.kind = WireMessageKind::VerifySpan;
        message.pos_start = 31;
        message.state.flags |= state_flags::IDENTIFIED_REPLY;
        message.state.checkpoint_generation = 5;
        message.state.prompt_token_count = 29;
        message.state.decode_step = 2;
        message.state.seq_id = 7;

        send_direct_prediction_return_for_message(&mut server, &message, predicted_token_reply(42))
            .unwrap();

        let received = recv_reply_envelope(&mut client).unwrap();
        assert_eq!(received.reply.predicted, 42);
        assert_eq!(
            received.identity,
            Some(StageReplyIdentity::from_message(&message))
        );
    }

    #[test]
    fn prediction_return_receiver_ignores_stale_stream_after_reconnect() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();

        let (mut stale_client, stale_server) = connected_stream_pair();
        receiver.attach_opened_stream(stale_server);
        wait_for_stream_generation(&receiver, 1);

        let (mut current_client, current_server) = connected_stream_pair();
        receiver.attach_opened_stream(current_server);
        wait_for_stream_generation(&receiver, 2);

        send_reply_predicted_with_stats(&mut stale_client, 11, Default::default()).unwrap();
        send_reply_predicted_with_stats(&mut current_client, 42, Default::default()).unwrap();

        let reply = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
    }

    #[test]
    fn prediction_return_receiver_ignores_stale_stream_error_after_reconnect() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();

        let (mut stale_client, stale_server) = connected_stream_pair();
        receiver.attach_opened_stream(stale_server);
        wait_for_stream_generation(&receiver, 1);

        let (mut current_client, current_server) = connected_stream_pair();
        receiver.attach_opened_stream(current_server);
        wait_for_stream_generation(&receiver, 2);

        stale_client.write_all(&123_456_i32.to_le_bytes()).unwrap();
        stale_client.flush().unwrap();
        send_reply_predicted_with_stats(&mut current_client, 42, Default::default()).unwrap();

        let reply = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
    }

    #[test]
    fn prediction_return_receiver_reopens_downstream_sink_after_current_stream_closes() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let reopen_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let reopen_addr = reopen_listener.local_addr().unwrap();
        let config = stage_config_with_downstream(reopen_addr);
        receiver
            .enable_downstream_reconnect(&config, WireActivationDType::F32)
            .unwrap();
        let reopen = thread::spawn(move || {
            let (mut stream, _) = reopen_listener.accept().unwrap();
            send_ready(&mut stream).unwrap();
            let open = read_stage_message(&mut stream, 0).unwrap();
            (stream, open)
        });

        let (current_client, current_server) = connected_stream_pair();
        receiver.attach_opened_stream(current_server);
        wait_for_stream_generation(&receiver, 1);
        drop(current_client);

        let mut replacement = wait_for_reopened_stream(&receiver, reopen);
        send_reply_predicted_with_stats(&mut replacement, 42, Default::default()).unwrap();

        let reply = poll_test_reply(&receiver, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
    }

    #[test]
    fn dropping_old_receiver_does_not_unregister_new_receiver_for_same_key() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let old_receiver = hub.register(request_id, session_id).unwrap();
        let new_receiver = hub.register(request_id, session_id).unwrap();
        drop(old_receiver);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let open = prediction_return_open_message(request_id, session_id);
        let handle = {
            let hub = hub.clone();
            thread::spawn(move || hub.handle_return_connection(open, server))
        };

        send_reply_predicted_with_stats(&mut client, 42, Default::default()).unwrap();

        let reply = poll_test_reply(&new_receiver, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        drop(client);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn prediction_return_sinks_store_upstream_opened_streams() {
        let request_id = 31;
        let session_id = 37;
        let sinks = PredictionReturnSinks::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                server,
            )
            .unwrap();

        let stream = sinks
            .take_wait(request_id, session_id, Duration::from_millis(1))
            .unwrap()
            .expect("registered prediction return sink");
        assert_eq!(stream.peer_addr().unwrap(), client.local_addr().unwrap());
    }

    #[test]
    fn prediction_return_sinks_replace_stale_upstream_opened_streams() {
        let request_id = 31;
        let session_id = 37;
        let sinks = PredictionReturnSinks::default();
        let (stale_client, stale_server) = connected_stream_pair();
        let (current_client, current_server) = connected_stream_pair();

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                stale_server,
            )
            .unwrap();
        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                current_server,
            )
            .unwrap();

        let stream = sinks
            .take_wait(request_id, session_id, Duration::from_millis(1))
            .unwrap()
            .expect("registered prediction return sink");
        assert_eq!(
            stream.peer_addr().unwrap(),
            current_client.local_addr().unwrap()
        );
        drop(stale_client);
    }

    fn connected_stream_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        (client, server)
    }

    fn stage_config_with_downstream(downstream_addr: SocketAddr) -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "model".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            tree_sequence_count: 0,
            selected_device: None,
            kv_cache: None,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: format!("tcp://{downstream_addr}"),
            }),
        }
    }

    fn poll_test_reply(receiver: &PredictionReturnReceiver, expected: WireReplyKind) -> StageReply {
        let started = std::time::Instant::now();
        loop {
            if let Some(reply) = receiver.try_recv_expected(expected).unwrap() {
                return reply.reply;
            }
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "timed out waiting for prediction return reply"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn wait_for_reopened_stream(
        receiver: &PredictionReturnReceiver,
        reopen: JoinHandle<(TcpStream, StageWireMessage)>,
    ) -> TcpStream {
        let started = std::time::Instant::now();
        loop {
            let _ = receiver
                .try_recv_expected(WireReplyKind::PredictedToken)
                .unwrap();
            if reopen.is_finished() {
                let (stream, open) = reopen.join().unwrap();
                assert_eq!(open.kind, WireMessageKind::PredictionReturnOpen);
                assert_eq!(open.request_id, receiver.key.request_id);
                assert_eq!(open.session_id, receiver.key.session_id);
                return stream;
            }
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "timed out waiting for replacement prediction return sink"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn spawn_reopened_sink_acceptor(
        listener: TcpListener,
        sinks: Arc<PredictionReturnSinks>,
        request_id: u64,
        session_id: u64,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            send_ready(&mut stream).unwrap();
            let open = read_stage_message(&mut stream, 0).unwrap();
            assert_eq!(open.kind, WireMessageKind::PredictionReturnOpen);
            assert_eq!(open.request_id, request_id);
            assert_eq!(open.session_id, session_id);
            sinks.insert_opened_sink(open, stream).unwrap();
        })
    }

    fn wait_for_stream_generation(receiver: &PredictionReturnReceiver, generation: u64) {
        let started = std::time::Instant::now();
        while receiver.waiter.stream_generation.load(Ordering::SeqCst) < generation {
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "timed out waiting for prediction return stream generation {generation}"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn predicted_token_reply(predicted: i32) -> StageReply {
        StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted,
            predicted_tokens: vec![predicted],
            stats: Default::default(),
        }
    }
}
