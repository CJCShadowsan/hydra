use std::{
    collections::{HashMap, VecDeque},
    io::{self, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{
        StageReply, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
        WireReplyKind, read_stage_message, recv_ready, recv_reply, send_ready,
        send_reply_ack_with_stats, send_reply_predicted_tokens_with_stats,
        send_reply_predicted_with_stats, send_reply_spd_tap_with_stats, state_flags,
        write_stage_message,
    },
};

use super::socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint};

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

const PREDICTION_RETURN_ORIGIN_MAGIC: i32 = 0x5350_4452; // "SPDR"
const PREDICTION_RETURN_ORIGIN_VERSION: i32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PredictionReturnOrigin {
    pub(crate) kind: WireMessageKind,
    pub(crate) pos_start: i32,
    pub(crate) token_count: i32,
    pub(crate) prompt_token_count: i32,
    pub(crate) decode_step: i32,
    pub(crate) checkpoint_generation: i32,
}

impl PredictionReturnOrigin {
    pub(crate) fn from_message(message: &StageWireMessage) -> Self {
        Self {
            kind: message.kind,
            pos_start: message.pos_start,
            token_count: message.token_count,
            prompt_token_count: message.state.prompt_token_count,
            decode_step: message.state.decode_step,
            checkpoint_generation: message.state.checkpoint_generation,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PredictionReturnItem {
    pub(crate) reply: StageReply,
    pub(crate) origin: Option<PredictionReturnOrigin>,
}

impl PredictionReturnItem {
    pub(crate) fn matches_origin(
        &self,
        expected: WireReplyKind,
        origin: PredictionReturnOrigin,
    ) -> bool {
        self.reply.kind == expected && self.origin.is_none_or(|actual| actual == origin)
    }
}

pub struct PredictionReturnHub {
    waiters:
        Mutex<HashMap<PredictionReturnKey, mpsc::Sender<Result<PredictionReturnItem, String>>>>,
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
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .insert(key, sender);
        Ok(PredictionReturnReceiver {
            key,
            hub: self.clone(),
            receiver,
            buffered: Mutex::new(VecDeque::new()),
            timeout: Duration::from_secs(300),
        })
    }

    pub(crate) fn unregister(&self, key: PredictionReturnKey) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&key);
        }
    }

    pub(crate) fn handle_return_connection(
        &self,
        open: StageWireMessage,
        mut stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let origin_enabled = (open.state.flags & state_flags::PREDICTION_RETURN_ORIGIN) != 0;
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        let sender = self
            .waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("no prediction return waiter for request {}", key.request_id))?;
        loop {
            match recv_direct_prediction_return(&mut stream, origin_enabled) {
                Ok(item) => {
                    if sender.send(Ok(item)).is_err() {
                        return Ok(());
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => {
                    let _ = sender.send(Err(error.to_string()));
                    return Err(error).context("read direct prediction return");
                }
            }
        }
    }
}

pub(crate) struct PredictionReturnReceiver {
    key: PredictionReturnKey,
    hub: Arc<PredictionReturnHub>,
    receiver: mpsc::Receiver<Result<PredictionReturnItem, String>>,
    buffered: Mutex<VecDeque<PredictionReturnItem>>,
    timeout: Duration,
}

impl PredictionReturnReceiver {
    pub(crate) fn recv_expected(&self, expected: WireReplyKind) -> Result<StageReply> {
        let reply = self.recv()?;
        if reply.kind != expected {
            bail!(
                "expected {expected:?} direct prediction return, got {:?}",
                reply.kind
            );
        }
        Ok(reply)
    }

    pub(crate) fn recv(&self) -> Result<StageReply> {
        Ok(self.recv_item()?.reply)
    }

    pub(crate) fn recv_item(&self) -> Result<PredictionReturnItem> {
        if let Some(item) = self
            .buffered
            .lock()
            .map_err(|_| anyhow!("prediction return buffer lock poisoned"))?
            .pop_front()
        {
            return Ok(item);
        }
        self.recv_channel_item()
    }

    pub(crate) fn recv_item_matching(
        &self,
        mut matches: impl FnMut(&PredictionReturnItem) -> bool,
    ) -> Result<PredictionReturnItem> {
        loop {
            if let Some(item) = self.take_buffered_matching(&mut matches)? {
                return Ok(item);
            }
            let item = self.recv_channel_item()?;
            if matches(&item) {
                return Ok(item);
            }
            self.buffered
                .lock()
                .map_err(|_| anyhow!("prediction return buffer lock poisoned"))?
                .push_back(item);
        }
    }

    fn take_buffered_matching(
        &self,
        matches: &mut impl FnMut(&PredictionReturnItem) -> bool,
    ) -> Result<Option<PredictionReturnItem>> {
        let mut buffered = self
            .buffered
            .lock()
            .map_err(|_| anyhow!("prediction return buffer lock poisoned"))?;
        let Some(index) = buffered.iter().position(matches) else {
            return Ok(None);
        };
        Ok(buffered.remove(index))
    }

    fn recv_channel_item(&self) -> Result<PredictionReturnItem> {
        self.receiver
            .recv_timeout(self.timeout)
            .context("timed out waiting for direct prediction return")?
            .map_err(|error| anyhow!(error))
    }
}

impl Drop for PredictionReturnReceiver {
    fn drop(&mut self) {
        self.hub.unregister(self.key);
    }
}

pub(crate) fn open_prediction_return_stream(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
    timeout_secs: u64,
) -> Result<TcpStream> {
    let endpoint = driver_stage_endpoint(config, topology)?;
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2)) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
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

pub(crate) fn send_direct_prediction_return(
    stream: &mut TcpStream,
    origin: PredictionReturnOrigin,
    reply: StageReply,
) -> Result<()> {
    write_direct_prediction_return(stream, Some(origin), reply)
}

fn write_direct_prediction_return(
    mut writer: impl Write,
    origin: Option<PredictionReturnOrigin>,
    reply: StageReply,
) -> Result<()> {
    if let Some(origin) = origin {
        write_prediction_return_origin(&mut writer, origin)
            .context("send direct prediction return origin")?;
    }
    match reply.kind {
        WireReplyKind::PredictedToken => {
            send_reply_predicted_with_stats(&mut writer, reply.predicted, reply.stats)
                .context("send direct predicted-token return")
        }
        WireReplyKind::PredictedTokens => send_reply_predicted_tokens_with_stats(
            &mut writer,
            &reply.predicted_tokens,
            reply.stats,
        )
        .context("send direct predicted-tokens return"),
        WireReplyKind::Ack => {
            send_reply_ack_with_stats(&mut writer, reply.stats).context("send direct ACK return")
        }
        WireReplyKind::SpdTap => {
            let tap = reply
                .spd_tap
                .as_ref()
                .context("missing SPD tap reply payload")?;
            send_reply_spd_tap_with_stats(&mut writer, tap, reply.stats)
                .context("send direct SPD tap return")
        }
    }
}

fn recv_direct_prediction_return(
    mut reader: impl Read,
    origin_enabled: bool,
) -> io::Result<PredictionReturnItem> {
    let origin = if origin_enabled {
        Some(read_prediction_return_origin(&mut reader)?)
    } else {
        None
    };
    let reply = recv_reply(&mut reader)?;
    Ok(PredictionReturnItem { reply, origin })
}

fn write_prediction_return_origin(
    mut writer: impl Write,
    origin: PredictionReturnOrigin,
) -> io::Result<()> {
    write_i32(&mut writer, PREDICTION_RETURN_ORIGIN_MAGIC)?;
    write_i32(&mut writer, PREDICTION_RETURN_ORIGIN_VERSION)?;
    write_i32(&mut writer, origin.kind as i32)?;
    write_i32(&mut writer, origin.pos_start)?;
    write_i32(&mut writer, origin.token_count)?;
    write_i32(&mut writer, origin.prompt_token_count)?;
    write_i32(&mut writer, origin.decode_step)?;
    write_i32(&mut writer, origin.checkpoint_generation)
}

fn read_prediction_return_origin(mut reader: impl Read) -> io::Result<PredictionReturnOrigin> {
    if read_i32(&mut reader)? != PREDICTION_RETURN_ORIGIN_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "direct prediction return origin magic mismatch",
        ));
    }
    if read_i32(&mut reader)? != PREDICTION_RETURN_ORIGIN_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported direct prediction return origin version",
        ));
    }
    Ok(PredictionReturnOrigin {
        kind: WireMessageKind::try_from(read_i32(&mut reader)?)?,
        pos_start: read_i32(&mut reader)?,
        token_count: read_i32(&mut reader)?,
        prompt_token_count: read_i32(&mut reader)?,
        decode_step: read_i32(&mut reader)?,
        checkpoint_generation: read_i32(&mut reader)?,
    })
}

fn write_i32(mut writer: impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i32(mut reader: impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
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
    let mut state = StageStateHeader::new(
        WireMessageKind::PredictionReturnOpen,
        WireActivationDType::F32,
    );
    state.flags |= state_flags::PREDICTION_RETURN_ORIGIN;
    StageWireMessage {
        kind: WireMessageKind::PredictionReturnOpen,
        pos_start: 0,
        token_count: 0,
        state,
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

    fn origin(pos_start: i32, decode_step: i32) -> PredictionReturnOrigin {
        PredictionReturnOrigin {
            kind: WireMessageKind::VerifySpan,
            pos_start,
            token_count: 1,
            prompt_token_count: 4,
            decode_step,
            checkpoint_generation: 0,
        }
    }

    fn predicted_tokens_reply(tokens: &[i32]) -> StageReply {
        StageReply {
            kind: WireReplyKind::PredictedTokens,
            predicted: tokens.first().copied().unwrap_or_default(),
            predicted_tokens: tokens.to_vec(),
            spd_tap: None,
            stats: Default::default(),
        }
    }

    #[test]
    fn direct_prediction_return_origin_round_trips_with_reply() {
        let expected_origin = origin(7, 3);
        let expected_reply = predicted_tokens_reply(&[101, 102]);
        let mut bytes = Vec::new();

        write_direct_prediction_return(&mut bytes, Some(expected_origin), expected_reply.clone())
            .expect("write direct prediction return");

        let item =
            recv_direct_prediction_return(bytes.as_slice(), true).expect("read direct return");
        assert_eq!(item.origin, Some(expected_origin));
        assert_eq!(item.reply, expected_reply);
    }

    #[test]
    fn prediction_return_receiver_buffers_unmatched_origins() {
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(11, 22).expect("register receiver");
        let sender = hub
            .waiters
            .lock()
            .expect("waiters lock")
            .get(&PredictionReturnKey::new(11, 22))
            .cloned()
            .expect("registered sender");
        let first_origin = origin(5, 1);
        let second_origin = origin(6, 2);

        sender
            .send(Ok(PredictionReturnItem {
                reply: predicted_tokens_reply(&[202]),
                origin: Some(second_origin),
            }))
            .expect("send second");
        sender
            .send(Ok(PredictionReturnItem {
                reply: predicted_tokens_reply(&[101]),
                origin: Some(first_origin),
            }))
            .expect("send first");

        let first = receiver
            .recv_item_matching(|item| {
                item.matches_origin(WireReplyKind::PredictedTokens, first_origin)
            })
            .expect("receive first origin");
        assert_eq!(first.origin, Some(first_origin));
        assert_eq!(first.reply.predicted_tokens, vec![101]);

        let buffered = receiver.recv_item().expect("receive buffered origin");
        assert_eq!(buffered.origin, Some(second_origin));
        assert_eq!(buffered.reply.predicted_tokens, vec![202]);
    }
}
