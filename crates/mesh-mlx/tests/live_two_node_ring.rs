//! Real **two-rank** MLX ring inference on a single machine — proves the
//! distributed leader/worker coordination actually rendezvous and generate,
//! without needing two physical Macs.
//!
//! Both ranks run the `ring_rank` example as separate processes (MLX's
//! distributed init is process-global, so two processes are required). They
//! join a TCP ring on two localhost ports from a shared hostfile; rank 0 (the
//! leader) broadcasts one request and generates, rank 1 runs the worker loop.
//!
//! This is the smallest viable shape of the distributed path. It catches the
//! failures unit tests can't: hostfile-format mismatches, group-init rendezvous
//! deadlocks, the request broadcast, and the worker loop staying in lock-step.
//!
//! Gated behind `link-mlx`; opt-in via `MLX_TWO_NODE_RING=1` because it loads
//! the model twice (2x memory) and binds localhost ports.
//!
//! ```bash
//! MLX_TWO_NODE_RING=1 cargo test -p mesh-mlx --features link-mlx \
//!   --test live_two_node_ring -- --nocapture
//! ```

#![cfg(feature = "link-mlx")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn enabled() -> bool {
    matches!(
        std::env::var("MLX_TWO_NODE_RING")
            .unwrap_or_default()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Pick two free localhost TCP ports by binding ephemeral sockets and releasing
/// them (a brief race window, acceptable for a test).
fn two_free_ports() -> (u16, u16) {
    let take = || {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        l.local_addr().expect("local addr").port()
    };
    let a = take();
    let mut b = take();
    while b == a {
        b = take();
    }
    (a, b)
}

fn spawn_rank(rank: usize, hostfile: &str, max_tokens: usize) -> std::process::Child {
    // Re-run the example binary for this rank. `cargo` rebuilds are cached, so
    // invoking through cargo keeps the example in sync with the crate.
    Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "-p",
            "mesh-mlx",
            "--features",
            "link-mlx",
            "--example",
            "ring_rank",
        ])
        .env("MLX_RANK", rank.to_string())
        .env("RING_HOSTFILE_JSON", hostfile)
        .env("RING_MAX_TOKENS", max_tokens.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ring_rank")
}

#[test]
fn two_rank_ring_generates_without_deadlock() {
    if !enabled() {
        eprintln!("skipping: set MLX_TWO_NODE_RING=1 to run the two-rank ring test");
        return;
    }

    let (p0, p1) = two_free_ports();
    // Shared ring hostfile: one row per rank, each its own listen address.
    let hostfile = format!(r#"[["127.0.0.1:{p0}"],["127.0.0.1:{p1}"]]"#);

    // Start the worker (rank 1) first so it is listening when the leader
    // connects, then the leader (rank 0).
    let mut worker = spawn_rank(1, &hostfile, 8);
    std::thread::sleep(Duration::from_millis(500));
    let mut leader = spawn_rank(0, &hostfile, 8);

    // Bound the whole exchange; a deadlock (the bug this test guards against)
    // would otherwise hang. Model download on a cold cache can be slow, so the
    // budget is generous.
    let deadline = Instant::now() + Duration::from_secs(600);
    let leader_status = loop {
        if let Some(status) = leader.try_wait().expect("leader wait") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = leader.kill();
            let _ = worker.kill();
            panic!("two-rank ring timed out (possible group-init or broadcast deadlock)");
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    let mut leader_out = String::new();
    if let Some(mut o) = leader.stdout.take() {
        let _ = o.read_to_string(&mut leader_out);
    }
    let mut leader_err = String::new();
    if let Some(mut e) = leader.stderr.take() {
        let _ = e.read_to_string(&mut leader_err);
    }

    // The worker should exit on its own once the leader broadcasts the
    // shutdown sentinel.
    let worker_status = loop {
        if let Some(status) = worker.try_wait().expect("worker wait") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = worker.kill();
            panic!("worker did not exit after leader shutdown (worker-loop sentinel gap)");
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    eprintln!("leader stdout:\n{leader_out}");
    eprintln!("leader stderr:\n{leader_err}");

    assert!(
        leader_status.success(),
        "leader exited with failure: {leader_status:?}"
    );
    assert!(
        worker_status.success(),
        "worker exited with failure: {worker_status:?}"
    );
    let answer = leader_out
        .lines()
        .find_map(|l| l.strip_prefix("LEADER_OUTPUT: "))
        .expect("leader printed LEADER_OUTPUT");
    assert!(
        answer.to_lowercase().contains("paris"),
        "distributed completion should mention Paris, got: {answer:?}"
    );
}
