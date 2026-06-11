//! Join an MLX ring group as a single rank. Used by the two-rank localhost
//! integration test (`tests/live_two_node_ring.rs`) to spawn a worker process
//! without needing a second physical machine.
//!
//! Driven entirely by env:
//!   - `MLX_TEST_MODEL`     — HF repo id to load.
//!   - `RING_HOSTFILE_JSON` — ring hostfile JSON (`[["ip:port"], ...]`).
//!   - `MLX_RANK`           — this process's rank.
//!   - `RING_MAX_TOKENS`    — generation budget (default 8).
//!
//! Rank 0 (leader) broadcasts one request and prints the completion as
//! `LEADER_OUTPUT: <text>`. Non-zero ranks run the worker loop until the leader
//! broadcasts the shutdown sentinel, then exit 0.
//!
//! Only built with `--features link-mlx` (needs the native engine).

#![recursion_limit = "256"]

#[cfg(not(feature = "link-mlx"))]
fn main() {
    eprintln!("ring_rank requires --features link-mlx");
    std::process::exit(2);
}

#[cfg(feature = "link-mlx")]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    use mesh_mlx::{
        Backend, ChatTurn, DistributedEngine, JoinParams, ModelRef, ParallelismMode, WorkerHandle,
    };

    let model = std::env::var("MLX_TEST_MODEL")
        .unwrap_or_else(|_| "mlx-community/Qwen2.5-0.5B-Instruct-bf16".to_string());
    let hostfile_json = std::env::var("RING_HOSTFILE_JSON").expect("RING_HOSTFILE_JSON");
    let rank: usize = std::env::var("MLX_RANK")
        .expect("MLX_RANK")
        .parse()
        .expect("rank parses");
    let max_tokens: usize = std::env::var("RING_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let join = JoinParams {
        hostfile_json,
        jaccl: None,
        rank,
        backend: Backend::Ring,
        mode: ParallelismMode::Pipeline,
    };

    let dengine = match DistributedEngine::join(&ModelRef::new(&model), join).await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("rank {rank} join failed: {e}");
            std::process::exit(3);
        }
    };

    if dengine.is_leader() {
        let turns = [ChatTurn::new("user", "What is the capital of France?")];
        match dengine.chat_turns(&turns, max_tokens) {
            Ok(text) => {
                println!("LEADER_OUTPUT: {text}");
                dengine.signal_shutdown();
            }
            Err(e) => {
                eprintln!("leader generation failed: {e}");
                dengine.signal_shutdown();
                std::process::exit(4);
            }
        }
    } else {
        let worker = WorkerHandle::spawn(dengine);
        if let Err(e) = worker.join().await {
            eprintln!("worker rank {rank} loop failed: {e}");
            std::process::exit(5);
        }
        eprintln!("worker rank {rank} exited cleanly");
    }
}
