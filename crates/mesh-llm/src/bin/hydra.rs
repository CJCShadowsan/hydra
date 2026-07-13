#![recursion_limit = "256"]

const DEFAULT_WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();

    let stack_size = std::env::var("MESH_TOKIO_STACK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WORKER_STACK_SIZE);
    builder.thread_stack_size(stack_size);

    let runtime = builder.build().expect("build tokio runtime");
    std::process::exit(runtime.block_on(mesh_llm::run_main()));
}
