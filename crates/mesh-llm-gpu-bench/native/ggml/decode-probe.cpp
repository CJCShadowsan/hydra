#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#if defined(MESH_LLM_GGML_PROBE_METAL)
#include "ggml-metal.h"
#endif

#if defined(MESH_LLM_GGML_PROBE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int WARMUP_RUNS = 3;
constexpr int TIMED_RUNS = 12;
constexpr int GRAPH_WARMUP_RUNS = 1;
constexpr int GRAPH_TIMED_RUNS = 3;
constexpr int64_t MAX_MODEL_SHAPED_MOE_EXPERTS = 128;
constexpr int64_t DEEP_LLAMA_GRAPH_LAYERS[] = {4, 8};
constexpr int GRAPH_FEATURE_ATTENTION_Q_NORM = 1 << 0;
constexpr int GRAPH_FEATURE_ATTENTION_K_NORM = 1 << 1;
constexpr int GRAPH_FEATURE_ATTENTION_POST_NORM = 1 << 2;
constexpr int GRAPH_FEATURE_FFN_POST_NORM = 1 << 3;

enum ProbeBackend {
    PROBE_BACKEND_METAL = 0,
    PROBE_BACKEND_CUDA = 1,
    PROBE_BACKEND_HIP = 2,
};

enum ProbeDepth {
    PROBE_DEPTH_STANDARD = 0,
    PROBE_DEPTH_DEEP = 1,
};

enum ProbeTensorType {
    PROBE_TENSOR_Q4_K = 0,
    PROBE_TENSOR_Q6_K = 1,
    PROBE_TENSOR_Q8_0 = 2,
    PROBE_TENSOR_F16 = 3,
};

struct ProbeResult {
    std::string name;
    std::string tensor_type;
    int64_t rows;
    int64_t cols;
    double effective_gbps;
    double tflops;
    double elapsed_ms;
    int graph_features;
    int runs;
};

struct ProbeShape {
    const char * suffix;
    int64_t rows;
    int64_t cols;
};

struct ScheduledGraph {
    ggml_backend_sched_t sched;
    ggml_backend_t cpu_backend;
};

struct EncodedWeightCache {
    std::unordered_map<std::string, std::vector<uint8_t>> encoded_by_shape;
};

constexpr ProbeShape DECODE_SHAPES[] = {
    {"square_2048", 2048, 2048},
    {"square_4096", 4096, 4096},
    {"ffn_up_4096_12288", 12288, 4096},
    {"ffn_down_12288_4096", 4096, 12288},
    {"expert_2048_128", 128, 2048},
};

constexpr ProbeShape LLAMA_GRAPH_SHAPES[] = {
    {"768_2048", 768, 2048},
    {"1024_4096", 1024, 4096},
    {"2048_6144", 2048, 6144},
    {"2560_9728", 2560, 9728},
    {"4096_12288", 4096, 12288},
};

constexpr ProbeShape LLAMA_GQA_GRAPH_SHAPES[] = {
    {"2048_kv1024_6144", 2048, 6144},
    {"2560_kv1024_9728", 2560, 9728},
    {"4096_kv1024_12288", 4096, 12288},
};

char * copy_c_string(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, value.c_str(), value.size() + 1);
    }
    return out;
}

void set_error(char ** error_out, const std::string & message) {
    if (error_out != nullptr) {
        *error_out = copy_c_string(message);
    }
}

ggml_backend_t init_backend(int backend_kind) {
    switch (backend_kind) {
        case PROBE_BACKEND_METAL:
#if defined(MESH_LLM_GGML_PROBE_METAL)
            return ggml_backend_metal_init();
#else
            return nullptr;
#endif
        case PROBE_BACKEND_CUDA:
#if defined(MESH_LLM_GGML_PROBE_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        case PROBE_BACKEND_HIP:
#if defined(MESH_LLM_GGML_PROBE_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        default:
            return nullptr;
    }
}

enum ggml_type probe_tensor_type(int tensor_type_kind) {
    switch (tensor_type_kind) {
        case PROBE_TENSOR_Q4_K:
            return GGML_TYPE_Q4_K;
        case PROBE_TENSOR_Q6_K:
            return GGML_TYPE_Q6_K;
        case PROBE_TENSOR_Q8_0:
            return GGML_TYPE_Q8_0;
        case PROBE_TENSOR_F16:
            return GGML_TYPE_F16;
        default:
            return GGML_TYPE_COUNT;
    }
}

bool quantized_row_width_supported(enum ggml_type type, int64_t cols) {
    const int64_t block_size = ggml_blck_size(type);
    return block_size > 0 && cols > 0 && cols % block_size == 0;
}

bool matrix_shape_supported(enum ggml_type type, int64_t rows, int64_t cols) {
    return rows > 0 && quantized_row_width_supported(type, cols);
}

bool dense_llama_shape_supported(
    enum ggml_type type,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn) {
    // GGML quantized tensor storage is row-blocked along ne[0]. Real GGUF
    // loading enforces `ne[0] % ggml_blck_size(type) == 0`; a model can still
    // be valid because mixed-quant files choose compatible tensor types per
    // tensor. The synthetic validator probes ask a stricter question by forcing
    // one candidate type across an entire llama-shaped block. When a small
    // model has widths such as hidden=576 and kv_width=192, an all-Q6_K graph is
    // not a source-plausible GGUF tensor layout even if the model file contains
    // other Q6_K tensors. Reject those probe shapes before graph construction so
    // backend planners never see an impossible quantized tensor.
    return matrix_shape_supported(type, hidden, hidden)
        && matrix_shape_supported(type, kv_width, hidden)
        && matrix_shape_supported(type, ffn, hidden)
        && matrix_shape_supported(type, hidden, ffn);
}

bool moe_shape_supported(
    enum ggml_type type,
    int64_t hidden,
    int64_t expert_width) {
    return matrix_shape_supported(type, expert_width, hidden)
        && matrix_shape_supported(type, hidden, expert_width);
}

const char * probe_tensor_type_name(int tensor_type_kind) {
    switch (tensor_type_kind) {
        case PROBE_TENSOR_Q4_K:
            return "q4_k";
        case PROBE_TENSOR_Q6_K:
            return "q6_k";
        case PROBE_TENSOR_Q8_0:
            return "q8_0";
        case PROBE_TENSOR_F16:
            return "f16";
        default:
            return "unknown";
    }
}

std::vector<float> deterministic_f32(int64_t count, uint32_t salt) {
    std::vector<float> values(static_cast<size_t>(count));
    uint32_t state = 0x9e3779b9u ^ salt;
    for (int64_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float centered = static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f;
        values[static_cast<size_t>(i)] = centered * 0.125f;
    }
    return values;
}

std::vector<uint8_t> encode_weights(
    enum ggml_type type,
    const std::vector<float> & weights,
    int64_t rows,
    int64_t cols) {
    const size_t encoded_bytes = ggml_row_size(type, cols) * rows;
    std::vector<uint8_t> encoded(encoded_bytes);
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(
            weights.data(),
            reinterpret_cast<ggml_fp16_t *>(encoded.data()),
            static_cast<int64_t>(weights.size()));
        return encoded;
    }
    ggml_quantize_chunk(type, weights.data(), encoded.data(), 0, rows, cols, nullptr);
    return encoded;
}

std::string encoded_weight_cache_key(enum ggml_type type, int64_t rows, int64_t cols) {
    return std::to_string(static_cast<int>(type))
        + ":"
        + std::to_string(rows)
        + "x"
        + std::to_string(cols);
}

const std::vector<uint8_t> & cached_encoded_weights(
    EncodedWeightCache & cache,
    enum ggml_type type,
    int64_t rows,
    int64_t cols) {
    const std::string key = encoded_weight_cache_key(type, rows, cols);
    auto existing = cache.encoded_by_shape.find(key);
    if (existing != cache.encoded_by_shape.end()) {
        return existing->second;
    }
    // The synthetic probes measure graph topology, backend scheduling, and
    // quantized kernel traffic. They do not test numerical accuracy. Repeated
    // model-shaped graphs contain the same small set of tensor shapes in every
    // layer, so quantizing fresh random weights for each layer only measures
    // CPU-side benchmark setup. Reusing one deterministic encoded blob per
    // `(type, rows, cols)` preserves tensor type, byte size, layout, and GGML op
    // support while keeping validation time proportional to the graph we time,
    // not to redundant host quantization.
    std::vector<float> weight_f32 = deterministic_f32(rows * cols, 17);
    auto inserted = cache.encoded_by_shape.emplace(
        key,
        encode_weights(type, weight_f32, rows, cols));
    return inserted.first->second;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5;
}

bool run_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    const ProbeShape & shape,
    ProbeResult & result) {
    if (!matrix_shape_supported(type, shape.rows, shape.cols)) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, shape.cols, shape.rows);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.cols, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(weights, "ggml_decode_probe_weights");
    ggml_set_name(input, "ggml_decode_probe_input");
    ggml_set_name(output, "ggml_decode_probe_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (cpu_backend == nullptr) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        nullptr,
        2,
        GGML_DEFAULT_GRAPH_SIZE,
        false,
        true);
    if (sched == nullptr) {
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> weight_f32 = deterministic_f32(shape.rows * shape.cols, 17);
    std::vector<uint8_t> weight_encoded = encode_weights(type, weight_f32, shape.rows, shape.cols);
    std::vector<float> input_f32 = deterministic_f32(shape.cols, 29);
    ggml_backend_tensor_set(weights, weight_encoded.data(), 0, weight_encoded.size());
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    auto compute_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(sched, graph);
        ggml_backend_sched_synchronize(sched);
        const auto finished = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (compute_once() <= 0.0) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu_backend);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        const double elapsed = compute_once();
        if (elapsed <= 0.0) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu_backend);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }

    // Decode-kernel probes feed the model-fit tok/s estimator, whose
    // validation target is median steady decode throughput. Use median elapsed
    // for this reusable kernel slope. The top-level streaming-memory benchmark
    // still reports p50/p90 separately; this probe has one effective bandwidth
    // field, and using p90/max here made metadata-only fit estimates
    // systematically conservative when one backend scheduler sample was slow.
    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }
    const double bytes = static_cast<double>(weight_encoded.size())
        + static_cast<double>(input_f32.size() * sizeof(float))
        + static_cast<double>(shape.rows * sizeof(float));
    const double flops = 2.0 * static_cast<double>(shape.rows) * static_cast<double>(shape.cols);
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops) || !std::isfinite(elapsed_ms)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }
    result = ProbeResult{
        name,
        tensor_type,
        shape.rows,
        shape.cols,
        effective_gbps,
        tflops,
        elapsed_ms,
        0,
        TIMED_RUNS,
    };

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);
    return true;
}

ScheduledGraph alloc_sched_for_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (cpu_backend == nullptr) {
        return ScheduledGraph{nullptr, nullptr};
    }
    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        nullptr,
        2,
        GGML_DEFAULT_GRAPH_SIZE,
        false,
        true);
    if (sched == nullptr) {
        ggml_backend_free(cpu_backend);
        return ScheduledGraph{nullptr, nullptr};
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return ScheduledGraph{nullptr, nullptr};
    }
    return ScheduledGraph{sched, cpu_backend};
}

void free_scheduled_graph(ScheduledGraph scheduled) {
    if (scheduled.sched != nullptr) {
        ggml_backend_sched_free(scheduled.sched);
    }
    if (scheduled.cpu_backend != nullptr) {
        ggml_backend_free(scheduled.cpu_backend);
    }
}

bool compute_graph_timed(
    ggml_cgraph * graph,
    ScheduledGraph scheduled,
    ProbeResult & result,
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    double bytes,
    double flops,
    int graph_features = 0,
    int warmup_runs = WARMUP_RUNS,
    int timed_runs = TIMED_RUNS) {
    if (scheduled.sched == nullptr) {
        return false;
    }
    auto compute_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        ggml_backend_sched_synchronize(scheduled.sched);
        const auto finished = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < warmup_runs; ++i) {
        if (compute_once() <= 0.0) {
            free_scheduled_graph(scheduled);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const double elapsed = compute_once();
        if (elapsed <= 0.0) {
            free_scheduled_graph(scheduled);
            return false;
        }
        seconds.push_back(elapsed);
    }

    // Same rationale as the isolated matvec probe above: this field is consumed
    // as a median decode-performance predictor, not as a worst-case latency
    // guardrail. Keep operational watchdogs outside the scoring model.
    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        free_scheduled_graph(scheduled);
        return false;
    }
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops) || !std::isfinite(elapsed_ms)) {
        free_scheduled_graph(scheduled);
        return false;
    }
    result = ProbeResult{
        name,
        tensor_type,
        rows,
        cols,
        effective_gbps,
        tflops,
        elapsed_ms,
        graph_features,
        timed_runs,
    };

    free_scheduled_graph(scheduled);
    return true;
}

bool set_encoded_weights(
    ggml_tensor * tensor,
    enum ggml_type type,
    int64_t rows,
    int64_t cols,
    EncodedWeightCache & cache,
    double & bytes,
    double & flops) {
    const std::vector<uint8_t> & encoded = cached_encoded_weights(cache, type, rows, cols);
    // Real GGUF files can mix tensor types per tensor, while these probes ask a
    // stricter question: "can the backend execute this whole model-shaped graph
    // if every matmul weight uses this candidate type?"  For very small or
    // unusual widths, especially K-quants, GGML may allocate fewer bytes for the
    // synthetic tensor than ggml_quantize_chunk produces for the same logical
    // rows/cols.  Calling ggml_backend_tensor_set in that state trips GGML's
    // bounds assertion and aborts the validator.  Treat it as an unsupported
    // synthetic probe shape instead; the estimator can then fall back to another
    // source of evidence without pretending this graph was measured.
    if (encoded.size() > ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    bytes += static_cast<double>(encoded.size());
    flops += 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    return true;
}

bool set_active_encoded_weights(
    ggml_tensor * tensor,
    enum ggml_type type,
    int64_t rows,
    int64_t cols,
    EncodedWeightCache & cache,
    double & bytes,
    double & flops) {
    // MoE probes route deterministically to expert ids 0..experts_used-1. The
    // full expert tensor must still exist so the GGML_OP_MUL_MAT_ID graph has
    // the same 3D expert-pool shape as llama.cpp, but initializing every unused
    // expert makes deep l4/l8 validation spend minutes in CPU-side quantization
    // before the timed graph even starts. Populate only the contiguous active
    // expert rows that the synthetic ids can read. The byte/flop counters here
    // also describe active traffic, not resident model size, matching the
    // model-fit active-expert accounting.
    const std::vector<uint8_t> & encoded = cached_encoded_weights(cache, type, rows, cols);
    if (encoded.size() > ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    bytes += static_cast<double>(encoded.size());
    flops += 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    return true;
}

bool set_f32_weights(
    ggml_tensor * tensor,
    int64_t rows,
    int64_t cols,
    uint32_t salt,
    double & bytes,
    double & flops) {
    std::vector<float> weights = deterministic_f32(rows * cols, salt);
    ggml_backend_tensor_set(tensor, weights.data(), 0, weights.size() * sizeof(float));
    bytes += static_cast<double>(weights.size() * sizeof(float));
    flops += 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    return true;
}

ggml_tensor * rms_norm_attention_projection(
    ggml_context * ctx,
    ggml_tensor * projection,
    int64_t total_width,
    int64_t norm_head_width) {
    if (norm_head_width > 0 && norm_head_width < total_width && total_width % norm_head_width == 0) {
        // llama.cpp's Q/K norm tensors are not normalized over the flattened
        // residual width. `build_qkv()` leaves Q and K shaped as
        // `[head_dim, heads, tokens]`, and `build_norm()` applies RMSNorm over
        // `ne[0]`, i.e. one attention head at a time. That matters for the
        // synthetic fit probe because backend kernels and graph scheduling see
        // many small head-width norms, not one hidden-width norm. Use the GGUF
        // head width when it is available, then reshape back to the flattened
        // vector shape consumed by this compact attention proxy.
        ggml_tensor * shaped = ggml_reshape_3d(
            ctx,
            projection,
            norm_head_width,
            total_width / norm_head_width,
            1);
        ggml_tensor * normed = ggml_rms_norm(ctx, shaped, 1e-5f);
        return ggml_reshape_2d(ctx, normed, total_width, 1);
    }
    return ggml_rms_norm(ctx, projection, 1e-5f);
}

bool run_llama_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    if (!dense_llama_shape_supported(type, hidden, kv_width, ffn)) {
        return false;
    }
    const bool use_q_norm = (graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0;
    const bool use_k_norm = (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0;
    const bool use_post_attention_norm = (graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0;
    const bool use_post_ffn_norm = (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0;
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(80 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int64_t layer = 0; layer < layers; ++layer) {
        // Keep the synthetic dense block close to llama.cpp's source graph, not
        // just to its resident tensor bytes. `build_llama()` normalizes the
        // residual stream before attention, adds the attention result back into
        // the residual stream, normalizes again before FFN, then adds the FFN
        // result. Earlier probes skipped the RMSNorm/residual structure and
        // therefore measured mostly quantized matvec throughput. That was
        // enough on some Metal rows but overestimated CUDA rows where the
        // quantized matvecs are very fast and the surrounding graph work is no
        // longer hidden under memory traffic.
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, kv_width), attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, kv_width), attn_input);
        if (use_q_norm) {
            q = rms_norm_attention_projection(ctx, q, hidden, norm_head_width);
        }
        if (use_k_norm) {
            k = rms_norm_attention_projection(ctx, k, kv_width, norm_head_width);
        }
        // llama.cpp's `llm_graph_context::build_attn()` does not leave Q/K/V
        // as anonymous dependencies under the final attention output. It calls
        // `ggml_build_forward_expand()` on the Q, K, and V projection nodes
        // before building attention, with an in-source note that this prevents
        // reordering and reduces graph splits. That scheduler shape matters for
        // decode probes: if this synthetic graph only expands from the final
        // output, Metal/CUDA can schedule a graph that is source-plausible at
        // the op level but not the graph llama.cpp actually submits. Keep the
        // probe source-shaped at the graph-boundary level too.
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        // The output projection consumes the result of attention, not a literal
        // elementwise blend of the raw Q/K/V projection vectors. We use Q as the
        // hidden-width proxy for that attention result and keep K/V scheduled
        // through the explicit graph expansion above. Earlier probes used
        // `q + k + v` when `kv_width == hidden`; that serialized the output
        // projection behind synthetic elementwise dependencies that are not in
        // llama.cpp's decode graph and made full-width attention probes too
        // pessimistic on Metal. KV-cache read traffic is accounted separately in
        // model-fit's metadata estimator, where it can scale with workload
        // prompt/context length without making this graph probe allocate a
        // production-sized cache.
        ggml_tensor * attn = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), q);
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, attn, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, attn);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        // llama.cpp's dense SWIGLU FFN path (`build_ffn(..., LLM_FFN_SILU,
        // LLM_FFN_PAR, ...)`) does not lower gate/up activation to a plain
        // elementwise multiply. It uses the source-visible GGML_SWIGLU_SPLIT op
        // before the down projection. The decode probe needs the same graph shape
        // because Metal/CUDA schedule and sometimes fuse these skinny activation
        // nodes differently from an isolated `ggml_mul`.
        ggml_tensor * gated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
            output = ggml_add(ctx, residual_after_attn, down);
        } else {
            output = ggml_add(ctx, residual_after_attn, down);
        }
        // For GQA/MQA the K/V projections are narrower than the hidden residual
        // stream. We still need those projections to be scheduled because
        // llama.cpp explicitly expands Q, K, and V before it builds attention.
        // The `ggml_build_forward_expand(graph, k/v)` calls above already do
        // that. Do not attach synthetic `sum(k)` / `sum(v)` dependencies to the
        // final output: those reductions are not part of llama decode and they
        // can dominate small Metal graph probes, which would make the metadata
        // estimator learn the benchmark artifact instead of the source graph.
    }
    ggml_set_name(input, "ggml_decode_llama_graph_input");
    ggml_set_name(output, "ggml_decode_llama_graph_output");
    ggml_set_output(output);

    ggml_build_forward_expand(graph, output);

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    std::vector<float> input_f32 = deterministic_f32(hidden, 101);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    EncodedWeightCache weight_cache;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != type || t->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t rows = t->ne[1];
        const int64_t cols = t->ne[0];
        if (!set_encoded_weights(t, type, rows, cols, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }
    bytes += static_cast<double>(layers * (4 * hidden + ffn) * sizeof(float));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        ffn,
        hidden,
        bytes,
        flops,
        graph_features,
        GRAPH_WARMUP_RUNS,
        GRAPH_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

bool run_linear_attention_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t hidden,
    int64_t qkv_width,
    int64_t gate_width,
    int64_t state_width,
    int64_t output_input_width,
    int64_t ffn,
    int64_t recurrent_layers,
    int64_t full_attention_layers,
    int64_t kv_width,
    int graph_features,
    int64_t norm_head_width,
    ProbeResult & result) {
    const int64_t recurrent = std::max<int64_t>(1, recurrent_layers);
    const int64_t full_attention = std::max<int64_t>(0, full_attention_layers);
    const bool use_q_norm = (graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0;
    const bool use_k_norm = (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0;
    const bool use_post_attention_norm = (graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0;
    const bool use_post_ffn_norm = (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0;
    const int64_t safe_qkv = std::max<int64_t>(1, qkv_width);
    const int64_t safe_gate = std::max<int64_t>(1, gate_width);
    const int64_t safe_state = std::max<int64_t>(1, state_width);
    const int64_t safe_output_input = std::max<int64_t>(1, output_input_width);
    const int64_t safe_kv = std::max<int64_t>(1, std::min(kv_width, hidden));
    if (!matrix_shape_supported(type, safe_qkv, hidden)
        || !matrix_shape_supported(type, safe_gate, hidden)
        || !matrix_shape_supported(type, safe_state, hidden)
        || !matrix_shape_supported(type, hidden, safe_output_input)
        || !matrix_shape_supported(type, ffn, hidden)
        || !matrix_shape_supported(type, hidden, ffn)
        || !dense_llama_shape_supported(type, hidden, safe_kv, ffn)) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(96 * (recurrent + full_attention))
        + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int64_t layer = 0; layer < recurrent; ++layer) {
        // Linear/recurrent attention blocks in llama.cpp are not simply dense
        // Q/K/V attention with a cheaper KV cache. In Qwen3.5-style graphs,
        // `build_layer_attn_linear()` submits independent source-visible
        // projections for qkv, z/gate, beta, alpha, and the final linear
        // attention output, with recurrent/SSM elementwise work between them.
        // The fit estimator needs a probe with that graph topology. This probe
        // intentionally stays structural: tensor-role widths come from GGUF
        // metadata and it does not embed a family/backend correction.
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * qkv = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_qkv), attn_input);
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_gate), attn_input);
        ggml_tensor * beta = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_state), attn_input);
        ggml_tensor * alpha = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_state), attn_input);
        ggml_build_forward_expand(graph, qkv);
        ggml_build_forward_expand(graph, gate);
        ggml_build_forward_expand(graph, beta);
        ggml_build_forward_expand(graph, alpha);

        ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha);
        ggml_tensor * recurrent_gate = ggml_mul(ctx, alpha_softplus, beta);
        ggml_tensor * qkv_activated = ggml_silu(ctx, qkv);
        ggml_tensor * state_proxy = ggml_mul(ctx, recurrent_gate, recurrent_gate);
        ggml_build_forward_expand(graph, state_proxy);

        // llama.cpp normalizes the recurrent attention output with the z/gate
        // projection before `ssm_out`. The full gated norm is family-specific;
        // this compact proxy keeps the source-visible dependency and
        // elementwise scheduling without manufacturing extra weight bytes.
        ggml_tensor * gate_reduced = ggml_mean(ctx, gate);
        ggml_tensor * gated_qkv = ggml_mul(ctx, qkv_activated, gate_reduced);
        ggml_tensor * output_input = gated_qkv;
        if (safe_output_input < safe_qkv) {
            output_input = ggml_view_2d(
                ctx,
                gated_qkv,
                safe_output_input,
                1,
                ggml_row_size(gated_qkv->type, safe_output_input),
                0);
        }
        ggml_tensor * projected = ggml_mul_mat(
            ctx,
            ggml_new_tensor_2d(ctx, type, safe_output_input, hidden),
            ggml_reshape_2d(ctx, output_input, safe_output_input, 1));
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, projected, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, projected);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * ffn_gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * ffn_up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * gated = ggml_swiglu_split(ctx, ffn_gate, ffn_up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
        }
        output = ggml_add(ctx, residual_after_attn, down);
    }

    for (int64_t layer = 0; layer < full_attention; ++layer) {
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_kv), attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_kv), attn_input);
        if (use_q_norm) {
            q = rms_norm_attention_projection(ctx, q, hidden, norm_head_width);
        }
        if (use_k_norm) {
            k = rms_norm_attention_projection(ctx, k, safe_kv, norm_head_width);
        }
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        ggml_tensor * attn = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), q);
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, attn, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, attn);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * gated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
        }
        output = ggml_add(ctx, residual_after_attn, down);
    }

    ggml_set_name(input, "ggml_decode_linear_attn_graph_input");
    ggml_set_name(output, "ggml_decode_linear_attn_graph_output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    std::vector<float> input_f32 = deterministic_f32(hidden, 131);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    EncodedWeightCache weight_cache;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != type || t->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t rows = t->ne[1];
        const int64_t cols = t->ne[0];
        if (!set_encoded_weights(t, type, rows, cols, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }
    bytes += static_cast<double>((recurrent + full_attention) * (5 * hidden + ffn) * sizeof(float));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        ffn,
        hidden,
        bytes,
        flops,
        graph_features,
        GRAPH_WARMUP_RUNS,
        GRAPH_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

bool run_moe_mul_mat_id_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    ProbeResult & result) {
    constexpr int64_t expert_count = 128;
    constexpr int64_t experts_used = 8;
    constexpr int64_t expert_width = 768;
    constexpr int64_t hidden = 2048;
    constexpr int64_t tokens = 1;
    if (!moe_shape_supported(type, hidden, expert_width)) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, experts_used, tokens);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, experts_used, tokens);
    ggml_tensor * output = ggml_mul_mat_id(ctx, experts, input, ids);
    ggml_set_name(experts, "ggml_decode_moe_experts");
    ggml_set_name(ids, "ggml_decode_moe_ids");
    ggml_set_name(input, "ggml_decode_moe_input");
    ggml_set_name(output, "ggml_decode_moe_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double ignored_resident_bytes = 0.0;
    double flops = 0.0;
    EncodedWeightCache weight_cache;
    if (!set_encoded_weights(
        experts,
        type,
        expert_width * expert_count,
        hidden,
        weight_cache,
        ignored_resident_bytes,
        flops)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    double bytes = static_cast<double>(ggml_row_size(type, hidden) * expert_width * experts_used);
    std::vector<float> input_f32 = deterministic_f32(hidden * experts_used * tokens, 223);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    std::vector<int32_t> ids_i32(static_cast<size_t>(experts_used * tokens));
    for (int64_t i = 0; i < experts_used * tokens; ++i) {
        ids_i32[static_cast<size_t>(i)] = static_cast<int32_t>(i % experts_used);
    }
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    bytes += static_cast<double>(ids_i32.size() * sizeof(int32_t));
    bytes += static_cast<double>(expert_width * experts_used * tokens * sizeof(float));
    flops = 2.0 * static_cast<double>(expert_width)
        * static_cast<double>(hidden)
        * static_cast<double>(experts_used)
        * static_cast<double>(tokens);
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        expert_width,
        hidden,
        bytes,
        flops);
    ggml_free(ctx);
    return ok;
}

bool run_moe_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t repeat_layers,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    constexpr int64_t tokens = 1;
    if (expert_count <= 0 || experts_used <= 0 || expert_width <= 0 || hidden <= 0) {
        return false;
    }
    if (experts_used > expert_count || expert_count > MAX_MODEL_SHAPED_MOE_EXPERTS) {
        return false;
    }
    if (!moe_shape_supported(type, hidden, expert_width)) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(96 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    std::vector<ggml_tensor *> routers;
    std::vector<ggml_tensor *> up_experts;
    std::vector<ggml_tensor *> gate_experts;
    std::vector<ggml_tensor *> down_experts;
    routers.reserve(static_cast<size_t>(layers));
    up_experts.reserve(static_cast<size_t>(layers));
    gate_experts.reserve(static_cast<size_t>(layers));
    down_experts.reserve(static_cast<size_t>(layers));
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, tokens);
    ggml_tensor * output = input;

    // Mirror the source-level shape of llama.cpp's `build_moe_ffn()` for the
    // common SILU routed-expert path used by OLMoE/Qwen-style GGUFs:
    //
    //   gate logits -> softmax -> argsort_top_k -> get_rows(weights)
    //   -> MUL_MAT_ID(up/gate) -> SWIGLU_SPLIT -> MUL_MAT_ID(down)
    //   -> multiply by routed weights -> view/add selected experts
    //
    // This is still a synthetic hardware probe: it does not run a GGUF model or
    // inspect model names. The dimensions come from GGUF metadata, and the
    // graph operations come from llama.cpp source. The cap on expert_count is a
    // resource guard so a validation probe cannot accidentally allocate a
    // production-scale expert pool just to measure one hardware row.
    //
    // Deep validation repeats this routed FFN subgraph l4/l8 in a single
    // scheduled graph. That is the MoE analogue of the dense stacked llama
    // graph probes: it measures source-visible graph depth and scheduler
    // amortization without fitting a multiplier to observed model throughput.
    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, expert_count);
        ggml_tensor * up_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * gate_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * down_exps = ggml_new_tensor_3d(ctx, type, expert_width, hidden, expert_count);
        routers.push_back(router);
        up_experts.push_back(up_exps);
        gate_experts.push_back(gate_exps);
        down_experts.push_back(down_exps);

        ggml_tensor * logits = ggml_mul_mat(ctx, router, output);
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * ids = ggml_argsort_top_k(ctx, probs, experts_used);
        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, expert_count, tokens), ids);
        ggml_tensor * routed_input = ggml_reshape_3d(ctx, output, hidden, 1, tokens);
        ggml_tensor * up = ggml_mul_mat_id(ctx, up_exps, routed_input, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_exps, routed_input, ids);
        ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, down_exps, activated, ids);
        experts = ggml_mul(ctx, experts, weights);
        ggml_tensor * layer_output = nullptr;
        for (int64_t expert = 0; expert < experts_used; ++expert) {
            ggml_tensor * expert_view = ggml_view_2d(
                ctx,
                experts,
                hidden,
                tokens,
                experts->nb[2],
                expert * experts->nb[1]);
            layer_output = layer_output == nullptr ? expert_view : ggml_add(ctx, layer_output, expert_view);
        }
        output = layer_output;
    }
    ggml_set_name(input, "ggml_decode_moe_graph_input");
    ggml_set_name(output, "ggml_decode_moe_graph_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double ignored_resident_bytes = 0.0;
    double ignored_flops = 0.0;
    EncodedWeightCache weight_cache;
    for (int64_t layer = 0; layer < layers; ++layer) {
        set_f32_weights(
            routers[static_cast<size_t>(layer)],
            expert_count,
            hidden,
            307 + static_cast<uint32_t>(layer * 17),
            ignored_resident_bytes,
            ignored_flops);
        if (!set_active_encoded_weights(
            up_experts[static_cast<size_t>(layer)],
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            ignored_resident_bytes,
            ignored_flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            gate_experts[static_cast<size_t>(layer)],
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            ignored_resident_bytes,
            ignored_flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            down_experts[static_cast<size_t>(layer)],
            type,
            hidden * experts_used,
            expert_width,
            weight_cache,
            ignored_resident_bytes,
            ignored_flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    double bytes = static_cast<double>(layers)
        * (static_cast<double>(expert_count * hidden * sizeof(float))
        + static_cast<double>(expert_count * tokens * sizeof(float))
        + static_cast<double>(expert_count * tokens * sizeof(float))
        + 2.0 * static_cast<double>(ggml_row_size(type, hidden) * expert_width * experts_used)
        + static_cast<double>(ggml_row_size(type, expert_width) * hidden * experts_used));
    std::vector<float> input_f32 = deterministic_f32(hidden * tokens, 331);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    bytes += static_cast<double>(layers)
        * static_cast<double>(experts_used * tokens * sizeof(int32_t));
    bytes += static_cast<double>(layers)
        * static_cast<double>((2 * expert_width + hidden) * experts_used * tokens * sizeof(float));
    bytes += static_cast<double>(layers)
        * static_cast<double>(hidden * experts_used * tokens * sizeof(float));
    const double flops = static_cast<double>(layers)
        * (2.0 * static_cast<double>(expert_count)
        * static_cast<double>(hidden)
        * static_cast<double>(tokens)
        + 6.0 * static_cast<double>(expert_width)
        * static_cast<double>(hidden)
        * static_cast<double>(experts_used)
        * static_cast<double>(tokens));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        expert_width,
        hidden,
        bytes,
        flops);
    ggml_free(ctx);
    return ok;
}

bool run_moe_block_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t kv_width,
    int64_t repeat_layers,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    constexpr int64_t tokens = 1;
    if (expert_count <= 0 || experts_used <= 0 || expert_width <= 0 || hidden <= 0 || kv_width <= 0) {
        return false;
    }
    if (experts_used > expert_count || expert_count > MAX_MODEL_SHAPED_MOE_EXPERTS) {
        return false;
    }
    kv_width = std::min(kv_width, hidden);
    if (!dense_llama_shape_supported(type, hidden, kv_width, expert_width)
        || !moe_shape_supported(type, hidden, expert_width)) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(160 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    struct LayerTensors {
        ggml_tensor * wq;
        ggml_tensor * wk;
        ggml_tensor * wv;
        ggml_tensor * wo;
        ggml_tensor * router;
        ggml_tensor * up_experts;
        ggml_tensor * gate_experts;
        ggml_tensor * down_experts;
    };
    std::vector<LayerTensors> layer_tensors;
    layer_tensors.reserve(static_cast<size_t>(layers));

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);

    // This probe intentionally models a sparse transformer block, not only the
    // routed expert inner loop. In llama.cpp an OLMoE/Qwen-MoE style decode
    // layer still pays the attention projections, scheduler boundaries,
    // residual adds, RMSNorms, and then the `build_moe_ffn()` routed path. The
    // older `moe_graph` row below remains useful diagnostic evidence for
    // GGML_OP_MUL_MAT_ID itself, but using that row as the sole estimator input
    // made sparse models look too fast because attention and graph depth were
    // composed from unlike probes. This block row keeps the operations and
    // dimensions source-shaped while still avoiding a production KV-cache
    // allocation; KV-cache read traffic is workload dependent and is charged by
    // model-fit from GGUF metadata.
    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * wq = ggml_new_tensor_2d(ctx, type, hidden, hidden);
        ggml_tensor * wk = ggml_new_tensor_2d(ctx, type, hidden, kv_width);
        ggml_tensor * wv = ggml_new_tensor_2d(ctx, type, hidden, kv_width);
        ggml_tensor * wo = ggml_new_tensor_2d(ctx, type, hidden, hidden);
        ggml_tensor * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, expert_count);
        ggml_tensor * up_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * gate_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * down_exps = ggml_new_tensor_3d(ctx, type, expert_width, hidden, expert_count);
        layer_tensors.push_back(LayerTensors{
            wq,
            wk,
            wv,
            wo,
            router,
            up_exps,
            gate_exps,
            down_exps,
        });

        ggml_tensor * attn_input = ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, wq, attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, wk, attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, wv, attn_input);
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        ggml_tensor * attn = ggml_mul_mat(ctx, wo, q);
        ggml_tensor * attn_residual = ggml_add(ctx, output, attn);

        ggml_tensor * ffn_input = ggml_rms_norm(ctx, attn_residual, 1e-5f);
        ggml_tensor * logits = ggml_mul_mat(ctx, router, ffn_input);
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * ids = ggml_argsort_top_k(ctx, probs, experts_used);
        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, expert_count, tokens), ids);
        ggml_build_forward_expand(graph, weights);
        ggml_tensor * routed_input = ggml_reshape_3d(ctx, ffn_input, hidden, 1, tokens);
        ggml_tensor * up = ggml_mul_mat_id(ctx, up_exps, routed_input, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_exps, routed_input, ids);
        ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, down_exps, activated, ids);
        experts = ggml_mul(ctx, experts, weights);
        ggml_build_forward_expand(graph, experts);
        ggml_tensor * moe_output = nullptr;
        for (int64_t expert = 0; expert < experts_used; ++expert) {
            ggml_tensor * expert_view = ggml_view_2d(
                ctx,
                experts,
                hidden,
                tokens,
                experts->nb[2],
                expert * experts->nb[1]);
            ggml_build_forward_expand(graph, expert_view);
            moe_output = moe_output == nullptr ? expert_view : ggml_add(ctx, moe_output, expert_view);
            if (moe_output != expert_view) {
                ggml_build_forward_expand(graph, moe_output);
            }
        }
        output = ggml_add(ctx, attn_residual, moe_output);
    }
    ggml_set_name(input, "ggml_decode_moe_block_graph_input");
    ggml_set_name(output, "ggml_decode_moe_block_graph_output");
    ggml_set_output(output);

    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    EncodedWeightCache weight_cache;
    for (int64_t layer = 0; layer < layers; ++layer) {
        const LayerTensors & tensors = layer_tensors[static_cast<size_t>(layer)];
        if (!set_encoded_weights(tensors.wq, type, hidden, hidden, weight_cache, bytes, flops)
            || !set_encoded_weights(tensors.wk, type, kv_width, hidden, weight_cache, bytes, flops)
            || !set_encoded_weights(tensors.wv, type, kv_width, hidden, weight_cache, bytes, flops)
            || !set_encoded_weights(tensors.wo, type, hidden, hidden, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        set_f32_weights(
            tensors.router,
            expert_count,
            hidden,
            607 + static_cast<uint32_t>(layer * 19),
            bytes,
            flops);
        if (!set_active_encoded_weights(
            tensors.up_experts,
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            bytes,
            flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            tensors.gate_experts,
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            bytes,
            flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            tensors.down_experts,
            type,
            hidden * experts_used,
            expert_width,
            weight_cache,
            bytes,
            flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<float> input_f32 = deterministic_f32(hidden, 631);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    bytes += static_cast<double>(layers)
        * static_cast<double>(
            (8 * hidden + 2 * kv_width + expert_count + (3 * experts_used)) * sizeof(float)
            + experts_used * sizeof(int32_t));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        expert_width,
        hidden,
        bytes,
        flops,
        0,
        GRAPH_WARMUP_RUNS,
        GRAPH_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

std::string results_json(const std::vector<ProbeResult> & results) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        const ProbeResult & result = results[i];
        if (i > 0) {
            out << ",";
        }
        out << "{\"name\":\"" << result.name << "\","
            << "\"tensor_type\":\"" << result.tensor_type << "\","
            << "\"rows\":" << result.rows << ","
            << "\"cols\":" << result.cols << ","
            << "\"batch_tokens\":1,"
            << "\"graph_features\":" << result.graph_features << ","
            << "\"effective_gbps\":" << result.effective_gbps << ","
            << "\"tflops\":" << result.tflops << ","
            << "\"elapsed_ms\":" << result.elapsed_ms << ","
            << "\"runs\":" << result.runs << "}";
    }
    out << "]";
    return out.str();
}

} // namespace

extern "C" char * mesh_llm_gpu_bench_ggml_output_projection_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t vocab,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported output projection probe tensor type");
        return nullptr;
    }
    if (hidden <= 0 || vocab <= 0) {
        set_error(error_out, "output projection probe dimensions must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_matvec_output_"
         << vocab
         << "_"
         << hidden;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            ProbeShape{nullptr, vocab, hidden},
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML output projection probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_decode_probe_json(
    int backend_kind,
    int probe_depth,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }
    const bool deep_probes = probe_depth == PROBE_DEPTH_DEEP;

    std::vector<ProbeResult> results;
    ProbeResult result{};
    for (const ProbeShape & shape : DECODE_SHAPES) {
        std::string f16_name = std::string("ggml_decode_f16_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_F16, f16_name.c_str(), "f16", shape, result)) {
            results.push_back(result);
        }
        std::string q8_name = std::string("ggml_decode_q8_0_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_Q8_0, q8_name.c_str(), "q8_0", shape, result)) {
            results.push_back(result);
        }
        std::string q4_name = std::string("ggml_decode_q4_k_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_Q4_K, q4_name.c_str(), "q4_k", shape, result)) {
            results.push_back(result);
        }
        std::string q6_name = std::string("ggml_decode_q6_k_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_Q6_K, q6_name.c_str(), "q6_k", shape, result)) {
            results.push_back(result);
        }
    }
    for (const ProbeShape & shape : LLAMA_GRAPH_SHAPES) {
        std::string q4_graph_name = std::string("ggml_decode_q4_k_llama_graph_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q4_K,
                q4_graph_name.c_str(),
                "q4_k",
                shape.rows,
                shape.rows,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
        // Deep validation can request a bounded graph-depth curve. Dense
        // llama.cpp decode submits many transformer blocks in one scheduled
        // graph, so source-shaped probes at l4/l8 let the estimator observe
        // how scheduler/allocator/kernel-launch behavior changes with graph
        // depth without constructing a full model-sized synthetic graph. The
        // full-depth experiment was too expensive even for a 28-layer 3B model,
        // and l16 was still too slow for a smoke-grade Metal deep benchmark on
        // an M1 Ultra. The curve therefore deliberately stops at small fixed
        // depths that can be gathered repeatedly.
        //
        // These rows are intentionally not part of the standard hardware
        // fingerprint: first-run Metal pipeline compilation plus deeper graphs
        // made `mesh-llm gpus detect` exceed its operator-facing timeout on an
        // M1 Ultra. The default benchmark should stay fast and broadly
        // portable; slow probes belong to validation.
        //
        // We also do not emit Q8 stack rows. Validation on the narrow SmolLM2 Q8
        // model falsified them as portable estimator inputs because they
        // over-amortized graph work relative to real llama.cpp decode.
        if (deep_probes && ((shape.rows == 2560 && shape.cols == 9728) ||
            (shape.rows == 4096 && shape.cols == 12288))) {
            for (int64_t layers : DEEP_LLAMA_GRAPH_LAYERS) {
                std::string q4_graph_l_name =
                    std::string("ggml_decode_q4_k_llama_graph_l")
                    + std::to_string(layers)
                    + "_"
                    + shape.suffix;
                if (run_llama_graph_probe(
                        backend,
                        GGML_TYPE_Q4_K,
                        q4_graph_l_name.c_str(),
                        "q4_k",
                        shape.rows,
                        shape.rows,
                        shape.cols,
                        layers,
                        0,
                        0,
                        result)) {
                    results.push_back(result);
                }
            }
        }
        if ((shape.rows != 768 || shape.cols != 2048) &&
            (shape.rows != 1024 || shape.cols != 4096) &&
            (shape.rows != 4096 || shape.cols != 12288)) {
            continue;
        }
        std::string q8_graph_name = std::string("ggml_decode_q8_0_llama_graph_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q8_0,
                q8_graph_name.c_str(),
                "q8_0",
                shape.rows,
                shape.rows,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
        std::string q6_graph_name = std::string("ggml_decode_q6_k_llama_graph_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q6_K,
                q6_graph_name.c_str(),
                "q6_k",
                shape.rows,
                shape.rows,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
    }
    for (const ProbeShape & shape : LLAMA_GQA_GRAPH_SHAPES) {
        constexpr int64_t kv_width = 1024;
        if (shape.rows <= kv_width) {
            continue;
        }
        std::string q4_graph_name = std::string("ggml_decode_q4_k_llama_graph_gqa_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q4_K,
                q4_graph_name.c_str(),
                "q4_k",
                shape.rows,
                kv_width,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
        if (deep_probes && shape.rows == 2560 && shape.cols == 9728) {
            for (int64_t layers : DEEP_LLAMA_GRAPH_LAYERS) {
                std::string q4_graph_l_name =
                    std::string("ggml_decode_q4_k_llama_graph_l")
                    + std::to_string(layers)
                    + "_gqa_"
                    + shape.suffix;
                if (run_llama_graph_probe(
                        backend,
                        GGML_TYPE_Q4_K,
                        q4_graph_l_name.c_str(),
                        "q4_k",
                        shape.rows,
                        kv_width,
                        shape.cols,
                        layers,
                        0,
                        0,
                        result)) {
                    results.push_back(result);
                }
            }
        }
    }
    if (run_moe_mul_mat_id_probe(
            backend,
            GGML_TYPE_Q4_K,
            "ggml_decode_moe_mul_mat_id_q4_k_128x8_768x2048",
            "q4_k",
            result)) {
        results.push_back(result);
    }
    if (run_moe_mul_mat_id_probe(
            backend,
            GGML_TYPE_Q6_K,
            "ggml_decode_moe_mul_mat_id_q6_k_128x8_768x2048",
            "q6_k",
            result)) {
        results.push_back(result);
    }
    if (run_moe_graph_probe(
            backend,
            GGML_TYPE_Q4_K,
            "ggml_decode_moe_graph_q4_k_128x8_768x2048",
            "q4_k",
            128,
            8,
            768,
            2048,
            1,
            result)) {
        results.push_back(result);
    }
    if (run_moe_graph_probe(
            backend,
            GGML_TYPE_Q6_K,
            "ggml_decode_moe_graph_q6_k_128x8_768x2048",
            "q6_k",
            128,
            8,
            768,
            2048,
            1,
            result)) {
        results.push_back(result);
    }

    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML decode probe did not produce supported matvec results");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_moe_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t repeat_layers,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported MoE probe tensor type");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_moe_graph_"
         << "l"
         << std::max<int64_t>(1, repeat_layers)
         << "_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_"
         << expert_count
         << "x"
         << experts_used
         << "_"
         << expert_width
         << "x"
         << hidden;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_moe_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            expert_count,
            experts_used,
            expert_width,
            hidden,
            repeat_layers,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML MoE graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_moe_block_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t kv_width,
    int64_t repeat_layers,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported MoE block probe tensor type");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_moe_block_graph_"
         << "l"
         << std::max<int64_t>(1, repeat_layers)
         << "_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_"
         << expert_count
         << "x"
         << experts_used
         << "_"
         << expert_width
         << "x"
         << hidden;
    if (kv_width > 0 && kv_width < hidden) {
        name << "_kv" << kv_width;
    }

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_moe_block_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            expert_count,
            experts_used,
            expert_width,
            hidden,
            kv_width,
            repeat_layers,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML MoE block graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported dense graph probe tensor type");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0) {
        set_error(error_out, "dense graph probe dimensions must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_llama_graph";
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_llama_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            repeat_layers,
            graph_features,
            norm_head_width,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_linear_attention_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t qkv_width,
    int64_t gate_width,
    int64_t state_width,
    int64_t output_input_width,
    int64_t ffn,
    int64_t recurrent_layers,
    int64_t full_attention_layers,
    int64_t kv_width,
    int graph_features,
    int64_t norm_head_width,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported linear attention graph probe tensor type");
        return nullptr;
    }
    if (hidden <= 0 || qkv_width <= 0 || gate_width <= 0 || state_width <= 0 ||
        output_input_width <= 0 || ffn <= 0 || recurrent_layers <= 0 || kv_width <= 0) {
        set_error(error_out, "linear attention graph probe dimensions must be positive");
        return nullptr;
    }
    if (output_input_width > qkv_width) {
        set_error(error_out, "linear attention output input width cannot exceed qkv width");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_linear_attn_graph"
         << "_r" << recurrent_layers
         << "_f" << std::max<int64_t>(0, full_attention_layers);
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    name << "_h" << hidden
         << "_qkv" << qkv_width
         << "_gate" << gate_width
         << "_state" << state_width
         << "_out" << output_input_width
         << "_kv" << kv_width
         << "_ffn" << ffn;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_linear_attention_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            hidden,
            qkv_width,
            gate_width,
            state_width,
            output_input_width,
            ffn,
            recurrent_layers,
            full_attention_layers,
            kv_width,
            graph_features,
            norm_head_width,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML linear attention graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" void mesh_llm_gpu_bench_ggml_decode_probe_free(void * ptr) {
    std::free(ptr);
}
