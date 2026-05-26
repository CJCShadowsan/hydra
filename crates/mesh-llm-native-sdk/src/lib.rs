//! Prebuilt native mesh-llm runtime.
//!
//! This crate's job is to *fetch and link* the matching `libmeshllm_ffi`
//! prebuilt shared library for the consumer's target platform and selected
//! backend. The actual download + link work happens in `build.rs`; the Rust
//! API surface lives elsewhere (currently `mesh-llm-ffi`'s UniFFI-generated
//! bindings; in the future, possibly a Rust-native wrapper layered on top).
//!
//! Consumers should not depend on this crate directly. Instead, depend on
//! `mesh-llm-api-server` with the appropriate `native-*` feature, which
//! pulls this crate in transparently.

// Force the linker to keep `libmeshllm_ffi` linked into the consumer's
// final binary. `build.rs` emits `cargo:rustc-link-search=...` so the
// linker can find the static archive; this `#[link]` attribute forces a
// `-l meshllm_ffi` even when the consumer hasn't yet referenced a symbol.
//
// `kind = "static"` matches the file the build script extracts on every
// platform — same shape as Swift's xcframework, which also ships a
// static archive that gets linked into the consumer app.
#[link(name = "meshllm_ffi", kind = "static")]
unsafe extern "C" {
    // We don't reference any FFI symbols here; the attribute alone is
    // enough to keep the link directive alive in the consumer.
}

/// Bring a tiny FFI symbol into scope so the consumer can sanity-check
/// that linking actually worked end-to-end. Useful for tests and the
/// faux-consumer trial.
///
/// Returns the UniFFI contract version baked into the linked
/// `libmeshllm_ffi`. Stable, no-arg, no allocation; safe to call from any
/// thread at any time.
pub fn uniffi_contract_version() -> u32 {
    unsafe extern "C" {
        fn ffi_meshllm_ffi_uniffi_contract_version() -> u32;
    }
    unsafe { ffi_meshllm_ffi_uniffi_contract_version() }
}
