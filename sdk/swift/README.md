# MeshLLM Swift SDK

Swift Package for connecting to mesh-llm meshes from iOS, Mac Catalyst, and macOS apps.

## Installation

Add to your app's `Package.swift` using a tagged release:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MeshLLM", package: "mesh-llm"),
        ]
    ),
]
```

The repo root is the Swift package entrypoint. Tagged releases resolve the
prebuilt XCFramework automatically through SwiftPM.

For development from a local checkout, build the native artifact first:

```bash
./sdk/swift/scripts/build-xcframework.sh
```

That generates `sdk/swift/Generated/MeshLLMFFI.xcframework`, which the root
Swift package picks up automatically for the real UniFFI-backed implementation.

Normal SDK builds and tests require the UniFFI-backed XCFramework. The package
does not ship a pure Swift fallback because the SDK must exercise the native
runtime it exposes.

## Usage

```swift
import MeshLLM

let ownerKeypair = generateOwnerKeypairHex()
let node = try Node(
    inviteToken: InviteToken("your-invite-token"),
    ownerKeypairBytesHex: ownerKeypair
)

let recommended = try await node.models.recommended()
let serving = try await node.serving.status()
try await node.start()

let models = try await node.inference.listModels()
let request = ChatRequest(model: models[0].id, messages: [
    ChatMessage(role: "user", content: "Hello!")
])

for try await event in node.inference.chatStream(request) {
    switch event {
    case .tokenDelta(_, let delta):
        print(delta, terminator: "")
    case .completed:
        print()
    default:
        break
    }
}
```

## Local Mac Inference Example

The Swift example uses the real UniFFI-backed SDK path for loading a model and
running inference from the current Mac:

```bash
./sdk/swift/scripts/build-host-macos-xcframework.sh
MESH_SDK_MODEL_REF=Qwen2.5-3B-Instruct-Q4_K_M \
swift run --package-path sdk/swift/example/MeshExampleApp
```

Useful environment overrides:

- `MESH_SDK_MODEL_REF` — catalog, Hugging Face, or local model reference to download/load.
- `MESH_SDK_CACHE_DIR` — Hugging Face cache location.
- `MESH_SDK_RUNTIME_DIR` — runtime scratch directory.
- `MESH_SDK_SKIP_DOWNLOAD=1` — skip `node.models.download` when the model is already installed.
- `MESH_SDK_PROMPT` — prompt text for the local inference request.

The host-enabled XCFramework is required for this path. The FFI node must also
be built with an attached in-process host runtime controller; without that
controller, `node.serving.load(...)` reports serving as unsupported.

## App Store Export Compliance

### Encryption

mesh-llm uses QUIC (via iroh) for transport, which uses TLS 1.3. This constitutes use of encryption.

**Required**: Set `ITSAppUsesNonExemptEncryption = YES` in your app's `Info.plist`.

If your app qualifies for an exemption (e.g., uses only standard encryption), you may set `ITSAppUsesNonExemptEncryption = NO` and provide justification.

### Privacy Manifest

The MeshLLM XCFramework includes a `PrivacyInfo.xcprivacy` manifest declaring:
- `NSPrivacyTracking = false` (no tracking)
- No data collection
- No required-reason API usage

This manifest is embedded inside each `.framework` bundle in the XCFramework, satisfying Apple's requirement since Spring 2024.

### Entitlements

No special entitlements are required. mesh-llm uses standard POSIX sockets via iroh/quinn — no `com.apple.security.network.client` entitlement is needed for macOS (it's allowed by default).

For iOS, network access is allowed by default. No special entitlements needed.

### App Store Submission Checklist

- [ ] Set `ITSAppUsesNonExemptEncryption` in `Info.plist`
- [ ] Verify `PrivacyInfo.xcprivacy` is embedded in XCFramework (run `find MeshLLM.xcframework -name PrivacyInfo.xcprivacy`)
- [ ] No subprocess spawning (mesh-llm SDK never calls `Process()`)
- [ ] No filesystem access for credentials (pass keys via constructor)
- [ ] Implement `reconnect()` in `UIApplication.willEnterForegroundNotification` observer

## iOS Backgrounding

Register for foreground notifications to reconnect after backgrounding:

```swift
NotificationCenter.default.addObserver(
    forName: UIApplication.willEnterForegroundNotification,
    object: nil,
    queue: .main
) { _ in
    Task {
        try? await node.reconnect()
    }
}
```
