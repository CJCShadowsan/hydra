# Lab notes — QUIC stability investigation

Working file. Brief, factual, append as findings come in.

## Setup

3-node private mesh, all v0.66.0 (release builds from `main` HEAD), joined via `--join <invite>`. Same home LAN, stable wifi.

| Node | Hostname | Model | Role |
|---|---|---|---|
| M4 | this | Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m | gateway / driver |
| Studio | BLKJVKK4F19N3 (192.168.86.24) | unsloth/MiniMax-M2.5-GGUF:Q4_K_M | big peer |
| Mini | mac (192.168.86.60) | unsloth/Qwen3.5-9B-GGUF:Q4_K_M | mid peer |

Probe loop in `/tmp/lab/probe.sh` cycles 6 probe types (auto_chat, mesh_chat, direct_studio, direct_mini, mesh_tool, stream_studio) with 5-12s gaps and a 2.5-min quiet pause every 4 iterations. Connection-event watcher in `/tmp/lab/watch-conn.sh` writes a unified CSV from all three nodes' logs. Status snapshot any time: `/tmp/lab/summary.sh`.

## Story of the QUIC keep-alive fix (#566 / f5cf4b86, May 20)

The fix added four lines:

```rust
.keep_alive_interval(10s)
.max_idle_timeout(300s)
.default_path_keep_alive_interval(10s)
.default_path_max_idle_timeout(300s)
```

After reading the iroh 1.0.0-rc.0 source:

- **`keep_alive_interval(10s)` and `max_idle_timeout(300s)` are the real fix.** iroh's default `keep_alive_interval` at the *connection* level is `None` (off) and `max_idle_timeout` is 30s. Long silent inference (MoA reducer call etc.) would hit 30s of silence and the connection would close. Our fix turns connection-level keep-alive on and extends idle to 5 min. **This is doing real work.**
- **`default_path_keep_alive_interval(10s)` and `default_path_max_idle_timeout(300s)` are no-ops.** iroh's builder already initializes these to `HEARTBEAT_INTERVAL = 5s` and `PATH_MAX_IDLE_TIMEOUT = 15s`. Our 10s is > 5s so iroh silently `return self` without applying it. Our 300s is clamped to 15s. The per-path defaults stay at 5s / 15s either way.

So the fix was correctly diagnosed but mis-attributed at the per-path level. The connection-level part is what matters. Worth a small cleanup PR: drop the per-path lines (or pin them at the iroh max so they're explicit no-ops) and document the connection-level rationale clearly.

iroh version timeline:
- 0.97 → 0.98 on 2026-04-21 (cb31b763)
- 0.98 → 1.0.0-rc.0 on 2026-05-11 (0daf9287)
- keep-alive code added on 2026-05-20 (f5cf4b86)

Both 0.98 and 1.0.0-rc.0 ship the same 5s/15s per-path defaults, so the iroh upgrade didn't change that. What did change in the keep-alive fix was strictly connection-level.

## Topology & multipath, in plain language

- iroh uses `quinn` + `noq-proto` multipath.
- A **connection** between two nodes is a logical thing. Under one connection there can be several **paths** (relay URL, LAN IP, holepunched UDP, etc.).
- A path can die; the connection survives if another path is up.
- `LastOpenPath` = "we tried to act on a connection but every path beneath it just closed." Symptom, not cause.
- The cause of a path closing: peer sent CONNECTION_CLOSE on it, OR local per-path idle timer fired, OR underlying UDP socket error, OR multipath layer abandoned a probed-but-unverified path.

## Asymmetric M4 ↔ mini connectivity — deeper dig

Mic's intuition was right: this M4 has incoming-traffic shape that breaks direct from M4→mini. But the precise cause isn't "M4 firewall blocking mesh-llm" — we checked: `socketfilterfw --getappblocked` on the release binary says "Incoming connection is permitted". So mesh-llm itself is allowed.

Facts we collected after the lab v2 restart:

* **All three nodes on Wi-Fi** (en0/en1) on the same 192.168.86.0/24 subnet. No wired uplinks.
* **M4: firewall on, stealth on.** Studio: firewall on, stealth on. Mini: firewall **off**. So firewall config doesn't explain the asymmetry — studio has the same firewall config as M4, and studio↔M4 is rock solid.
* **Tailscale: NOT running anywhere** (`pgrep -lf tailscale` empty on all three). My earlier hypothesis about Tailscale was wrong. The `utun*` interfaces are owned by other macOS daemons.
* **Mini has a macOS VPN system extension running** (`/System/Library/ExtensionKit/Extensions/VPN.appex/Contents/MacOS/VPN`). 6 utun interfaces on mini. Strong candidate for capturing/rerouting UDP packets in a way that interferes with iroh's holepunching to mini specifically.
* macOS versions identical-ish (26.5 / 26.5 / 26.4.1).
* OpenDirectoryD pegged at 37% CPU on mini at the moment of inspection. Probably orthogonal but worth keeping in mind.

After restarting M4 with the new retry binary, **mini's RTT trail no longer flaps direct↔relay** — instead it stays "direct" but the direct RTT climbs steeply: `9ms → 10ms → 607ms → 192ms → 181ms → 257ms → 428ms`. That looks like UDP packet loss / retransmits on the direct path, which iroh's RTT estimator picks up. So direct is alive but degraded. Iroh chose not to fall back to relay this time, probably because no fatal path-level error fired (only RTT degradation).

This is **net better** for the connection-pair: paths don't get torn down and reopened repeatedly. But individual requests on a degraded direct path can still see >300s first-byte timeouts.

Takeaway: mini's VPN system extension is the most likely environmental cause. Asking the user to disable it would confirm. But the wild won't always cooperate — we need to be resilient regardless.

## Topological resilience experiments still worth running

1. **Direction reversal**: have mini initiate connections to M4 (mini becomes the proxy). Does the asymmetry flip with direction, or is it always M4→mini that's broken?
2. **Key rotation**: delete `~/.mesh-llm/key` on either side and restart. Forces a fresh holepunch from scratch. Does the new connection pick different addrs / behave differently?
3. **Disable mini's VPN system extension**: would confirm or rule out as the cause.
4. **Capture iroh debug logs** during a flap and see exactly which addresses it tried (`endpoint=` / `path=` events at debug level).

These are cheap to try in the lab but not in the wild. The point of the resilience work (same-target retry, sticky relay) is to be okay regardless.

## Asymmetric M4 ↔ mini connectivity (original framing, kept for context)

Mic's note: this M4 has incoming-traffic firewall rules. Asymmetric outbound‐fine, inbound‐from‐M4‐flaky is expected and is **not mini's fault**. Mini is doing the right thing; the home-network shape just doesn't favour the direct path in M4→mini direction.

Observations that confirm this read:
- All `LastOpenPath` events in the lab CSV are on connections where mini (`462bcc96fb`) is one end. Studio↔M4 has had **zero** path teardowns in ~60 min.
- Mini's mesh-llm process is alive the whole time (no crashes, no restarts).
- Mini→M4 RTT trace is steady (12–16ms direct). M4→mini trace flaps `direct↔relay`.

So the takeaway isn't "fix mini" — it's "make the mesh stable over relay when direct is unhealthy." Relay is fine and we should be willing to use it consistently for that one peer-pair without thrashing.

Mini has 5 `utun*` interfaces (Tailscale userspace mode etc.) which probably contribute to the multipath probe noise, but the firewall on this M4 side is the immediate cause.

### Errors observable on `direct_mini` probes

503/429/curl_fail happen when:
1. Direct path drops mid-stream → in-flight tunnel stream errors out before relay fallback resumes (the resilience bug below).
2. Mini's slot-admission queue overflows under back-to-back probes ("timed out waiting for an execution lane after 10 seconds") — unrelated to QUIC, this is mesh-llm's own admission control under load.

## What I think is actually broken (the resilience gap, not keep-alive)

When `LastOpenPath` fires *mid-request*, the in-flight HTTP-tunnel stream in `network::tunnel.rs` errors out:

```
WARN mesh_llm_host_runtime::network::tunnel: Inbound HTTP tunnel stream error: connection lost
```

This bubbles all the way back to the caller as a curl_fail or 503. The mesh underneath usually reconnects within ~1s. So **a request that hits a path drop is doomed today**, even though the mesh is fine 1s later.

The resilience fix lives in `network::tunnel.rs` and possibly `network::proxy.rs`: detect the specific "connection lost" / `LastOpenPath` class of error on an idempotent read, and either:
- Retry the same outbound request once on a fresh connection.
- Or block briefly on the underlying mesh reconnect and resume the stream.

That's the real follow-up. It's worth doing regardless of mini, because users will have networks like mini.

### Adjacent improvement: stickier relay preference

When a peer has been bouncing direct↔relay multiple times in a short window, the mesh could mark that path-pair as "prefer relay for now" and stop reprobing direct so eagerly. Right now we get the worst of both worlds: direct succeeds briefly, gets used for the next request, then dies. Relay would have answered every request in that window.

This would be an mesh-llm change, not an iroh change. Probably lives in `mesh::mod.rs` near where we already track per-peer RTT and `[path info]` events. A small hysteresis (`recent_direct_failures > N → demote direct for M seconds`) would cut the visible failure rate on M4↔mini significantly.

## A/B confirmation: M4↔Studio under stress (30 concurrent + 5 sequential)

- 30 concurrent direct requests to MiniMax on studio: many 429s, but **zero** new LastOpenPath events on M4. All 429s were studio's own application-level admission control (`generation queue is full`) — not network errors.
- 5 sequential follow-ups: all 200 in 1.2–2s.
- Confirms M4↔Studio is genuinely stable under load. The asymmetry is M4↔mini-specific.

This was a stress test on studio's compute too — don't repeat unnecessarily. Studio is doing real work hosting MiniMax.

## Independent confirmation: it's an OS-layer asymmetry, not QUIC

While collecting OS state from mini, ssh from M4→mini also started timing out, same direction. mesh-llm's peer connection (mesh QUIC) was still alive (peers_count=2 throughout). SSH (TCP) and QUIC see the same M4→mini packet-loss / NAT-mapping issue at the OS layer. So this is **not a QUIC quirk** — it's home-network firewall/routing between these two specific hosts, and mesh-llm inherits it.

What this means for the lesson:

## The lesson

Asymmetric connectivity is the common case, not an edge case. We will see it on:
- Home networks where one side has incoming-traffic filtering (like this M4).
- Corporate LANs with split-horizon firewalls.
- Fly machines behind cloud NATs.
- Mobile carriers with CGNAT.
- VPN overlays (Tailscale, ZeroTier) that change paths under us.

Direct P2P is the ideal when both sides cooperate. **The mesh's correctness should not depend on it.** Relay is supposed to be the resilient fallback, and we should be willing to **use relay consistently** for a peer pair where direct has been flaky in the recent past, instead of thrashing direct↔relay every few seconds.

The two concrete improvements that follow:

1. **Stream-level retry on `connection lost`** in `network::tunnel.rs` / `network/openai/transport.rs`. Today: a single mid-stream `LastOpenPath` kills the request. The mesh reconnects in ~1s; we just need to detect the error class and retry once. `RouteAttemptResult` already has `RetryableUnavailable`; the missing piece is mapping `connection lost` to that variant on the response path, not only the connect path.
2. **Sticky relay preference per peer**. Track `recent_direct_path_failures` per peer in `mesh::mod.rs`. If we've had ≥2 path teardowns in the last minute on a peer-pair, demote direct for that pair for N minutes — just use relay until the dust settles. Re-probe direct on a slow timer.

Neither requires changes to iroh. The iroh per-path defaults are fine. The mesh-llm-layer logic on top is what's missing.

## Where the resilience gap lives, exactly

`network::openai::transport.rs::route_remote_attempt`:

- **Pre-forward errors** (`node.open_http_tunnel` fails, `quic_send.write_all` fails before commit) → mapped to `RouteAttemptResult::RetryableUnavailable` → outer loop retries on next host. **This already works.**
- **Post-forward errors** (response stream read fails, relay loop fails) → propagated as `Result::Err` out of `relay_adapted_response` → caller sees an error and the request dies. **No retry here today.** This is the bug we're hitting on mini.

The natural fix: in `route_remote_attempt_after_forward`, if we haven't yet written any bytes to the client TCP stream (i.e. we're still in `probe_http_response` or just finished it, before `relay_*` starts streaming), classify `connection lost` as `RetryableUnavailable` and let the outer routing loop pick a new target / retry the same one with a fresh connection.

Once we've started streaming body bytes back to the client, we can't safely retry (partial bytes already on the wire), but we *can* at that point send a clean SSE error event so the client knows it's a stream interruption vs. a content error.

## Same-target retry on RetryableUnavailable (now in test on branch `micn/quic-retry-on-connection-lost`)

Implemented in `crates/mesh-llm-host-runtime/src/network/openai/transport.rs`:

* Split `route_remote_attempt` into `route_remote_attempt_once` (one shot) + `route_remote_attempt` (wrapper).
* Wrapper inspects the first attempt result. If it's `RetryableUnavailable` (the only variant we ever set BEFORE any bytes hit the client TCP stream), sleeps 750ms and retries once against the same host on a fresh tunnel.
* All other variants pass through unchanged — `Delivered`, `ClientDisconnected`, `RetryableTimeout`, `RetryableContextOverflow` either succeeded, partially streamed, or have a different remediation.
* Pure-function `should_retry_remote_attempt` extracted for unit testing. 5 tests pin the policy (yes on unavailable; no on the other four variants).

Why 750ms: iroh typically reopens a fresh connection within ~1s after a path teardown. 750ms is below the user-perceptible-stall threshold and matches the observed reconnect window in the lab logs.

Baseline (before fix, ~1h40min):
- `direct_mini` 8/30 OK = 27% success
- `mesh_chat` 30/31 OK = 97%
- everything else 100%

Running lab now with the fix to see if `direct_mini` success rate improves and whether the retry actually fires under real `connection lost` events. Most `direct_mini` failures so far are http_err 429 (mini's own admission control), NOT pre-commit connection drops, so the retry is design-correctly a no-op for those — we'll see if the `curl_fail` rate (which IS pre-commit drops) goes down.

## Sticky relay design sketch (separate change)

- Per-peer counter: `recent_direct_path_failures: u32` decaying over time.
- On `LastOpenPath` / `connection lost` whose `remote_info` indicates direct was the active path, increment.
- When opening a new tunnel to a peer with `recent_direct_path_failures >= N`, configure the connection to prefer relay (iroh has knobs for this; need to verify whether they're public API).
- Reset after M minutes of clean operation.

## Open questions still worth answering

- Is iroh's path-selection actually configurable per-connection? Or only globally via transport config?
- Does iroh's relay path have its own keep-alive we can confirm is enabled? (`RELAY_PATH_MAX_IDLE_TIMEOUT = 30s` in iroh socket.rs.) Important for the sticky-relay case where relay is the only path for many seconds.
- The reverse OS-level finding: M4→mini SSH (TCP/22) also times out during the same windows where M4→mini QUIC paths drop. So at least some of the iroh "direct path closed" events trace back to actual UDP packet loss / NAT mapping expiry at the OS layer, not iroh bookkeeping. This is reassuring — it means iroh is doing the right thing; we just need to recover gracefully.

## Probe summary at ~53 min (latest snapshot)

```
auto_chat        16/16 OK
mesh_chat        15/16 OK (1 curl_fail)
direct_studio    16/16 OK
direct_mini       6/15 OK   ← the problem, 60% fail
mesh_tool        15/15 OK
stream_studio    15/15 OK
```

- Studio-side surfaces (`auto_chat`, `direct_studio`, `mesh_chat`, `mesh_tool`, `stream_studio`) are essentially 100% successful even though MoA fan-out crosses to mini under the hood.
- Direct mini surface fails ~60% of the time — a mix of curl_fail (path teardown mid-stream) and http_err 429/503 (mini's queue + relay-fallback timeout).
- Slow `mesh_chat` runs (60–83s) are MoA's grace timer waiting on mini, then giving up.

All LastOpenPath events in the lab CSV are on connections involving mini. Studio↔M4 has zero. Studio's `LastOpenPath` count of 6 is *studio losing its connection to mini*, not to M4 — confirmed by grepping `remote=...` on those events. So the framing "all about mini" is right.

## Sticky relay: prior art in the iroh API

`iroh::Endpoint::remote_info(node_id)` returns `RemoteInfo` which exposes path observations (relay URL, direct addresses, last activity, etc.). We can already see what path is in use. So the sticky-relay heuristic could be a thin layer over that without forking iroh.
