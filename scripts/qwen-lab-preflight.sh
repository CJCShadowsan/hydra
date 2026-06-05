#!/usr/bin/env bash
set -euo pipefail

HOSTS="${HOSTS:-192.168.0.2,192.168.0.4,192.168.0.3}"
PORTS="${PORTS:-9337 14317 19031 19032 19033 19131 19132 19214 19227 19231 19232 19241}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"
KILL_STALE=0
CLEAN_TMP=0
MIN_FREE_GB="${MIN_FREE_GB:-20}"
OUT=""

usage() {
  cat <<'USAGE'
Usage: scripts/qwen-lab-preflight.sh [--kill] [--clean-tmp] [--min-free-gb N] [--hosts A,B,C] [--ports "P ..."] [--ssh-opts "OPTS"] [--out path]

Checks Qwen lab hosts for SSH reachability, stale stage, llama, KV, Mesh, or
Ollama processes, and listeners on common lab ports. With --kill, matching
processes are stopped. With --clean-tmp, generated remote lab scratch
directories under /tmp are removed. The command exits non-zero if any host is
unreachable or if any stale process, lab-port listener, or low-disk condition
remains.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill)
      KILL_STALE=1
      shift
      ;;
    --clean-tmp)
      CLEAN_TMP=1
      shift
      ;;
    --min-free-gb)
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --hosts)
      HOSTS="$2"
      shift 2
      ;;
    --ports)
      PORTS="$2"
      shift 2
      ;;
    --ssh-opts)
      SSH_OPTS="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  exec > >(tee "$OUT") 2>&1
fi

IFS=',' read -r -a HOST_ARRAY <<<"$HOSTS"
overall_status=0
ssh_failures=0
remote_failures=0
FAILED_SSH_HOSTS=()

print_local_network_diagnostics() {
  local host
  echo "local_network_diagnostics:"
  echo "local_ipv4_addresses:"
  if command -v ip >/dev/null 2>&1; then
    ip -4 addr show 2>/dev/null | awk '/inet / { print "  " $2 " dev " $NF }' || true
  elif command -v ifconfig >/dev/null 2>&1; then
    ifconfig 2>/dev/null | awk '
      /^[a-zA-Z0-9]/ { iface = $1; sub(/:.*/, "", iface) }
      /inet / && $2 != "127.0.0.1" { print "  " $2 " dev " iface }
    ' || true
  else
    echo "  unavailable"
  fi
  echo "routes_to_failed_hosts:"
  for host in "${FAILED_SSH_HOSTS[@]}"; do
    echo "  == $host =="
    if command -v ip >/dev/null 2>&1; then
      ip route get "$host" 2>&1 | sed 's/^/  /' || true
    elif command -v route >/dev/null 2>&1; then
      route -n get "$host" 2>&1 | sed 's/^/  /' || true
    else
      echo "  route command unavailable"
    fi
  done
}

for host in "${HOST_ARRAY[@]}"; do
  echo "== qwen lab preflight: $host =="
  remote_cmd="$(printf 'PORTS=%q KILL_STALE=%q CLEAN_TMP=%q MIN_FREE_GB=%q bash -s' "$PORTS" "$KILL_STALE" "$CLEAN_TMP" "$MIN_FREE_GB")"
  set +e
  ssh $SSH_OPTS "$host" "$remote_cmd" <<'REMOTE'
set -euo pipefail
PROCESS_PATTERN='(skippy-server|skippy-correctness|skippy-prompt|kv-server|/(llama-server|llama-cli|llama-bench|llama-run|main)( |$)|(^| )llama-(server|cli|bench|run)( |$)|mesh-llm|mesh-server|/(mesh)( |$)|(^| )mesh( |$)|ollama)'

scan_processes() {
  ps -axo pid=,etime=,user=,command= \
    | awk -v pat="$PROCESS_PATTERN" '
      BEGIN { IGNORECASE = 1 }
      $0 ~ pat && $0 !~ /awk -v pat/ && $0 !~ /bash -s/ { print }
    '
}

scan_ports() {
  local port
  for port in $PORTS; do
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | tail -n +2 || true
  done
}

free_gb_for_path() {
  df -Pk "$1" 2>/dev/null | awk 'NR == 2 { printf "%.0f", $4 / 1024 / 1024 }'
}

host_name="$(hostname)"
echo "host=$host_name"

matches="$(scan_processes || true)"
if [[ -n "$matches" ]]; then
  echo "process_matches:"
  printf '%s\n' "$matches"
  if [[ "$KILL_STALE" == "1" ]]; then
    pids="$(printf '%s\n' "$matches" | awk '{print $1}')"
    echo "stopping_pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    kill -9 $pids 2>/dev/null || true
  fi
else
  echo "process_matches: none"
fi

listeners="$(scan_ports || true)"
if [[ -n "$listeners" ]]; then
  echo "lab_port_listeners:"
  printf '%s\n' "$listeners"
else
  echo "lab_port_listeners: none"
fi

if [[ "$CLEAN_TMP" == "1" ]]; then
  echo "cleaning_tmp: /tmp/skippy-remote-prompt /tmp/skippy-runtime-bench"
  rm -rf /tmp/skippy-remote-prompt /tmp/skippy-runtime-bench
fi

free_root_gb="$(free_gb_for_path /)"
free_tmp_gb="$(free_gb_for_path /tmp)"
echo "free_gb_root=$free_root_gb"
echo "free_gb_tmp=$free_tmp_gb"
low_disk=""
if [[ -n "$free_root_gb" && "$free_root_gb" -lt "$MIN_FREE_GB" ]]; then
  low_disk="root:${free_root_gb}GB"
fi
if [[ -n "$free_tmp_gb" && "$free_tmp_gb" -lt "$MIN_FREE_GB" ]]; then
  low_disk="${low_disk:+$low_disk }tmp:${free_tmp_gb}GB"
fi
if [[ -n "$low_disk" ]]; then
  echo "low_disk: $low_disk (min ${MIN_FREE_GB}GB)"
else
  echo "low_disk: none"
fi

if [[ "$KILL_STALE" == "1" ]]; then
  echo "recheck_after_kill:"
  matches="$(scan_processes || true)"
  listeners="$(scan_ports || true)"
  if [[ -n "$matches" ]]; then
    echo "process_matches:"
    printf '%s\n' "$matches"
  else
    echo "process_matches: none"
  fi
  if [[ -n "$listeners" ]]; then
    echo "lab_port_listeners:"
    printf '%s\n' "$listeners"
  else
    echo "lab_port_listeners: none"
  fi
fi

if [[ -n "${matches:-}" || -n "${listeners:-}" || -n "$low_disk" ]]; then
  echo "remote_status: dirty"
  exit 42
fi
echo "remote_status: clean"
REMOTE
  ssh_rc=$?
  set -e
  if [[ "$ssh_rc" -eq 0 ]]; then
    echo "preflight_status: ok"
  elif [[ "$ssh_rc" -eq 255 ]]; then
    echo "preflight_status: ssh_failed rc=$ssh_rc" >&2
    ssh_failures=$((ssh_failures + 1))
    FAILED_SSH_HOSTS+=("$host")
    overall_status=1
  else
    echo "preflight_status: remote_failed rc=$ssh_rc" >&2
    remote_failures=$((remote_failures + 1))
    overall_status=1
  fi
done

if [[ "$overall_status" -eq 0 ]]; then
  echo "qwen lab preflight: clean"
else
  if [[ "$ssh_failures" -gt 0 ]]; then
    echo "qwen lab preflight: ssh connection failed for $ssh_failures host(s); check --hosts, --ssh-opts, and remote sshd" >&2
    print_local_network_diagnostics >&2
  fi
  if [[ "$remote_failures" -gt 0 ]]; then
    echo "qwen lab preflight: remote stale process, lab-port listener, low-disk, or remote command failure remains on $remote_failures host(s)" >&2
  fi
fi
exit "$overall_status"
