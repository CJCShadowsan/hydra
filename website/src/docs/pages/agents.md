---
title: Coding agents
---

# Coding agents

Use the console first. Once chat works at `http://localhost:3131`, connect an agent to the same local Mesh API.

## Base URL

```text
http://localhost:9337/v1
```

If an agent asks for an API key, use any placeholder value:

```sh
export OPENAI_BASE_URL=http://localhost:9337/v1
export OPENAI_API_KEY=dummy
```

## Recommended first agent

```sh
MODEL_ID=$(curl -s http://localhost:9337/v1/models | jq -r '.data[0].id')
mesh-llm goose --model "$MODEL_ID"
```

## Other launchers

```sh
mesh-llm claude --model "$MODEL_ID"
mesh-llm opencode --host 127.0.0.1:9337 --model "$MODEL_ID"
mesh-llm pi --host 127.0.0.1:9337 --model "$MODEL_ID"
```

The built-in launchers point the agent at Mesh for you. If `--model` is omitted, Mesh chooses from models available on the local mesh.

## Manual setup

For tools without a Mesh launcher, configure an OpenAI-compatible provider:

| Setting | Value |
|---|---|
| Base URL | `http://localhost:9337/v1` |
| API key | `dummy` |
| Model | Any id from `/v1/models` |

List model ids:

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

## If the agent fails

Confirm console chat still works, then check the API:

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

If no models are listed, restart the serving node with a model that fits this machine.
