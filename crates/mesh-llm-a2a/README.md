# mesh-llm-a2a

A2A agent directory primitives for Mesh LLM.

This crate owns the local, configuration-backed agent directory and conversion
to official `a2a-lf` agent cards. It intentionally does not start agent
processes, publish gossip, or expose network servers.
