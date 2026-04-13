# 2048 AI

A pure Rust reinforcement learning system for 2048 — training, live inference,
and a playable web demo. No Python, no ML frameworks, no GPU.

## Live Demo

Watch trained models play at [2048.redact.ing](https://2048.redact.ing). Switch
between models, or take over and play yourself.

## How It Works

N-tuple networks learn board evaluation functions through self-play using TD(0)
with afterstate values. N-tuple networks are lookup tables, not neural networks
— inference is a handful of table reads per position, and training is pure
integer arithmetic. A GPU would have nothing to do.

Train a model, point it at the server, and it appears in the live demo
automatically — no restart, no manual steps. A dashboard streams training
progress in real-time as it runs.

## Architecture

Everything is Rust, structured as four crates:

- **engine** — `u64` bitboard with precomputed move tables
- **training** — TD(0) learning loop, evaluation, CLI with named flags
- **model** — serialization format, `Agent` trait for inference
- **server** — axum websocket server, hot model loading via inotify, SSE
  training stream

The frontend is Rust/WASM, rendering game state received over websocket.

## Status

Reproducing known results from the literature (Wu et al., Szubert &
Jaśkowski). Training runs in progress. Goal is to match and then surpass
state of the art.
