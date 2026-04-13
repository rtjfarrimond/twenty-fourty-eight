# 2048 Solver

A reinforcement learning-based solver for the game 2048, with the goal of
reproducing and ultimately surpassing the current state of the art.

## Tech Stack

Pure Rust. The entire pipeline — game engine, training, inference, and server —
is implemented in Rust. N-tuple networks are lookup tables, not matrix
operations, so CPU training is the natural fit (not a compromise). This keeps
the cost (carbon and cash) of the project down.

### Game Engine

A bitboard representation of the game state (`u64`, 4 bits per tile storing
exponents), with precomputed lookup tables for moves and scoring. This enables
simulating millions of games per second, which is critical for training
throughput. The engine is encapsulated behind a clean API so the internal
representation can be widened to `u128` (5 bits per tile) later if needed to
support tiles above 32768.

### Front End

A Rust/WASM frontend that renders the game state received from the server over
websocket. The frontend does not run inference — it purely visualises the board.

A single agent game runs continuously on the server at a configurable pace
(e.g. one move per second), and all visitors watch it in real-time. A button
enables the user to take over and play the game themselves (the agent's game
continues in the background). All game state — both agent and user — lives on
the server, identified by session.

### Training

N-tuple networks with TD(0) learning and afterstate value functions, following
the approach of the current SOTA (Wu et al.). Training is pure Rust, using the
same game engine API as the server. The training pipeline produces a serialized
model artefact (weight tables + pattern definitions) that is consumed by the
server for inference.

A training dashboard (lightweight web UI, separate from the game frontend)
visualises score progression, tile-reach percentages, and loss curves over
training. This is part of the product, not an afterthought.

### Model Inference

N-tuple inference is just table lookups — the same Rust code handles both
training and inference. The server loads a model artefact at startup and
evaluates positions via an `Agent` trait (`best_move`, `evaluate`), keeping the
server decoupled from training code. The trait interface also enables swapping
in different model architectures in future without changing the server.

### Project Structure

- **Game engine crate** — bitboard representation, move logic, scoring
- **Model format crate** — serialization format, read-only inference struct,
  `Agent` trait
- **Training crate** — TD learning loop, evaluation, artefact output. Depends
  on game engine and model format crates.
- **Server crate** — web server, websocket, agent game loop. Depends on game
  engine and model format crates. No dependency on training.
- **Frontend** — Rust/WASM rendering
