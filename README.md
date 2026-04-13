# 2048 Solver

This project contains source code files for a reinforcement learning-based
solver of the popular game 2048.

## Tech Stack

We use Rust as our primary language because it is fast and portable, enabling
us to train on CPUs and keep the cost (carbon and cash) of the project down.

### Front End

Rust compiled to WebAssemby visualises the trained model playing 2048 in
real-time. The agent is continually playing games until it loses and starting a
new one.

A button enables the user visiting the site to take over from the agent and
play the game themselves.

### Backend

A highly efficient Rust core representation of the game state and engine,
optimized to enable agents to play the game both at training and inference
time.

A web-server that tracks the current state of the game that the agent is
playing over a websocket, and enables users visiting the front end to play
games themselves. The server does not keep any local state of user initiated
games, instead it sends representations of the whole game state between the
back and frontend, enabling the user's device to keep track of the state and
reduce server load.

### Model Training

The only part of the codebase that may use python where it is strictly required
to train the models, since this is the gold standard ecosystem for this type of
work. Rather than using convolutional networks, as prior work in this area
does, we instead use the same core Rust API that the backend uses to update
model state, enabling highly optimized CPU bound training.

### Model Inference

TODO: Figure out how this will work. Should not be python at this point, should
compile to something fast.
