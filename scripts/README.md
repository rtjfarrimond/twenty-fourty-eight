# Deployment

## Filesystem Layout

Follows the Linux Filesystem Hierarchy Standard:

```
/opt/2048-solver/
  bin/                    server, training, generate_models, benchmark
  frontend/               HTML, CSS, WASM — static web assets

/etc/2048-solver/
  config.toml             server configuration

/var/lib/2048-solver/
  models/                 trained model .bin files + .meta.toml descriptions
  training/               per-run training logs (.log.jsonl, .config.json)
```

## Prerequisites

- Rust toolchain (`rustup`, `cargo`)
- `wasm-pack` (`cargo install wasm-pack`)
- `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
- `sudo` access (for installing to system paths and systemd)

## Build

Builds all crates in release mode and compiles the WASM frontend:

```sh
./scripts/build.sh
```

## Deploy

Runs the full build, installs to FHS paths, and starts the systemd service:

```sh
./scripts/deploy.sh
```

This will:
1. Build everything in release mode
2. Create `/opt/2048-solver/`, `/etc/2048-solver/`, `/var/lib/2048-solver/`
3. Install binaries, frontend, config, models, and training logs
4. Install and enable the `2048-server` systemd service
5. Start the server

Config is only installed if it doesn't already exist (won't overwrite
customisations). Model metadata `.meta.toml` files are always updated from
the repo.

## Redeployment

```sh
git pull
./scripts/deploy.sh
```

## Server Configuration

Edit `/etc/2048-solver/config.toml`:

```toml
port = 3000
move_interval_ms = 500
frontend_dir = "/opt/2048-solver/frontend"
models_dir = "/var/lib/2048-solver/models"
training_dir = "/var/lib/2048-solver/training"
```

The server accepts an optional config file path as its first argument.
If none is given, it looks for `/etc/2048-solver/config.toml`. If that
doesn't exist, it uses built-in defaults.

## Service Management

```sh
sudo systemctl status 2048-server    # check status
sudo systemctl restart 2048-server   # restart after changes
sudo journalctl -u 2048-server -f    # follow logs
```

## Training

Training runs independently of the server:

```sh
cd /var/lib/2048-solver/training

# Train a model (args: num_games, eval_interval, model_name)
/opt/2048-solver/bin/training 1000000 100000 "my-model"

# This produces in the current directory:
#   my-model.bin           — model weights (~257MB)
#   my-model.log.jsonl     — training eval log
#   my-model.config.json   — training config
```

To make a trained model available to the server:

```sh
# Move model + metadata to the models directory
sudo mv my-model.bin /var/lib/2048-solver/models/

# Create a description file
sudo tee /var/lib/2048-solver/models/my-model.meta.toml << 'EOF'
name = "my-model"
description = "Description of this model and its training."
EOF

# Regenerate the results dashboard data
cd /var/lib/2048-solver/models
sudo /opt/2048-solver/bin/generate_models

# Restart to load the new model
sudo systemctl restart 2048-server
```

## Adding Models

The server automatically scans `/var/lib/2048-solver/models/` on startup
and loads every `.bin` file it finds. Each model can have an optional
`.meta.toml` sidecar file providing a name and description. If no sidecar
exists, the filename is used as the name.

## Reverse Proxy (optional)

The server binds to `[::]:3000` (dual-stack IPv4/IPv6). If running behind
Traefik or another reverse proxy, add a route for your domain pointing to
`localhost:3000`. WebSocket upgrade is required for the `/ws` endpoint.

Example Traefik config (drop into your rules directory):

```toml
[http.routers.router_2048]
rule = "Host(`2048.example.com`)"
entryPoints = ["https"]
service = "service_2048"

[http.routers.router_2048.tls]
certResolver = "letsencrypt"

[http.services.service_2048.loadBalancer]
passHostHeader = true
[[http.services.service_2048.loadBalancer.servers]]
url = "http://127.0.0.1:3000"
```

## Development

For local development without installing to system paths, pass a config
file pointing to your repo paths:

```sh
cd server
cargo run -- ../config/config-dev.toml
```
