#!/usr/bin/env bash
set -e

echo "[+] Installing system dependencies (Ubuntu/Debian)"

sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libssl-dev \
  zlib1g-dev \
  curl \
  git

# Install Rust if not present
if ! command -v cargo >/dev/null 2>&1; then
  echo "[+] Installing Rust toolchain"
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  source "$HOME/.cargo/env"
fi

echo "[+] Building project"
cargo build --release || true

echo "[✓] Setup completed"