#!/usr/bin/env bash
set -e

echo "==> Updating system"
sudo apt update
sudo apt upgrade -y

echo "==> Installing basic build tools"
sudo apt install -y \
  build-essential \
  gcc \
  g++ \
  make \
  cmake \
  pkg-config \
  curl \
  git \
  unzip \
  wget

echo "==> Installing SSL and system libs (for Rust crypto crates)"
sudo apt install -y \
  libssl-dev \
  ca-certificates \
  zlib1g-dev

echo "==> Installing clang / llvm (for bindgen, some crypto crates)"
sudo apt install -y \
  clang \
  llvm \
  llvm-dev

echo "==> Installing perf / cpu tools (optional but useful)"
sudo apt install -y \
  linux-tools-common \
  linux-tools-$(uname -r) || true

echo "==> Installing Rust via rustup"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

echo "==> Loading Rust environment"
source $HOME/.cargo/env

echo "==> Setting Rust toolchain to stable"
rustup default stable

echo "==> Adding useful Rust components"
rustup component add rustfmt clippy || true

echo "==> Setting performance-related env vars"
cat << 'EOF' >> ~/.bashrc

# ==== Rust / HPC performance settings ====
export RUSTFLAGS="-C target-cpu=native"
export CARGO_PROFILE_RELEASE_LTO=true
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

# Parallelism (adjust if needed)
export RAYON_NUM_THREADS=128
export OMP_NUM_THREADS=128
EOF

echo "==> Reloading shell config"
source ~/.bashrc

echo "==> Verifying toolchain"
echo "--- cc ---"
which cc
cc --version

echo "--- rustc ---"
rustc --version

echo "--- cargo ---"
cargo --version

echo "==> Environment setup complete 🎉"