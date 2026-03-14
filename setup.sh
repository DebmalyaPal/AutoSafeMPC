#!/bin/bash
set -e

echo "🚀 Starting setup..."
echo "🍎 macOS detected"
OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" != "Darwin" ]]; then
    echo "❌ This script is only for macOS."
    exit 1
fi

echo "📦 Installing system dependencies via Homebrew..."
brew install gmp cmake pkgconf python@3.11 || true

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "🔗 Activating virtual environment..."
source venv/bin/activate

echo "⬆️ Upgrading pip, wheel, setuptools, cython, packaging..."
pip install --upgrade pip wheel setuptools cython packaging

# Install PyBullet prebuilt wheel for Apple Silicon
echo "📦 Installing PyBullet prebuilt wheel..."
PYBULLET_WHL="https://github.com/bulletphysics/bullet3/releases/download/pybullet-3.2.7/pybullet-3.2.7-cp311-cp311-macosx_11_0_arm64.whl"
pip uninstall pybullet -y || true
pip install "$PYBULLET_WHL"

# Install safe-control-gym from source
if [ ! -d "safe-control-gym" ]; then
    echo "📂 Cloning safe-control-gym from GitHub..."
    git clone https://github.com/Stanford-ILIAD/safe-control-gym.git
fi

echo "📦 Installing safe-control-gym from source..."
cd safe-control-gym
pip install -e .

cd ..

echo "🎉 Setup complete! ✅"