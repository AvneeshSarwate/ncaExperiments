#!/bin/bash
# RunPod First-Time Setup
#
# Installs all tools into /workspace/.persist/ so they survive pod restarts.
# Run this once on a fresh pod. After that, runpod-persistence.sh (sourced
# automatically via ~/.bashrc) restores everything on each restart.
#
# Usage: bash runpod-setup.sh

set -e

PERSIST="/workspace/.persist"
mkdir -p "$PERSIST/bin" "$PERSIST/claude-versions" "$PERSIST/deno" \
         "$PERSIST/deno-cache" "$PERSIST/uv-cache" "$PERSIST/claude-home"

echo "=== RunPod Setup ==="

# --- uv (Python package manager) ---
if [ ! -f "$PERSIST/bin/uv" ]; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cp "$HOME/.local/bin/uv" "$PERSIST/bin/uv"
    cp "$HOME/.local/bin/uvx" "$PERSIST/bin/uvx" 2>/dev/null || true
    echo "[setup] uv installed."
else
    echo "[setup] uv already present."
fi

# --- Deno ---
if [ ! -f "$PERSIST/deno/deno" ]; then
    echo "[setup] Installing deno..."
    # unzip is required by the deno installer
    if ! command -v unzip &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq unzip >/dev/null 2>&1
    fi
    curl -fsSL https://deno.land/install.sh | sh
    cp "$HOME/.deno/bin/deno" "$PERSIST/deno/deno"
    echo "[setup] deno installed."
else
    echo "[setup] deno already present."
fi

# --- Claude Code ---
if [ -z "$(ls -A "$PERSIST/claude-versions/" 2>/dev/null)" ]; then
    echo "[setup] Installing Claude Code..."
    # Ensure PATH includes uv/deno for the rest of the script
    export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH"
    npx @anthropic-ai/claude-code --version 2>/dev/null || npm install -g @anthropic-ai/claude-code
    # Claude Code self-installs its binary; find and persist it
    CLAUDE_BIN=$(command -v claude 2>/dev/null)
    if [ -n "$CLAUDE_BIN" ]; then
        VERSION=$(claude --version 2>/dev/null | head -1)
        cp "$CLAUDE_BIN" "$PERSIST/claude-versions/${VERSION:-latest}"
        echo "[setup] Claude Code installed (${VERSION:-unknown version})."
    else
        echo "[setup] WARNING: Claude Code binary not found after install. You may need to install it manually."
    fi
else
    echo "[setup] Claude Code already present."
fi

# --- micro (terminal editor) ---
if [ ! -f "$PERSIST/bin/micro" ]; then
    echo "[setup] Installing micro..."
    cd /tmp
    curl -sL https://getmic.ro/r | bash
    mv micro "$PERSIST/bin/micro"
    chmod +x "$PERSIST/bin/micro"
    cd -
    echo "[setup] micro installed."
else
    echo "[setup] micro already present."
fi

# --- tmux ---
if [ ! -f "$PERSIST/bin/tmux" ]; then
    echo "[setup] Installing tmux..."
    apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1
    cp /usr/bin/tmux "$PERSIST/bin/tmux"
    chmod +x "$PERSIST/bin/tmux"
    echo "[setup] tmux installed."
else
    echo "[setup] tmux already present."
fi

# --- Copy persistence script into place ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/runpod-persistence.sh" "$PERSIST/init.sh"
chmod +x "$PERSIST/init.sh"

# --- Create persist-save helper ---
cat > "$PERSIST/persist-save.sh" << 'SAVESCRIPT'
#!/bin/bash
PERSIST="/workspace/.persist"
CLAUDE_HOME="${HOME:-/root}/.claude"
CLAUDE_PERSIST="$PERSIST/claude-home"
[ -d "$CLAUDE_HOME" ] || exit 0
mkdir -p "$CLAUDE_PERSIST"
for item in .credentials.json history.jsonl sessions projects settings.json plugins; do
    if [ -e "$CLAUDE_HOME/$item" ]; then
        cp -a "$CLAUDE_HOME/$item" "$CLAUDE_PERSIST/$item"
    fi
done
SAVESCRIPT
chmod +x "$PERSIST/persist-save.sh"

# --- Python deps for NCA training ---
echo "[setup] Installing Python dependencies..."
export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH"
cd "$SCRIPT_DIR"
uv sync

# --- NVIDIA MPS for parallel training ---
echo "[setup] Starting NVIDIA MPS daemon..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1
nvidia-cuda-mps-control -d 2>/dev/null && echo "[setup] MPS daemon started." || echo "[setup] MPS not available (no GPU?)."

# --- Activate persistence ---
echo ""
echo "[setup] Activating persistence..."
source "$PERSIST/init.sh"

echo ""
echo "=== Setup Complete ==="
echo "Tools installed to $PERSIST/ and will be restored on pod restart."
echo "Run 'persist-save' to manually save Claude Code state."
