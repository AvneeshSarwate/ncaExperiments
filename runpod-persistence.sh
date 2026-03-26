#!/bin/bash
# RunPod Persistence Script
#
# On RunPod, everything outside /workspace is destroyed between pod restarts.
# This script restores tools and state from /workspace/.persist back into the
# ephemeral filesystem so they're usable again.
#
# What it restores:
#   - Claude Code binary (from .persist/claude-versions/)
#   - Claude Code auth, history, sessions, and settings (from .persist/claude-home/)
#   - uv + uvx package manager (from .persist/bin/)
#   - Deno runtime (from .persist/deno/)
#   - micro terminal editor (from .persist/bin/)
#   - tmux terminal multiplexer (from .persist/bin/)
#
# It also:
#   - Installs unzip via apt if missing (needed by deno installer)
#   - Sets up PATH to include ~/.local/bin and ~/.deno/bin
#   - Points uv and deno caches to persistent storage
#   - Hooks itself into ~/.bashrc so it runs automatically on new shells
#   - Registers a cron job to save Claude Code state every 5 minutes
#   - Exports a `persist-save` shell function for manual saves
#
# First-time setup: run runpod-setup.sh to install all tools into .persist/
# Subsequent restarts: source this script (or open a new shell after first run)
#
# Usage: source /workspace/.persist/init.sh

PERSIST="/workspace/.persist"

# --- Claude Code ---
if [ -d "$PERSIST/claude-versions" ] && [ ! -f "$HOME/.local/bin/claude" ]; then
    LATEST=$(ls -t "$PERSIST/claude-versions/" 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        mkdir -p "$HOME/.local/share/claude/versions" "$HOME/.local/bin"
        cp "$PERSIST/claude-versions/$LATEST" "$HOME/.local/share/claude/versions/$LATEST"
        chmod +x "$HOME/.local/share/claude/versions/$LATEST"
        ln -sf "$HOME/.local/share/claude/versions/$LATEST" "$HOME/.local/bin/claude"
        echo "[init] Claude Code $LATEST restored."
    fi
fi

# --- uv ---
if [ -f "$PERSIST/bin/uv" ] && ! command -v uv &>/dev/null; then
    mkdir -p "$HOME/.local/bin"
    cp "$PERSIST/bin/uv" "$HOME/.local/bin/uv"
    cp "$PERSIST/bin/uvx" "$HOME/.local/bin/uvx" 2>/dev/null
    chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" 2>/dev/null
    echo "[init] uv restored."
fi

# --- Deno ---
if [ -f "$PERSIST/deno/deno" ] && ! command -v deno &>/dev/null; then
    mkdir -p "$HOME/.deno/bin"
    cp "$PERSIST/deno/deno" "$HOME/.deno/bin/deno"
    chmod +x "$HOME/.deno/bin/deno"
    echo "[init] Deno restored."
fi

# --- micro (terminal editor) ---
if [ -f "$PERSIST/bin/micro" ] && ! command -v micro &>/dev/null; then
    cp "$PERSIST/bin/micro" "$HOME/.local/bin/micro"
    chmod +x "$HOME/.local/bin/micro"
    echo "[init] micro restored."
fi

# --- tmux ---
if [ -f "$PERSIST/bin/tmux" ] && ! command -v tmux &>/dev/null; then
    cp "$PERSIST/bin/tmux" "$HOME/.local/bin/tmux"
    chmod +x "$HOME/.local/bin/tmux"
    echo "[init] tmux restored."
fi

# --- SSH keys (git deploy key) ---
if [ -d "$PERSIST/ssh" ] && [ ! -f "$HOME/.ssh/github_one_repo" ]; then
    mkdir -p "$HOME/.ssh"
    cp "$PERSIST/ssh/github_one_repo" "$HOME/.ssh/github_one_repo"
    cp "$PERSIST/ssh/github_one_repo.pub" "$HOME/.ssh/github_one_repo.pub"
    cp "$PERSIST/ssh/config" "$HOME/.ssh/config"
    cp "$PERSIST/ssh/known_hosts" "$HOME/.ssh/known_hosts" 2>/dev/null
    chmod 700 "$HOME/.ssh"
    chmod 600 "$HOME/.ssh/github_one_repo" "$HOME/.ssh/config"
    echo "[init] SSH keys restored."
fi

# --- Claude Code auth, history & settings ---
CLAUDE_HOME="$HOME/.claude"
CLAUDE_PERSIST="$PERSIST/claude-home"
if [ -d "$CLAUDE_PERSIST" ]; then
    mkdir -p "$CLAUDE_HOME"
    for item in .credentials.json history.jsonl sessions projects settings.json plugins; do
        if [ -e "$CLAUDE_PERSIST/$item" ]; then
            cp -a "$CLAUDE_PERSIST/$item" "$CLAUDE_HOME/$item"
        fi
    done
    echo "[init] Claude Code auth & history restored."
fi

# --- unzip (needed by deno installer, may be missing) ---
if ! command -v unzip &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq unzip >/dev/null 2>&1
    echo "[init] unzip installed."
fi

# --- PATH ---
export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH"

# --- Cache dirs on persistent storage ---
export UV_CACHE_DIR="$PERSIST/uv-cache"
export DENO_DIR="$PERSIST/deno-cache"
mkdir -p "$UV_CACHE_DIR" "$DENO_DIR"

# --- Auto-hook into ~/.bashrc if not already there ---
HOOK='[ -f /workspace/.persist/init.sh ] && source /workspace/.persist/init.sh'
if ! grep -qF 'workspace/.persist/init.sh' "$HOME/.bashrc" 2>/dev/null; then
    echo "$HOOK" >> "$HOME/.bashrc"
    echo "[init] Added to ~/.bashrc (will auto-run on new shells this session)."
fi

# --- Save function: also available as shell command ---
persist-save() { /workspace/.persist/persist-save.sh && echo "[persist-save] Done."; }
export -f persist-save

# --- Cron: auto-save Claude Code state every 5 minutes ---
CRON_CMD="/workspace/.persist/persist-save.sh"
if ! crontab -l 2>/dev/null | grep -qF "$CRON_CMD"; then
    service cron start >/dev/null 2>&1 || true
    (crontab -l 2>/dev/null; echo "*/5 * * * * $CRON_CMD") | crontab -
    echo "[init] Cron job added: persist-save every 5 minutes."
fi

echo "[init] Done. claude=$(command -v claude) uv=$(command -v uv) deno=$(command -v deno)"
