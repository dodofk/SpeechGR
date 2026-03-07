#!/usr/bin/env bash
set -euo pipefail

# Installs common tooling for ephemeral GPU rental servers.
# - uv
# - tmux
# - nano
#
# Usage:
#   bash scripts/setup_rental_server.sh

log() {
  printf '[setup] %s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  if have_cmd sudo; then
    SUDO="sudo"
  else
    log "This script needs root privileges and 'sudo' is not available."
    exit 1
  fi
fi

install_with_apt() {
  log "Using apt package manager"
  ${SUDO} apt-get update -y
  ${SUDO} apt-get install -y \
    ca-certificates \
    curl \
    nano \
    tmux
}

install_with_dnf() {
  log "Using dnf package manager"
  ${SUDO} dnf makecache -y
  ${SUDO} dnf install -y \
    ca-certificates \
    curl \
    nano \
    tmux
}

install_with_yum() {
  log "Using yum package manager"
  ${SUDO} yum makecache -y
  ${SUDO} yum install -y \
    ca-certificates \
    curl \
    nano \
    tmux
}

install_with_pacman() {
  log "Using pacman package manager"
  ${SUDO} pacman -Sy --noconfirm \
    ca-certificates \
    curl \
    nano \
    tmux
}

install_with_zypper() {
  log "Using zypper package manager"
  ${SUDO} zypper --gpg-auto-import-keys refresh
  ${SUDO} zypper install -y \
    ca-certificates \
    curl \
    nano \
    tmux
}

install_base_packages() {
  if have_cmd apt-get; then
    install_with_apt
    return
  fi
  if have_cmd dnf; then
    install_with_dnf
    return
  fi
  if have_cmd yum; then
    install_with_yum
    return
  fi
  if have_cmd pacman; then
    install_with_pacman
    return
  fi
  if have_cmd zypper; then
    install_with_zypper
    return
  fi

  log "Unsupported OS/package manager. Please install curl, tmux, and nano manually."
  exit 1
}

ensure_uv_on_path() {
  local uv_bin_dir
  uv_bin_dir="${HOME}/.local/bin"

  if ! have_cmd uv; then
    log "Installing uv"
    if ! have_cmd curl; then
      log "curl is required to install uv"
      exit 1
    fi
    # Do not edit shell rc automatically; we handle PATH below.
    UV_NO_MODIFY_PATH=1 curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    log "uv already installed: $(command -v uv)"
  fi

  if ! have_cmd uv && [ -x "${uv_bin_dir}/uv" ]; then
    export PATH="${uv_bin_dir}:${PATH}"
  fi

  if ! have_cmd uv; then
    log "uv installed but not on PATH. Add this to your shell rc:"
    log "  export PATH=\"${uv_bin_dir}:\$PATH\""
    exit 1
  fi

  local shell_rc
  shell_rc="${HOME}/.bashrc"
  if [ -n "${ZSH_VERSION:-}" ]; then
    shell_rc="${HOME}/.zshrc"
  fi

  if [ -f "${shell_rc}" ] && ! grep -q 'PATH="\$HOME/.local/bin:\$PATH"' "${shell_rc}"; then
    printf '\n# Added by SpeechGR setup script\nexport PATH="$HOME/.local/bin:$PATH"\n' >>"${shell_rc}"
    log "Added ~/.local/bin to PATH in ${shell_rc}"
  fi
}

main() {
  install_base_packages
  ensure_uv_on_path

  log "Done. Installed tools:"
  log "  uv:   $(command -v uv)"
  log "  tmux: $(command -v tmux || true)"
  log "  nano: $(command -v nano || true)"
  log "You may need to open a new shell for PATH updates to take effect."
}

main "$@"

