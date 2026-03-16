#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

if [[ ! -r /etc/os-release ]]; then
  echo "No se pudo leer /etc/os-release." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"

if [[ -z "${CODENAME}" ]]; then
  echo "No se pudo detectar el codename de Ubuntu (ej: jammy)." >&2
  exit 1
fi

fix_invalid_canonical_universe() {
  local changed=0
  local file

  for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
    [[ -f "${file}" ]] || continue
    if grep -Eq "^[[:space:]]*deb[[:space:]].*archive\\.canonical\\.com/ubuntu/?[[:space:]]+${CODENAME}\\b.*\\buniverse\\b" "${file}"; then
      echo "Corrigiendo componente invalido 'universe' en ${file} (archive.canonical.com)..."
      ${SUDO} sed -Ei "/^[[:space:]]*deb[[:space:]].*archive\\.canonical\\.com\\/ubuntu\\/?[[:space:]]+${CODENAME}\\b/{
        s/(^|[[:space:]])universe([[:space:]]|$)/ /g
        s/[[:space:]]+/ /g
        s/[[:space:]]$//
      }" "${file}"
      changed=1
    fi
  done

  if [[ "${changed}" -eq 1 ]]; then
    echo "Repositorio Canonical ajustado (partner sin universe)."
  fi
}

ensure_universe_enabled() {
  if command -v add-apt-repository >/dev/null 2>&1; then
    ${SUDO} add-apt-repository -y universe >/dev/null
  fi
}

pick_aravis_pkg() {
  local pkg
  for pkg in libaravis-0.8-dev libaravis-dev; do
    if apt-cache show "${pkg}" >/dev/null 2>&1; then
      echo "${pkg}"
      return 0
    fi
  done
  return 1
}

fix_invalid_canonical_universe
ensure_universe_enabled
${SUDO} apt-get update

ARAVIS_PKG="$(pick_aravis_pkg || true)"
if [[ -z "${ARAVIS_PKG}" ]]; then
  echo "No se encontro paquete Aravis en APT (probado: libaravis-0.8-dev, libaravis-dev)." >&2
  echo "Revisa tus repositorios Ubuntu/${CODENAME} y que 'universe' este habilitado." >&2
  exit 1
fi

${SUDO} apt-get install -y \
  "${ARAVIS_PKG}" \
  pkg-config

echo "Aravis instalado (${ARAVIS_PKG}). Vuelve a configurar CMake para detectar HAVE_ARAVIS."
