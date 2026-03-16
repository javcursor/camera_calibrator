#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  build_opencv_contrib_world.sh <OPENCV_VERSION> [opciones]

Ejemplos:
  ./build_opencv_contrib_world.sh 4.10.0
  ./build_opencv_contrib_world.sh 4.10.0 --keep-workroot
  ./build_opencv_contrib_world.sh 4.9.0 --prefix /usr/local/opencv-4.9.0 --nonfree
  ./build_opencv_contrib_world.sh 4.10.0 --clean
  ./build_opencv_contrib_world.sh 4.10.0 --cuda

Opciones:
  --prefix <ruta>      Prefijo de instalación (default: /usr/local/opencv-<VERSION>)
  --clean              Borra el árbol de trabajo/build de esa versión y recompila
  --nonfree            Activa OPENCV_ENABLE_NONFREE (por defecto OFF)
  --cuda               Intenta compilar con soporte CUDA (requiere toolchain CUDA instalado)
  --keep-workroot      NO borrar el WORKROOT al finalizar (por defecto se borra si todo va bien)
  -h|--help            Muestra esta ayuda

Notas:
  - WORKROOT se define por defecto como:
      $HOME/opencv_build/opencv-<VERSION>
    (puedes sobreescribirlo con la variable de entorno WORKROOT)
  - Se genera un fichero reproducible de configuración:
      <workroot>/build/opencv-<VERSION>/cmake_configure.sh
  - Se generan scripts de entorno en el prefijo (estilo ROS2):
      <PREFIX>/env.bash y <PREFIX>/env.zsh (ambos incluyen env.sh)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

OPENCV_VERSION="$1"
shift

PREFIX="/usr/local/opencv-${OPENCV_VERSION}"
CLEAN=0
ENABLE_NONFREE=OFF
WITH_CUDA=OFF
CLEANUP_WORKROOT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --nonfree)
      ENABLE_NONFREE=ON
      shift
      ;;
    --cuda)
      WITH_CUDA=ON
      shift
      ;;
    --keep-workroot)
      CLEANUP_WORKROOT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Opción desconocida: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# WORKROOT por versión (evita borrar otras builds)
WORKROOT_DEFAULT="$HOME/opencv_build/opencv-${OPENCV_VERSION}"
WORKROOT="${WORKROOT:-$WORKROOT_DEFAULT}"

SRC_ROOT="${WORKROOT}/src"
BUILD_ROOT="${WORKROOT}/build"

OPENCV_SRC="${SRC_ROOT}/opencv-${OPENCV_VERSION}"
CONTRIB_SRC="${SRC_ROOT}/opencv_contrib-${OPENCV_VERSION}"
BUILD_DIR="${BUILD_ROOT}/opencv-${OPENCV_VERSION}"

echo "==> OpenCV version: ${OPENCV_VERSION}"
echo "==> Install prefix: ${PREFIX}"
echo "==> Workroot:       ${WORKROOT}"
echo

echo "==> Se requerirán credenciales de administrador para instalar dependencias e instalar OpenCV."
sudo -v

# Helper: instalar el primer paquete disponible (para compatibilidad entre Ubuntu)
install_first_available() {
  for pkg in "$@"; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
      sudo apt-get install -y "$pkg"
      return 0
    fi
  done
  echo "AVISO: Ninguno de estos paquetes está disponible: $* (se omite)."
  return 0
}

echo "==> Instalando dependencias de compilación (apt)..."
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git pkg-config unzip curl ca-certificates \
  software-properties-common

# (universe) útil en algunas versiones/configuraciones; inocuo si ya está
sudo add-apt-repository -y universe >/dev/null 2>&1 || true
sudo apt-get update

sudo apt-get install -y \
  libgtk-3-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libjpeg-dev libpng-dev libtiff-dev \
  libtbb-dev \
  libopenexr-dev

# dc1394: en Ubuntu recientes el dev correcto es libdc1394-dev (y libdc1394-22-dev puede no existir)
install_first_available libdc1394-dev libdc1394-22-dev

if [[ "${WITH_CUDA}" == "ON" ]]; then
  echo "==> Opción --cuda activada. Se asume CUDA toolkit ya instalado."
fi

mkdir -p "${SRC_ROOT}" "${BUILD_ROOT}"

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "==> Limpieza solicitada (--clean). Eliminando WORKROOT de esta versión..."
  rm -rf "${WORKROOT}"
  mkdir -p "${SRC_ROOT}" "${BUILD_ROOT}"
fi

download_and_unpack() {
  local name="$1"
  local url="$2"
  local dst_dir="$3"

  if [[ -d "${dst_dir}" ]]; then
    echo "==> ${name} ya existe: ${dst_dir} (se reutiliza)"
    return 0
  fi

  local tmp_zip="${WORKROOT}/${name}-${OPENCV_VERSION}.zip"
  echo "==> Descargando ${name} desde:"
  echo "    ${url}"
  curl -L --fail -o "${tmp_zip}" "${url}"

  echo "==> Descomprimiendo ${name}..."
  unzip -q "${tmp_zip}" -d "${SRC_ROOT}"
  rm -f "${tmp_zip}"

  if [[ ! -d "${dst_dir}" ]]; then
    local extracted
    extracted="$(find "${SRC_ROOT}" -maxdepth 1 -type d -name "${name}-*" | head -n1 || true)"
    if [[ -n "${extracted}" && -d "${extracted}" ]]; then
      mv "${extracted}" "${dst_dir}"
    fi
  fi

  if [[ ! -d "${dst_dir}" ]]; then
    echo "ERROR: No se encontró el directorio esperado tras descomprimir: ${dst_dir}" >&2
    exit 1
  fi
}

OPENCV_URL="https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip"
CONTRIB_URL="https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip"

download_and_unpack "opencv" "${OPENCV_URL}" "${OPENCV_SRC}"
download_and_unpack "opencv_contrib" "${CONTRIB_URL}" "${CONTRIB_SRC}"

mkdir -p "${BUILD_DIR}"

echo "==> Generando script de configuración de CMake (reproducible)..."
CONFIG_SCRIPT="${BUILD_DIR}/cmake_configure.sh"
cat > "${CONFIG_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cmake -S "${OPENCV_SRC}" -B "${BUILD_DIR}" -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \\
  -DOPENCV_EXTRA_MODULES_PATH="${CONTRIB_SRC}/modules" \\
  -DBUILD_opencv_world=ON \\
  -DBUILD_SHARED_LIBS=ON \\
  -DOPENCV_GENERATE_PKGCONFIG=ON \\
  -DOPENCV_ENABLE_NONFREE=${ENABLE_NONFREE} \\
  -DBUILD_TESTS=OFF \\
  -DBUILD_PERF_TESTS=OFF \\
  -DBUILD_EXAMPLES=OFF \\
  -DBUILD_DOCS=OFF \\
  -DWITH_CUDA=${WITH_CUDA}
EOF
chmod +x "${CONFIG_SCRIPT}"

echo "==> Configurando (CMake)..."
"${CONFIG_SCRIPT}"

echo "==> Compilando..."
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo "==> Instalando en el sistema..."
sudo cmake --install "${BUILD_DIR}"

# Detectar libdir (lib vs lib64)
LIBDIR="lib"
if [[ -d "${PREFIX}/lib64" && ! -d "${PREFIX}/lib" ]]; then
  LIBDIR="lib64"
fi

echo "==> Verificando que se ha generado libopencv_world.so..."
if ! ls "${PREFIX}/${LIBDIR}/"libopencv_world.so* >/dev/null 2>&1; then
  echo "ERROR: No encuentro libopencv_world.so en ${PREFIX}/${LIBDIR}." >&2
  exit 1
fi

echo "==> Registrando librerías para el linker (ldconfig)..."
LDCONF_FILE="/etc/ld.so.conf.d/opencv-${OPENCV_VERSION}.conf"
echo "${PREFIX}/${LIBDIR}" | sudo tee "${LDCONF_FILE}" >/dev/null
sudo ldconfig

# ---------------------------
# Generación de env.sh / env.bash / env.zsh (estilo ROS2)
# (corregido: sin bad substitution en bash/zsh)
# ---------------------------
echo "==> Generando scripts de entorno en el prefijo de instalación..."

TMP_ENV_DIR="${WORKROOT}/_env_tmp"
mkdir -p "${TMP_ENV_DIR}"

ENV_TEMPLATE="${TMP_ENV_DIR}/env.sh.in"
ENV_COMMON_TMP="${TMP_ENV_DIR}/env.sh"
ENV_BASH_TMP="${TMP_ENV_DIR}/env.bash"
ENV_ZSH_TMP="${TMP_ENV_DIR}/env.zsh"

cat > "${ENV_TEMPLATE}" <<'EOF'
# OpenCV environment (common)
# Usage:
#   source <PREFIX>/env.bash   (bash)
#   source <PREFIX>/env.zsh    (zsh)

_opencv_prefix="__OPENCV_PREFIX__"
_opencv_libdir="__OPENCV_LIBDIR__"

_prepend_path() {
  dir="$1"
  var="$2"

  # Leer el valor actual de la variable cuyo nombre está en $var
  # Ej: var=PATH -> cur=${PATH-}
  eval "cur=\${$var-}"

  case ":$cur:" in
    *":$dir:"*) : ;;
    *)
      if [ -n "$cur" ]; then
        eval "export $var=\"$dir:$cur\""
      else
        eval "export $var=\"$dir\""
      fi
      ;;
  esac

  unset cur
}

_prepend_path "${_opencv_prefix}/bin" PATH
_prepend_path "${_opencv_libdir}/pkgconfig" PKG_CONFIG_PATH

# CMake package location (OpenCVConfig.cmake)
export OpenCV_DIR="${_opencv_libdir}/cmake/opencv4"

# Opcional: ayuda a CMake a encontrar prefijos sin pasar OpenCV_DIR explícitamente
_prepend_path "${_opencv_prefix}" CMAKE_PREFIX_PATH

unset _prepend_path _opencv_prefix _opencv_libdir
EOF

escape_sed_repl() {
  # Escapa caracteres especiales en el replacement de sed (delim: |)
  printf '%s' "$1" | sed -e 's/[\\/&|]/\\&/g'
}

PREFIX_ESCAPED="$(escape_sed_repl "${PREFIX}")"
LIBPATH_ESCAPED="$(escape_sed_repl "${PREFIX}/${LIBDIR}")"

sed \
  -e "s|__OPENCV_PREFIX__|${PREFIX_ESCAPED}|g" \
  -e "s|__OPENCV_LIBDIR__|${LIBPATH_ESCAPED}|g" \
  "${ENV_TEMPLATE}" > "${ENV_COMMON_TMP}"

cat > "${ENV_BASH_TMP}" <<'EOF'
# OpenCV environment for Bash
_opencv_this="${BASH_SOURCE[0]}"
_opencv_dir="$(cd "$(dirname "$_opencv_this")" && pwd)"
# shellcheck source=/dev/null
source "${_opencv_dir}/env.sh"
unset _opencv_this _opencv_dir
EOF

cat > "${ENV_ZSH_TMP}" <<'EOF'
# OpenCV environment for Zsh
_opencv_this="${(%):-%x}"
_opencv_dir="${_opencv_this:A:h}"
# shellcheck source=/dev/null
source "${_opencv_dir}/env.sh"
unset _opencv_this _opencv_dir
EOF

# Instalar (root) en el prefijo
sudo install -m 0644 "${ENV_COMMON_TMP}" "${PREFIX}/env.sh"
sudo install -m 0644 "${ENV_BASH_TMP}"   "${PREFIX}/env.bash"
sudo install -m 0644 "${ENV_ZSH_TMP}"    "${PREFIX}/env.zsh"

# ---------------------------
# Smoketest + Cleanup WORKROOT
# ---------------------------
smoke_test() {
  local ok=1

  [[ -f "${PREFIX}/${LIBDIR}/cmake/opencv4/OpenCVConfig.cmake" ]] || ok=0
  ls "${PREFIX}/${LIBDIR}/"libopencv_world.so* >/dev/null 2>&1 || ok=0

  if [[ -x "${PREFIX}/bin/opencv_version" ]]; then
    "${PREFIX}/bin/opencv_version" >/dev/null 2>&1 || ok=0
  fi

  if command -v pkg-config >/dev/null 2>&1; then
    PKG_CONFIG_PATH_TEST="${PREFIX}/${LIBDIR}/pkgconfig:${PKG_CONFIG_PATH:-}"
    PKG_CONFIG_PATH="${PKG_CONFIG_PATH_TEST}" pkg-config --exists opencv4 || ok=0
  fi

  [[ "${ok}" -eq 1 ]]
}

safe_delete_workroot() {
  local dir="$1"
  local dir_real
  dir_real="$(realpath -m "${dir}")"

  # Guardas anti-desastre
  if [[ -z "${dir_real}" || "${dir_real}" == "/" ]]; then
    echo "ERROR: WORKROOT inválido para borrado: '${dir_real}'" >&2
    return 1
  fi

  local home_real
  home_real="$(realpath -m "$HOME")"

  if [[ "${dir_real}" == "${home_real}" || "${dir_real}" == "/home" || "${dir_real}" == "/usr" || "${dir_real}" == "/usr/local" ]]; then
    echo "ERROR: WORKROOT demasiado genérico para borrado: '${dir_real}'" >&2
    return 1
  fi

  # Requisito de seguridad: debe contener '/opencv_build/'
  if [[ "${dir_real}" != *"/opencv_build/"* ]]; then
    echo "ERROR: Por seguridad, solo borro WORKROOT que contenga '/opencv_build/' (actual: '${dir_real}')." >&2
    echo "       Si has definido WORKROOT manualmente, usa --keep-workroot o ajusta a una ruta segura." >&2
    return 1
  fi

  # El prefijo de instalación NO puede estar dentro del WORKROOT
  local prefix_real
  prefix_real="$(realpath -m "${PREFIX}")"
  if [[ "${prefix_real}" == "${dir_real}"* ]]; then
    echo "ERROR: El prefijo de instalación (${prefix_real}) está dentro de WORKROOT (${dir_real})." >&2
    echo "       No se borra para evitar eliminar la instalación." >&2
    return 1
  fi

  # Calcular padre antes de borrar
  local parent_dir
  parent_dir="$(dirname "${dir_real}")"

  echo "==> Eliminando WORKROOT: ${dir_real}"
  rm -rf --one-file-system "${dir_real}"

  # Si queda <...>/opencv_build vacío, eliminarlo también (solo si está vacío).
  if [[ "$(basename "${parent_dir}")" == "opencv_build" ]]; then
    rmdir --ignore-fail-on-non-empty "${parent_dir}" 2>/dev/null || true
  fi
}

echo
if smoke_test; then
  echo "==> Instalación verificada (smoketest OK)."
  if [[ "${CLEANUP_WORKROOT}" -eq 1 ]]; then
    safe_delete_workroot "${WORKROOT}"
    echo "==> WORKROOT eliminado. Espacio liberado."
  else
    echo "==> --keep-workroot activo: WORKROOT conservado."
  fi
else
  echo "ERROR: Smoketest falló; no se borra WORKROOT." >&2
  exit 1
fi

echo
echo "==> Instalación completada."
echo "==> Para usar OpenCV en tu shell:"
echo "    Bash: source ${PREFIX}/env.bash"
echo "    Zsh:  source ${PREFIX}/env.zsh"
echo
echo "==> Comprobaciones recomendadas (tras 'source'):"
echo "    opencv_version"
echo "    pkg-config --modversion opencv4"

