#include "camera/GenTLBackend.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <vector>

namespace {
using bool8_t = uint8_t;
using GC_ERROR = int32_t;
using INFO_DATATYPE = int32_t;
using TL_HANDLE = void*;
using IF_HANDLE = void*;
using DEV_HANDLE = void*;
using DS_HANDLE = void*;
using EVENT_HANDLE = void*;
using BUFFER_HANDLE = void*;
using EVENT_TYPE = int32_t;
using STREAM_INFO_CMD = int32_t;
using BUFFER_INFO_CMD = int32_t;
using ACQ_START_FLAGS = int32_t;
using ACQ_STOP_FLAGS = int32_t;
using ACQ_QUEUE_TYPE = int32_t;
using DEVICE_ACCESS_FLAGS = int32_t;

constexpr GC_ERROR GC_SUCCESS = 0;
constexpr GC_ERROR GC_ERR_TIMEOUT = -1011;
constexpr GC_ERROR GC_ERR_ABORT = -1012;
constexpr uint64_t GENTL_INFINITE = std::numeric_limits<uint64_t>::max();

constexpr DEVICE_ACCESS_FLAGS DEVICE_ACCESS_CONTROL = 3;

constexpr EVENT_TYPE EVENT_NEW_BUFFER = 1;

constexpr ACQ_START_FLAGS ACQ_START_FLAGS_DEFAULT = 0;
constexpr ACQ_STOP_FLAGS ACQ_STOP_FLAGS_DEFAULT = 0;
constexpr ACQ_QUEUE_TYPE ACQ_QUEUE_ALL_TO_INPUT = 2;

constexpr STREAM_INFO_CMD STREAM_INFO_PAYLOAD_SIZE = 10;
constexpr STREAM_INFO_CMD STREAM_INFO_BUF_ANNOUNCE_MIN = 12;

constexpr BUFFER_INFO_CMD BUFFER_INFO_BASE = 0;
constexpr BUFFER_INFO_CMD BUFFER_INFO_SIZE_FILLED = 9;
constexpr BUFFER_INFO_CMD BUFFER_INFO_WIDTH = 10;
constexpr BUFFER_INFO_CMD BUFFER_INFO_HEIGHT = 11;
constexpr BUFFER_INFO_CMD BUFFER_INFO_XPADDING = 14;
constexpr BUFFER_INFO_CMD BUFFER_INFO_FRAMEID = 16;
constexpr BUFFER_INFO_CMD BUFFER_INFO_IMAGEPRESENT = 17;
constexpr BUFFER_INFO_CMD BUFFER_INFO_IMAGEOFFSET = 18;
constexpr BUFFER_INFO_CMD BUFFER_INFO_PAYLOADTYPE = 19;
constexpr BUFFER_INFO_CMD BUFFER_INFO_PIXELFORMAT = 20;
constexpr BUFFER_INFO_CMD BUFFER_INFO_PIXELFORMAT_NAMESPACE = 21;
constexpr BUFFER_INFO_CMD BUFFER_INFO_DATA_SIZE = 27;
constexpr BUFFER_INFO_CMD BUFFER_INFO_TIMESTAMP_NS = 28;
constexpr BUFFER_INFO_CMD BUFFER_INFO_IS_INCOMPLETE = 7;
constexpr BUFFER_INFO_CMD BUFFER_INFO_TIMESTAMP = 3;

constexpr uint64_t PAYLOAD_TYPE_IMAGE = 1;

constexpr uint64_t PFNC_MONO8 = 0x01080001ULL;
constexpr uint64_t PFNC_BAYER_GR8 = 0x01080008ULL;
constexpr uint64_t PFNC_BAYER_RG8 = 0x01080009ULL;
constexpr uint64_t PFNC_BAYER_GB8 = 0x0108000AULL;
constexpr uint64_t PFNC_BAYER_BG8 = 0x0108000BULL;
constexpr uint64_t PFNC_RGB8 = 0x02180014ULL;
constexpr uint64_t PFNC_BGR8 = 0x02180015ULL;

struct EVENT_NEW_BUFFER_DATA {
  BUFFER_HANDLE buffer_handle = nullptr;
  void* user_pointer = nullptr;
};

struct GenTLFns {
  GC_ERROR (*GCInitLib)() = nullptr;
  GC_ERROR (*GCCloseLib)() = nullptr;
  GC_ERROR (*GCGetLastError)(GC_ERROR* piErrorCode, char* sErrorText, size_t* piSize) = nullptr;

  GC_ERROR (*TLOpen)(TL_HANDLE* phTL) = nullptr;
  GC_ERROR (*TLClose)(TL_HANDLE hTL) = nullptr;
  GC_ERROR (*TLUpdateInterfaceList)(TL_HANDLE hTL, bool8_t* pbChanged, uint64_t iTimeout) = nullptr;
  GC_ERROR (*TLGetNumInterfaces)(TL_HANDLE hTL, uint32_t* piNumInterfaces) = nullptr;
  GC_ERROR (*TLGetInterfaceID)(TL_HANDLE hTL, uint32_t iIndex, char* sInterfaceID, size_t* piSize) = nullptr;
  GC_ERROR (*TLOpenInterface)(TL_HANDLE hTL, const char* sInterfaceID, IF_HANDLE* phInterface) = nullptr;

  GC_ERROR (*IFUpdateDeviceList)(IF_HANDLE hIF, bool8_t* pbChanged, uint64_t iTimeout) = nullptr;
  GC_ERROR (*IFGetNumDevices)(IF_HANDLE hIF, uint32_t* piNumDevices) = nullptr;
  GC_ERROR (*IFGetDeviceID)(IF_HANDLE hIF, uint32_t iIndex, char* sDeviceID, size_t* piSize) = nullptr;
  GC_ERROR (*IFOpenDevice)(IF_HANDLE hIF, const char* sDeviceID, DEVICE_ACCESS_FLAGS iOpenFlag,
                           DEV_HANDLE* phDevice) = nullptr;
  GC_ERROR (*IFClose)(IF_HANDLE hIF) = nullptr;

  GC_ERROR (*DevClose)(DEV_HANDLE hDevice) = nullptr;
  GC_ERROR (*DevGetNumDataStreams)(DEV_HANDLE hDevice, uint32_t* piNumDataStreams) = nullptr;
  GC_ERROR (*DevGetDataStreamID)(DEV_HANDLE hDevice, uint32_t iIndex, char* sDataStreamID,
                                 size_t* piSize) = nullptr;
  GC_ERROR (*DevOpenDataStream)(DEV_HANDLE hDevice, const char* sDataStreamID,
                                DS_HANDLE* phDataStream) = nullptr;

  GC_ERROR (*DSClose)(DS_HANDLE hDataStream) = nullptr;
  GC_ERROR (*DSGetInfo)(DS_HANDLE hDataStream, STREAM_INFO_CMD iInfoCmd, INFO_DATATYPE* piType,
                        void* pBuffer, size_t* piSize) = nullptr;
  GC_ERROR (*DSAllocAndAnnounceBuffer)(DS_HANDLE hDataStream, size_t iSize, void* pPrivate,
                                       BUFFER_HANDLE* phBuffer) = nullptr;
  GC_ERROR (*DSQueueBuffer)(DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer) = nullptr;
  GC_ERROR (*DSStartAcquisition)(DS_HANDLE hDataStream, ACQ_START_FLAGS iStartFlags,
                                 uint64_t iNumToAcquire) = nullptr;
  GC_ERROR (*DSStopAcquisition)(DS_HANDLE hDataStream, ACQ_STOP_FLAGS iStopFlags) = nullptr;
  GC_ERROR (*DSFlushQueue)(DS_HANDLE hDataStream, ACQ_QUEUE_TYPE iOperation) = nullptr;
  GC_ERROR (*DSGetBufferInfo)(DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer,
                              BUFFER_INFO_CMD iInfoCmd, INFO_DATATYPE* piType, void* pBuffer,
                              size_t* piSize) = nullptr;
  GC_ERROR (*DSRevokeBuffer)(DS_HANDLE hDataStream, BUFFER_HANDLE hBuffer, void** ppBuffer,
                             void** ppPrivate) = nullptr;

  GC_ERROR (*GCRegisterEvent)(void* hObject, EVENT_TYPE iEventID, EVENT_HANDLE* phEvent) = nullptr;
  GC_ERROR (*GCUnregisterEvent)(void* hObject, EVENT_TYPE iEventID) = nullptr;
  GC_ERROR (*EventGetData)(EVENT_HANDLE hEvent, void* pBuffer, size_t* piSize,
                           uint64_t iTimeout) = nullptr;
  GC_ERROR (*EventKill)(EVENT_HANDLE hEvent) = nullptr;
};

GenTLFns g_fns;

bool loadSymbol(void* lib, void** fn, const char* name, std::string& status) {
  *fn = dlsym(lib, name);
  if (!*fn) {
    status = std::string("Symbol missing: ") + name;
    return false;
  }
  return true;
}

std::string trimTrailingNulls(std::string value) {
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

template <typename T>
bool getStreamInfo(DS_HANDLE stream, STREAM_INFO_CMD cmd, T& out) {
  if (!g_fns.DSGetInfo) return false;
  size_t size = sizeof(T);
  return g_fns.DSGetInfo(stream, cmd, nullptr, &out, &size) == GC_SUCCESS && size == sizeof(T);
}

template <typename T>
bool getBufferInfo(DS_HANDLE stream, BUFFER_HANDLE buffer, BUFFER_INFO_CMD cmd, T& out) {
  if (!g_fns.DSGetBufferInfo) return false;
  size_t size = sizeof(T);
  return g_fns.DSGetBufferInfo(stream, buffer, cmd, nullptr, &out, &size) == GC_SUCCESS &&
         size == sizeof(T);
}

bool mapPixelFormat(uint64_t pixel_format, PixelFormat& format, int& channels) {
  switch (pixel_format) {
    case PFNC_MONO8:
      format = PixelFormat::Mono8;
      channels = 1;
      return true;
    case PFNC_BAYER_RG8:
      format = PixelFormat::BayerRG8;
      channels = 1;
      return true;
    case PFNC_BAYER_BG8:
      format = PixelFormat::BayerBG8;
      channels = 1;
      return true;
    case PFNC_BAYER_GB8:
      format = PixelFormat::BayerGB8;
      channels = 1;
      return true;
    case PFNC_BAYER_GR8:
      format = PixelFormat::BayerGR8;
      channels = 1;
      return true;
    case PFNC_RGB8:
      format = PixelFormat::RGB8;
      channels = 3;
      return true;
    case PFNC_BGR8:
      format = PixelFormat::BGR8;
      channels = 3;
      return true;
    default:
      format = PixelFormat::Unknown;
      channels = 0;
      return false;
  }
}
}  // namespace

GenTLBackend::~GenTLBackend() {
  unload();
}

bool GenTLBackend::setOption(const std::string& key, const std::string& value) {
  if (key == "cti_path") {
    if (cti_path_ != value) {
      cti_path_ = value;
      unload();
    }
    return true;
  }
  return false;
}

bool GenTLBackend::load() {
  unload();
  if (cti_path_.empty()) {
    status_ = "CTI path vacio";
    return false;
  }

  lib_ = dlopen(cti_path_.c_str(), RTLD_NOW);
  if (!lib_) {
    status_ = std::string("dlopen failed: ") + dlerror();
    return false;
  }

  bool ok = true;
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCInitLib), "GCInitLib", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCCloseLib), "GCCloseLib", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCGetLastError), "GCGetLastError", status_);

  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.TLOpen), "TLOpen", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.TLClose), "TLClose", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.TLUpdateInterfaceList),
                   "TLUpdateInterfaceList", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.TLGetNumInterfaces), "TLGetNumInterfaces",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.TLGetInterfaceID), "TLGetInterfaceID",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.TLOpenInterface), "TLOpenInterface", status_);

  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.IFUpdateDeviceList), "IFUpdateDeviceList",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.IFGetNumDevices), "IFGetNumDevices", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.IFGetDeviceID), "IFGetDeviceID", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.IFOpenDevice), "IFOpenDevice", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.IFClose), "IFClose", status_);

  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DevClose), "DevClose", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DevGetNumDataStreams),
                   "DevGetNumDataStreams", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DevGetDataStreamID), "DevGetDataStreamID",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DevOpenDataStream), "DevOpenDataStream",
                   status_);

  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSClose), "DSClose", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSGetInfo), "DSGetInfo", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSAllocAndAnnounceBuffer),
                   "DSAllocAndAnnounceBuffer", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSQueueBuffer), "DSQueueBuffer", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSStartAcquisition), "DSStartAcquisition",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSStopAcquisition), "DSStopAcquisition",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSFlushQueue), "DSFlushQueue", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSGetBufferInfo), "DSGetBufferInfo", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.DSRevokeBuffer), "DSRevokeBuffer", status_);

  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCRegisterEvent), "GCRegisterEvent", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCUnregisterEvent), "GCUnregisterEvent",
                   status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.EventGetData), "EventGetData", status_);
  ok &= loadSymbol(lib_, reinterpret_cast<void**>(&g_fns.EventKill), "EventKill", status_);

  if (!ok) {
    unload();
    return false;
  }

  GC_ERROR err = g_fns.GCInitLib();
  if (err != GC_SUCCESS) {
    setStatusFromError("GCInitLib fallo", err);
    unload();
    return false;
  }

  TL_HANDLE tl = nullptr;
  err = g_fns.TLOpen(&tl);
  if (err != GC_SUCCESS || !tl) {
    setStatusFromError("TLOpen fallo", err);
    unload();
    return false;
  }
  tl_ = tl;

  status_ = "GenTL cargado";
  return true;
}

void GenTLBackend::unload() {
  close();

  if (tl_ && g_fns.TLClose) {
    g_fns.TLClose(tl_);
  }
  tl_ = nullptr;

  if (lib_) {
    if (g_fns.GCCloseLib) {
      g_fns.GCCloseLib();
    }
    dlclose(lib_);
  }
  lib_ = nullptr;

  g_fns = GenTLFns{};
}

std::vector<CameraInfo> GenTLBackend::listDevices() {
  std::vector<CameraInfo> devices;
  if (!lib_ || !tl_) {
    if (!load()) return devices;
  }

  bool8_t changed = 0;
  g_fns.TLUpdateInterfaceList(tl_, &changed, 100);

  uint32_t num_if = 0;
  GC_ERROR err = g_fns.TLGetNumInterfaces(tl_, &num_if);
  if (err != GC_SUCCESS) {
    setStatusFromError("TLGetNumInterfaces fallo", err);
    return devices;
  }

  for (uint32_t i = 0; i < num_if; ++i) {
    size_t size = 0;
    if (g_fns.TLGetInterfaceID(tl_, i, nullptr, &size) != GC_SUCCESS || size == 0) {
      continue;
    }

    std::string if_id(size, '\0');
    if (g_fns.TLGetInterfaceID(tl_, i, if_id.data(), &size) != GC_SUCCESS) {
      continue;
    }
    if_id = trimTrailingNulls(if_id);
    if (if_id.empty()) continue;

    IF_HANDLE if_handle = nullptr;
    if (g_fns.TLOpenInterface(tl_, if_id.c_str(), &if_handle) != GC_SUCCESS || !if_handle) {
      continue;
    }

    bool8_t dev_changed = 0;
    g_fns.IFUpdateDeviceList(if_handle, &dev_changed, 100);

    uint32_t num_dev = 0;
    if (g_fns.IFGetNumDevices(if_handle, &num_dev) == GC_SUCCESS) {
      for (uint32_t d = 0; d < num_dev; ++d) {
        size_t dev_size = 0;
        if (g_fns.IFGetDeviceID(if_handle, d, nullptr, &dev_size) != GC_SUCCESS || dev_size == 0) {
          continue;
        }

        std::string dev_id(dev_size, '\0');
        if (g_fns.IFGetDeviceID(if_handle, d, dev_id.data(), &dev_size) != GC_SUCCESS) {
          continue;
        }
        dev_id = trimTrailingNulls(dev_id);
        if (dev_id.empty()) continue;

        CameraInfo info;
        info.id = if_id + "|" + dev_id;
        info.label = dev_id;
        devices.push_back(std::move(info));
      }
    }

    if (g_fns.IFClose) {
      g_fns.IFClose(if_handle);
    }
  }

  status_ = devices.empty() ? "Sin dispositivos GenTL" : "Dispositivos GenTL detectados";
  return devices;
}

bool GenTLBackend::open(const CameraInfo& info) {
  close();
  if (!lib_ || !tl_) {
    if (!load()) return false;
  }

  auto sep = info.id.find('|');
  if (sep == std::string::npos) {
    status_ = "ID GenTL invalido";
    return false;
  }

  std::string if_id = info.id.substr(0, sep);
  std::string dev_id = info.id.substr(sep + 1);
  if (if_id.empty() || dev_id.empty()) {
    status_ = "ID GenTL invalido";
    return false;
  }

  IF_HANDLE if_handle = nullptr;
  GC_ERROR err = g_fns.TLOpenInterface(tl_, if_id.c_str(), &if_handle);
  if (err != GC_SUCCESS || !if_handle) {
    setStatusFromError("TLOpenInterface fallo", err);
    return false;
  }
  if_handle_ = if_handle;

  DEV_HANDLE dev = nullptr;
  err = g_fns.IFOpenDevice(if_handle_, dev_id.c_str(), DEVICE_ACCESS_CONTROL, &dev);
  if (err != GC_SUCCESS || !dev) {
    setStatusFromError("IFOpenDevice fallo", err);
    if (g_fns.IFClose) g_fns.IFClose(if_handle_);
    if_handle_ = nullptr;
    return false;
  }
  dev_ = dev;

  status_ = "GenTL device abierto";
  return true;
}

bool GenTLBackend::openDataStream() {
  if (ds_) return true;
  if (!dev_) return false;

  uint32_t num_streams = 0;
  GC_ERROR err = g_fns.DevGetNumDataStreams(dev_, &num_streams);
  if (err != GC_SUCCESS) {
    setStatusFromError("DevGetNumDataStreams fallo", err);
    return false;
  }
  if (num_streams == 0) {
    status_ = "GenTL device sin data streams";
    return false;
  }

  size_t stream_id_size = 0;
  err = g_fns.DevGetDataStreamID(dev_, 0, nullptr, &stream_id_size);
  if (err != GC_SUCCESS || stream_id_size == 0) {
    setStatusFromError("DevGetDataStreamID fallo", err);
    return false;
  }

  std::string stream_id(stream_id_size, '\0');
  err = g_fns.DevGetDataStreamID(dev_, 0, stream_id.data(), &stream_id_size);
  if (err != GC_SUCCESS) {
    setStatusFromError("DevGetDataStreamID fallo", err);
    return false;
  }

  stream_id = trimTrailingNulls(stream_id);
  if (stream_id.empty()) {
    status_ = "DevGetDataStreamID vacio";
    return false;
  }

  DS_HANDLE stream = nullptr;
  err = g_fns.DevOpenDataStream(dev_, stream_id.c_str(), &stream);
  if (err != GC_SUCCESS || !stream) {
    setStatusFromError("DevOpenDataStream fallo", err);
    return false;
  }

  ds_ = stream;
  return true;
}

bool GenTLBackend::registerNewBufferEvent() {
  if (new_buffer_event_) return true;
  if (!ds_) return false;

  EVENT_HANDLE event_handle = nullptr;
  GC_ERROR err = g_fns.GCRegisterEvent(ds_, EVENT_NEW_BUFFER, &event_handle);
  if (err != GC_SUCCESS || !event_handle) {
    setStatusFromError("GCRegisterEvent(EVENT_NEW_BUFFER) fallo", err);
    return false;
  }

  new_buffer_event_ = event_handle;
  return true;
}

bool GenTLBackend::queryPayloadSize(size_t& payload_size) const {
  return getStreamInfo(ds_, STREAM_INFO_PAYLOAD_SIZE, payload_size);
}

bool GenTLBackend::queryMinAnnouncedBuffers(size_t& min_buffers) const {
  return getStreamInfo(ds_, STREAM_INFO_BUF_ANNOUNCE_MIN, min_buffers);
}

void GenTLBackend::setStatusFromError(const char* operation, GenTLError err) {
  std::string detail;
  if (g_fns.GCGetLastError) {
    GC_ERROR error_code = err;
    size_t msg_size = 0;
    if (g_fns.GCGetLastError(&error_code, nullptr, &msg_size) == GC_SUCCESS && msg_size > 0) {
      detail.assign(msg_size, '\0');
      if (g_fns.GCGetLastError(&error_code, detail.data(), &msg_size) == GC_SUCCESS) {
        detail = trimTrailingNulls(detail);
      } else {
        detail.clear();
      }
    }
  }

  if (detail.empty()) {
    status_ = std::string(operation) + " [err=" + std::to_string(err) + "]";
  } else {
    status_ = std::string(operation) + ": " + detail + " [err=" + std::to_string(err) + "]";
  }
}

bool GenTLBackend::queueBuffer(GenTLHandle buffer) {
  if (!ds_ || !buffer) return false;
  GC_ERROR err = g_fns.DSQueueBuffer(ds_, buffer);
  if (err != GC_SUCCESS) {
    setStatusFromError("DSQueueBuffer fallo", err);
    return false;
  }
  return true;
}

void GenTLBackend::close() {
  stop();

  if (dev_ && g_fns.DevClose) {
    g_fns.DevClose(dev_);
  }
  dev_ = nullptr;

  if (if_handle_ && g_fns.IFClose) {
    g_fns.IFClose(if_handle_);
  }
  if_handle_ = nullptr;
}

bool GenTLBackend::start() {
  if (!dev_) {
    status_ = "No hay dispositivo GenTL abierto";
    return false;
  }
  if (acquisition_started_) return true;

  if (!openDataStream()) {
    return false;
  }
  if (!registerNewBufferEvent()) {
    stop();
    return false;
  }

  if (!queryPayloadSize(payload_size_) || payload_size_ == 0) {
    status_ = "STREAM_INFO_PAYLOAD_SIZE no disponible";
    stop();
    return false;
  }

  size_t min_buffers = 0;
  if (!queryMinAnnouncedBuffers(min_buffers)) {
    status_ = "STREAM_INFO_BUF_ANNOUNCE_MIN no disponible";
    stop();
    return false;
  }

  const size_t buffer_count = std::max<size_t>(16, min_buffers);
  announced_buffers_.reserve(buffer_count);
  for (size_t i = 0; i < buffer_count; ++i) {
    BUFFER_HANDLE buffer = nullptr;
    GC_ERROR err = g_fns.DSAllocAndAnnounceBuffer(ds_, payload_size_, nullptr, &buffer);
    if (err != GC_SUCCESS || !buffer) {
      setStatusFromError("DSAllocAndAnnounceBuffer fallo", err);
      stop();
      return false;
    }

    announced_buffers_.push_back(buffer);
    if (!queueBuffer(buffer)) {
      stop();
      return false;
    }
  }

  GC_ERROR err = g_fns.DSStartAcquisition(ds_, ACQ_START_FLAGS_DEFAULT, GENTL_INFINITE);
  if (err != GC_SUCCESS) {
    setStatusFromError("DSStartAcquisition fallo", err);
    stop();
    return false;
  }

  acquisition_started_ = true;
  status_ = "GenTL streaming";
  return true;
}

void GenTLBackend::stop() {
  if (new_buffer_event_ && g_fns.EventKill) {
    g_fns.EventKill(new_buffer_event_);
  }

  if (acquisition_started_ && ds_ && g_fns.DSStopAcquisition) {
    GC_ERROR err = g_fns.DSStopAcquisition(ds_, ACQ_STOP_FLAGS_DEFAULT);
    if (err != GC_SUCCESS && err != GC_ERR_ABORT) {
      setStatusFromError("DSStopAcquisition fallo", err);
    }
  }
  acquisition_started_ = false;

  if (ds_ && g_fns.DSFlushQueue) {
    g_fns.DSFlushQueue(ds_, ACQ_QUEUE_ALL_TO_INPUT);
  }

  if (ds_ && g_fns.DSRevokeBuffer) {
    for (auto buffer : announced_buffers_) {
      if (!buffer) continue;
      void* base = nullptr;
      void* priv = nullptr;
      g_fns.DSRevokeBuffer(ds_, buffer, &base, &priv);
    }
  }
  announced_buffers_.clear();

  if (ds_ && g_fns.GCUnregisterEvent) {
    g_fns.GCUnregisterEvent(ds_, EVENT_NEW_BUFFER);
  }
  new_buffer_event_ = nullptr;

  if (ds_ && g_fns.DSClose) {
    g_fns.DSClose(ds_);
  }
  ds_ = nullptr;
  payload_size_ = 0;
}

bool GenTLBackend::grab(Frame& out) {
  if (!acquisition_started_ || !ds_ || !new_buffer_event_) return false;

  EVENT_NEW_BUFFER_DATA event_data{};
  size_t event_size = sizeof(event_data);
  GC_ERROR err = g_fns.EventGetData(new_buffer_event_, &event_data, &event_size, 200);
  if (err == GC_ERR_TIMEOUT || err == GC_ERR_ABORT) {
    return false;
  }
  if (err != GC_SUCCESS) {
    setStatusFromError("EventGetData fallo", err);
    return false;
  }

  BUFFER_HANDLE buffer = event_data.buffer_handle;
  if (!buffer) {
    status_ = "EVENT_NEW_BUFFER sin handle de buffer";
    return false;
  }

  bool ok = false;
  auto finish = [&]() {
    if (!queueBuffer(buffer)) {
      return false;
    }
    return ok;
  };

  void* base_ptr = nullptr;
  if (!getBufferInfo(ds_, buffer, BUFFER_INFO_BASE, base_ptr) || !base_ptr) {
    status_ = "BUFFER_INFO_BASE invalido";
    return finish();
  }

  size_t data_size = 0;
  if (!getBufferInfo(ds_, buffer, BUFFER_INFO_SIZE_FILLED, data_size)) {
    if (!getBufferInfo(ds_, buffer, BUFFER_INFO_DATA_SIZE, data_size)) {
      data_size = payload_size_;
    }
  }

  bool8_t image_present = 1;
  if (getBufferInfo(ds_, buffer, BUFFER_INFO_IMAGEPRESENT, image_present) && !image_present) {
    status_ = "Buffer sin imagen";
    return finish();
  }

  bool8_t incomplete = 0;
  if (getBufferInfo(ds_, buffer, BUFFER_INFO_IS_INCOMPLETE, incomplete) && incomplete) {
    status_ = "Buffer incompleto";
    return finish();
  }

  size_t width = 0;
  size_t height = 0;
  if (!getBufferInfo(ds_, buffer, BUFFER_INFO_WIDTH, width) ||
      !getBufferInfo(ds_, buffer, BUFFER_INFO_HEIGHT, height) || width == 0 || height == 0) {
    status_ = "BUFFER_INFO_WIDTH/HEIGHT no disponible";
    return finish();
  }

  uint64_t payload_type = PAYLOAD_TYPE_IMAGE;
  if (getBufferInfo(ds_, buffer, BUFFER_INFO_PAYLOADTYPE, payload_type) &&
      payload_type != PAYLOAD_TYPE_IMAGE) {
    status_ = "Payload no es imagen";
    return finish();
  }

  uint64_t pixel_format = 0;
  if (!getBufferInfo(ds_, buffer, BUFFER_INFO_PIXELFORMAT, pixel_format)) {
    status_ = "BUFFER_INFO_PIXELFORMAT no disponible";
    return finish();
  }

  uint64_t pixel_format_ns = 0;
  (void)getBufferInfo(ds_, buffer, BUFFER_INFO_PIXELFORMAT_NAMESPACE, pixel_format_ns);

  PixelFormat format = PixelFormat::Unknown;
  int channels = 0;
  if (!mapPixelFormat(pixel_format, format, channels)) {
    status_ = "PixelFormat GenTL no soportado";
    return finish();
  }

  if (width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      height > static_cast<size_t>(std::numeric_limits<int>::max())) {
    status_ = "Tamano de imagen fuera de rango";
    return finish();
  }

  size_t image_offset = 0;
  (void)getBufferInfo(ds_, buffer, BUFFER_INFO_IMAGEOFFSET, image_offset);

  size_t xpadding = 0;
  (void)getBufferInfo(ds_, buffer, BUFFER_INFO_XPADDING, xpadding);

  if (width > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels)) {
    status_ = "Overflow en calculo de stride";
    return finish();
  }
  const size_t stride = width * static_cast<size_t>(channels) + xpadding;

  if (height > 0 && stride > std::numeric_limits<size_t>::max() / height) {
    status_ = "Overflow en tamano de imagen";
    return finish();
  }
  const size_t required_bytes = stride * height;

  if (image_offset > data_size || required_bytes > (data_size - image_offset)) {
    status_ = "Buffer demasiado pequeno para la imagen";
    return finish();
  }

  const uint8_t* image_ptr = static_cast<const uint8_t*>(base_ptr) + image_offset;
  auto buffer_copy = makeFrameBuffer(required_bytes);
  std::memcpy(buffer_copy.get(), image_ptr, required_bytes);

  out.data = std::move(buffer_copy);
  out.data_size = required_bytes;
  out.width = static_cast<int>(width);
  out.height = static_cast<int>(height);
  out.stride = static_cast<int>(stride);
  out.format = format;
  out.timestamp_ns = 0;

  uint64_t timestamp_ns = 0;
  if (getBufferInfo(ds_, buffer, BUFFER_INFO_TIMESTAMP_NS, timestamp_ns)) {
    out.timestamp_ns = timestamp_ns;
  } else {
    uint64_t timestamp = 0;
    if (getBufferInfo(ds_, buffer, BUFFER_INFO_TIMESTAMP, timestamp)) {
      out.timestamp_ns = timestamp;
    }
  }

  uint64_t frame_id = 0;
  if (getBufferInfo(ds_, buffer, BUFFER_INFO_FRAMEID, frame_id)) {
    out.seq = frame_id;
  }

  ok = true;
  return finish();
}

bool GenTLBackend::isOpen() const {
  return dev_ != nullptr;
}
