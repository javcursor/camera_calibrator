#include "camera/GenTLBackend.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef HAVE_GENTL_GENAPI
#include <Base/GCException.h>
#include <GenApi/IBoolean.h>
#include <GenApi/ICommand.h>
#include <GenApi/IFloat.h>
#include <GenApi/IInteger.h>
#include <GenApi/INodeMap.h>
#include <GenApi/IPort.h>
#include <GenApi/IString.h>
#include <GenApi/NodeMapFactory.h>
#include <GenApi/NodeMapRef.h>
#include <GenApi/Pointer.h>
#endif

namespace {
using bool8_t = uint8_t;
using GC_ERROR = int32_t;
using INFO_DATATYPE = int32_t;
using TL_HANDLE = void*;
using IF_HANDLE = void*;
using DEV_HANDLE = void*;
using PORT_HANDLE = void*;
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
using URL_INFO_CMD = int32_t;

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

constexpr URL_INFO_CMD URL_INFO_URL = 0;
constexpr URL_INFO_CMD URL_INFO_FILE_REGISTER_ADDRESS = 7;
constexpr URL_INFO_CMD URL_INFO_FILE_SIZE = 8;
constexpr URL_INFO_CMD URL_INFO_SCHEME = 9;
constexpr URL_INFO_CMD URL_INFO_FILENAME = 10;

constexpr int32_t URL_SCHEME_LOCAL = 0;
constexpr int32_t URL_SCHEME_FILE = 2;

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
  GC_ERROR (*DevGetPort)(DEV_HANDLE hDevice, PORT_HANDLE* phRemoteDevice) = nullptr;
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
  GC_ERROR (*GCGetNumPortURLs)(PORT_HANDLE hPort, uint32_t* piNumURLs) = nullptr;
  GC_ERROR (*GCGetPortURL)(PORT_HANDLE hPort, char* sURL, size_t* piSize) = nullptr;
  GC_ERROR (*GCGetPortURLInfo)(PORT_HANDLE hPort, uint32_t iURLIndex, URL_INFO_CMD iInfoCmd,
                               INFO_DATATYPE* piType, void* pBuffer, size_t* piSize) = nullptr;
  GC_ERROR (*GCReadPort)(PORT_HANDLE hPort, uint64_t iAddress, void* pBuffer, size_t* piSize) =
      nullptr;
  GC_ERROR (*GCWritePort)(PORT_HANDLE hPort, uint64_t iAddress, const void* pBuffer,
                          size_t* piSize) = nullptr;
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

void loadOptionalSymbol(void* lib, void** fn, const char* name) {
  *fn = dlsym(lib, name);
}

std::string trimTrailingNulls(std::string value) {
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

char toLowerAscii(char c) {
  if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
  return c;
}

bool startsWithIgnoreCase(const std::string& value, const std::string& prefix) {
  if (value.size() < prefix.size()) return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (toLowerAscii(value[i]) != toLowerAscii(prefix[i])) {
      return false;
    }
  }
  return true;
}

std::string stripUrlQuery(const std::string& value) {
  const size_t query = value.find('?');
  return query == std::string::npos ? value : value.substr(0, query);
}

#ifdef HAVE_GENTL_GENAPI
constexpr size_t kPortIoChunkSize = 4096;

std::string toStdString(const GENICAM_NAMESPACE::gcstring& value) {
  return value.c_str() ? std::string(value.c_str()) : std::string();
}

struct PortUrl {
  enum class Kind { Local, File };

  Kind kind = Kind::Local;
  std::string path;
  uint64_t address = 0;
  size_t length = 0;
  bool zipped = false;
};

bool parseHexUint64(const std::string& text, uint64_t& out) {
  if (text.empty()) return false;
  try {
    size_t used = 0;
    out = std::stoull(text, &used, 16);
    return used == text.size();
  } catch (...) {
    return false;
  }
}

bool parsePortUrl(const std::string& url, PortUrl& out) {
  const std::string without_query = stripUrlQuery(url);

  if (startsWithIgnoreCase(without_query, "local:")) {
    size_t path_start = std::strlen("local:");
    while (path_start < without_query.size() && without_query[path_start] == '/') {
      ++path_start;
    }
    const std::string rest = without_query.substr(path_start);
    const size_t first_sep = rest.find(';');
    const size_t second_sep = first_sep == std::string::npos ? std::string::npos : rest.find(';', first_sep + 1);
    if (first_sep == std::string::npos || second_sep == std::string::npos) return false;

    out.kind = PortUrl::Kind::Local;
    out.path = rest.substr(0, first_sep);
    uint64_t length = 0;
    if (!parseHexUint64(rest.substr(first_sep + 1, second_sep - first_sep - 1), out.address) ||
        !parseHexUint64(rest.substr(second_sep + 1), length)) {
      return false;
    }
    out.length = static_cast<size_t>(length);
    out.zipped = out.path.size() >= 4 &&
                 out.path.compare(out.path.size() - 4, 4, ".zip") == 0;
    return out.length > 0;
  }

  if (startsWithIgnoreCase(without_query, "file:")) {
    out.kind = PortUrl::Kind::File;
    out.path = without_query.substr(std::strlen("file:"));
    while (out.path.size() >= 2 && out.path[0] == '/' && out.path[1] == '/') {
      out.path.erase(0, 1);
    }
    out.zipped = out.path.size() >= 4 &&
                 out.path.compare(out.path.size() - 4, 4, ".zip") == 0;
    return !out.path.empty();
  }

  return false;
}

bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  input.seekg(0, std::ios::end);
  const std::streamoff length = input.tellg();
  if (length <= 0) return false;
  input.seekg(0, std::ios::beg);
  out.resize(static_cast<size_t>(length));
  input.read(reinterpret_cast<char*>(out.data()), length);
  return input.good() || input.eof();
}

bool readPortRange(PORT_HANDLE port, uint64_t address, void* buffer, size_t length) {
  if (!g_fns.GCReadPort || !port) return false;
  auto* out = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    const size_t chunk = std::min(remaining, kPortIoChunkSize);
    size_t transferred = chunk;
    if (g_fns.GCReadPort(port, address, out, &transferred) != GC_SUCCESS || transferred != chunk) {
      return false;
    }
    out += chunk;
    address += chunk;
    remaining -= chunk;
  }
  return true;
}

bool writePortRange(PORT_HANDLE port, uint64_t address, const void* buffer, size_t length) {
  if (!g_fns.GCWritePort || !port) return false;
  const auto* in = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    const size_t chunk = std::min(remaining, kPortIoChunkSize);
    size_t transferred = chunk;
    if (g_fns.GCWritePort(port, address, in, &transferred) != GC_SUCCESS || transferred != chunk) {
      return false;
    }
    in += chunk;
    address += chunk;
    remaining -= chunk;
  }
  return true;
}

bool getPortUrlInfoString(PORT_HANDLE port, uint32_t index, URL_INFO_CMD cmd, std::string& out) {
  if (!g_fns.GCGetPortURLInfo) return false;
  INFO_DATATYPE type = 0;
  size_t size = 0;
  if (g_fns.GCGetPortURLInfo(port, index, cmd, &type, nullptr, &size) != GC_SUCCESS || size == 0) {
    return false;
  }
  std::string value(size, '\0');
  if (g_fns.GCGetPortURLInfo(port, index, cmd, &type, value.data(), &size) != GC_SUCCESS) {
    return false;
  }
  out = trimTrailingNulls(value);
  return !out.empty();
}

template <typename T>
bool getPortUrlInfoNumeric(PORT_HANDLE port, uint32_t index, URL_INFO_CMD cmd, T& out) {
  if (!g_fns.GCGetPortURLInfo) return false;
  std::array<uint8_t, sizeof(uint64_t)> raw{};
  INFO_DATATYPE type = 0;
  size_t size = raw.size();
  if (g_fns.GCGetPortURLInfo(port, index, cmd, &type, raw.data(), &size) != GC_SUCCESS ||
      size == 0 || size > raw.size()) {
    return false;
  }
  uint64_t value = 0;
  std::memcpy(&value, raw.data(), size);
  if constexpr (std::is_integral_v<T>) {
    if (value > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
      return false;
    }
  }
  out = static_cast<T>(value);
  return true;
}

bool readPortBytes(PORT_HANDLE port, uint64_t address, size_t length, std::vector<uint8_t>& out) {
  if (!port || length == 0) return false;
  out.resize(length);
  return readPortRange(port, address, out.data(), length);
}

bool fetchPortXml(PORT_HANDLE port, const std::vector<std::string>& urls, std::vector<uint8_t>& data,
                  bool& zipped) {
  for (const std::string& url_text : urls) {
    PortUrl url;
    if (!parsePortUrl(url_text, url)) continue;

    std::vector<uint8_t> candidate;
    const bool ok = url.kind == PortUrl::Kind::Local ? readPortBytes(port, url.address, url.length, candidate)
                                                     : readFileBytes(url.path, candidate);
    if (!ok || candidate.empty()) continue;

    data = std::move(candidate);
    zipped = url.zipped;
    return true;
  }

  return false;
}

bool trySetEnumValue(GenApi::INodeMap& node_map, const char* name, const char* value) {
  GenApi::CEnumerationPtr param(node_map.GetNode(name));
  if (!GenApi::IsWritable(param)) {
    return false;
  }
  GenApi::CEnumEntryPtr entry(param->GetEntryByName(value));
  if (!GenApi::IsReadable(entry)) {
    return false;
  }
  param->FromString(value, true);
  return true;
}

bool tryExecuteCommand(GenApi::INodeMap& node_map, const char* name) {
  GenApi::CCommandPtr command(node_map.GetNode(name));
  if (!GenApi::IsWritable(command)) {
    return false;
  }
  command->Execute(true);
  return true;
}

bool trySetIntegerValue(GenApi::INodeMap& node_map, const char* name, int64_t value) {
  GenApi::CIntegerPtr param(node_map.GetNode(name));
  if (!GenApi::IsWritable(param)) {
    return false;
  }
  param->SetValue(value, true);
  return true;
}

void tryDisableTrigger(GenApi::INodeMap& node_map, const char* selector) {
  if (!trySetEnumValue(node_map, "TriggerSelector", selector)) {
    return;
  }
  (void)trySetEnumValue(node_map, "TriggerMode", "Off");
}

FeatureType featureTypeFromNode(const GenApi::INode& node) {
  switch (node.GetPrincipalInterfaceType()) {
    case GenApi::intfIInteger:
      return FeatureType::Integer;
    case GenApi::intfIFloat:
      return FeatureType::Float;
    case GenApi::intfIBoolean:
      return FeatureType::Boolean;
    case GenApi::intfIEnumeration:
      return FeatureType::Enumeration;
    case GenApi::intfIString:
      return FeatureType::String;
    case GenApi::intfICommand:
      return FeatureType::Command;
    case GenApi::intfICategory:
      return FeatureType::Category;
    default:
      return FeatureType::Unknown;
  }
}

bool isUserFeatureNode(GenApi::INode* node) {
  if (!node || !node->IsFeature() || !GenApi::IsImplemented(node) || !GenApi::IsAvailable(node)) {
    return false;
  }
  const std::string name = toStdString(node->GetName());
  return !name.empty() && name.rfind("d_", 0) != 0;
}

void fillEnumEntries(GenApi::INode& node, FeatureInfo& info) {
  GenApi::CEnumerationPtr enumeration(&node);
  if (!GenApi::IsAvailable(enumeration)) {
    return;
  }
  GenApi::NodeList_t entries;
  enumeration->GetEntries(entries);
  info.enum_entries.reserve(entries.size());
  for (GenApi::INode* entry_node : entries) {
    GenApi::CEnumEntryPtr entry(entry_node);
    if (!GenApi::IsAvailable(entry)) {
      continue;
    }
    const std::string symbolic = toStdString(entry->GetSymbolic());
    if (!symbolic.empty()) {
      info.enum_entries.push_back(symbolic);
    }
  }
}

void fillFeatureInfo(GenApi::INode& node, FeatureInfo& info) {
  info.id = toStdString(node.GetName());
  info.display_name = toStdString(node.GetDisplayName());
  if (info.display_name.empty()) {
    info.display_name = info.id;
  }
  info.type = featureTypeFromNode(node);
  info.readable = GenApi::IsReadable(&node);
  info.writable = GenApi::IsWritable(&node);
  info.backend_handle = &node;

  if (info.type == FeatureType::Integer) {
    GenApi::CIntegerPtr integer(&node);
    if (GenApi::IsAvailable(integer)) {
      info.min = static_cast<double>(integer->GetMin());
      info.max = static_cast<double>(integer->GetMax());
    }
  } else if (info.type == FeatureType::Float) {
    GenApi::CFloatPtr floating(&node);
    if (GenApi::IsAvailable(floating)) {
      info.min = floating->GetMin();
      info.max = floating->GetMax();
    }
  } else if (info.type == FeatureType::Enumeration) {
    fillEnumEntries(node, info);
  }
}

bool readFeatureValueFromNode(GenApi::INode& node, FeatureType type, FeatureValue& out) {
  switch (type) {
    case FeatureType::Integer: {
      GenApi::CIntegerPtr integer(&node);
      if (!GenApi::IsReadable(integer)) return false;
      out.value = integer->GetValue(false, false);
      return true;
    }
    case FeatureType::Float: {
      GenApi::CFloatPtr floating(&node);
      if (!GenApi::IsReadable(floating)) return false;
      out.value = floating->GetValue(false, false);
      return true;
    }
    case FeatureType::Boolean: {
      GenApi::CBooleanPtr boolean(&node);
      if (!GenApi::IsReadable(boolean)) return false;
      out.value = boolean->GetValue(false, false);
      return true;
    }
    case FeatureType::Enumeration:
    case FeatureType::String: {
      GenApi::CValuePtr value(&node);
      if (!GenApi::IsReadable(value)) return false;
      out.value = toStdString(value->ToString(false, false));
      return true;
    }
    default:
      return false;
  }
}

bool writeFeatureValueToNode(GenApi::INode& node, FeatureType type, const FeatureValue& value) {
  switch (type) {
    case FeatureType::Integer: {
      GenApi::CIntegerPtr integer(&node);
      const auto* v = std::get_if<int64_t>(&value.value);
      if (!GenApi::IsWritable(integer) || !v) return false;
      integer->SetValue(*v, true);
      return true;
    }
    case FeatureType::Float: {
      GenApi::CFloatPtr floating(&node);
      const auto* v = std::get_if<double>(&value.value);
      if (!GenApi::IsWritable(floating) || !v) return false;
      floating->SetValue(*v, true);
      return true;
    }
    case FeatureType::Boolean: {
      GenApi::CBooleanPtr boolean(&node);
      const auto* v = std::get_if<bool>(&value.value);
      if (!GenApi::IsWritable(boolean) || !v) return false;
      boolean->SetValue(*v, true);
      return true;
    }
    case FeatureType::Enumeration:
    case FeatureType::String: {
      GenApi::CValuePtr string_value(&node);
      const auto* v = std::get_if<std::string>(&value.value);
      if (!GenApi::IsWritable(string_value) || !v) return false;
      string_value->FromString(v->c_str(), true);
      return true;
    }
    default:
      return false;
  }
}
#endif

template <typename T, typename QueryFn>
bool readInfoValue(QueryFn&& query, T& out) {
  INFO_DATATYPE type = 0;

  if constexpr (std::is_pointer_v<T>) {
    size_t size = sizeof(T);
    T value = nullptr;
    if (query(&type, &value, &size) != GC_SUCCESS || size != sizeof(T)) {
      return false;
    }
    out = value;
    return true;
  } else if constexpr (std::is_same_v<T, bool8_t>) {
    std::array<uint8_t, sizeof(uint64_t)> raw{};
    size_t size = raw.size();
    if (query(&type, raw.data(), &size) != GC_SUCCESS || size == 0 || size > raw.size()) {
      return false;
    }

    uint64_t value = 0;
    std::memcpy(&value, raw.data(), size);
    out = value != 0 ? 1 : 0;
    return true;
  } else if constexpr (std::is_integral_v<T>) {
    std::array<uint8_t, sizeof(uint64_t)> raw{};
    size_t size = raw.size();
    if (query(&type, raw.data(), &size) != GC_SUCCESS || size == 0 || size > raw.size()) {
      return false;
    }

    uint64_t value = 0;
    std::memcpy(&value, raw.data(), size);
    if (value > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
      return false;
    }
    out = static_cast<T>(value);
    return true;
  } else {
    size_t size = sizeof(T);
    return query(&type, &out, &size) == GC_SUCCESS && size == sizeof(T);
  }
}

template <typename T>
bool getStreamInfo(DS_HANDLE stream, STREAM_INFO_CMD cmd, T& out) {
  if (!g_fns.DSGetInfo) return false;
  return readInfoValue([&](INFO_DATATYPE* type, void* buffer, size_t* size) {
    return g_fns.DSGetInfo(stream, cmd, type, buffer, size);
  }, out);
}

template <typename T>
bool getBufferInfo(DS_HANDLE stream, BUFFER_HANDLE buffer, BUFFER_INFO_CMD cmd, T& out) {
  if (!g_fns.DSGetBufferInfo) return false;
  return readInfoValue([&](INFO_DATATYPE* type, void* data, size_t* size) {
    return g_fns.DSGetBufferInfo(stream, buffer, cmd, type, data, size);
  }, out);
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

struct GenTLBackend::RemoteControlState {
#ifdef HAVE_GENTL_GENAPI
  class PortProxy : public GenApi::IPort {
   public:
    explicit PortProxy(PORT_HANDLE port) : port_(port) {}

    GenApi::EAccessMode GetAccessMode() const override { return GenApi::RW; }

    void Read(void* buffer, int64_t address, int64_t length) override {
      const size_t size = length > 0 ? static_cast<size_t>(length) : 0;
      if (size == 0) return;
      if (!readPortRange(port_, static_cast<uint64_t>(address), buffer, size)) {
        throw RUNTIME_EXCEPTION("GCReadPort failed");
      }
    }

    void Write(const void* buffer, int64_t address, int64_t length) override {
      const size_t size = length > 0 ? static_cast<size_t>(length) : 0;
      if (size == 0) return;
      if (!writePortRange(port_, static_cast<uint64_t>(address), buffer, size)) {
        throw RUNTIME_EXCEPTION("GCWritePort failed");
      }
    }

   private:
    PORT_HANDLE port_ = nullptr;
  };

  PORT_HANDLE port = nullptr;
  std::vector<uint8_t> xml_data;
  std::unique_ptr<PortProxy> port_proxy;
  GenApi::INodeMap* node_map_ptr = nullptr;
  GenApi::CNodeMapRef node_map;
  bool loaded = false;
#endif
};

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

  loadOptionalSymbol(lib_, reinterpret_cast<void**>(&g_fns.DevGetPort), "DevGetPort");
  loadOptionalSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCGetNumPortURLs), "GCGetNumPortURLs");
  loadOptionalSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCGetPortURL), "GCGetPortURL");
  loadOptionalSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCGetPortURLInfo), "GCGetPortURLInfo");
  loadOptionalSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCReadPort), "GCReadPort");
  loadOptionalSymbol(lib_, reinterpret_cast<void**>(&g_fns.GCWritePort), "GCWritePort");

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
  remote_status_.clear();

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

bool GenTLBackend::ensureRemoteControl() {
  if (remote_control_ && remote_control_->loaded) {
    return true;
  }
  return initializeRemoteControl();
}

bool GenTLBackend::initializeRemoteControl() {
  releaseRemoteControl();
#ifndef HAVE_GENTL_GENAPI
  status_ = "GenApi remoto no compilado";
  remote_status_ = status_;
  return false;
#else
  if (!dev_ || !g_fns.DevGetPort || !g_fns.GCGetNumPortURLs ||
      (!g_fns.GCGetPortURL && !g_fns.GCGetPortURLInfo)) {
    status_ = "GenTL sin API de puerto remoto";
    remote_status_ = status_;
    return false;
  }

  PORT_HANDLE port = nullptr;
  GC_ERROR err = g_fns.DevGetPort(dev_, &port);
  if (err != GC_SUCCESS || !port) {
    setStatusFromError("DevGetPort fallo", err);
    remote_status_ = status_;
    return false;
  }

  uint32_t num_urls = 0;
  err = g_fns.GCGetNumPortURLs(port, &num_urls);
  if (err != GC_SUCCESS || num_urls == 0) {
    if (err != GC_SUCCESS) {
      setStatusFromError("GCGetNumPortURLs fallo", err);
    } else {
      status_ = "RemoteDevice sin PortURLs GenApi";
    }
    remote_status_ = status_;
    return false;
  }

  std::vector<std::string> urls;
  urls.reserve(num_urls);
  for (uint32_t i = 0; i < num_urls; ++i) {
    std::string url;
    if (getPortUrlInfoString(port, i, URL_INFO_URL, url)) {
      urls.push_back(std::move(url));
      continue;
    }

    int32_t scheme = -1;
    uint64_t address = 0;
    uint64_t file_size = 0;
    std::string filename;
    const bool have_scheme = getPortUrlInfoNumeric(port, i, URL_INFO_SCHEME, scheme);
    const bool have_address = getPortUrlInfoNumeric(port, i, URL_INFO_FILE_REGISTER_ADDRESS, address);
    const bool have_size = getPortUrlInfoNumeric(port, i, URL_INFO_FILE_SIZE, file_size);
    (void)getPortUrlInfoString(port, i, URL_INFO_FILENAME, filename);

    if (have_scheme) {
      if (scheme == URL_SCHEME_LOCAL && have_address && have_size && file_size > 0) {
        if (filename.empty()) filename = "remote_device.xml";
        urls.push_back("local:///" + filename + ";" + std::to_string(address) + ";" +
                       std::to_string(file_size));
        continue;
      }
      if (scheme == URL_SCHEME_FILE && !filename.empty()) {
        urls.push_back("file:///" + filename);
        continue;
      }
    }

    if (i == 0 && g_fns.GCGetPortURL) {
      size_t size = 0;
      if (g_fns.GCGetPortURL(port, nullptr, &size) == GC_SUCCESS && size > 0) {
        std::string legacy_url(size, '\0');
        if (g_fns.GCGetPortURL(port, legacy_url.data(), &size) == GC_SUCCESS) {
          legacy_url = trimTrailingNulls(legacy_url);
          if (!legacy_url.empty()) {
            urls.push_back(std::move(legacy_url));
          }
        }
      }
    }
  }
  if (urls.empty()) {
    status_ = "PortURLs GenApi vacias";
    remote_status_ = status_;
    return false;
  }

  std::vector<uint8_t> xml_data;
  bool zipped = false;
  if (!fetchPortXml(port, urls, xml_data, zipped) || xml_data.empty()) {
    status_ = "No se pudo cargar XML remoto GenApi";
    remote_status_ = status_;
    return false;
  }

  try {
    GenApi::CNodeMapFactory factory(zipped ? GenApi::ContentType_ZippedXml : GenApi::ContentType_Xml,
                                    xml_data.data(), xml_data.size(), GenApi::CacheUsage_Ignore, true);
    GenApi::INodeMap* node_map = factory.CreateNodeMap("RemoteDevice", true);
    if (!node_map) {
      return false;
    }

    std::unique_ptr<RemoteControlState> state(new RemoteControlState());
    state->port = port;
    state->xml_data = std::move(xml_data);
    state->port_proxy = std::make_unique<RemoteControlState::PortProxy>(port);
    state->node_map_ptr = node_map;
    state->node_map = node_map;
    if (!state->node_map._Connect(state->port_proxy.get())) {
      return false;
    }
    state->loaded = true;
    remote_control_ = state.release();
    remote_status_ = "GenApi remoto activo";
    return true;
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("GenApi remoto fallo: ") + e.what();
    remote_status_ = status_;
    releaseRemoteControl();
    return false;
  } catch (const std::exception& e) {
    status_ = std::string("GenApi remoto fallo: ") + e.what();
    remote_status_ = status_;
    releaseRemoteControl();
    return false;
  }
#endif
}

void GenTLBackend::releaseRemoteControl() {
  delete remote_control_;
  remote_control_ = nullptr;
}

bool GenTLBackend::prepareRemoteAcquisition() {
#ifndef HAVE_GENTL_GENAPI
  return false;
#else
  if (!remote_control_ || !remote_control_->loaded) {
    return false;
  }

  try {
    GenApi::INodeMap& node_map = *remote_control_->node_map_ptr;
    (void)trySetIntegerValue(node_map, "TLParamsLocked", 1);
    trySetEnumValue(node_map, "AcquisitionMode", "Continuous");
    tryDisableTrigger(node_map, "AcquisitionStart");
    tryDisableTrigger(node_map, "FrameBurstStart");
    tryDisableTrigger(node_map, "FrameStart");
    return true;
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("Preparacion GenApi fallo: ") + e.what();
    return false;
  }
#endif
}

bool GenTLBackend::startRemoteAcquisition() {
#ifndef HAVE_GENTL_GENAPI
  return false;
#else
  if (!remote_control_ || !remote_control_->loaded) {
    return true;
  }

  try {
    if (remote_control_->node_map_ptr && remote_control_->node_map_ptr->GetNode("AcquisitionStart")) {
      return tryExecuteCommand(*remote_control_->node_map_ptr, "AcquisitionStart");
    }
    return true;
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("AcquisitionStart fallo: ") + e.what();
    return false;
  }
#endif
}

void GenTLBackend::stopRemoteAcquisition() {
#ifdef HAVE_GENTL_GENAPI
  if (!remote_control_ || !remote_control_->loaded) {
    return;
  }

  try {
    if (remote_control_->node_map_ptr && remote_control_->node_map_ptr->GetNode("AcquisitionStop")) {
      (void)tryExecuteCommand(*remote_control_->node_map_ptr, "AcquisitionStop");
    }
    if (remote_control_->node_map_ptr) {
      (void)trySetIntegerValue(*remote_control_->node_map_ptr, "TLParamsLocked", 0);
    }
  } catch (...) {
  }
#endif
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
  releaseRemoteControl();
  remote_status_.clear();

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
  const bool remote_ready = ensureRemoteControl();
  const std::string remote_status = remote_status_;
  if (remote_ready && !prepareRemoteAcquisition()) {
    remote_status_ = status_;
    stop();
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
  if (remote_ready && !startRemoteAcquisition()) {
    stop();
    return false;
  }

  acquisition_started_ = true;
  status_ = remote_ready ? "GenTL streaming (control remoto activo)"
                         : (remote_status.empty() ? "GenTL streaming (sin control remoto)"
                                                  : "GenTL streaming (sin control remoto: " +
                                                        remote_status + ")");
  return true;
}

void GenTLBackend::stop() {
  stopRemoteAcquisition();

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
    status_ = remote_control_ && remote_control_->loaded
                  ? "Esperando buffers GenTL (control remoto activo)"
                  : (remote_status_.empty() ? "Esperando buffers GenTL"
                                            : "Esperando buffers GenTL (" + remote_status_ + ")");
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

bool GenTLBackend::supportsFeatures() const {
#ifdef HAVE_GENTL_GENAPI
  return dev_ != nullptr;
#else
  return false;
#endif
}

std::vector<FeatureInfo> GenTLBackend::listFeatures() {
  std::vector<FeatureInfo> features;
#ifndef HAVE_GENTL_GENAPI
  status_ = "GenApi remoto no compilado";
  return features;
#else
  if (!dev_) {
    status_ = "No hay dispositivo GenTL abierto";
    return features;
  }
  if (!ensureRemoteControl()) {
    status_ = remote_status_.empty() ? "No se pudo inicializar GenApi remoto" : remote_status_;
    return features;
  }

  try {
    GenApi::NodeList_t nodes;
    remote_control_->node_map_ptr->GetNodes(nodes);
    features.reserve(nodes.size());
    for (GenApi::INode* node : nodes) {
      if (!isUserFeatureNode(node)) {
        continue;
      }

      FeatureInfo info;
      fillFeatureInfo(*node, info);
      if (info.id.empty()) {
        continue;
      }
      features.push_back(std::move(info));
    }

    status_ = features.empty() ? "Sin features GenApi" : "Features GenApi cargadas";
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("Listado GenApi fallo: ") + e.what();
    features.clear();
  } catch (const std::exception& e) {
    status_ = std::string("Listado GenApi fallo: ") + e.what();
    features.clear();
  }
  return features;
#endif
}

bool GenTLBackend::getFeatureValue(const FeatureInfo& info, FeatureValue& out) {
#ifndef HAVE_GENTL_GENAPI
  (void)info;
  (void)out;
  status_ = "GenApi remoto no compilado";
  return false;
#else
  if (!dev_) {
    status_ = "No hay dispositivo GenTL abierto";
    return false;
  }
  if (!ensureRemoteControl()) {
    status_ = remote_status_.empty() ? "No se pudo inicializar GenApi remoto" : remote_status_;
    return false;
  }

  GenApi::INode* node = remote_control_->node_map_ptr->GetNode(info.id.c_str());
  if (!isUserFeatureNode(node)) {
    status_ = "Feature GenApi no disponible: " + info.id;
    return false;
  }

  try {
    const bool ok = readFeatureValueFromNode(*node, info.type, out);
    if (!ok) {
      status_ = "No se pudo leer feature GenApi: " + info.id;
    }
    return ok;
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("Lectura GenApi fallo: ") + e.what();
    return false;
  } catch (const std::exception& e) {
    status_ = std::string("Lectura GenApi fallo: ") + e.what();
    return false;
  }
#endif
}

bool GenTLBackend::setFeatureValue(const FeatureInfo& info, const FeatureValue& value) {
#ifndef HAVE_GENTL_GENAPI
  (void)info;
  (void)value;
  status_ = "GenApi remoto no compilado";
  return false;
#else
  if (!dev_) {
    status_ = "No hay dispositivo GenTL abierto";
    return false;
  }
  if (!ensureRemoteControl()) {
    status_ = remote_status_.empty() ? "No se pudo inicializar GenApi remoto" : remote_status_;
    return false;
  }

  GenApi::INode* node = remote_control_->node_map_ptr->GetNode(info.id.c_str());
  if (!isUserFeatureNode(node)) {
    status_ = "Feature GenApi no disponible: " + info.id;
    return false;
  }

  try {
    const bool ok = writeFeatureValueToNode(*node, info.type, value);
    if (ok) {
      status_ = "Feature GenApi actualizada";
    } else {
      status_ = "No se pudo escribir feature GenApi: " + info.id;
    }
    return ok;
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("Escritura GenApi fallo: ") + e.what();
    return false;
  } catch (const std::exception& e) {
    status_ = std::string("Escritura GenApi fallo: ") + e.what();
    return false;
  }
#endif
}

bool GenTLBackend::executeCommand(const FeatureInfo& info) {
#ifndef HAVE_GENTL_GENAPI
  (void)info;
  status_ = "GenApi remoto no compilado";
  return false;
#else
  if (!dev_) {
    status_ = "No hay dispositivo GenTL abierto";
    return false;
  }
  if (!ensureRemoteControl()) {
    status_ = remote_status_.empty() ? "No se pudo inicializar GenApi remoto" : remote_status_;
    return false;
  }

  GenApi::INode* node = remote_control_->node_map_ptr->GetNode(info.id.c_str());
  if (!isUserFeatureNode(node)) {
    status_ = "Feature GenApi no disponible: " + info.id;
    return false;
  }

  try {
    if (info.type != FeatureType::Command) {
      status_ = "La feature no es un comando: " + info.id;
      return false;
    }
    const bool ok = tryExecuteCommand(*remote_control_->node_map_ptr, info.id.c_str());
    if (ok) {
      status_ = "Comando GenApi ejecutado";
    } else {
      status_ = "No se pudo ejecutar comando GenApi: " + info.id;
    }
    return ok;
  } catch (const GenICam::GenericException& e) {
    status_ = std::string("Comando GenApi fallo: ") + e.what();
    return false;
  } catch (const std::exception& e) {
    status_ = std::string("Comando GenApi fallo: ") + e.what();
    return false;
  }
#endif
}
