#include "camera/AravisBackend.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>

#ifdef HAVE_ARAVIS
namespace {
constexpr ArvPixelFormat kInvalidPixelFormat = static_cast<ArvPixelFormat>(0);

bool mapPixelFormat(ArvPixelFormat fmt, PixelFormat& out_format, int& channels) {
  switch (fmt) {
    case ARV_PIXEL_FORMAT_MONO_8:
      out_format = PixelFormat::Mono8;
      channels = 1;
      return true;
    case ARV_PIXEL_FORMAT_RGB_8_PACKED:
      out_format = PixelFormat::RGB8;
      channels = 3;
      return true;
    case ARV_PIXEL_FORMAT_BGR_8_PACKED:
      out_format = PixelFormat::BGR8;
      channels = 3;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_RG_8:
      out_format = PixelFormat::BayerRG8;
      channels = 1;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_BG_8:
      out_format = PixelFormat::BayerBG8;
      channels = 1;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GB_8:
      out_format = PixelFormat::BayerGB8;
      channels = 1;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GR_8:
      out_format = PixelFormat::BayerGR8;
      channels = 1;
      return true;
    default:
      out_format = PixelFormat::Unknown;
      channels = 0;
      return false;
  }
}

bool mapHighBitDepthPixelFormat(ArvPixelFormat fmt, PixelFormat& out_format, int& bits_per_pixel) {
  switch (fmt) {
    case ARV_PIXEL_FORMAT_MONO_10:
      out_format = PixelFormat::Mono8;
      bits_per_pixel = 10;
      return true;
    case ARV_PIXEL_FORMAT_MONO_12:
      out_format = PixelFormat::Mono8;
      bits_per_pixel = 12;
      return true;
    case ARV_PIXEL_FORMAT_MONO_14:
      out_format = PixelFormat::Mono8;
      bits_per_pixel = 14;
      return true;
    case ARV_PIXEL_FORMAT_MONO_16:
      out_format = PixelFormat::Mono8;
      bits_per_pixel = 16;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_RG_10:
      out_format = PixelFormat::BayerRG8;
      bits_per_pixel = 10;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_BG_10:
      out_format = PixelFormat::BayerBG8;
      bits_per_pixel = 10;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GB_10:
      out_format = PixelFormat::BayerGB8;
      bits_per_pixel = 10;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GR_10:
      out_format = PixelFormat::BayerGR8;
      bits_per_pixel = 10;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_RG_12:
      out_format = PixelFormat::BayerRG8;
      bits_per_pixel = 12;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_BG_12:
      out_format = PixelFormat::BayerBG8;
      bits_per_pixel = 12;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GB_12:
      out_format = PixelFormat::BayerGB8;
      bits_per_pixel = 12;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GR_12:
      out_format = PixelFormat::BayerGR8;
      bits_per_pixel = 12;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_RG_16:
      out_format = PixelFormat::BayerRG8;
      bits_per_pixel = 16;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_BG_16:
      out_format = PixelFormat::BayerBG8;
      bits_per_pixel = 16;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GB_16:
      out_format = PixelFormat::BayerGB8;
      bits_per_pixel = 16;
      return true;
    case ARV_PIXEL_FORMAT_BAYER_GR_16:
      out_format = PixelFormat::BayerGR8;
      bits_per_pixel = 16;
      return true;
    default:
      out_format = PixelFormat::Unknown;
      bits_per_pixel = 0;
      return false;
  }
}

bool convertHighBitDepthTo8Bit(const uint8_t* src, size_t src_size, int width, int height,
                               int bits_per_pixel, PixelFormat out_format, Frame& out) {
  if (!src || width <= 0 || height <= 0 || bits_per_pixel <= 8) return false;
  const size_t w = static_cast<size_t>(width);
  const size_t h = static_cast<size_t>(height);
  if (w > std::numeric_limits<size_t>::max() / h) return false;
  const size_t pixel_count = w * h;
  if (pixel_count > std::numeric_limits<size_t>::max() / 2) return false;
  const size_t required_bytes = pixel_count * 2;
  if (src_size < required_bytes) return false;

  auto converted = makeFrameBuffer(pixel_count);
  const int shift = bits_per_pixel - 8;
  for (size_t i = 0; i < pixel_count; ++i) {
    const size_t src_i = i * 2;
    uint16_t sample = static_cast<uint16_t>(src[src_i]) |
                      (static_cast<uint16_t>(src[src_i + 1]) << 8);
    sample = static_cast<uint16_t>(sample >> shift);
    if (sample > 255) sample = 255;
    converted.get()[i] = static_cast<uint8_t>(sample);
  }

  out.data = std::move(converted);
  out.data_size = pixel_count;
  out.width = width;
  out.height = height;
  out.stride = width;
  out.format = out_format;
  return true;
}

const char* bufferStatusToString(ArvBufferStatus status) {
  switch (status) {
    case ARV_BUFFER_STATUS_SUCCESS:
      return "success";
    case ARV_BUFFER_STATUS_CLEARED:
      return "cleared";
    case ARV_BUFFER_STATUS_TIMEOUT:
      return "timeout";
    case ARV_BUFFER_STATUS_MISSING_PACKETS:
      return "missing_packets";
    case ARV_BUFFER_STATUS_WRONG_PACKET_ID:
      return "wrong_packet_id";
    case ARV_BUFFER_STATUS_SIZE_MISMATCH:
      return "size_mismatch";
    case ARV_BUFFER_STATUS_FILLING:
      return "filling";
    case ARV_BUFFER_STATUS_ABORTED:
      return "aborted";
    case ARV_BUFFER_STATUS_UNKNOWN:
    default:
      return "unknown";
  }
}

void clearError(GError*& err) {
  if (err) {
    g_error_free(err);
    err = nullptr;
  }
}

bool cameraFeatureAvailable(ArvCamera* camera, const char* feature) {
  if (!camera || !feature || !*feature) return false;
  GError* err = nullptr;
  const gboolean available = arv_camera_is_feature_available(camera, feature, &err);
  clearError(err);
  return available == TRUE;
}

void setCameraStringFeatureIfAvailable(ArvCamera* camera, const char* feature,
                                       const char* value) {
  if (!camera || !feature || !value) return;
  if (!cameraFeatureAvailable(camera, feature)) return;
  GError* err = nullptr;
  arv_camera_set_string(camera, feature, value, &err);
  clearError(err);
}

void setCameraBooleanFeatureIfAvailable(ArvCamera* camera, const char* feature, bool value) {
  if (!camera || !feature) return;
  if (!cameraFeatureAvailable(camera, feature)) return;
  GError* err = nullptr;
  arv_camera_set_boolean(camera, feature, value ? TRUE : FALSE, &err);
  clearError(err);
}

std::string cameraPixelFormatString(ArvCamera* camera) {
  if (!camera) return {};
  GError* err = nullptr;
  const char* fmt = arv_camera_get_pixel_format_as_string(camera, &err);
  if (err) {
    clearError(err);
    return {};
  }
  return fmt ? std::string(fmt) : std::string{};
}
}  // namespace
#endif

std::vector<CameraInfo> AravisBackend::listDevices() {
  std::vector<CameraInfo> devices;
#ifdef HAVE_ARAVIS
  arv_update_device_list();
  int n = arv_get_n_devices();
  devices.reserve(n);
  for (int i = 0; i < n; ++i) {
    const char* id = arv_get_device_id(i);
    if (!id) continue;
    CameraInfo info;
    info.id = id;
    info.label = id;
    devices.push_back(info);
  }
  status_ = devices.empty() ? "No devices" : "Devices detected";
#else
  status_ = "Aravis no disponible";
#endif
  return devices;
}

bool AravisBackend::open(const CameraInfo& info) {
#ifdef HAVE_ARAVIS
  close();
  GError* err = nullptr;
  camera_ = arv_camera_new(info.id.c_str(), &err);
  if (err) {
    status_ = err->message ? err->message : "No se pudo abrir la camara";
    g_error_free(err);
    return false;
  }
  if (!camera_) {
    status_ = "No se pudo abrir la camara";
    return false;
  }

  if (arv_camera_is_gv_device(camera_)) {
    // Packet socket can fail silently on some systems if CAP_NET_RAW is unavailable.
    arv_camera_gv_set_stream_options(camera_, ARV_GV_STREAM_OPTION_PACKET_SOCKET_DISABLED);
  }

  device_ = arv_camera_get_device(camera_);
  genicam_ = device_ ? arv_device_get_genicam(device_) : nullptr;
  status_ = "Camara abierta";
  return true;
#else
  (void)info;
  status_ = "Aravis no disponible";
  return false;
#endif
}

void AravisBackend::close() {
#ifdef HAVE_ARAVIS
  stop();
  if (stream_) {
    g_object_unref(stream_);
    stream_ = nullptr;
  }
  if (camera_) {
    g_object_unref(camera_);
    camera_ = nullptr;
  }
  device_ = nullptr;
  genicam_ = nullptr;
  feature_nodes_.clear();
#endif
}

bool AravisBackend::start() {
#ifdef HAVE_ARAVIS
  if (!camera_) return false;

  GError* err = nullptr;
  setCameraStringFeatureIfAvailable(camera_, "AcquisitionMode", "Continuous");
  setCameraStringFeatureIfAvailable(camera_, "TriggerSelector", "FrameStart");
  setCameraStringFeatureIfAvailable(camera_, "TriggerMode", "Off");
  setCameraStringFeatureIfAvailable(camera_, "TriggerSelector", "AcquisitionStart");
  setCameraStringFeatureIfAvailable(camera_, "TriggerMode", "Off");
  setCameraStringFeatureIfAvailable(camera_, "TriggerSelector", "FrameBurstStart");
  setCameraStringFeatureIfAvailable(camera_, "TriggerMode", "Off");

  arv_camera_set_acquisition_mode(camera_, ARV_ACQUISITION_MODE_CONTINUOUS, &err);
  if (err) {
    clearError(err);
  }
  arv_camera_clear_triggers(camera_, &err);
  if (err) {
    clearError(err);
  }

  if (!stream_) {
    stream_ = arv_camera_create_stream(camera_, nullptr, nullptr, &err);
    if (err) {
      status_ = err->message ? err->message : "No se pudo crear stream";
      g_error_free(err);
      return false;
    }
    if (!stream_) return false;
    payload_ = static_cast<size_t>(arv_camera_get_payload(camera_, &err));
    if (err) {
      status_ = err->message ? err->message : "No se pudo obtener payload";
      g_error_free(err);
      return false;
    }
    if (payload_ == 0) {
      status_ = "Payload invalido";
      return false;
    }
    for (int i = 0; i < 16; ++i) {
      arv_stream_push_buffer(stream_, arv_buffer_new(payload_, nullptr));
    }
  }

  arv_camera_start_acquisition(camera_, &err);
  if (err) {
    status_ = err->message ? err->message : "No se pudo iniciar adquisicion";
    clearError(err);
    return false;
  }
  const std::string fmt_name = cameraPixelFormatString(camera_);
  if (!fmt_name.empty()) {
    status_ = "Streaming (" + fmt_name + ")";
  } else {
    status_ = "Streaming";
  }
  return true;
#else
  return false;
#endif
}

void AravisBackend::stop() {
#ifdef HAVE_ARAVIS
  if (camera_) {
    arv_camera_stop_acquisition(camera_, nullptr);
  }
#endif
}

bool AravisBackend::grab(Frame& out) {
#ifdef HAVE_ARAVIS
  if (!stream_) return false;
  ArvBuffer* buffer = arv_stream_timeout_pop_buffer(stream_, 50000);
  if (!buffer) return false;

  bool ok = false;
  const ArvBufferStatus buffer_status = arv_buffer_get_status(buffer);
  if (buffer_status == ARV_BUFFER_STATUS_SUCCESS) {
    size_t size = 0;
    const uint8_t* data = static_cast<const uint8_t*>(arv_buffer_get_data(buffer, &size));
    int width = arv_buffer_get_image_width(buffer);
    int height = arv_buffer_get_image_height(buffer);
    const ArvPixelFormat fmt = arv_buffer_get_image_pixel_format(buffer);

    if (!data || width <= 0 || height <= 0 || size == 0) {
      status_ = "Buffer de imagen vacio";
    } else {
      PixelFormat mapped = PixelFormat::Unknown;
      int channels = 0;
      if (mapPixelFormat(fmt, mapped, channels)) {
        const size_t w = static_cast<size_t>(width);
        const size_t h = static_cast<size_t>(height);
        if (w > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels)) {
          status_ = "Overflow en stride de imagen";
        } else {
          const size_t stride = w * static_cast<size_t>(channels);
          if (h > std::numeric_limits<size_t>::max() / stride) {
            status_ = "Overflow en tamano de imagen";
          } else {
            const size_t required_bytes = stride * h;
            if (size < required_bytes) {
              status_ = "Buffer demasiado pequeno para imagen";
            } else {
              auto buffer_copy = makeFrameBuffer(required_bytes);
              std::memcpy(buffer_copy.get(), data, required_bytes);
              out.data = std::move(buffer_copy);
              out.data_size = required_bytes;
              out.width = width;
              out.height = height;
              out.stride = static_cast<int>(stride);
              out.format = mapped;
              ok = true;
            }
          }
        }
      } else {
        PixelFormat converted_format = PixelFormat::Unknown;
        int bits_per_pixel = 0;
        if (mapHighBitDepthPixelFormat(fmt, converted_format, bits_per_pixel) &&
            convertHighBitDepthTo8Bit(data, size, width, height, bits_per_pixel,
                                      converted_format, out)) {
          ok = true;
        } else {
          char fmt_text[32];
          std::snprintf(fmt_text, sizeof(fmt_text), "0x%08x",
                        static_cast<unsigned int>(fmt));
          status_ = std::string("PixelFormat no soportado: ") + fmt_text +
                    " (cambialo en Device properties)";
        }
      }

      if (ok) {
        out.timestamp_ns = static_cast<uint64_t>(arv_buffer_get_timestamp(buffer));
        if (out.timestamp_ns == 0) {
          out.timestamp_ns = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count());
        }
        out.seq = static_cast<uint64_t>(arv_buffer_get_frame_id(buffer));
        status_ = "Streaming";
      }
    }
  } else {
    status_ = std::string("Buffer status: ") + bufferStatusToString(buffer_status);
  }

  arv_stream_push_buffer(stream_, buffer);
  return ok;
#else
  (void)out;
  return false;
#endif
}

bool AravisBackend::isOpen() const {
#ifdef HAVE_ARAVIS
  return camera_ != nullptr;
#else
  return false;
#endif
}

std::vector<FeatureInfo> AravisBackend::listFeatures() {
  std::vector<FeatureInfo> features;
#ifdef HAVE_ARAVIS
  feature_nodes_.clear();
  if (!genicam_) return features;
  ArvDomNode* root = ARV_DOM_NODE(arv_dom_document_get_document_element(ARV_DOM_DOCUMENT(genicam_)));
  if (!root) {
    status_ = "GenICam sin nodo raiz";
    return features;
  }

  std::unordered_set<ArvDomNode*> visited;
  std::unordered_set<std::string> seen_ids;
  collectFeatures(root, features, visited, seen_ids);
  status_ = features.empty() ? "Sin features" : "Features cargadas";
#else
  status_ = "Aravis no disponible";
#endif
  return features;
}

bool AravisBackend::getFeatureValue(const FeatureInfo& info, FeatureValue& out) {
#ifdef HAVE_ARAVIS
  ArvGcFeatureNode* node = findFeatureNode(info.id);
  if (!node) return false;
  if (info.type == FeatureType::Integer) {
    if (!ARV_IS_GC_INTEGER(node)) return false;
    GError* err = nullptr;
    gint64 value = arv_gc_integer_get_value(ARV_GC_INTEGER(node), &err);
    if (err) {
      g_error_free(err);
      return false;
    }
    out.value = static_cast<int64_t>(value);
    return true;
  }
  if (info.type == FeatureType::Float) {
    if (!ARV_IS_GC_FLOAT(node)) return false;
    GError* err = nullptr;
    double value = arv_gc_float_get_value(ARV_GC_FLOAT(node), &err);
    if (err) {
      g_error_free(err);
      return false;
    }
    out.value = value;
    return true;
  }
  if (info.type == FeatureType::Boolean || info.type == FeatureType::Enumeration || info.type == FeatureType::String) {
    if (!ARV_IS_GC_FEATURE_NODE(node)) return false;
    GError* err = nullptr;
    const char* value = arv_gc_feature_node_get_value_as_string(node, &err);
    if (err) {
      g_error_free(err);
      return false;
    }
    std::string result = value ? value : "";
    if (info.type == FeatureType::Boolean) {
      out.value = (result == "True" || result == "true" || result == "1");
    } else {
      out.value = result;
    }
    return true;
  }
#endif
  return false;
}

bool AravisBackend::setFeatureValue(const FeatureInfo& info, const FeatureValue& value) {
#ifdef HAVE_ARAVIS
  ArvGcFeatureNode* node = findFeatureNode(info.id);
  if (!node) return false;
  if (info.type == FeatureType::Integer) {
    if (!ARV_IS_GC_INTEGER(node)) return false;
    if (auto v = std::get_if<int64_t>(&value.value)) {
      arv_gc_integer_set_value(ARV_GC_INTEGER(node), static_cast<gint64>(*v), nullptr);
      return true;
    }
  }
  if (info.type == FeatureType::Float) {
    if (!ARV_IS_GC_FLOAT(node)) return false;
    if (auto v = std::get_if<double>(&value.value)) {
      arv_gc_float_set_value(ARV_GC_FLOAT(node), *v, nullptr);
      return true;
    }
  }
  if (info.type == FeatureType::Boolean) {
    if (!ARV_IS_GC_FEATURE_NODE(node)) return false;
    if (auto v = std::get_if<bool>(&value.value)) {
      arv_gc_feature_node_set_value_from_string(node, *v ? "True" : "False", nullptr);
      return true;
    }
  }
  if (info.type == FeatureType::Enumeration || info.type == FeatureType::String) {
    if (!ARV_IS_GC_FEATURE_NODE(node)) return false;
    if (auto v = std::get_if<std::string>(&value.value)) {
      arv_gc_feature_node_set_value_from_string(node, v->c_str(), nullptr);
      return true;
    }
  }
#endif
  return false;
}

bool AravisBackend::executeCommand(const FeatureInfo& info) {
#ifdef HAVE_ARAVIS
  ArvGcFeatureNode* node = findFeatureNode(info.id);
  if (!node) return false;
  if (info.type == FeatureType::Command) {
    arv_gc_command_execute(ARV_GC_COMMAND(node), nullptr);
    return true;
  }
#endif
  return false;
}

#ifdef HAVE_ARAVIS
void AravisBackend::collectFeatures(ArvDomNode* node, std::vector<FeatureInfo>& out,
                                    std::unordered_set<ArvDomNode*>& visited,
                                    std::unordered_set<std::string>& seen_ids) {
  if (!node) return;
  if (!visited.insert(node).second) return;

  if (ARV_IS_GC_FEATURE_NODE(node) && !ARV_IS_GC_ENUM_ENTRY(node)) {
    auto* feat = ARV_GC_FEATURE_NODE(node);
    const char* name = arv_gc_feature_node_get_name(feat);
    if (name && *name) {
      // Internal GenICam helper nodes are not user-facing controls.
      if (std::strncmp(name, "d_", 2) != 0) {
        GError* err = nullptr;
        const gboolean implemented = arv_gc_feature_node_is_implemented(feat, &err);
        if (err) {
          g_error_free(err);
          err = nullptr;
        }
        const gboolean available = arv_gc_feature_node_is_available(feat, &err);
        if (err) {
          g_error_free(err);
          err = nullptr;
        }

        if (implemented && available && seen_ids.insert(name).second) {
          FeatureInfo info;
          info.id = name;
          const char* display = arv_gc_feature_node_get_display_name(feat);
          info.display_name = (display && *display) ? display : info.id;

          const ArvGcAccessMode access = arv_gc_feature_node_get_actual_access_mode(feat);
          info.readable = (access == ARV_GC_ACCESS_MODE_RO || access == ARV_GC_ACCESS_MODE_RW);
          info.writable = (access == ARV_GC_ACCESS_MODE_WO || access == ARV_GC_ACCESS_MODE_RW);

          if (ARV_IS_GC_ENUMERATION(node)) {
            info.type = FeatureType::Enumeration;
            const GSList* entries = arv_gc_enumeration_get_entries(ARV_GC_ENUMERATION(node));
            for (const GSList* it = entries; it; it = it->next) {
              if (!ARV_IS_GC_ENUM_ENTRY(it->data)) continue;
              const char* enum_name = arv_gc_feature_node_get_name(ARV_GC_FEATURE_NODE(it->data));
              if (enum_name && *enum_name) {
                info.enum_entries.emplace_back(enum_name);
              }
            }
          } else if (ARV_IS_GC_INTEGER(node)) {
            info.type = FeatureType::Integer;
            info.min = static_cast<double>(arv_gc_integer_get_min(ARV_GC_INTEGER(node), nullptr));
            info.max = static_cast<double>(arv_gc_integer_get_max(ARV_GC_INTEGER(node), nullptr));
          } else if (ARV_IS_GC_FLOAT(node)) {
            info.type = FeatureType::Float;
            info.min = arv_gc_float_get_min(ARV_GC_FLOAT(node), nullptr);
            info.max = arv_gc_float_get_max(ARV_GC_FLOAT(node), nullptr);
          } else if (ARV_IS_GC_BOOLEAN(node)) {
            info.type = FeatureType::Boolean;
          } else if (ARV_IS_GC_STRING(node)) {
            info.type = FeatureType::String;
          } else if (ARV_IS_GC_COMMAND(node)) {
            info.type = FeatureType::Command;
          } else if (ARV_IS_GC_CATEGORY(node)) {
            info.type = FeatureType::Category;
          } else {
            info.type = FeatureType::Unknown;
          }

          feature_nodes_[info.id] = feat;
          out.push_back(std::move(info));
        }
      }
    }
  }

  ArvDomNodeList* children = arv_dom_node_get_child_nodes(node);
  if (!children) return;
  const int len = arv_dom_node_list_get_length(children);
  for (int i = 0; i < len; ++i) {
    ArvDomNode* child = arv_dom_node_list_get_item(children, i);
    collectFeatures(child, out, visited, seen_ids);
  }
}

ArvGcFeatureNode* AravisBackend::findFeatureNode(const std::string& id) const {
  auto it = feature_nodes_.find(id);
  if (it == feature_nodes_.end()) return nullptr;
  return it->second;
}
#endif
