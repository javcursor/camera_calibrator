#include "ui/ImGuiApp.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl2.h>
#include <imgui_stdlib.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

#ifdef HAVE_OPENCV_ARUCO
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#endif

namespace fs = std::filesystem;

namespace {
constexpr const char* kOpenCVGStreamerDeviceIdPrefix = "__opencv_gstreamer_pipeline__";

static bool frameToBGR(const Frame& frame, cv::Mat& bgr, double scale = 1.0, bool bayer_swap_rb = false) {
  if (frame.empty()) return false;
  if (!std::isfinite(scale) || scale <= 0.0) return false;
  const bool resize_needed = scale < 0.999;
  cv::Mat wrapped;
  switch (frame.format) {
    case PixelFormat::Mono8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data.get(), frame.stride);
      if (resize_needed) {
        cv::Mat mono_small;
        cv::resize(wrapped, mono_small, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::cvtColor(mono_small, bgr, cv::COLOR_GRAY2BGR);
      } else {
        cv::cvtColor(wrapped, bgr, cv::COLOR_GRAY2BGR);
      }
      return true;
    }
    case PixelFormat::BGR8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC3, frame.data.get(), frame.stride);
      if (resize_needed) {
        cv::resize(wrapped, bgr, cv::Size(), scale, scale, cv::INTER_AREA);
      } else {
        bgr = wrapped.clone();
      }
      return true;
    }
    case PixelFormat::RGB8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC3, frame.data.get(), frame.stride);
      if (resize_needed) {
        cv::Mat rgb_small;
        cv::resize(wrapped, rgb_small, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::cvtColor(rgb_small, bgr, cv::COLOR_RGB2BGR);
      } else {
        cv::cvtColor(wrapped, bgr, cv::COLOR_RGB2BGR);
      }
      return true;
    }
    case PixelFormat::BayerRG8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data.get(), frame.stride);
      const int code = bayer_swap_rb ? cv::COLOR_BayerBG2BGR : cv::COLOR_BayerRG2BGR;
      if (resize_needed) {
        cv::Mat bgr_full;
        cv::cvtColor(wrapped, bgr_full, code);
        cv::resize(bgr_full, bgr, cv::Size(), scale, scale, cv::INTER_AREA);
      } else {
        cv::cvtColor(wrapped, bgr, code);
      }
      return true;
    }
    case PixelFormat::BayerBG8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data.get(), frame.stride);
      const int code = bayer_swap_rb ? cv::COLOR_BayerRG2BGR : cv::COLOR_BayerBG2BGR;
      if (resize_needed) {
        cv::Mat bgr_full;
        cv::cvtColor(wrapped, bgr_full, code);
        cv::resize(bgr_full, bgr, cv::Size(), scale, scale, cv::INTER_AREA);
      } else {
        cv::cvtColor(wrapped, bgr, code);
      }
      return true;
    }
    case PixelFormat::BayerGB8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data.get(), frame.stride);
      const int code = bayer_swap_rb ? cv::COLOR_BayerGR2BGR : cv::COLOR_BayerGB2BGR;
      if (resize_needed) {
        cv::Mat bgr_full;
        cv::cvtColor(wrapped, bgr_full, code);
        cv::resize(bgr_full, bgr, cv::Size(), scale, scale, cv::INTER_AREA);
      } else {
        cv::cvtColor(wrapped, bgr, code);
      }
      return true;
    }
    case PixelFormat::BayerGR8: {
      wrapped = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data.get(), frame.stride);
      const int code = bayer_swap_rb ? cv::COLOR_BayerGB2BGR : cv::COLOR_BayerGR2BGR;
      if (resize_needed) {
        cv::Mat bgr_full;
        cv::cvtColor(wrapped, bgr_full, code);
        cv::resize(bgr_full, bgr, cv::Size(), scale, scale, cv::INTER_AREA);
      } else {
        cv::cvtColor(wrapped, bgr, code);
      }
      return true;
    }
    default:
      return false;
  }
}

static void drawDetection(cv::Mat& image, const PatternConfig& config, const DetectionResult& det) {
  if (!det.found) return;
  if (config.type == PatternType::Charuco) {
#ifdef HAVE_OPENCV_ARUCO
    cv::Mat corners_mat(det.corners, true);
    cv::Mat ids_mat(det.ids, true);
    cv::aruco::drawDetectedCornersCharuco(image, corners_mat, ids_mat);
#else
    for (const auto& p : det.corners) {
      cv::circle(image, p, 4, cv::Scalar(0, 255, 0), -1);
    }
#endif
  } else {
    cv::drawChessboardCorners(image, config.board_size, det.corners, det.found);
  }
}

enum class UiIcon {
  Link,
  Gear,
  Theme,
  Folder,
};

static void drawLinkIcon(ImDrawList* draw_list, ImVec2 top_left, float size, ImU32 color) {
  const float r = size * 0.24f;
  const ImVec2 c1(top_left.x + size * 0.38f, top_left.y + size * 0.50f);
  const ImVec2 c2(top_left.x + size * 0.62f, top_left.y + size * 0.50f);
  draw_list->AddCircle(c1, r, color, 16, 2.0f);
  draw_list->AddCircle(c2, r, color, 16, 2.0f);
  draw_list->AddLine(ImVec2(c1.x + r * 0.65f, c1.y), ImVec2(c2.x - r * 0.65f, c2.y), color, 2.0f);
}

static void drawGearIcon(ImDrawList* draw_list, ImVec2 top_left, float size, ImU32 color) {
  const ImVec2 center(top_left.x + size * 0.5f, top_left.y + size * 0.5f);
  const float outer_r = size * 0.28f;
  const float inner_r = size * 0.12f;
  for (int i = 0; i < 8; ++i) {
    const float a = static_cast<float>(i) * 0.785398163f;  // 2*pi/8
    const float x0 = center.x + std::cos(a) * (outer_r + 1.0f);
    const float y0 = center.y + std::sin(a) * (outer_r + 1.0f);
    const float x1 = center.x + std::cos(a) * (outer_r + 4.0f);
    const float y1 = center.y + std::sin(a) * (outer_r + 4.0f);
    draw_list->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), color, 1.5f);
  }
  draw_list->AddCircle(center, outer_r, color, 24, 1.8f);
  draw_list->AddCircle(center, inner_r, color, 20, 1.8f);
}

static void drawThemeIcon(ImDrawList* draw_list, ImVec2 top_left, float size, ImU32 color) {
  const ImVec2 center(top_left.x + size * 0.5f, top_left.y + size * 0.5f);
  const float inner_r = size * 0.18f;
  draw_list->AddCircle(center, inner_r, color, 20, 1.8f);
  for (int i = 0; i < 8; ++i) {
    const float a = static_cast<float>(i) * 0.785398163f;  // 2*pi/8
    const float x0 = center.x + std::cos(a) * (size * 0.28f);
    const float y0 = center.y + std::sin(a) * (size * 0.28f);
    const float x1 = center.x + std::cos(a) * (size * 0.40f);
    const float y1 = center.y + std::sin(a) * (size * 0.40f);
    draw_list->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), color, 1.5f);
  }
}

static void drawFolderIcon(ImDrawList* draw_list, ImVec2 top_left, float size, ImU32 color) {
  const float x = top_left.x + size * 0.13f;
  const float y = top_left.y + size * 0.34f;
  const float w = size * 0.74f;
  const float h = size * 0.42f;
  const float tab_h = size * 0.14f;
  const float tab_start = x + size * 0.10f;
  const float tab_end = x + size * 0.34f;
  const std::array<ImVec2, 8> outline = {
      ImVec2(x, y + h),
      ImVec2(x, y),
      ImVec2(tab_start, y),
      ImVec2(tab_start + size * 0.06f, y - tab_h),
      ImVec2(tab_end, y - tab_h),
      ImVec2(tab_end + size * 0.08f, y),
      ImVec2(x + w, y),
      ImVec2(x + w, y + h),
  };
  draw_list->AddPolyline(outline.data(), static_cast<int>(outline.size()), color, ImDrawFlags_Closed, 1.8f);
}

static ImU32 defaultIconColor(float alpha = 1.0f) {
  ImVec4 text_color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
  text_color.w *= alpha;
  return ImGui::GetColorU32(text_color);
}

static bool iconButton(const char* id, UiIcon icon, float size, ImU32 color) {
  if (ImGui::InvisibleButton(id, ImVec2(size, size))) {
    return true;
  }
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ImVec2 min = ImGui::GetItemRectMin();
  if (icon == UiIcon::Link) {
    drawLinkIcon(draw_list, min, size, color);
  } else if (icon == UiIcon::Folder) {
    drawFolderIcon(draw_list, min, size, color);
  } else if (icon == UiIcon::Theme) {
    drawThemeIcon(draw_list, min, size, color);
  } else {
    drawGearIcon(draw_list, min, size, color);
  }
  return false;
}

static ImVec4 progressColor(float value) {
  if (value < 0.35f) return ImVec4(0.85f, 0.10f, 0.10f, 1.0f);
  if (value < 0.70f) return ImVec4(0.95f, 0.50f, 0.05f, 1.0f);
  return ImVec4(0.16f, 0.65f, 0.24f, 1.0f);
}

static void drawCoverageInlineRow(const CalibrationProgress& progress) {
  const std::array<const char*, 4> labels = {
      "Coverage X",
      "Coverage Y",
      "Coverage Size",
      "Coverage Skew",
  };
  const std::array<float, 4> values = {
      progress.x,
      progress.y,
      progress.scale,
      progress.skew,
  };

  const float avail_w = ImGui::GetContentRegionAvail().x;
  const float base_x = ImGui::GetCursorPosX();
  const float side_margin = 18.0f;
  const float prefix_gap = 10.0f;
  const float label_gap = 6.0f;
  const float sep_gap = 8.0f;
  const float min_bar_w = 48.0f;

  const float prefix_w = ImGui::CalcTextSize("Coverage:").x;
  const float sep_w = ImGui::CalcTextSize("|").x;
  float label_sum_w = 0.0f;
  for (const char* label : labels) {
    label_sum_w += ImGui::CalcTextSize(label).x;
  }

  const float fixed_w =
      prefix_w + prefix_gap +
      label_sum_w + static_cast<float>(labels.size()) * label_gap +
      static_cast<float>(labels.size() - 1) * (sep_w + 2.0f * sep_gap);

  float bar_w = (avail_w - 2.0f * side_margin - fixed_w) / static_cast<float>(labels.size());
  bar_w = std::max(min_bar_w, bar_w);

  const float row_w = fixed_w + static_cast<float>(labels.size()) * bar_w;
  float start_x = base_x + std::max(0.0f, (avail_w - row_w) * 0.5f);
  if (avail_w > row_w + 2.0f * side_margin) {
    start_x = std::max(start_x, base_x + side_margin);
  }
  ImGui::SetCursorPosX(start_x);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Coverage:");
  ImGui::SameLine(0.0f, prefix_gap);

  for (size_t i = 0; i < labels.size(); ++i) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(labels[i]);
    ImGui::SameLine(0.0f, label_gap);

    const float clamped = std::clamp(values[i], 0.0f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, progressColor(clamped));
    ImGui::ProgressBar(clamped, ImVec2(bar_w, 0.0f), "");
    ImGui::PopStyleColor();

    if (i + 1 < labels.size()) {
      ImGui::SameLine(0.0f, sep_gap);
      ImGui::TextUnformatted("|");
      ImGui::SameLine(0.0f, sep_gap);
    }
  }
}

static void itemTooltip(const char* text) {
  if (!text || !*text) return;
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip("%s", text);
  }
}

static bool comboWithTooltips(const char* label, int* current_index,
                              const char* const* items, int items_count,
                              const char* const* item_tooltips = nullptr) {
  if (!current_index || !items || items_count <= 0) {
    return false;
  }

  int safe_index = std::clamp(*current_index, 0, items_count - 1);
  bool changed = false;
  const char* preview = items[safe_index] ? items[safe_index] : "";

  if (ImGui::BeginCombo(label, preview)) {
    for (int i = 0; i < items_count; ++i) {
      const bool selected = (i == safe_index);
      if (ImGui::Selectable(items[i] ? items[i] : "", selected)) {
        *current_index = i;
        safe_index = i;
        changed = true;
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
      if (item_tooltips && item_tooltips[i] && *item_tooltips[i]) {
        itemTooltip(item_tooltips[i]);
      }
    }
    ImGui::EndCombo();
  }

  return changed;
}

#ifdef HAVE_OPENCV_ARUCO
struct ArucoDictionaryOption {
  int value;
  const char* label;
  const char* tooltip;
};

static const std::array<ArucoDictionaryOption, 17>& arucoDictionaryOptions() {
  static const std::array<ArucoDictionaryOption, 17> kOptions = {{
      {cv::aruco::DICT_4X4_50, "4x4 / 50", "Small 4x4 marker family with 50 unique ids."},
      {cv::aruco::DICT_4X4_100, "4x4 / 100", "Small 4x4 marker family with 100 unique ids."},
      {cv::aruco::DICT_4X4_250, "4x4 / 250", "Small 4x4 marker family with 250 unique ids."},
      {cv::aruco::DICT_4X4_1000, "4x4 / 1000", "Small 4x4 marker family with 1000 unique ids."},
      {cv::aruco::DICT_5X5_50, "5x5 / 50", "5x5 marker family with 50 unique ids."},
      {cv::aruco::DICT_5X5_100, "5x5 / 100", "5x5 marker family with 100 unique ids."},
      {cv::aruco::DICT_5X5_250, "5x5 / 250", "5x5 marker family with 250 unique ids."},
      {cv::aruco::DICT_5X5_1000, "5x5 / 1000", "5x5 marker family with 1000 unique ids."},
      {cv::aruco::DICT_6X6_50, "6x6 / 50", "6x6 marker family with 50 unique ids."},
      {cv::aruco::DICT_6X6_100, "6x6 / 100", "6x6 marker family with 100 unique ids."},
      {cv::aruco::DICT_6X6_250, "6x6 / 250", "6x6 marker family with 250 unique ids."},
      {cv::aruco::DICT_6X6_1000, "6x6 / 1000", "6x6 marker family with 1000 unique ids."},
      {cv::aruco::DICT_7X7_50, "7x7 / 50", "7x7 marker family with 50 unique ids."},
      {cv::aruco::DICT_7X7_100, "7x7 / 100", "7x7 marker family with 100 unique ids."},
      {cv::aruco::DICT_7X7_250, "7x7 / 250", "7x7 marker family with 250 unique ids."},
      {cv::aruco::DICT_7X7_1000, "7x7 / 1000", "7x7 marker family with 1000 unique ids."},
      {cv::aruco::DICT_ARUCO_ORIGINAL, "ArUco original", "Legacy ArUco dictionary used by older board generators."},
  }};
  return kOptions;
}
#endif

static std::string trimWhitespace(std::string value) {
  auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                                          [&](unsigned char c) { return !is_ws(c); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](unsigned char c) { return !is_ws(c); })
                  .base(),
              value.end());
  return value;
}

#if defined(__linux__) || defined(__APPLE__)
struct ShellCommandResult {
  bool started = false;
  int exit_status = -1;
  std::string output;
};

static bool commandExistsInPath(const char* name) {
  if (!name || !*name) return false;
  const char* path_env = std::getenv("PATH");
  if (!path_env || !*path_env) return false;

  std::stringstream path_stream(path_env);
  std::string dir;
  while (std::getline(path_stream, dir, ':')) {
    fs::path candidate = fs::path(dir.empty() ? "." : dir) / name;
    if (fs::exists(candidate) && access(candidate.c_str(), X_OK) == 0) {
      return true;
    }
  }
  return false;
}

static std::string shellQuote(const std::string& value) {
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

static ShellCommandResult runShellCommand(const std::string& command) {
  ShellCommandResult result;
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return result;

  result.started = true;
  std::array<char, 256> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    result.output += buffer.data();
  }
  result.exit_status = pclose(pipe);
  return result;
}
#endif

static std::string normalizeBrowseStartDirectory(const std::string& current_value) {
  std::error_code ec;
  if (!current_value.empty()) {
    fs::path current_path = fs::path(current_value);
    if (fs::is_directory(current_path, ec)) {
      return current_path.lexically_normal().string();
    }
    ec.clear();
    fs::path parent = current_path.parent_path();
    if (!parent.empty() && fs::is_directory(parent, ec)) {
      return parent.lexically_normal().string();
    }
  }

  ec.clear();
  fs::path cwd = fs::current_path(ec);
  return ec ? std::string() : cwd.string();
}

static std::optional<std::string> pickDirectory(const std::string& current_value, std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }

  const std::string start_dir = normalizeBrowseStartDirectory(current_value);

#if defined(__APPLE__)
  auto appleScriptQuote = [](const std::string& value) {
    std::string quoted = "\"";
    for (char c : value) {
      if (c == '\\' || c == '"') {
        quoted.push_back('\\');
      }
      quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
  };

  std::string script = "POSIX path of (choose folder";
  if (!start_dir.empty()) {
    script += " default location POSIX file " + appleScriptQuote(start_dir);
  }
  script += ")";

  const ShellCommandResult result = runShellCommand("osascript -e " + shellQuote(script) + " 2>/dev/null");
  if (!result.started) {
    if (error_message) {
      *error_message = "Failed to start the folder picker.";
    }
    return std::nullopt;
  }
  if (result.exit_status != 0) {
    return std::nullopt;
  }

  std::string selection = trimWhitespace(result.output);
  if (selection.empty()) {
    return std::nullopt;
  }
  return selection;
#elif defined(__linux__)
  const auto start_dir_arg = [&]() {
    if (start_dir.empty()) return std::string("./");
    return fs::path(start_dir).lexically_normal().string() + "/";
  }();

  struct PickerCommand {
    const char* executable;
    std::string command;
  };

  const std::array<PickerCommand, 4> pickers = {
      PickerCommand{
          "zenity",
          "zenity --file-selection --directory --filename=" + shellQuote(start_dir_arg) + " 2>/dev/null",
      },
      PickerCommand{
          "kdialog",
          "kdialog --getexistingdirectory " + shellQuote(start_dir.empty() ? "." : start_dir) + " 2>/dev/null",
      },
      PickerCommand{
          "yad",
          "yad --file-selection --directory --filename=" + shellQuote(start_dir_arg) + " 2>/dev/null",
      },
      PickerCommand{
          "qarma",
          "qarma --file-selection --directory --filename=" + shellQuote(start_dir_arg) + " 2>/dev/null",
      },
  };

  for (const PickerCommand& picker : pickers) {
    if (!commandExistsInPath(picker.executable)) {
      continue;
    }

    const ShellCommandResult result = runShellCommand(picker.command);
    if (!result.started) {
      if (error_message) {
        *error_message = "Failed to start the folder picker.";
      }
      return std::nullopt;
    }
    if (result.exit_status != 0) {
      return std::nullopt;
    }

    std::string selection = trimWhitespace(result.output);
    if (!selection.empty()) {
      return selection;
    }
    return std::nullopt;
  }

  if (error_message) {
    *error_message = "No folder picker available. Install zenity, kdialog, yad, or qarma, or type the path manually.";
  }
  return std::nullopt;
#else
  if (error_message) {
    *error_message = "Folder picker is not available on this platform. Type the path manually.";
  }
  return std::nullopt;
#endif
}

static std::optional<std::string> pickFile(const std::string& current_value, std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }

  const std::string start_dir = normalizeBrowseStartDirectory(current_value);

#if defined(__APPLE__)
  auto appleScriptQuote = [](const std::string& value) {
    std::string quoted = "\"";
    for (char c : value) {
      if (c == '\\' || c == '"') {
        quoted.push_back('\\');
      }
      quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
  };

  std::string script = "POSIX path of (choose file";
  if (!start_dir.empty()) {
    script += " default location POSIX file " + appleScriptQuote(start_dir);
  }
  script += ")";

  const ShellCommandResult result = runShellCommand("osascript -e " + shellQuote(script) + " 2>/dev/null");
  if (!result.started) {
    if (error_message) {
      *error_message = "Failed to start the file picker.";
    }
    return std::nullopt;
  }
  if (result.exit_status != 0) {
    return std::nullopt;
  }

  std::string selection = trimWhitespace(result.output);
  if (selection.empty()) {
    return std::nullopt;
  }
  return selection;
#elif defined(__linux__)
  const std::string start_arg = start_dir.empty() ? std::string("./") : start_dir;

  struct PickerCommand {
    const char* executable;
    std::string command;
  };

  const std::array<PickerCommand, 4> pickers = {
      PickerCommand{
          "zenity",
          "zenity --file-selection --filename=" + shellQuote(start_arg) + " 2>/dev/null",
      },
      PickerCommand{
          "kdialog",
          "kdialog --getopenfilename " + shellQuote(start_arg) + " 2>/dev/null",
      },
      PickerCommand{
          "yad",
          "yad --file-selection --filename=" + shellQuote(start_arg) + " 2>/dev/null",
      },
      PickerCommand{
          "qarma",
          "qarma --file-selection --filename=" + shellQuote(start_arg) + " 2>/dev/null",
      },
  };

  for (const PickerCommand& picker : pickers) {
    if (!commandExistsInPath(picker.executable)) {
      continue;
    }

    const ShellCommandResult result = runShellCommand(picker.command);
    if (!result.started) {
      if (error_message) {
        *error_message = "Failed to start the file picker.";
      }
      return std::nullopt;
    }
    if (result.exit_status != 0) {
      return std::nullopt;
    }

    std::string selection = trimWhitespace(result.output);
    if (!selection.empty()) {
      return selection;
    }
    return std::nullopt;
  }

  if (error_message) {
    *error_message = "No file picker available. Install zenity, kdialog, yad, or qarma, or type the path manually.";
  }
  return std::nullopt;
#else
  if (error_message) {
    *error_message = "File picker is not available on this platform. Type the path manually.";
  }
  return std::nullopt;
#endif
}

static bool isSupportedImageExtension(const fs::path& path) {
  const std::string ext = path.extension().string();
  if (ext.empty()) return false;
  std::string lower = ext;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return (lower == ".png" || lower == ".jpg" || lower == ".jpeg" ||
          lower == ".bmp" || lower == ".tiff" || lower == ".tif");
}

static uint64_t monotonicNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct OfflineFrameAnalysis {
  cv::Mat preview_bgr;
  DetectionResult detection;
  cv::Size image_size;
  bool found = false;
};

static bool loadOfflineImageForEncoding(const std::string& path, ImGuiApp::OfflineImageEncoding encoding,
                                        cv::Mat& bgr, std::string& error_message) {
  error_message.clear();

  if (encoding == ImGuiApp::OfflineImageEncoding::Standard) {
    bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      error_message = "No se pudo leer imagen";
      return false;
    }
    return true;
  }

  cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (raw.empty()) {
    error_message = "No se pudo leer imagen Bayer";
    return false;
  }
  if (raw.channels() != 1 || raw.depth() != CV_8U) {
    error_message = "La imagen Bayer debe ser monocanal de 8 bits";
    return false;
  }

  int code = cv::COLOR_BayerBG2BGR;
  switch (encoding) {
    case ImGuiApp::OfflineImageEncoding::BayerRG:
      code = cv::COLOR_BayerRG2BGR;
      break;
    case ImGuiApp::OfflineImageEncoding::BayerBG:
      code = cv::COLOR_BayerBG2BGR;
      break;
    case ImGuiApp::OfflineImageEncoding::BayerGB:
      code = cv::COLOR_BayerGB2BGR;
      break;
    case ImGuiApp::OfflineImageEncoding::BayerGR:
      code = cv::COLOR_BayerGR2BGR;
      break;
    case ImGuiApp::OfflineImageEncoding::Standard:
      break;
  }

  cv::cvtColor(raw, bgr, code);
  if (bgr.empty()) {
    error_message = "No se pudo demosaicar imagen Bayer";
    return false;
  }
  return true;
}

static cv::Mat downscaleOfflinePreview(const cv::Mat& preview_bgr) {
  if (preview_bgr.empty()) {
    return {};
  }

  constexpr int kOfflinePreviewMaxDim = 1400;
  const int full_max_dim = std::max(preview_bgr.cols, preview_bgr.rows);
  if (full_max_dim <= kOfflinePreviewMaxDim) {
    return preview_bgr.clone();
  }

  const double scale = static_cast<double>(kOfflinePreviewMaxDim) / static_cast<double>(full_max_dim);
  cv::Mat preview_small;
  cv::resize(preview_bgr, preview_small, cv::Size(), scale, scale, cv::INTER_AREA);
  return preview_small;
}

static cv::Mat makeOfflineDisplayPreview(const cv::Mat& frame_bgr, bool offline_ir_mode, bool offline_invert_ir) {
  if (frame_bgr.empty()) {
    return {};
  }

  cv::Mat preview_bgr;
  if (offline_ir_mode || offline_invert_ir) {
    cv::Mat gray_for_preview;
    cv::cvtColor(frame_bgr, gray_for_preview, cv::COLOR_BGR2GRAY);
    if (offline_invert_ir) {
      cv::bitwise_not(gray_for_preview, gray_for_preview);
    }
    cv::cvtColor(gray_for_preview, preview_bgr, cv::COLOR_GRAY2BGR);
  } else {
    preview_bgr = frame_bgr.clone();
  }
  return downscaleOfflinePreview(preview_bgr);
}

static OfflineFrameAnalysis analyzeOfflineFrame(const cv::Mat& frame_bgr, const PatternConfig& pattern_config,
                                                bool offline_ir_mode, bool offline_invert_ir) {
  OfflineFrameAnalysis analysis;
  if (frame_bgr.empty()) {
    return analysis;
  }

  cv::Mat gray_for_detection;
  cv::cvtColor(frame_bgr, gray_for_detection, cv::COLOR_BGR2GRAY);
  if (offline_invert_ir) {
    cv::bitwise_not(gray_for_detection, gray_for_detection);
  }

  if (offline_ir_mode || offline_invert_ir) {
    cv::cvtColor(gray_for_detection, analysis.preview_bgr, cv::COLOR_GRAY2BGR);
  } else {
    analysis.preview_bgr = frame_bgr.clone();
  }

  PatternDetector detector(pattern_config);
  analysis.detection = detector.detect(gray_for_detection);
  analysis.found = analysis.detection.found;
  analysis.image_size = frame_bgr.size();

  if (analysis.found && !analysis.detection.corners.empty()) {
    drawDetection(analysis.preview_bgr, pattern_config, analysis.detection);
  }

  analysis.preview_bgr = downscaleOfflinePreview(analysis.preview_bgr);

  return analysis;
}

static void drawResidualGrid(const ResidualGrid& grid, const ImVec2& size) {
  if (grid.cols <= 0 || grid.rows <= 0 || grid.mean_error.empty()) {
    ImGui::TextUnformatted("Residual grid unavailable");
    return;
  }

  ImGui::BeginChild("residual_grid", size, true);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float avail_w = std::max(20.0f, ImGui::GetContentRegionAvail().x);
  const float avail_h = std::max(20.0f, ImGui::GetContentRegionAvail().y);
  const float cell_w = avail_w / static_cast<float>(grid.cols);
  const float cell_h = avail_h / static_cast<float>(grid.rows);

  float max_e = 0.0f;
  for (float e : grid.mean_error) max_e = std::max(max_e, e);
  if (max_e < 1e-6f) max_e = 1.0f;

  for (int r = 0; r < grid.rows; ++r) {
    for (int c = 0; c < grid.cols; ++c) {
      const size_t idx = static_cast<size_t>(r * grid.cols + c);
      if (idx >= grid.mean_error.size()) continue;
      const float v = std::clamp(grid.mean_error[idx] / max_e, 0.0f, 1.0f);
      const int red = static_cast<int>(60.0f + 190.0f * v);
      const int green = static_cast<int>(180.0f - 120.0f * v);
      const int blue = static_cast<int>(80.0f - 60.0f * v);
      const ImU32 color = IM_COL32(red, green, blue, 220);
      const ImVec2 a(p0.x + static_cast<float>(c) * cell_w,
                     p0.y + static_cast<float>(r) * cell_h);
      const ImVec2 b(a.x + cell_w, a.y + cell_h);
      dl->AddRectFilled(a, b, color);
      dl->AddRect(a, b, IM_COL32(35, 35, 35, 140));
    }
  }

  ImGui::Dummy(ImVec2(avail_w, avail_h));
  ImGui::EndChild();
}

static const char* cameraStateLabel(CameraState state) {
  switch (state) {
    case CameraState::Disconnected:
      return "Disconnected";
    case CameraState::Connected:
      return "Connected";
    case CameraState::Streaming:
      return "Streaming";
    case CameraState::Error:
      return "Error";
    default:
      return "Unknown";
  }
}

static void drawPreviewFrame(const char* child_id, const cv::Mat& display_frame, ImageTexture& texture,
                             const ImVec2& child_size) {
  ImGui::BeginChild(child_id, child_size, true);
  if (!display_frame.empty()) {
    cv::Mat rgba;
    cv::cvtColor(display_frame, rgba, cv::COLOR_BGR2RGBA);
    texture.update(rgba);

    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;
    float tex_w = static_cast<float>(texture.width());
    float tex_h = static_cast<float>(texture.height());
    if (tex_w > 0 && tex_h > 0 && avail_w > 0 && avail_h > 0) {
      const float fit_scale = std::min(avail_w / tex_w, avail_h / tex_h);
      const float scale = std::min(1.0f, fit_scale);
      ImVec2 size(tex_w * scale, tex_h * scale);
      ImVec2 cursor = ImGui::GetCursorPos();
      if (size.x < avail_w) {
        cursor.x += (avail_w - size.x) * 0.5f;
      }
      if (size.y < avail_h) {
        cursor.y += (avail_h - size.y) * 0.5f;
      }
      ImGui::SetCursorPos(cursor);
      ImGui::Image(texture.id(), size);
    } else {
      ImGui::Text("No image");
    }
  } else {
    ImGui::Text("No image");
  }
  ImGui::EndChild();
}

static fs::path executableDir() {
#if defined(__linux__)
  std::array<char, 4096> exe_path{};
  const ssize_t n = readlink("/proc/self/exe", exe_path.data(), exe_path.size() - 1);
  if (n > 0) {
    exe_path[static_cast<size_t>(n)] = '\0';
    return fs::path(exe_path.data()).parent_path();
  }
#endif
  return {};
}

static bool loadWindowIconRGBA(std::vector<uint8_t>& rgba, int& width, int& height) {
  std::vector<fs::path> candidates = {
      fs::path("assets") / "cam_calibrator.png",
      fs::path("..") / "assets" / "cam_calibrator.png",
      fs::path("..") / ".." / "assets" / "cam_calibrator.png",
  };

  const fs::path exe_dir = executableDir();
  if (!exe_dir.empty()) {
    candidates.push_back(exe_dir / "assets" / "cam_calibrator.png");
    candidates.push_back(exe_dir.parent_path() / "assets" / "cam_calibrator.png");
    candidates.push_back(exe_dir.parent_path() / "share" / "camera_calibrator" / "assets" /
                         "cam_calibrator.png");
  }

  for (const auto& icon_path : candidates) {
    if (!fs::exists(icon_path)) continue;

    cv::Mat icon = cv::imread(icon_path.string(), cv::IMREAD_UNCHANGED);
    if (icon.empty()) continue;

    cv::Mat icon_rgba;
    if (icon.channels() == 4) {
      cv::cvtColor(icon, icon_rgba, cv::COLOR_BGRA2RGBA);
    } else if (icon.channels() == 3) {
      cv::cvtColor(icon, icon_rgba, cv::COLOR_BGR2RGBA);
    } else if (icon.channels() == 1) {
      cv::cvtColor(icon, icon_rgba, cv::COLOR_GRAY2RGBA);
    } else {
      continue;
    }

    if (!icon_rgba.isContinuous()) {
      icon_rgba = icon_rgba.clone();
    }

    width = icon_rgba.cols;
    height = icon_rgba.rows;
    const size_t bytes = icon_rgba.total() * icon_rgba.elemSize();
    rgba.assign(icon_rgba.data, icon_rgba.data + bytes);
    return true;
  }

  return false;
}

static void applyWindowIcon(GLFWwindow* window) {
  if (!window) return;

  std::vector<uint8_t> rgba;
  int width = 0;
  int height = 0;
  if (!loadWindowIconRGBA(rgba, width, height) || width <= 0 || height <= 0 || rgba.empty()) {
    return;
  }

  GLFWimage image;
  image.width = width;
  image.height = height;
  image.pixels = rgba.data();
  glfwSetWindowIcon(window, 1, &image);
}
}  // namespace

int ImGuiApp::run() {
  if (!glfwInit()) return 1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

  GLFWwindow* window = glfwCreateWindow(1380, 820, "Camera Calibrator", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  applyWindowIcon(window);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  applyTheme();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL2_Init();

  addCameraSlot();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    updateCalibrationTask();
    updateFrameProcessing();
    updateOfflineImageLoader();
    updateOfflineVideoLoader();
    updateOfflinePathPicker();
    updateOfflineImageFrameLoader();
    updateOfflineDetectionTask();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->WorkPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(main_viewport->WorkSize, ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);

    ImGuiWindowFlags main_window_flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::Begin("Camera Calibrator", nullptr, main_window_flags);
    ImGui::PopStyleVar();

    {
      ImGui::PushID("global_theme");
      int theme_index = static_cast<int>(ui_theme_);
      const char* theme_items[] = {"Dark", "Light"};
      const float icon_size = 18.0f;
      const float combo_width = 120.0f;
      const float total_width = icon_size + ImGui::GetStyle().ItemInnerSpacing.x + combo_width;
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - total_width);
      iconButton("theme_icon", UiIcon::Theme, icon_size, defaultIconColor(0.95f));
      ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
      ImGui::SetNextItemWidth(combo_width);
      const char* theme_tooltips[] = {
          "Dark theme for low-glare use and dim environments.",
          "Light theme for bright environments and clearer screenshots.",
      };
      if (comboWithTooltips("##theme_combo", &theme_index, theme_items, IM_ARRAYSIZE(theme_items),
                            theme_tooltips)) {
        ui_theme_ = static_cast<UiTheme>(theme_index);
        applyTheme();
      }
      ImGui::PopID();
      ImGui::Separator();
    }

    CameraSlot* active = activeSlot();

    const float splitter_width = 8.0f;
    const float min_left_width = 300.0f;
    const float min_right_width = 420.0f;
    const float total_width = ImGui::GetContentRegionAvail().x;
    const float max_left_width = std::max(min_left_width, total_width - min_right_width - splitter_width);
    left_panel_width_ = std::clamp(left_panel_width_, min_left_width, max_left_width);

    ImGui::BeginChild("left_panel", ImVec2(left_panel_width_, 0), true);
    drawCalibrationSourceControls();
    if (calibration_input_mode_ == CalibrationInputMode::Live) {
      ImGui::Separator();
      ImGui::TextUnformatted("Camera selector");
      if (active) {
        drawCameraSlot(static_cast<size_t>(active_slot_), *active);
      } else {
        ImGui::Text("No camera slot available");
      }
      ImGui::Separator();
    }
    drawCalibrationPanel(active);
    ImGui::EndChild();

    const float panel_height = ImGui::GetItemRectSize().y;
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::InvisibleButton("left_right_splitter", ImVec2(splitter_width, panel_height));
    {
      ImDrawList* dl = ImGui::GetWindowDrawList();
      ImVec2 a = ImGui::GetItemRectMin();
      ImVec2 b = ImGui::GetItemRectMax();
      ImGuiCol col = ImGuiCol_Separator;
      if (ImGui::IsItemActive()) {
        col = ImGuiCol_SeparatorActive;
      } else if (ImGui::IsItemHovered()) {
        col = ImGuiCol_SeparatorHovered;
      }
      dl->AddRectFilled(a, b, ImGui::GetColorU32(col));
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    if (ImGui::IsItemActive()) {
      left_panel_width_ = std::clamp(left_panel_width_ + ImGui::GetIO().MouseDelta.x,
                                     min_left_width, max_left_width);
    }

    ImGui::SameLine(0.0f, 0.0f);

    ImGui::BeginChild("right_panel", ImVec2(0, 0), true);
    constexpr float coverage_row_height = 60.0f;
    ImGui::BeginChild("right_tabs_area", ImVec2(0, -coverage_row_height), false);
    if (calibration_input_mode_ == CalibrationInputMode::Live) {
      int close_tab = -1;
      if (ImGui::BeginTabBar("preview_tabs")) {
        if (stereo_mode_enabled_ &&
            stereo_left_slot_ >= 0 &&
            stereo_right_slot_ >= 0 &&
            stereo_left_slot_ < static_cast<int>(slots_.size()) &&
            stereo_right_slot_ < static_cast<int>(slots_.size()) &&
            stereo_left_slot_ != stereo_right_slot_) {
          if (ImGui::BeginTabItem("Stereo Pair###stereo_pair_tab")) {
            auto& left = *slots_[static_cast<size_t>(stereo_left_slot_)];
            auto& right = *slots_[static_cast<size_t>(stereo_right_slot_)];
            const CameraInfo left_info = left.device.currentInfo();
            const CameraInfo right_info = right.device.currentInfo();
            const std::string left_label = left_info.label.empty()
                                               ? ("Slot " + std::to_string(stereo_left_slot_ + 1))
                                               : left_info.label;
            const std::string right_label = right_info.label.empty()
                                                ? ("Slot " + std::to_string(stereo_right_slot_ + 1))
                                                : right_info.label;

            ImGui::Text("Left: %s", left_label.c_str());
            ImGui::SameLine();
            ImGui::Text("Right: %s", right_label.c_str());
            ImGui::Separator();

	            if (ImGui::BeginTable("stereo_pair_preview", 2,
	                                  ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_BordersInnerV)) {
	              ImGui::TableNextColumn();
	              ImGui::Text("Left (%s)", cameraStateLabel(left.device.state()));
	              cv::Mat left_preview;
	              {
	                std::lock_guard<std::mutex> left_lock(left.frame_mutex);
	                left_preview = left.display_frame;
	              }
	              drawPreviewFrame("stereo_left_preview", left_preview, left.texture, ImVec2(0, 0));

	              ImGui::TableNextColumn();
	              ImGui::Text("Right (%s)", cameraStateLabel(right.device.state()));
	              cv::Mat right_preview;
	              {
	                std::lock_guard<std::mutex> right_lock(right.frame_mutex);
	                right_preview = right.display_frame;
	              }
	              drawPreviewFrame("stereo_right_preview", right_preview, right.texture, ImVec2(0, 0));
	              ImGui::EndTable();
	            }

            ImGui::EndTabItem();
          }
        }

        std::vector<size_t> tab_order;
        tab_order.reserve(slots_.size());
        if (stereo_mode_enabled_ &&
            stereo_left_slot_ >= 0 &&
            stereo_right_slot_ >= 0 &&
            stereo_left_slot_ < static_cast<int>(slots_.size()) &&
            stereo_right_slot_ < static_cast<int>(slots_.size()) &&
            stereo_left_slot_ != stereo_right_slot_) {
          tab_order.push_back(static_cast<size_t>(stereo_left_slot_));
          tab_order.push_back(static_cast<size_t>(stereo_right_slot_));
        }
        for (size_t i = 0; i < slots_.size(); ++i) {
          if (std::find(tab_order.begin(), tab_order.end(), i) == tab_order.end()) {
            tab_order.push_back(i);
          }
        }

        for (size_t order_idx = 0; order_idx < tab_order.size(); ++order_idx) {
          const size_t i = tab_order[order_idx];
          auto& slot = *slots_[i];
          CameraInfo info = slot.device.currentInfo();

          std::string tab_label;
          const CameraState slot_state = slot.device.state();
          if (!info.label.empty() &&
              (slot_state == CameraState::Connected || slot_state == CameraState::Streaming)) {
            tab_label = info.label;
          }
          if (!tab_label.empty() && slot_state == CameraState::Streaming) {
            tab_label += " *";
          }
          if (tab_label.empty()) {
            tab_label = "    ";
          }
          tab_label += "###preview_tab_" + std::to_string(i);

          bool open = true;
          bool* open_ptr = slots_.size() > 1 ? &open : nullptr;
          if (ImGui::BeginTabItem(tab_label.c_str(), open_ptr)) {
            active_slot_ = static_cast<int>(i);

            if (!info.label.empty()) {
              ImGui::Text("Device: %s", info.label.c_str());
            } else {
              ImGui::Text("Device: not connected");
            }

	            ImGui::Separator();

	            std::string preview_id = "preview_frame##slot_" + std::to_string(i);
	            cv::Mat preview_frame;
	            {
	              std::lock_guard<std::mutex> frame_lock(slot.frame_mutex);
	              preview_frame = slot.display_frame;
	            }
	            drawPreviewFrame(preview_id.c_str(), preview_frame, slot.texture, ImVec2(0, -42));

            const bool can_start = slot.device.state() == CameraState::Connected;
            const bool can_stop = slot.device.state() == CameraState::Streaming;

            if (!can_start) ImGui::BeginDisabled();
            if (ImGui::Button("Start")) {
              slot.device.startStreaming();
            }
            if (!can_start) ImGui::EndDisabled();

            ImGui::SameLine();

            if (!can_stop) ImGui::BeginDisabled();
            if (ImGui::Button("Stop")) {
              slot.device.stopStreaming();
            }
            if (!can_stop) ImGui::EndDisabled();

            ImGui::EndTabItem();
          }

          if (open_ptr && !open && close_tab < 0) {
            close_tab = static_cast<int>(i);
          }
        }
        ImGui::EndTabBar();
      }

      if (close_tab >= 0) {
        removeCameraSlot(static_cast<size_t>(close_tab));
      }
    } else {
      const char* source_label =
          (calibration_input_mode_ == CalibrationInputMode::ImageFolder) ? "Image folder" : "Video file";
      ImGui::Text("Offline source: %s", source_label);
      if (!offline_status_.empty()) {
        ImGui::TextWrapped("Status: %s", offline_status_.c_str());
      }
      ImGui::Separator();

      const cv::Mat& offline_preview =
          offline_preview_frame_bgr_.empty() ? offline_frame_bgr_ : offline_preview_frame_bgr_;
      if (!offline_preview.empty()) {
        drawPreviewFrame("offline_preview_right_panel", offline_preview, offline_texture_, ImVec2(0, 0));
      } else {
        ImGui::BeginChild("offline_preview_right_panel", ImVec2(0, 0), true);
        if (calibration_input_mode_ == CalibrationInputMode::ImageFolder) {
          ImGui::TextUnformatted("No offline image loaded. Use 'Load list' and 'Prev/Next'.");
        } else {
          ImGui::TextUnformatted("No video frame loaded. Use 'Open video' and 'Next frame'.");
        }
        ImGui::EndChild();
      }
    }

    ImGui::EndChild();
    ImGui::Separator();
    drawCoverageInlineRow(calibrator_.progress());

    ImGui::EndChild();
    ImGui::End();

    for (size_t i = 0; i < slots_.size(); ++i) {
      drawConfigDialog(i, *slots_[i]);
    }

#ifndef HAVE_ARAVIS
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGui::Begin("Backend status", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.20f, 1.0f),
                       "Aravis no disponible: backend GigE/GenICam deshabilitado.");
    ImGui::Text("Script: scripts/install_aravis_ubuntu.sh");
    ImGui::End();
#endif

    drawOfflineImageLoaderPopup();
    drawOfflineVideoLoaderPopup();
    drawOfflinePathPickerPopup();
    drawOfflineImageFrameLoaderPopup();
    drawCalibrationPopup();

    ImGui::Render();
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.10f, 0.10f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  stopOfflineImageLoader();
  stopOfflineVideoLoader();
  stopOfflinePathPicker();
  stopOfflineImageFrameLoader();
  stopCalibrationTask();
  stopOfflineDetectionWorker();

  for (auto& slot : slots_) {
    stopProcessingWorker(*slot);
    slot->device.stopStreaming();
    stopFeatureWorker(*slot);
    slot->device.disconnect();
  }

  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

void ImGuiApp::applyTheme() {
  switch (ui_theme_) {
    case UiTheme::Dark:
      ImGui::StyleColorsDark();
      break;
    case UiTheme::Light:
      ImGui::StyleColorsLight();
      break;
    default:
      ImGui::StyleColorsDark();
      break;
  }
}

void ImGuiApp::startProcessingWorker(CameraSlot& slot) {
  if (slot.processing_worker_running) return;
  slot.processing_worker_stop.store(false);
  slot.processing_worker_running = true;
  slot.processing_worker = std::thread(&ImGuiApp::processingWorkerLoop, this, &slot);
}

void ImGuiApp::stopProcessingWorker(CameraSlot& slot) {
  if (!slot.processing_worker_running) return;
  slot.processing_worker_stop.store(true);
  if (slot.processing_worker.joinable()) {
    slot.processing_worker.join();
  }
  {
    std::lock_guard<std::mutex> lock(slot.frame_mutex);
    slot.processing_fps = 0.0;
    slot.processed_frames = 0;
  }
  slot.processing_worker_running = false;
}

void ImGuiApp::processingWorkerLoop(CameraSlot* slot) {
  if (!slot) return;

  struct AsyncDetectionResult {
    uint64_t seq = 0;
    uint64_t timestamp_ns = 0;
    cv::Size image_size;
    PatternConfig config;
    DetectionResult detection;
    cv::Mat frame_bgr_full;
  };

  struct AsyncDetectionState {
    std::mutex mutex;
    bool busy = false;
    bool has_result = false;
    AsyncDetectionResult result;
  };

  PatternConfig local_config;
  uint64_t local_revision = 0;
  {
    std::lock_guard<std::mutex> lock(pattern_mutex_);
    local_config = pattern_config_;
    local_revision = pattern_revision_;
  }
  auto detection_state = std::make_shared<AsyncDetectionState>();
  DetectionResult latest_detection;
  PatternConfig latest_detection_config = local_config;
  cv::Mat latest_detection_gray;
  cv::Size latest_detection_size;
  bool has_latest_detection = false;
  uint64_t last_processed_seq = 0;
  uint64_t last_processed_ts = 0;
  uint64_t processed_frames = 0;
  uint64_t processed_in_window = 0;
  double processing_fps = 0.0;
  auto processing_window_start = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(slot->frame_mutex);
    slot->detection = DetectionResult{};
    slot->found = false;
    slot->metrics = SampleMetrics{};
  }

  while (!slot->processing_worker_stop.load()) {
    uint64_t revision_snapshot = local_revision;
    PatternConfig config_snapshot;
    {
      std::lock_guard<std::mutex> lock(pattern_mutex_);
      revision_snapshot = pattern_revision_;
      if (revision_snapshot != local_revision) {
        config_snapshot = pattern_config_;
      }
    }
    if (revision_snapshot != local_revision) {
      local_config = config_snapshot;
      local_revision = revision_snapshot;
      has_latest_detection = false;
      latest_detection = DetectionResult{};
      latest_detection_gray.release();
      latest_detection_size = cv::Size();
      latest_detection_config = local_config;
    }

    Frame frame;
    if (!slot->device.latestFrame(frame)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
      continue;
    }

    // Avoid reprocessing the same frame repeatedly when capture FPS is lower than worker loop rate.
    const bool same_seq = (frame.seq != 0 && frame.seq == last_processed_seq);
    const bool same_ts = (frame.seq == 0 && frame.timestamp_ns != 0 && frame.timestamp_ns == last_processed_ts);
    if (same_seq || same_ts) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    last_processed_seq = frame.seq;
    last_processed_ts = frame.timestamp_ns;

    constexpr int kPreviewMaxDim = 1400;
    const int full_max_dim = std::max(frame.width, frame.height);
    double display_scale = 1.0;
    if (full_max_dim > kPreviewMaxDim) {
      display_scale = static_cast<double>(kPreviewMaxDim) / static_cast<double>(full_max_dim);
    }

    cv::Mat bgr_full;
    const bool bayer_swap_rb = slot->bayer_swap_rb.load(std::memory_order_relaxed);
    if (!frameToBGR(frame, bgr_full, 1.0, bayer_swap_rb)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    cv::Mat gray_for_detection;
    cv::cvtColor(bgr_full, gray_for_detection, cv::COLOR_BGR2GRAY);
    if (slot->invert_ir) {
      cv::bitwise_not(gray_for_detection, gray_for_detection);
    }

    cv::Mat processed_bgr_full = bgr_full;
    if (slot->ir_mode) {
      cv::cvtColor(gray_for_detection, processed_bgr_full, cv::COLOR_GRAY2BGR);
    }

    bool detection_updated = false;
    AsyncDetectionResult new_detection;
    {
      std::lock_guard<std::mutex> lock(detection_state->mutex);
      if (detection_state->has_result) {
        new_detection = std::move(detection_state->result);
        detection_state->has_result = false;
        detection_updated = true;
      }
    }
    if (detection_updated) {
      if (!new_detection.frame_bgr_full.empty() &&
          new_detection.image_size.width > 0 && new_detection.image_size.height > 0 &&
          new_detection.detection.found && !new_detection.detection.corners.empty()) {
        latest_detection = new_detection.detection;
        latest_detection_config = new_detection.config;
        latest_detection_size = new_detection.image_size;
        cv::cvtColor(new_detection.frame_bgr_full, latest_detection_gray, cv::COLOR_BGR2GRAY);
        has_latest_detection = !latest_detection_gray.empty();
      } else if (latest_detection_size.width > 0 &&
                 latest_detection_size.height > 0 &&
                 new_detection.image_size != latest_detection_size) {
        has_latest_detection = false;
        latest_detection = DetectionResult{};
        latest_detection_gray.release();
        latest_detection_size = cv::Size();
      }
    }

    cv::Mat display;
    if (display_scale < 0.999) {
      cv::resize(processed_bgr_full, display, cv::Size(), display_scale, display_scale, cv::INTER_AREA);
    } else {
      display = processed_bgr_full.clone();
    }

    const bool should_detect = true;
    if (should_detect) {
      bool launch_detection = false;
      {
        std::lock_guard<std::mutex> lock(detection_state->mutex);
        if (!detection_state->busy) {
          detection_state->busy = true;
          launch_detection = true;
        }
      }
      if (launch_detection) {
        cv::Mat frame_for_detection = processed_bgr_full.clone();
        cv::Mat gray = gray_for_detection.clone();
        const uint64_t detect_seq = frame.seq;
        const uint64_t detect_ts = frame.timestamp_ns;
        const cv::Size detect_size(frame.width, frame.height);
        const PatternConfig detect_config = local_config;
        std::thread([state = detection_state,
                     gray = std::move(gray),
                     frame_for_detection = std::move(frame_for_detection),
                     detect_seq,
                     detect_ts,
                     detect_size,
                     detect_config]() mutable {
          AsyncDetectionResult result;
          result.seq = detect_seq;
          result.timestamp_ns = detect_ts;
          result.image_size = detect_size;
          result.config = detect_config;
          result.frame_bgr_full = std::move(frame_for_detection);
          try {
            PatternDetector worker_detector(detect_config);
            result.detection = worker_detector.detect(gray);
          } catch (...) {
            result.detection = DetectionResult{};
          }
          std::lock_guard<std::mutex> lock(state->mutex);
          state->result = std::move(result);
          state->has_result = true;
          state->busy = false;
        }).detach();
      }
    }

    if (has_latest_detection &&
        latest_detection_size.width == gray_for_detection.cols &&
        latest_detection_size.height == gray_for_detection.rows &&
        !latest_detection_gray.empty() &&
        latest_detection.found && !latest_detection.corners.empty()) {
      std::vector<cv::Point2f> tracked_corners;
      std::vector<uchar> track_status;
      std::vector<float> track_error;
      cv::calcOpticalFlowPyrLK(
          latest_detection_gray,
          gray_for_detection,
          latest_detection.corners,
          tracked_corners,
          track_status,
          track_error,
          cv::Size(21, 21),
          3,
          cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));

      size_t good_tracks = 0;
      for (uchar s : track_status) {
        if (s) ++good_tracks;
      }
      const size_t min_good_tracks = std::max<size_t>(6, (latest_detection.corners.size() * 3) / 4);
      if (!tracked_corners.empty() && good_tracks >= min_good_tracks) {
        DetectionResult det_tracked = latest_detection;
        for (size_t i = 0; i < det_tracked.corners.size() && i < tracked_corners.size(); ++i) {
          if (i < track_status.size() && track_status[i]) {
            det_tracked.corners[i] = tracked_corners[i];
          }
        }

        DetectionResult det_display = det_tracked;
        const float sx = static_cast<float>(display.cols) /
                         static_cast<float>(latest_detection_size.width);
        const float sy = static_cast<float>(display.rows) /
                         static_cast<float>(latest_detection_size.height);
        for (auto& c : det_display.corners) {
          c.x *= sx;
          c.y *= sy;
        }
        drawDetection(display, latest_detection_config, det_display);

        // Keep tracking state synchronized with the most recent preview frame.
        latest_detection = std::move(det_tracked);
        latest_detection_gray = gray_for_detection.clone();
      } else {
        has_latest_detection = false;
        latest_detection = DetectionResult{};
        latest_detection_gray.release();
        latest_detection_size = cv::Size();
      }
    }

    ++processed_frames;
    ++processed_in_window;
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - processing_window_start).count();
    if (elapsed_ms >= 500) {
      processing_fps = (static_cast<double>(processed_in_window) * 1000.0) /
                       static_cast<double>(elapsed_ms);
      processing_window_start = now;
      processed_in_window = 0;
    }

    std::lock_guard<std::mutex> lock(slot->frame_mutex);
    slot->display_frame = std::move(display);
    slot->processing_fps = processing_fps;
    slot->processed_frames = processed_frames;
    if (detection_updated) {
      slot->last_frame_raw = std::move(new_detection.frame_bgr_full);
      slot->last_image_size = new_detection.image_size;
      slot->detection = std::move(new_detection.detection);
      slot->detection_seq = new_detection.seq;
      slot->detection_timestamp_ns = new_detection.timestamp_ns;
      slot->found = slot->detection.found;
      if (!slot->found) {
        slot->metrics = SampleMetrics{};
      }
    }
  }
}

void ImGuiApp::startFeatureWorker(CameraSlot& slot) {
  std::lock_guard<std::mutex> lock(slot.feature_mutex);
  if (slot.feature_worker_running) return;
  slot.feature_worker_stop = false;
  slot.feature_worker_running = true;
  slot.feature_worker = std::thread(&ImGuiApp::featureWorkerLoop, this, &slot);
}

void ImGuiApp::stopFeatureWorker(CameraSlot& slot) {
  {
    std::lock_guard<std::mutex> lock(slot.feature_mutex);
    if (!slot.feature_worker_running) return;
    slot.feature_worker_stop = true;
    slot.feature_tasks.clear();
    slot.feature_refresh_queued = false;
    slot.feature_busy = false;
  }
  slot.feature_cv.notify_all();
  if (slot.feature_worker.joinable()) {
    slot.feature_worker.join();
  }
  std::lock_guard<std::mutex> lock(slot.feature_mutex);
  slot.feature_worker_running = false;
  slot.feature_busy = false;
  slot.feature_refresh_queued = false;
  slot.feature_tasks.clear();
}

void ImGuiApp::clearFeatureCache(CameraSlot& slot) {
  std::lock_guard<std::mutex> lock(slot.feature_mutex);
  slot.features.clear();
  slot.feature_values.clear();
  slot.feature_status.clear();
  slot.feature_busy = false;
  slot.feature_refresh_queued = false;
  slot.feature_tasks.clear();
}

void ImGuiApp::queueFeatureRefresh(CameraSlot& slot) {
  startFeatureWorker(slot);
  {
    std::lock_guard<std::mutex> lock(slot.feature_mutex);
    if (slot.feature_refresh_queued) return;
    FeatureTask task;
    task.type = FeatureTask::Type::Refresh;
    slot.feature_tasks.push_back(std::move(task));
    slot.feature_refresh_queued = true;
    slot.feature_busy = true;
  }
  slot.feature_cv.notify_one();
}

void ImGuiApp::queueFeatureSet(CameraSlot& slot, const FeatureInfo& feature, const FeatureValue& value) {
  startFeatureWorker(slot);
  {
    std::lock_guard<std::mutex> lock(slot.feature_mutex);
    FeatureTask task;
    task.type = FeatureTask::Type::SetValue;
    task.feature = feature;
    task.value = value;
    slot.feature_tasks.push_back(std::move(task));
    slot.feature_busy = true;
  }
  slot.feature_cv.notify_one();
}

void ImGuiApp::queueFeatureCommand(CameraSlot& slot, const FeatureInfo& feature) {
  startFeatureWorker(slot);
  {
    std::lock_guard<std::mutex> lock(slot.feature_mutex);
    FeatureTask task;
    task.type = FeatureTask::Type::ExecuteCommand;
    task.feature = feature;
    slot.feature_tasks.push_back(std::move(task));
    slot.feature_busy = true;
  }
  slot.feature_cv.notify_one();
}

void ImGuiApp::featureWorkerLoop(CameraSlot* slot) {
  if (!slot) return;

  while (true) {
    FeatureTask task;
    {
      std::unique_lock<std::mutex> lock(slot->feature_mutex);
      slot->feature_cv.wait(lock, [&]() {
        return slot->feature_worker_stop || !slot->feature_tasks.empty();
      });

      if (slot->feature_worker_stop) {
        slot->feature_tasks.clear();
        slot->feature_refresh_queued = false;
        break;
      }

      task = std::move(slot->feature_tasks.front());
      slot->feature_tasks.pop_front();
      if (task.type == FeatureTask::Type::Refresh) {
        slot->feature_refresh_queued = false;
      }
    }

    switch (task.type) {
      case FeatureTask::Type::Refresh: {
        std::vector<FeatureInfo> features = slot->device.listFeatures();
        std::unordered_map<std::string, FeatureValue> values;
        values.reserve(features.size());
        for (const auto& feature : features) {
          if (!feature.readable) continue;
          if (feature.type == FeatureType::Category ||
              feature.type == FeatureType::Command ||
              feature.type == FeatureType::Unknown) {
            continue;
          }
          FeatureValue value;
          if (slot->device.getFeatureValue(feature, value)) {
            values[feature.id] = std::move(value);
          }
        }

        std::lock_guard<std::mutex> lock(slot->feature_mutex);
        slot->features = std::move(features);
        slot->feature_values = std::move(values);
        slot->feature_status = slot->device.status();
        break;
      }

      case FeatureTask::Type::SetValue: {
        bool ok = slot->device.setFeatureValue(task.feature, task.value);
        if (ok && task.feature.readable) {
          FeatureValue read_back;
          if (slot->device.getFeatureValue(task.feature, read_back)) {
            std::lock_guard<std::mutex> lock(slot->feature_mutex);
            slot->feature_values[task.feature.id] = std::move(read_back);
          }
        }
        std::lock_guard<std::mutex> lock(slot->feature_mutex);
        if (!ok) {
          slot->feature_status = "Failed to set feature: " + task.feature.id;
        } else {
          slot->feature_status = slot->device.status();
        }
        break;
      }

      case FeatureTask::Type::ExecuteCommand: {
        const bool ok = slot->device.executeCommand(task.feature);
        std::lock_guard<std::mutex> lock(slot->feature_mutex);
        if (!ok) {
          slot->feature_status = "Failed to execute command: " + task.feature.id;
        } else {
          slot->feature_status = slot->device.status();
        }
        break;
      }
    }

    {
      std::lock_guard<std::mutex> lock(slot->feature_mutex);
      slot->feature_busy = !slot->feature_tasks.empty();
    }
  }

  std::lock_guard<std::mutex> lock(slot->feature_mutex);
  slot->feature_worker_running = false;
  slot->feature_busy = false;
  slot->feature_refresh_queued = false;
}

void ImGuiApp::addCameraSlot() {
  slots_.push_back(std::make_unique<CameraSlot>());
  CameraSlot& slot = *slots_.back();
  startProcessingWorker(slot);
  refreshDevices(slot);
  active_slot_ = static_cast<int>(slots_.size() - 1);
}

void ImGuiApp::removeCameraSlot(size_t index) {
  if (index >= slots_.size()) return;
  stopProcessingWorker(*slots_[index]);
  slots_[index]->device.stopStreaming();
  stopFeatureWorker(*slots_[index]);
  slots_[index]->device.disconnect();
  slots_.erase(slots_.begin() + static_cast<long>(index));
  if (slots_.empty()) {
    active_slot_ = 0;
  } else if (active_slot_ >= static_cast<int>(slots_.size())) {
    active_slot_ = static_cast<int>(slots_.size() - 1);
  }
}

ImGuiApp::CameraSlot* ImGuiApp::activeSlot() {
  if (slots_.empty()) return nullptr;
  if (active_slot_ < 0 || active_slot_ >= static_cast<int>(slots_.size())) return nullptr;
  return slots_[static_cast<size_t>(active_slot_)].get();
}

bool ImGuiApp::isDeviceInUse(const std::string& id, const CameraSlot* ignore) const {
  if (id.empty()) return false;
  for (const auto& slot_ptr : slots_) {
    if (!slot_ptr) continue;
    if (ignore && slot_ptr.get() == ignore) continue;
    CameraInfo info = slot_ptr->device.currentInfo();
    if (!info.id.empty() && info.id == id) return true;
  }
  return false;
}

int ImGuiApp::findSlotByDevice(const std::string& id) const {
  if (id.empty()) return -1;
  for (size_t i = 0; i < slots_.size(); ++i) {
    CameraInfo info = slots_[i]->device.currentInfo();
    if (!info.id.empty() && info.id == id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void ImGuiApp::refreshDevices(CameraSlot& slot) {
  std::string previously_selected;
  if (slot.selected_device >= 0 && slot.selected_device < static_cast<int>(slot.devices.size())) {
    previously_selected = slot.devices[slot.selected_device].id;
  }

  CameraInfo current = slot.device.currentInfo();
  slot.devices = slot.device.listDevices();

  if (!current.id.empty()) {
    bool found_current = false;
    for (const auto& dev : slot.devices) {
      if (dev.id == current.id) {
        found_current = true;
        break;
      }
    }
    if (!found_current) {
      slot.devices.push_back(current);
    }
  }

  std::string target_id = previously_selected;
  if (target_id.empty() && !current.id.empty()) {
    target_id = current.id;
  }

  slot.selected_device = -1;
  for (size_t i = 0; i < slot.devices.size(); ++i) {
    if (!target_id.empty() && slot.devices[i].id == target_id) {
      slot.selected_device = static_cast<int>(i);
      break;
    }
  }

  if (slot.selected_device < 0 && !slot.devices.empty()) {
    slot.selected_device = 0;
  }
}

void ImGuiApp::updateFrameProcessing() {
  if (calibration_input_mode_ != CalibrationInputMode::Live) {
    return;
  }

  CameraSlot* active = activeSlot();
  if (!active) return;

  DetectionResult active_detection;
  cv::Size active_image_size;
  cv::Mat active_last_frame;
  bool active_found = false;
  {
    std::lock_guard<std::mutex> lock(active->frame_mutex);
    active_detection = active->detection;
    active_image_size = active->last_image_size;
    active_last_frame = active->last_frame_raw;
    active_found = active->found;
  }

  SampleMetrics metrics{};
  if (active_found && !active_detection.corners.empty() && !active_image_size.empty()) {
    metrics = calibrator_.computeMetrics(active_detection.corners, active_detection.ids, active_image_size);
  }
  {
    std::lock_guard<std::mutex> lock(active->frame_mutex);
    active->metrics = metrics;
  }

  bool calibration_busy = false;
  {
    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    calibration_busy = calibration_task_.running || calibration_task_.result_ready;
  }

  if (!calibration_busy &&
      auto_add_ && active_found && !active_detection.corners.empty() && !active_image_size.empty()) {
    PatternConfig current_pattern;
    {
      std::lock_guard<std::mutex> lock(pattern_mutex_);
      current_pattern = pattern_config_;
    }
    calibrator_.setPattern(current_pattern);
    calibrator_.setCameraModel(camera_model_);
    calibrator_.setPinholeDistortionModel(pinhole_distortion_model_);
    calibrator_.setTargetWarpCompensation(target_warp_compensation_);
    if (calibrator_.addSample(active_detection.corners, active_detection.ids,
                              active_image_size, allow_duplicates_)) {
      if (save_samples_ && !active_last_frame.empty()) {
        fs::create_directories(samples_dir_);
        std::ostringstream name;
        name << samples_dir_ << "/sample_" << calibrator_.sampleCount() << ".png";
        cv::imwrite(name.str(), active_last_frame);
      }
    }
  }
}

void ImGuiApp::updateOfflineImageLoader() {
  std::vector<std::string> loaded_files;
  std::string result_status;
  bool has_result = false;

  {
    std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
    if (!offline_image_loader_.result_ready) {
      return;
    }
    loaded_files = std::move(offline_image_loader_.files);
    result_status = offline_image_loader_.result_status;
    offline_image_loader_.files.clear();
    offline_image_loader_.result_ready = false;
    has_result = true;
  }

  if (offline_image_loader_.worker.joinable()) {
    offline_image_loader_.worker.join();
  }
  if (!has_result) {
    return;
  }

  offline_image_files_.clear();
  offline_image_index_ = -1;
  offline_image_files_ = std::move(loaded_files);

  if (!offline_image_files_.empty()) {
    requestOfflineImageLoad(0);
  } else if (!result_status.empty()) {
    offline_status_ = result_status;
  }

  {
    std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
    offline_image_loader_.show_popup = false;
  }
}

void ImGuiApp::stopOfflineImageLoader() {
  if (offline_image_loader_.worker.joinable()) {
    offline_image_loader_.worker.join();
  }
  std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
  offline_image_loader_.running = false;
  offline_image_loader_.result_ready = false;
  offline_image_loader_.show_popup = false;
}

void ImGuiApp::drawOfflineImageLoaderPopup() {
  bool show_popup = false;
  bool running = false;
  std::string target_dir;
  std::string progress_message;
  size_t scanned_entries = 0;
  size_t found_images = 0;

  {
    std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
    show_popup = offline_image_loader_.show_popup;
    running = offline_image_loader_.running;
    target_dir = offline_image_loader_.target_dir;
    progress_message = offline_image_loader_.progress_message;
    scanned_entries = offline_image_loader_.scanned_entries;
    found_images = offline_image_loader_.found_images;
  }

  if (show_popup) {
    ImGui::OpenPopup("Loading image folder");
  }

  if (ImGui::BeginPopupModal("Loading image folder", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
    if (running) {
      ImGui::TextWrapped("Scanning folder:");
      ImGui::TextWrapped("%s", target_dir.c_str());
      ImGui::Spacing();
      ImGui::TextUnformatted(progress_message.empty() ? "Scanning image files..." : progress_message.c_str());
      ImGui::Text("Entries scanned: %zu", scanned_entries);
      ImGui::Text("Images found: %zu", found_images);
      ImGui::Spacing();
      const float t = static_cast<float>(glfwGetTime());
      const float pulse = 0.25f + 0.65f * (0.5f + 0.5f * std::sin(t * 5.0f));
      ImGui::ProgressBar(pulse, ImVec2(320.0f, 0.0f), "");
      ImGui::TextDisabled("The UI stays responsive while the folder is being indexed.");
    } else {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void ImGuiApp::updateOfflineVideoLoader() {
  bool has_result = false;
  bool open_succeeded = false;
  std::string result_status;

  {
    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    if (!offline_video_loader_.result_ready) {
      return;
    }
    open_succeeded = offline_video_loader_.open_succeeded;
    result_status = offline_video_loader_.result_status;
    offline_video_loader_.result_ready = false;
    has_result = true;
  }

  if (offline_video_loader_.worker.joinable()) {
    offline_video_loader_.worker.join();
  }
  if (!has_result) {
    return;
  }

  offline_video_open_ = open_succeeded;
  offline_video_frame_index_ = -1;
  if (!result_status.empty()) {
    offline_status_ = result_status;
  }

  {
    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    offline_video_loader_.show_popup = false;
  }
}

void ImGuiApp::stopOfflineVideoLoader() {
  if (offline_video_loader_.worker.joinable()) {
    offline_video_loader_.worker.join();
  }
  std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
  offline_video_loader_.running = false;
  offline_video_loader_.result_ready = false;
  offline_video_loader_.show_popup = false;
}

void ImGuiApp::drawOfflineVideoLoaderPopup() {
  bool show_popup = false;
  bool running = false;
  std::string target_path;
  std::string progress_message;

  {
    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    show_popup = offline_video_loader_.show_popup;
    running = offline_video_loader_.running;
    target_path = offline_video_loader_.target_path;
    progress_message = offline_video_loader_.progress_message;
  }

  if (show_popup) {
    ImGui::OpenPopup("Opening video file");
  }

  if (ImGui::BeginPopupModal("Opening video file", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
    if (running) {
      ImGui::TextWrapped("Opening video:");
      ImGui::TextWrapped("%s", target_path.c_str());
      ImGui::Spacing();
      ImGui::TextUnformatted(progress_message.empty() ? "Opening video stream..." : progress_message.c_str());
      ImGui::Spacing();
      const float t = static_cast<float>(glfwGetTime());
      const float pulse = 0.25f + 0.65f * (0.5f + 0.5f * std::sin(t * 5.0f));
      ImGui::ProgressBar(pulse, ImVec2(320.0f, 0.0f), "");
      ImGui::TextDisabled("The UI stays responsive while the video backend is opening the file.");
    } else {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void ImGuiApp::updateOfflinePathPicker() {
  OfflinePathPickerTarget target = OfflinePathPickerTarget::None;
  bool pick_directory = false;
  bool selection_made = false;
  int slot_index = -1;
  std::string selected_path;
  std::string error_message;
  bool has_result = false;

  {
    std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
    if (!offline_path_picker_.result_ready) {
      return;
    }
    target = offline_path_picker_.target;
    pick_directory = offline_path_picker_.pick_directory;
    selection_made = offline_path_picker_.selection_made;
    slot_index = offline_path_picker_.slot_index;
    selected_path = offline_path_picker_.selected_path;
    error_message = offline_path_picker_.error_message;
    offline_path_picker_.result_ready = false;
    has_result = true;
  }

  if (offline_path_picker_.worker.joinable()) {
    offline_path_picker_.worker.join();
  }
  if (!has_result) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
    offline_path_picker_.show_popup = false;
  }

  if (!error_message.empty()) {
    offline_status_ = error_message;
    return;
  }
  if (!selection_made) {
    return;
  }

  if (target == OfflinePathPickerTarget::ImageFolder && pick_directory) {
    offline_images_dir_ = selected_path;
    refreshOfflineImageList();
  } else if (target == OfflinePathPickerTarget::VideoFile && !pick_directory) {
    offline_video_path_ = selected_path;
    openOfflineVideo();
  } else if (target == OfflinePathPickerTarget::GenTLCtiFile && !pick_directory &&
             slot_index >= 0 && static_cast<size_t>(slot_index) < slots_.size()) {
    slots_[static_cast<size_t>(slot_index)]->gentl_cti_path = selected_path;
  }
}

void ImGuiApp::stopOfflinePathPicker() {
  if (offline_path_picker_.worker.joinable()) {
    offline_path_picker_.worker.join();
  }
  std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
  offline_path_picker_.running = false;
  offline_path_picker_.result_ready = false;
  offline_path_picker_.show_popup = false;
  offline_path_picker_.slot_index = -1;
  offline_path_picker_.target = OfflinePathPickerTarget::None;
}

void ImGuiApp::drawOfflinePathPickerPopup() {
  bool show_popup = false;
  bool running = false;
  bool pick_directory = false;
  std::string current_value;
  std::string progress_message;

  {
    std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
    show_popup = offline_path_picker_.show_popup;
    running = offline_path_picker_.running;
    pick_directory = offline_path_picker_.pick_directory;
    current_value = offline_path_picker_.current_value;
    progress_message = offline_path_picker_.progress_message;
  }

  const char* popup_name = pick_directory ? "Selecting folder" : "Selecting file";
  if (show_popup) {
    ImGui::OpenPopup(popup_name);
  }

  if (ImGui::BeginPopupModal(popup_name, nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
    if (running) {
      ImGui::TextUnformatted(pick_directory ? "Waiting for folder picker..." : "Waiting for file picker...");
      if (!current_value.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", current_value.c_str());
      }
      ImGui::Spacing();
      ImGui::TextUnformatted(progress_message.empty() ? "The system picker is open." : progress_message.c_str());
      ImGui::Spacing();
      const float t = static_cast<float>(glfwGetTime());
      const float pulse = 0.25f + 0.65f * (0.5f + 0.5f * std::sin(t * 5.0f));
      ImGui::ProgressBar(pulse, ImVec2(320.0f, 0.0f), "");
      ImGui::TextDisabled("The app remains responsive while the external picker is open.");
    } else {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

bool ImGuiApp::requestOfflinePathPicker(OfflinePathPickerTarget target, bool pick_directory,
                                        const std::string& current_value, int slot_index) {
  bool is_running = false;
  {
    std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
    is_running = offline_path_picker_.running;
  }
  if (is_running) {
    offline_status_ = pick_directory ? "Seleccion de carpeta en curso..." : "Seleccion de archivo en curso...";
    return false;
  }

  if (offline_path_picker_.worker.joinable()) {
    offline_path_picker_.worker.join();
  }

  {
    std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
    offline_path_picker_.running = true;
    offline_path_picker_.result_ready = false;
    offline_path_picker_.show_popup = true;
    offline_path_picker_.pick_directory = pick_directory;
    offline_path_picker_.selection_made = false;
    offline_path_picker_.slot_index = slot_index;
    offline_path_picker_.target = target;
    offline_path_picker_.current_value = current_value;
    offline_path_picker_.progress_message =
        pick_directory ? "The folder picker is running in the background."
                       : "The file picker is running in the background.";
    offline_path_picker_.selected_path.clear();
    offline_path_picker_.error_message.clear();
  }

  offline_status_ = pick_directory ? "Seleccionando carpeta..." : "Seleccionando archivo...";
  offline_path_picker_.worker = std::thread([this, target, pick_directory, current_value, slot_index]() {
    std::string error_message;
    std::optional<std::string> selected_path =
        pick_directory ? pickDirectory(current_value, &error_message)
                       : pickFile(current_value, &error_message);

    std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
    offline_path_picker_.selection_made = selected_path.has_value();
    offline_path_picker_.selected_path = selected_path.value_or(std::string{});
    offline_path_picker_.error_message = std::move(error_message);
    offline_path_picker_.slot_index = slot_index;
    offline_path_picker_.target = target;
    offline_path_picker_.running = false;
    offline_path_picker_.result_ready = true;
  });

  return true;
}

void ImGuiApp::updateOfflineImageFrameLoader() {
  cv::Mat preview_update_bgr;
  std::string preview_status;
  bool has_preview_update = false;
  cv::Mat frame_bgr;
  cv::Mat preview_bgr;
  cv::Size image_size;
  int target_index = -1;
  std::string result_status;
  bool has_result = false;

  {
    std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
    if (offline_image_frame_loader_.preview_ready) {
      preview_update_bgr = std::move(offline_image_frame_loader_.preview_bgr);
      preview_status = offline_image_frame_loader_.result_status;
      offline_image_frame_loader_.preview_bgr.release();
      offline_image_frame_loader_.preview_ready = false;
      has_preview_update = true;
    }
    if (!offline_image_frame_loader_.result_ready) {
      if (has_preview_update && !preview_update_bgr.empty()) {
        offline_preview_frame_bgr_ = std::move(preview_update_bgr);
        if (!preview_status.empty()) {
          offline_status_ = preview_status;
        }
      }
      return;
    }
    frame_bgr = std::move(offline_image_frame_loader_.frame_bgr);
    preview_bgr = std::move(offline_image_frame_loader_.preview_bgr);
    image_size = offline_image_frame_loader_.image_size;
    target_index = offline_image_frame_loader_.target_index;
    result_status = offline_image_frame_loader_.result_status;
    offline_image_frame_loader_.frame_bgr.release();
    offline_image_frame_loader_.preview_bgr.release();
    offline_image_frame_loader_.detection = DetectionResult{};
    offline_image_frame_loader_.image_size = cv::Size();
    offline_image_frame_loader_.result_ready = false;
    has_result = true;
  }

  if (has_preview_update && !preview_update_bgr.empty()) {
    offline_preview_frame_bgr_ = std::move(preview_update_bgr);
    if (!preview_status.empty()) {
      offline_status_ = preview_status;
    }
  }

  if (offline_image_frame_loader_.worker.joinable()) {
    offline_image_frame_loader_.worker.join();
  }
  if (!has_result) {
    return;
  }

  if (!frame_bgr.empty()) {
    offline_image_index_ = target_index;
    offline_frame_bgr_ = std::move(frame_bgr);
    offline_preview_frame_bgr_ = std::move(preview_bgr);
    offline_image_size_ = image_size;
    offline_video_frame_index_ = -1;
    offline_seq_ += 1;
    offline_timestamp_ns_ = monotonicNowNs();
    processOfflineFrame();
  }

  if (frame_bgr.empty() && !result_status.empty()) {
    offline_status_ = result_status;
  }

  {
    std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
    offline_image_frame_loader_.show_popup = false;
  }
}

void ImGuiApp::stopOfflineImageFrameLoader() {
  if (offline_image_frame_loader_.worker.joinable()) {
    offline_image_frame_loader_.worker.join();
  }
  std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
  offline_image_frame_loader_.running = false;
  offline_image_frame_loader_.preview_ready = false;
  offline_image_frame_loader_.result_ready = false;
  offline_image_frame_loader_.show_popup = false;
}

void ImGuiApp::drawOfflineImageFrameLoaderPopup() {
  bool show_popup = false;
  bool running = false;
  int target_index = -1;
  std::string target_path;
  std::string progress_message;

  {
    std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
    show_popup = offline_image_frame_loader_.show_popup;
    running = offline_image_frame_loader_.running;
    target_index = offline_image_frame_loader_.target_index;
    target_path = offline_image_frame_loader_.target_path;
    progress_message = offline_image_frame_loader_.progress_message;
  }

  if (show_popup) {
    ImGui::OpenPopup("Loading image");
  }

  if (ImGui::BeginPopupModal("Loading image", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
    if (running) {
      ImGui::Text("Loading image %d / %d", target_index + 1, static_cast<int>(offline_image_files_.size()));
      ImGui::TextWrapped("%s", target_path.c_str());
      ImGui::Spacing();
      ImGui::TextUnformatted(progress_message.empty() ? "Reading image and building preview..." : progress_message.c_str());
      ImGui::Spacing();
      const float t = static_cast<float>(glfwGetTime());
      const float pulse = 0.25f + 0.65f * (0.5f + 0.5f * std::sin(t * 5.0f));
      ImGui::ProgressBar(pulse, ImVec2(320.0f, 0.0f), "");
      ImGui::TextDisabled("The image is shown first; pattern detection continues in the background.");
    } else {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void ImGuiApp::updateCalibrationTask() {
  bool has_result = false;
  CalibrationSession result_session;

  {
    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    if (!calibration_task_.result_ready) {
      return;
    }
    result_session = std::move(calibration_task_.result_session);
    calibration_task_.result_ready = false;
    calibration_task_.show_popup = false;
    has_result = true;
  }

  if (calibration_task_.worker.joinable()) {
    calibration_task_.worker.join();
  }
  if (!has_result) {
    return;
  }

  calibrator_ = std::move(result_session);
  PatternConfig current_pattern;
  {
    std::lock_guard<std::mutex> lock(pattern_mutex_);
    current_pattern = pattern_config_;
  }
  calibrator_.setPattern(current_pattern);
  calibrator_.setCameraModel(camera_model_);
  calibrator_.setPinholeDistortionModel(pinhole_distortion_model_);
  calibrator_.setTargetWarpCompensation(target_warp_compensation_);
}

void ImGuiApp::stopCalibrationTask() {
  if (calibration_task_.worker.joinable()) {
    calibration_task_.worker.join();
  }
  std::lock_guard<std::mutex> lock(calibration_task_.mutex);
  calibration_task_.running = false;
  calibration_task_.result_ready = false;
  calibration_task_.show_popup = false;
  calibration_task_.progress = 0.0f;
  calibration_task_.progress_message.clear();
}

void ImGuiApp::drawCalibrationPopup() {
  bool show_popup = false;
  bool running = false;
  float progress = 0.0f;
  std::string progress_message;

  {
    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    show_popup = calibration_task_.show_popup;
    running = calibration_task_.running;
    progress = calibration_task_.progress;
    progress_message = calibration_task_.progress_message;
  }

  if (show_popup) {
    ImGui::OpenPopup("Calibrating camera");
  }

  if (ImGui::BeginPopupModal("Calibrating camera", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
    if (running) {
      const float clamped_progress = std::clamp(progress, 0.0f, 1.0f);
      const int progress_percent = static_cast<int>(std::round(clamped_progress * 100.0f));
      std::string overlay = std::to_string(progress_percent) + "%";

      ImGui::TextUnformatted("Running intrinsic calibration...");
      ImGui::Spacing();
      ImGui::TextWrapped("%s",
                         progress_message.empty() ? "Calibrating with the current sample set..."
                                                  : progress_message.c_str());
      ImGui::Spacing();
      ImGui::ProgressBar(clamped_progress, ImVec2(320.0f, 0.0f), overlay.c_str());
      ImGui::TextDisabled("The UI stays responsive while OpenCV solves the calibration.");
    } else {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

bool ImGuiApp::startCalibrationTask(const cv::Size& image_size, const PatternConfig& pattern_config) {
  if (image_size.empty()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    if (calibration_task_.running || calibration_task_.result_ready) {
      return false;
    }
  }

  if (calibration_task_.worker.joinable()) {
    calibration_task_.worker.join();
  }

  CalibrationSession worker_session = calibrator_;
  worker_session.setPattern(pattern_config);
  worker_session.setCameraModel(camera_model_);
  worker_session.setPinholeDistortionModel(pinhole_distortion_model_);
  worker_session.setTargetWarpCompensation(target_warp_compensation_);

  {
    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    calibration_task_.running = true;
    calibration_task_.result_ready = false;
    calibration_task_.show_popup = true;
    calibration_task_.progress = 0.0f;
    calibration_task_.progress_message = "Iniciando calibracion...";
    calibration_task_.result_session = CalibrationSession{};
  }

  calibration_task_.worker = std::thread([this,
                                          image_size,
                                          worker_session = std::move(worker_session)]() mutable {
    auto progress_callback = [this](float progress, const std::string& message) {
      std::lock_guard<std::mutex> lock(calibration_task_.mutex);
      calibration_task_.progress = std::clamp(progress, 0.0f, 1.0f);
      calibration_task_.progress_message = message;
    };

    worker_session.calibrate(image_size, progress_callback);

    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    calibration_task_.progress = 1.0f;
    if (!worker_session.lastStatus().empty()) {
      calibration_task_.progress_message = worker_session.lastStatus();
    }
    calibration_task_.result_session = std::move(worker_session);
    calibration_task_.running = false;
    calibration_task_.result_ready = true;
  });

  return true;
}

void ImGuiApp::updateOfflineDetectionTask() {
  cv::Mat preview_bgr;
  DetectionResult detection;
  cv::Size image_size;
  bool found = false;
  std::string result_status;
  uint64_t result_request_id = 0;
  bool has_result = false;

  {
    std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
    if (!offline_detection_task_.result_ready) {
      return;
    }
    preview_bgr = std::move(offline_detection_task_.preview_bgr);
    detection = std::move(offline_detection_task_.detection);
    image_size = offline_detection_task_.image_size;
    found = offline_detection_task_.found;
    result_status = std::move(offline_detection_task_.result_status);
    result_request_id = offline_detection_task_.result_request_id;
    offline_detection_task_.preview_bgr.release();
    offline_detection_task_.detection = DetectionResult{};
    offline_detection_task_.image_size = cv::Size();
    offline_detection_task_.found = false;
    offline_detection_task_.result_ready = false;
    has_result = true;
  }

  if (!has_result || result_request_id != offline_detection_request_id_ || offline_frame_bgr_.empty()) {
    return;
  }

  offline_preview_frame_bgr_ = std::move(preview_bgr);
  offline_detection_ = std::move(detection);
  offline_found_ = found;
  offline_image_size_ = image_size;
  offline_detection_state_ = OfflineDetectionState::Ready;

  if (offline_found_ && !offline_detection_.corners.empty()) {
    offline_metrics_ = calibrator_.computeMetrics(offline_detection_.corners, offline_detection_.ids,
                                                  offline_image_size_);
  } else {
    offline_metrics_ = SampleMetrics{};
  }

  if (!result_status.empty()) {
    offline_status_ = result_status;
  }
}

void ImGuiApp::startOfflineDetectionWorker() {
  if (offline_detection_task_.worker.joinable()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
    offline_detection_task_.stop_requested = false;
  }

  offline_detection_task_.worker = std::thread(&ImGuiApp::offlineDetectionWorkerLoop, this);
}

void ImGuiApp::stopOfflineDetectionWorker() {
  {
    std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
    offline_detection_task_.stop_requested = true;
    offline_detection_task_.has_request = false;
  }
  offline_detection_task_.cv.notify_all();

  if (offline_detection_task_.worker.joinable()) {
    offline_detection_task_.worker.join();
  }

  std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
  offline_detection_task_.stop_requested = false;
  offline_detection_task_.busy = false;
  offline_detection_task_.has_request = false;
  offline_detection_task_.result_ready = false;
  offline_detection_task_.pending_request_id = 0;
  offline_detection_task_.result_request_id = 0;
  offline_detection_task_.pending_frame_bgr.release();
  offline_detection_task_.preview_bgr.release();
  offline_detection_task_.detection = DetectionResult{};
  offline_detection_task_.image_size = cv::Size();
  offline_detection_task_.found = false;
  offline_detection_task_.progress_message.clear();
  offline_detection_task_.result_status.clear();
}

void ImGuiApp::offlineDetectionWorkerLoop() {
  while (true) {
    uint64_t request_id = 0;
    cv::Mat frame_bgr;
    PatternConfig pattern_config;
    bool offline_ir_mode = false;
    bool offline_invert_ir = false;

    {
      std::unique_lock<std::mutex> lock(offline_detection_task_.mutex);
      offline_detection_task_.cv.wait(lock, [&]() {
        return offline_detection_task_.stop_requested || offline_detection_task_.has_request;
      });

      if (offline_detection_task_.stop_requested) {
        break;
      }

      request_id = offline_detection_task_.pending_request_id;
      frame_bgr = offline_detection_task_.pending_frame_bgr;
      pattern_config = offline_detection_task_.pending_pattern_config;
      offline_ir_mode = offline_detection_task_.pending_ir_mode;
      offline_invert_ir = offline_detection_task_.pending_invert_ir;
      offline_detection_task_.pending_frame_bgr.release();
      offline_detection_task_.has_request = false;
      offline_detection_task_.busy = true;
      offline_detection_task_.progress_message = "Detecting pattern on original image...";
    }

    OfflineFrameAnalysis analysis =
        analyzeOfflineFrame(frame_bgr, pattern_config, offline_ir_mode, offline_invert_ir);
    const std::string result_status = analysis.found ? "Patron detectado" : "Patron no detectado";

    {
      std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
      offline_detection_task_.preview_bgr = std::move(analysis.preview_bgr);
      offline_detection_task_.detection = std::move(analysis.detection);
      offline_detection_task_.image_size = analysis.image_size;
      offline_detection_task_.found = analysis.found;
      offline_detection_task_.result_status = result_status;
      offline_detection_task_.result_request_id = request_id;
      offline_detection_task_.result_ready = true;
      offline_detection_task_.busy = false;
      if (!offline_detection_task_.has_request) {
        offline_detection_task_.progress_message.clear();
      }
    }
  }

  std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
  offline_detection_task_.busy = false;
  offline_detection_task_.has_request = false;
  offline_detection_task_.progress_message.clear();
}

void ImGuiApp::queueOfflineDetection(const cv::Mat& frame_bgr, const PatternConfig& pattern_config,
                                     bool offline_ir_mode, bool offline_invert_ir,
                                     const std::string& status_message) {
  if (frame_bgr.empty()) {
    invalidateOfflineDetection();
    return;
  }

  startOfflineDetectionWorker();

  offline_detection_state_ = OfflineDetectionState::Detecting;
  offline_detection_ = DetectionResult{};
  offline_found_ = false;
  offline_metrics_ = SampleMetrics{};
  offline_detection_request_id_ += 1;
  if (!status_message.empty()) {
    offline_status_ = status_message;
  }

  {
    std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
    offline_detection_task_.pending_request_id = offline_detection_request_id_;
    offline_detection_task_.pending_frame_bgr = frame_bgr;
    offline_detection_task_.pending_pattern_config = pattern_config;
    offline_detection_task_.pending_ir_mode = offline_ir_mode;
    offline_detection_task_.pending_invert_ir = offline_invert_ir;
    offline_detection_task_.has_request = true;
    offline_detection_task_.result_ready = false;
    offline_detection_task_.progress_message = "Detecting pattern on original image...";
  }

  offline_detection_task_.cv.notify_one();
}

void ImGuiApp::invalidateOfflineDetection() {
  offline_detection_request_id_ += 1;
  offline_detection_state_ = OfflineDetectionState::Idle;
  offline_detection_ = DetectionResult{};
  offline_found_ = false;
  offline_metrics_ = SampleMetrics{};

  std::lock_guard<std::mutex> lock(offline_detection_task_.mutex);
  offline_detection_task_.has_request = false;
  offline_detection_task_.result_ready = false;
  offline_detection_task_.pending_frame_bgr.release();
  offline_detection_task_.preview_bgr.release();
  offline_detection_task_.detection = DetectionResult{};
  offline_detection_task_.image_size = cv::Size();
  offline_detection_task_.found = false;
  offline_detection_task_.result_status.clear();
}

bool ImGuiApp::requestOfflineImageLoad(int index) {
  bool is_running = false;
  {
    std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
    is_running = offline_image_frame_loader_.running;
  }
  if (is_running) {
    offline_status_ = "Cargando imagen...";
    return false;
  }

  if (index < 0 || index >= static_cast<int>(offline_image_files_.size())) {
    offline_status_ = "Indice de imagen fuera de rango";
    return false;
  }

  if (offline_image_frame_loader_.worker.joinable()) {
    offline_image_frame_loader_.worker.join();
  }

  invalidateOfflineDetection();
  const bool offline_ir_mode = offline_ir_mode_;
  const bool offline_invert_ir = offline_invert_ir_;
  const ImGuiApp::OfflineImageEncoding offline_image_encoding = offline_image_encoding_;
  const std::string target_path = offline_image_files_[static_cast<size_t>(index)];
  offline_status_ = "Cargando imagen...";

  {
    std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
    offline_image_frame_loader_.running = true;
    offline_image_frame_loader_.preview_ready = false;
    offline_image_frame_loader_.result_ready = false;
    offline_image_frame_loader_.show_popup = true;
    offline_image_frame_loader_.target_index = index;
    offline_image_frame_loader_.target_path = target_path;
    offline_image_frame_loader_.progress_message = "Reading image and building preview...";
    offline_image_frame_loader_.frame_bgr.release();
    offline_image_frame_loader_.preview_bgr.release();
    offline_image_frame_loader_.detection = DetectionResult{};
    offline_image_frame_loader_.image_size = cv::Size();
    offline_image_frame_loader_.found = false;
    offline_image_frame_loader_.result_status.clear();
  }

  offline_image_frame_loader_.worker = std::thread([this, target_path, index, offline_ir_mode, offline_invert_ir,
                                                    offline_image_encoding]() {
    cv::Mat image;
    cv::Mat preview_bgr;
    cv::Size image_size;
    std::string result_status;

    if (loadOfflineImageForEncoding(target_path, offline_image_encoding, image, result_status)) {
      preview_bgr = makeOfflineDisplayPreview(image, offline_ir_mode, offline_invert_ir);
      image_size = image.size();
      if (result_status.empty()) {
        result_status = "Imagen cargada";
      }
    }

    std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
    offline_image_frame_loader_.frame_bgr = std::move(image);
    offline_image_frame_loader_.preview_bgr = std::move(preview_bgr);
    offline_image_frame_loader_.image_size = image_size;
    offline_image_frame_loader_.target_index = index;
    offline_image_frame_loader_.progress_message = result_status.empty() ? "Load complete" : result_status;
    offline_image_frame_loader_.result_status = std::move(result_status);
    offline_image_frame_loader_.running = false;
    offline_image_frame_loader_.result_ready = true;
  });

  return true;
}

void ImGuiApp::refreshOfflineImageList() {
  bool is_running = false;
  {
    std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
    is_running = offline_image_loader_.running;
  }
  if (is_running) {
    offline_status_ = "Cargando lista de imagenes...";
    return;
  }

  if (offline_image_loader_.worker.joinable()) {
    offline_image_loader_.worker.join();
  }

  invalidateOfflineDetection();
  offline_found_ = false;
  offline_detection_ = DetectionResult{};
  offline_metrics_ = SampleMetrics{};
  offline_image_size_ = cv::Size();
  offline_preview_frame_bgr_.release();
  offline_status_ = "Cargando lista de imagenes...";

  const std::string target_dir = offline_images_dir_;
  {
    std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
    offline_image_loader_.running = true;
    offline_image_loader_.result_ready = false;
    offline_image_loader_.show_popup = true;
    offline_image_loader_.target_dir = target_dir;
    offline_image_loader_.progress_message = "Scanning image files...";
    offline_image_loader_.scanned_entries = 0;
    offline_image_loader_.found_images = 0;
    offline_image_loader_.files.clear();
    offline_image_loader_.result_status.clear();
  }

  offline_image_loader_.worker = std::thread([this, target_dir]() {
    std::vector<std::string> loaded_files;
    size_t scanned_entries = 0;
    size_t found_images = 0;
    std::string result_status;

    auto publish_progress = [&](const std::string& message) {
      std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
      offline_image_loader_.progress_message = message;
      offline_image_loader_.scanned_entries = scanned_entries;
      offline_image_loader_.found_images = found_images;
    };

    if (target_dir.empty()) {
      result_status = "Ruta de carpeta vacia";
    } else {
      fs::path dir(target_dir);
      std::error_code ec;
      const bool exists = fs::exists(dir, ec);
      if (ec || !exists || !fs::is_directory(dir, ec)) {
        result_status = "Carpeta no encontrada";
      } else {
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
          ++scanned_entries;
          const fs::directory_entry& entry = *it;
          std::error_code entry_ec;
          if (entry.is_regular_file(entry_ec) && !entry_ec) {
            const fs::path path = entry.path();
            if (isSupportedImageExtension(path)) {
              loaded_files.push_back(path.string());
              ++found_images;
            }
          }

          if ((scanned_entries % 64) == 0) {
            publish_progress("Scanning image files...");
          }
        }

        if (ec) {
          result_status = "No se pudo leer la carpeta";
        } else {
          std::sort(loaded_files.begin(), loaded_files.end());
          if (loaded_files.empty()) {
            result_status = "No se encontraron imagenes compatibles";
          } else {
            result_status = "Imagenes cargadas: " + std::to_string(loaded_files.size());
          }
        }
      }
    }

    std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
    offline_image_loader_.progress_message = result_status.empty() ? "Loading complete" : result_status;
    offline_image_loader_.scanned_entries = scanned_entries;
    offline_image_loader_.found_images = found_images;
    offline_image_loader_.files = std::move(loaded_files);
    offline_image_loader_.result_status = std::move(result_status);
    offline_image_loader_.running = false;
    offline_image_loader_.result_ready = true;
  });
}

bool ImGuiApp::loadOfflineImage(int index) {
  return requestOfflineImageLoad(index);
}

bool ImGuiApp::openOfflineVideo() {
  bool is_running = false;
  {
    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    is_running = offline_video_loader_.running;
  }
  if (is_running) {
    offline_status_ = "Abriendo video...";
    return false;
  }

  if (offline_video_loader_.worker.joinable()) {
    offline_video_loader_.worker.join();
  }

  invalidateOfflineDetection();
  offline_video_capture_.release();
  offline_video_open_ = false;
  offline_video_frame_index_ = -1;

  if (offline_video_path_.empty()) {
    offline_status_ = "Ruta de video vacia";
    return false;
  }

  offline_status_ = "Abriendo video...";
  const std::string target_path = offline_video_path_;
  {
    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    offline_video_loader_.running = true;
    offline_video_loader_.result_ready = false;
    offline_video_loader_.show_popup = true;
    offline_video_loader_.open_succeeded = false;
    offline_video_loader_.target_path = target_path;
    offline_video_loader_.progress_message = "Opening video stream...";
    offline_video_loader_.result_status.clear();
  }

  offline_video_loader_.worker = std::thread([this, target_path]() {
    std::string result_status;
    bool open_succeeded = false;

    if (target_path.empty()) {
      result_status = "Ruta de video vacia";
    } else {
      offline_video_capture_.release();
      if (offline_video_capture_.open(target_path)) {
        open_succeeded = true;
        result_status = "Video abierto";
      } else {
        result_status = "No se pudo abrir el video";
      }
    }

    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    offline_video_loader_.progress_message = result_status.empty() ? "Open complete" : result_status;
    offline_video_loader_.open_succeeded = open_succeeded;
    offline_video_loader_.result_status = std::move(result_status);
    offline_video_loader_.running = false;
    offline_video_loader_.result_ready = true;
  });

  return true;
}

bool ImGuiApp::stepOfflineVideo() {
  {
    std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
    if (offline_video_loader_.running) {
      offline_status_ = "Abriendo video...";
      return false;
    }
  }

  if (!offline_video_open_) {
    if (!openOfflineVideo()) return false;
    return false;
  }

  cv::Mat frame;
  if (!offline_video_capture_.read(frame) || frame.empty()) {
    offline_status_ = "Fin de video o frame invalido";
    return false;
  }

  offline_video_frame_index_ += 1;
  offline_frame_bgr_ = frame;
  offline_image_size_ = frame.size();
  offline_seq_ += 1;
  offline_timestamp_ns_ = monotonicNowNs();
  processOfflineFrame();
  return true;
}

void ImGuiApp::processOfflineFrame() {
  if (offline_frame_bgr_.empty()) {
    invalidateOfflineDetection();
    offline_image_size_ = cv::Size();
    offline_preview_frame_bgr_.release();
    return;
  }

  PatternConfig pattern_config;
  {
    std::lock_guard<std::mutex> lock(pattern_mutex_);
    pattern_config = pattern_config_;
  }
  offline_preview_frame_bgr_ = makeOfflineDisplayPreview(offline_frame_bgr_, offline_ir_mode_, offline_invert_ir_);
  offline_image_size_ = offline_frame_bgr_.size();
  const std::string status_message =
      calibration_input_mode_ == CalibrationInputMode::VideoFile
          ? "Frame cargado, detectando patron..."
          : "Imagen cargada, detectando patron...";
  queueOfflineDetection(offline_frame_bgr_, pattern_config, offline_ir_mode_, offline_invert_ir_,
                        status_message);
}

void ImGuiApp::drawCalibrationSourceControls() {
  ImGui::TextUnformatted("Calibration source");
  const char* input_mode_labels[] = {"Live camera", "Image folder", "Video file"};
  const char* input_mode_tooltips[] = {
      "Use a connected live stream and detect the pattern frame by frame.",
      "Load still images from a directory and browse them offline.",
      "Open a recorded video file and step through frames offline.",
  };
  int input_mode_index = static_cast<int>(calibration_input_mode_);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (comboWithTooltips("##calib_input_mode", &input_mode_index, input_mode_labels,
                        IM_ARRAYSIZE(input_mode_labels), input_mode_tooltips)) {
    calibration_input_mode_ = static_cast<CalibrationInputMode>(input_mode_index);
    if (calibration_input_mode_ == CalibrationInputMode::ImageFolder && offline_image_files_.empty()) {
      refreshOfflineImageList();
    }
    if (calibration_input_mode_ == CalibrationInputMode::VideoFile && !offline_video_open_) {
      openOfflineVideo();
    }
  }
  itemTooltip("Select live camera stream or reproducible offline input.");

  if (calibration_input_mode_ == CalibrationInputMode::ImageFolder) {
    bool image_folder_loading = false;
    bool image_frame_loading = false;
    bool image_folder_picker_running = false;
    {
      std::lock_guard<std::mutex> lock(offline_image_loader_.mutex);
      image_folder_loading = offline_image_loader_.running;
    }
    {
      std::lock_guard<std::mutex> lock(offline_image_frame_loader_.mutex);
      image_frame_loading = offline_image_frame_loader_.running;
    }
    {
      std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
      image_folder_picker_running =
          offline_path_picker_.running &&
          offline_path_picker_.target == OfflinePathPickerTarget::ImageFolder;
    }
    const bool image_folder_busy = image_folder_loading || image_frame_loading || image_folder_picker_running;

    ImGui::TextUnformatted("Image folder");
    const ImGuiStyle& style = ImGui::GetStyle();
    const float browse_btn_size = ImGui::GetFrameHeight();
    const float input_width =
        std::max(1.0f, ImGui::GetContentRegionAvail().x - browse_btn_size - style.ItemInnerSpacing.x);
    if (image_folder_busy) ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(input_width);
    ImGui::InputText("##offline_images_dir", &offline_images_dir_);
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    if (iconButton("##offline_images_dir_browse", UiIcon::Folder, browse_btn_size, defaultIconColor())) {
      requestOfflinePathPicker(OfflinePathPickerTarget::ImageFolder, true, offline_images_dir_);
    }
    itemTooltip("Browse for the folder that contains calibration images.");
    ImGui::TextUnformatted("Image encoding");
    const char* offline_image_encoding_labels[] = {
        "Standard image",
        "Bayer RG",
        "Bayer BG",
        "Bayer GB",
        "Bayer GR",
    };
    const char* offline_image_encoding_tooltips[] = {
        "Read the file as a normal image already encoded in grayscale or color.",
        "Decode the file as raw Bayer RG mosaic before preview and detection.",
        "Decode the file as raw Bayer BG mosaic before preview and detection.",
        "Decode the file as raw Bayer GB mosaic before preview and detection.",
        "Decode the file as raw Bayer GR mosaic before preview and detection.",
    };
    int offline_image_encoding_index = static_cast<int>(offline_image_encoding_);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (comboWithTooltips("##offline_image_encoding", &offline_image_encoding_index,
                          offline_image_encoding_labels, IM_ARRAYSIZE(offline_image_encoding_labels),
                          offline_image_encoding_tooltips)) {
      offline_image_encoding_ = static_cast<OfflineImageEncoding>(offline_image_encoding_index);
      if (!offline_image_files_.empty()) {
        const int reload_index = std::max(0, offline_image_index_);
        loadOfflineImage(reload_index);
      }
    }
    itemTooltip("How to decode images from the folder before preview and pattern detection.");
    const float btn_w = (ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    if (ImGui::Button("Load list", ImVec2(btn_w, 0))) {
      refreshOfflineImageList();
    }
    ImGui::SameLine();
    const bool can_prev = !offline_image_files_.empty() && offline_image_index_ > 0;
    if (!can_prev) ImGui::BeginDisabled();
    if (ImGui::Button("Prev", ImVec2(btn_w, 0))) {
      loadOfflineImage(offline_image_index_ - 1);
    }
    if (!can_prev) ImGui::EndDisabled();
    ImGui::SameLine();
    const bool can_next =
        !offline_image_files_.empty() && offline_image_index_ >= 0 &&
        offline_image_index_ + 1 < static_cast<int>(offline_image_files_.size());
    if (!can_next) ImGui::BeginDisabled();
    if (ImGui::Button("Next", ImVec2(btn_w, 0))) {
      loadOfflineImage(offline_image_index_ + 1);
    }
    if (!can_next) ImGui::EndDisabled();
    if (image_folder_busy) ImGui::EndDisabled();

    if (!offline_status_.empty()) {
      ImGui::TextWrapped("Offline status: %s", offline_status_.c_str());
    }
    if (!offline_image_files_.empty() && offline_image_index_ >= 0) {
      ImGui::Text("Image %d / %d", offline_image_index_ + 1, static_cast<int>(offline_image_files_.size()));
      ImGui::TextWrapped("%s",
                         fs::path(offline_image_files_[static_cast<size_t>(offline_image_index_)])
                             .filename()
                             .string()
                             .c_str());
    }
  } else if (calibration_input_mode_ == CalibrationInputMode::VideoFile) {
    bool video_loading = false;
    bool video_picker_running = false;
    {
      std::lock_guard<std::mutex> lock(offline_video_loader_.mutex);
      video_loading = offline_video_loader_.running;
    }
    {
      std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
      video_picker_running =
          offline_path_picker_.running &&
          offline_path_picker_.target == OfflinePathPickerTarget::VideoFile;
    }
    const bool video_busy = video_loading || video_picker_running;

    ImGui::TextUnformatted("Video file");
    const ImGuiStyle& style = ImGui::GetStyle();
    const float browse_btn_size = ImGui::GetFrameHeight();
    const float input_width =
        std::max(1.0f, ImGui::GetContentRegionAvail().x - browse_btn_size - style.ItemInnerSpacing.x);
    if (video_busy) ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(input_width);
    ImGui::InputText("##offline_video_path", &offline_video_path_);
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    if (iconButton("##offline_video_path_browse", UiIcon::Folder, browse_btn_size, defaultIconColor())) {
      requestOfflinePathPicker(OfflinePathPickerTarget::VideoFile, false, offline_video_path_);
    }
    itemTooltip("Browse for the video file used as calibration input.");
    const float btn_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    if (ImGui::Button("Open video", ImVec2(btn_w, 0))) {
      openOfflineVideo();
    }
    ImGui::SameLine();
    bool can_step = offline_video_open_ && !video_busy;
    if (!can_step) ImGui::BeginDisabled();
    if (ImGui::Button("Next frame", ImVec2(btn_w, 0))) {
      stepOfflineVideo();
    }
    if (!can_step) ImGui::EndDisabled();
    if (video_busy) ImGui::EndDisabled();

    if (!offline_status_.empty()) {
      ImGui::TextWrapped("Offline status: %s", offline_status_.c_str());
    }
    if (offline_video_frame_index_ >= 0) {
      ImGui::Text("Frame index: %d", offline_video_frame_index_);
    }
  }

  if (calibration_input_mode_ != CalibrationInputMode::Live) {
    bool offline_preprocess_changed = false;
    ImGui::TextUnformatted("Offline IR mode (mono preprocess)");
    if (ImGui::Checkbox("##offline_ir_mode", &offline_ir_mode_)) {
      offline_preprocess_changed = true;
    }
    itemTooltip("Convert offline frame to grayscale before pattern detection.");
    ImGui::TextUnformatted("Offline invert IR image");
    if (ImGui::Checkbox("##offline_invert_ir", &offline_invert_ir_)) {
      offline_preprocess_changed = true;
    }
    itemTooltip("Invert offline grayscale image before detection for reversed thermal polarity.");

    if (offline_preprocess_changed && !offline_frame_bgr_.empty()) {
      processOfflineFrame();
    }
  }
}

void ImGuiApp::drawCameraSlot(size_t index, CameraSlot& slot) {
  CameraInfo current = slot.device.currentInfo();

  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - 22.0f);
  if (iconButton(("cfg_btn##" + std::to_string(index)).c_str(), UiIcon::Gear, 18.0f,
                 defaultIconColor())) {
    slot.open_config_dialog = true;
    if (slot.device.supportsFeatures()) {
      queueFeatureRefresh(slot);
    }
  }

  std::vector<BackendType> backends = availableBackends();
  std::vector<std::string> labels;
  std::vector<const char*> label_ptrs;
  labels.reserve(backends.size());
  label_ptrs.reserve(backends.size());
  int backend_index = 0;
  for (size_t i = 0; i < backends.size(); ++i) {
    labels.push_back(backendLabel(backends[i]));
    label_ptrs.push_back(labels.back().c_str());
    if (backends[i] == slot.device.backendType()) {
      backend_index = static_cast<int>(i);
    }
  }
  std::vector<const char*> backend_tooltips;
  backend_tooltips.reserve(backends.size());
  for (BackendType backend : backends) {
    switch (backend) {
      case BackendType::Aravis:
        backend_tooltips.push_back("Native GenICam/GigE Vision backend with feature access for supported cameras.");
        break;
      case BackendType::GenTL:
        backend_tooltips.push_back("Use a vendor GenTL producer via a .cti file to access industrial cameras.");
        break;
      case BackendType::OpenCV:
        backend_tooltips.push_back("Generic OpenCV VideoCapture backend for webcams and simple USB sources.");
        break;
      case BackendType::OpenCVGStreamer:
        backend_tooltips.push_back("Open a custom GStreamer pipeline through OpenCV, useful for network streams.");
        break;
      default:
        backend_tooltips.push_back("");
        break;
    }
  }

  const std::string backend_combo_id = "##backend_combo_" + std::to_string(index);
  ImGui::TextUnformatted("Backend");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (comboWithTooltips(backend_combo_id.c_str(), &backend_index, label_ptrs.data(),
                        static_cast<int>(label_ptrs.size()), backend_tooltips.data())) {
    stopFeatureWorker(slot);
    const BackendType selected_backend = backends[backend_index];
    slot.device.setBackend(selected_backend);
    if (selected_backend == BackendType::OpenCVGStreamer) {
      slot.use_gstreamer_pipeline = true;
    } else if (selected_backend == BackendType::OpenCV) {
      slot.use_gstreamer_pipeline = false;
    }
    if (selected_backend == BackendType::OpenCV || selected_backend == BackendType::OpenCVGStreamer) {
      slot.device.setBackendOption("gstreamer_enabled",
                                   selected_backend == BackendType::OpenCVGStreamer ? "true" : "false");
      slot.device.setBackendOption("gstreamer_pipeline", slot.gstreamer_pipeline);
    }
    clearFeatureCache(slot);
    refreshDevices(slot);
  }

  const bool is_opencv_backend = slot.device.backendType() == BackendType::OpenCV;
  const bool is_opencv_gstreamer_backend = slot.device.backendType() == BackendType::OpenCVGStreamer;
  const bool is_any_opencv_backend = is_opencv_backend || is_opencv_gstreamer_backend;

  if (slot.device.backendType() == BackendType::GenTL) {
    bool cti_picker_running = false;
    {
      std::lock_guard<std::mutex> lock(offline_path_picker_.mutex);
      cti_picker_running =
          offline_path_picker_.running &&
          offline_path_picker_.target == OfflinePathPickerTarget::GenTLCtiFile &&
          offline_path_picker_.slot_index == static_cast<int>(index);
    }
    const std::string cti_input_id = "##cti_path_" + std::to_string(index);
    const std::string cti_browse_id = "##cti_path_browse_" + std::to_string(index);
    ImGui::TextUnformatted("CTI path");
    const ImGuiStyle& style = ImGui::GetStyle();
    const float browse_btn_size = ImGui::GetFrameHeight();
    const float input_width =
        std::max(1.0f, ImGui::GetContentRegionAvail().x - browse_btn_size - style.ItemInnerSpacing.x);
    if (cti_picker_running) ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(input_width);
    ImGui::InputText(cti_input_id.c_str(), &slot.gentl_cti_path);
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    if (iconButton(cti_browse_id.c_str(), UiIcon::Folder, browse_btn_size, defaultIconColor())) {
      requestOfflinePathPicker(OfflinePathPickerTarget::GenTLCtiFile, false, slot.gentl_cti_path,
                               static_cast<int>(index));
    }
    itemTooltip("Browse for the GenTL producer (.cti) file.");
    if (cti_picker_running) ImGui::EndDisabled();
    if (ImGui::Button("Load CTI", ImVec2(-1, 0))) {
      slot.device.setBackendOption("cti_path", slot.gentl_cti_path);
      refreshDevices(slot);
    }
  }
  if (slot.device.backendType() == BackendType::Aravis) {
    bool swap_rb = slot.bayer_swap_rb.load(std::memory_order_relaxed);
    if (ImGui::Checkbox(("Swap Bayer R/B##" + std::to_string(index)).c_str(), &swap_rb)) {
      slot.bayer_swap_rb.store(swap_rb, std::memory_order_relaxed);
    }
  }
  if (is_opencv_gstreamer_backend) {
    const std::string gst_input_id = "##gst_pipeline_" + std::to_string(index);
    ImGui::TextUnformatted("GStreamer pipeline");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputText(gst_input_id.c_str(), &slot.gstreamer_pipeline);
    itemTooltip("Example SRT: srtsrc uri=\"srt://127.0.0.1:9000?mode=caller&latency=120\" ! tsdemux ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false");
    if (ImGui::Button("Apply OpenCV source", ImVec2(-1, 0))) {
      slot.device.setBackendOption("gstreamer_enabled", "true");
      slot.device.setBackendOption("gstreamer_pipeline", slot.gstreamer_pipeline);
    }
    ImGui::TextWrapped("Device selection is disabled in OpenCV GStreamer mode.");
  }

  ImGui::Checkbox(("IR mode (mono preprocess)##" + std::to_string(index)).c_str(), &slot.ir_mode);
  itemTooltip("Convert input to grayscale preview/detection, useful for IR streams.");
  ImGui::Checkbox(("Invert IR image##" + std::to_string(index)).c_str(), &slot.invert_ir);
  itemTooltip("Invert image before pattern detection (common in some IR setups).");

  if (!is_opencv_gstreamer_backend) {
    if (ImGui::Button("Refresh devices", ImVec2(-1, 0))) {
      refreshDevices(slot);
    }

    const std::string devices_list_id = "##devices_list_" + std::to_string(index);
    if (ImGui::BeginListBox(devices_list_id.c_str(), ImVec2(-FLT_MIN, 170.0f))) {
      for (size_t i = 0; i < slot.devices.size(); ++i) {
        const auto& dev = slot.devices[i];
        bool selected = slot.selected_device == static_cast<int>(i);
        bool is_current = !current.id.empty() && dev.id == current.id;
        bool in_use_other = isDeviceInUse(dev.id, &slot);
        bool disabled = in_use_other && !is_current;

        std::string label = dev.label;
        if (in_use_other && !is_current) label += " [in use]";

        ImGui::PushID(static_cast<int>(i));
        if (disabled) ImGui::BeginDisabled();
        if (ImGui::Selectable(label.c_str(), selected)) {
          slot.selected_device = static_cast<int>(i);
        }
        if (disabled) ImGui::EndDisabled();

        ImVec2 item_min = ImGui::GetItemRectMin();
        ImVec2 item_max = ImGui::GetItemRectMax();
        const float icon_size = 14.0f;
        ImVec2 icon_pos(item_max.x - icon_size - 6.0f,
                        item_min.y + (item_max.y - item_min.y - icon_size) * 0.5f);
        ImU32 icon_color = IM_COL32(170, 170, 170, 255);
        if (is_current) {
          icon_color = IM_COL32(125, 210, 45, 255);
        } else if (in_use_other) {
          icon_color = IM_COL32(235, 180, 60, 255);
        }
        drawLinkIcon(ImGui::GetWindowDrawList(), icon_pos, icon_size, icon_color);
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }
  }

  bool has_selection = slot.selected_device >= 0 && slot.selected_device < static_cast<int>(slot.devices.size());
  CameraInfo selected = has_selection ? slot.devices[slot.selected_device] : CameraInfo{};
  const bool use_gstreamer_target = is_opencv_gstreamer_backend;
  auto isBlank = [](const std::string& text) {
    return text.empty() || std::all_of(text.begin(), text.end(), [](unsigned char c) {
             return std::isspace(c) != 0;
           });
  };

  CameraInfo target = selected;
  if (use_gstreamer_target && !isBlank(slot.gstreamer_pipeline)) {
    target.id = std::string(kOpenCVGStreamerDeviceIdPrefix) + ":" + slot.gstreamer_pipeline;
    target.label = "OpenCV GStreamer pipeline";
  }

  const bool has_target = use_gstreamer_target ? !target.id.empty() : has_selection;
  bool target_in_use = false;
  bool target_is_current = false;
  if (has_target) {
    if (use_gstreamer_target) {
      const int existing_target = findSlotByDevice(target.id);
      target_in_use = existing_target >= 0 && existing_target != static_cast<int>(index);
      target_is_current = !current.id.empty() && current.id == target.id;
    } else {
      target_in_use = isDeviceInUse(target.id, &slot);
      target_is_current = !current.id.empty() && target.id == current.id;
    }
  }

  bool can_connect = has_target && !target_in_use;
  CameraState state = slot.device.state();

  if (!can_connect || (target_is_current && state != CameraState::Disconnected && state != CameraState::Error)) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Connect", ImVec2(-1, 0))) {
    if (use_gstreamer_target) {
      slot.device.setBackendOption("gstreamer_enabled", "true");
      slot.device.setBackendOption("gstreamer_pipeline", slot.gstreamer_pipeline);
    }

    int existing = findSlotByDevice(target.id);
    if (existing >= 0 && existing != static_cast<int>(index)) {
      active_slot_ = existing;
    } else if (state == CameraState::Disconnected || state == CameraState::Error || current.id.empty()) {
      stopFeatureWorker(slot);
      if (slot.device.connect(target)) {
        clearFeatureCache(slot);
      }
    } else {
      addCameraSlot();
      CameraSlot* new_slot = activeSlot();
      if (new_slot) {
        new_slot->device.setBackend(slot.device.backendType());
        new_slot->gentl_cti_path = slot.gentl_cti_path;
        new_slot->use_gstreamer_pipeline = slot.use_gstreamer_pipeline;
        new_slot->gstreamer_pipeline = slot.gstreamer_pipeline;
        new_slot->ir_mode = slot.ir_mode;
        new_slot->invert_ir = slot.invert_ir;
        if (slot.device.backendType() == BackendType::GenTL && !slot.gentl_cti_path.empty()) {
          new_slot->device.setBackendOption("cti_path", slot.gentl_cti_path);
        }
        if (is_any_opencv_backend) {
          new_slot->device.setBackendOption("gstreamer_enabled",
                                            slot.device.backendType() == BackendType::OpenCVGStreamer ? "true"
                                                                                                      : "false");
          new_slot->device.setBackendOption("gstreamer_pipeline", new_slot->gstreamer_pipeline);
        }
        refreshDevices(*new_slot);
        if (use_gstreamer_target) {
          CameraInfo new_target = target;
          new_target.id = std::string(kOpenCVGStreamerDeviceIdPrefix) + ":" + new_slot->gstreamer_pipeline;
          new_target.label = "OpenCV GStreamer pipeline";
          if (new_slot->device.connect(new_target)) {
            clearFeatureCache(*new_slot);
          }
        } else {
          for (size_t i = 0; i < new_slot->devices.size(); ++i) {
            if (new_slot->devices[i].id == target.id) {
              new_slot->selected_device = static_cast<int>(i);
              break;
            }
          }
          if (new_slot->selected_device >= 0 &&
              new_slot->selected_device < static_cast<int>(new_slot->devices.size()) &&
              new_slot->device.connect(new_slot->devices[new_slot->selected_device])) {
            clearFeatureCache(*new_slot);
          }
        }
      }
    }
  }
  if (!can_connect || (target_is_current && state != CameraState::Disconnected && state != CameraState::Error)) {
    ImGui::EndDisabled();
  }

  const bool can_disconnect = state == CameraState::Connected || state == CameraState::Streaming;
  if (!can_disconnect) ImGui::BeginDisabled();
  if (ImGui::Button("Disconnect", ImVec2(-1, 0))) {
    slot.device.stopStreaming();
    stopFeatureWorker(slot);
    slot.device.disconnect();
    clearFeatureCache(slot);
    refreshDevices(slot);
  }
  if (!can_disconnect) ImGui::EndDisabled();

  const char* state_label = "Unknown";
  switch (state) {
    case CameraState::Disconnected:
      state_label = "Disconnected";
      break;
    case CameraState::Connected:
      state_label = "Connected";
      break;
    case CameraState::Streaming:
      state_label = "Streaming";
      break;
    case CameraState::Error:
      state_label = "Error";
      break;
  }
  ImGui::Text("State: %s", state_label);
  if (!slot.device.status().empty()) {
    ImGui::TextWrapped("Status: %s", slot.device.status().c_str());
  }
  if (state == CameraState::Streaming) {
    double preview_fps = 0.0;
    uint64_t processed_frames = 0;
    {
      std::lock_guard<std::mutex> frame_lock(slot.frame_mutex);
      preview_fps = slot.processing_fps;
      processed_frames = slot.processed_frames;
    }
    ImGui::Text("Capture FPS: %.2f", slot.device.captureFps());
    ImGui::Text("Preview FPS: %.2f", preview_fps);
    ImGui::Text("Frames: %llu  Grab fails: %llu",
                static_cast<unsigned long long>(slot.device.capturedFrames()),
                static_cast<unsigned long long>(slot.device.captureFailures()));
    ImGui::Text("Processed frames: %llu", static_cast<unsigned long long>(processed_frames));
  }

  (void)index;
}

void ImGuiApp::drawConfigDialog(size_t index, CameraSlot& slot) {
  std::string popup_id = "Device properties##slot_" + std::to_string(index);
  if (slot.open_config_dialog) {
    ImGui::OpenPopup(popup_id.c_str());
    slot.open_config_dialog = false;
  }

  bool open = true;
  ImGui::SetNextWindowSize(ImVec2(920.0f, 620.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(560.0f, 380.0f), ImVec2(FLT_MAX, FLT_MAX));
  if (ImGui::BeginPopupModal(popup_id.c_str(), &open, ImGuiWindowFlags_NoCollapse)) {
    CameraInfo info = slot.device.currentInfo();
    if (!info.label.empty()) {
      ImGui::Text("Device: %s", info.label.c_str());
      ImGui::Separator();
    }

    if (!slot.device.supportsFeatures()) {
      ImGui::Text("No configurable properties for this backend/device.");
    } else {
      ImGui::PushID(static_cast<int>(index));
      drawFeaturePanel(slot);
      ImGui::PopID();
    }

    if (ImGui::Button("Close")) {
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }
}

void ImGuiApp::drawCalibrationPanel(CameraSlot* active) {
  ImGui::Text("Calibration");

  std::string active_label = "None";
  if (active) {
    CameraInfo current = active->device.currentInfo();
    if (!current.label.empty()) {
      active_label = current.label;
    } else {
      active_label = "Tab " + std::to_string(active_slot_ + 1);
    }
  }
  if (calibration_input_mode_ == CalibrationInputMode::Live) {
    ImGui::TextWrapped("Active camera: %s", active_label.c_str());
  } else {
    ImGui::TextWrapped("Active source: offline input");
  }
  PatternConfig pattern_config;
  {
    std::lock_guard<std::mutex> lock(pattern_mutex_);
    pattern_config = pattern_config_;
  }

  DetectionResult live_detection;
  cv::Size live_image_size;
  cv::Mat live_last_frame;
  SampleMetrics live_metrics{};
  bool live_found = false;
  uint64_t live_ts_ns = 0;
  if (active) {
    std::lock_guard<std::mutex> lock(active->frame_mutex);
    live_detection = active->detection;
    live_image_size = active->last_image_size;
    live_last_frame = active->last_frame_raw;
    live_metrics = active->metrics;
    live_found = active->found;
    live_ts_ns = active->detection_timestamp_ns;
  }

  std::vector<PatternType> pattern_types;
  std::vector<const char*> pattern_labels;
  std::vector<const char*> pattern_tooltips;
  pattern_types.push_back(PatternType::ChessboardAuto);
  pattern_labels.push_back("Chessboard (auto)");
  pattern_tooltips.push_back("Try the sector-based detector first and automatically fall back to the classic detector if needed.");
  pattern_types.push_back(PatternType::Chessboard);
  pattern_labels.push_back("Chessboard (classic)");
  pattern_tooltips.push_back("Original OpenCV chessboard detector. More compatible in some cases, but usually slower on large images.");
  pattern_types.push_back(PatternType::ChessboardSB);
  pattern_labels.push_back("Chessboard (SB)");
  pattern_tooltips.push_back("Sector-based chessboard detector. Usually faster on large images and often more precise, but it can miss some boards.");
  pattern_types.push_back(PatternType::CirclesSymmetric);
  pattern_labels.push_back("Circles (symmetric)");
  pattern_tooltips.push_back("Detect a regular symmetric circles grid instead of a checkerboard.");
  pattern_types.push_back(PatternType::CirclesAsymmetric);
  pattern_labels.push_back("Circles (asymmetric)");
  pattern_tooltips.push_back("Detect an asymmetric circles grid with staggered rows. OpenCV is sensitive to the board geometry here; the common reference board is 4x11, and some layouts such as 6x8 may fail even when the circles are clean.");
#ifdef HAVE_OPENCV_ARUCO
  pattern_types.push_back(PatternType::Charuco);
  pattern_labels.push_back("ChArUco");
  pattern_tooltips.push_back("Hybrid ArUco plus chessboard target, robust to partial views and useful for precise calibration.");
#endif

  int pattern_index = 0;
  for (size_t i = 0; i < pattern_types.size(); ++i) {
    if (pattern_types[i] == pattern_config.type) {
      pattern_index = static_cast<int>(i);
      break;
    }
  }

  bool pattern_changed = false;
  ImGui::TextUnformatted("Pattern");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (comboWithTooltips("##pattern_type", &pattern_index, pattern_labels.data(),
                        static_cast<int>(pattern_labels.size()), pattern_tooltips.data())) {
    pattern_config.type = pattern_types[pattern_index];
    pattern_changed = true;
  }
  itemTooltip("Pattern geometry to detect.");

  const bool is_charuco_pattern = pattern_config.type == PatternType::Charuco;
  const bool is_circles_pattern =
      pattern_config.type == PatternType::CirclesSymmetric ||
      pattern_config.type == PatternType::CirclesAsymmetric;
  int cols = pattern_config.board_size.width;
  int rows = pattern_config.board_size.height;

  ImGui::TextUnformatted(is_charuco_pattern ? "Squares X" : (is_circles_pattern ? "Points per row" : "Columns"));
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::InputInt("##pattern_cols", &cols)) {
    cols = std::max(2, cols);
    pattern_config.board_size.width = cols;
    pattern_changed = true;
  }
  itemTooltip(is_charuco_pattern
                  ? "Number of chessboard squares across the ChArUco board width. This is not the number of inner corners or marker columns."
                  : (is_circles_pattern
                         ? "Number of circle centers in each row. For asymmetric grids, each staggered row still counts the same number of points. If you are using the standard OpenCV-style reference board, start with 4 points per row."
                         : "Number of internal corners/points along the board width."));

  ImGui::TextUnformatted(is_charuco_pattern ? "Squares Y" : "Rows");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::InputInt("##pattern_rows", &rows)) {
    rows = std::max(2, rows);
    pattern_config.board_size.height = rows;
    pattern_changed = true;
  }
  itemTooltip(is_charuco_pattern
                  ? "Number of chessboard squares across the ChArUco board height. This is not the number of inner corners or marker rows."
                  : (is_circles_pattern
                         ? "Number of rows of circle centers in the grid. For asymmetric grids, count every staggered row. If you are using the standard OpenCV-style reference board, start with 11 rows."
                         : "Number of internal corners/points along the board height."));

  ImGui::TextUnformatted(is_charuco_pattern ? "Square size" : "Size");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::InputFloat("##pattern_size", &pattern_config.square_size, 0.001f, 0.01f, "%.4f")) {
    pattern_config.square_size = std::max(0.0001f, pattern_config.square_size);
    pattern_changed = true;
  }
  itemTooltip(is_charuco_pattern
                  ? "Real size of one chessboard square in meters."
                  : "Real size of one square/cell in meters.");

#ifdef HAVE_OPENCV_ARUCO
  if (pattern_config.type == PatternType::Charuco) {
    const auto& dictionary_options = arucoDictionaryOptions();
    std::vector<const char*> dictionary_labels;
    std::vector<const char*> dictionary_tooltips;
    dictionary_labels.reserve(dictionary_options.size());
    dictionary_tooltips.reserve(dictionary_options.size());
    int dictionary_index = 0;
    for (size_t i = 0; i < dictionary_options.size(); ++i) {
      dictionary_labels.push_back(dictionary_options[i].label);
      dictionary_tooltips.push_back(dictionary_options[i].tooltip);
      if (dictionary_options[i].value == pattern_config.aruco_dictionary) {
        dictionary_index = static_cast<int>(i);
      }
    }

    ImGui::TextUnformatted("ArUco dictionary");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (comboWithTooltips("##charuco_dictionary", &dictionary_index,
                          dictionary_labels.data(), static_cast<int>(dictionary_labels.size()),
                          dictionary_tooltips.data())) {
      pattern_config.aruco_dictionary = dictionary_options[static_cast<size_t>(dictionary_index)].value;
      pattern_changed = true;
    }
    itemTooltip("Must match the dictionary used to generate the printed ChArUco board.");

    ImGui::TextUnformatted("Marker size");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputFloat("##charuco_marker_size", &pattern_config.marker_size, 0.001f, 0.01f, "%.4f")) {
      pattern_config.marker_size = std::max(0.0001f, pattern_config.marker_size);
      pattern_changed = true;
    }
    itemTooltip("Real marker side length in meters (ChArUco only).");
  }
#endif

  if (pattern_changed) {
    {
      std::lock_guard<std::mutex> lock(pattern_mutex_);
      pattern_config_ = pattern_config;
      ++pattern_revision_;
    }
    calibrator_.setPattern(pattern_config);
    stereo_calibrator_.setPattern(pattern_config);
    if (calibration_input_mode_ != CalibrationInputMode::Live && !offline_frame_bgr_.empty()) {
      processOfflineFrame();
    }
  }

  ImGui::TextUnformatted("Camera model");
  const char* camera_model_labels[] = {"Standard (pinhole)", "Fisheye"};
  const char* camera_model_tooltips[] = {
      "Use the standard pinhole model for conventional lenses.",
      "Use the fisheye model for very wide-angle lenses with strong distortion.",
  };
  int camera_model_index = (camera_model_ == CameraModel::Fisheye) ? 1 : 0;
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (comboWithTooltips("##camera_model", &camera_model_index, camera_model_labels,
                        IM_ARRAYSIZE(camera_model_labels), camera_model_tooltips)) {
    camera_model_ = (camera_model_index == 1) ? CameraModel::Fisheye : CameraModel::Pinhole;
    calibrator_.setCameraModel(camera_model_);
  }
  itemTooltip("Select intrinsic model: pinhole (standard) or fisheye.");

  if (camera_model_ == CameraModel::Pinhole) {
    ImGui::TextUnformatted("Pinhole distortion");
    const char* pinhole_labels[] = {"Plumb bob", "Rational polynomial"};
    const char* pinhole_tooltips[] = {
        "Classic radial and tangential distortion model used by most standard calibrations.",
        "Adds higher-order radial terms for lenses that need a more flexible pinhole distortion model.",
    };
    int pinhole_index = (pinhole_distortion_model_ == PinholeDistortionModel::RationalPolynomial) ? 1 : 0;
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (comboWithTooltips("##pinhole_distortion_model", &pinhole_index, pinhole_labels,
                          IM_ARRAYSIZE(pinhole_labels), pinhole_tooltips)) {
      pinhole_distortion_model_ =
          (pinhole_index == 1) ? PinholeDistortionModel::RationalPolynomial : PinholeDistortionModel::PlumbBob;
      calibrator_.setPinholeDistortionModel(pinhole_distortion_model_);
    }
    itemTooltip("Advanced model with higher-order radial terms.");

    ImGui::Checkbox("Target warp compensation##target_warp_compensation", &target_warp_compensation_);
    calibrator_.setTargetWarpCompensation(target_warp_compensation_);
    itemTooltip("Refine board geometry with calibrateCameraRO to reduce non-planar target bias.");
  }

  ImGui::Checkbox("Auto add samples##auto_add_samples", &auto_add_);
  itemTooltip("Automatically add a sample whenever a valid pattern is detected (live mode).");
  ImGui::Checkbox("Allow duplicates##allow_duplicates", &allow_duplicates_);
  itemTooltip("Allow very similar views to be added; disabling improves sample diversity.");
  ImGui::Checkbox("Save sample images##save_sample_images", &save_samples_);
  itemTooltip("Save each accepted sample image to disk.");
  ImGui::TextUnformatted("Samples dir");
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputText("##samples_dir", &samples_dir_);
  itemTooltip("Directory where sample images are stored when saving is enabled.");

  DetectionResult current_detection;
  cv::Size current_image_size;
  cv::Mat current_last_frame;
  SampleMetrics current_metrics{};
  bool current_found = false;
  bool current_detection_pending = false;
  uint64_t current_ts_ns = 0;

  if (calibration_input_mode_ == CalibrationInputMode::Live) {
    current_detection = live_detection;
    current_image_size = live_image_size;
    current_last_frame = live_last_frame;
    current_metrics = live_metrics;
    current_found = live_found;
    current_ts_ns = live_ts_ns;
  } else {
    current_detection = offline_detection_;
    current_image_size = offline_image_size_;
    current_last_frame = offline_frame_bgr_;
    current_metrics = offline_metrics_;
    current_found = offline_found_;
    current_detection_pending = offline_detection_state_ == OfflineDetectionState::Detecting;
    current_ts_ns = offline_timestamp_ns_;
  }

  ImGui::Text("Current: X=%.2f  Y=%.2f  Size=%.2f  Skew=%.2f",
              current_metrics.px, current_metrics.py, current_metrics.scale, current_metrics.skew);
  const char* detection_label = current_detection_pending ? "Detecting..." : (current_found ? "OK" : "No pattern");
  ImGui::Text("Detection: %s", detection_label);
  (void)current_ts_ns;

  bool calibration_busy = false;
  std::string calibration_progress_message;
  {
    std::lock_guard<std::mutex> lock(calibration_task_.mutex);
    calibration_busy = calibration_task_.running || calibration_task_.result_ready;
    calibration_progress_message = calibration_task_.progress_message;
  }

  bool can_add = current_found && !current_detection.corners.empty() && !current_image_size.empty() &&
                 !calibration_busy;
  const float duo_button_width =
      (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
  if (!can_add) ImGui::BeginDisabled();
  if (ImGui::Button("Add sample", ImVec2(duo_button_width, 0))) {
    calibrator_.setPattern(pattern_config);
    calibrator_.setCameraModel(camera_model_);
    calibrator_.setPinholeDistortionModel(pinhole_distortion_model_);
    calibrator_.setTargetWarpCompensation(target_warp_compensation_);
    if (calibrator_.addSample(current_detection.corners, current_detection.ids,
                              current_image_size, allow_duplicates_)) {
      if (save_samples_ && !current_last_frame.empty()) {
        fs::create_directories(samples_dir_);
        std::ostringstream name;
        name << samples_dir_ << "/sample_" << calibrator_.sampleCount() << ".png";
        cv::imwrite(name.str(), current_last_frame);
      }
    }
  }
  itemTooltip("Add the current detection as a calibration sample.");
  if (!can_add) ImGui::EndDisabled();

  ImGui::SameLine();
  if (calibration_busy) ImGui::BeginDisabled();
  if (ImGui::Button("Clear samples", ImVec2(duo_button_width, 0))) {
    calibrator_.clear();
  }
  itemTooltip("Remove all collected samples and reset progress.");
  if (calibration_busy) ImGui::EndDisabled();

  bool can_calib = !current_image_size.empty() && !calibration_busy;
  if (!can_calib) ImGui::BeginDisabled();
  if (ImGui::Button("Calibrate", ImVec2(-1, 0))) {
    if (startCalibrationTask(current_image_size, pattern_config)) {
      ImGui::OpenPopup("Calibrating camera");
    }
  }
  itemTooltip("Run intrinsic calibration with current samples.");
  if (!can_calib) ImGui::EndDisabled();

  ImGui::TextUnformatted("Save YAML");
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputText("##save_yaml_path", &calibration_save_path_);
  itemTooltip("Output path for the calibration YAML file.");
  if (calibration_busy) ImGui::BeginDisabled();
  if (ImGui::Button("Save calibration", ImVec2(-1, 0))) {
    calibrator_.saveResult(calibration_save_path_);
  }
  itemTooltip("Save the current calibration result to YAML.");
  if (calibration_busy) ImGui::EndDisabled();

  ImGui::Text("Samples: %d", calibrator_.sampleCount());
  const std::string status_text =
      (calibration_busy && !calibration_progress_message.empty())
          ? calibration_progress_message
          : calibrator_.lastStatus();
  ImGui::Text("Status: %s", status_text.c_str());
  ImGui::Text("Good enough: %s", calibrator_.goodEnough() ? "Yes" : "No");

  const auto& calib_result = calibrator_.result();
  if (calib_result.valid) {
    ImGui::Text("Model: %s", calib_result.model == CameraModel::Fisheye ? "fisheye" : "pinhole");
    ImGui::Text("RMS: %.5f", calib_result.rms);
    ImGui::Text("Inliers: %d  Rejected: %d",
                static_cast<int>(calib_result.inlier_sample_indices.size()),
                static_cast<int>(calib_result.rejected_sample_indices.size()));
    ImGui::Text("Mean/Median/P95 reproj: %.4f / %.4f / %.4f",
                calib_result.mean_reprojection_error,
                calib_result.median_reprojection_error,
                calib_result.p95_reprojection_error);

    if (!calib_result.per_view_errors.empty()) {
      std::vector<float> per_view;
      per_view.reserve(calib_result.per_view_errors.size());
      float max_v = 0.0f;
      for (double v : calib_result.per_view_errors) {
        const float fv = static_cast<float>(v);
        per_view.push_back(fv);
        max_v = std::max(max_v, fv);
      }
      ImGui::PlotHistogram("Per-view RMS", per_view.data(), static_cast<int>(per_view.size()),
                           0, nullptr, 0.0f, max_v * 1.05f, ImVec2(-1, 90));

      std::vector<size_t> order(per_view.size());
      std::iota(order.begin(), order.end(), 0U);
      std::sort(order.begin(), order.end(),
                [&](size_t a, size_t b) { return per_view[a] > per_view[b]; });
      const int top_n = std::min<int>(3, static_cast<int>(order.size()));
      for (int i = 0; i < top_n; ++i) {
        const size_t idx = order[static_cast<size_t>(i)];
        int sample_id = -1;
        if (idx < calib_result.inlier_sample_indices.size()) {
          sample_id = calib_result.inlier_sample_indices[idx] + 1;
        }
        ImGui::Text("Worst #%d: sample %d  RMS=%.4f", i + 1, sample_id, per_view[idx]);
      }
    }

    ImGui::TextUnformatted("Spatial residual map");
    drawResidualGrid(calib_result.residual_grid, ImVec2(-1, 140));
  }

  ImGui::Separator();
  ImGui::TextUnformatted("Stereo Pair");
  if (calibration_input_mode_ != CalibrationInputMode::Live) {
    ImGui::TextWrapped("Stereo calibration requires live synchronized streams.");
    return;
  }
  if (camera_model_ == CameraModel::Fisheye) {
    ImGui::TextWrapped("Stereo calibration currently uses the standard pinhole model.");
  }
  ImGui::TextUnformatted("Enable stereo pair");
  ImGui::Checkbox("##stereo_enable_pair", &stereo_mode_enabled_);
  itemTooltip("Enable paired capture and stereo calibration (left/right cameras).");

  if (!stereo_mode_enabled_) return;

  std::vector<int> candidate_indices;
  std::vector<std::string> candidate_labels;
  std::vector<const char*> candidate_label_ptrs;
  std::vector<std::string> left_candidate_tooltips;
  std::vector<std::string> right_candidate_tooltips;
  std::vector<const char*> left_candidate_tooltip_ptrs;
  std::vector<const char*> right_candidate_tooltip_ptrs;
  candidate_indices.reserve(slots_.size());
  candidate_labels.reserve(slots_.size());
  candidate_label_ptrs.reserve(slots_.size());
  left_candidate_tooltips.reserve(slots_.size());
  right_candidate_tooltips.reserve(slots_.size());
  left_candidate_tooltip_ptrs.reserve(slots_.size());
  right_candidate_tooltip_ptrs.reserve(slots_.size());

  for (size_t i = 0; i < slots_.size(); ++i) {
    const auto& slot = *slots_[i];
    const CameraState state = slot.device.state();
    if (state != CameraState::Connected && state != CameraState::Streaming) continue;

    CameraInfo info = slot.device.currentInfo();
    std::string label = info.label.empty() ? ("Slot " + std::to_string(i + 1)) : info.label;
    label += " (" + std::string(cameraStateLabel(state)) + ")";
    candidate_indices.push_back(static_cast<int>(i));
    candidate_labels.push_back(std::move(label));
  }

  for (const auto& label : candidate_labels) {
    candidate_label_ptrs.push_back(label.c_str());
    left_candidate_tooltips.push_back("Use " + label + " as the left image of the stereo pair.");
    right_candidate_tooltips.push_back("Use " + label + " as the right image of the stereo pair.");
  }
  for (const auto& tooltip : left_candidate_tooltips) {
    left_candidate_tooltip_ptrs.push_back(tooltip.c_str());
  }
  for (const auto& tooltip : right_candidate_tooltips) {
    right_candidate_tooltip_ptrs.push_back(tooltip.c_str());
  }

  if (candidate_indices.size() < 2) {
    ImGui::TextWrapped("Connect and start at least two camera streams to use stereo mode.");
    return;
  }

  auto findCandidatePos = [&](int slot_index) {
    for (size_t i = 0; i < candidate_indices.size(); ++i) {
      if (candidate_indices[i] == slot_index) return static_cast<int>(i);
    }
    return -1;
  };

  int left_pos = findCandidatePos(stereo_left_slot_);
  if (left_pos < 0) {
    left_pos = 0;
    stereo_left_slot_ = candidate_indices[0];
  }

  int right_pos = findCandidatePos(stereo_right_slot_);
  if (right_pos < 0 || stereo_right_slot_ == stereo_left_slot_) {
    right_pos = (left_pos == 0) ? 1 : 0;
    stereo_right_slot_ = candidate_indices[static_cast<size_t>(right_pos)];
  }

  ImGui::TextUnformatted("Stereo left camera");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (comboWithTooltips("##stereo_left_camera", &left_pos, candidate_label_ptrs.data(),
                        static_cast<int>(candidate_label_ptrs.size()),
                        left_candidate_tooltip_ptrs.data())) {
    stereo_left_slot_ = candidate_indices[static_cast<size_t>(left_pos)];
    if (stereo_right_slot_ == stereo_left_slot_) {
      right_pos = (left_pos == 0) ? 1 : 0;
      stereo_right_slot_ = candidate_indices[static_cast<size_t>(right_pos)];
    }
  }
  itemTooltip("Camera used as left image in the stereo pair.");

  right_pos = findCandidatePos(stereo_right_slot_);
  if (right_pos < 0) {
    right_pos = (left_pos == 0) ? 1 : 0;
    stereo_right_slot_ = candidate_indices[static_cast<size_t>(right_pos)];
  }
  ImGui::TextUnformatted("Stereo right camera");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (comboWithTooltips("##stereo_right_camera", &right_pos, candidate_label_ptrs.data(),
                        static_cast<int>(candidate_label_ptrs.size()),
                        right_candidate_tooltip_ptrs.data())) {
    stereo_right_slot_ = candidate_indices[static_cast<size_t>(right_pos)];
    if (stereo_right_slot_ == stereo_left_slot_) {
      right_pos = (left_pos == 0) ? 1 : 0;
      stereo_right_slot_ = candidate_indices[static_cast<size_t>(right_pos)];
    }
  }
  itemTooltip("Camera used as right image in the stereo pair.");

  if (stereo_right_slot_ == stereo_left_slot_) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.35f, 1.0f),
                       "Left and right cameras must be different.");
    return;
  }

  ImGui::TextUnformatted("Sync tolerance (ms)");
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputFloat("##stereo_sync_tolerance_ms", &stereo_sync_tolerance_ms_, 1.0f, 5.0f, "%.1f");
  stereo_sync_tolerance_ms_ = std::max(0.0f, stereo_sync_tolerance_ms_);
  itemTooltip("Max timestamp delta allowed between left/right detections.");

  ImGui::TextUnformatted("Expected baseline (m)");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::InputFloat("##stereo_expected_baseline", &stereo_expected_baseline_m_, 0.001f, 0.01f,
                        "%.4f")) {
    stereo_expected_baseline_m_ = std::max(0.0f, stereo_expected_baseline_m_);
  }
  itemTooltip("Physical distance between camera optical centers in meters (optional).");

  ImGui::TextWrapped("Fix intrinsics in stereo solve");
  ImGui::Checkbox("##stereo_fix_intrinsics", &stereo_fix_intrinsics_);
  itemTooltip("If enabled, estimate each camera intrinsics first and keep them fixed in stereo solve.");

  stereo_calibrator_.setPattern(pattern_config);
  stereo_calibrator_.setExpectedBaseline(static_cast<double>(stereo_expected_baseline_m_));

  CameraSlot* left_slot = slots_[static_cast<size_t>(stereo_left_slot_)].get();
  CameraSlot* right_slot = slots_[static_cast<size_t>(stereo_right_slot_)].get();
  DetectionResult left_detection;
  DetectionResult right_detection;
  cv::Size left_image_size;
  cv::Size right_image_size;
  bool left_found = false;
  bool right_found = false;
  uint64_t left_ts_ns = 0;
  uint64_t right_ts_ns = 0;
  if (left_slot) {
    std::lock_guard<std::mutex> left_lock(left_slot->frame_mutex);
    left_detection = left_slot->detection;
    left_image_size = left_slot->last_image_size;
    left_found = left_slot->found;
    left_ts_ns = left_slot->detection_timestamp_ns;
  }
  if (right_slot) {
    std::lock_guard<std::mutex> right_lock(right_slot->frame_mutex);
    right_detection = right_slot->detection;
    right_image_size = right_slot->last_image_size;
    right_found = right_slot->found;
    right_ts_ns = right_slot->detection_timestamp_ns;
  }

  const double sync_delta_ms = (left_ts_ns > 0 && right_ts_ns > 0)
                                   ? std::abs(static_cast<double>(left_ts_ns) - static_cast<double>(right_ts_ns)) / 1e6
                                   : std::numeric_limits<double>::infinity();
  const bool synced = std::isfinite(sync_delta_ms) && sync_delta_ms <= static_cast<double>(stereo_sync_tolerance_ms_);
  if (std::isfinite(sync_delta_ms)) {
    ImGui::Text("Current sync delta: %.2f ms", sync_delta_ms);
  } else {
    ImGui::Text("Current sync delta: n/a");
  }

  const bool can_add_stereo =
      left_slot && right_slot &&
      left_found && right_found &&
      !left_detection.corners.empty() &&
      !right_detection.corners.empty() &&
      !left_image_size.empty() &&
      !right_image_size.empty() &&
      synced;

  const float stereo_duo_button_width =
      (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
  if (!can_add_stereo) ImGui::BeginDisabled();
  if (ImGui::Button("Add stereo sample", ImVec2(stereo_duo_button_width, 0))) {
    stereo_calibrator_.addSample(left_detection.corners, left_detection.ids, left_image_size,
                                 right_detection.corners, right_detection.ids, right_image_size,
                                 allow_duplicates_);
  }
  itemTooltip("Add a synchronized pair when both cameras detect the pattern.");
  if (!can_add_stereo) ImGui::EndDisabled();

  ImGui::SameLine();
  if (ImGui::Button("Clear stereo samples", ImVec2(stereo_duo_button_width, 0))) {
    stereo_calibrator_.clear();
    stereo_calibrator_.setPattern(pattern_config);
    stereo_calibrator_.setExpectedBaseline(static_cast<double>(stereo_expected_baseline_m_));
  }
  itemTooltip("Remove all stereo pair samples.");

  const bool can_stereo_calib = stereo_calibrator_.sampleCount() >= 5;
  if (!can_stereo_calib) ImGui::BeginDisabled();
  if (ImGui::Button("Calibrate stereo", ImVec2(-1, 0))) {
    stereo_calibrator_.calibrate(stereo_fix_intrinsics_);
  }
  itemTooltip("Compute stereo extrinsics and rectification from paired samples.");
  if (!can_stereo_calib) ImGui::EndDisabled();

  ImGui::TextUnformatted("Save stereo YAML");
  ImGui::SetNextItemWidth(-FLT_MIN);
  ImGui::InputText("##save_stereo_yaml_path", &stereo_calibration_save_path_);
  itemTooltip("Output path for stereo calibration YAML.");

  const bool can_save_stereo = stereo_calibrator_.result().valid;
  if (!can_save_stereo) ImGui::BeginDisabled();
  if (ImGui::Button("Save stereo calibration", ImVec2(-1, 0))) {
    stereo_calibrator_.saveResult(stereo_calibration_save_path_);
  }
  itemTooltip("Save stereo calibration result to YAML.");
  if (!can_save_stereo) ImGui::EndDisabled();

  const auto& stereo_result = stereo_calibrator_.result();
  ImGui::Text("Stereo samples: %d", stereo_calibrator_.sampleCount());
  ImGui::Text("Stereo status: %s", stereo_calibrator_.lastStatus().c_str());
  if (stereo_result.valid) {
    ImGui::Text("Stereo RMS: %.5f", stereo_result.rms);
    ImGui::Text("Estimated baseline: %.4f m", stereo_result.baseline);
    ImGui::Text("Rectification: %s", stereo_result.disparity_to_depth.empty() ? "no" : "yes");
    if (stereo_expected_baseline_m_ > 0.0f) {
      const double err = std::abs(stereo_result.baseline - static_cast<double>(stereo_expected_baseline_m_));
      const double err_pct = (err / static_cast<double>(stereo_expected_baseline_m_)) * 100.0;
      ImGui::Text("Expected baseline: %.4f m (error %.2f%%)",
                  stereo_expected_baseline_m_, err_pct);
    }
  }
}

void ImGuiApp::drawFeaturePanel(CameraSlot& slot) {
  if (ImGui::Button("Refresh features")) {
    queueFeatureRefresh(slot);
  }

  bool busy = false;
  std::string feature_status;
  std::vector<FeatureInfo> features;
  std::unordered_map<std::string, FeatureValue> values;
  {
    std::lock_guard<std::mutex> lock(slot.feature_mutex);
    busy = slot.feature_busy;
    feature_status = slot.feature_status;
    features = slot.features;
    values = slot.feature_values;
  }

  if (busy) {
    ImGui::SameLine();
    ImGui::TextDisabled("Updating...");
  }

  ImGui::InputTextWithHint("##feature_filter", "Name contains...", &slot.feature_filter);
  ImGui::SameLine();
  ImGui::TextUnformatted("Filter");

  if (!feature_status.empty()) {
    ImGui::TextWrapped("Status: %s", feature_status.c_str());
  }

  auto setCachedValue = [&](const std::string& id, const FeatureValue& value) {
    std::lock_guard<std::mutex> lock(slot.feature_mutex);
    slot.feature_values[id] = value;
  };

  ImGui::BeginChild("features", ImVec2(-FLT_MIN, 340), true);
  for (auto& feature : features) {
    if (!slot.feature_filter.empty()) {
      if (feature.id.find(slot.feature_filter) == std::string::npos &&
          feature.display_name.find(slot.feature_filter) == std::string::npos) {
        continue;
      }
    }

    if (feature.type == FeatureType::Category) continue;

    std::string label = feature.display_name.empty() ? feature.id : feature.display_name;
    ImGui::PushID(feature.id.c_str());

    if (!feature.readable && !feature.writable) {
      ImGui::TextDisabled("%s", label.c_str());
      ImGui::PopID();
      continue;
    }

    bool disabled = !feature.writable;
    if (disabled) ImGui::BeginDisabled();

    FeatureValue value;
    const auto value_it = values.find(feature.id);
    if (value_it != values.end()) {
      value = value_it->second;
    }

    switch (feature.type) {
      case FeatureType::Integer: {
        int64_t v = 0;
        if (auto p = std::get_if<int64_t>(&value.value)) v = *p;
        int64_t minv = static_cast<int64_t>(feature.min);
        int64_t maxv = static_cast<int64_t>(feature.max);
        if (minv < maxv) {
          if (ImGui::SliderScalar(label.c_str(), ImGuiDataType_S64, &v, &minv, &maxv)) {
            FeatureValue updated{v};
            setCachedValue(feature.id, updated);
            queueFeatureSet(slot, feature, updated);
          }
        } else {
          if (ImGui::InputScalar(label.c_str(), ImGuiDataType_S64, &v)) {
            FeatureValue updated{v};
            setCachedValue(feature.id, updated);
            queueFeatureSet(slot, feature, updated);
          }
        }
        break;
      }
      case FeatureType::Float: {
        double v = 0.0;
        if (auto p = std::get_if<double>(&value.value)) v = *p;
        double minv = feature.min;
        double maxv = feature.max;
        if (minv < maxv) {
          if (ImGui::SliderScalar(label.c_str(), ImGuiDataType_Double, &v, &minv, &maxv)) {
            FeatureValue updated{v};
            setCachedValue(feature.id, updated);
            queueFeatureSet(slot, feature, updated);
          }
        } else {
          if (ImGui::InputScalar(label.c_str(), ImGuiDataType_Double, &v)) {
            FeatureValue updated{v};
            setCachedValue(feature.id, updated);
            queueFeatureSet(slot, feature, updated);
          }
        }
        break;
      }
      case FeatureType::Enumeration: {
        std::string current;
        if (auto p = std::get_if<std::string>(&value.value)) current = *p;
        int current_index = 0;
        for (size_t i = 0; i < feature.enum_entries.size(); ++i) {
          if (feature.enum_entries[i] == current) {
            current_index = static_cast<int>(i);
            break;
          }
        }

        std::vector<const char*> items;
        items.reserve(feature.enum_entries.size());
        for (const auto& entry : feature.enum_entries) {
          items.push_back(entry.c_str());
        }

        if (!items.empty()) {
          if (ImGui::Combo(label.c_str(), &current_index, items.data(), static_cast<int>(items.size()))) {
            FeatureValue updated{std::string(feature.enum_entries[current_index])};
            setCachedValue(feature.id, updated);
            queueFeatureSet(slot, feature, updated);
          }
        } else {
          ImGui::Text("%s", label.c_str());
        }
        break;
      }
      case FeatureType::Boolean: {
        bool enabled = false;
        if (auto p = std::get_if<bool>(&value.value)) enabled = *p;
        if (ImGui::Checkbox(label.c_str(), &enabled)) {
          FeatureValue updated{enabled};
          setCachedValue(feature.id, updated);
          queueFeatureSet(slot, feature, updated);
        }
        break;
      }
      case FeatureType::String: {
        std::string text;
        if (auto p = std::get_if<std::string>(&value.value)) text = *p;
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer), "%s", text.c_str());
        if (ImGui::InputText(label.c_str(), buffer, sizeof(buffer))) {
          FeatureValue updated{std::string(buffer)};
          setCachedValue(feature.id, updated);
          queueFeatureSet(slot, feature, updated);
        }
        break;
      }
      case FeatureType::Command: {
        if (ImGui::Button(label.c_str())) {
          queueFeatureCommand(slot, feature);
        }
        break;
      }
      default:
        ImGui::Text("%s", label.c_str());
        break;
    }

    if (disabled) ImGui::EndDisabled();
    ImGui::PopID();
  }
  ImGui::EndChild();
}
