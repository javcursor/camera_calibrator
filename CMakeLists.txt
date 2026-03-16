cmake_minimum_required(VERSION 3.20)
project(camera_calibrator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.8 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

# GLFW
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.9
  )
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endif()

if(TARGET glfw)
  set(GLFW_TARGET glfw)
elseif(TARGET glfw3)
  set(GLFW_TARGET glfw3)
else()
  message(FATAL_ERROR "GLFW target not found")
endif()

# Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.9
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
)

target_include_directories(imgui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${imgui_SOURCE_DIR}/misc/cpp
)

target_link_libraries(imgui PUBLIC ${GLFW_TARGET} OpenGL::GL)

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_NONE)

# Aravis (GenICam/GigE Vision)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ARAVIS QUIET aravis-0.8)
endif()

if(ARAVIS_FOUND)
  add_compile_definitions(HAVE_ARAVIS)
endif()

set(CALIB_SOURCES
  src/main.cpp
  src/ui/ImGuiApp.cpp
  src/ui/ImageTexture.cpp
  src/camera/CameraDevice.cpp
  src/camera/BackendFactory.cpp
  src/camera/OpenCVBackend.cpp
  src/processing/PatternDetector.cpp
  src/processing/Calibration.cpp
)

if(UNIX)
  list(APPEND CALIB_SOURCES src/camera/GenTLBackend.cpp)
endif()

if(ARAVIS_FOUND)
  list(APPEND CALIB_SOURCES src/camera/AravisBackend.cpp)
endif()

add_executable(camera_calibrator ${CALIB_SOURCES})

target_link_libraries(camera_calibrator PRIVATE
  imgui
  ${GLFW_TARGET}
  OpenGL::GL
  Threads::Threads
  ${OpenCV_LIBS}
)

target_include_directories(camera_calibrator PRIVATE include)

if(UNIX)
  target_compile_definitions(camera_calibrator PRIVATE HAVE_GENTL)
  target_link_libraries(camera_calibrator PRIVATE ${CMAKE_DL_LIBS})
endif()

if(ARAVIS_FOUND)
  target_include_directories(camera_calibrator PRIVATE ${ARAVIS_INCLUDE_DIRS})
  target_link_libraries(camera_calibrator PRIVATE ${ARAVIS_LIBRARIES})
  target_compile_options(camera_calibrator PRIVATE ${ARAVIS_CFLAGS_OTHER})
endif()

include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
check_include_file_cxx("opencv2/aruco.hpp" OPENCV_ARUCO_HEADER_FOUND)
unset(CMAKE_REQUIRED_INCLUDES)

if(OPENCV_ARUCO_HEADER_FOUND)
  if(TARGET opencv_aruco)
    target_link_libraries(camera_calibrator PRIVATE opencv_aruco)
  endif()
  target_compile_definitions(camera_calibrator PRIVATE HAVE_OPENCV_ARUCO)
endif()

set(APP_ASSETS_DIR "${CMAKE_SOURCE_DIR}/assets")
if(EXISTS "${APP_ASSETS_DIR}")
  add_custom_command(TARGET camera_calibrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:camera_calibrator>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${APP_ASSETS_DIR}" "$<TARGET_FILE_DIR:camera_calibrator>/assets"
    COMMENT "Copying app assets next to executable"
  )
endif()

install(TARGETS camera_calibrator
  RUNTIME DESTINATION bin
)

if(EXISTS "${APP_ASSETS_DIR}")
  install(DIRECTORY "${APP_ASSETS_DIR}/"
    DESTINATION "share/camera_calibrator/assets"
  )
endif()
