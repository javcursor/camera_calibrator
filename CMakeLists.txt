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

set(GENICAM_ROOT "" CACHE PATH "Optional root containing GenICam headers and libraries")

set(_genicam_roots
  "${GENICAM_ROOT}"
  "$ENV{GENICAM_ROOT}"
  "/usr"
  "/usr/local"
  "/opt"
)

file(GLOB _opt_roots LIST_DIRECTORIES true "/opt/*")
list(SORT _opt_roots)
list(APPEND _genicam_roots ${_opt_roots})

set(_genicam_search_roots)
foreach(root IN LISTS _genicam_roots)
  if(root AND EXISTS "${root}")
    list(APPEND _genicam_search_roots "${root}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _genicam_search_roots)

find_path(GENICAM_INCLUDE_DIR
  NAMES GenApi/NodeMapFactory.h Base/GCException.h
  HINTS ${_genicam_search_roots}
  PATH_SUFFIXES include
)

function(find_genicam_library out_var stem)
  foreach(root IN LISTS _genicam_search_roots)
    foreach(suffix IN ITEMS lib lib64 bin/Linux64_x64 bin/Linux64_ARM64)
      file(GLOB candidates LIST_DIRECTORIES false
        "${root}/${suffix}/lib${stem}*.so"
        "${root}/${suffix}/lib${stem}*.so.*"
      )
      list(SORT candidates)
      if(candidates)
        list(GET candidates 0 candidate)
        set(${out_var} "${candidate}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

find_genicam_library(GENICAM_GENAPI_LIBRARY GenApi)
find_genicam_library(GENICAM_GCBASE_LIBRARY GCBase)

if(GENICAM_INCLUDE_DIR AND GENICAM_GENAPI_LIBRARY AND GENICAM_GCBASE_LIBRARY)
  set(GENICAM_FOUND TRUE)
else()
  set(GENICAM_FOUND FALSE)
endif()

if(ARAVIS_FOUND)
  add_compile_definitions(HAVE_ARAVIS)
endif()

if(GENICAM_FOUND)
  add_compile_definitions(HAVE_GENTL_GENAPI)
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

if(GENICAM_FOUND)
  get_filename_component(GENICAM_LIBRARY_DIR "${GENICAM_GENAPI_LIBRARY}" DIRECTORY)
  target_include_directories(camera_calibrator PRIVATE ${GENICAM_INCLUDE_DIR})
  target_link_libraries(camera_calibrator PRIVATE ${GENICAM_GENAPI_LIBRARY} ${GENICAM_GCBASE_LIBRARY})
  set_property(TARGET camera_calibrator APPEND PROPERTY BUILD_RPATH "${GENICAM_LIBRARY_DIR}")
  set_property(TARGET camera_calibrator APPEND PROPERTY INSTALL_RPATH "${GENICAM_LIBRARY_DIR}")
endif()

if(ARAVIS_FOUND)
  target_include_directories(camera_calibrator PRIVATE ${ARAVIS_INCLUDE_DIRS})
  target_link_libraries(camera_calibrator PRIVATE ${ARAVIS_LIBRARIES})
  target_compile_options(camera_calibrator PRIVATE ${ARAVIS_CFLAGS_OTHER})
endif()

function(opencv_has_header out_var relative_path)
  set(found FALSE)
  foreach(include_dir IN LISTS OpenCV_INCLUDE_DIRS)
    if(EXISTS "${include_dir}/${relative_path}")
      set(found TRUE)
      break()
    endif()
  endforeach()
  set(${out_var} ${found} PARENT_SCOPE)
  set(${out_var} ${found} CACHE INTERNAL "OpenCV optional header presence" FORCE)
endfunction()

opencv_has_header(OPENCV_ARUCO_HEADER_FOUND "opencv2/aruco.hpp")
opencv_has_header(OPENCV_OMNIDIR_HEADER_FOUND "opencv2/ccalib/omnidir.hpp")

if(OPENCV_ARUCO_HEADER_FOUND)
  if(TARGET opencv_aruco)
    target_link_libraries(camera_calibrator PRIVATE opencv_aruco)
  endif()
  target_compile_definitions(camera_calibrator PRIVATE HAVE_OPENCV_ARUCO)
endif()

if(OPENCV_OMNIDIR_HEADER_FOUND)
  if(TARGET opencv_ccalib)
    target_link_libraries(camera_calibrator PRIVATE opencv_ccalib)
  endif()
  target_compile_definitions(camera_calibrator PRIVATE HAVE_OPENCV_OMNIDIR)
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
