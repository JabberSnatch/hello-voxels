cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(explorer)

option(ENABLE_GL_DEBUG_CONTEXT "Create OpenGL debug context" OFF)

set(SOURCE_DIR src)
set(LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED) # PATHS ${LIB_DIR})
find_package(X11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")

set(QVDB_BUILD_TESTS ON CACHE BOOL "Disable qvdb tests")
set(QVDB_ENABLE_CACHE OFF CACHE BOOL "Enable qvdb cache")
add_subdirectory(src/quick_vdb)

add_subdirectory(src/oglbase)

add_library(engine SHARED ${SOURCE_DIR}/proto.cc ${SOURCE_DIR}/rng.cc)
target_link_libraries(engine PRIVATE oglbase qvdb)


add_executable(explorer ${SOURCE_DIR}/main.cc)
target_link_libraries(explorer
	PRIVATE
		dl
		X11
		X11::Xfixes
		OpenGL::GL
		GLEW::GLEW
)

if (ENABLE_GL_DEBUG_CONTEXT)
  add_compile_definitions(ENABLE_GL_DEBUG_CONTEXT)

  target_link_libraries(explorer
	PRIVATE
		oglbase
  )
endif()
