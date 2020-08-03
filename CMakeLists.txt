cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(explorer)

option(GL_DEBUG_CONTEXT "Create OpenGL debug context" OFF)

set(SOURCE_DIR src)
set(LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED PATHS ${LIB_DIR})
find_package(X11 REQUIRED)

add_library(engine SHARED ${SOURCE_DIR}/proto.cc)

add_executable(explorer ${SOURCE_DIR}/main.cc)
target_link_libraries(explorer
	PRIVATE
		dl
		X11
		X11::Xfixes
		OpenGL::GL
		GLEW::GLEW
)