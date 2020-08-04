cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

set( OGLBASE_SOURCES
	 src/error.cc
	 src/framebuffer.cc
	 src/handle.cc
	 src/shader.cc
	 )

add_library(oglbase STATIC ${OGLBASE_SOURCES})
target_link_libraries(oglbase PRIVATE OpenGL::GL GLEW::GLEW)
target_include_directories(oglbase PUBLIC include)
