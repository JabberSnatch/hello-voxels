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

set(SHADER_SOURCES
  ${SOURCE_DIR}/voxeltraversal.frag.glsl
  ${SOURCE_DIR}/transmittance.frag.glsl
  ${SOURCE_DIR}/directirradiance.frag.glsl
  ${SOURCE_DIR}/singlescattering.frag.glsl
  ${SOURCE_DIR}/scatteringdensity.frag.glsl
  ${SOURCE_DIR}/indirectirradiance.frag.glsl
  ${SOURCE_DIR}/multiplescattering.frag.glsl
  )

set(SHADER_INCLUDES)
set(SHADER_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/shaders")
foreach (INPUT_SHADER IN LISTS SHADER_SOURCES)
  cmake_path(GET INPUT_SHADER FILENAME FILENAME)
  set(OUTPUT_FILE "${SHADER_INCLUDE_PATH}/${FILENAME}.inc")
  list(APPEND SHADER_INCLUDES ${OUTPUT_FILE})
  if (${CMAKE_GENERATOR} STREQUAL "Ninja")
	set(COMMAND sed -e '1 i R\"__lstr__\(' -e '$$ a \)__lstr__\"' ${INPUT_SHADER} > ${OUTPUT_FILE})
  else()
	set(COMMAND sed -e '1 i R\"__lstr__\(' -e '$ a \)__lstr__\"' ${INPUT_SHADER} > ${OUTPUT_FILE})
  endif()
  add_custom_command(OUTPUT ${OUTPUT_FILE}
	COMMAND ${COMMAND}
	DEPENDS ${INPUT_SHADER}
	WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
	)
endforeach()

add_custom_target(shaders DEPENDS ${SHADER_INCLUDES})

add_library(engine SHARED ${SOURCE_DIR}/proto.cc ${SOURCE_DIR}/rng.cc)
add_dependencies(engine shaders)
target_link_libraries(engine PRIVATE oglbase qvdb)
target_include_directories(engine PRIVATE ${SHADER_INCLUDE_PATH})

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
