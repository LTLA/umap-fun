cmake_minimum_required(VERSION 3.14)

project(umap_wasm
    VERSION 1.0.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  umappp
  GIT_REPOSITORY https://github.com/LTLA/umappp
  GIT_TAG master
)
FetchContent_MakeAvailable(umappp)

add_executable(
    umaplib 
    run_umap.cpp
)

target_compile_options(
    umaplib PUBLIC "-O2"
)

target_link_libraries(
    umaplib 
    umappp
)

set_target_properties(umaplib PROPERTIES LINK_FLAGS "--bind -O2 -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -s FORCE_FILESYSTEM=1 -s USE_ZLIB=1 -lworkerfs.js")

add_custom_command(TARGET umaplib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE_BASE_NAME:umaplib>.js ${CMAKE_SOURCE_DIR}/../src
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE_BASE_NAME:umaplib>.wasm ${CMAKE_SOURCE_DIR}/../src
)
