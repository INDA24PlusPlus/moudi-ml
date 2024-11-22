cmake_minimum_required(VERSION 3.23)

project(ML VERSION 1.0.0)

find_package(GSL REQUIRED)
add_subdirectory("submodules/String-Formatter")

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "-pg -g")
set(CMAKE_C_FLAGS_RELEASE "-Wall")

message(STATUS "FLAGS: ${CMAKE_C_FLAGS_DEBUG}")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
file(GLOB_RECURSE INCLUDES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/includes/*.h)

message(STATUS "Sources : ${SOURCES}")
message(STATUS "Includes : ${INCLUDES}")

add_library(ml SHARED 
    ${SOURCES}
    ${INCLUDES})

target_include_directories(ml PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/String-Formatter/include)

target_link_libraries(ml PRIVATE GSL::gsl GSL::gslcblas)
