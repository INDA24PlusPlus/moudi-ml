cmake_minimum_required(VERSION 3.23)

project(ML VERSION 1.0.0)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "-pg -g -lm")
set(CMAKE_C_FLAGS_RELEASE "-Wall -lm")

message(STATUS "FLAGS: ${CMAKE_C_FLAGS_DEBUG}")

add_subdirectory("lib")
find_package(raylib REQUIRED)
find_package(GSL REQUIRED)

get_target_property(LIB_INCLUDES ml INCLUDE_DIRECTORIES)

message(STATUS "Lib includes: ${LIB_INCLUDES}")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
file(GLOB_RECURSE MNISC_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mnisc/*.c)

add_executable(exec ${MNISC_SOURCES})

# Add other flags to the compiler
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Include header files
target_include_directories(exec PRIVATE 
	${CMAKE_CURRENT_SOURCE_DIR}/lib/includes
    ${LIB_INCLUDES})

# Link libraries
target_link_libraries(exec PRIVATE ml fmt raylib GSL::gsl GSL::gslcblas)
