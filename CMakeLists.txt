
cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.8.0 HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/solutions)

set(CMAKE_CXX_STANDARD 20)
set(TARGET poisson)
set(TARGET_SRC src/main.cc)
project(${TARGET})

add_executable(test_kelly src/test_kelly.cc)
deal_ii_setup_target(test_kelly)

deal_ii_initialize_cached_variables()
deal_ii_invoke_autopilot()

add_custom_target(run_kelly
    COMMAND test_kelly
    DEPENDS test_kelly
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
