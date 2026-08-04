cmake_minimum_required(VERSION 3.16)
project(slu_kokkos LANGUAGES C CXX)

find_package(Kokkos REQUIRED)
find_program(NICSLU_CC NAMES gcc cc REQUIRED)

set(NICSLU_DIR "${CMAKE_CURRENT_BINARY_DIR}/nicslu-src/nicslu")
if(NOT EXISTS "${NICSLU_DIR}/include/nicslu.h")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nicslu-src")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xjf "${CMAKE_CURRENT_LIST_DIR}/../slu-cuda/src/nicslu.tar.bz2"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nicslu-src"
        RESULT_VARIABLE NICSLU_EXTRACT_RESULT
    )
    if(NOT NICSLU_EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract bundled NicSLU")
    endif()
endif()

set(NICSLU_LIB "${NICSLU_DIR}/lib/nicslu.a")
set(NICSLU_UTIL_LIB "${NICSLU_DIR}/util/nicslu_util.a")
add_custom_command(
    OUTPUT "${NICSLU_LIB}" "${NICSLU_UTIL_LIB}"
    COMMAND ${CMAKE_MAKE_PROGRAM} -C "${NICSLU_DIR}" CC=${NICSLU_CC}
    WORKING_DIRECTORY "${NICSLU_DIR}"
    COMMENT "Building bundled NicSLU"
)
add_custom_target(nicslu_bundled DEPENDS "${NICSLU_LIB}" "${NICSLU_UTIL_LIB}")

add_executable(main
    main.cpp
    ../slu-cuda/src/main.cpp
    ../slu-cuda/src/preprocess.cpp
    ../slu-cuda/src/symbolic.cpp
    ../slu-cuda/src/Timer.cpp
)
add_dependencies(main nicslu_bundled)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_link_libraries(main PRIVATE "${NICSLU_LIB}" "${NICSLU_UTIL_LIB}")
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../slu-cuda/include
    ../slu-cuda/src
    "${NICSLU_DIR}/include"
    "${NICSLU_DIR}/util"
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        -i
        "${NICSLU_DIR}/test/add32.mtx"
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
