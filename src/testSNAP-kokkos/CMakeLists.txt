cmake_minimum_required(VERSION 3.16)
project(testSNAP_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

set(REFDATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/refdata")
if(NOT EXISTS "${REFDATA_DIR}/refdata_2J14_W.h")
    file(MAKE_DIRECTORY "${REFDATA_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_LIST_DIR}/../testSNAP-omp/refdata.tar.gz"
        WORKING_DIRECTORY "${REFDATA_DIR}"
        RESULT_VARIABLE REFDATA_EXTRACT_RESULT
    )
    if(NOT REFDATA_EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract testSNAP reference data")
    endif()
endif()

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ${REFDATA_DIR}
    ../testSNAP-omp
)
target_compile_definitions(main PRIVATE
    REFDATA_TWOJ=14
)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        --nsteps
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
