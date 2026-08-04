cmake_minimum_required(VERSION 3.16)
project(mcpr_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../mcpr-cuda/alphas.csv"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../mcpr-cuda/alphas.csv.bz2")
    execute_process(
        COMMAND bzip2 -dk "${CMAKE_CURRENT_LIST_DIR}/../mcpr-cuda/alphas.csv.bz2"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../mcpr-cuda"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../mcpr-cuda/alphas.csv
        1
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
