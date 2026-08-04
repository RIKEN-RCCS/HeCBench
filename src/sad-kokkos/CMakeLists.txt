cmake_minimum_required(VERSION 3.16)
project(sad_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PRIVATE
    ../sad-cuda
)

if((NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda/coins.bmp"
    OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda/coin.bmp"
    OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda/bitmap_image.hpp")
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda/data.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda/data.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../sad-cuda"
    )
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        ../sad-cuda/coins.bmp
        ../sad-cuda/coin.bmp
        100
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
