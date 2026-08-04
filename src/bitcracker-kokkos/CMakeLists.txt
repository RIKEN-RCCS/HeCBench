cmake_minimum_required(VERSION 3.16)
project(bitcracker_kokkos LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    w_blocks.cpp
    attack.cpp
    utils.cpp
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
        -f
        hash_pass/img_win8_user_hash.txt
        -d
        hash_pass/user_passwords.txt
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
