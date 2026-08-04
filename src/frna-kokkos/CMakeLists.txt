cmake_minimum_required(VERSION 3.16)
project(frna_kokkos LANGUAGES C CXX)

find_package(Kokkos REQUIRED)

add_executable(main
    main.cpp
    fbase.c
    fparam.c
    util.c
)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
target_compile_definitions(main PRIVATE
    INT
)
target_compile_definitions(main PRIVATE $<$<COMPILE_LANGUAGE:C>:INT>)

set(FRNA_DATAPATH "${CMAKE_CURRENT_LIST_DIR}/../prna-cuda/data_tables")
if(NOT EXISTS "${FRNA_DATAPATH}"
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../prna-cuda/data_tables.tar.gz")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../prna-cuda/data_tables.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../prna-cuda"
    )
endif()

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env DATAPATH=${FRNA_DATAPATH} $<TARGET_FILE:main>
        ../frna-cuda/RD0260.seq
        rd0260.out
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
