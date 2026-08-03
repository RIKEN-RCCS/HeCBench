# all-pairs-distance-acc/CMakeLists.txt

add_hecbench_benchmark(
    NAME all-pairs-distance
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
