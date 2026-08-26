# sss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sss
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
