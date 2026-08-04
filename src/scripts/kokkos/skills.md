# HeCBench Kokkos Porting Context

This file is intended as context for an LLM or developer resuming work on the
HeCBench Kokkos benchmark ports. It captures the conventions, scripts, known
pitfalls, and review rules used during the porting effort.

## Goal

Each `src/<benchmark>-kokkos` directory should be a faithful Kokkos version of
the existing reference benchmark (`-omp`, `-cuda`, or `-sycl`). The expected
differences should mainly be the parallelization mechanism and backend API, not
the benchmark behavior, inputs, validation, or overall structure.

## Main Coherence Rule

Before modifying or creating a Kokkos benchmark:

1. Identify the available reference versions:
   - `src/<benchmark>-omp`
   - `src/<benchmark>-cuda`
   - `src/<benchmark>-sycl`
2. Read their `CMakeLists.txt`, `Makefile`, `main`, `driver`, `kernel`, `utils`,
   and validation files.
3. Preserve the same application logic:
   - same inputs,
   - same run arguments,
   - same validation,
   - same file separation when the reference is split across files.
4. Adapt only the parallel/backend layer to Kokkos.

If the reference version is multi-file (`main`, `driver`, `kernel`, `utils`),
the Kokkos version should also remain multi-file. Avoid compacting everything
into `main.cpp` unless the reference versions are also compact.

Important example: `bsw-kokkos` is currently suspicious because it keeps
everything inside `main.cpp`, while `bsw-cuda` and `bsw-sycl` have `main`,
`driver`, `kernel`, and `utils`. Also, `bsw-kokkos` generates synthetic
sequences, whereas CUDA/SYCL read FASTA files. That is not a clean performance
comparison.

## Cross-Version Dependencies

The goal is for each Kokkos directory to be as self-contained and as coherent
with the reference versions as possible.

Rules:

- If the OMP version already depends on CUDA or SYCL files, the Kokkos version
  may use the same dependency.
- If the OMP version contains support files locally, copy those files into the
  Kokkos directory instead of depending on `../*-omp`.
- Avoid artificial folders such as `_external`.
- Do not invent a local `reference.h` if the reference version does not have
  one.
- For datasets, it is acceptable to read the same file as the reference version
  when that is the benchmark convention.

## CUDA Built-In Type Conflicts

With NVHPC/CUDA, types such as `int3`, `uchar3`, `short4`, `double2`, `ulong4`,
and similar names may already be defined by `vector_types.h`.

In Kokkos ports, if a benchmark defines a local type with the same name, rename
the local type with the `hec_` prefix, for example:

- `int3` -> `hec_int3`
- `uchar3` -> `hec_uchar3`
- `short4` -> `hec_short4`
- `double2` -> `hec_double2`
- `ulong4` -> `hec_ulong4`

Only rename the local type and its local uses. Do not change the algorithm.

## Important Scripts

### Running Benchmarks

Main script:

```sh
python3 ../scripts/run_kokkos_benchmarks.py
```

When launched from `src/`, the script detects `*-kokkos` directories with a
`CMakeLists.txt`.

Related files:

- `src/kokkos-logs/kokkos-passed.txt`: benchmarks currently considered passed.
- `scripts/kokkos-benchmark-list.txt`: target batch to test.
- `src/kokkos-logs/<benchmark>.log`: individual benchmark log.

The script skips benchmarks already listed in `kokkos-passed.txt`.

Warning: a benchmark can compile and return `rc=0` while still printing `FAIL`
or `FAILED` in its log. Scan passed benchmark logs with:

```sh
rg -n "\\bFAIL(ED)?\\b" src/kokkos-logs/*.log
```

Known example: `minisweep-kokkos` previously printed `verify: FAIL`.

### Cleaning Builds

Script:

```sh
python3 scripts/clean_kokkos_builds.py --dry-run
python3 scripts/clean_kokkos_builds.py
```

It removes `build/` directories under `src/*-kokkos`.

## CMake and Makefile Expectations

Each Kokkos benchmark should have:

- a `CMakeLists.txt`,
- a `Makefile` when that is expected by this repository's benchmark workflow,
- a `run` target consistent with the OMP/CUDA/SYCL reference.

Typical CMake pattern:

```cmake
find_package(Kokkos REQUIRED)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_compile_features(main PRIVATE cxx_std_17)
```

Do not hardcode CUDA unless necessary. Kokkos ports should remain portable
across multiple backends. However, if the reference benchmark already depends on
CUDA/SYCL helpers or datasets, reproducing that convention is acceptable.

## Datasets and Archives

If a benchmark fails because a dataset is stored in a local archive in the repo
(`.tar.gz`, `.tar.bz2`, `.zip`, `.bz2`), add a clean CMake extraction step
before the run target, as already done for several benchmarks.

If the README says an external dataset must be downloaded from a link, do not
fake or synthesize the data.


## Known Errors and Typical Fixes

### `has already been defined`

Cause: conflict with CUDA-provided vector types (`double2`, `int3`, etc.).

Fix: rename the local type with the `hec_` prefix.

### `undefined reference`

Typical cause: `CMakeLists.txt` or `Makefile` does not compile all required
source files (`utils.cpp`, `driver.cpp`, `kernel.cpp`, etc.).

Fix: compare with OMP/CUDA/SYCL and add the missing sources.

Example: `sw4ck-kokkos` needed to compile `curvilinear4sg.cpp` and `utils.cpp`
in addition to `main.cpp`.

### `No such file or directory`

Check whether it is:

- an external dependency header (`mpi.h`, `gsl/gsl_math.h`,
  `KokkosBlas3_gemm.hpp`),
- a missing dataset,
- a support file that should be copied from OMP/CUDA/SYCL.

### `Kokkos::deep_copy` Deduction Errors

Avoid unmanaged `View<..., HostSpace, MemoryUnmanaged>` when deduction becomes
fragile. Prefer explicit mirrors:

```cpp
auto h = Kokkos::create_mirror_view(device_view);
for (...) h(i) = host_ptr[i];
Kokkos::deep_copy(device_view, h);
```

For device-to-host copies:

```cpp
auto h = Kokkos::create_mirror_view(device_view);
Kokkos::deep_copy(h, device_view);
for (...) host_ptr[i] = h(i);
```

## Manual Coherence Review

For each suspicious benchmark:

1. Compare file layout:

   ```sh
   find src/<bench>-kokkos src/<bench>-omp src/<bench>-cuda src/<bench>-sycl -maxdepth 2 -type f | sort
   ```

2. Compare run arguments in CMake/Makefile.
3. Check whether Kokkos reads the same inputs as the reference versions.
4. Check whether final validation is the same.
5. Check whether the Kokkos code replaced a real benchmark workflow with a
   non-equivalent synthetic data generator.

## Expected Mindset

Do not merely make the benchmark compile. The goal is to enable meaningful
performance comparison between OMP/CUDA/SYCL/Kokkos.

Therefore:

- prioritize fidelity to the benchmark,
- preserve the existing architecture,
- avoid shortcuts that change input data or the application path,
- clearly document benchmarks that are set aside.
