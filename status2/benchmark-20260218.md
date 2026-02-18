# 実行時間 (sec.)

| name | cuda | sycl | acc | omp_nvc | 分類 |
| -- | -- | -- | -- | -- | -- |
| accuracy | 2.54 | 2.49 | 2.48 | 2.53 | A |
| ace | 2.90 | 3.13 | 2.84 | 2.79 | A |
| adam | 5.20 | 5.24 | 6.72 | 6.70 | A |
| adamw | 13.20 | 12.32 | 20.30 | 20.34 | A |
| adjacent | 7.66 | 9.13 | 7.03 | 15.22 | A |
| adv | 3.05 | 3.35 | build err | 3.49 | B |
| aes | 2.72 | 2.79 | build err | 2.70 | B |
| affine | 5.62 | 5.42 | 12.78 | 12.37 | A |
| aidw | 4.58 | 12.87 | 5.42 | 6.21 | B |
| aligned-types | 3.82 | 3.68 | 4.40 | 4.71 | A |
| all-pairs-distance | 4.96 | 5.23 | 4.81 | 21.34 | B |
| amgmk | 2.51 | 2.45 | 2.42 | 2.44 | B |
| ans | 5.19 | 5.42 | exe err | exe err | A |
| aobench | 2.46 | 0.36 | 2.42 | 2.47 | A |
| aop | 7.03 | 7.02 | build err | build err | B |
| asmooth | 12.47 | 5.70 | 7.38 | 7.33 | A |
| assert | 2.51 | exe err | build err | 2.58 | B |
| asta | 4.36 | build err | build err | 10.71 | B |
| atan2 | 2.68 | 3.24 | 2.52 | 2.54 | A |
| atomicCost | 16.01 | 18.89 | 18.84 | 5.36 | A |
| atomicIntrinsics | 3.90 | over 600 | 3.18 | 3.21 | A |
| atomicPerf | 14.33 | 14.66 | 13.47 | 13.88 | B |
| atomicReduction | 2.70 | 3.44 | 2.65 | 2.63 | A |
| attention | 2.48 | 2.47 | 83.27 | 83.85 | A |
| axhelm | 2.46 | 2.41 | build err | 2.47 | B |
| babelstream | 2.74 | 0.56 | 2.72 | 2.85 | A |
| background-subtract | 6.52 | 8.07 | 8.66 | 8.33 | A |
| backprop | 2.53 | 2.45 | 2.44 | 2.47 | B |
| bezier-surface | 12.43 | 12.93 | 14.25 | 10.71 | A |
| bfs | 0.98 | 1.03 | 2.48 | 2.46 | A |
| bilateral | 7.64 | 15.09 | 205.59 | 7.85 | A |
| binomial | 3.24 | 3.10 | | -- | B |
| bitonic-sort | 13.52 | 10.81 | 7.80 | 8.49 | A |
| black-scholes | 4.80 | 5.00 | 39.24 | 4.88 | A |
| blas-gemm | 5.70 | build err | | -- | A |
| bn | 4.47 | 5.00 | build err | exe err | B |
| bonds | 11.93 | 7.66 | 11.50 | 11.19 | A |
| boxfilter | 9.06 | 8.46 | 6.47 | 9.55 | B |
| bsearch | 33.96 | 34.85 | 32.68 | 31.84 | B |
| bspline-vgh | 9.73 | 7.74 | 4.73 | 4.81 | A |
| b+tree | 2.87 | 2.46 | 2.46 | 2.46 | A |
| burger | 84.00 | 72.44 | 68.61 | 112.53 | A |
| bwt | 204.75 | 199.29 | 188.93 | exe err | A |
| car | 7.41 | 7.18 | 6.64 | 6.65 | A |
| cbsfil | 32.08 | 31.48 | 35.01 | 33.19 | A |
| ccsd-trpdrv | 2.61 | 2.67 | 2.55 | 2.63 | A |
| ccs | 3.36 | 3.57 | 2.97 | 3.65 | B |
| cfd | 0.82 | 0.58 | 3.02 | 3.08 | A |
| chacha20 | 4.74 | 4.68 | 9.24 | 6.66 | B |
| channelShuffle | 18.46 | 538.46 | 19.23 | 19.24 | A |
| channelSum | 65.58 | 64.97 | 65.55 | 68.17 | A |
| chemv | 3.21 | 3.20 | 2.85 | 3.21 | B |
| che | 3.62 | 3.05 | 3.27 | 3.38 | A |
| chi2 | 197.09 | 217.04 | 220.78 | 251.54 | A |
| clenergy | 2.68 | 2.66 | 3.13 | 3.16 | A |
| clink | 4.98 | 56.20 | 8.31 | 8.15 | A |
| cmp | exe err | exe err | build err | exe err | A |
| cm | exe err | exe err | build err | exe err | B |
| cobahh | 487.96 | 406.65 | 383.64 | 388.03 | A |
| colorwheel | 39.48 | 37.38 | 27.55 | 27.96 | A |
| columnarSolver | 9.83 | 10.16 | 16.87 | 10.22 | B |
| complex | 23.04 | 22.28 | 12.59 | 12.96 | A |
| compute-score | 3.28 | 3.24 | 3.27 | 4.34 | B |
| concat | 17.45 | 15.08 | 16.24 | 15.73 | A |
| contract | 9.28 | 11.40 | 28.95 | 10.82 | A |
| conversion | 5.09 | 4.98 | 3.56 | 3.44 | A |
| convolution1D | 382.64 | 380.96 | 333.42 | 383.27 | B |
| convolution3D | 2.55 | 2.48 | 3.17 | 2.51 | A |
| convolutionSeparable | 13.37 | 13.58 | build err | 22.05 | B |
| cooling | 381.03 | 154.60 | 452.57 | 454.09 | A |
| crc64 | 56.48 | 58.47 | 57.16 | 57.25 | A |
| cross | 12.82 | 14.58 | 7.67 | 8.04 | A |
| crs | 25.43 | 24.78 | build err | over 600 | B |
| d2q9-bgk | 2.59 | 2.85 | build err | 4.37 | B |
| damage | 29.66 | 24.45 | 36.68 | 243.28 | B |
| dct8x8 | 85.64 | 130.74 | build err | build err | B |
| ddbp | 16.32 | 17.41 | build err | 17.27 | B |
| debayer | 6.75 | 6.37 | 6.68 | 6.79 | B |
| degrid | 14.49 | 14.97 | 23.24 | 17.04 | A |
| dense-embedding | 112.61 | 142.22 | 99.47 | 184.08 | B |
| depixel | 105.99 | 157.40 | 208.93 | 168.04 | A |
| deredundancy | 16.29 | 17.89 | 15.82 | 15.76 | B |
| diamond | 36.76 | build err | build err | 37.30 | A |
| distort | 2.52 | 2.45 | 2.45 | 2.50 | A |
| divergence | 4.74 | 5.32 | 3.15 | 3.15 | A |
| doh | 14.61 | 14.97 | 2.39 | 2.41 | A |
| dp | 53.24 | build err | 25.04 | 25.25 | A |
| dslash | 2.68 | 2.70 | 2.68 | 2.66 | A |
| dxtc2 | 2.49 | 2.49 | build err | over 600 | B |
| easyWave | 3.09 | 2.97 | 2.79 | 3.08 | A |
| ecdh | 14.96 | 11.26 | 18.09 | 19.13 | A |
| eigenvalue | 21.76 | 19.92 | over 600 | 22.02 | A |
| entropy | 48.31 | 47.60 | 94.05 | 49.43 | B |
| epistasis | 54.59 | 53.82 | 259.15 | 110.41 | A |
| expdist | 7.57 | 7.45 | build err | 12.43 | B |
| extend2 | 5.80 | 5.17 | 4.68 | 4.94 | A |
| extrema | 7.05 | 6.75 | 6.99 | 7.10 | A |
| face | 2.72 | 2.64 | 2.45 | exe err | A |
| fdtd3d | 30.14 | 31.50 | build err | 43.29 | B |
| feynman-kac | 81.22 | 41.84 | 48.24 | 22.16 | A |
| fft | 3.17 | 3.19 | build err | 3.19 | B |
| fhd | over 600 | 183.93 | over 600 | exe err | A |
| filter | 168.70 | build err | build err | 154.77 | B |
| flip | 10.43 | 37.32 | 14.45 | 15.50 | A |
| floydwarshall | 74.91 | 73.45 | 72.59 | 83.41 | A |
| fluidSim | 19.45 | 17.67 | 17.30 | 17.36 | A |
| fpc | 67.41 | 66.38 | build err | exe err | B |
| fpdc | 4.82 | 5.37 | build err | over 600 | B |
| fresnel | 6.64 | 6.00 | 6.61 | 6.61 | A |
| frna | 103.73 | 104.63 | build err | build err | B |
| fwt | 3.13 | 3.12 | 3.35 | 3.47 | B |
| gabor | 21.34 | 21.57 | 21.27 | 57.42 | A |
| gamma-correction | 68.53 | 63.44 | 105.30 | 104.46 | A |
| ga | 48.84 | 36.53 | 32.72 | 33.46 | A |
| gaussian | 15.65 | 4.12 | 3.96 | 4.07 | A |
| gc | 2.59 | 2.41 | build err | build err | A |
| gd | 12.38 | 13.02 | 12.66 | 12.83 | A |
| geglu | 27.67 | 94.40 | 22.13 | 22.07 | A |
| geodesic | 2.54 | 2.60 | 2.51 | 2.52 | A |
| glu | 59.42 | 121.58 | 124.43 | 120.93 | A |
| gmm | 35.58 | 35.97 | build err | 37.28 | B |
| goulash | 44.04 | 27.35 | 36.03 | 36.52 | A |
| gpp | 5.15 | 4.29 | 4.10 | 4.15 | A |
| grep | 55.53 | 29.24 | build err | 1.72 | B |
| grrt | 14.22 | 14.54 | 13.31 | 12.82 | A |
| haccmk | 4.70 | 4.33 | 2.57 | 4.42 | A |
| hausdorff | 30.92 | 30.37 | 32.21 | 31.92 | A |
| haversine | 2.22 | 2.18 | 2.49 | 2.43 | A |
| heartwall | 7.48 | exe err | build err | build err | A |
| heat2d | 6.14 | 61.50 | 25.22 | 5.92 | A |
| heat | 22.95 | 22.04 | 15.34 | 16.22 | B |
| hellinger | 4.94 | 4.80 | 6.88 | 4.52 | A |
| henry | 3.91 | 3.77 | 2.86 | 2.74 | A |
| hexciton | 4.19 | 6.03 | build err | 4.96 | B |
| histogram | 3.06 | 2.98 | build err | 3.30 | B |
| hmm | 4.22 | 8.56 | 14.71 | 2.81 | A |
| hogbom | 0.53 | 2.67 | 2.65 | 2.76 | B |
| hotspot3D | 18.45 | 16.21 | 18.11 | 18.00 | A |
| hwt1d | 3.15 | 3.13 | 3.27 | 3.23 | B |
| hybridsort | 12.83 | 15.27 | build err | 12.96 | B |
| hypterm | 5.75 | 5.86 | 13.14 | 13.12 | A |
| idivide | 6.12 | 8.31 | 8.02 | 9.39 | A |
| interleave | 4.15 | 4.13 | 4.07 | 3.44 | A |
| interval | 7.59 | 7.15 | 13.47 | 10.78 | A |
| inversek2j | 2.84 | 1.05 | 225.58 | 111.49 | A |
| ising | 10.77 | 258.64 | 13.36 | 13.31 | A |
| iso2dfd | 17.76 | 17.67 | 13.97 | 14.20 | A |
| jacobi | 2.80 | 2.77 | 3.56 | 3.50 | A |
| jenkins-hash | 5.05 | 4.98 | 5.14 | 5.17 | A |
| kalman | 335.35 | over 600 | build err | exe err | A |
| keccaktreehash | 7.76 | 8.26 | 11.46 | 10.08 | A |
| keogh | 22.46 | 7.03 | 5.80 | 5.89 | A |
| kernelLaunch | 13.80 | 14.84 | 281.99 | 73.99 | B |
| kmeans | 28.57 | 28.93 | 27.05 | 22.68 | B |
| knn | 4.43 | 4.48 | 4.28 | 4.53 | B |
| lanczos | 2.61 | 2.61 | build err | 2.81 | B |
| langevin | 21.81 | 17.73 | 6.26 | 31.31 | A |
| langford | 2.76 | 2.54 | build err | build err | B |
| laplace3d | 65.95 | 64.92 | 14.60 | 14.64 | A |
| laplace | 10.89 | 333.46 | 11.93 | 11.34 | A |
| lavaMD | 104.27 | 102.55 | build err | 116.55 | B |
| layout | 3.28 | 3.27 | 2.66 | 2.93 | A |
| lci | 17.78 | 10.00 | 4.36 | 4.36 | A |
| lda | 3.10 | 3.10 | build err | build err | B |
| ldpc | 4.83 | 4.91 | build err | 9.60 | B |
| lebesgue | 39.25 | 41.65 | 39.43 | 39.11 | A |
| leukocyte | 0.64 | build err | build err | 0.92 | A |
| libor | 2.87 | 2.87 | 13.25 | 2.84 | A |
| lid-driven-cavity | 10.21 | 10.91 | build err | 22.40 | B |
| lif | 100.17 | 116.49 | 125.87 | 124.42 | A |
| linearprobing | 105.14 | 102.65 | build err | build err | A |
| log2 | 2.80 | 0.74 | 2.77 | 2.81 | A |
| lombscargle | 3.14 | 2.99 | 3.06 | 3.05 | A |
| loopback | 5.73 | 5.33 | 5.38 | 8.16 | B |
| lrn | 104.40 | 105.73 | 111.48 | 78.63 | A |
| lr | 6.18 | 6.36 | build err | build err | B |
| lsqt | 19.97 | 19.99 | 32.63 | 32.86 | A |
| lud | 5.54 | 8.19 | 9.54 | 46.07 | B |
| lulesh | 7.81 | 11.28 | 11.70 | 8.41 | A |
| mallocFree | 3.72 | 3.69 | 2.68 | 2.67 | A |
| mandelbrot | 5.75 | 5.29 | 3.75 | 3.82 | A |
| mask | 22.03 | 55.31 | 21.60 | 29.68 | A |
| match | 40.66 | 42.51 | build err | 41.46 | B |
| matern | 6.11 | 6.24 | 3.85 | 31.37 | B |
| matrix-rotate | 32.32 | 35.98 | 35.18 | 35.00 | A |
| maxFlops | 25.53 | 25.56 | 26.46 | 25.46 | A |
| maxpool3d | 32.54 | 33.75 | 32.80 | 32.24 | A |
| mcmd | 12.38 | 9.03 | 10.79 | 11.06 | A |
| mcpr | 25.42 | 14.07 | 21.86 | 15.00 | B |
| md5hash | 15.45 | 15.14 | 15.35 | 15.50 | A |
| mdh | 2.80 | 48.13 | 35.15 | 2.67 | A |
| md | 14.56 | 14.68 | 14.24 | 14.22 | A |
| meanshift | 3.76 | 4.42 | 4.71 | 4.40 | B |
| medianfilter | 3.54 | 3.75 | build err | 4.68 | B |
| memcpy | 5.61 | 3.54 | 6.52 | 6.54 | A |
| memtest | 18.43 | 18.85 | 19.78 | 19.45 | A |
| merge | 432.85 | 442.02 | build err | 425.74 | B |
| metropolis | 12.73 | 14.03 | build err | 65.32 | B |
| michalewicz | 43.63 | 150.55 | over 600 | 63.30 | A |
| minibude | 2.89 | 3.31 | build err | 3.54 | B |
| miniFE | | | | -- | A |
| minimap2 | 1.51 | 2.32 | | 2.19 | A |
| minisweep | 50.83 | 56.83 | 35.43 | 35.90 | A |
| miniWeather | 8.49 | 8.19 | 33.30 | 33.14 | A |
| minkowski | 23.94 | 24.25 | 22.27 | 22.36 | A |
| mis | 24.24 | 13.22 | build err | 2.60 | B |
| mixbench | 3.77 | 3.62 | build err | 3.80 | B |
| morphology | 5.03 | 4.81 | build err | over 600 | B |
| mrc | 21.72 | 83.40 | 17.88 | 17.83 | A |
| mriQ | 4.76 | 5.31 | 5.04 | 5.21 | A |
| mr | 3.31 | 3.33 | build err | 3.31 | A |
| mt | 5.66 | 4.65 | 4.29 | 4.08 | A |
| multimaterial | 28.40 | 14.67 | 12.11 | 21.91 | B |
| murmurhash3 | 2.78 | 2.80 | 2.67 | 2.69 | A |
| myocyte | 33.92 | 3.03 | build err | exe err | A |
| nbody | 23.95 | 25.24 | 19.63 | 19.61 | A |
| ne | 4.83 | 5.37 | 4.90 | 4.94 | A |
| nlll | 3.07 | 3.08 | build err | 3.06 | B |
| nms | 2.60 | 2.59 | build err | 2.73 | B |
| nn | 2.55 | 2.47 | 2.56 | 2.52 | A |
| norm2 | 3.77 | build err | 3.71 | 4.97 | A |
| nqueen | 6.31 | 6.94 | 23.85 | 6.87 | A |
| ntt | 6.21 | 6.63 | build err | 6.65 | B |
| nw | 16.68 | 26.82 | build err | 33.48 | B |
| openmp | 13.88 | 81.27 | build err | 13.65 | B |
| overlay | 5.87 | 5.33 | 9.20 | 10.50 | A |
| p4 | 3.78 | 3.77 | build err | 16.94 | B |
| page-rank | 50.87 | 375.35 | 46.20 | 46.72 | A |
| particle-diffusion | 11.92 | 11.94 | 13.04 | 12.23 | A |
| particlefilter | 222.50 | 198.39 | build err | 219.65 | B |
| particles | 3.67 | 3.46 | build err | over 600 | B |
| pathfinder | 63.24 | 64.27 | build err | exe err | B |
| permutate | 26.61 | 24.87 | build err | 25.87 | B |
| permute | 61.55 | 61.55 | 85.77 | 85.40 | A |
| perplexity | 186.41 | 192.24 | 97.16 | 96.53 | A |
| phmm | 3.72 | 3.53 | build err | 7.57 | B |
| pnpoly | 7.07 | 21.12 | build err | 74.17 | A |
| pns | 5.50 | 5.44 | build err | exe err | B |
| pointwise | 21.34 | 21.28 | 53.08 | 75.86 | A |
| pool | 27.23 | 51.58 | 41.75 | 41.00 | A |
| popcount | 241.58 | 178.30 | 279.36 | 134.99 | A |
| present | 2.87 | 2.67 | 3.04 | 3.05 | A |
| prna | 79.29 | build err | build err | exe err | B |
| projectile | 2.66 | 0.91 | 2.66 | 2.55 | A |
| pso | 2.93 | 2.83 | build err | 2.69 | A |
| qrg | 7.23 | 6.30 | 8.57 | 10.24 | A |
| qtclustering | 35.55 | 34.48 | build err | exe err | B |
| quantBnB | 18.55 | 121.16 | 125.38 | 125.68 | A |
| quicksort | 36.33 | 36.99 | build err | 33.98 | B |
| radixsort | 2.94 | 2.90 | build err | 8.11 | B |
| rainflow | 29.26 | 29.99 | 29.64 | 29.52 | A |
| randomAccess | 6.66 | 6.90 | build err | 24.54 | A |
| reaction | 4.18 | 4.08 | build err | 9.17 | B |
| recursiveGaussian | 4.03 | 4.04 | build err | 4.27 | B |
| resize | 6.38 | 5.98 | 6.03 | 6.01 | A |
| reverse | 3.47 | 3.67 | build err | 8.08 | B |
| rfs | 46.68 | 46.70 | 48.57 | 48.26 | A |
| rng-wallace | 2.62 | 2.56 | build err | 3.55 | B |
| rodrigues | 96.39 | 101.46 | 114.74 | 101.38 | A |
| romberg | 2.75 | 2.72 | build err | 2.81 | B |
| rsbench | 3.01 | 3.39 | build err | build err | B |
| rsc | 2.75 | 0.50 | build err | 2.73 | B |
| rtm8 | 4.35 | 17.67 | 15.76 | 4.22 | A |
| rushlarsen | 140.36 | 129.52 | 135.46 | 135.06 | A |
| s3d | 2.92 | 2.94 | build err | 2.90 | A |
| s8n | 2.58 | 2.49 | 2.47 | 2.43 | A |
| sad | 2.40 | 2.48 | 2.63 | 81.81 | A |
| sampling | 5.85 | 5.77 | build err | 5.65 | B |
| scan2 | 2.93 | 2.91 | build err | 6.76 | B |
| scan | 33.88 | 31.87 | build err | 33.05 | B |
| scatterAdd | 7.95 | 8.10 | 8.05 | 7.79 | A |
| scel | 17.04 | 322.81 | 11.91 | 11.29 | A |
| secp256k1 | 2.77 | 2.68 | 2.79 | 2.69 | A |
| sheath | 5.92 | 5.85 | 5.81 | 5.82 | A |
| shmembench | 5.25 | 9.16 | build err | exe err | B |
| simplemoc | 233.38 | 400.17 | 293.35 | 294.34 | B |
| simpleSpmv | 317.74 | 319.81 | 320.00 | 317.47 | A |
| slu | | build err | build err | build err | B |
| snake | 8.17 | 8.43 | 10.57 | 10.42 | A |
| sobel | 2.81 | 2.89 | 3.18 | 3.32 | A |
| sobol | 4.71 | 3.89 | build err | 4.51 | B |
| softmax-online | 11.10 | 13.69 | build err | 15.10 | B |
| softmax | 44.19 | 39.36 | 40.91 | 36.96 | A |
| sort | 4.69 | build err | build err | 36.92 | B |
| sosfil | 4.65 | 4.75 | build err | 13.32 | B |
| sph | 5.45 | 5.41 | 22.53 | 5.40 | A |
| split | 35.89 | 35.78 | build err | 37.73 | B |
| spm | 100.86 | 104.39 | 98.00 | 98.41 | A |
| sptrsv | 2.44 | 2.70 | build err | 2.81 | B |
| srad | 36.80 | 37.49 | build err | exe err | B |
| ss | 7.45 | 7.23 | 9.95 | 18.10 | B |
| stddev | 48.33 | 46.55 | build err | 51.00 | B |
| stencil1d | 3.75 | 4.68 | 30.81 | 35.77 | A |
| stencil3d | 4.01 | 4.35 | build err | 4.24 | B |
| streamcluster | 5.89 | 11.84 | build err | 26.24 | B |
| su3 | 2.99 | 5.17 | 7.47 | 7.48 | A |
| surfel | 13.03 | 510.01 | 12.02 | 11.97 | A |
| svd3x3 | 1.91 | 2.00 | 2.86 | 2.89 | A |
| sw4ck | 3.58 | 6.40 | 29.28 | 28.11 | A |
| swish | 7.49 | 24.54 | 5.48 | 5.39 | A |
| tensorT | 2.73 | 2.56 | build err | 2.87 | B |
| testSNAP | 3.69 | 3.73 | 66.16 | 4.19 | A |
| thomas | 125.54 | 131.17 | 59.68 | 49.73 | A |
| threadfence | 25.80 | 26.09 | build err | 35.11 | B |
| tissue | 11.96 | 12.45 | 11.82 | 18.23 | A |
| tonemapping | 3.24 | 3.15 | 3.02 | 2.95 | A |
| tqs | 2.71 | 2.57 | build err | 2.59 | B |
| triad | 2.72 | 2.62 | 2.82 | 2.75 | A |
| tridiagonal | 8.42 | 8.80 | build err | 24.55 | B |
| tsa | 54.11 | 36.86 | build err | 29.09 | B |
| tsp | 5.10 | 7.82 | build err | build err | B |
| urng | 2.47 | 2.43 | 2.21 | 2.20 | B |
| vanGenuchten | 21.39 | 21.48 | 30.99 | 30.92 | A |
| vmc | 5.07 | 4.38 | build err | build err | B |
| vol2col | 8.27 | 5.26 | 7.87 | 8.13 | A |
| wedford | exe err | build err | build err | exe err | B |
| winograd | 0.84 | 0.65 | 3.25 | 3.20 | A |
| wlcpow | 4.53 | 4.50 | build err | 12.37 | B |
| wordcount | 8.03 | 7.98 | 8.18 | 7.93 | A |
| wsm5 | 5.57 | 4.78 | 5.59 | 5.73 | B |
| wyllie | 107.36 | 112.64 | 108.06 | over 600 | B |
| xlqc | build err | build err | build err | build err | B |
| xsbench | 46.19 | 37.40 | 37.18 | 3.89 | A |
| zeropoint | 18.63 | 55.54 | 21.82 | 24.23 | A |
| zmddft | 2.63 | 2.63 | build err | 14.28 | B |
| | | | | | |
| completed | 318 | 306 | 214 | 285 | |

sycl と acc がともに完了した件数 211
