# 実行時間 (sec.)

| name | sycl | acc | omp_nvc |
| -- | -- | -- | -- |
| accuracy | 2.48 | 2.49 | 2.55 |
| ace | 3.13 | 2.69 | 2.92 |
| adam | 7.02 | 6.68 | 6.68 |
| adamw | 14.09 | 20.24 | 20.69 |
| adjacent | 10.74 | 13.08 | 19.10 |
| adv | 4.89 | build err | 3.39 |
| aes | 2.78 | build err | 2.72 |
| affine | 7.13 | 13.00 | 12.35 |
| aidw | build err | build err | 4.42 |
| aligned-types | 5.40 | 4.41 | 4.64 |
| all-pairs-distance | 7.06 | 4.81 | 21.35 |
| amgmk | 4.29 | 2.46 | 2.45 |
| ans | 7.02 | exe err | exe error |
| aobench | 0.28 | 0.24 | 2.55 |
| aop | 8.74 | build err | exe error |
| asmooth | 5.66 | 7.25 | 7.35 |
| assert | exe err | build err | 2.54 |
| asta | build err | build err | 10.73 |
| atan2 | 5.01 | 2.61 | 2.53 |
| atomicCost | 20.59 | 18.86 | 5.35 |
| atomicIntrinsics | over 600 | exe error | 3.13 |
| atomicPerf | 16.44 | build err | 13.91 |
| atomicReduction | 5.21 | build err | 2.72 |
| attention | 4.36 | 82.97 | exe error |
| axhelm | build err | build err | build err |
| babelstream | 0.49 | 2.78 | 2.85 |
| background-subtract | 9.86 | 8.33 | 7.91 |
| backprop | 4.09 | build err | 2.45 |
| bezier-surface | 13.05 | 13.98 | 10.75 |
| bfs | 1.49 | 0.99 | 2.49 |
| bilateral | 16.75 | 205.56 | 7.77 |
| binomial | 4.83 | | -- |
| bitonic-sort | 12.63 | build err | 8.38 |
| black-scholes | 6.75 | 39.29 | 5.05 |
| blas-gemm | build err | | -- |
| bn | 4.92 | build err | exe error |
| bonds | 7.71 | 11.41 | 11.28 |
| boxfilter | 10.09 | build err | 9.55 |
| bsearch | 34.65 | build err | 31.67 |
| bspline-vgh | 7.67 | 4.77 | 4.43 |
| b+tree | 3.96 | build err | 2.41 |
| burger | 74.42 | 74.27 | 113.21 |
| bwt | 201.02 | 187.72 | exe error |
| car | 7.45 | 6.69 | 6.54 |
| cbsfil | 33.23 | 35.06 | 33.54 |
| ccsd-trpdrv | 2.57 | 2.58 | 2.60 |
| ccs | build err | build err | exe error |
| cfd | 0.58 | 1.01 | 3.09 |
| chacha20 | 6.40 | build err | 6.66 |
| channelShuffle | build err | 19.67 | 19.51 |
| channelSum | 66.86 | 65.17 | 68.33 |
| chemv | 4.91 | build err | 3.24 |
| che | 4.78 | 3.29 | 5.08 |
| chi2 | 219.39 | 222.83 | 261.09 |
| clenergy | 4.45 | 3.14 | 3.16 |
| clink | build err | 6.38 | exe error |
| cmp | exe error | build err | exe error |
| cm | exe error | build err | exe error |
| cobahh | 435.01 | 358.67 | 358.72 |
| colorwheel | 37.59 | 27.53 | 27.85 |
| columnarSolver | 9.64 | build err | 10.09 |
| complex | 20.01 | 12.71 | 12.94 |
| compute-score | 5.00 | build err | 4.35 |
| concat | build err | 16.79 | 17.35 |
| contract | 13.01 | 28.91 | 10.81 |
| conversion | 6.76 | 3.52 | 3.43 |
| convolution1D | over 600 | build err | over 600 |
| convolution3D | 0.32 | 3.17 | 2.60 |
| convolutionSeparable | 13.72 | build err | 22.11 |
| cooling | over 600 | over 600 | over 600 |
| crc64 | exe error | 2.52 | 0.23 |
| cross | 15.72 | 8.28 | 7.66 |
| crs | 26.47 | build err | over 600 |
| d2q9-bgk | 0.81 | build err | 4.12 |
| damage | 25.33 | build err | 243.25 |
| dct8x8 | 119.46 | build err | exe error |
| ddbp | 18.41 | build err | 14.86 |
| debayer | 6.48 | build err | 6.62 |
| degrid | 13.43 | 23.73 | 17.16 |
| dense-embedding | 170.40 | build err | 166.89 |
| depixel | over 600 | build err | over 600 |
| deredundancy | 15.77 | build err | 16.03 |
| diamond | exe error | build err | 37.47 |
| distort | 4.29 | 2.49 | 2.44 |
| divergence | 6.91 | 3.17 | |
| doh | 14.92 | 2.40 | 2.39 |
| dp | build err | 24.99 | 25.20 |
| dslash | 4.41 | 2.63 | 2.66 |
| dxtc2 | 2.48 | build err | over 600 |
| easyWave | 1.80 | 2.81 | 3.02 |
| ecdh | 12.94 | 18.21 | 19.19 |
| eigenvalue | 23.71 | | |
| entropy | 47.12 | build err | 49.70 |
| epistasis | 55.50 | 258.02 | 110.39 |
| expdist | 9.20 | build err | 12.38 |
| extend2 | 6.89 | build err | 4.99 |
| extrema | 6.77 | 7.08 | 7.07 |
| face | 2.70 | 2.33 | exe error |
| fdtd3d | 31.50 | build err | 42.19 |
| feynman-kac | 43.54 | 47.14 | 22.27 |
| fft | 4.93 | build err | 3.20 |
| fhd | 184.03 | over 600 | exe error |
| filter | build err | build err | 155.75 |
| flip | build err | 15.14 | 16.78 |
| floydwarshall | 75.35 | 72.61 | 82.75 |
| fluidSim | 17.43 | 17.08 | 17.38 |
| fpc | 66.75 | build err | exe error |
| fpdc | build err | build err | over 600 |
| fresnel | 5.94 | 6.63 | 6.63 |
| frna | 107.16 | build err | exe error |
| fwt | 1.00 | build err | 3.35 |
| gabor | 21.62 | 21.30 | 22.22 |
| gamma-correction | 64.06 | 105.92 | 104.75 |
| ga | 37.89 | 32.97 | 32.89 |
| gaussian | 4.55 | 4.02 | 4.04 |
| gc | 2.44 | build err | build err |
| gd | 12.97 | 12.70 | 12.83 |
| geglu | 92.27 | 22.04 | 21.92 |
| geodesic | 4.20 | 2.52 | 2.54 |
| glu | 108.91 | 136.70 | 127.86 |
| gmm | 32.92 | build err | 37.19 |
| goulash | 25.00 | 36.18 | 37.08 |
| gpp | 6.13 | 4.13 | 4.14 |
| grep | 30.68 | build err | 1.76 |
| grrt | 16.67 | build err | 12.79 |
| haccmk | 6.16 | 2.48 | 4.44 |
| hausdorff | 33.81 | build err | 32.02 |
| haversine | 2.22 | 2.22 | 2.44 |
| heartwall | exe error | build err | exe error |
| heat2d | build err | 25.62 | 4.22 |
| heat | 23.77 | build err | 16.27 |
| hellinger | 6.59 | 6.89 | 4.59 |
| henry | 5.50 | 2.66 | 3.08 |
| hexciton | 7.78 | build err | 4.95 |
| histogram | 4.49 | build err | 3.25 |
| hmm | 10.41 | 14.95 | 3.03 |
| hogbom | 0.42 | build err | 2.82 |
| hotspot3D | 16.23 | 16.38 | 17.97 |
| hwt1d | 5.21 | build err | 3.48 |
| hybridsort | 15.20 | build err | 13.01 |
| hypterm | 5.84 | 13.31 | 13.00 |
| idivide | 10.18 | 7.93 | 9.39 |
| interleave | 5.81 | 4.06 | 3.47 |
| interval | 8.99 | 13.51 | 10.74 |
| inversek2j | build err | over 600 | 14.96 |
| ising | build err | 53.66 | 54.21 |
| iso2dfd | exe error | 15.37 | 14.36 |
| jacobi | 4.29 | 3.62 | 3.47 |
| jenkins-hash | 6.46 | 5.19 | 5.28 |
| kalman | over 600 | build err | exe error |
| keccaktreehash | 8.29 | 11.42 | 9.99 |
| keogh | 8.78 | 5.73 | 5.83 |
| kernelLaunch | 16.57 | build err | 73.59 |
| kmeans | exe error | 25.78 | 20.98 |
| knn | 4.70 | build err | 4.57 |
| lanczos | 2.69 | build err | 2.80 |
| langevin | build err | 6.19 | 4.94 |
| langford | 4.45 | build err | build err |
| laplace3d | 65.39 | 14.51 | 14.65 |
| laplace | build err | 11.58 | 11.16 |
| lavaMD | 101.76 | build err | 112.48 |
| layout | 4.86 | 2.72 | 2.87 |
| lci | 11.81 | 4.37 | 4.36 |
| lda | 4.85 | build err | build err |
| ldpc | 4.98 | build err | 9.45 |
| lebesgue | 43.01 | 39.24 | 39.63 |
| leukocyte | build err | build err | 0.92 |
| libor | 4.48 | 13.24 | 2.83 |
| lid-driven-cavity | 12.37 | build err | 22.51 |
| lif | 115.87 | 125.69 | 126.46 |
| linearprobing | 105.10 | build err | build err |
| log2 | exe error | 2.75 | exe error |
| lombscargle | 0.96 | 3.05 | |
| loopback | 5.33 | build err | 8.13 |
| lrn | 107.91 | 110.82 | 112.27 |
| lr | 8.19 | build err | exe error |
| lsqt | 20.03 | | |
| lud | 6.21 | build err | 46.13 |
| lulesh | 13.17 | 11.66 | 11.35 |
| mallocFree | 5.47 | 2.74 | build err |
| mandelbrot | 5.34 | 3.82 | 5.59 |
| mask | build err | 35.53 | 27.27 |
| match | 44.09 | build err | 41.42 |
| matern | 15.38 | build err | 99.88 |
| matrix-rotate | 35.18 | 19.10 | 35.40 |
| maxFlops | 27.29 | 26.54 | 25.41 |
| maxpool3d | 33.66 | 32.35 | 32.04 |
| mcmd | 9.10 | 10.78 | 10.98 |
| mcpr | 13.78 | build err | 14.94 |
| md5hash | 16.82 | 15.40 | 15.39 |
| mdh | build err | 34.88 | 2.67 |
| md | 14.41 | 14.22 | 14.07 |
| meanshift | 4.40 | build err | 4.48 |
| medianfilter | 5.64 | build err | 4.70 |
| memcpy | 7.49 | 6.55 | 6.56 |
| memtest | 20.55 | 19.83 | 19.45 |
| merge | over 600 | build err | over 600 |
| metropolis | 11.88 | build err | 65.27 |
| michalewicz | 151.69 | over 600 | 62.87 |
| minibude | 4.95 | build err | 3.53 |
| miniFE | | | -- |
| minimap2 | 4.27 | | 3.84 |
| minisweep | 58.65 | 35.50 | 35.87 |
| miniWeather | build err | 33.04 | 33.23 |
| minkowski | 24.00 | 22.52 | 22.11 |
| mis | 14.83 | build err | 2.45 |
| mixbench | 3.60 | build err | 3.85 |
| morphology | 2.72 | build err | over 600 |
| mrc | 82.56 | 17.78 | 17.63 |
| mriQ | 7.14 | 5.09 | 5.40 |
| mr | 3.36 | build err | 3.34 |
| mt | 2.45 | 4.23 | 4.38 |
| multimaterial | 14.61 | 12.08 | 21.98 |
| murmurhash3 | 4.65 | 2.71 | 2.72 |
| myocyte | 2.93 | build err | exe error |
| nbody | over 600 | over 600 | over 600 |
| ne | 5.29 | 5.26 | 5.95 |
| nlll | 3.09 | build err | 3.10 |
| nms | 0.40 | build err | 2.71 |
| nn | 4.23 | 2.57 | 2.51 |
| norm2 | build err | 3.71 | 5.00 |
| nqueen | 8.71 | 23.81 | 6.86 |
| ntt | 6.59 | build err | 6.64 |
| nw | 27.16 | build err | 34.48 |
| openmp | build err | build err | 13.65 |
| overlay | 7.13 | 8.96 | 10.61 |
| p4 | 5.52 | build err | 16.63 |
| page-rank | build err | 49.57 | 47.58 |
| particle-diffusion | 11.74 | 12.60 | 11.80 |
| particlefilter | 196.26 | build err | 219.78 |
| particles | 1.32 | build err | over 600 |
| pathfinder | 63.99 | build err | exe error |
| permutate | 24.33 | build err | 25.76 |
| permute | 61.12 | 86.54 | 85.98 |
| perplexity | 190.17 | 122.81 | 122.75 |
| phmm | 5.41 | build err | 7.61 |
| pnpoly | 21.12 | build err | 74.18 |
| pns | 7.27 | build err | exe error |
| pointwise | 21.36 | 80.05 | 51.51 |
| pool | 53.22 | 40.55 | 26.23 |
| popcount | 165.37 | 278.41 | 182.43 |
| present | 4.17 | 3.04 | 3.03 |
| prna | build err | build err | exe error |
| projectile | 2.15 | 2.56 | 2.57 |
| pso | 4.63 | build err | 2.72 |
| qrg | 7.85 | 8.63 | 10.22 |
| qtclustering | 36.14 | build err | exe error |
| quantBnB | 119.86 | 124.66 | 122.89 |
| quicksort | 37.63 | build err | 33.53 |
| radixsort | 4.81 | build err | 8.14 |
| rainflow | 30.44 | 30.15 | 32.16 |
| randomAccess | 8.68 | build err | 24.53 |
| reaction | 5.87 | build err | 9.21 |
| recursiveGaussian | 1.96 | build err | 4.23 |
| resize | 3.76 | 6.03 | 5.99 |
| reverse | 5.31 | build err | 8.03 |
| rfs | 47.71 | 48.32 | 46.12 |
| rng-wallace | 4.43 | build err | 3.55 |
| rodrigues | 103.33 | 102.26 | 101.55 |
| romberg | 4.41 | build err | 2.80 |
| rsbench | 5.02 | build err | exe error |
| rsc | 0.43 | build err | 2.75 |
| rtm8 | 18.95 | 15.75 | 4.20 |
| rushlarsen | over 600 | over 600 | over 600 |
| s3d | 4.85 | build err | 2.89 |
| s8n | 4.32 | 2.57 | 2.49 |
| sad | build err | exe err | build err |
| sampling | 6.75 | build err | 5.92 |
| scan2 | 4.55 | build err | 6.79 |
| scan | over 600 | build err | over 600 |
| scatterAdd | 8.13 | 7.93 | 7.90 |
| scel | 317.39 | 11.89 | 11.14 |
| secp256k1 | 4.52 | 2.81 | 2.72 |
| sheath | 7.70 | 5.80 | 5.76 |
| shmembench | 10.98 | build err | exe err |
| simplemoc | 399.87 | 294.12 | 293.60 |
| simpleSpmv | 317.86 | 315.32 | 316.96 |
| slu | build err | build err | build err |
| snake | 8.39 | 10.35 | 10.47 |
| sobel | 4.74 | 3.12 | 3.51 |
| sobol | 5.51 | build err | 4.52 |
| softmax-online | 13.73 | build err | 15.80 |
| softmax | 39.66 | 40.66 | 37.36 |
| sort | build err | build err | 36.90 |
| sosfil | 6.46 | build err | 13.49 |
| sph | 7.02 | 22.61 | 5.39 |
| split | 35.04 | build err | 37.03 |
| spm | 103.26 | 94.87 | 94.80 |
| sptrsv | 4.47 | build err | 2.71 |
| srad | 58.15 | build err | exe error |
| ss | 5.01 | build err | 18.27 |
| stddev | 48.71 | build err | 49.55 |
| stencil1d | 6.48 | 10.83 | build err |
| stencil3d | 4.41 | build err | 4.23 |
| streamcluster | exe error | build err | over 600 |
| su3 | exe error | 12.20 | 14.08 |
| surfel | build err | 117.62 | 118.90 |
| svd3x3 | exe error | 1.92 | build err |
| sw4ck | exe error | 28.81 | exe error |
| swish | 21.91 | 5.43 | 5.42 |
| tensorT | 4.42 | build err | 2.88 |
| testSNAP | 4.14 | exe error | 4.05 |
| thomas | 136.01 | 47.56 | 59.43 |
| threadfence | 27.80 | build err | 35.10 |
| tissue | 13.35 | 11.60 | 18.00 |
| tonemapping | 4.95 | 2.95 | 3.18 |
| tqs | 4.14 | build err | 2.64 |
| triad | 4.41 | 2.69 | 2.72 |
| tridiagonal | 8.19 | build err | 24.24 |
| tsa | 44.03 | build err | 27.23 |
| tsp | 5.74 | build err | build err |
| urng | 2.40 | 2.19 | 2.47 |
| vanGenuchten | 21.05 | 31.08 | 31.30 |
| vmc | 4.38 | build err | exe error |
| vol2col | 5.23 | 7.86 | 8.16 |
| wedford | build err | build err | exe error |
| winograd | 0.56 | 3.11 | 3.15 |
| wlcpow | 6.37 | build err | 12.24 |
| wordcount | build err | 8.17 | 7.94 |
| wsm5 | 6.52 | 5.52 | 5.75 |
| wyllie | 111.04 | 112.29 | over 600 |
| xlqc | exe error | build err | 0.27 |
| xsbench | 38.76 | 37.04 | 3.84 |
| zeropoint | 51.34 | 21.47 | 21.45 |
| zmddft | 4.31 | build err | 14.12 |
| | | | |
| completed | 270 | 171 | 264 |

sycl と acc が動作した件数 148
