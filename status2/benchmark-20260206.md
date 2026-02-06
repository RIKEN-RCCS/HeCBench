# 実行時間 (sec.)

| name | sycl | acc | omp_nvc |
| -- | -- | -- | -- |
| accuracy | 2.49 | 2.48 | 2.53 |
| ace | 3.13 | 2.84 | 2.79 |
| adam | 5.24 | 6.72 | 6.70 |
| adamw | 12.32 | 20.30 | 20.34 |
| adjacent | 9.13 | 7.03 | 15.22 |
| adv | 3.35 | build err | 3.49 |
| aes | 2.79 | build err | 2.70 |
| affine | 5.42 | 12.78 | 12.37 |
| aidw | 12.83 | build err | 4.46 |
| aligned-types | 3.68 | 4.40 | 4.71 |
| all-pairs-distance | 5.23 | 4.81 | 21.34 |
| amgmk | 2.45 | 2.42 | 2.44 |
| ans | 5.42 | exe err | exe err |
| aobench | 0.36 | 2.42 | 2.47 |
| aop | 7.02 | build err | build err |
| asmooth | 5.70 | 7.38 | 7.33 |
| assert | exe err | build err | 2.58 |
| asta | build err | build err | 10.71 |
| atan2 | 3.24 | 2.52 | 2.54 |
| atomicCost | 18.89 | 18.84 | 5.36 |
| atomicIntrinsics | over 600 | 3.18 | 3.21 |
| atomicPerf | 14.74 | build err | 13.89 |
| atomicReduction | 3.46 | build err | 2.67 |
| attention | 2.47 | 83.27 | 83.85 |
| axhelm | 2.41 | build err | 2.47 |
| babelstream | 0.56 | 2.72 | 2.85 |
| background-subtract | 8.07 | 8.66 | 8.33 |
| backprop | 2.45 | 2.44 | 2.47 |
| bezier-surface | 12.93 | 14.25 | 10.71 |
| bfs | 1.03 | 2.48 | 2.46 |
| bilateral | 15.09 | 205.59 | 7.85 |
| binomial | 3.10 | | -- |
| bitonic-sort | 10.68 | build err | 8.25 |
| black-scholes | 5.00 | 39.24 | 4.88 |
| blas-gemm | build err | | -- |
| bn | 5.00 | build err | exe err |
| bonds | 7.66 | 11.50 | 11.19 |
| boxfilter | 8.17 | build err | 9.58 |
| bsearch | 35.25 | build err | 31.81 |
| bspline-vgh | 7.74 | 4.73 | 4.81 |
| b+tree | 2.33 | build err | 2.45 |
| burger | 72.44 | 68.61 | 112.53 |
| bwt | 199.29 | 188.93 | exe err |
| car | 7.18 | 6.64 | 6.65 |
| cbsfil | 31.48 | 35.01 | 33.19 |
| ccsd-trpdrv | 2.67 | 2.55 | 2.63 |
| ccs | 3.59 | build err | 3.57 |
| cfd | 0.58 | 3.02 | 3.08 |
| chacha20 | 4.68 | 9.24 | 6.66 |
| channelShuffle | 538.46 | 19.23 | 19.24 |
| channelSum | 64.97 | 65.55 | 68.17 |
| chemv | 3.19 | build err | 3.24 |
| che | 3.05 | 3.27 | 3.38 |
| chi2 | 217.04 | 220.78 | 251.54 |
| clenergy | 2.66 | 3.13 | 3.16 |
| clink | 56.20 | 8.31 | 8.15 |
| cmp | exe err | build err | exe err |
| cm | exe err | build err | exe err |
| cobahh | 406.65 | 383.64 | 388.03 |
| colorwheel | 37.38 | 27.55 | 27.96 |
| columnarSolver | 10.16 | 16.87 | 10.22 |
| complex | 22.28 | 12.59 | 12.96 |
| compute-score | 3.19 | build err | 4.41 |
| concat | 15.08 | 16.24 | 15.73 |
| contract | 11.40 | 28.95 | 10.82 |
| conversion | 4.98 | 3.56 | 3.44 |
| convolution1D | 346.88 | build err | 353.39 |
| convolution3D | 2.48 | 3.17 | 2.51 |
| convolutionSeparable | 13.58 | build err | 22.05 |
| cooling | 154.60 | 452.57 | 454.09 |
| crc64 | 58.47 | 57.16 | 57.25 |
| cross | 14.58 | 7.67 | 8.04 |
| crs | 24.78 | build err | over 600 |
| d2q9-bgk | 2.85 | build err | 4.37 |
| damage | 24.32 | build err | 243.23 |
| dct8x8 | 130.74 | build err | build err |
| ddbp | 17.41 | build err | 17.27 |
| debayer | 6.74 | build err | 6.75 |
| degrid | 14.97 | 23.24 | 17.04 |
| dense-embedding | 166.23 | build err | 164.57 |
| depixel | 157.40 | 208.93 | 168.04 |
| deredundancy | 17.83 | build err | 15.72 |
| diamond | build err | build err | 37.30 |
| distort | 2.45 | 2.45 | 2.50 |
| divergence | 3.11 | 3.12 | 3.13 |
| doh | 14.97 | 2.39 | 2.41 |
| dp | build err | 25.04 | 25.25 |
| dslash | 2.70 | 2.68 | 2.66 |
| dxtc2 | 2.49 | build err | over 600 |
| easyWave | 2.97 | 2.79 | 3.08 |
| ecdh | 11.26 | 18.09 | 19.13 |
| eigenvalue | 19.92 | over 600 | 22.02 |
| entropy | 46.86 | build err | 49.60 |
| epistasis | 53.82 | 259.15 | 110.41 |
| expdist | 7.45 | build err | 12.43 |
| extend2 | 5.17 | 4.68 | 4.94 |
| extrema | 6.75 | 6.99 | 7.10 |
| face | 2.64 | 2.45 | exe err |
| fdtd3d | 31.50 | build err | 43.29 |
| feynman-kac | 41.84 | 48.24 | 22.16 |
| fft | 3.19 | build err | 3.19 |
| fhd | 183.93 | over 600 | exe err |
| filter | build err | build err | 154.77 |
| flip | 37.32 | 14.45 | 15.50 |
| floydwarshall | 73.45 | 72.59 | 83.41 |
| fluidSim | 17.67 | 17.30 | 17.36 |
| fpc | 66.38 | build err | exe err |
| fpdc | 5.37 | build err | over 600 |
| fresnel | 6.00 | 6.61 | 6.61 |
| frna | 104.63 | build err | build err |
| fwt | 1.08 | build err | 3.38 |
| gabor | 21.57 | 21.27 | 57.42 |
| gamma-correction | 63.44 | 105.30 | 104.46 |
| ga | 36.53 | 32.72 | 33.46 |
| gaussian | 4.12 | 3.96 | 4.07 |
| gc | 2.41 | build err | build err |
| gd | 13.02 | 12.66 | 12.83 |
| geglu | 94.40 | 22.13 | 22.07 |
| geodesic | 2.60 | 2.51 | 2.52 |
| glu | 121.58 | 124.43 | 120.93 |
| gmm | 35.97 | build err | 37.28 |
| goulash | 27.35 | 36.03 | 36.52 |
| gpp | 4.29 | 4.10 | 4.15 |
| grep | 29.24 | build err | 1.72 |
| grrt | 14.59 | build err | 12.78 |
| haccmk | 4.33 | 2.57 | 4.42 |
| hausdorff | 30.37 | 32.21 | 31.92 |
| haversine | 2.18 | 2.49 | 2.43 |
| heartwall | exe err | build err | build err |
| heat2d | 61.50 | 25.22 | 5.92 |
| heat | 22.03 | build err | 16.23 |
| hellinger | 4.80 | 6.88 | 4.52 |
| henry | 3.77 | 2.86 | 2.74 |
| hexciton | 6.03 | build err | 4.96 |
| histogram | 2.98 | build err | 3.30 |
| hmm | 8.56 | 14.71 | 2.81 |
| hogbom | 0.50 | build err | 2.71 |
| hotspot3D | 16.21 | 18.11 | 18.00 |
| hwt1d | 3.13 | 3.27 | 3.23 |
| hybridsort | 15.27 | build err | 12.96 |
| hypterm | 5.86 | 13.14 | 13.12 |
| idivide | 8.31 | 8.02 | 9.39 |
| interleave | 4.13 | 4.07 | 3.44 |
| interval | 7.15 | 13.47 | 10.78 |
| inversek2j | 1.05 | 225.58 | 111.49 |
| ising | 258.64 | 13.36 | 13.31 |
| iso2dfd | 17.67 | 13.97 | 14.20 |
| jacobi | 2.77 | 3.56 | 3.50 |
| jenkins-hash | 4.98 | 5.14 | 5.17 |
| kalman | over 600 | build err | exe err |
| keccaktreehash | 8.26 | 11.46 | 10.08 |
| keogh | 7.03 | 5.80 | 5.89 |
| kernelLaunch | 14.57 | build err | 73.69 |
| kmeans | 28.93 | 27.05 | 22.68 |
| knn | 4.48 | 4.28 | 4.53 |
| lanczos | 2.61 | build err | 2.81 |
| langevin | 17.73 | 6.26 | 31.31 |
| langford | 2.54 | build err | build err |
| laplace3d | 64.92 | 14.60 | 14.64 |
| laplace | 333.46 | 11.93 | 11.34 |
| lavaMD | 102.55 | build err | 116.55 |
| layout | 3.27 | 2.66 | 2.93 |
| lci | 10.00 | 4.36 | 4.36 |
| lda | 3.10 | build err | build err |
| ldpc | 4.91 | build err | 9.60 |
| lebesgue | 41.65 | 39.43 | 39.11 |
| leukocyte | build err | build err | 0.92 |
| libor | 2.87 | 13.25 | 2.84 |
| lid-driven-cavity | 10.91 | build err | 22.40 |
| lif | 116.49 | 125.87 | 124.42 |
| linearprobing | 102.65 | build err | build err |
| log2 | 0.74 | 2.77 | 2.81 |
| lombscargle | 2.99 | 3.06 | 3.05 |
| loopback | 5.38 | build err | 8.20 |
| lrn | 105.73 | 111.48 | 78.63 |
| lr | 6.36 | build err | build err |
| lsqt | 19.99 | 32.63 | 32.86 |
| lud | 6.29 | build err | 46.11 |
| lulesh | 11.28 | 11.70 | 8.41 |
| mallocFree | 3.69 | 2.68 | 2.67 |
| mandelbrot | 5.29 | 3.75 | 3.82 |
| mask | 55.31 | 21.60 | 29.68 |
| match | 42.51 | build err | 41.46 |
| matern | 6.24 | 3.85 | 31.37 |
| matrix-rotate | 35.98 | 35.18 | 35.00 |
| maxFlops | 25.56 | 26.46 | 25.46 |
| maxpool3d | 33.75 | 32.80 | 32.24 |
| mcmd | 9.03 | 10.79 | 11.06 |
| mcpr | 13.97 | build err | 15.03 |
| md5hash | 15.14 | 15.35 | 15.50 |
| mdh | 48.13 | 35.15 | 2.67 |
| md | 14.68 | 14.24 | 14.22 |
| meanshift | 4.43 | build err | 4.21 |
| medianfilter | 3.75 | build err | 4.68 |
| memcpy | 3.54 | 6.52 | 6.54 |
| memtest | 18.85 | 19.78 | 19.45 |
| merge | over 600 | build err | over 600 |
| metropolis | 14.03 | build err | 65.32 |
| michalewicz | 150.55 | over 600 | 63.30 |
| minibude | 3.31 | build err | 3.54 |
| miniFE | | | -- |
| minimap2 | 2.32 | | 2.19 |
| minisweep | 56.83 | 35.43 | 35.90 |
| miniWeather | 8.19 | 33.30 | 33.14 |
| minkowski | 24.25 | 22.27 | 22.36 |
| mis | 13.22 | build err | 2.60 |
| mixbench | 3.62 | build err | 3.80 |
| morphology | 4.81 | build err | over 600 |
| mrc | 83.40 | 17.88 | 17.83 |
| mriQ | 5.31 | 5.04 | 5.21 |
| mr | 3.33 | build err | 3.31 |
| mt | 4.65 | 4.29 | 4.08 |
| multimaterial | 14.67 | 12.11 | 21.91 |
| murmurhash3 | 2.80 | 2.67 | 2.69 |
| myocyte | 3.03 | build err | exe err |
| nbody | 25.24 | 19.63 | 19.61 |
| ne | 5.37 | 4.90 | 4.94 |
| nlll | 3.08 | build err | 3.06 |
| nms | 2.59 | build err | 2.73 |
| nn | 2.47 | 2.56 | 2.52 |
| norm2 | build err | 3.71 | 4.97 |
| nqueen | 6.94 | 23.85 | 6.87 |
| ntt | 6.63 | build err | 6.65 |
| nw | 26.82 | build err | 33.48 |
| openmp | 81.27 | build err | 13.65 |
| overlay | 5.33 | 9.20 | 10.50 |
| p4 | 3.77 | build err | 16.94 |
| page-rank | 375.35 | 46.20 | 46.72 |
| particle-diffusion | 11.94 | 13.04 | 12.23 |
| particlefilter | 198.39 | build err | 219.65 |
| particles | 3.46 | build err | over 600 |
| pathfinder | 64.27 | build err | exe err |
| permutate | 24.87 | build err | 25.87 |
| permute | 61.55 | 85.77 | 85.40 |
| perplexity | 192.24 | 97.16 | 96.53 |
| phmm | 3.53 | build err | 7.57 |
| pnpoly | 21.12 | build err | 74.17 |
| pns | 5.44 | build err | exe err |
| pointwise | 21.28 | 53.08 | 75.86 |
| pool | 51.58 | 41.75 | 41.00 |
| popcount | 178.30 | 279.36 | 134.99 |
| present | 2.67 | 3.04 | 3.05 |
| prna | build err | build err | exe err |
| projectile | 0.91 | 2.66 | 2.55 |
| pso | 2.83 | build err | 2.69 |
| qrg | 6.30 | 8.57 | 10.24 |
| qtclustering | 34.48 | build err | exe err |
| quantBnB | 121.16 | 125.38 | 125.68 |
| quicksort | 36.99 | build err | 33.98 |
| radixsort | 2.90 | build err | 8.11 |
| rainflow | 29.99 | 29.64 | 29.52 |
| randomAccess | 6.90 | build err | 24.54 |
| reaction | 4.08 | build err | 9.17 |
| recursiveGaussian | 4.04 | build err | 4.27 |
| resize | 5.98 | 6.03 | 6.01 |
| reverse | 3.67 | build err | 8.08 |
| rfs | 46.70 | 48.57 | 48.26 |
| rng-wallace | 2.56 | build err | 3.55 |
| rodrigues | 101.46 | 114.74 | 101.38 |
| romberg | 2.72 | build err | 2.81 |
| rsbench | 3.39 | build err | build err |
| rsc | 0.50 | build err | 2.73 |
| rtm8 | 17.67 | 15.76 | 4.22 |
| rushlarsen | over 600 | over 600 | over 600 |
| s3d | 2.94 | build err | 2.90 |
| s8n | 2.49 | 2.47 | 2.43 |
| sad | 2.53 | 2.16 | 8.52 |
| sampling | 5.77 | build err | 5.65 |
| scan2 | 2.91 | build err | 6.79 |
| scan | over 600 | build err | over 600 |
| scatterAdd | 8.10 | 8.05 | 7.79 |
| scel | 322.81 | 11.91 | 11.29 |
| secp256k1 | 2.68 | 2.79 | 2.69 |
| sheath | 5.85 | 5.81 | 5.82 |
| shmembench | 9.16 | build err | exe err |
| simplemoc | 400.17 | 293.35 | 294.34 |
| simpleSpmv | 319.81 | 320.00 | 317.47 |
| slu | build err | build err | build err |
| snake | 8.43 | 10.57 | 10.42 |
| sobel | 2.89 | 3.18 | 3.32 |
| sobol | 3.89 | build err | 4.51 |
| softmax-online | 13.69 | build err | 15.10 |
| softmax | 39.36 | 40.91 | 36.96 |
| sort | build err | build err | 36.92 |
| sosfil | 4.75 | build err | 13.32 |
| sph | 5.41 | 22.53 | 5.40 |
| split | 35.78 | build err | 37.73 |
| spm | 104.39 | 98.00 | 98.41 |
| sptrsv | 2.70 | build err | 2.81 |
| srad | 37.49 | build err | exe err |
| ss | 7.23 | 9.95 | 18.10 |
| stddev | 46.55 | build err | 51.00 |
| stencil1d | 4.68 | 30.81 | 35.77 |
| stencil3d | 4.35 | build err | 4.24 |
| streamcluster | 11.84 | build err | 26.24 |
| su3 | 5.17 | 7.47 | 7.48 |
| surfel | 510.01 | 12.02 | 11.97 |
| svd3x3 | 2.00 | 2.86 | 2.89 |
| sw4ck | 6.40 | 29.28 | 28.11 |
| swish | 24.54 | 5.48 | 5.39 |
| tensorT | 2.56 | build err | 2.87 |
| testSNAP | 3.73 | 66.16 | 4.19 |
| thomas | 131.17 | 59.68 | 49.73 |
| threadfence | 26.09 | build err | 35.11 |
| tissue | 12.06 | 11.74 | 18.48 |
| tonemapping | 3.15 | 3.02 | 2.95 |
| tqs | 2.57 | build err | 2.59 |
| triad | 2.62 | 2.82 | 2.75 |
| tridiagonal | 8.80 | build err | 24.55 |
| tsa | 36.86 | build err | 29.09 |
| tsp | 7.82 | build err | build err |
| urng | 2.43 | 2.21 | 2.20 |
| vanGenuchten | 21.48 | 30.99 | 30.92 |
| vmc | 4.38 | build err | build err |
| vol2col | 5.26 | 7.87 | 8.13 |
| wedford | build err | build err | exe err |
| winograd | 0.65 | 3.25 | 3.20 |
| wlcpow | 4.50 | build err | 12.37 |
| wordcount | 7.98 | 8.18 | 7.93 |
| wsm5 | 4.78 | 5.59 | 5.73 |
| wyllie | 112.64 | 108.06 | over 600 |
| xlqc | build err | build err | build err |
| xsbench | 37.40 | 37.18 | 3.89 |
| zeropoint | 55.54 | 21.82 | 24.23 |
| zmddft | 2.63 | build err | 14.28 |
| | | | |
| completed | 303 | 188 | 282 |

sycl と acc がともに完了した件数 185
