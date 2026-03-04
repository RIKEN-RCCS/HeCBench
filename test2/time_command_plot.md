# time コマンドにより出力される時間 (単位: 秒)

| 名称 | cuda | sycl | acc | omp_nvc | 分類 | メモリ(GB) | plot |
| -- | -- | -- | -- | -- | -- | -- | -- |
| accuracy | 2.54 | 2.49 | 2.48 | 2.53 | A | 0.9 |![accuracy](SVGs_command/accuracy.svg) |
| ace | 2.90 | 3.13 | 2.84 | 2.79 | A | 4.0 |![ace](SVGs_command/ace.svg) |
| adam | 5.20 | 5.24 | 6.72 | 6.70 | A | 0.6 |![adam](SVGs_command/adam.svg) |
| adamw | 13.20 | 12.32 | 20.30 | 20.34 | A | 0.6 |![adamw](SVGs_command/adamw.svg) |
| adjacent | 7.66 | 9.13 | 7.03 | 15.22 | A | 4.5 |![adjacent](SVGs_command/adjacent.svg) |
| adv | 3.05 | 3.35 | build err | 3.49 | B | 4.9 |![adv](SVGs_command/adv.svg) |
| aes | 2.72 | 2.79 | build err | 2.70 | B | 0.6 |![aes](SVGs_command/aes.svg) |
| affine | 5.62 | 5.42 | 12.78 | 12.37 | A | 0.6 |![affine](SVGs_command/affine.svg) |
| aidw | 4.58 | 12.87 | 5.42 | 6.21 | B | 0.6 |![aidw](SVGs_command/aidw.svg) |
| aligned-types | 3.82 | 3.68 | 4.40 | 4.71 | A | 0.7 |![aligned-types](SVGs_command/aligned-types.svg) |
| all-pairs-distance | 4.96 | 5.23 | 4.81 | 21.34 | B | 0.6 |![all-pairs-distance](SVGs_command/all-pairs-distance.svg) |
| amgmk | 2.51 | 2.45 | 2.42 | 2.44 | B | 0.6 |![amgmk](SVGs_command/amgmk.svg) |
| ans | 5.19 | 5.42 | exe err | exe err | A | 0.6 |![ans](SVGs_command/ans.svg) |
| aobench | 2.46 | 0.36 | 2.42 | 2.47 | A | 0.6 |![aobench](SVGs_command/aobench.svg) |
| aop | 7.03 | 7.02 | build err | build err | B | 0.6 |![aop](SVGs_command/aop.svg) |
| asmooth | 12.47 | 5.70 | 7.38 | 7.33 | A | 1.6 |![asmooth](SVGs_command/asmooth.svg) |
| assert | 2.51 | exe err | build err | 2.58 | B | 0.6 |![assert](SVGs_command/assert.svg) |
| asta | 4.36 | build err | build err | 10.71 | B | 0.6 |![asta](SVGs_command/asta.svg) |
| atan2 | 2.68 | 3.24 | 2.52 | 2.54 | A | 0.7 |![atan2](SVGs_command/atan2.svg) |
| atomicCost | 16.01 | 18.89 | 18.84 | 5.36 | A | 14.6 |![atomicCost](SVGs_command/atomicCost.svg) |
| atomicIntrinsics | 3.90 | over 600 | 3.18 | 3.21 | A | 0.6 |![atomicIntrinsics](SVGs_command/atomicIntrinsics.svg) |
| atomicPerf | 14.33 | 14.66 | 13.47 | 13.88 | B | 0.6 |![atomicPerf](SVGs_command/atomicPerf.svg) |
| atomicReduction | 2.70 | 3.44 | 2.65 | 2.63 | A | 0.8 |![atomicReduction](SVGs_command/atomicReduction.svg) |
| attention | 2.48 | 2.47 | 83.27 | 83.85 | A | 1.1 |![attention](SVGs_command/attention.svg) |
| axhelm | 2.46 | 2.41 | build err | 2.47 | B | 0.7 |![axhelm](SVGs_command/axhelm.svg) |
| babelstream | 2.74 | 0.56 | 2.72 | 2.85 | A | 1.3 |![babelstream](SVGs_command/babelstream.svg) |
| background-subtract | 6.52 | 8.07 | 8.66 | 8.33 | A | 0.6 |![background-subtract](SVGs_command/background-subtract.svg) |
| backprop | 2.53 | 2.45 | 2.44 | 2.47 | B | 0.6 |![backprop](SVGs_command/backprop.svg) |
| bezier-surface | 12.43 | 12.93 | 14.25 | 10.71 | A | 1.3 |![bezier-surface](SVGs_command/bezier-surface.svg) |
| bfs | 0.98 | 1.03 | 2.48 | 2.46 | A | 0.6 |![bfs](SVGs_command/bfs.svg) |
| bilateral | 7.64 | 15.09 | 205.59 | 7.85 | A | 0.6 |![bilateral](SVGs_command/bilateral.svg) |
| binomial | 3.24 | 3.10 | | -- | B | 0.6 |![binomial](SVGs_command/binomial.svg) |
| bitonic-sort | 13.52 | 10.81 | 7.80 | 8.49 | A | 0.7 |![bitonic-sort](SVGs_command/bitonic-sort.svg) |
| black-scholes | 4.80 | 5.00 | 39.24 | 4.88 | A | 2.5 |![black-scholes](SVGs_command/black-scholes.svg) |
| blas-gemm | 5.70 | build err | | -- | A | 1.1 |![blas-gemm](SVGs_command/blas-gemm.svg) |
| bn | 4.47 | 5.00 | build err | exe err | B | 0.6 |![bn](SVGs_command/bn.svg) |
| bonds | 11.93 | 7.66 | 11.50 | 11.19 | A | 0.8 |![bonds](SVGs_command/bonds.svg) |
| boxfilter | 9.06 | 8.46 | 6.47 | 9.55 | B | 0.6 |![boxfilter](SVGs_command/boxfilter.svg) |
| bsearch | 33.96 | 34.85 | 32.68 | 31.84 | B | 27.3 |![bsearch](SVGs_command/bsearch.svg) |
| bspline-vgh | 9.73 | 7.74 | 4.73 | 4.81 | A | 2.6 |![bspline-vgh](SVGs_command/bspline-vgh.svg) |
| b+tree | 2.87 | 2.46 | 2.46 | 2.46 | A | 0.6 |![b+tree](SVGs_command/b+tree.svg) |
| burger | 84.00 | 72.44 | 68.61 | 112.53 | A | 8.4 |![burger](SVGs_command/burger.svg) |
| bwt | 204.75 | 199.29 | 188.93 | exe err | A | 1.3 |![bwt](SVGs_command/bwt.svg) |
| car | 7.41 | 7.18 | 6.64 | 6.65 | A | 6.6 |![car](SVGs_command/car.svg) |
| cbsfil | 32.08 | 31.48 | 35.01 | 33.19 | A | 2.4 |![cbsfil](SVGs_command/cbsfil.svg) |
| ccsd-trpdrv | 2.61 | 2.67 | 2.55 | 2.63 | A | 0.6 |![ccsd-trpdrv](SVGs_command/ccsd-trpdrv.svg) |
| ccs | 3.36 | 3.57 | 2.97 | 3.65 | B | 0.6 |![ccs](SVGs_command/ccs.svg) |
| cfd | 0.82 | 0.58 | 3.02 | 3.08 | A | 0.6 |![cfd](SVGs_command/cfd.svg) |
| chacha20 | 4.74 | 4.68 | 9.24 | 6.66 | B | 0.6 |![chacha20](SVGs_command/chacha20.svg) |
| channelShuffle | 18.46 | 538.46 | 19.23 | 19.24 | A | 13.1 |![channelShuffle](SVGs_command/channelShuffle.svg) |
| channelSum | 65.58 | 64.97 | 65.55 | 68.17 | A | 6.8 |![channelSum](SVGs_command/channelSum.svg) |
| chemv | 3.21 | 3.20 | 2.85 | 3.21 | B | 0.6 |![chemv](SVGs_command/chemv.svg) |
| che | 3.62 | 3.05 | 3.27 | 3.38 | A | 1.1 |![che](SVGs_command/che.svg) |
| chi2 | 197.09 | 217.04 | 220.78 | 251.54 | A | 51.0 |![chi2](SVGs_command/chi2.svg) |
| clenergy | 2.68 | 2.66 | 3.13 | 3.16 | A | 0.6 |![clenergy](SVGs_command/clenergy.svg) |
| clink | 4.98 | 56.20 | 8.31 | 8.15 | A | 1.8 |![clink](SVGs_command/clink.svg) |
| cmp | exe err | exe err | build err | exe err | A | |![cmp](SVGs_command/cmp.svg) |
| cm | exe err | exe err | build err | exe err | B | |![cm](SVGs_command/cm.svg) |
| cobahh | 487.96 | 406.65 | 383.64 | 388.03 | A | 55.9 |![cobahh](SVGs_command/cobahh.svg) |
| colorwheel | 39.48 | 37.38 | 27.55 | 27.96 | A | 2.3 |![colorwheel](SVGs_command/colorwheel.svg) |
| columnarSolver | 9.83 | 10.16 | 16.87 | 10.22 | B | 1.6 |![columnarSolver](SVGs_command/columnarSolver.svg) |
| complex | 23.04 | 22.28 | 12.59 | 12.96 | A | 0.7 |![complex](SVGs_command/complex.svg) |
| compute-score | 3.28 | 3.24 | 3.27 | 4.34 | B | 1.2 |![compute-score](SVGs_command/compute-score.svg) |
| concat | 17.45 | 15.08 | 16.24 | 15.73 | A | 4.7 |![concat](SVGs_command/concat.svg) |
| contract | 9.28 | 11.40 | 28.95 | 10.82 | A | 0.9 |![contract](SVGs_command/contract.svg) |
| conversion | 5.09 | 4.98 | 3.56 | 3.44 | A | 16.6 |![conversion](SVGs_command/conversion.svg) |
| convolution1D | 382.64 | 380.96 | 333.42 | 383.27 | B | 16.7 |![convolution1D](SVGs_command/convolution1D.svg) |
| convolution3D | 2.55 | 2.48 | 3.17 | 2.51 | A | 0.6 |![convolution3D](SVGs_command/convolution3D.svg) |
| convolutionSeparable | 13.37 | 13.58 | build err | 22.05 | B | 3.6 |![convolutionSeparable](SVGs_command/convolutionSeparable.svg) |
| cooling | 381.03 | 154.60 | 452.57 | 454.09 | A | 14.3 |![cooling](SVGs_command/cooling.svg) |
| crc64 | 56.48 | 58.47 | 57.16 | 57.25 | A | 1.9 |![crc64](SVGs_command/crc64.svg) |
| cross | 12.82 | 14.58 | 7.67 | 8.04 | A | 6.1 |![cross](SVGs_command/cross.svg) |
| crs | 25.43 | 24.78 | build err | over 600 | B | 0.6 |![crs](SVGs_command/crs.svg) |
| d2q9-bgk | 2.59 | 2.85 | build err | 4.37 | B | 0.9 |![d2q9-bgk](SVGs_command/d2q9-bgk.svg) |
| damage | 29.66 | 24.45 | 36.68 | 243.28 | B | 8.7 |![damage](SVGs_command/damage.svg) |
| dct8x8 | 85.64 | 130.74 | build err | build err | B | 32.8 |![dct8x8](SVGs_command/dct8x8.svg) |
| ddbp | 16.32 | 17.41 | build err | 17.27 | B | 3.9 |![ddbp](SVGs_command/ddbp.svg) |
| debayer | 6.75 | 6.37 | 6.68 | 6.79 | B | 2.7 |![debayer](SVGs_command/debayer.svg) |
| degrid | 14.49 | 14.97 | 23.24 | 17.04 | A | 1.7 |![degrid](SVGs_command/degrid.svg) |
| dense-embedding | 112.61 | 142.22 | 99.47 | 184.08 | B | 14.6 |![dense-embedding](SVGs_command/dense-embedding.svg) |
| depixel | 105.99 | 157.40 | 208.93 | 168.04 | A | 8.2 |![depixel](SVGs_command/depixel.svg) |
| deredundancy | 16.29 | 17.89 | 15.82 | 15.76 | B | 3.5 |![deredundancy](SVGs_command/deredundancy.svg) |
| diamond | 36.76 | build err | build err | 37.30 | A | 0.7 |![diamond](SVGs_command/diamond.svg) |
| distort | 2.52 | 2.45 | 2.45 | 2.50 | A | 0.6 |![distort](SVGs_command/distort.svg) |
| divergence | 4.74 | 5.32 | 3.15 | 3.15 | A | 0.6 |![divergence](SVGs_command/divergence.svg) |
| doh | 14.61 | 14.97 | 2.39 | 2.41 | A | 0.6 |![doh](SVGs_command/doh.svg) |
| dp | 53.24 | build err | 25.04 | 25.25 | A | 49.5 |![dp](SVGs_command/dp.svg) |
| dslash | 2.68 | 2.70 | 2.68 | 2.66 | A | 1.9 |![dslash](SVGs_command/dslash.svg) |
| dxtc2 | 2.49 | 2.49 | build err | over 600 | B | 0.6 |![dxtc2](SVGs_command/dxtc2.svg) |
| easyWave | 3.09 | 2.97 | 2.79 | 3.08 | A | 0.8 |![easyWave](SVGs_command/easyWave.svg) |
| ecdh | 14.96 | 11.26 | 18.09 | 19.13 | A | 4.4 |![ecdh](SVGs_command/ecdh.svg) |
| eigenvalue | 21.76 | 19.92 | over 600 | 22.02 | A | 0.6 |![eigenvalue](SVGs_command/eigenvalue.svg) |
| entropy | 48.31 | 47.60 | 94.05 | 49.43 | B | 2.5 |![entropy](SVGs_command/entropy.svg) |
| epistasis | 54.59 | 53.82 | 259.15 | 110.41 | A | 4.3 |![epistasis](SVGs_command/epistasis.svg) |
| expdist | 7.57 | 7.45 | build err | 12.43 | B | 0.6 |![expdist](SVGs_command/expdist.svg) |
| extend2 | 5.80 | 5.17 | 4.68 | 4.94 | A | 0.6 |![extend2](SVGs_command/extend2.svg) |
| extrema | 7.05 | 6.75 | 6.99 | 7.10 | A | 0.6 |![extrema](SVGs_command/extrema.svg) |
| face | 2.72 | 2.64 | 2.45 | exe err | A | 0.6 |![face](SVGs_command/face.svg) |
| fdtd3d | 30.14 | 31.50 | build err | 43.29 | B | 1.0 |![fdtd3d](SVGs_command/fdtd3d.svg) |
| feynman-kac | 81.22 | 41.84 | 48.24 | 22.16 | A | 0.6 |![feynman-kac](SVGs_command/feynman-kac.svg) |
| fft | 3.17 | 3.19 | build err | 3.19 | B | 0.8 |![fft](SVGs_command/fft.svg) |
| fhd | over 600 | 183.93 | over 600 | exe err | A | 2.5 |![fhd](SVGs_command/fhd.svg) |
| filter | 168.70 | build err | build err | 154.77 | B | 11.3 |![filter](SVGs_command/filter.svg) |
| flip | 10.43 | 37.32 | 14.45 | 15.50 | A | 26.9 |![flip](SVGs_command/flip.svg) |
| floydwarshall | 74.91 | 73.45 | 72.59 | 83.41 | A | 0.7 |![floydwarshall](SVGs_command/floydwarshall.svg) |
| fluidSim | 19.45 | 17.67 | 17.30 | 17.36 | A | 0.6 |![fluidSim](SVGs_command/fluidSim.svg) |
| fpc | 67.41 | 66.38 | build err | exe err | B | 2.6 |![fpc](SVGs_command/fpc.svg) |
| fpdc | 4.82 | 5.37 | build err | over 600 | B | 2.1 |![fpdc](SVGs_command/fpdc.svg) |
| fresnel | 6.64 | 6.00 | 6.61 | 6.61 | A | 1.8 |![fresnel](SVGs_command/fresnel.svg) |
| frna | 103.73 | 104.63 | build err | build err | B | 3.9 |![frna](SVGs_command/frna.svg) |
| fwt | 3.13 | 3.12 | 3.35 | 3.47 | B | 0.6 |![fwt](SVGs_command/fwt.svg) |
| gabor | 21.34 | 21.57 | 21.27 | 57.42 | A | 9.7 |![gabor](SVGs_command/gabor.svg) |
| gamma-correction | 68.53 | 63.44 | 105.30 | 104.46 | A | 1.1 |![gamma-correction](SVGs_command/gamma-correction.svg) |
| ga | 48.84 | 36.53 | 32.72 | 33.46 | A | 0.6 |![ga](SVGs_command/ga.svg) |
| gaussian | 15.65 | 4.12 | 3.96 | 4.07 | A | 0.7 |![gaussian](SVGs_command/gaussian.svg) |
| gc | 2.59 | 2.41 | build err | build err | A | 0.6 |![gc](SVGs_command/gc.svg) |
| gd | 12.38 | 13.02 | 12.66 | 12.83 | A | 0.8 |![gd](SVGs_command/gd.svg) |
| geglu | 27.67 | 94.40 | 22.13 | 22.07 | A | 8.2 |![geglu](SVGs_command/geglu.svg) |
| geodesic | 2.54 | 2.60 | 2.51 | 2.52 | A | 0.8 |![geodesic](SVGs_command/geodesic.svg) |
| glu | 59.42 | 121.58 | 124.43 | 120.93 | A | 15.8 |![glu](SVGs_command/glu.svg) |
| gmm | 35.58 | 35.97 | build err | 37.28 | B | 1.5 |![gmm](SVGs_command/gmm.svg) |
| goulash | 44.04 | 27.35 | 36.03 | 36.52 | A | 26.2 |![goulash](SVGs_command/goulash.svg) |
| gpp | 5.15 | 4.29 | 4.10 | 4.15 | A | 2.7 |![gpp](SVGs_command/gpp.svg) |
| grep | 55.53 | 57.42 | build err | 30.15 | B | 4.9 |![grep](SVGs_command/grep.svg) |
| grrt | 14.22 | 14.54 | 13.31 | 12.82 | A | 0.6 |![grrt](SVGs_command/grrt.svg) |
| haccmk | 4.70 | 4.33 | 2.57 | 4.42 | A | 0.6 |![haccmk](SVGs_command/haccmk.svg) |
| hausdorff | 30.92 | 30.37 | 32.21 | 31.92 | A | 0.6 |![hausdorff](SVGs_command/hausdorff.svg) |
| haversine | 2.22 | 2.18 | 2.49 | 2.43 | A | 1.0 |![haversine](SVGs_command/haversine.svg) |
| heartwall | 7.48 | exe err | build err | build err | A | 2.7 |![heartwall](SVGs_command/heartwall.svg) |
| heat2d | 6.14 | 61.50 | 25.22 | 5.92 | A | 0.8 |![heat2d](SVGs_command/heat2d.svg) |
| heat | 22.95 | 22.04 | 15.34 | 16.22 | A | 16.9 |![heat](SVGs_command/heat.svg) |
| hellinger | 4.94 | 4.80 | 6.88 | 4.52 | A | 0.6 |![hellinger](SVGs_command/hellinger.svg) |
| henry | 3.91 | 3.77 | 2.86 | 2.74 | A | 0.6 |![henry](SVGs_command/henry.svg) |
| hexciton | 4.19 | 6.03 | build err | 4.96 | B | 0.6 |![hexciton](SVGs_command/hexciton.svg) |
| histogram | 3.06 | 2.98 | build err | 3.30 | B | 0.6 |![histogram](SVGs_command/histogram.svg) |
| hmm | 4.22 | 8.56 | 14.71 | 2.81 | A | 0.7 |![hmm](SVGs_command/hmm.svg) |
| hogbom | 0.53 | 2.67 | 2.65 | 2.76 | B | 0.7 |![hogbom](SVGs_command/hogbom.svg) |
| hotspot3D | 18.45 | 16.21 | 18.11 | 18.00 | A | 0.6 |![hotspot3D](SVGs_command/hotspot3D.svg) |
| hwt1d | 3.15 | 3.13 | 3.27 | 3.23 | B | 0.6 |![hwt1d](SVGs_command/hwt1d.svg) |
| hybridsort | 12.83 | 15.27 | build err | 12.96 | B | 1.2 |![hybridsort](SVGs_command/hybridsort.svg) |
| hypterm | 5.75 | 5.86 | 13.14 | 13.12 | A | 3.5 |![hypterm](SVGs_command/hypterm.svg) |
| idivide | 6.12 | 8.31 | 8.02 | 9.39 | A | 0.6 |![idivide](SVGs_command/idivide.svg) |
| interleave | 4.15 | 4.13 | 4.07 | 3.44 | A | 0.6 |![interleave](SVGs_command/interleave.svg) |
| interval | 7.59 | 7.15 | 13.47 | 10.78 | A | 1.0 |![interval](SVGs_command/interval.svg) |
| inversek2j | 2.84 | 1.05 | 225.58 | 111.49 | A | 0.6 |![inversek2j](SVGs_command/inversek2j.svg) |
| ising | 10.77 | 258.64 | 13.36 | 13.31 | A | 2.4 |![ising](SVGs_command/ising.svg) |
| iso2dfd | 17.76 | 17.67 | 13.97 | 14.20 | A | 49.7 |![iso2dfd](SVGs_command/iso2dfd.svg) |
| jacobi | 2.80 | 2.77 | 3.56 | 3.50 | A | 0.6 |![jacobi](SVGs_command/jacobi.svg) |
| jenkins-hash | 5.05 | 4.98 | 5.14 | 5.17 | A | 10.0 |![jenkins-hash](SVGs_command/jenkins-hash.svg) |
| kalman | 335.35 | over 600 | build err | exe err | A | 10.8 |![kalman](SVGs_command/kalman.svg) |
| keccaktreehash | 7.76 | 8.26 | 11.46 | 10.08 | A | 0.7 |![keccaktreehash](SVGs_command/keccaktreehash.svg) |
| keogh | 22.46 | 7.03 | 5.80 | 5.89 | A | 0.9 |![keogh](SVGs_command/keogh.svg) |
| kernelLaunch | 13.80 | 14.84 | 281.99 | 73.99 | B | 0.6 |![kernelLaunch](SVGs_command/kernelLaunch.svg) |
| kmeans | 28.57 | 28.93 | 27.05 | 22.68 | A | 0.7 |![kmeans](SVGs_command/kmeans.svg) |
| knn | 4.43 | 4.48 | 4.28 | 4.53 | B | 0.6 |![knn](SVGs_command/knn.svg) |
| lanczos | 2.61 | 2.61 | build err | 2.81 | B | 0.6 |![lanczos](SVGs_command/lanczos.svg) |
| langevin | 21.81 | 17.73 | 6.26 | 31.31 | A | 31.1 |![langevin](SVGs_command/langevin.svg) |
| langford | 2.76 | 2.54 | build err | build err | B | 2.1 |![langford](SVGs_command/langford.svg) |
| laplace3d | 65.95 | 64.92 | 14.60 | 14.64 | A | 8.8 |![laplace3d](SVGs_command/laplace3d.svg) |
| laplace | 10.89 | 333.46 | 11.93 | 11.34 | A | 0.6 |![laplace](SVGs_command/laplace.svg) |
| lavaMD | 104.27 | 102.55 | build err | 116.55 | B | 49.9 |![lavaMD](SVGs_command/lavaMD.svg) |
| layout | 3.28 | 3.27 | 2.66 | 2.93 | A | 0.6 |![layout](SVGs_command/layout.svg) |
| lci | 17.78 | 10.00 | 4.36 | 4.36 | A | 0.6 |![lci](SVGs_command/lci.svg) |
| lda | 3.10 | 3.10 | build err | build err | B | 0.6 |![lda](SVGs_command/lda.svg) |
| ldpc | 4.83 | 4.91 | build err | 9.60 | B | 0.6 |![ldpc](SVGs_command/ldpc.svg) |
| lebesgue | 39.25 | 41.65 | 39.43 | 39.11 | A | 1.0 |![lebesgue](SVGs_command/lebesgue.svg) |
| leukocyte | 0.64 | build err | build err | 0.92 | A | 0.6 |![leukocyte](SVGs_command/leukocyte.svg) |
| libor | 2.87 | 2.87 | 13.25 | 2.84 | A | 4.0 |![libor](SVGs_command/libor.svg) |
| lid-driven-cavity | 10.21 | 10.91 | build err | 22.40 | B | 0.6 |![lid-driven-cavity](SVGs_command/lid-driven-cavity.svg) |
| lif | 100.17 | 116.49 | 125.87 | 124.42 | A | 31.9 |![lif](SVGs_command/lif.svg) |
| linearprobing | 105.14 | 102.65 | build err | build err | A | 1.4 |![linearprobing](SVGs_command/linearprobing.svg) |
| log2 | 2.80 | 0.74 | 2.77 | 2.81 | A | 1.3 |![log2](SVGs_command/log2.svg) |
| lombscargle | 3.14 | 2.99 | 3.06 | 3.05 | A | 0.6 |![lombscargle](SVGs_command/lombscargle.svg) |
| loopback | 5.73 | 5.33 | 5.38 | 8.16 | B | 0.6 |![loopback](SVGs_command/loopback.svg) |
| lrn | 104.40 | 105.73 | 111.48 | 78.63 | A | 22.5 |![lrn](SVGs_command/lrn.svg) |
| lr | 6.18 | 4.12 | 16.99 | 31.82 | B | 0.6 |![lr](SVGs_command/lr.svg) |
| lsqt | 19.97 | 19.99 | 32.63 | 32.86 | A | 1.7 |![lsqt](SVGs_command/lsqt.svg) |
| lud | 5.54 | 8.19 | 9.54 | 46.07 | B | 4.7 |![lud](SVGs_command/lud.svg) |
| lulesh | 7.81 | 11.28 | 11.70 | 8.41 | A | 54.2 |![lulesh](SVGs_command/lulesh.svg) |
| mallocFree | 3.72 | 3.69 | 2.68 | 2.67 | A | 8.4 |![mallocFree](SVGs_command/mallocFree.svg) |
| mandelbrot | 5.75 | 5.29 | 3.75 | 3.82 | A | 0.6 |![mandelbrot](SVGs_command/mandelbrot.svg) |
| mask | 22.03 | 55.31 | 21.60 | 29.68 | A | 8.8 |![mask](SVGs_command/mask.svg) |
| match | 40.66 | 42.51 | build err | 41.46 | B | 0.6 |![match](SVGs_command/match.svg) |
| matern | 6.11 | 6.24 | 3.85 | 31.37 | B | 0.7 |![matern](SVGs_command/matern.svg) |
| matrix-rotate | 32.32 | 35.98 | 35.18 | 35.00 | A | 8.6 |![matrix-rotate](SVGs_command/matrix-rotate.svg) |
| maxFlops | 25.53 | 25.56 | 26.46 | 25.46 | A | 0.6 |![maxFlops](SVGs_command/maxFlops.svg) |
| maxpool3d | 32.54 | 33.75 | 32.80 | 32.24 | A | 10.2 |![maxpool3d](SVGs_command/maxpool3d.svg) |
| mcmd | 12.38 | 9.03 | 10.79 | 11.06 | A | 0.6 |![mcmd](SVGs_command/mcmd.svg) |
| mcpr | 25.42 | 14.07 | 21.86 | 15.00 | B | 0.6 |![mcpr](SVGs_command/mcpr.svg) |
| md5hash | 15.45 | 15.14 | 15.35 | 15.50 | A | 0.6 |![md5hash](SVGs_command/md5hash.svg) |
| mdh | 2.80 | 48.13 | 35.15 | 2.67 | A | 0.6 |![mdh](SVGs_command/mdh.svg) |
| md | 14.56 | 14.68 | 14.24 | 14.22 | A | 0.6 |![md](SVGs_command/md.svg) |
| meanshift | 3.76 | 4.42 | 4.71 | 4.40 | B | 0.6 |![meanshift](SVGs_command/meanshift.svg) |
| medianfilter | 3.54 | exe err | 1.36 | 2.84 | B | 0.6 |![medianfilter](SVGs_command/medianfilter.svg) |
| memcpy | 5.61 | 3.54 | 6.52 | 6.54 | A | 0.6 |![memcpy](SVGs_command/memcpy.svg) |
| memtest | 18.43 | 18.85 | 19.78 | 19.45 | A | 2.6 |![memtest](SVGs_command/memtest.svg) |
| merge | 432.85 | 442.02 | build err | 425.74 | B | 9.7 |![merge](SVGs_command/merge.svg) |
| metropolis | 12.73 | 14.03 | build err | 65.32 | B | 17.2 |![metropolis](SVGs_command/metropolis.svg) |
| michalewicz | 43.63 | 150.37 | 59.07 | 63.13 | A | 50.2 |![michalewicz](SVGs_command/michalewicz.svg) |
| minibude | 2.89 | 1.24 | 1.07 | 1.49 | B | 0.6 |![minibude](SVGs_command/minibude.svg) |
| miniFE | | | | -- | A | |![miniFE](SVGs_command/miniFE.svg) |
| minimap2 | 1.51 | 2.37 | 2.18 | 3.68 | A | 1.0 |![minimap2](SVGs_command/minimap2.svg) |
| minisweep | 50.83 | 56.83 | 35.43 | 35.90 | A | 7.7 |![minisweep](SVGs_command/minisweep.svg) |
| miniWeather | 8.49 | 8.19 | 33.30 | 33.14 | A | 0.6 |![miniWeather](SVGs_command/miniWeather.svg) |
| minkowski | 23.94 | 24.25 | 22.27 | 22.36 | A | 0.6 |![minkowski](SVGs_command/minkowski.svg) |
| mis | 24.24 | 10.99 | 0.24 | 0.23 | B | 0.6 |![mis](SVGs_command/mis.svg) |
| mixbench | 3.77 | 1.42 | 1.49 | 1.55 | B | 0.6 |![mixbench](SVGs_command/mixbench.svg) |
| morphology | 5.03 | 4.81 | build err | over 600 | B | 4.4 |![morphology](SVGs_command/morphology.svg) |
| mrc | 21.72 | 83.40 | 17.88 | 17.83 | A | 23.5 |![mrc](SVGs_command/mrc.svg) |
| mriQ | 4.76 | 5.31 | 5.04 | 5.21 | A | 0.6 |![mriQ](SVGs_command/mriQ.svg) |
| mr | 3.31 | 3.33 | build err | 3.31 | A | 0.6 |![mr](SVGs_command/mr.svg) |
| mt | 5.66 | 4.65 | 4.29 | 4.08 | A | 0.7 |![mt](SVGs_command/mt.svg) |
| multimaterial | 28.40 | 14.67 | 12.11 | 21.91 | A | 3.1 |![multimaterial](SVGs_command/multimaterial.svg) |
| murmurhash3 | 2.78 | 2.80 | 2.67 | 2.69 | A | 2.6 |![murmurhash3](SVGs_command/murmurhash3.svg) |
| myocyte | 33.92 | 46.17 | 13.21 | 12.83 | A | 0.6 |![myocyte](SVGs_command/myocyte.svg) |
| nbody | 23.95 | 25.24 | 19.63 | 19.61 | A | 0.6 |![nbody](SVGs_command/nbody.svg) |
| ne | 4.83 | 5.37 | 4.90 | 4.94 | A | 3.2 |![ne](SVGs_command/ne.svg) |
| nlll | 3.07 | 3.06 | 82.52 | 3.08 | B | 0.6 |![nlll](SVGs_command/nlll.svg) |
| nms | 2.60 | 2.59 | build err | 2.73 | B | 0.6 |![nms](SVGs_command/nms.svg) |
| nn | 2.55 | 2.47 | 2.56 | 2.52 | A | 0.6 |![nn](SVGs_command/nn.svg) |
| norm2 | 3.77 | build err | 3.71 | 4.97 | A | 2.7 |![norm2](SVGs_command/norm2.svg) |
| nqueen | 6.31 | 6.94 | 23.85 | 6.87 | A | 0.6 |![nqueen](SVGs_command/nqueen.svg) |
| ntt | 6.21 | 6.63 | build err | 6.65 | B | 0.6 |![ntt](SVGs_command/ntt.svg) |
| nw | 16.68 | 26.82 | build err | 33.48 | B | 15.3 |![nw](SVGs_command/nw.svg) |
| openmp | 13.88 | 80.16 | 11.49 | 11.52 | B | 0.7 |![openmp](SVGs_command/openmp.svg) |
| overlay | 5.87 | 5.33 | 9.20 | 10.50 | A | 0.6 |![overlay](SVGs_command/overlay.svg) |
| p4 | 3.78 | 3.77 | build err | 16.94 | B | 0.6 |![p4](SVGs_command/p4.svg) |
| page-rank | 50.87 | 375.35 | 46.20 | 46.72 | A | 12.8 |![page-rank](SVGs_command/page-rank.svg) |
| particle-diffusion | 11.92 | 11.94 | 13.04 | 12.23 | A | 3.3 |![particle-diffusion](SVGs_command/particle-diffusion.svg) |
| particlefilter | 222.50 | 198.39 | build err | 219.65 | B | 4.2 |![particlefilter](SVGs_command/particlefilter.svg) |
| particles | 3.67 | 3.46 | build err | over 600 | B | 0.6 |![particles](SVGs_command/particles.svg) |
| pathfinder | 63.24 | 63.28 | 69.10 | 824.34 | B | 6.7 |![pathfinder](SVGs_command/pathfinder.svg) |
| permutate | 26.61 | 24.87 | build err | 25.87 | B | 2.9 |![permutate](SVGs_command/permutate.svg) |
| permute | 61.55 | 61.55 | 85.77 | 85.40 | A | 9.8 |![permute](SVGs_command/permute.svg) |
| perplexity | 186.41 | 192.24 | 97.16 | 96.53 | A | 15.8 |![perplexity](SVGs_command/perplexity.svg) |
| phmm | 3.72 | 3.52 | 7.07 | 9.31 | B | 0.6 |![phmm](SVGs_command/phmm.svg) |
| pnpoly | 7.07 | 21.12 | build err | 74.17 | A | 0.9 |![pnpoly](SVGs_command/pnpoly.svg) |
| pns | 5.50 | 5.44 | build err | exe err | B | 1.3 |![pns](SVGs_command/pns.svg) |
| pointwise | 21.34 | 21.28 | 53.08 | 75.86 | A | 15.8 |![pointwise](SVGs_command/pointwise.svg) |
| pool | 27.23 | 51.58 | 41.75 | 41.00 | A | 8.2 |![pool](SVGs_command/pool.svg) |
| popcount | 241.58 | 178.30 | 279.36 | 134.99 | A | 23.4 |![popcount](SVGs_command/popcount.svg) |
| present | 2.87 | 2.67 | 3.04 | 3.05 | A | 0.6 |![present](SVGs_command/present.svg) |
| prna | 79.29 | build err | build err | exe err | B | 4.8 |![prna](SVGs_command/prna.svg) |
| projectile | 2.66 | 0.91 | 2.66 | 2.55 | A | 0.9 |![projectile](SVGs_command/projectile.svg) |
| pso | 2.93 | 2.89 | 2.81 | 4.37 | A | 0.6 |![pso](SVGs_command/pso.svg) |
| qrg | 7.23 | 6.30 | 8.57 | 10.24 | A | 0.6 |![qrg](SVGs_command/qrg.svg) |
| qtclustering | 35.55 | 34.48 | build err | exe err | B | 0.8 |![qtclustering](SVGs_command/qtclustering.svg) |
| quantBnB | 18.55 | 121.16 | 125.38 | 125.68 | A | 2.9 |![quantBnB](SVGs_command/quantBnB.svg) |
| quicksort | 36.33 | 36.99 | build err | 33.98 | B | 0.8 |![quicksort](SVGs_command/quicksort.svg) |
| radixsort | 2.94 | 2.90 | build err | 8.11 | B | 0.6 |![radixsort](SVGs_command/radixsort.svg) |
| rainflow | 29.26 | 29.99 | 29.64 | 29.52 | A | 49.0 |![rainflow](SVGs_command/rainflow.svg) |
| randomAccess | 6.66 | 6.90 | build err | 24.54 | A | 1.1 |![randomAccess](SVGs_command/randomAccess.svg) |
| reaction | 4.18 | 4.08 | build err | 9.17 | B | 0.6 |![reaction](SVGs_command/reaction.svg) |
| recursiveGaussian | 4.03 | 4.04 | build err | 4.27 | B | 0.6 |![recursiveGaussian](SVGs_command/recursiveGaussian.svg) |
| resize | 6.38 | 5.98 | 6.03 | 6.01 | A | 4.7 |![resize](SVGs_command/resize.svg) |
| reverse | 3.47 | 1.56 | 4.46 | 5.70 | B | 0.6 |![reverse](SVGs_command/reverse.svg) |
| rfs | 46.68 | 46.70 | 48.57 | 48.26 | A | 6.3 |![rfs](SVGs_command/rfs.svg) |
| rng-wallace | 2.62 | 2.56 | build err | 3.55 | B | 0.8 |![rng-wallace](SVGs_command/rng-wallace.svg) |
| rodrigues | 96.39 | 101.46 | 114.74 | 101.38 | A | 54.0 |![rodrigues](SVGs_command/rodrigues.svg) |
| romberg | 2.75 | 0.60 | 0.57 | 0.54 | B | 0.6 |![romberg](SVGs_command/romberg.svg) |
| rsbench | 3.01 | 1.29 | 9.73 | 2.98 | A | 0.6 |![rsbench](SVGs_command/rsbench.svg) |
| rsc | 2.75 | 0.50 | 2.68 | 2.80 | B | 0.6 |![rsc](SVGs_command/rsc.svg) |
| rtm8 | 4.35 | 17.67 | 15.76 | 4.22 | A | 1.5 |![rtm8](SVGs_command/rtm8.svg) |
| rushlarsen | 140.36 | 129.52 | 135.46 | 135.06 | A | 6.0 |![rushlarsen](SVGs_command/rushlarsen.svg) |
| s3d | 2.92 | 2.94 | build err | 2.90 | A | 0.9 |![s3d](SVGs_command/s3d.svg) |
| s8n | 2.58 | 2.49 | 2.47 | 2.43 | A | 0.6 |![s8n](SVGs_command/s8n.svg) |
| sad | 2.40 | 2.48 | 2.63 | 81.81 | A | 0.6 |![sad](SVGs_command/sad.svg) |
| sampling | 5.85 | 5.77 | 5.80 | 5.64 | B | 0.7 |![sampling](SVGs_command/sampling.svg) |
| scan2 | 2.93 | 2.91 | build err | 6.76 | B | 0.8 |![scan2](SVGs_command/scan2.svg) |
| scan | 33.88 | 31.87 | build err | 33.05 | B | 1.3 |![scan](SVGs_command/scan.svg) |
| scatterAdd | 7.95 | 8.10 | 8.05 | 7.79 | A | 8.0 |![scatterAdd](SVGs_command/scatterAdd.svg) |
| scel | 17.04 | 322.81 | 11.91 | 11.29 | A | 3.6 |![scel](SVGs_command/scel.svg) |
| secp256k1 | 2.77 | 2.68 | 2.79 | 2.69 | A | 0.6 |![secp256k1](SVGs_command/secp256k1.svg) |
| sheath | 5.92 | 5.85 | 5.81 | 5.82 | A | 0.6 |![sheath](SVGs_command/sheath.svg) |
| shmembench | 5.25 | 9.16 | build err | exe err | B | 0.6 |![shmembench](SVGs_command/shmembench.svg) |
| simplemoc | 233.38 | 400.17 | 293.35 | 294.34 | B | 15.4 |![simplemoc](SVGs_command/simplemoc.svg) |
| simpleSpmv | 317.74 | 319.81 | 320.00 | 317.47 | A | 7.0 |![simpleSpmv](SVGs_command/simpleSpmv.svg) |
| slu | build err | build err | build err | build err | B | |![slu](SVGs_command/slu.svg) |
| snake | 8.17 | 8.43 | 10.57 | 10.42 | A | 0.6 |![snake](SVGs_command/snake.svg) |
| sobel | 2.81 | 2.89 | 3.18 | 3.32 | A | 0.6 |![sobel](SVGs_command/sobel.svg) |
| sobol | 4.71 | 3.89 | build err | 4.51 | B | 4.4 |![sobol](SVGs_command/sobol.svg) |
| softmax-online | 11.10 | 13.69 | build err | 15.10 | B | 3.7 |![softmax-online](SVGs_command/softmax-online.svg) |
| softmax | 44.19 | 39.36 | 40.91 | 36.96 | A | 25.9 |![softmax](SVGs_command/softmax.svg) |
| sort | 4.69 | build err | build err | 36.92 | B | 0.8 |![sort](SVGs_command/sort.svg) |
| sosfil | 4.65 | 4.75 | 5.74 | 13.04 | B | 0.6 |![sosfil](SVGs_command/sosfil.svg) |
| sph | 5.45 | 5.41 | 22.53 | 5.40 | A | 0.6 |![sph](SVGs_command/sph.svg) |
| split | 35.89 | 35.78 | build err | 37.73 | B | 15.8 |![split](SVGs_command/split.svg) |
| spm | 100.86 | 104.39 | 98.00 | 98.41 | A | 10.6 |![spm](SVGs_command/spm.svg) |
| sptrsv | 2.44 | 2.70 | build err | 2.81 | B | 0.6 |![sptrsv](SVGs_command/sptrsv.svg) |
| srad | 36.80 | 37.49 | build err | exe err | B | 14.7 |![srad](SVGs_command/srad.svg) |
| ss | 7.45 | 7.23 | 9.95 | 18.10 | B | 0.6 |![ss](SVGs_command/ss.svg) |
| stddev | 48.33 | 46.55 | 45.23 | 50.49 | B | 8.7 |![stddev](SVGs_command/stddev.svg) |
| stencil1d | 3.75 | 4.68 | 30.81 | 35.77 | A | 4.7 |![stencil1d](SVGs_command/stencil1d.svg) |
| stencil3d | 4.01 | 4.35 | 3.17 | 4.27 | B | 11.8 |![stencil3d](SVGs_command/stencil3d.svg) |
| streamcluster | 5.89 | 11.84 | 9.58 | 7.74 | B | 0.7 |![streamcluster](SVGs_command/streamcluster.svg) |
| su3 | 2.99 | 5.17 | 7.47 | 7.48 | A | 2.5 |![su3](SVGs_command/su3.svg) |
| surfel | 13.03 | 510.01 | 12.02 | 11.97 | A | 0.6 |![surfel](SVGs_command/surfel.svg) |
| svd3x3 | 1.91 | 2.00 | 2.86 | 2.89 | A | 0.7 |![svd3x3](SVGs_command/svd3x3.svg) |
| sw4ck | 3.58 | 6.40 | 29.28 | 28.11 | A | 1.9 |![sw4ck](SVGs_command/sw4ck.svg) |
| swish | 7.49 | 24.54 | 5.48 | 5.39 | A | 8.2 |![swish](SVGs_command/swish.svg) |
| tensorT | 2.73 | 2.56 | 2.67 | 2.95 | B | 3.6 |![tensorT](SVGs_command/tensorT.svg) |
| testSNAP | 3.69 | 3.73 | 66.16 | 4.19 | A | 3.2 |![testSNAP](SVGs_command/testSNAP.svg) |
| thomas | 125.54 | 131.17 | 59.68 | 49.73 | A | 51.8 |![thomas](SVGs_command/thomas.svg) |
| threadfence | 25.80 | 26.09 | 15.40 | 33.08 | B | 6.0 |![threadfence](SVGs_command/threadfence.svg) |
| tissue | 11.96 | 12.45 | 11.82 | 18.23 | A | 0.6 |![tissue](SVGs_command/tissue.svg) |
| tonemapping | 3.24 | 3.15 | 3.02 | 2.95 | A | 0.6 |![tonemapping](SVGs_command/tonemapping.svg) |
| tqs | 2.71 | 2.57 | build err | 2.59 | B | 0.6 |![tqs](SVGs_command/tqs.svg) |
| triad | 2.72 | 2.62 | 2.82 | 2.75 | A | 0.7 |![triad](SVGs_command/triad.svg) |
| tridiagonal | 8.42 | 8.80 | build err | 24.55 | B | 4.0 |![tridiagonal](SVGs_command/tridiagonal.svg) |
| tsa | 54.11 | 36.86 | build err | 29.09 | B | 20.6 |![tsa](SVGs_command/tsa.svg) |
| tsp | 5.10 | 7.82 | build err | build err | B | 0.6 |![tsp](SVGs_command/tsp.svg) |
| urng | 2.47 | 2.43 | 2.21 | 2.20 | B | 0.6 |![urng](SVGs_command/urng.svg) |
| vanGenuchten | 21.39 | 21.48 | 30.99 | 30.92 | A | 6.2 |![vanGenuchten](SVGs_command/vanGenuchten.svg) |
| vmc | 5.07 | 4.38 | build err | build err | B | 0.6 |![vmc](SVGs_command/vmc.svg) |
| vol2col | 8.27 | 5.26 | 7.87 | 8.13 | A | 3.5 |![vol2col](SVGs_command/vol2col.svg) |
| wedford | exe err | build err | build err | exe err | B | |![wedford](SVGs_command/wedford.svg) |
| winograd | 0.84 | 0.65 | 3.25 | 3.20 | A | 0.6 |![winograd](SVGs_command/winograd.svg) |
| wlcpow | 4.53 | 4.50 | build err | 12.37 | B | 0.7 |![wlcpow](SVGs_command/wlcpow.svg) |
| wordcount | 8.03 | 7.98 | 8.18 | 7.93 | A | 0.8 |![wordcount](SVGs_command/wordcount.svg) |
| wsm5 | 5.57 | 4.78 | 5.59 | 5.73 | B | 0.6 |![wsm5](SVGs_command/wsm5.svg) |
| wyllie | 107.36 | 112.64 | 108.06 | over 600 | B | 1.3 |![wyllie](SVGs_command/wyllie.svg) |
| xlqc | 2.30 | exe err | build err | 0.36 | B | 0.6 |![xlqc](SVGs_command/xlqc.svg) |
| xsbench | 46.19 | 37.40 | 37.18 | 3.89 | A | 6.3 |![xsbench](SVGs_command/xsbench.svg) |
| zeropoint | 18.63 | 55.54 | 21.82 | 24.23 | A | 15.8 |![zeropoint](SVGs_command/zeropoint.svg) |
| zmddft | 2.63 | 2.63 | build err | 14.28 | B | 1.6 |![zmddft](SVGs_command/zmddft.svg) |
| | | | | | | |
| 完了した件数 | 319 | 305 | 238 (A:186) | 290 | | ||

sycl と acc がともに完了した件数 234 (A:183)|

分類: OpenMP コードが (omp_get_wtime 以外の) omp_get_* を (A) 含まない (B) 含む
