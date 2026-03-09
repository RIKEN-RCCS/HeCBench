# time コマンドにより出力される時間 (単位: 秒)

| 名称 | cuda | sycl | acc | omp | 分類 | メモリ(GB) | グラフ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accuracy | 2.54 | 2.49 | 2.47 | 2.53 | A | 0.9 |![accuracy](Graph_command/accuracy.svg) |
| ace | 2.90 | 3.11 | 3.01 | 2.79 | A | 4.0 |![ace](Graph_command/ace.svg) |
| adam | 5.20 | 5.38 | 6.72 | 6.70 | A | 0.6 |![adam](Graph_command/adam.svg) |
| adamw | 13.20 | 12.34 | 20.47 | 20.34 | A | 0.6 |![adamw](Graph_command/adamw.svg) |
| adjacent | 7.66 | 9.23 | 15.92 | 15.22 | A | 4.5 |![adjacent](Graph_command/adjacent.svg) |
| adv | 3.05 | 3.39 | build err | 3.49 | B | 4.9 |![adv](Graph_command/adv.svg) |
| aes | 2.72 | 2.69 | build err | 2.70 | B | 0.6 |![aes](Graph_command/aes.svg) |
| affine | 5.62 | 5.46 | 12.76 | 12.37 | A | 0.6 |![affine](Graph_command/affine.svg) |
| aidw | 4.58 | 12.83 | 5.42 | 6.21 | B | 0.6 |![aidw](Graph_command/aidw.svg) |
| aligned-types | 3.82 | 3.66 | 4.39 | 4.71 | A | 0.7 |![aligned-types](Graph_command/aligned-types.svg) |
| all-pairs-distance | 4.96 | 5.24 | 4.81 | 21.34 | B | 0.6 |![all-pairs-distance](Graph_command/all-pairs-distance.svg) |
| amgmk | 2.51 | 2.42 | 2.40 | 2.44 | B | 0.6 |![amgmk](Graph_command/amgmk.svg) |
| ans | 5.19 | 5.46 | exe err | exe err | A | 0.6 |![ans](Graph_command/ans.svg) |
| aobench | 2.46 | 0.27 | 2.57 | 2.47 | A | 0.6 |![aobench](Graph_command/aobench.svg) |
| aop | 7.03 | 6.99 | build err | build err | B | 0.6 |![aop](Graph_command/aop.svg) |
| asmooth | 12.47 | 6.83 | 6.43 | 7.33 | A | 1.6 |![asmooth](Graph_command/asmooth.svg) |
| assert | 2.51 | exe err | build err | 2.58 | B | 0.6 |![assert](Graph_command/assert.svg) |
| asta | 4.36 | build err | build err | 10.71 | B | 0.6 |![asta](Graph_command/asta.svg) |
| atan2 | 2.68 | 3.26 | 2.53 | 2.54 | A | 0.7 |![atan2](Graph_command/atan2.svg) |
| atomicCost | 16.01 | 5.32 | 5.29 | 5.36 | A | 14.6 |![atomicCost](Graph_command/atomicCost.svg) |
| atomicIntrinsics | 3.90 | over 600 | 3.19 | 3.21 | A | 0.6 |![atomicIntrinsics](Graph_command/atomicIntrinsics.svg) |
| atomicPerf | 14.33 | 14.78 | 13.40 | 13.88 | B | 0.6 |![atomicPerf](Graph_command/atomicPerf.svg) |
| atomicReduction | 2.70 | 3.45 | 2.65 | 2.63 | A | 0.8 |![atomicReduction](Graph_command/atomicReduction.svg) |
| attention | 2.48 | 2.44 | 82.59 | 83.85 | A | 1.1 |![attention](Graph_command/attention.svg) |
| axhelm | 2.46 | 2.42 | build err | 2.47 | B | 0.7 |![axhelm](Graph_command/axhelm.svg) |
| babelstream | 2.74 | 2.71 | 2.64 | 2.85 | A | 1.3 |![babelstream](Graph_command/babelstream.svg) |
| background-subtract | 6.52 | 8.11 | 8.51 | 8.33 | A | 0.6 |![background-subtract](Graph_command/background-subtract.svg) |
| backprop | 2.53 | 2.42 | 2.42 | 2.47 | B | 0.6 |![backprop](Graph_command/backprop.svg) |
| bezier-surface | 12.43 | 13.00 | 14.05 | 10.71 | A | 1.3 |![bezier-surface](Graph_command/bezier-surface.svg) |
| bfs | 0.98 | 1.00 | 2.43 | 2.46 | A | 0.6 |![bfs](Graph_command/bfs.svg) |
| bilateral | 7.64 | 15.20 | 205.58 | 7.85 | A | 0.6 |![bilateral](Graph_command/bilateral.svg) |
| binomial | 3.24 | 3.09 | | -- | B | 0.6 |![binomial](Graph_command/binomial.svg) |
| bitonic-sort | 13.52 | 10.77 | 7.74 | 8.49 | A | 0.7 |![bitonic-sort](Graph_command/bitonic-sort.svg) |
| black-scholes | 4.80 | 5.03 | 39.34 | 4.88 | A | 2.5 |![black-scholes](Graph_command/black-scholes.svg) |
| blas-gemm | 5.70 | build err | | -- | A | 1.1 |![blas-gemm](Graph_command/blas-gemm.svg) |
| bn | 4.47 | 4.97 | build err | exe err | B | 0.6 |![bn](Graph_command/bn.svg) |
| bonds | 11.93 | 9.73 | 11.47 | 11.19 | A | 0.8 |![bonds](Graph_command/bonds.svg) |
| boxfilter | 9.06 | 8.12 | 6.45 | 9.55 | B | 0.6 |![boxfilter](Graph_command/boxfilter.svg) |
| bsearch | 33.96 | 35.63 | 32.09 | 31.84 | B | 27.3 |![bsearch](Graph_command/bsearch.svg) |
| bspline-vgh | 9.73 | 7.46 | 4.95 | 4.81 | A | 2.6 |![bspline-vgh](Graph_command/bspline-vgh.svg) |
| b+tree | 2.87 | 2.34 | 2.48 | 2.46 | A | 0.6 |![b+tree](Graph_command/b+tree.svg) |
| burger | 84.00 | 72.53 | 74.38 | 112.53 | A | 8.4 |![burger](Graph_command/burger.svg) |
| bwt | 204.75 | 191.14 | 190.22 | exe err | A | 1.3 |![bwt](Graph_command/bwt.svg) |
| car | 7.41 | 7.48 | 6.45 | 6.65 | A | 6.6 |![car](Graph_command/car.svg) |
| cbsfil | 32.08 | 33.27 | 33.71 | 33.19 | A | 2.4 |![cbsfil](Graph_command/cbsfil.svg) |
| ccsd-trpdrv | 2.61 | 2.64 | 2.59 | 2.63 | A | 0.6 |![ccsd-trpdrv](Graph_command/ccsd-trpdrv.svg) |
| ccs | 3.36 | 3.63 | 3.14 | 3.65 | B | 0.6 |![ccs](Graph_command/ccs.svg) |
| cfd | 0.82 | 0.57 | 2.99 | 3.08 | A | 0.6 |![cfd](Graph_command/cfd.svg) |
| chacha20 | 4.74 | 4.66 | 9.32 | 6.66 | B | 0.6 |![chacha20](Graph_command/chacha20.svg) |
| channelShuffle | 18.46 | 542.77 | 19.58 | 19.24 | A | 13.1 |![channelShuffle](Graph_command/channelShuffle.svg) |
| channelSum | 65.58 | 66.66 | 66.43 | 68.17 | A | 6.8 |![channelSum](Graph_command/channelSum.svg) |
| chemv | 3.21 | 3.13 | 2.89 | 3.21 | B | 0.6 |![chemv](Graph_command/chemv.svg) |
| che | 3.62 | 3.05 | 3.40 | 3.38 | A | 1.1 |![che](Graph_command/che.svg) |
| chi2 | 197.09 | 243.23 | 223.07 | 251.54 | A | 51.0 |![chi2](Graph_command/chi2.svg) |
| clenergy | 2.68 | 2.64 | 3.14 | 3.16 | A | 0.6 |![clenergy](Graph_command/clenergy.svg) |
| clink | 4.98 | 54.58 | 6.21 | 8.15 | A | 1.8 |![clink](Graph_command/clink.svg) |
| cmp | exe err | exe err | build err | exe err | A | |![cmp](Graph_command/cmp.svg) |
| cm | exe err | exe err | build err | exe err | B | |![cm](Graph_command/cm.svg) |
| cobahh | 487.96 | 408.69 | 386.46 | 388.03 | A | 55.9 |![cobahh](Graph_command/cobahh.svg) |
| colorwheel | 39.48 | 37.38 | 27.65 | 27.96 | A | 2.3 |![colorwheel](Graph_command/colorwheel.svg) |
| columnarSolver | 9.83 | 10.23 | 16.88 | 10.22 | B | 1.6 |![columnarSolver](Graph_command/columnarSolver.svg) |
| complex | 23.04 | 22.26 | 12.59 | 12.96 | A | 0.7 |![complex](Graph_command/complex.svg) |
| compute-score | 3.28 | 3.26 | 3.28 | 4.34 | B | 1.2 |![compute-score](Graph_command/compute-score.svg) |
| concat | 17.45 | 18.29 | 16.96 | 15.73 | A | 4.7 |![concat](Graph_command/concat.svg) |
| contract | 9.28 | 11.37 | 28.99 | 10.82 | A | 0.9 |![contract](Graph_command/contract.svg) |
| conversion | 5.09 | 5.03 | 3.58 | 3.44 | A | 16.6 |![conversion](Graph_command/conversion.svg) |
| convolution1D | 382.64 | 329.18 | 369.85 | 383.27 | B | 16.7 |![convolution1D](Graph_command/convolution1D.svg) |
| convolution3D | 2.55 | 2.45 | 3.13 | 2.51 | A | 0.6 |![convolution3D](Graph_command/convolution3D.svg) |
| convolutionSeparable | 13.37 | 13.62 | build err | 22.05 | B | 3.6 |![convolutionSeparable](Graph_command/convolutionSeparable.svg) |
| cooling | 381.03 | 156.60 | 447.95 | 454.09 | A | 14.3 |![cooling](Graph_command/cooling.svg) |
| crc64 | 56.48 | 58.40 | 57.23 | 57.25 | A | 1.9 |![crc64](Graph_command/crc64.svg) |
| cross | 12.82 | 15.25 | 7.71 | 8.04 | A | 6.1 |![cross](Graph_command/cross.svg) |
| crs | 25.43 | 25.87 | build err | over 600 | B | 0.6 |![crs](Graph_command/crs.svg) |
| d2q9-bgk | 2.59 | 2.87 | build err | 4.37 | B | 0.9 |![d2q9-bgk](Graph_command/d2q9-bgk.svg) |
| damage | 29.66 | 24.37 | 36.64 | 243.28 | B | 8.7 |![damage](Graph_command/damage.svg) |
| dct8x8 | 85.64 | 103.52 | build err | build err | B | 32.8 |![dct8x8](Graph_command/dct8x8.svg) |
| ddbp | 16.32 | 17.50 | build err | 17.27 | B | 3.9 |![ddbp](Graph_command/ddbp.svg) |
| debayer | 6.75 | 6.40 | 6.63 | 6.79 | B | 2.7 |![debayer](Graph_command/debayer.svg) |
| degrid | 14.49 | 14.13 | 23.75 | 17.04 | A | 1.7 |![degrid](Graph_command/degrid.svg) |
| dense-embedding | 112.61 | 169.13 | 102.69 | 184.08 | B | 14.6 |![dense-embedding](Graph_command/dense-embedding.svg) |
| depixel | 105.99 | 158.76 | 210.60 | 168.04 | A | 8.2 |![depixel](Graph_command/depixel.svg) |
| deredundancy | 16.29 | 17.83 | 15.69 | 15.76 | B | 3.5 |![deredundancy](Graph_command/deredundancy.svg) |
| diamond | 36.76 | build err | build err | 37.30 | A | 0.7 |![diamond](Graph_command/diamond.svg) |
| distort | 2.52 | 0.40 | 2.45 | 2.50 | A | 0.6 |![distort](Graph_command/distort.svg) |
| divergence | 4.74 | 5.23 | 3.17 | 3.15 | A | 0.6 |![divergence](Graph_command/divergence.svg) |
| doh | 14.61 | 14.82 | 2.41 | 2.41 | A | 0.6 |![doh](Graph_command/doh.svg) |
| dp | 53.24 | build err | 25.09 | 25.25 | A | 49.5 |![dp](Graph_command/dp.svg) |
| dslash | 2.68 | 2.67 | 2.66 | 2.66 | A | 1.9 |![dslash](Graph_command/dslash.svg) |
| dxtc2 | 2.49 | 2.55 | build err | over 600 | B | 0.6 |![dxtc2](Graph_command/dxtc2.svg) |
| easyWave | 3.09 | 3.03 | 2.82 | 3.08 | A | 0.8 |![easyWave](Graph_command/easyWave.svg) |
| ecdh | 14.96 | 11.26 | 18.10 | 19.13 | A | 4.4 |![ecdh](Graph_command/ecdh.svg) |
| eigenvalue | 21.76 | 22.13 | over 600 | 22.02 | A | 0.6 |![eigenvalue](Graph_command/eigenvalue.svg) |
| entropy | 48.31 | 47.51 | 94.92 | 49.43 | B | 2.5 |![entropy](Graph_command/entropy.svg) |
| epistasis | 54.59 | 54.50 | 259.10 | 110.41 | A | 4.3 |![epistasis](Graph_command/epistasis.svg) |
| expdist | 7.57 | 7.57 | build err | 12.43 | B | 0.6 |![expdist](Graph_command/expdist.svg) |
| extend2 | 5.80 | 5.32 | 4.60 | 4.94 | A | 0.6 |![extend2](Graph_command/extend2.svg) |
| extrema | 7.05 | 6.78 | 7.05 | 7.10 | A | 0.6 |![extrema](Graph_command/extrema.svg) |
| face | 2.72 | 2.64 | 2.46 | exe err | A | 0.6 |![face](Graph_command/face.svg) |
| fdtd3d | 30.14 | 31.28 | build err | 43.29 | B | 1.0 |![fdtd3d](Graph_command/fdtd3d.svg) |
| feynman-kac | 81.22 | 41.86 | 48.19 | 22.16 | A | 0.6 |![feynman-kac](Graph_command/feynman-kac.svg) |
| fft | 3.17 | 3.20 | build err | 3.19 | B | 0.8 |![fft](Graph_command/fft.svg) |
| fhd | over 600 | 183.97 | over 600 | exe err | A | 2.5 |![fhd](Graph_command/fhd.svg) |
| filter | 168.70 | build err | build err | 154.77 | B | 11.3 |![filter](Graph_command/filter.svg) |
| flip | 10.43 | 36.05 | 12.45 | 15.50 | A | 26.9 |![flip](Graph_command/flip.svg) |
| floydwarshall | 74.91 | 73.40 | 72.54 | 83.41 | A | 0.7 |![floydwarshall](Graph_command/floydwarshall.svg) |
| fluidSim | 19.45 | 17.74 | 17.21 | 17.36 | A | 0.6 |![fluidSim](Graph_command/fluidSim.svg) |
| fpc | 67.41 | 65.55 | build err | exe err | B | 2.6 |![fpc](Graph_command/fpc.svg) |
| fpdc | 4.82 | 5.01 | build err | over 600 | B | 2.1 |![fpdc](Graph_command/fpdc.svg) |
| fresnel | 6.64 | 8.03 | 6.72 | 6.61 | A | 1.8 |![fresnel](Graph_command/fresnel.svg) |
| frna | 103.73 | 106.60 | build err | build err | B | 3.9 |![frna](Graph_command/frna.svg) |
| fwt | 3.13 | 3.08 | 3.26 | 3.47 | B | 0.6 |![fwt](Graph_command/fwt.svg) |
| gabor | 21.34 | 21.65 | 21.24 | 57.42 | A | 9.7 |![gabor](Graph_command/gabor.svg) |
| gamma-correction | 68.53 | 63.52 | 105.19 | 104.46 | A | 1.1 |![gamma-correction](Graph_command/gamma-correction.svg) |
| ga | 48.84 | 35.91 | 32.95 | 33.46 | A | 0.6 |![ga](Graph_command/ga.svg) |
| gaussian | 15.65 | 4.11 | 4.08 | 4.07 | A | 0.7 |![gaussian](Graph_command/gaussian.svg) |
| gc | 2.59 | 2.50 | build err | build err | A | 0.6 |![gc](Graph_command/gc.svg) |
| gd | 12.38 | 13.11 | 12.67 | 12.83 | A | 0.8 |![gd](Graph_command/gd.svg) |
| geglu | 27.67 | 94.31 | 22.15 | 22.07 | A | 8.2 |![geglu](Graph_command/geglu.svg) |
| geodesic | 2.54 | 2.61 | 2.52 | 2.52 | A | 0.8 |![geodesic](Graph_command/geodesic.svg) |
| glu | 59.42 | 109.24 | 97.27 | 120.93 | A | 15.8 |![glu](Graph_command/glu.svg) |
| gmm | 35.58 | 34.84 | build err | 37.28 | B | 1.5 |![gmm](Graph_command/gmm.svg) |
| goulash | 44.04 | 27.38 | 36.42 | 36.52 | A | 26.2 |![goulash](Graph_command/goulash.svg) |
| gpp | 5.15 | 4.32 | 4.15 | 4.15 | A | 2.7 |![gpp](Graph_command/gpp.svg) |
| grep | 55.53 | 57.43 | build err | 30.15 | B | 4.9 |![grep](Graph_command/grep.svg) |
| grrt | 14.22 | 14.59 | 13.38 | 12.82 | A | 0.6 |![grrt](Graph_command/grrt.svg) |
| haccmk | 4.70 | 4.34 | 2.49 | 4.42 | A | 0.6 |![haccmk](Graph_command/haccmk.svg) |
| hausdorff | 30.92 | 32.10 | 32.24 | 31.92 | A | 0.6 |![hausdorff](Graph_command/hausdorff.svg) |
| haversine | 2.22 | 2.20 | 2.48 | 2.43 | A | 1.0 |![haversine](Graph_command/haversine.svg) |
| heartwall | 7.48 | exe err | build err | build err | A | 2.7 |![heartwall](Graph_command/heartwall.svg) |
| heat2d | 6.14 | 62.53 | 25.48 | 5.92 | A | 0.8 |![heat2d](Graph_command/heat2d.svg) |
| heat | 22.95 | 22.27 | 15.34 | 16.22 | A | 16.9 |![heat](Graph_command/heat.svg) |
| hellinger | 4.94 | 4.86 | 6.80 | 4.52 | A | 0.6 |![hellinger](Graph_command/hellinger.svg) |
| henry | 3.91 | 3.71 | 2.82 | 2.74 | A | 0.6 |![henry](Graph_command/henry.svg) |
| hexciton | 4.19 | 6.07 | build err | 4.96 | B | 0.6 |![hexciton](Graph_command/hexciton.svg) |
| histogram | 3.06 | 3.06 | build err | 3.30 | B | 0.6 |![histogram](Graph_command/histogram.svg) |
| hmm | 4.22 | 8.60 | 14.59 | 2.81 | A | 0.7 |![hmm](Graph_command/hmm.svg) |
| hogbom | 0.53 | 0.41 | 2.57 | 2.76 | B | 0.7 |![hogbom](Graph_command/hogbom.svg) |
| hotspot3D | 18.45 | 16.47 | 18.17 | 18.00 | A | 0.6 |![hotspot3D](Graph_command/hotspot3D.svg) |
| hwt1d | 3.15 | 3.15 | 3.14 | 3.23 | B | 0.6 |![hwt1d](Graph_command/hwt1d.svg) |
| hybridsort | 12.83 | 15.37 | build err | 12.96 | B | 1.2 |![hybridsort](Graph_command/hybridsort.svg) |
| hypterm | 5.75 | 5.88 | 13.12 | 13.12 | A | 3.5 |![hypterm](Graph_command/hypterm.svg) |
| idivide | 6.12 | 8.42 | 8.03 | 9.39 | A | 0.6 |![idivide](Graph_command/idivide.svg) |
| interleave | 4.15 | 4.10 | 4.03 | 3.44 | A | 0.6 |![interleave](Graph_command/interleave.svg) |
| interval | 7.59 | 7.20 | 13.47 | 10.78 | A | 1.0 |![interval](Graph_command/interval.svg) |
| inversek2j | 2.84 | 2.76 | 86.59 | 111.49 | A | 0.6 |![inversek2j](Graph_command/inversek2j.svg) |
| ising | 10.77 | 259.03 | 13.64 | 13.31 | A | 2.4 |![ising](Graph_command/ising.svg) |
| iso2dfd | 17.76 | 20.00 | 14.48 | 14.20 | A | 49.7 |![iso2dfd](Graph_command/iso2dfd.svg) |
| jacobi | 2.80 | 2.79 | 3.63 | 3.50 | A | 0.6 |![jacobi](Graph_command/jacobi.svg) |
| jenkins-hash | 5.05 | 5.02 | 5.19 | 5.17 | A | 10.0 |![jenkins-hash](Graph_command/jenkins-hash.svg) |
| kalman | 335.35 | over 600 | build err | exe err | A | 10.8 |![kalman](Graph_command/kalman.svg) |
| keccaktreehash | 7.76 | 8.16 | 11.48 | 10.08 | A | 0.7 |![keccaktreehash](Graph_command/keccaktreehash.svg) |
| keogh | 22.46 | 7.05 | 5.78 | 5.89 | A | 0.9 |![keogh](Graph_command/keogh.svg) |
| kernelLaunch | 13.80 | 14.57 | 282.44 | 73.99 | B | 0.6 |![kernelLaunch](Graph_command/kernelLaunch.svg) |
| kmeans | 28.57 | 29.26 | 27.23 | 22.68 | A | 0.7 |![kmeans](Graph_command/kmeans.svg) |
| knn | 4.43 | 4.44 | 4.30 | 4.53 | B | 0.6 |![knn](Graph_command/knn.svg) |
| lanczos | 2.61 | 2.61 | build err | 2.81 | B | 0.6 |![lanczos](Graph_command/lanczos.svg) |
| langevin | 21.81 | 19.92 | 6.19 | 31.31 | A | 31.1 |![langevin](Graph_command/langevin.svg) |
| langford | 2.76 | 2.64 | build err | build err | B | 2.1 |![langford](Graph_command/langford.svg) |
| laplace3d | 65.95 | 65.29 | 14.70 | 14.64 | A | 8.8 |![laplace3d](Graph_command/laplace3d.svg) |
| laplace | 10.89 | 340.70 | 11.47 | 11.34 | A | 0.6 |![laplace](Graph_command/laplace.svg) |
| lavaMD | 104.27 | 102.82 | build err | 116.55 | B | 49.9 |![lavaMD](Graph_command/lavaMD.svg) |
| layout | 3.28 | 3.25 | 2.68 | 2.93 | A | 0.6 |![layout](Graph_command/layout.svg) |
| lci | 17.78 | 10.01 | 4.40 | 4.36 | A | 0.6 |![lci](Graph_command/lci.svg) |
| lda | 3.10 | 3.13 | build err | build err | B | 0.6 |![lda](Graph_command/lda.svg) |
| ldpc | 4.83 | 4.97 | build err | 9.60 | B | 0.6 |![ldpc](Graph_command/ldpc.svg) |
| lebesgue | 39.25 | 41.72 | 39.40 | 39.11 | A | 1.0 |![lebesgue](Graph_command/lebesgue.svg) |
| leukocyte | 0.64 | build err | build err | 0.92 | A | 0.6 |![leukocyte](Graph_command/leukocyte.svg) |
| libor | 2.87 | 2.81 | 13.21 | 2.84 | A | 4.0 |![libor](Graph_command/libor.svg) |
| lid-driven-cavity | 10.21 | 10.86 | build err | 22.40 | B | 0.6 |![lid-driven-cavity](Graph_command/lid-driven-cavity.svg) |
| lif | 100.17 | 124.57 | 125.51 | 124.42 | A | 31.9 |![lif](Graph_command/lif.svg) |
| linearprobing | 105.14 | 103.66 | build err | build err | A | 1.4 |![linearprobing](Graph_command/linearprobing.svg) |
| log2 | 2.80 | 2.76 | 2.74 | 2.81 | A | 1.3 |![log2](Graph_command/log2.svg) |
| lombscargle | 3.14 | 3.09 | 3.09 | 3.05 | A | 0.6 |![lombscargle](Graph_command/lombscargle.svg) |
| loopback | 5.73 | 5.31 | 5.44 | 8.16 | B | 0.6 |![loopback](Graph_command/loopback.svg) |
| lrn | 104.40 | 79.53 | 134.66 | 78.63 | A | 22.5 |![lrn](Graph_command/lrn.svg) |
| lr | 6.18 | 6.19 | 14.73 | 31.82 | B | 0.6 |![lr](Graph_command/lr.svg) |
| lsqt | 19.97 | 20.11 | 32.68 | 32.86 | A | 1.7 |![lsqt](Graph_command/lsqt.svg) |
| lud | 5.54 | 8.26 | 9.45 | 46.07 | B | 4.7 |![lud](Graph_command/lud.svg) |
| lulesh | 7.81 | 11.36 | 11.75 | 8.41 | A | 54.2 |![lulesh](Graph_command/lulesh.svg) |
| mallocFree | 3.72 | 3.74 | 2.60 | 2.67 | A | 8.4 |![mallocFree](Graph_command/mallocFree.svg) |
| mandelbrot | 5.75 | 5.52 | 3.75 | 3.82 | A | 0.6 |![mandelbrot](Graph_command/mandelbrot.svg) |
| mask | 22.03 | 55.68 | 27.45 | 29.68 | A | 8.8 |![mask](Graph_command/mask.svg) |
| match | 40.66 | 42.53 | build err | 41.46 | B | 0.6 |![match](Graph_command/match.svg) |
| matern | 6.11 | 6.28 | 3.79 | 31.37 | B | 0.7 |![matern](Graph_command/matern.svg) |
| matrix-rotate | 32.32 | 35.67 | 34.75 | 35.00 | A | 8.6 |![matrix-rotate](Graph_command/matrix-rotate.svg) |
| maxFlops | 25.53 | 25.61 | 26.46 | 25.46 | A | 0.6 |![maxFlops](Graph_command/maxFlops.svg) |
| maxpool3d | 32.54 | 33.07 | 32.43 | 32.24 | A | 10.2 |![maxpool3d](Graph_command/maxpool3d.svg) |
| mcmd | 12.38 | 9.08 | 10.78 | 11.06 | A | 0.6 |![mcmd](Graph_command/mcmd.svg) |
| mcpr | 25.42 | 14.03 | 21.95 | 15.00 | B | 0.6 |![mcpr](Graph_command/mcpr.svg) |
| md5hash | 15.45 | 15.21 | 15.33 | 15.50 | A | 0.6 |![md5hash](Graph_command/md5hash.svg) |
| mdh | 2.80 | 48.72 | 35.43 | 2.67 | A | 0.6 |![mdh](Graph_command/mdh.svg) |
| md | 14.56 | 14.70 | 14.20 | 14.22 | A | 0.6 |![md](Graph_command/md.svg) |
| meanshift | 3.76 | 4.36 | 4.76 | 4.40 | B | 0.6 |![meanshift](Graph_command/meanshift.svg) |
| medianfilter | 3.54 | 3.86 | 2.89 | 2.84 | B | 0.6 |![medianfilter](Graph_command/medianfilter.svg) |
| memcpy | 5.61 | 5.85 | 6.57 | 6.54 | A | 0.6 |![memcpy](Graph_command/memcpy.svg) |
| memtest | 18.43 | 18.79 | 19.76 | 19.45 | A | 2.6 |![memtest](Graph_command/memtest.svg) |
| merge | 432.85 | 451.29 | build err | 425.74 | B | 9.7 |![merge](Graph_command/merge.svg) |
| metropolis | 12.73 | 14.09 | build err | 65.32 | B | 17.2 |![metropolis](Graph_command/metropolis.svg) |
| michalewicz | 43.63 | 150.72 | 59.72 | 63.13 | A | 50.2 |![michalewicz](Graph_command/michalewicz.svg) |
| minibude | 2.89 | 3.36 | 3.12 | 1.49 | B | 0.6 |![minibude](Graph_command/minibude.svg) |
| miniFE | | | | -- | A | |![miniFE](Graph_command/miniFE.svg) |
| minimap2 | 1.51 | 2.30 | 2.13 | 3.68 | A | 1.0 |![minimap2](Graph_command/minimap2.svg) |
| minisweep | 50.83 | 56.85 | 36.24 | 35.90 | A | 7.7 |![minisweep](Graph_command/minisweep.svg) |
| miniWeather | 8.49 | 7.14 | 33.18 | 33.14 | A | 0.6 |![miniWeather](Graph_command/miniWeather.svg) |
| minkowski | 23.94 | 24.15 | 22.45 | 22.36 | A | 0.6 |![minkowski](Graph_command/minkowski.svg) |
| mis | 24.24 | 13.27 | 2.50 | 0.23 | B | 0.6 |![mis](Graph_command/mis.svg) |
| mixbench | 3.77 | 3.59 | 3.72 | 1.55 | B | 0.6 |![mixbench](Graph_command/mixbench.svg) |
| morphology | 5.03 | 4.83 | build err | over 600 | B | 4.4 |![morphology](Graph_command/morphology.svg) |
| mrc | 21.72 | 84.22 | 17.78 | 17.83 | A | 23.5 |![mrc](Graph_command/mrc.svg) |
| mriQ | 4.76 | 5.31 | 5.19 | 5.21 | A | 0.6 |![mriQ](Graph_command/mriQ.svg) |
| mr | 3.31 | 3.40 | build err | 3.31 | A | 0.6 |![mr](Graph_command/mr.svg) |
| mt | 5.66 | 4.64 | 4.21 | 4.08 | A | 0.7 |![mt](Graph_command/mt.svg) |
| multimaterial | 28.40 | 15.15 | 12.10 | 21.91 | A | 3.1 |![multimaterial](Graph_command/multimaterial.svg) |
| murmurhash3 | 2.78 | 2.73 | 2.69 | 2.69 | A | 2.6 |![murmurhash3](Graph_command/murmurhash3.svg) |
| myocyte | 33.92 | 3.50 | 11.18 | 12.83 | A | 0.6 |![myocyte](Graph_command/myocyte.svg) |
| nbody | 23.95 | 25.53 | 19.55 | 19.61 | A | 0.6 |![nbody](Graph_command/nbody.svg) |
| ne | 4.83 | 5.17 | 5.22 | 4.94 | A | 3.2 |![ne](Graph_command/ne.svg) |
| nlll | 3.07 | 3.04 | 82.52 | 3.08 | B | 0.6 |![nlll](Graph_command/nlll.svg) |
| nms | 2.60 | 2.66 | build err | 2.73 | B | 0.6 |![nms](Graph_command/nms.svg) |
| nn | 2.55 | 2.45 | 2.57 | 2.52 | A | 0.6 |![nn](Graph_command/nn.svg) |
| norm2 | 3.77 | build err | 3.76 | 4.97 | A | 2.7 |![norm2](Graph_command/norm2.svg) |
| nqueen | 6.31 | 6.96 | 23.79 | 6.87 | A | 0.6 |![nqueen](Graph_command/nqueen.svg) |
| ntt | 6.21 | 6.65 | build err | 6.65 | B | 0.6 |![ntt](Graph_command/ntt.svg) |
| nw | 16.68 | 26.56 | build err | 33.48 | B | 15.3 |![nw](Graph_command/nw.svg) |
| openmp | 13.88 | 82.22 | 13.80 | 11.52 | B | 0.7 |![openmp](Graph_command/openmp.svg) |
| overlay | 5.87 | 5.51 | 9.28 | 10.50 | A | 0.6 |![overlay](Graph_command/overlay.svg) |
| p4 | 3.78 | 3.74 | build err | 16.94 | B | 0.6 |![p4](Graph_command/p4.svg) |
| page-rank | 50.87 | 369.64 | 49.39 | 46.72 | A | 12.8 |![page-rank](Graph_command/page-rank.svg) |
| particle-diffusion | 11.92 | 11.98 | 11.94 | 12.23 | A | 3.3 |![particle-diffusion](Graph_command/particle-diffusion.svg) |
| particlefilter | 222.50 | 198.23 | build err | 219.65 | B | 4.2 |![particlefilter](Graph_command/particlefilter.svg) |
| particles | 3.67 | 3.44 | build err | over 600 | B | 0.6 |![particles](Graph_command/particles.svg) |
| pathfinder | 63.24 | 63.67 | 68.33 | 824.34 | B | 6.7 |![pathfinder](Graph_command/pathfinder.svg) |
| permutate | 26.61 | 26.60 | build err | 25.87 | B | 2.9 |![permutate](Graph_command/permutate.svg) |
| permute | 61.55 | 61.68 | 85.69 | 85.40 | A | 9.8 |![permute](Graph_command/permute.svg) |
| perplexity | 186.41 | 194.30 | 124.21 | 96.53 | A | 15.8 |![perplexity](Graph_command/perplexity.svg) |
| phmm | 3.72 | 3.50 | 7.15 | 9.31 | B | 0.6 |![phmm](Graph_command/phmm.svg) |
| pnpoly | 7.07 | 21.26 | build err | 74.17 | A | 0.9 |![pnpoly](Graph_command/pnpoly.svg) |
| pns | 5.50 | 5.53 | build err | exe err | B | 1.3 |![pns](Graph_command/pns.svg) |
| pointwise | 21.34 | 23.43 | 87.37 | 75.86 | A | 15.8 |![pointwise](Graph_command/pointwise.svg) |
| pool | 27.23 | 55.17 | 27.74 | 41.00 | A | 8.2 |![pool](Graph_command/pool.svg) |
| popcount | 241.58 | 171.64 | 281.17 | 134.99 | A | 23.4 |![popcount](Graph_command/popcount.svg) |
| present | 2.87 | 2.66 | 3.08 | 3.05 | A | 0.6 |![present](Graph_command/present.svg) |
| prna | 79.29 | build err | build err | exe err | B | 4.8 |![prna](Graph_command/prna.svg) |
| projectile | 2.66 | 2.63 | 2.54 | 2.55 | A | 0.9 |![projectile](Graph_command/projectile.svg) |
| pso | 2.93 | 2.84 | 2.71 | 4.37 | A | 0.6 |![pso](Graph_command/pso.svg) |
| qrg | 7.23 | 6.33 | 8.62 | 10.24 | A | 0.6 |![qrg](Graph_command/qrg.svg) |
| qtclustering | 35.55 | 34.48 | build err | exe err | B | 0.8 |![qtclustering](Graph_command/qtclustering.svg) |
| quantBnB | 18.55 | 122.13 | 124.49 | 125.68 | A | 2.9 |![quantBnB](Graph_command/quantBnB.svg) |
| quicksort | 36.33 | 37.30 | build err | 33.98 | B | 0.8 |![quicksort](Graph_command/quicksort.svg) |
| radixsort | 2.94 | 2.95 | build err | 8.11 | B | 0.6 |![radixsort](Graph_command/radixsort.svg) |
| rainflow | 29.26 | 30.54 | 30.56 | 29.52 | A | 49.0 |![rainflow](Graph_command/rainflow.svg) |
| randomAccess | 6.66 | 6.98 | build err | 24.54 | A | 1.1 |![randomAccess](Graph_command/randomAccess.svg) |
| reaction | 4.18 | 4.08 | build err | 9.17 | B | 0.6 |![reaction](Graph_command/reaction.svg) |
| recursiveGaussian | 4.03 | 4.11 | build err | 4.27 | B | 0.6 |![recursiveGaussian](Graph_command/recursiveGaussian.svg) |
| resize | 6.38 | 6.04 | 6.01 | 6.01 | A | 4.7 |![resize](Graph_command/resize.svg) |
| reverse | 3.47 | 3.71 | 6.76 | 5.70 | B | 0.6 |![reverse](Graph_command/reverse.svg) |
| rfs | 46.68 | 48.23 | 48.97 | 48.26 | A | 6.3 |![rfs](Graph_command/rfs.svg) |
| rng-wallace | 2.62 | 2.56 | build err | 3.55 | B | 0.8 |![rng-wallace](Graph_command/rng-wallace.svg) |
| rodrigues | 96.39 | 101.95 | 104.01 | 101.38 | A | 54.0 |![rodrigues](Graph_command/rodrigues.svg) |
| romberg | 2.75 | 2.81 | 2.75 | 0.54 | B | 0.6 |![romberg](Graph_command/romberg.svg) |
| rsbench | 3.01 | 3.42 | 9.72 | 2.98 | A | 0.6 |![rsbench](Graph_command/rsbench.svg) |
| rsc | 2.75 | 2.65 | 2.65 | 2.80 | B | 0.6 |![rsc](Graph_command/rsc.svg) |
| rtm8 | 4.35 | 17.75 | 16.01 | 4.22 | A | 1.5 |![rtm8](Graph_command/rtm8.svg) |
| rushlarsen | 140.36 | 132.18 | 138.89 | 135.06 | A | 6.0 |![rushlarsen](Graph_command/rushlarsen.svg) |
| s3d | 2.92 | 3.00 | build err | 2.90 | A | 0.9 |![s3d](Graph_command/s3d.svg) |
| s8n | 2.58 | 2.48 | 2.48 | 2.43 | A | 0.6 |![s8n](Graph_command/s8n.svg) |
| sad | 2.40 | 2.50 | 2.78 | 81.81 | A | 0.6 |![sad](Graph_command/sad.svg) |
| sampling | 5.85 | 5.79 | 5.87 | 5.64 | B | 0.7 |![sampling](Graph_command/sampling.svg) |
| scan2 | 2.93 | 2.88 | build err | 6.76 | B | 0.8 |![scan2](Graph_command/scan2.svg) |
| scan | 33.88 | 34.35 | build err | 33.05 | B | 1.3 |![scan](Graph_command/scan.svg) |
| scatterAdd | 7.95 | 7.96 | 7.90 | 7.79 | A | 8.0 |![scatterAdd](Graph_command/scatterAdd.svg) |
| scel | 17.04 | 324.29 | 12.00 | 11.29 | A | 3.6 |![scel](Graph_command/scel.svg) |
| secp256k1 | 2.77 | 2.76 | 2.77 | 2.69 | A | 0.6 |![secp256k1](Graph_command/secp256k1.svg) |
| sheath | 5.92 | 5.88 | 5.80 | 5.82 | A | 0.6 |![sheath](Graph_command/sheath.svg) |
| shmembench | 5.25 | 9.24 | build err | exe err | B | 0.6 |![shmembench](Graph_command/shmembench.svg) |
| simplemoc | 233.38 | 400.99 | 295.21 | 294.34 | B | 15.4 |![simplemoc](Graph_command/simplemoc.svg) |
| simpleSpmv | 317.74 | 320.94 | 318.85 | 317.47 | A | 7.0 |![simpleSpmv](Graph_command/simpleSpmv.svg) |
| slu | build err | build err | build err | build err | B | |![slu](Graph_command/slu.svg) |
| snake | 8.17 | 8.47 | 10.41 | 10.42 | A | 0.6 |![snake](Graph_command/snake.svg) |
| sobel | 2.81 | 2.93 | 3.42 | 3.32 | A | 0.6 |![sobel](Graph_command/sobel.svg) |
| sobol | 4.71 | 3.98 | build err | 4.51 | B | 4.4 |![sobol](Graph_command/sobol.svg) |
| softmax-online | 11.10 | 13.74 | build err | 15.10 | B | 3.7 |![softmax-online](Graph_command/softmax-online.svg) |
| softmax | 44.19 | 39.95 | 47.59 | 36.96 | A | 25.9 |![softmax](Graph_command/softmax.svg) |
| sort | 4.69 | exe err | build err | 36.92 | B | 0.8 |![sort](Graph_command/sort.svg) |
| sosfil | 4.65 | 2.56 | 5.71 | 13.04 | B | 0.6 |![sosfil](Graph_command/sosfil.svg) |
| sph | 5.45 | 5.33 | 22.55 | 5.40 | A | 0.6 |![sph](Graph_command/sph.svg) |
| split | 35.89 | 35.22 | build err | 37.73 | B | 15.8 |![split](Graph_command/split.svg) |
| spm | 100.86 | 107.20 | 95.88 | 98.41 | A | 10.6 |![spm](Graph_command/spm.svg) |
| sptrsv | 2.44 | 2.67 | build err | 2.81 | B | 0.6 |![sptrsv](Graph_command/sptrsv.svg) |
| srad | 36.80 | 37.48 | build err | exe err | B | 14.7 |![srad](Graph_command/srad.svg) |
| ss | 7.45 | 7.16 | 9.75 | 18.10 | B | 0.6 |![ss](Graph_command/ss.svg) |
| stddev | 48.33 | 48.29 | 48.56 | 50.49 | B | 8.7 |![stddev](Graph_command/stddev.svg) |
| stencil1d | 3.75 | 4.76 | 26.58 | 35.77 | A | 4.7 |![stencil1d](Graph_command/stencil1d.svg) |
| stencil3d | 4.01 | 4.39 | 3.07 | 4.27 | B | 11.8 |![stencil3d](Graph_command/stencil3d.svg) |
| streamcluster | 5.89 | 7.41 | 9.67 | 7.74 | B | 0.7 |![streamcluster](Graph_command/streamcluster.svg) |
| su3 | 2.99 | 5.09 | 7.42 | 7.48 | A | 2.5 |![su3](Graph_command/su3.svg) |
| surfel | 13.03 | 518.05 | 12.16 | 11.97 | A | 0.6 |![surfel](Graph_command/surfel.svg) |
| svd3x3 | 1.91 | 1.96 | 2.85 | 2.89 | A | 0.7 |![svd3x3](Graph_command/svd3x3.svg) |
| sw4ck | 3.58 | 6.42 | 28.30 | 28.11 | A | 1.9 |![sw4ck](Graph_command/sw4ck.svg) |
| swish | 7.49 | 24.43 | 5.46 | 5.39 | A | 8.2 |![swish](Graph_command/swish.svg) |
| tensorT | 2.73 | 2.60 | 2.60 | 2.95 | B | 3.6 |![tensorT](Graph_command/tensorT.svg) |
| testSNAP | 3.69 | 3.75 | 66.06 | 4.19 | A | 3.2 |![testSNAP](Graph_command/testSNAP.svg) |
| thomas | 125.54 | 137.36 | 54.85 | 49.73 | A | 51.8 |![thomas](Graph_command/thomas.svg) |
| threadfence | 25.80 | 26.25 | 17.46 | 33.08 | B | 6.0 |![threadfence](Graph_command/threadfence.svg) |
| tissue | 11.96 | 12.18 | 11.78 | 18.23 | A | 0.6 |![tissue](Graph_command/tissue.svg) |
| tonemapping | 3.24 | 3.15 | 2.99 | 2.95 | A | 0.6 |![tonemapping](Graph_command/tonemapping.svg) |
| tqs | 2.71 | 2.59 | build err | 2.59 | B | 0.6 |![tqs](Graph_command/tqs.svg) |
| triad | 2.72 | 2.68 | 2.83 | 2.75 | A | 0.7 |![triad](Graph_command/triad.svg) |
| tridiagonal | 8.42 | 7.70 | build err | 24.55 | B | 4.0 |![tridiagonal](Graph_command/tridiagonal.svg) |
| tsa | 54.11 | 51.39 | build err | 29.09 | B | 20.6 |![tsa](Graph_command/tsa.svg) |
| tsp | 5.10 | 7.91 | build err | build err | B | 0.6 |![tsp](Graph_command/tsp.svg) |
| urng | 2.47 | 2.44 | 2.09 | 2.20 | B | 0.6 |![urng](Graph_command/urng.svg) |
| vanGenuchten | 21.39 | 21.34 | 31.07 | 30.92 | A | 6.2 |![vanGenuchten](Graph_command/vanGenuchten.svg) |
| vmc | 5.07 | 4.41 | build err | build err | B | 0.6 |![vmc](Graph_command/vmc.svg) |
| vol2col | 8.27 | 7.41 | 7.92 | 8.13 | A | 3.5 |![vol2col](Graph_command/vol2col.svg) |
| wedford | 34.48 | 35.74 | build err | 33.65 | B | 8.8 |![wedford](Graph_command/wedford.svg) |
| winograd | 0.84 | 2.73 | 3.24 | 3.20 | A | 0.6 |![winograd](Graph_command/winograd.svg) |
| wlcpow | 4.53 | 4.55 | build err | 12.37 | B | 0.7 |![wlcpow](Graph_command/wlcpow.svg) |
| wordcount | 8.03 | 8.16 | 7.99 | 7.93 | A | 0.8 |![wordcount](Graph_command/wordcount.svg) |
| wsm5 | 5.57 | 4.83 | 5.56 | 5.73 | B | 0.6 |![wsm5](Graph_command/wsm5.svg) |
| wyllie | 107.36 | 113.04 | 109.34 | over 600 | B | 1.3 |![wyllie](Graph_command/wyllie.svg) |
| xlqc | 2.07 | exe err | build err | 0.36 | B | 0.6 |![xlqc](Graph_command/xlqc.svg) |
| xsbench | 46.19 | 37.77 | 37.47 | 3.89 | A | 6.3 |![xsbench](Graph_command/xsbench.svg) |
| zeropoint | 18.63 | 52.46 | 21.66 | 24.23 | A | 15.8 |![zeropoint](Graph_command/zeropoint.svg) |
| zmddft | 2.63 | 2.57 | build err | 14.28 | B | 1.6 |![zmddft](Graph_command/zmddft.svg) |
| | | | | | | |
| 完了した件数 | 320 | 307 | 238 (A:186) | 291 | | ||

sycl と acc がともに完了した件数 235 (A:183)|

※ 分類: OpenMP コードが (omp_get_wtime 以外の) omp_get_* を (A) 含まない (B) 含む

※ メモリ: cuda 版での値
