# ベンチマークが標準出力に出力する値

| 名称 | cuda | sycl | acc | omp  | 単位 | 分類 | グラフ |
|  ---: | ---:  | ---: | ---: | ---: | ---:   | ---:   | ---: |
| accuracy | 6.02e+02 | 6.07e+02 | 6.85e+02 | 8.27e+02 | us | A |![accuracy](Graph_builtin/accuracy.svg) |
| ace | 2.35e+03 | 2.64e+03 | 2.52e+03 | 2.52e+03 | ms | A |![ace](Graph_builtin/ace.svg) |
| adam | 6.02e-02 | 6.23e-02 | 7.61e-02 | 1.06e-01 | ms | A |![adam](Graph_builtin/adam.svg) |
| adamw | 5.17e-02 | 5.42e-02 | 9.97e-02 | 6.95e-01 | ms | A |![adamw](Graph_builtin/adamw.svg) |
| adjacent | 2.40e+03 | 2.41e+03 | 3.04e+03 | 6.03e+03 | us | A |![adjacent](Graph_builtin/adjacent.svg) |
| adv | 3.82e+06 | 5.34e+06 | | 6.92e+06 | ns | B |![adv](Graph_builtin/adv.svg) |
| aes | 3.14e-04 | 2.98e-04 | | 4.23e-04 | s | B |![aes](Graph_builtin/aes.svg) |
| affine | 3.13e-06 | 2.95e-06 | 1.03e-05 | 9.82e-06 | s | A |![affine](Graph_builtin/affine.svg) |
| aidw | 7.70e-03 | 4.73e-02 | 1.95e-02 | 9.10e-03 | s | B |![aidw](Graph_builtin/aidw.svg) |
| aligned-types | 6.29e-01 | 7.25e-01 | 1.47e+00 | 1.69e+00 | ms | A |![aligned-types](Graph_builtin/aligned-types.svg) |
| all-pairs-distance | 2.12e+02 | 2.54e+02 | 2.15e+02 | 1.87e+03 | us | B |![all-pairs-distance](Graph_builtin/all-pairs-distance.svg) |
| amgmk | 2.45e+00 | 2.37e+00 | 2.35e+00 | 2.38e+00 | s | B |![amgmk](Graph_builtin/amgmk.svg) |
| ans | 5.14e+00 | 5.40e+00 | | | s | A |![ans](Graph_builtin/ans.svg) |
| aobench | 8.90e+01 | 1.02e+02 | 7.87e+01 | 1.06e+02 | us | A |![aobench](Graph_builtin/aobench.svg) |
| aop | 4.00e+00 | 4.01e+00 | | | ms | B |![aop](Graph_builtin/aop.svg) |
| asmooth | 5.16e-03 | 5.68e-03 | 5.65e-03 | 5.80e-03 | s | A |![asmooth](Graph_builtin/asmooth.svg) |
| assert | 2.43e+00 | 3.34e-02 | | 5.02e+00 | ns | B |![assert](Graph_builtin/assert.svg) |
| asta | 1.53e-02 | | | 7.29e-02 | s | B |![asta](Graph_builtin/asta.svg) |
| atan2 | 1.38e+02 | 1.41e+02 | 1.70e+02 | 1.89e+02 | us | A |![atan2](Graph_builtin/atan2.svg) |
| atomicCost | 1.09e+04 | 1.08e+04 | 1.15e+04 | 1.42e+04 | us | A |![atomicCost](Graph_builtin/atomicCost.svg) |
| atomicIntrinsics | 4.23e+05 | | 3.12e+04 | 3.38e+04 | us | A |![atomicIntrinsics](Graph_builtin/atomicIntrinsics.svg) |
| atomicPerf | 6.61e+03 | 6.62e+03 | 6.69e+03 | 7.20e+03 | us | B |![atomicPerf](Graph_builtin/atomicPerf.svg) |
| atomicReduction | 5.12e+04 | 5.11e+04 | 4.55e+04 | 4.59e+04 | GB/s | A |![atomicReduction](Graph_builtin/atomicReduction.svg) |
| attention | 4.31e-01 | 4.31e-01 | 4.31e-01 | 9.30e-01 | ms | A |![attention](Graph_builtin/attention.svg) |
| axhelm | 2.22e+07 | 2.18e+07 | | 2.18e+07 | ns | B |![axhelm](Graph_builtin/axhelm.svg) |
| babelstream | 6.69e+06 | 6.68e+06 | 6.59e+06 | 6.04e+06 | s | A |![babelstream](Graph_builtin/babelstream.svg) |
| background-subtract | 1.03e+02 | 1.06e+02 | 1.14e+02 | 1.63e+02 | us | A |![background-subtract](Graph_builtin/background-subtract.svg) |
| backprop | 2.45e+00 | 2.34e+00 | 2.35e+00 | 2.39e+00 | s | B |![backprop](Graph_builtin/backprop.svg) |
| bezier-surface | 8.80e+01 | 1.13e+02 | 3.18e+03 | 9.70e+01 | ms | A |![bezier-surface](Graph_builtin/bezier-surface.svg) |
| bfs | 3.84e+02 | 5.30e+02 | 4.43e+02 | 1.27e+03 | us | A |![bfs](Graph_builtin/bfs.svg) |
| bilateral | 4.86e+00 | 4.90e+00 | 2.03e+02 | 5.22e+00 | ms | A |![bilateral](Graph_builtin/bilateral.svg) |
| binomial | 2.99e+03 | 2.77e+03 | | | ms | B |![binomial](Graph_builtin/binomial.svg) |
| bitonic-sort | 3.29e+01 | 3.28e+01 | 3.72e+01 | 5.55e+01 | ms | A |![bitonic-sort](Graph_builtin/bitonic-sort.svg) |
| black-scholes | 2.08e+03 | 2.25e+03 | 1.93e+03 | 2.25e+03 | ms | A |![black-scholes](Graph_builtin/black-scholes.svg) |
| blas-gemm | 5.06e+03 | | | | us | A |![blas-gemm](Graph_builtin/blas-gemm.svg) |
| bn | 1.62e-01 | 2.21e-01 | | | s | B |![bn](Graph_builtin/bn.svg) |
| bonds | 3.76e+01 | 2.21e+01 | 5.24e+01 | 5.05e+01 | ms | A |![bonds](Graph_builtin/bonds.svg) |
| boxfilter | 6.43e+02 | 5.68e+02 | 3.96e+02 | 7.06e+02 | us | B |![boxfilter](Graph_builtin/boxfilter.svg) |
| bsearch | 3.12e+00 | 2.30e+00 | 2.73e-01 | 2.27e-01 | s | B |![bsearch](Graph_builtin/bsearch.svg) |
| bspline-vgh | 1.32e-01 | 1.65e-01 | 1.40e-01 | 1.43e-01 | s | A |![bspline-vgh](Graph_builtin/bspline-vgh.svg) |
| b+tree | 2.06e+02 | 3.28e+02 | 1.64e+03 | 1.39e+03 | us | A |![b+tree](Graph_builtin/b+tree.svg) |
| burger | 5.62e-01 | 5.18e-01 | 6.35e-01 | 5.78e-01 | s | A |![burger](Graph_builtin/burger.svg) |
| bwt | 1.58e+04 | 1.53e+04 | 1.20e+03 | | ms | A |![bwt](Graph_builtin/bwt.svg) |
| car | 4.84e-03 | 4.54e-03 | 5.59e-03 | 5.65e-03 | s | A |![car](Graph_builtin/car.svg) |
| cbsfil | 3.66e-02 | 3.43e-02 | 5.26e-02 | 3.64e-02 | s | A |![cbsfil](Graph_builtin/cbsfil.svg) |
| ccsd-trpdrv | 4.20e-05 | 4.60e-05 | 4.80e-05 | 8.70e-05 | s | A |![ccsd-trpdrv](Graph_builtin/ccsd-trpdrv.svg) |
| ccs | 1.09e-02 | 1.17e-02 | 8.89e-03 | 1.31e-02 | s | B |![ccs](Graph_builtin/ccs.svg) |
| cfd | 8.72e-02 | 7.91e-02 | 3.00e-01 | 4.22e-01 | s | A |![cfd](Graph_builtin/cfd.svg) |
| chacha20 | 2.25e+01 | 2.24e+01 | 6.79e+01 | 4.18e+01 | us | B |![chacha20](Graph_builtin/chacha20.svg) |
| channelShuffle | 2.40e+01 | 2.43e+01 | 3.85e+01 | 3.83e+01 | ms | A |![channelShuffle](Graph_builtin/channelShuffle.svg) |
| channelSum | 6.26e+01 | 6.28e+01 | 5.93e+01 | 8.76e+01 | ms | A |![channelSum](Graph_builtin/channelSum.svg) |
| chemv | | 6.82e+02 | 3.52e+02 | 7.67e+02 | us | B |![chemv](Graph_builtin/chemv.svg) |
| che | 7.66e-01 | 6.10e-01 | 9.72e-01 | 9.08e-01 | s | A |![che](Graph_builtin/che.svg) |
| chi2 | | 1.47e-02 | 1.40e-02 | 1.40e-02 | s | A |![chi2](Graph_builtin/chi2.svg) |
| clenergy | 2.10e-01 | 2.23e-01 | 7.11e-01 | 7.32e-01 | s | A |![clenergy](Graph_builtin/clenergy.svg) |
| clink | 3.20e+01 | 3.58e+01 | 4.97e+01 | 4.28e+01 | ms | A |![clink](Graph_builtin/clink.svg) |
| cmp | | | | |  | A |![cmp](Graph_builtin/cmp.svg) |
| cm | | | | |  | B |![cm](Graph_builtin/cm.svg) |
| cobahh | 4.61e+04 | 3.44e+04 | 4.96e+04 | 1.40e+04 | us | A |![cobahh](Graph_builtin/cobahh.svg) |
| colorwheel | 1.25e+02 | 1.24e+02 | 1.23e+02 | 1.22e+02 | ms | A |![colorwheel](Graph_builtin/colorwheel.svg) |
| columnarSolver | 7.30e+00 | 7.76e+00 | 1.46e+01 | 7.75e+00 | s | B |![columnarSolver](Graph_builtin/columnarSolver.svg) |
| complex | 1.02e-02 | 1.04e-02 | 1.01e-02 | 1.04e-02 | s | A |![complex](Graph_builtin/complex.svg) |
| compute-score | 6.12e-01 | 6.14e-01 | 7.00e-01 | 1.79e+00 | ms | B |![compute-score](Graph_builtin/compute-score.svg) |
| concat | 2.98e+03 | 2.97e+03 | 3.06e+03 | 3.67e+03 | us | A |![concat](Graph_builtin/concat.svg) |
| contract | 6.65e+00 | 8.82e+00 | 2.64e+01 | 8.29e+00 | s | A |![contract](Graph_builtin/contract.svg) |
| conversion | 1.74e-01 | 1.69e-01 | 9.22e-02 | 7.96e-02 | , | A |![conversion](Graph_builtin/conversion.svg) |
| convolution1D | 1.42e+05 | 1.72e+05 | 3.12e+05 | 1.58e+06 | us | B |![convolution1D](Graph_builtin/convolution1D.svg) |
| convolution3D | 3.16e+01 | 3.48e+01 | 1.22e+03 | 2.76e+01 | us | A |![convolution3D](Graph_builtin/convolution3D.svg) |
| convolutionSeparable | 2.30e-03 | 2.30e-03 | | 1.13e-02 | s | B |![convolutionSeparable](Graph_builtin/convolutionSeparable.svg) |
| cooling | 5.13e+02 | 3.97e+02 | 4.19e+02 | 3.80e+02 | ms | A |![cooling](Graph_builtin/cooling.svg) |
| crc64 | 4.77e+04 | 5.14e+04 | 5.58e+04 | 5.60e+04 | MB/s | A |![crc64](Graph_builtin/crc64.svg) |
| cross | 6.96e+03 | 6.96e+03 | 7.01e+03 | 7.00e+03 | us | A |![cross](Graph_builtin/cross.svg) |
| crs | 6.64e-02 | 7.19e-02 | | | s | B |![crs](Graph_builtin/crs.svg) |
| d2q9-bgk | 5.83e+00 | 5.49e+00 | | 2.45e+01 | us | B |![d2q9-bgk](Graph_builtin/d2q9-bgk.svg) |
| damage | 1.89e-02 | 1.99e-02 | 3.22e-02 | 2.39e-01 | s | B |![damage](Graph_builtin/damage.svg) |
| dct8x8 | 9.43e-03 | 9.43e-03 | | | s | B |![dct8x8](Graph_builtin/dct8x8.svg) |
| ddbp | 3.42e-01 | | | 2.63e-01 | s | B |![ddbp](Graph_builtin/ddbp.svg) |
| debayer | 3.62e-03 | 3.69e-03 | 7.15e-02 | 8.39e-02 | s | B |![debayer](Graph_builtin/debayer.svg) |
| degrid | 1.29e-02 | 1.28e-02 | 1.01e-01 | 3.28e-02 | s | A |![degrid](Graph_builtin/degrid.svg) |
| dense-embedding | 2.37e+04 | 2.36e+04 | 2.35e+04 | 3.88e+04 | us | B |![dense-embedding](Graph_builtin/dense-embedding.svg) |
| depixel | 1.01e+02 | 1.53e+02 | 2.05e+02 | 1.63e+02 | s | A |![depixel](Graph_builtin/depixel.svg) |
| deredundancy | 1.62e+01 | 1.55e+01 | 1.56e+01 | 1.57e+01 | s | B |![deredundancy](Graph_builtin/deredundancy.svg) |
| diamond | 3.67e+01 | 1.61e+01 | | 3.72e+01 | s | A |![diamond](Graph_builtin/diamond.svg) |
| distort | 1.24e-02 | 7.77e-03 | 2.09e-02 | 1.97e-02 | ms | A |![distort](Graph_builtin/distort.svg) |
| divergence | 1.09e+03 | 1.35e+03 | 3.34e+02 | 3.39e+02 | ms | A |![divergence](Graph_builtin/divergence.svg) |
| doh | 6.92e+00 | 7.73e+00 | 1.38e+01 | 2.32e+01 | us | A |![doh](Graph_builtin/doh.svg) |
| dp | 2.11e+01 | | 2.02e+01 | 2.04e+01 | ms | A |![dp](Graph_builtin/dp.svg) |
| dslash | 4.09e-02 | 5.13e-02 | 6.16e-02 | 6.45e-02 | s | A |![dslash](Graph_builtin/dslash.svg) |
| dxtc2 | 3.16e+02 | 3.62e+02 | | | us | B |![dxtc2](Graph_builtin/dxtc2.svg) |
| easyWave | 2.14e+04 | 2.13e+04 | 1.98e+04 | 2.07e+04 | ms | A |![easyWave](Graph_builtin/easyWave.svg) |
| ecdh | 1.19e-01 | 8.15e-02 | 1.50e-01 | 1.64e-01 | s | A |![ecdh](Graph_builtin/ecdh.svg) |
| eigenvalue | 2.04e+01 | 2.43e+01 | | 2.34e+01 | us | A |![eigenvalue](Graph_builtin/eigenvalue.svg) |
| entropy | 4.86e-02 | 5.30e-02 | 5.17e-01 | 7.12e-02 | s | B |![entropy](Graph_builtin/entropy.svg) |
| epistasis | 2.01e-02 | 1.99e-02 | 1.52e+00 | 2.47e-02 | s | A |![epistasis](Graph_builtin/epistasis.svg) |
| expdist | 3.32e-03 | 3.29e-03 | | 8.57e-03 | s | B |![expdist](Graph_builtin/expdist.svg) |
| extend2 | 2.60e+03 | 1.14e+03 | 2.02e+03 | 2.18e+03 | us | A |![extend2](Graph_builtin/extend2.svg) |
| extrema | 2.54e-01 | 3.09e-01 | 9.41e-02 | 1.09e-01 | s | A |![extrema](Graph_builtin/extrema.svg) |
| face | 2.64e+00 | 2.55e+00 | 2.38e+00 | 2.11e-01 | s | A |![face](Graph_builtin/face.svg) |
| fdtd3d | 4.69e-04 | 3.92e-04 | | 4.78e-03 | s | B |![fdtd3d](Graph_builtin/fdtd3d.svg) |
| feynman-kac | 7.87e+00 | 3.94e+00 | 4.57e+00 | 1.98e+00 | s | A |![feynman-kac](Graph_builtin/feynman-kac.svg) |
| fft | 3.55e-04 | 6.13e-04 | | 1.15e-03 | s | B |![fft](Graph_builtin/fft.svg) |
| fhd | | 1.79e+02 | | | s | A |![fhd](Graph_builtin/fhd.svg) |
| filter | 6.17e+00 | | | 7.69e+01 | ms | B |![filter](Graph_builtin/filter.svg) |
| flip | 4.12e+01 | 4.72e+01 | 6.36e+01 | 7.60e+01 | ms | A |![flip](Graph_builtin/flip.svg) |
| floydwarshall | 2.02e-01 | 2.08e-01 | 2.39e-01 | 3.47e-01 | s | A |![floydwarshall](Graph_builtin/floydwarshall.svg) |
| fluidSim | 1.20e-05 | 8.00e-06 | 2.20e-05 | 2.90e-05 | s | A |![fluidSim](Graph_builtin/fluidSim.svg) |
| fpc | 1.19e-02 | 1.23e-02 | | | s | B |![fpc](Graph_builtin/fpc.svg) |
| fpdc | 6.77e-04 | 6.42e-04 | | | s | B |![fpdc](Graph_builtin/fpdc.svg) |
| fresnel | 3.13e-03 | 1.48e-03 | 8.26e-03 | 8.23e-03 | s | A |![fresnel](Graph_builtin/fresnel.svg) |
| frna | 7.78e+01 | 8.04e+01 | | | s | B |![frna](Graph_builtin/frna.svg) |
| fwt | 4.81e-04 | 4.88e-04 | 7.90e-04 | 1.56e-03 | s | B |![fwt](Graph_builtin/fwt.svg) |
| gabor | 7.95e+03 | 8.18e+03 | 1.01e+04 | 1.11e+04 | us | A |![gabor](Graph_builtin/gabor.svg) |
| gamma-correction | 5.21e-04 | 6.09e-04 | 4.97e-04 | 1.79e-03 | s | A |![gamma-correction](Graph_builtin/gamma-correction.svg) |
| ga | 2.36e+00 | 2.82e+00 | 2.85e+00 | 2.68e+00 | s | A |![ga](Graph_builtin/ga.svg) |
| gaussian | 4.61e+05 | 3.92e+05 | 3.77e+05 | 3.82e+05 | us | A |![gaussian](Graph_builtin/gaussian.svg) |
| gc | 6.20e-05 | | | | s | A |![gc](Graph_builtin/gc.svg) |
| gd | 3.99e-01 | 5.12e-01 | 5.31e-01 | 7.87e-01 | s | A |![gd](Graph_builtin/gd.svg) |
| geglu | 7.58e+03 | 1.83e+04 | 1.97e+04 | 2.02e+04 | us | A |![geglu](Graph_builtin/geglu.svg) |
| geodesic | 2.89e+02 | 2.60e+02 | 2.86e+02 | 2.96e+02 | us | A |![geodesic](Graph_builtin/geodesic.svg) |
| glu | 1.86e+04 | 1.52e+04 | 1.90e+04 | 2.06e+04 | us | A |![glu](Graph_builtin/glu.svg) |
| gmm | 3.26e+00 | 3.23e+00 | | 4.23e+00 | s | B |![gmm](Graph_builtin/gmm.svg) |
| goulash | 1.25e+00 | 1.25e+00 | 1.26e+00 | 1.75e+00 | s | A |![goulash](Graph_builtin/goulash.svg) |
| gpp | 2.42e-01 | 1.84e-01 | 1.67e-01 | 1.72e-01 | s | A |![gpp](Graph_builtin/gpp.svg) |
| grep | | | | |  | B |![grep](Graph_builtin/grep.svg) |
| grrt | 1.01e+01 | 1.05e+01 | 9.26e+00 | 8.74e+00 | s | A |![grrt](Graph_builtin/grrt.svg) |
| haccmk | 2.15e-03 | 1.91e-03 | 4.60e-05 | 1.94e-03 | s | A |![haccmk](Graph_builtin/haccmk.svg) |
| hausdorff | 6.49e+00 | 6.76e+00 | 1.01e+01 | 7.26e+00 | ms | A |![hausdorff](Graph_builtin/hausdorff.svg) |
| haversine | 2.09e-04 | 2.35e-04 | | | s | A |![haversine](Graph_builtin/haversine.svg) |
| heartwall | 7.42e+00 | | | | s | A |![heartwall](Graph_builtin/heartwall.svg) |
| heat2d | 1.22e+03 | 1.18e+03 | 7.75e+02 | 6.89e+02 | GFLOPS | A |![heat2d](Graph_builtin/heat2d.svg) |
| heat | 2.29e+01 | 2.22e+01 | 1.53e+01 | 1.61e+01 | s | A |![heat](Graph_builtin/heat.svg) |
| hellinger | 2.87e-03 | 1.62e-03 | 2.72e-02 | 3.79e-03 | s | A |![hellinger](Graph_builtin/hellinger.svg) |
| henry | 1.25e-03 | 1.14e-03 | 4.65e-04 | 4.28e-04 | s | A |![henry](Graph_builtin/henry.svg) |
| hexciton | 1.38e+00 | 3.28e+00 | | 2.13e+00 | s | B |![hexciton](Graph_builtin/hexciton.svg) |
| histogram | 3.32e+02 | 3.02e+02 | | 7.93e+02 | us  | B |![histogram](Graph_builtin/histogram.svg) |
| hmm | 1.39e-01 | 1.44e-01 | 2.83e-01 | 1.90e-01 | s | A |![hmm](Graph_builtin/hmm.svg) |
| hogbom | 2.60e-01 | 1.40e-01 | 1.60e-01 | 3.10e-01 | ms | B |![hogbom](Graph_builtin/hogbom.svg) |
| hotspot3D | 7.44e+00 | 7.35e+00 | 1.72e+01 | 1.66e+01 | us | A |![hotspot3D](Graph_builtin/hotspot3D.svg) |
| hwt1d | 6.11e-04 | 6.20e-04 | 6.19e-04 | 7.06e-04 | s | B |![hwt1d](Graph_builtin/hwt1d.svg) |
| hybridsort | 2.22e+02 | 5.79e+01 | | 2.25e+02 | ms | B |![hybridsort](Graph_builtin/hybridsort.svg) |
| hypterm | 2.30e+00 | 2.62e+00 | 2.39e+00 | 2.42e+00 | ms | A |![hypterm](Graph_builtin/hypterm.svg) |
| idivide | 1.08e-02 | 4.31e-03 | 7.45e-03 | 1.82e-02 | s | A |![idivide](Graph_builtin/idivide.svg) |
| interleave | 1.47e-02 | 1.47e-02 | 1.40e-02 | 7.96e-03 | s | A |![interleave](Graph_builtin/interleave.svg) |
| interval | 4.33e+01 | 3.88e+01 | 1.02e+02 | 7.41e+01 | us | A |![interval](Graph_builtin/interval.svg) |
| inversek2j | 2.71e+00 | 3.12e+00 | 8.41e+02 | 1.11e+03 | us | A |![inversek2j](Graph_builtin/inversek2j.svg) |
| ising | 3.09e-01 | 3.50e-01 | 3.89e-01 | 4.86e-01 | s | A |![ising](Graph_builtin/ising.svg) |
| iso2dfd | 2.40e+01 | 2.45e+01 | 2.24e+01 | 3.20e+01 | ms | A |![iso2dfd](Graph_builtin/iso2dfd.svg) |
| jacobi | 2.74e+00 | 2.73e+00 | 3.57e+00 | 3.44e+00 | s | A |![jacobi](Graph_builtin/jacobi.svg) |
| jenkins-hash | 3.10e-03 | 3.56e-03 | 2.30e-01 | 2.28e-01 | s | A |![jenkins-hash](Graph_builtin/jenkins-hash.svg) |
| kalman | 3.22e-01 | | | | s | A |![kalman](Graph_builtin/kalman.svg) |
| keccaktreehash | 2.91e+07 | 3.08e+07 | 7.36e+05 | 2.86e+07 | kB/s | A |![keccaktreehash](Graph_builtin/keccaktreehash.svg) |
| keogh | 4.56e-03 | 5.04e-03 | 5.09e-02 | 1.52e-01 | s | A |![keogh](Graph_builtin/keogh.svg) |
| kernelLaunch | 1.83e+00 | 2.02e+00 | 1.13e+02 | 1.24e+01 | us | B |![kernelLaunch](Graph_builtin/kernelLaunch.svg) |
| kmeans | 2.79e+04 | 2.86e+04 | 2.66e+04 | 2.21e+04 | ms | A |![kmeans](Graph_builtin/kmeans.svg) |
| knn | 6.45e-01 | 4.71e-01 | 5.76e-01 | 7.78e-01 | s | B |![knn](Graph_builtin/knn.svg) |
| lanczos | 3.14e-01 | 3.11e-01 | | 4.95e-01 | s | B |![lanczos](Graph_builtin/lanczos.svg) |
| langevin | 6.30e-03 | 6.19e-03 | 6.31e-03 | 9.40e-03 | s | A |![langevin](Graph_builtin/langevin.svg) |
| langford | 1.24e-01 | 1.21e-01 | | | s | B |![langford](Graph_builtin/langford.svg) |
| laplace3d | 1.01e-02 | 1.03e-02 | 1.80e-02 | 1.95e-02 | s | A |![laplace3d](Graph_builtin/laplace3d.svg) |
| laplace | 1.16e+00 | 8.32e+00 | 4.92e+00 | 4.61e+00 | s | A |![laplace](Graph_builtin/laplace.svg) |
| lavaMD | 4.99e+00 | 5.82e+00 | | 1.82e+01 | s | B |![lavaMD](Graph_builtin/lavaMD.svg) |
| layout | 1.33e+02 | 1.41e+02 | 8.98e+01 | 8.85e+01 | us | A |![layout](Graph_builtin/layout.svg) |
| lci | 1.53e+01 | 7.50e+00 | 4.32e+00 | 4.28e+00 | s | A |![lci](Graph_builtin/lci.svg) |
| lda | 6.31e-03 | 6.53e-03 | | | s | B |![lda](Graph_builtin/lda.svg) |
| ldpc | 1.19e+00 | 1.27e+00 | | 5.92e+00 | s | B |![ldpc](Graph_builtin/ldpc.svg) |
| lebesgue | 2.66e+00 | 1.79e-01 | 2.43e+00 | 2.47e+00 | s | A |![lebesgue](Graph_builtin/lebesgue.svg) |
| leukocyte | 5.89e-01 | | | 8.64e-01 | s | A |![leukocyte](Graph_builtin/leukocyte.svg) |
| libor | 2.99e-03 | 3.20e-03 | 1.07e-01 | 3.30e-03 | s | A |![libor](Graph_builtin/libor.svg) |
| lid-driven-cavity | 7.56e+00 | 8.23e+00 | | 1.98e+01 | s | B |![lid-driven-cavity](Graph_builtin/lid-driven-cavity.svg) |
| lif | 1.37e+04 | 1.27e+04 | 1.70e+04 | 1.09e+04 | us | A |![lif](Graph_builtin/lif.svg) |
| linearprobing | 1.76e-03 | 1.77e-03 | | | s | A |![linearprobing](Graph_builtin/linearprobing.svg) |
| log2 | 2.02e+03 | 1.84e+03 | 2.41e+03 | 2.68e+03 | us | A |![log2](Graph_builtin/log2.svg) |
| lombscargle | 3.05e+02 | 1.80e+02 | 1.95e+02 | 2.17e+02 | us | A |![lombscargle](Graph_builtin/lombscargle.svg) |
| loopback | 3.26e-02 | 2.90e-02 | 2.98e-02 | 5.74e-02 | s | B |![loopback](Graph_builtin/loopback.svg) |
| lrn | 4.28e-01 | 6.49e-01 | 5.27e-01 | 6.42e-01 | s | A |![lrn](Graph_builtin/lrn.svg) |
| lr | 7.56e+00 | 7.50e+00 | 2.55e+01 | 5.51e+01 | us | B |![lr](Graph_builtin/lr.svg) |
| lsqt | 1.30e+01 | 1.29e+01 | 1.77e+01 | 1.74e+01 | s | A |![lsqt](Graph_builtin/lsqt.svg) |
| lud | 5.07e+00 | 5.72e+00 | 7.02e+00 | 4.36e+01 | s | B |![lud](Graph_builtin/lud.svg) |
| lulesh | 4.59e+00 | 5.66e+00 | 8.51e+00 | 5.18e+00 | s | A |![lulesh](Graph_builtin/lulesh.svg) |
| mallocFree | 2.53e+06 | 3.10e+03 | 2.34e+06 | 5.09e+02 | us | A |![mallocFree](Graph_builtin/mallocFree.svg) |
| mandelbrot | 5.78e-02 | 6.43e-02 | 5.90e-02 | 6.16e-02 | ms | A |![mandelbrot](Graph_builtin/mandelbrot.svg) |
| mask | 2.79e+04 | 2.76e+04 | 2.74e+04 | 2.78e+04 | us | A |![mask](Graph_builtin/mask.svg) |
| match | 2.38e+02 | 2.51e+02 | | 3.57e+11 | ms | B |![match](Graph_builtin/match.svg) |
| matern | 1.81e+04 | 1.92e+04 | 6.98e+03 | 1.45e+05 | us | B |![matern](Graph_builtin/matern.svg) |
| matrix-rotate | 7.03e-02 | 7.02e-02 | 7.13e-02 | 6.75e-02 | s | A |![matrix-rotate](Graph_builtin/matrix-rotate.svg) |
| maxFlops | 4.61e+00 | 4.62e+00 | 4.80e+00 | 4.59e+00 | s | A |![maxFlops](Graph_builtin/maxFlops.svg) |
| maxpool3d | 3.94e-03 | 4.14e-03 | 5.74e-03 | 6.76e-03 | s | A |![maxpool3d](Graph_builtin/maxpool3d.svg) |
| mcmd | 1.23e+01 | 9.01e+00 | 1.07e+01 | 1.10e+01 | s | A |![mcmd](Graph_builtin/mcmd.svg) |
| mcpr | 2.93e-02 | 5.56e-03 | 1.28e-01 | 2.83e-02 | s | B |![mcpr](Graph_builtin/mcpr.svg) |
| md5hash | 1.44e+04 | 1.40e+04 | 1.44e+04 | 1.45e+04 | ms | A |![md5hash](Graph_builtin/md5hash.svg) |
| mdh | 1.34e+00 | 5.90e-01 | 3.17e-01 | 1.23e+00 | s | A |![mdh](Graph_builtin/mdh.svg) |
| md | 6.56e-05 | 6.63e-05 | 7.48e-05 | 7.23e-05 | s | A |![md](Graph_builtin/md.svg) |
| meanshift | 4.41e-01 | 8.78e-01 | 1.37e+00 | 9.38e-01 | ms | B |![meanshift](Graph_builtin/meanshift.svg) |
| medianfilter | 7.60e-05 | 7.60e-05 | 8.50e-05 | 2.40e-04 | s | B |![medianfilter](Graph_builtin/medianfilter.svg) |
| memcpy | 1.82e+02 | 2.01e+02 | 2.48e+02 | 2.49e+02 | us | A |![memcpy](Graph_builtin/memcpy.svg) |
| memtest | 3.25e-02 | 3.46e-02 | 4.32e-02 | 4.31e-02 | s | A |![memtest](Graph_builtin/memtest.svg) |
| merge | 6.73e+04 | 6.65e+04 | | 1.62e+05 | us | B |![merge](Graph_builtin/merge.svg) |
| metropolis | 9.72e+00 | 1.11e+01 | | 6.14e+01 | s | B |![metropolis](Graph_builtin/metropolis.svg) |
| michalewicz | 1.09e+05 | 1.05e+06 | 1.45e+05 | 1.90e+05 | us | A |![michalewicz](Graph_builtin/michalewicz.svg) |
| minibude | 7.66e+02 | 9.10e+02 | 8.43e+02 | 1.25e+03 | ms | B |![minibude](Graph_builtin/minibude.svg) |
| miniFE | | | | |  | A |![miniFE](Graph_builtin/miniFE.svg) |
| minimap2 | 4.20e-02 | 5.79e-03 | 1.13e-02 | 1.09e-01 | s | A |![minimap2](Graph_builtin/minimap2.svg) |
| minisweep | 5.03e+01 | 5.41e+01 | 3.37e+01 | 3.34e+01 | s | A |![minisweep](Graph_builtin/minisweep.svg) |
| miniWeather | 6.96e-01 | 3.90e+00 | 2.99e+01 | 2.99e+01 | s | A |![miniWeather](Graph_builtin/miniWeather.svg) |
| minkowski | 2.31e-03 | 2.65e-03 | 2.22e-03 | 2.20e-03 | s | A |![minkowski](Graph_builtin/minkowski.svg) |
| mis | 2.17e-04 | 1.08e-04 | 0.00e+00 | 0.00e+00 | s | B |![mis](Graph_builtin/mis.svg) |
| mixbench | 6.06e-01 | 5.55e-01 | 6.31e-01 | 6.67e-01 | s | B |![mixbench](Graph_builtin/mixbench.svg) |
| morphology | 2.23e-01 | 2.24e-01 | | | s | B |![morphology](Graph_builtin/morphology.svg) |
| mrc | 1.38e+04 | 1.38e+04 | 1.39e+04 | 1.41e+04 | us | A |![mrc](Graph_builtin/mrc.svg) |
| mriQ | 2.60e+00 | 3.24e+00 | 3.29e+00 | 3.30e+00 | s | A |![mriQ](Graph_builtin/mriQ.svg) |
| mr | 2.25e+00 | 1.12e+00 | | 2.30e+00 | ms | A |![mr](Graph_builtin/mr.svg) |
| mt | 2.95e-03 | 1.88e-03 | 1.67e-03 | 1.68e-03 | s | A |![mt](Graph_builtin/mt.svg) |
| multimaterial | 1.14e+02 | 2.49e+00 | 3.14e+00 | 4.37e+00 | ms | A |![multimaterial](Graph_builtin/multimaterial.svg) |
| murmurhash3 | 3.97e-03 | 3.88e-03 | 3.99e-03 | 3.91e-03 | s | A |![murmurhash3](Graph_builtin/murmurhash3.svg) |
| myocyte | 1.49e+01 | | 3.22e+00 | 4.00e+00 | s | A |![myocyte](Graph_builtin/myocyte.svg) |
| nbody | 3.54e-02 | 3.33e-02 | 2.31e+00 | 2.40e+00 |  | A |![nbody](Graph_builtin/nbody.svg) |
| ne | 9.06e-04 | 1.18e-03 | 1.33e-03 | 1.61e-03 | s | A |![ne](Graph_builtin/ne.svg) |
| nlll | 8.41e+01 | 9.40e+01 | 1.33e+04 | 1.00e+02 | us | B |![nlll](Graph_builtin/nlll.svg) |
| nms | 7.40e-05 | 1.36e-04 | | 2.18e-04 | s | B |![nms](Graph_builtin/nms.svg) |
| nn | 2.43e+00 | 2.58e+00 | 7.74e+00 | 8.28e+00 | us | A |![nn](Graph_builtin/nn.svg) |
| norm2 | 2.05e+03 | | 2.91e+03 | 1.59e+04 | us | A |![norm2](Graph_builtin/norm2.svg) |
| nqueen | 3.85e-02 | 4.50e-02 | 2.14e-01 | 4.39e-02 | s | A |![nqueen](Graph_builtin/nqueen.svg) |
| ntt | 3.62e+01 | 4.14e+01 | | 4.19e+01 | us | B |![ntt](Graph_builtin/ntt.svg) |
| nw | 4.97e-02 | 4.95e-02 | | 1.23e-01 | s | B |![nw](Graph_builtin/nw.svg) |
| openmp | 2.55e+00 | 9.59e-02 | 2.44e+00 | 2.29e-01 | s | B |![openmp](Graph_builtin/openmp.svg) |
| overlay | 5.51e-01 | 5.91e-01 | 3.45e+00 | 4.89e+00 | s | A |![overlay](Graph_builtin/overlay.svg) |
| p4 | 8.25e+01 | 8.53e+01 | | 1.39e+03 | us | B |![p4](Graph_builtin/p4.svg) |
| page-rank | 3.93e+00 | 3.83e+00 | 4.08e+00 | 4.36e+00 | s | A |![page-rank](Graph_builtin/page-rank.svg) |
| particle-diffusion | 2.20e-03 | 1.69e-03 | 2.27e-03 | 2.27e-03 | s | A |![particle-diffusion](Graph_builtin/particle-diffusion.svg) |
| particlefilter | 2.22e+02 | 1.98e+02 | | 2.20e+02 | s | B |![particlefilter](Graph_builtin/particlefilter.svg) |
| particles | 1.07e+00 | 9.65e-01 | | | s | B |![particles](Graph_builtin/particles.svg) |
| pathfinder | 3.88e+01 | 3.88e+01 | 4.35e+01 | 7.99e+02 | s | B |![pathfinder](Graph_builtin/pathfinder.svg) |
| permutate | 2.05e+00 | 2.42e+00 | | 2.26e+00 | s | B |![permutate](Graph_builtin/permutate.svg) |
| permute | 2.00e+01 | 2.22e+01 | 2.23e+01 | 2.28e+01 | ms | A |![permute](Graph_builtin/permute.svg) |
| perplexity | 1.64e+00 | 2.24e+00 | 1.53e+00 | 1.69e+00 | s | A |![perplexity](Graph_builtin/perplexity.svg) |
| phmm | 1.16e+03 | 1.06e+03 | 4.63e+03 | 5.23e+03 | ms | B |![phmm](Graph_builtin/phmm.svg) |
| pnpoly | 4.59e-02 | 5.18e-02 | | 7.17e-01 | s | A |![pnpoly](Graph_builtin/pnpoly.svg) |
| pns | 5.44e+00 | 5.48e+00 | | | s | B |![pns](Graph_builtin/pns.svg) |
| pointwise | 9.21e-02 | 1.13e-01 | 1.84e-01 | 1.66e-01 | s | A |![pointwise](Graph_builtin/pointwise.svg) |
| pool | 8.88e-03 | 9.02e-03 | 1.83e-02 | 1.57e-02 | s | A |![pool](Graph_builtin/pool.svg) |
| popcount | 1.71e+05 | 8.28e+04 | 2.00e+05 | 7.06e+04 | us | A |![popcount](Graph_builtin/popcount.svg) |
| present | 3.81e+01 | 4.82e+01 | 3.69e+01 | 9.36e+01 | us | A |![present](Graph_builtin/present.svg) |
| prna | | | | |  | B |![prna](Graph_builtin/prna.svg) |
| projectile | 1.13e-04 | 1.13e-04 | 1.18e-04 | 1.19e-04 | s | A |![projectile](Graph_builtin/projectile.svg) |
| pso | 2.55e+01 | 2.42e+01 | 2.48e+01 | 2.19e+01 | us | A |![pso](Graph_builtin/pso.svg) |
| qrg | 4.48e+01 | 3.69e+01 | 5.94e+01 | 7.56e+01 | us | A |![qrg](Graph_builtin/qrg.svg) |
| qtclustering | 3.31e+01 | 3.20e+01 | | | s | B |![qtclustering](Graph_builtin/qtclustering.svg) |
| quantBnB | 6.09e+03 | 6.66e+03 | 2.74e+03 | 3.26e+03 | us | A |![quantBnB](Graph_builtin/quantBnB.svg) |
| quicksort | 6.92e+03 | 6.93e+03 | | 2.53e+03 | ms | B |![quicksort](Graph_builtin/quicksort.svg) |
| radixsort | 5.06e-04 | 5.17e-04 | | 5.71e-03 | s | B |![radixsort](Graph_builtin/radixsort.svg) |
| rainflow | 1.71e+05 | 1.49e+05 | 1.69e+05 | 1.27e+05 | us | A |![rainflow](Graph_builtin/rainflow.svg) |
| randomAccess | 1.58e-01 | 1.58e-01 | | 1.92e+00 | s | A |![randomAccess](Graph_builtin/randomAccess.svg) |
| reaction | 1.63e+00 | 1.64e+00 | | 6.75e+00 | s | B |![reaction](Graph_builtin/reaction.svg) |
| recursiveGaussian | 1.32e-03 | 1.49e-03 | | 1.67e-03 | s | B |![recursiveGaussian](Graph_builtin/recursiveGaussian.svg) |
| resize | 6.98e+02 | 7.58e+02 | 7.14e+02 | 8.35e+02 | us | A |![resize](Graph_builtin/resize.svg) |
| reverse | 9.29e-01 | 1.21e+00 | 4.23e+00 | 5.49e+00 | s | B |![reverse](Graph_builtin/reverse.svg) |
| rfs | 8.78e-03 | 8.78e-03 | 8.79e-03 | 8.79e-03 | s | A |![rfs](Graph_builtin/rfs.svg) |
| rng-wallace | 1.25e+02 | 1.18e+02 | | 1.12e+03 | us | B |![rng-wallace](Graph_builtin/rng-wallace.svg) |
| rodrigues | 3.10e+04 | 3.50e+04 | 3.49e+04 | 2.01e+04 | us | A |![rodrigues](Graph_builtin/rodrigues.svg) |
| romberg | 2.85e-04 | 2.91e-04 | 3.50e-04 | 3.26e-04 | s | B |![romberg](Graph_builtin/romberg.svg) |
| rsbench | 2.94e+00 | 3.35e+00 | 9.65e+00 | 2.91e+00 | s | A |![rsbench](Graph_builtin/rsbench.svg) |
| rsc | 1.55e+02 | 1.43e+02 | 2.17e+02 | 2.28e+02 | ms | B |![rsc](Graph_builtin/rsc.svg) |
| rtm8 | 1.02e-03 | 1.06e-03 | 8.59e-04 | 8.90e-04 | s | A |![rtm8](Graph_builtin/rtm8.svg) |
| rushlarsen | 2.29e-01 | 3.24e-01 | 2.32e-01 | 2.45e-01 | s | A |![rushlarsen](Graph_builtin/rushlarsen.svg) |
| s3d | 2.86e+00 | 4.81e-01 | | 2.84e+00 | s | A |![s3d](Graph_builtin/s3d.svg) |
| s8n | 1.16e+01 | 1.33e+01 | 3.26e+01 | 5.00e+01 | us | A |![s8n](Graph_builtin/s8n.svg) |
| sad | 4.00e+01 | 4.30e+01 | 4.84e+02 | 8.18e+04 | ms | A |![sad](Graph_builtin/sad.svg) |
| sampling | 2.39e+02 | 2.82e+02 | 3.71e+02 | 5.63e+02 | us | B |![sampling](Graph_builtin/sampling.svg) |
| scan2 | 3.55e+02 | 4.18e+02 | | 4.32e+03 | us | B |![scan2](Graph_builtin/scan2.svg) |
| scan | 1.75e+04 | 1.76e+04 | | 2.05e+05 | us | B |![scan](Graph_builtin/scan.svg) |
| scatterAdd | 3.91e+03 | 4.84e+03 | 1.60e+05 | 1.61e+05 | us | A |![scatterAdd](Graph_builtin/scatterAdd.svg) |
| scel | 2.92e+03 | 2.92e+03 | 4.27e+04 | 6.24e+03 | us | A |![scel](Graph_builtin/scel.svg) |
| secp256k1 | 2.91e-03 | 2.77e-03 | 2.92e-03 | 2.87e-03 | s | A |![secp256k1](Graph_builtin/secp256k1.svg) |
| sheath | 3.18e+00 | 3.44e+00 | 3.38e+00 | 3.40e+00 | ms | A |![sheath](Graph_builtin/sheath.svg) |
| shmembench | 2.70e+00 | 6.74e+00 | | | ms | B |![shmembench](Graph_builtin/shmembench.svg) |
| simplemoc | 1.76e+02 | 3.42e+02 | 2.35e+02 | 2.35e+02 | s | B |![simplemoc](Graph_builtin/simplemoc.svg) |
| simpleSpmv | 4.46e+01 | 3.58e+01 | 2.28e+01 | 4.70e+01 | ms | A |![simpleSpmv](Graph_builtin/simpleSpmv.svg) |
| slu | | | | | ms | B |![slu](Graph_builtin/slu.svg) |
| snake | 4.39e+03 | 4.85e+03 | 7.15e+03 | 7.15e+03 | us | A |![snake](Graph_builtin/snake.svg) |
| sobel | 4.80e+00 | 4.68e+00 | 1.01e+01 | 1.11e+01 | us | A |![sobel](Graph_builtin/sobel.svg) |
| sobol | 1.22e-03 | 1.28e-03 | | 3.35e-03 | s | B |![sobol](Graph_builtin/sobol.svg) |
| softmax-online | 1.44e+01 | 1.45e+01 | | 2.92e+01 | ms | B |![softmax-online](Graph_builtin/softmax-online.svg) |
| softmax | 1.12e+01 | 1.12e+01 | 3.43e+03 | 5.39e+01 | ms | A |![softmax](Graph_builtin/softmax.svg) |
| sort | 2.13e-02 | | | 3.44e-01 | s | B |![sort](Graph_builtin/sort.svg) |
| sosfil | 2.02e-02 | 2.29e-02 | 3.16e-02 | 1.05e-01 | s | B |![sosfil](Graph_builtin/sosfil.svg) |
| sph | 5.81e+00 | 5.85e+00 | 4.01e+01 | 5.91e+00 | ms | A |![sph](Graph_builtin/sph.svg) |
| split | 1.27e+04 | 1.28e+04 | | 1.49e+05 | us | B |![split](Graph_builtin/split.svg) |
| spm | 7.59e+01 | 6.59e+01 | 6.34e+01 | 7.27e+01 | ms | A |![spm](Graph_builtin/spm.svg) |
| sptrsv | 1.18e+02 | 1.37e+02 | | 1.17e+02 | us | B |![sptrsv](Graph_builtin/sptrsv.svg) |
| srad | 3.67e+01 | 3.74e+01 | | | s | B |![srad](Graph_builtin/srad.svg) |
| ss | 2.49e+02 | 2.36e+02 | 3.80e+02 | 7.90e+02 | us | B |![ss](Graph_builtin/ss.svg) |
| stddev | 3.77e-03 | 5.04e-03 | 5.56e-03 | 6.51e-02 | s | B |![stddev](Graph_builtin/stddev.svg) |
| stencil1d | 2.25e-03 | 2.25e-03 | 2.57e-01 | 3.50e-01 | s | A |![stencil1d](Graph_builtin/stencil1d.svg) |
| stencil3d | 6.27e-03 | 6.31e-03 | 6.40e-03 | 1.74e-02 | s | B |![stencil3d](Graph_builtin/stencil3d.svg) |
| streamcluster | 5.84e+00 | 5.00e+00 | 9.61e+00 | 7.69e+00 | s | B |![streamcluster](Graph_builtin/streamcluster.svg) |
| su3 | 1.92e+00 | 2.32e+00 | 2.34e+00 | 2.35e+00 | s | A |![su3](Graph_builtin/su3.svg) |
| surfel | 2.55e+02 | 2.59e+02 | 3.07e+02 | 3.59e+02 | ms | A |![surfel](Graph_builtin/surfel.svg) |
| svd3x3 | 4.60e+01 | 4.88e+01 | 6.86e+01 | 7.09e+01 | us | A |![svd3x3](Graph_builtin/svd3x3.svg) |
| sw4ck | 8.78e+00 | 1.02e+01 | 2.38e+02 | 2.36e+02 | ms | A |![sw4ck](Graph_builtin/sw4ck.svg) |
| swish | 4.13e+03 | 4.34e+03 | 4.59e+03 | 6.21e+03 | us | A |![swish](Graph_builtin/swish.svg) |
| tensorT | 1.41e+00 | 1.28e+00 | 1.36e+00 | 4.44e+00 | ms | B |![tensorT](Graph_builtin/tensorT.svg) |
| testSNAP | 1.11e+01 | 1.16e+01 | 6.34e+02 | 1.48e+01 | ms | A |![testSNAP](Graph_builtin/testSNAP.svg) |
| thomas | 1.44e+04 | 1.45e+04 | 1.26e+04 | 1.24e+04 | ms | A |![thomas](Graph_builtin/thomas.svg) |
| threadfence | 2.30e+02 | 2.34e+02 | 1.50e+02 | 3.27e+02 | ms | B |![threadfence](Graph_builtin/threadfence.svg) |
| tissue | 7.68e-03 | 8.00e-03 | 1.07e-02 | 8.15e-03 | s | A |![tissue](Graph_builtin/tissue.svg) |
| tonemapping | 1.69e+01 | 1.62e+01 | 1.91e+01 | 1.82e+01 | us | A |![tonemapping](Graph_builtin/tonemapping.svg) |
| tqs | 1.31e+02 | 8.24e+01 | | 1.42e+02 | ms | B |![tqs](Graph_builtin/tqs.svg) |
| triad | 1.86e+03 | 1.25e+03 | 5.74e+02 | 6.10e+02 | GFLOPS | A |![triad](Graph_builtin/triad.svg) |
| tridiagonal | 3.83e-02 | 3.82e-02 | | 1.55e-01 | s | B |![tridiagonal](Graph_builtin/tridiagonal.svg) |
| tsa | 1.71e+04 | 1.76e+04 | | 1.70e+05 | us | B |![tsa](Graph_builtin/tsa.svg) |
| tsp | 2.52e-02 | 5.34e-02 | | | s | B |![tsp](Graph_builtin/tsp.svg) |
| urng | 4.87e+00 | 4.99e+00 | 1.17e+01 | 2.37e+01 | us | B |![urng](Graph_builtin/urng.svg) |
| vanGenuchten | 7.29e-03 | 7.25e-03 | 7.44e-03 | 7.55e-03 | s | A |![vanGenuchten](Graph_builtin/vanGenuchten.svg) |
| vmc | 2.52e-03 | 1.89e-03 | | | s | B |![vmc](Graph_builtin/vmc.svg) |
| vol2col | 2.53e+03 | 2.55e+03 | 2.02e+03 | 2.73e+03 | us | A |![vol2col](Graph_builtin/vol2col.svg) |
| wedford | 3.87e+00 | 3.99e+00 | | 1.65e+01 | ms | B |![wedford](Graph_builtin/wedford.svg) |
| winograd | 7.85e-01 | 2.93e-01 | 3.18e+00 | 3.14e+00 | s | A |![winograd](Graph_builtin/winograd.svg) |
| wlcpow | 2.03e+03 | 2.09e+03 | | 6.74e+03 | us | B |![wlcpow](Graph_builtin/wlcpow.svg) |
| wordcount | 1.74e-03 | 3.75e-03 | 1.51e-03 | 2.73e-03 | s | A |![wordcount](Graph_builtin/wordcount.svg) |
| wsm5 | 2.98e+02 | 2.32e+02 | 3.10e+02 | 3.28e+02 | ms | B |![wsm5](Graph_builtin/wsm5.svg) |
| wyllie | 2.93e+01 | 2.96e+01 | 3.08e+01 | | ms | B |![wyllie](Graph_builtin/wyllie.svg) |
| xlqc | 2.01e+06 | | | 3.03e+05 | us | B |![xlqc](Graph_builtin/xlqc.svg) |
| xsbench | 4.45e+01 | 3.64e+01 | 3.59e+01 | 2.39e+00 | s | A |![xsbench](Graph_builtin/xsbench.svg) |
| zeropoint | 1.39e+04 | 1.51e+04 | 1.72e+04 | 2.01e+04 | us | A |![zeropoint](Graph_builtin/zeropoint.svg) |
| zmddft | 1.16e+00 | 1.40e+00 | | 2.01e+00 | ms | B |![zmddft](Graph_builtin/zmddft.svg) |

※ 分類: OpenMP コードが (omp_get_wtime 以外の) omp_get_* を (A) 含まない (B) 含む

※ メモリ: cuda 版での値
