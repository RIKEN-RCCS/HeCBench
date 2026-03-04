# ベンチマークが標準出力に出力する値

| 名称 | cuda | sycl | acc | omp_nvc  | 単位 | 分類 | plot |
|  --  |  --  | --   |  -- |   --     | --   | --   | -- |
| accuracy | 6.02e+02 | 6.07e+02 | 6.89e+02 | 8.27e+02 | us | A |![accuracy](SVGs_builtin/accuracy.svg) |
| ace | 2.35e+03 | 2.66e+03 | 2.34e+03 | 2.52e+03 | ms | A |![ace](SVGs_builtin/ace.svg) |
| adam | 6.02e-02 | 6.22e-02 | 7.60e-02 | 1.06e-01 | ms | A |![adam](SVGs_builtin/adam.svg) |
| adamw | 5.17e-02 | 5.44e-02 | 9.98e-02 | 6.95e-01 | ms | A |![adamw](SVGs_builtin/adamw.svg) |
| adjacent | 2.40e+03 | 2.43e+03 | 3.04e+03 | 6.03e+03 | us | A |![adjacent](SVGs_builtin/adjacent.svg) |
| adv | 3.82e+06 | 5.35e+06 | | 6.92e+06 | ns | B |![adv](SVGs_builtin/adv.svg) |
| aes | 3.14e-04 | 2.98e-04 | | 4.23e-04 | s | B |![aes](SVGs_builtin/aes.svg) |
| affine | 3.13e-06 | 2.92e-06 | 1.02e-05 | 9.82e-06 | s | A |![affine](SVGs_builtin/affine.svg) |
| aidw | 7.70e-03 | 4.73e-02 | 1.95e-02 | 9.10e-03 | s | B |![aidw](SVGs_builtin/aidw.svg) |
| aligned-types | 6.29e-01 | 7.27e-01 | 1.47e+00 | 1.69e+00 | ms | A |![aligned-types](SVGs_builtin/aligned-types.svg) |
| all-pairs-distance | 2.12e+02 | 2.54e+02 | 2.16e+02 | 1.87e+03 | us | B |![all-pairs-distance](SVGs_builtin/all-pairs-distance.svg) |
| amgmk | 2.45e+00 | 2.39e+00 | 2.37e+00 | 2.38e+00 | s | B |![amgmk](SVGs_builtin/amgmk.svg) |
| ans | 5.14e+00 | 5.36e+00 | | | s | A |![ans](SVGs_builtin/ans.svg) |
| aobench | 8.90e+01 | 1.02e+02 | 7.88e+01 | 1.06e+02 | us | A |![aobench](SVGs_builtin/aobench.svg) |
| aop | 4.00e+00 | 4.01e+00 | | | ms | B |![aop](SVGs_builtin/aop.svg) |
| asmooth | 5.16e-03 | 5.67e-03 | 5.65e-03 | 5.80e-03 | s | A |![asmooth](SVGs_builtin/asmooth.svg) |
| assert | 2.43e+00 | 3.25e-02 | | 5.02e+00 | ns | B |![assert](SVGs_builtin/assert.svg) |
| asta | 1.53e-02 | | | 7.29e-02 | s | B |![asta](SVGs_builtin/asta.svg) |
| atan2 | 1.38e+02 | 1.41e+02 | 1.70e+02 | 1.89e+02 | us | A |![atan2](SVGs_builtin/atan2.svg) |
| atomicCost | 1.09e+04 | 1.08e+04 | 1.14e+04 | 1.42e+04 | us | A |![atomicCost](SVGs_builtin/atomicCost.svg) |
| atomicIntrinsics | 4.23e+05 | | 3.11e+04 | 3.38e+04 | us | A |![atomicIntrinsics](SVGs_builtin/atomicIntrinsics.svg) |
| atomicPerf | 6.61e+03 | 6.61e+03 | 6.69e+03 | 7.20e+03 | us | B |![atomicPerf](SVGs_builtin/atomicPerf.svg) |
| atomicReduction | 5.12e+04 | 4.71e+04 | 4.56e+04 | 4.59e+04 | GB/s | A |![atomicReduction](SVGs_builtin/atomicReduction.svg) |
| attention | 4.31e-01 | 4.31e-01 | 4.32e-01 | 9.30e-01 | ms | A |![attention](SVGs_builtin/attention.svg) |
| axhelm | 2.22e+07 | 2.17e+07 | | 2.18e+07 | ns | B |![axhelm](SVGs_builtin/axhelm.svg) |
| babelstream | 6.69e+06 | 6.66e+06 | 6.60e+06 | 6.04e+06 | s | A |![babelstream](SVGs_builtin/babelstream.svg) |
| background-subtract | 1.03e+02 | 1.06e+02 | 1.14e+02 | 1.63e+02 | us | A |![background-subtract](SVGs_builtin/background-subtract.svg) |
| backprop | 2.45e+00 | 2.37e+00 | 2.36e+00 | 2.39e+00 | s | B |![backprop](SVGs_builtin/backprop.svg) |
| bezier-surface | 8.80e+01 | 1.13e+02 | 3.19e+03 | 9.70e+01 | ms | A |![bezier-surface](SVGs_builtin/bezier-surface.svg) |
| bfs | 3.84e+02 | 5.31e+02 | 4.35e+02 | 1.27e+03 | us | A |![bfs](SVGs_builtin/bfs.svg) |
| bilateral | 4.86e+00 | 4.90e+00 | 2.03e+02 | 5.22e+00 | ms | A |![bilateral](SVGs_builtin/bilateral.svg) |
| binomial | 2.99e+03 | 2.78e+03 | | | ms | B |![binomial](SVGs_builtin/binomial.svg) |
| bitonic-sort | 3.29e+01 | 3.46e+01 | 3.74e+01 | 5.55e+01 | ms | A |![bitonic-sort](SVGs_builtin/bitonic-sort.svg) |
| black-scholes | 2.08e+03 | 2.24e+03 | 1.85e+03 | 2.25e+03 | ms | A |![black-scholes](SVGs_builtin/black-scholes.svg) |
| blas-gemm | 5.06e+03 | | | | us | A |![blas-gemm](SVGs_builtin/blas-gemm.svg) |
| bn | 1.62e-01 | 2.22e-01 | | | s | B |![bn](SVGs_builtin/bn.svg) |
| bonds | 3.76e+01 | 2.21e+01 | 5.30e+01 | 5.05e+01 | ms | A |![bonds](SVGs_builtin/bonds.svg) |
| boxfilter | 6.43e+02 | 5.97e+02 | 3.94e+02 | 7.06e+02 | us | B |![boxfilter](SVGs_builtin/boxfilter.svg) |
| bsearch | 3.12e+00 | 2.30e+00 | 2.73e-01 | 2.27e-01 | s | B |![bsearch](SVGs_builtin/bsearch.svg) |
| bspline-vgh | 1.32e-01 | 1.59e-01 | 1.41e-01 | 1.43e-01 | s | A |![bspline-vgh](SVGs_builtin/bspline-vgh.svg) |
| b+tree | 2.06e+02 | 3.34e+02 | 1.35e+03 | 1.39e+03 | us | A |![b+tree](SVGs_builtin/b+tree.svg) |
| burger | 5.62e-01 | 5.18e-01 | 6.35e-01 | 5.78e-01 | s | A |![burger](SVGs_builtin/burger.svg) |
| bwt | 1.58e+04 | 1.59e+04 | 1.21e+03 | | ms | A |![bwt](SVGs_builtin/bwt.svg) |
| car | 4.84e-03 | 4.55e-03 | 5.60e-03 | 5.65e-03 | s | A |![car](SVGs_builtin/car.svg) |
| cbsfil | 3.66e-02 | 3.43e-02 | 5.26e-02 | 3.64e-02 | s | A |![cbsfil](SVGs_builtin/cbsfil.svg) |
| ccsd-trpdrv | 4.20e-05 | 4.50e-05 | 4.90e-05 | 8.70e-05 | s | A |![ccsd-trpdrv](SVGs_builtin/ccsd-trpdrv.svg) |
| ccs | 1.09e-02 | 1.17e-02 | 8.89e-03 | 1.31e-02 | s | B |![ccs](SVGs_builtin/ccs.svg) |
| cfd | 8.72e-02 | 8.18e-02 | 3.01e-01 | 4.22e-01 | s | A |![cfd](SVGs_builtin/cfd.svg) |
| chacha20 | 2.25e+01 | 2.28e+01 | 6.75e+01 | 4.18e+01 | us | B |![chacha20](SVGs_builtin/chacha20.svg) |
| channelShuffle | 2.40e+01 | 2.43e+01 | 3.85e+01 | 3.83e+01 | ms | A |![channelShuffle](SVGs_builtin/channelShuffle.svg) |
| channelSum | 6.26e+01 | 6.27e+01 | 5.93e+01 | 8.76e+01 | ms | A |![channelSum](SVGs_builtin/channelSum.svg) |
| chemv | | 6.82e+02 | 3.50e+02 | 7.67e+02 | us | B |![chemv](SVGs_builtin/chemv.svg) |
| che | 7.66e-01 | 6.09e-01 | 8.56e-01 | 9.08e-01 | s | A |![che](SVGs_builtin/che.svg) |
| chi2 | | 1.47e-02 | 1.40e-02 | 1.40e-02 | s | A |![chi2](SVGs_builtin/chi2.svg) |
| clenergy | 2.10e-01 | 2.23e-01 | 7.12e-01 | 7.32e-01 | s | A |![clenergy](SVGs_builtin/clenergy.svg) |
| clink | 3.20e+01 | 3.58e+01 | 4.99e+01 | 4.28e+01 | ms | A |![clink](SVGs_builtin/clink.svg) |
| cmp | | | | |  | A |![cmp](SVGs_builtin/cmp.svg) |
| cm | | | | |  | B |![cm](SVGs_builtin/cm.svg) |
| cobahh | 4.61e+04 | 3.44e+04 | 4.96e+04 | 1.40e+04 | us | A |![cobahh](SVGs_builtin/cobahh.svg) |
| colorwheel | 1.25e+02 | 1.24e+02 | 1.23e+02 | 1.22e+02 | ms | A |![colorwheel](SVGs_builtin/colorwheel.svg) |
| columnarSolver | 7.30e+00 | 7.76e+00 | 1.47e+01 | 7.75e+00 | s | B |![columnarSolver](SVGs_builtin/columnarSolver.svg) |
| complex | 1.02e-02 | 1.04e-02 | 1.01e-02 | 1.04e-02 | s | A |![complex](SVGs_builtin/complex.svg) |
| compute-score | 6.12e-01 | 6.14e-01 | 7.00e-01 | 1.79e+00 | ms | B |![compute-score](SVGs_builtin/compute-score.svg) |
| concat | 2.98e+03 | 2.97e+03 | 3.06e+03 | 3.67e+03 | us | A |![concat](SVGs_builtin/concat.svg) |
| contract | 6.65e+00 | 8.82e+00 | 2.63e+01 | 8.29e+00 | s | A |![contract](SVGs_builtin/contract.svg) |
| conversion | 1.74e-01 | 1.69e-01 | 9.22e-02 | 7.96e-02 | , | A |![conversion](SVGs_builtin/conversion.svg) |
| convolution1D | 1.42e+05 | 1.72e+05 | 3.11e+05 | 1.58e+06 | us | B |![convolution1D](SVGs_builtin/convolution1D.svg) |
| convolution3D | 3.16e+01 | 3.53e+01 | 1.22e+03 | 2.76e+01 | us | A |![convolution3D](SVGs_builtin/convolution3D.svg) |
| convolutionSeparable | 2.30e-03 | 2.30e-03 | | 1.13e-02 | s | B |![convolutionSeparable](SVGs_builtin/convolutionSeparable.svg) |
| cooling | 5.13e+02 | 3.97e+02 | 4.19e+02 | 3.80e+02 | ms | A |![cooling](SVGs_builtin/cooling.svg) |
| crc64 | 4.77e+04 | 5.24e+04 | 5.54e+04 | 5.60e+04 | MB/s | A |![crc64](SVGs_builtin/crc64.svg) |
| cross | 6.96e+03 | 6.96e+03 | 7.00e+03 | 7.00e+03 | us | A |![cross](SVGs_builtin/cross.svg) |
| crs | 6.64e-02 | 7.09e-02 | | | s | B |![crs](SVGs_builtin/crs.svg) |
| d2q9-bgk | 5.83e+00 | 5.53e+00 | | 2.45e+01 | us | B |![d2q9-bgk](SVGs_builtin/d2q9-bgk.svg) |
| damage | 1.89e-02 | 1.99e-02 | 3.22e-02 | 2.39e-01 | s|ms|us | B |![damage](SVGs_builtin/damage.svg) |
| dct8x8 | 9.43e-03 | 9.43e-03 | | | s | B |![dct8x8](SVGs_builtin/dct8x8.svg) |
| ddbp | 3.42e-01 | | | 2.63e-01 | s | B |![ddbp](SVGs_builtin/ddbp.svg) |
| debayer | 3.62e-03 | 3.68e-03 | 7.34e-02 | 8.39e-02 | s | B |![debayer](SVGs_builtin/debayer.svg) |
| degrid | 1.29e-02 | 1.28e-02 | 1.01e-01 | 3.28e-02 | s | A |![degrid](SVGs_builtin/degrid.svg) |
| dense-embedding | 2.37e+04 | 2.37e+04 | 2.35e+04 | 3.88e+04 | us | B |![dense-embedding](SVGs_builtin/dense-embedding.svg) |
| depixel | 1.01e+02 | 1.52e+02 | 2.04e+02 | 1.63e+02 | s | A |![depixel](SVGs_builtin/depixel.svg) |
| deredundancy | 1.62e+01 | 1.55e+01 | 1.57e+01 | 1.57e+01 | s | B |![deredundancy](SVGs_builtin/deredundancy.svg) |
| diamond | 3.67e+01 | 1.61e+01 | | 3.72e+01 | s | A |![diamond](SVGs_builtin/diamond.svg) |
| distort | 1.24e-02 | 7.93e-03 | 2.20e-02 | 1.97e-02 | ms | A |![distort](SVGs_builtin/distort.svg) |
| divergence | 1.09e+03 | 1.44e+03 | 3.26e+02 | 3.39e+02 | ms | A |![divergence](SVGs_builtin/divergence.svg) |
| doh | 6.92e+00 | 7.75e+00 | 1.39e+01 | 2.32e+01 | us | A |![doh](SVGs_builtin/doh.svg) |
| dp | 2.11e+01 | | 2.02e+01 | 2.04e+01 | ms | A |![dp](SVGs_builtin/dp.svg) |
| dslash | 4.09e-02 | 5.08e-02 | 6.24e-02 | 6.45e-02 | s | A |![dslash](SVGs_builtin/dslash.svg) |
| dxtc2 | 3.16e+02 | 3.61e+02 | | | us | B |![dxtc2](SVGs_builtin/dxtc2.svg) |
| easyWave | 2.14e+04 | 2.05e+04 | 1.94e+04 | 2.07e+04 | ms | A |![easyWave](SVGs_builtin/easyWave.svg) |
| ecdh | 1.19e-01 | 8.15e-02 | 1.50e-01 | 1.64e-01 | s | A |![ecdh](SVGs_builtin/ecdh.svg) |
| eigenvalue | 2.04e+01 | 2.43e+01 | | 2.34e+01 | us | A |![eigenvalue](SVGs_builtin/eigenvalue.svg) |
| entropy | 4.86e-02 | 5.30e-02 | 5.17e-01 | 7.12e-02 | s | B |![entropy](SVGs_builtin/entropy.svg) |
| epistasis | 2.01e-02 | 1.99e-02 | 1.52e+00 | 2.47e-02 | s | A |![epistasis](SVGs_builtin/epistasis.svg) |
| expdist | 3.32e-03 | 3.29e-03 | | 8.57e-03 | s | B |![expdist](SVGs_builtin/expdist.svg) |
| extend2 | 2.60e+03 | 1.10e+03 | 2.05e+03 | 2.18e+03 | us | A |![extend2](SVGs_builtin/extend2.svg) |
| extrema | 2.54e-01 | 3.09e-01 | 9.44e-02 | 1.09e-01 | s | A |![extrema](SVGs_builtin/extrema.svg) |
| face | 2.64e+00 | 2.56e+00 | 2.37e+00 | 2.11e-01 | s | A |![face](SVGs_builtin/face.svg) |
| fdtd3d | 4.69e-04 | 3.96e-04 | | 4.78e-03 | s | B |![fdtd3d](SVGs_builtin/fdtd3d.svg) |
| feynman-kac | 7.87e+00 | 3.94e+00 | 4.58e+00 | 1.98e+00 | s | A |![feynman-kac](SVGs_builtin/feynman-kac.svg) |
| fft | 3.55e-04 | 6.14e-04 | | 1.15e-03 | s | B |![fft](SVGs_builtin/fft.svg) |
| fhd | | 1.79e+02 | | | s | A |![fhd](SVGs_builtin/fhd.svg) |
| filter | 6.17e+00 | | | 7.69e+01 | ms | B |![filter](SVGs_builtin/filter.svg) |
| flip | 4.12e+01 | 4.72e+01 | 6.36e+01 | 7.60e+01 | ms | A |![flip](SVGs_builtin/flip.svg) |
| floydwarshall | 2.02e-01 | 2.08e-01 | 2.39e-01 | 3.47e-01 | s | A |![floydwarshall](SVGs_builtin/floydwarshall.svg) |
| fluidSim | 1.20e-05 | 8.00e-06 | 2.20e-05 | 2.90e-05 | s | A |![fluidSim](SVGs_builtin/fluidSim.svg) |
| fpc | 1.19e-02 | 1.21e-02 | | | s | B |![fpc](SVGs_builtin/fpc.svg) |
| fpdc | 6.77e-04 | 6.42e-04 | | | s | B |![fpdc](SVGs_builtin/fpdc.svg) |
| fresnel | 3.13e-03 | 1.49e-03 | 8.27e-03 | 8.23e-03 | s | A |![fresnel](SVGs_builtin/fresnel.svg) |
| frna | 7.78e+01 | 8.04e+01 | | | s | B |![frna](SVGs_builtin/frna.svg) |
| fwt | 4.81e-04 | 4.84e-04 | 7.83e-04 | 1.56e-03 | s | B |![fwt](SVGs_builtin/fwt.svg) |
| gabor | 7.95e+03 | 8.18e+03 | 1.01e+04 | 1.11e+04 | us | A |![gabor](SVGs_builtin/gabor.svg) |
| gamma-correction | 5.21e-04 | 6.17e-04 | 4.96e-04 | 1.79e-03 | s | A |![gamma-correction](SVGs_builtin/gamma-correction.svg) |
| ga | 2.36e+00 | 2.81e+00 | 2.85e+00 | 2.68e+00 | s | A |![ga](SVGs_builtin/ga.svg) |
| gaussian | 4.61e+05 | 3.96e+05 | 3.72e+05 | 3.82e+05 | us | A |![gaussian](SVGs_builtin/gaussian.svg) |
| gc | 6.20e-05 | | | | s | A |![gc](SVGs_builtin/gc.svg) |
| gd | 3.99e-01 | 5.10e-01 | 5.42e-01 | 7.87e-01 | s | A |![gd](SVGs_builtin/gd.svg) |
| geglu | 7.58e+03 | 1.83e+04 | 1.97e+04 | 2.02e+04 | us | A |![geglu](SVGs_builtin/geglu.svg) |
| geodesic | 2.89e+02 | 2.60e+02 | 2.86e+02 | 2.96e+02 | us | A |![geodesic](SVGs_builtin/geodesic.svg) |
| glu | 1.86e+04 | 1.52e+04 | 1.90e+04 | 2.06e+04 | us | A |![glu](SVGs_builtin/glu.svg) |
| gmm | 3.26e+00 | 3.58e+00 | | 4.23e+00 | s | B |![gmm](SVGs_builtin/gmm.svg) |
| goulash | 1.25e+00 | 1.25e+00 | 1.26e+00 | 1.75e+00 | s | A |![goulash](SVGs_builtin/goulash.svg) |
| gpp | 2.42e-01 | 1.83e-01 | 1.67e-01 | 1.72e-01 | s | A |![gpp](SVGs_builtin/gpp.svg) |
| grep | | | | |  | B |![grep](SVGs_builtin/grep.svg) |
| grrt | 1.01e+01 | 1.05e+01 | 9.26e+00 | 8.74e+00 | s | A |![grrt](SVGs_builtin/grrt.svg) |
| haccmk | 2.15e-03 | 1.91e-03 | 4.60e-05 | 1.94e-03 | s | A |![haccmk](SVGs_builtin/haccmk.svg) |
| hausdorff | 6.49e+00 | 6.76e+00 | 1.01e+01 | 7.26e+00 | ms | A |![hausdorff](SVGs_builtin/hausdorff.svg) |
| haversine | 2.09e-04 | 2.35e-04 | | | s | A |![haversine](SVGs_builtin/haversine.svg) |
| heartwall | 7.42e+00 | | | | s | A |![heartwall](SVGs_builtin/heartwall.svg) |
| heat2d | 1.22e+03 | 1.18e+03 | 7.72e+02 | 6.89e+02 | GFLOPS | A |![heat2d](SVGs_builtin/heat2d.svg) |
| heat | 2.29e+01 | 2.20e+01 | 1.53e+01 | 1.61e+01 | s | A |![heat](SVGs_builtin/heat.svg) |
| hellinger | 2.87e-03 | 1.62e-03 | 2.73e-02 | 3.79e-03 | s | A |![hellinger](SVGs_builtin/hellinger.svg) |
| henry | 1.25e-03 | 1.14e-03 | 4.64e-04 | 4.28e-04 | s | A |![henry](SVGs_builtin/henry.svg) |
| hexciton | 1.38e+00 | 3.27e+00 | | 2.13e+00 | s | B |![hexciton](SVGs_builtin/hexciton.svg) |
| histogram | 3.32e+02 | 2.93e+02 | | 7.93e+02 | us  | B |![histogram](SVGs_builtin/histogram.svg) |
| hmm | 1.39e-01 | 1.44e-01 | 2.83e-01 | 1.90e-01 | s | A |![hmm](SVGs_builtin/hmm.svg) |
| hogbom | 2.60e-01 | 1.40e-01 | 1.60e-01 | 3.10e-01 | ms | B |![hogbom](SVGs_builtin/hogbom.svg) |
| hotspot3D | 7.44e+00 | 7.55e+00 | 1.73e+01 | 1.66e+01 | us | A |![hotspot3D](SVGs_builtin/hotspot3D.svg) |
| hwt1d | 6.11e-04 | 6.22e-04 | 6.34e-04 | 7.06e-04 | s | B |![hwt1d](SVGs_builtin/hwt1d.svg) |
| hybridsort | 2.22e+02 | 5.28e+01 | | 2.25e+02 | ms | B |![hybridsort](SVGs_builtin/hybridsort.svg) |
| hypterm | 2.30e+00 | 2.61e+00 | 2.39e+00 | 2.42e+00 | ms | A |![hypterm](SVGs_builtin/hypterm.svg) |
| idivide | 1.08e-02 | 4.31e-03 | 7.27e-03 | 1.82e-02 | s | A |![idivide](SVGs_builtin/idivide.svg) |
| interleave | 1.47e-02 | 1.47e-02 | 1.40e-02 | 7.96e-03 | s | A |![interleave](SVGs_builtin/interleave.svg) |
| interval | 4.33e+01 | 3.86e+01 | 1.02e+02 | 7.41e+01 | us | A |![interval](SVGs_builtin/interval.svg) |
| inversek2j | 2.71e+00 | 7.76e+00 | 2.25e+03 | 1.11e+03 | us | A |![inversek2j](SVGs_builtin/inversek2j.svg) |
| ising | 3.09e-01 | 3.50e-01 | 3.89e-01 | 4.86e-01 | s | A |![ising](SVGs_builtin/ising.svg) |
| iso2dfd | 2.40e+01 | 2.44e+01 | 2.24e+01 | 3.20e+01 | ms | A |![iso2dfd](SVGs_builtin/iso2dfd.svg) |
| jacobi | 2.74e+00 | 2.72e+00 | 3.51e+00 | 3.44e+00 | s | A |![jacobi](SVGs_builtin/jacobi.svg) |
| jenkins-hash | 3.10e-03 | 3.59e-03 | 2.34e-01 | 2.28e-01 | s | A |![jenkins-hash](SVGs_builtin/jenkins-hash.svg) |
| kalman | 3.22e-01 | | | | s | A |![kalman](SVGs_builtin/kalman.svg) |
| keccaktreehash | 2.91e+07 | 3.15e+07 | 7.36e+05 | 2.86e+07 | kB/s | A |![keccaktreehash](SVGs_builtin/keccaktreehash.svg) |
| keogh | 4.56e-03 | 5.05e-03 | 5.09e-02 | 1.52e-01 | s | A |![keogh](SVGs_builtin/keogh.svg) |
| kernelLaunch | 1.83e+00 | 2.07e+00 | 1.12e+02 | 1.24e+01 | us | B |![kernelLaunch](SVGs_builtin/kernelLaunch.svg) |
| kmeans | 2.79e+04 | 2.83e+04 | 2.64e+04 | 2.21e+04 | ms | A |![kmeans](SVGs_builtin/kmeans.svg) |
| knn | 6.45e-01 | 4.69e-01 | 5.92e-01 | 7.78e-01 | s | B |![knn](SVGs_builtin/knn.svg) |
| lanczos | 3.14e-01 | 3.12e-01 | | 4.95e-01 | s | B |![lanczos](SVGs_builtin/lanczos.svg) |
| langevin | 6.30e-03 | 6.19e-03 | 6.31e-03 | 9.40e-03 | s | A |![langevin](SVGs_builtin/langevin.svg) |
| langford | 1.24e-01 | 1.21e-01 | | | s | B |![langford](SVGs_builtin/langford.svg) |
| laplace3d | 1.01e-02 | 1.03e-02 | 1.80e-02 | 1.95e-02 | s | A |![laplace3d](SVGs_builtin/laplace3d.svg) |
| laplace | 1.16e+00 | 8.27e+00 | 4.94e+00 | 4.61e+00 | s | A |![laplace](SVGs_builtin/laplace.svg) |
| lavaMD | 4.99e+00 | 5.83e+00 | | 1.82e+01 | s | B |![lavaMD](SVGs_builtin/lavaMD.svg) |
| layout | 1.33e+02 | 1.41e+02 | 9.07e+01 | 8.85e+01 | us | A |![layout](SVGs_builtin/layout.svg) |
| lci | 1.53e+01 | 7.50e+00 | 4.28e+00 | 4.28e+00 | s | A |![lci](SVGs_builtin/lci.svg) |
| lda | 6.31e-03 | 6.53e-03 | | | s | B |![lda](SVGs_builtin/lda.svg) |
| ldpc | 1.19e+00 | 1.30e+00 | | 5.92e+00 | s | B |![ldpc](SVGs_builtin/ldpc.svg) |
| lebesgue | 2.66e+00 | 1.75e-01 | 2.53e+00 | 2.47e+00 | s | A |![lebesgue](SVGs_builtin/lebesgue.svg) |
| leukocyte | 5.89e-01 | | | 8.64e-01 | s | A |![leukocyte](SVGs_builtin/leukocyte.svg) |
| libor | 2.99e-03 | 3.21e-03 | 1.07e-01 | 3.30e-03 | s | A |![libor](SVGs_builtin/libor.svg) |
| lid-driven-cavity | 7.56e+00 | 8.37e+00 | | 1.98e+01 | s | B |![lid-driven-cavity](SVGs_builtin/lid-driven-cavity.svg) |
| lif | 1.37e+04 | 1.27e+04 | 1.70e+04 | 1.09e+04 | us | A |![lif](SVGs_builtin/lif.svg) |
| linearprobing | 1.76e-03 | 1.78e-03 | | | s | A |![linearprobing](SVGs_builtin/linearprobing.svg) |
| log2 | 2.02e+03 | 1.84e+03 | 2.41e+03 | 2.68e+03 | us | A |![log2](SVGs_builtin/log2.svg) |
| lombscargle | 3.05e+02 | 1.80e+02 | 2.02e+02 | 2.17e+02 | us | A |![lombscargle](SVGs_builtin/lombscargle.svg) |
| loopback | 3.26e-02 | 2.90e-02 | 2.98e-02 | 5.74e-02 | s | B |![loopback](SVGs_builtin/loopback.svg) |
| lrn | 4.28e-01 | 6.49e-01 | 5.27e-01 | 6.42e-01 | s | A |![lrn](SVGs_builtin/lrn.svg) |
| lr | 7.56e+00 | 7.51e+00 | 2.50e+01 | 5.51e+01 | us | B |![lr](SVGs_builtin/lr.svg) |
| lsqt | 1.30e+01 | 1.28e+01 | 1.77e+01 | 1.74e+01 | s | A |![lsqt](SVGs_builtin/lsqt.svg) |
| lud | 5.07e+00 | 5.72e+00 | 7.03e+00 | 4.36e+01 | s | B |![lud](SVGs_builtin/lud.svg) |
| lulesh | 4.59e+00 | 5.58e+00 | 8.48e+00 | 5.18e+00 | s | A |![lulesh](SVGs_builtin/lulesh.svg) |
| mallocFree | 2.53e+06 | 3.07e+03 | 2.43e+06 | 5.09e+02 | us | A |![mallocFree](SVGs_builtin/mallocFree.svg) |
| mandelbrot | 5.78e-02 | 6.43e-02 | 5.88e-02 | 6.16e-02 | ms | A |![mandelbrot](SVGs_builtin/mandelbrot.svg) |
| mask | 2.79e+04 | 2.75e+04 | 2.74e+04 | 2.78e+04 | us | A |![mask](SVGs_builtin/mask.svg) |
| match | 2.38e+02 | 2.51e+02 | | 3.57e+11 | ms | B |![match](SVGs_builtin/match.svg) |
| matern | 1.81e+04 | 1.92e+04 | 6.98e+03 | 1.45e+05 | us | B |![matern](SVGs_builtin/matern.svg) |
| matrix-rotate | 7.03e-02 | 7.02e-02 | 7.14e-02 | 6.75e-02 | s | A |![matrix-rotate](SVGs_builtin/matrix-rotate.svg) |
| maxFlops | 4.61e+00 | 4.62e+00 | 4.80e+00 | 4.59e+00 | s | A |![maxFlops](SVGs_builtin/maxFlops.svg) |
| maxpool3d | 3.94e-03 | 4.15e-03 | 5.74e-03 | 6.76e-03 | s | A |![maxpool3d](SVGs_builtin/maxpool3d.svg) |
| mcmd | 1.23e+01 | 8.96e+00 | 1.07e+01 | 1.10e+01 | s | A |![mcmd](SVGs_builtin/mcmd.svg) |
| mcpr | 2.93e-02 | 5.56e-03 | 1.28e-01 | 2.83e-02 | s | B |![mcpr](SVGs_builtin/mcpr.svg) |
| md5hash | 1.44e+04 | 1.39e+04 | 1.44e+04 | 1.45e+04 | ms | A |![md5hash](SVGs_builtin/md5hash.svg) |
| mdh | 8.79e-02 | 2.39e+00 | 1.74e+00 | 6.86e-02 |  | A |![mdh](SVGs_builtin/mdh.svg) |
| md | 6.56e-05 | 6.64e-05 | 7.49e-05 | 7.23e-05 | s | A |![md](SVGs_builtin/md.svg) |
| meanshift | 4.41e-01 | 8.77e-01 | 1.38e+00 | 9.38e-01 | ms | B |![meanshift](SVGs_builtin/meanshift.svg) |
| medianfilter | 7.60e-05 | | 8.50e-05 | 2.40e-04 | s | B |![medianfilter](SVGs_builtin/medianfilter.svg) |
| memcpy | 1.82e+02 | 2.00e+02 | 2.49e+02 | 2.49e+02 | us | A |![memcpy](SVGs_builtin/memcpy.svg) |
| memtest | 3.25e-02 | 3.46e-02 | 4.32e-02 | 4.31e-02 | s | A |![memtest](SVGs_builtin/memtest.svg) |
| merge | 6.73e+04 | 6.65e+04 | | 1.62e+05 | us | B |![merge](SVGs_builtin/merge.svg) |
| metropolis | 9.72e+00 | 1.11e+01 | | 6.14e+01 | s | B |![metropolis](SVGs_builtin/metropolis.svg) |
| michalewicz | 1.09e+05 | 1.05e+06 | 1.45e+05 | 1.90e+05 | us | A |![michalewicz](SVGs_builtin/michalewicz.svg) |
| minibude | 7.66e+02 | 9.10e+02 | 8.43e+02 | 1.25e+03 | ms | B |![minibude](SVGs_builtin/minibude.svg) |
| miniFE | | | | |  | A |![miniFE](SVGs_builtin/miniFE.svg) |
| minimap2 | 4.20e-02 | 5.82e-03 | 1.13e-02 | 1.09e-01 | s | A |![minimap2](SVGs_builtin/minimap2.svg) |
| minisweep | 5.03e+01 | 5.41e+01 | 3.29e+01 | 3.34e+01 | s | A |![minisweep](SVGs_builtin/minisweep.svg) |
| miniWeather | 6.96e-01 | 6.94e-01 | 3.00e+01 | 2.99e+01 | s | A |![miniWeather](SVGs_builtin/miniWeather.svg) |
| minkowski | 2.31e-03 | 2.65e-03 | 2.22e-03 | 2.20e-03 | s | A |![minkowski](SVGs_builtin/minkowski.svg) |
| mis | 2.17e-04 | 1.07e-04 | 0.00e+00 | 0.00e+00 | s | B |![mis](SVGs_builtin/mis.svg) |
| mixbench | 6.06e-01 | 5.55e-01 | 6.31e-01 | 6.67e-01 | s | B |![mixbench](SVGs_builtin/mixbench.svg) |
| morphology | 2.23e-01 | 2.24e-01 | | | s | B |![morphology](SVGs_builtin/morphology.svg) |
| mrc | 1.38e+04 | 1.38e+04 | 1.39e+04 | 1.41e+04 | us | A |![mrc](SVGs_builtin/mrc.svg) |
| mriQ | 2.60e+00 | 3.24e+00 | 3.29e+00 | 3.30e+00 | s | A |![mriQ](SVGs_builtin/mriQ.svg) |
| mr | 2.25e+00 | 1.12e+00 | | 2.30e+00 | ms | A |![mr](SVGs_builtin/mr.svg) |
| mt | 2.95e-03 | 1.88e-03 | 1.67e-03 | 1.68e-03 | s | A |![mt](SVGs_builtin/mt.svg) |
| multimaterial | 1.14e+02 | 2.46e+00 | 3.15e+00 | 4.37e+00 | ms | A |![multimaterial](SVGs_builtin/multimaterial.svg) |
| murmurhash3 | 3.97e-03 | 3.88e-03 | 3.98e-03 | 3.91e-03 | s | A |![murmurhash3](SVGs_builtin/murmurhash3.svg) |
| myocyte | 1.49e+01 | | 3.26e+00 | 4.00e+00 | s | A |![myocyte](SVGs_builtin/myocyte.svg) |
| nbody | 3.54e-02 | 3.35e-02 | 2.35e+00 | 2.40e+00 |  | A |![nbody](SVGs_builtin/nbody.svg) |
| ne | 9.06e-04 | 1.18e-03 | 1.33e-03 | 1.61e-03 | s | A |![ne](SVGs_builtin/ne.svg) |
| nlll | 8.41e+01 | 9.41e+01 | 1.33e+04 | 1.00e+02 | us | B |![nlll](SVGs_builtin/nlll.svg) |
| nms | 7.40e-05 | 1.36e-04 | | 2.18e-04 | s | B |![nms](SVGs_builtin/nms.svg) |
| nn | 2.43e+00 | 2.42e+00 | 7.62e+00 | 8.28e+00 | us | A |![nn](SVGs_builtin/nn.svg) |
| norm2 | 2.05e+03 | | 2.91e+03 | 1.59e+04 | us | A |![norm2](SVGs_builtin/norm2.svg) |
| nqueen | 3.85e-02 | 4.50e-02 | 2.14e-01 | 4.39e-02 | s | A |![nqueen](SVGs_builtin/nqueen.svg) |
| ntt | 3.62e+01 | 4.16e+01 | | 4.19e+01 | us | B |![ntt](SVGs_builtin/ntt.svg) |
| nw | 4.97e-02 | 5.02e-02 | | 1.23e-01 | s | B |![nw](SVGs_builtin/nw.svg) |
| openmp | 2.55e+00 | 9.79e-02 | 2.22e-01 | 2.29e-01 | s | B |![openmp](SVGs_builtin/openmp.svg) |
| overlay | 5.51e-01 | 5.85e-01 | 3.44e+00 | 4.89e+00 | s | A |![overlay](SVGs_builtin/overlay.svg) |
| p4 | 8.25e+01 | 8.52e+01 | | 1.39e+03 | us | B |![p4](SVGs_builtin/p4.svg) |
| page-rank | 3.93e+00 | 3.84e+00 | 4.09e+00 | 4.36e+00 | s | A |![page-rank](SVGs_builtin/page-rank.svg) |
| particle-diffusion | 2.20e-03 | 1.70e-03 | 2.29e-03 | 2.27e-03 | s | A |![particle-diffusion](SVGs_builtin/particle-diffusion.svg) |
| particlefilter | 2.22e+02 | 1.98e+02 | | 2.20e+02 | s | B |![particlefilter](SVGs_builtin/particlefilter.svg) |
| particles | 1.07e+00 | 9.94e-01 | | | s | B |![particles](SVGs_builtin/particles.svg) |
| pathfinder | 3.88e+01 | 3.88e+01 | 4.35e+01 | 7.99e+02 | s | B |![pathfinder](SVGs_builtin/pathfinder.svg) |
| permutate | 2.05e+00 | 2.42e+00 | | 2.26e+00 | s | B |![permutate](SVGs_builtin/permutate.svg) |
| permute | 2.00e+01 | 2.22e+01 | 2.23e+01 | 2.28e+01 | ms | A |![permute](SVGs_builtin/permute.svg) |
| perplexity | 1.64e+00 | 2.24e+00 | 1.53e+00 | 1.69e+00 | s | A |![perplexity](SVGs_builtin/perplexity.svg) |
| phmm | 1.16e+03 | 1.10e+03 | 4.62e+03 | 5.23e+03 | millis | B |![phmm](SVGs_builtin/phmm.svg) |
| pnpoly | 4.59e-02 | 5.17e-02 | | 7.17e-01 | s | A |![pnpoly](SVGs_builtin/pnpoly.svg) |
| pns | 5.44e+00 | 5.38e+00 | | | s | B |![pns](SVGs_builtin/pns.svg) |
| pointwise | 9.21e-02 | 1.14e-01 | 1.80e-01 | 1.66e-01 | s | A |![pointwise](SVGs_builtin/pointwise.svg) |
| pool | 8.88e-03 | 9.02e-03 | 1.83e-02 | 1.57e-02 | s | A |![pool](SVGs_builtin/pool.svg) |
| popcount | 1.71e+05 | 8.28e+04 | 2.00e+05 | 7.06e+04 | us | A |![popcount](SVGs_builtin/popcount.svg) |
| present | 3.81e+01 | 3.88e+01 | 3.70e+01 | 9.36e+01 | us | A |![present](SVGs_builtin/present.svg) |
| prna | | | | |  | B |![prna](SVGs_builtin/prna.svg) |
| projectile | 1.13e-04 | 1.13e-04 | 1.18e-04 | 1.19e-04 | s | A |![projectile](SVGs_builtin/projectile.svg) |
| pso | 2.55e+01 | 2.44e+01 | 2.48e+01 | 2.19e+01 | us | A |![pso](SVGs_builtin/pso.svg) |
| qrg | 4.48e+01 | 3.66e+01 | 5.93e+01 | 7.56e+01 | us | A |![qrg](SVGs_builtin/qrg.svg) |
| qtclustering | 3.31e+01 | 3.20e+01 | | | s | B |![qtclustering](SVGs_builtin/qtclustering.svg) |
| quantBnB | 6.09e+03 | 6.66e+03 | 2.74e+03 | 3.26e+03 | us | A |![quantBnB](SVGs_builtin/quantBnB.svg) |
| quicksort | 6.92e+03 | 6.87e+03 | | 2.53e+03 | ms | B |![quicksort](SVGs_builtin/quicksort.svg) |
| radixsort | 5.06e-04 | 5.06e-04 | | 5.71e-03 | s | B |![radixsort](SVGs_builtin/radixsort.svg) |
| rainflow | 1.71e+05 | 1.50e+05 | 1.69e+05 | 1.27e+05 | us | A |![rainflow](SVGs_builtin/rainflow.svg) |
| randomAccess | 1.58e-01 | 1.58e-01 | | 1.92e+00 | s | A |![randomAccess](SVGs_builtin/randomAccess.svg) |
| reaction | 1.63e+00 | 1.65e+00 | | 6.75e+00 | s | B |![reaction](SVGs_builtin/reaction.svg) |
| recursiveGaussian | 1.32e-03 | 1.48e-03 | | 1.67e-03 | s | B |![recursiveGaussian](SVGs_builtin/recursiveGaussian.svg) |
| resize | 6.98e+02 | 7.58e+02 | 7.14e+02 | 8.35e+02 | us | A |![resize](SVGs_builtin/resize.svg) |
| reverse | 9.29e-01 | 1.25e+00 | 4.25e+00 | 5.49e+00 | s | B |![reverse](SVGs_builtin/reverse.svg) |
| rfs | 8.78e-03 | 8.78e-03 | 8.79e-03 | 8.79e-03 | s | A |![rfs](SVGs_builtin/rfs.svg) |
| rng-wallace | 1.25e+02 | 1.18e+02 | | 1.12e+03 | us | B |![rng-wallace](SVGs_builtin/rng-wallace.svg) |
| rodrigues | 3.10e+04 | 3.50e+04 | 3.49e+04 | 2.01e+04 | us | A |![rodrigues](SVGs_builtin/rodrigues.svg) |
| romberg | 2.85e-04 | 2.92e-04 | 3.51e-04 | 3.26e-04 | s | B |![romberg](SVGs_builtin/romberg.svg) |
| rsbench | 2.94e+00 | 1.22e+00 | 9.66e+00 | 2.91e+00 | s | A |![rsbench](SVGs_builtin/rsbench.svg) |
| rsc | 1.55e+02 | 1.31e+02 | 2.17e+02 | 2.28e+02 | ms | B |![rsc](SVGs_builtin/rsc.svg) |
| rtm8 | 1.02e-03 | 1.06e-03 | 8.59e-04 | 8.90e-04 | s | A |![rtm8](SVGs_builtin/rtm8.svg) |
| rushlarsen | 2.29e-01 | 3.24e-01 | 2.32e-01 | 2.45e-01 | s | A |![rushlarsen](SVGs_builtin/rushlarsen.svg) |
| s3d | 2.86e+00 | 4.80e-01 | | 2.84e+00 | s | A |![s3d](SVGs_builtin/s3d.svg) |
| s8n | 1.16e+01 | 1.33e+01 | 4.06e+01 | 5.00e+01 | us | A |![s8n](SVGs_builtin/s8n.svg) |
| sad | 4.00e+01 | 4.20e+01 | 4.84e+02 | 8.18e+04 | ms | A |![sad](SVGs_builtin/sad.svg) |
| sampling | 2.39e+02 | 2.81e+02 | 3.67e+02 | 5.63e+02 | us | B |![sampling](SVGs_builtin/sampling.svg) |
| scan2 | 3.55e+02 | 4.19e+02 | | 4.32e+03 | us | B |![scan2](SVGs_builtin/scan2.svg) |
| scan | 1.75e+04 | 1.87e+04 | | 2.05e+05 | us | B |![scan](SVGs_builtin/scan.svg) |
| scatterAdd | 3.91e+03 | 4.85e+03 | 1.61e+05 | 1.61e+05 | us | A |![scatterAdd](SVGs_builtin/scatterAdd.svg) |
| scel | 2.92e+03 | 2.93e+03 | 4.27e+04 | 6.24e+03 | us | A |![scel](SVGs_builtin/scel.svg) |
| secp256k1 | 2.91e-03 | 2.77e-03 | 2.92e-03 | 2.87e-03 | s | A |![secp256k1](SVGs_builtin/secp256k1.svg) |
| sheath | 3.18e+00 | 3.42e+00 | 3.36e+00 | 3.40e+00 | ms | A |![sheath](SVGs_builtin/sheath.svg) |
| shmembench | 2.70e+00 | 6.74e+00 | | | ms | B |![shmembench](SVGs_builtin/shmembench.svg) |
| simplemoc | 1.76e+02 | 3.42e+02 | 2.35e+02 | 2.35e+02 | s | B |![simplemoc](SVGs_builtin/simplemoc.svg) |
| simpleSpmv | 4.46e+01 | 3.57e+01 | 2.28e+01 | 4.70e+01 | ms | A |![simpleSpmv](SVGs_builtin/simpleSpmv.svg) |
| slu | | | | | ms | B |![slu](SVGs_builtin/slu.svg) |
| snake | 4.39e+03 | 4.85e+03 | 7.15e+03 | 7.15e+03 | us | A |![snake](SVGs_builtin/snake.svg) |
| sobel | 4.80e+00 | 4.85e+00 | 1.01e+01 | 1.11e+01 | us | A |![sobel](SVGs_builtin/sobel.svg) |
| sobol | 1.22e-03 | 1.28e-03 | | 3.35e-03 | s | B |![sobol](SVGs_builtin/sobol.svg) |
| softmax-online | 1.44e+01 | 1.45e+01 | | 2.92e+01 | ms | B |![softmax-online](SVGs_builtin/softmax-online.svg) |
| softmax | 1.12e+01 | 1.12e+01 | 3.43e+03 | 5.39e+01 | ms | A |![softmax](SVGs_builtin/softmax.svg) |
| sort | 2.13e-02 | | | 3.44e-01 | s | B |![sort](SVGs_builtin/sort.svg) |
| sosfil | 2.02e-02 | 2.29e-02 | 3.14e-02 | 1.05e-01 | s | B |![sosfil](SVGs_builtin/sosfil.svg) |
| sph | 5.81e+00 | 5.84e+00 | 4.01e+01 | 5.91e+00 | ms | A |![sph](SVGs_builtin/sph.svg) |
| split | 1.27e+04 | 1.28e+04 | | 1.49e+05 | us | B |![split](SVGs_builtin/split.svg) |
| spm | 7.59e+01 | 6.59e+01 | 6.33e+01 | 7.27e+01 | ms | A |![spm](SVGs_builtin/spm.svg) |
| sptrsv | 1.18e+02 | 1.37e+02 | | 1.17e+02 | us | B |![sptrsv](SVGs_builtin/sptrsv.svg) |
| srad | 3.67e+01 | 3.74e+01 | | | s | B |![srad](SVGs_builtin/srad.svg) |
| ss | 2.49e+02 | 2.37e+02 | 3.80e+02 | 7.90e+02 | us | B |![ss](SVGs_builtin/ss.svg) |
| stddev | 3.77e-03 | 5.05e-03 | 5.56e-03 | 6.51e-02 | s | B |![stddev](SVGs_builtin/stddev.svg) |
| stencil1d | 2.25e-03 | 2.25e-03 | 3.00e-01 | 3.50e-01 | s | A |![stencil1d](SVGs_builtin/stencil1d.svg) |
| stencil3d | 6.27e-03 | 6.31e-03 | 6.40e-03 | 1.74e-02 | s | B |![stencil3d](SVGs_builtin/stencil3d.svg) |
| streamcluster | 5.84e+00 | 1.15e+01 | 9.53e+00 | 7.69e+00 | s | B |![streamcluster](SVGs_builtin/streamcluster.svg) |
| su3 | 1.92e+00 | 2.32e+00 | 2.38e+00 | 2.35e+00 | s | A |![su3](SVGs_builtin/su3.svg) |
| surfel | | | 3.07e+02 | 3.59e+02 | ms | A |![surfel](SVGs_builtin/surfel.svg) |
| svd3x3 | 4.60e+01 | 4.84e+01 | 6.81e+01 | 7.09e+01 | us | A |![svd3x3](SVGs_builtin/svd3x3.svg) |
| sw4ck | 8.78e+00 | 1.02e+01 | 2.48e+02 | 2.36e+02 | ms | A |![sw4ck](SVGs_builtin/sw4ck.svg) |
| swish | 4.13e+03 | 4.35e+03 | 4.58e+03 | 6.21e+03 | us | A |![swish](SVGs_builtin/swish.svg) |
| tensorT | 1.41e+00 | 1.28e+00 | 1.36e+00 | 4.44e+00 | ms | B |![tensorT](SVGs_builtin/tensorT.svg) |
| testSNAP | 1.11e+01 | 1.16e+01 | 6.35e+02 | 1.48e+01 | ms | A |![testSNAP](SVGs_builtin/testSNAP.svg) |
| thomas | 1.44e+04 | 1.44e+04 | 1.24e+04 | 1.24e+04 | ms | A |![thomas](SVGs_builtin/thomas.svg) |
| threadfence | 2.30e+02 | 2.34e+02 | 1.50e+02 | 3.27e+02 | ms | B |![threadfence](SVGs_builtin/threadfence.svg) |
| tissue | 7.68e-03 | 7.99e-03 | 1.07e-02 | 8.15e-03 | s | A |![tissue](SVGs_builtin/tissue.svg) |
| tonemapping | 1.69e+01 | 1.62e+01 | 1.91e+01 | 1.82e+01 | us | A |![tonemapping](SVGs_builtin/tonemapping.svg) |
| tqs | 1.31e+02 | 7.99e+01 | | 1.42e+02 | ms | B |![tqs](SVGs_builtin/tqs.svg) |
| triad | 1.86e+03 | 1.23e+03 | 5.85e+02 | 6.10e+02 | GFLOPS | A |![triad](SVGs_builtin/triad.svg) |
| tridiagonal | 3.83e-02 | 3.83e-02 | | 1.55e-01 | s | B |![tridiagonal](SVGs_builtin/tridiagonal.svg) |
| tsa | 1.71e+04 | 1.76e+04 | | 1.70e+05 | us | B |![tsa](SVGs_builtin/tsa.svg) |
| tsp | 2.52e-02 | 5.34e-02 | | | s | B |![tsp](SVGs_builtin/tsp.svg) |
| urng | 4.87e+00 | 5.05e+00 | 1.17e+01 | 2.37e+01 | us | B |![urng](SVGs_builtin/urng.svg) |
| vanGenuchten | 7.29e-03 | 7.25e-03 | 7.45e-03 | 7.55e-03 | s | A |![vanGenuchten](SVGs_builtin/vanGenuchten.svg) |
| vmc | 2.52e-03 | 1.88e-03 | | | s | B |![vmc](SVGs_builtin/vmc.svg) |
| vol2col | 2.53e+03 | 2.55e+03 | 2.02e+03 | 2.73e+03 | us | A |![vol2col](SVGs_builtin/vol2col.svg) |
| wedford | | | | | ms | B |![wedford](SVGs_builtin/wedford.svg) |
| winograd | 7.85e-01 | 2.94e-01 | 3.19e+00 | 3.14e+00 | s | A |![winograd](SVGs_builtin/winograd.svg) |
| wlcpow | 2.03e+03 | 2.09e+03 | | 6.74e+03 | us | B |![wlcpow](SVGs_builtin/wlcpow.svg) |
| wordcount | 1.74e-03 | 3.75e-03 | 1.75e-03 | 2.73e-03 | s | A |![wordcount](SVGs_builtin/wordcount.svg) |
| wsm5 | 2.98e+02 | 2.32e+02 | 3.10e+02 | 3.28e+02 | ms | B |![wsm5](SVGs_builtin/wsm5.svg) |
| wyllie | 2.93e+01 | 2.95e+01 | 3.08e+01 | | ms | B |![wyllie](SVGs_builtin/wyllie.svg) |
| xlqc | 2.24e+06 | | | 3.03e+05 | us | B |![xlqc](SVGs_builtin/xlqc.svg) |
| xsbench | 4.45e+01 | 3.61e+01 | 3.57e+01 | 2.39e+00 | s | A |![xsbench](SVGs_builtin/xsbench.svg) |
| zeropoint | 1.39e+04 | 1.51e+04 | 1.72e+04 | 2.01e+04 | us | A |![zeropoint](SVGs_builtin/zeropoint.svg) |
| zmddft | 1.16e+00 | 1.40e+00 | | 2.01e+00 | ms | B |![zmddft](SVGs_builtin/zmddft.svg) |

分類: OpenMP コードが (omp_get_wtime 以外の) omp_get_* を (A) 含まない (B) 含む
