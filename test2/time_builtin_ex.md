# ベンチマークが標準出力に出力する値

| 名称 | cuda | sycl | acc | omp_nvc  | 単位 | plot |
|  --  |  --  | --   |  -- |   --     | --   | -- |
| accuracy | 6.02e+02 | 6.07e+02 | 6.89e+02 | 8.27e+02 | us |![accuracy](SVGs_builtin/accuracy.svg) |
| ace | 2.35e+03 | 2.66e+03 | 2.34e+03 | 2.52e+03 | ms |![ace](SVGs_builtin/ace.svg) |
| adam | 6.02e-02 | 6.22e-02 | 7.60e-02 | 1.06e-01 | ms |![adam](SVGs_builtin/adam.svg) |
| adamw | 5.17e-02 | 5.44e-02 | 9.98e-02 | 6.95e-01 | ms |![adamw](SVGs_builtin/adamw.svg) |
| adjacent | 2.40e+03 | 2.43e+03 | 3.04e+03 | 6.03e+03 | us |![adjacent](SVGs_builtin/adjacent.svg) |
| adv | 3.82e+06 | 5.35e+06 | | 6.92e+06 | ns |![adv](SVGs_builtin/adv.svg) |
| aes | 3.14e-04 | 2.98e-04 | | 4.23e-04 | s |![aes](SVGs_builtin/aes.svg) |
| affine | 3.13e-06 | 2.92e-06 | 1.02e-05 | 9.82e-06 | s |![affine](SVGs_builtin/affine.svg) |
| aidw | 7.70e-03 | 4.73e-02 | 1.95e-02 | 9.10e-03 | s |![aidw](SVGs_builtin/aidw.svg) |
| aligned-types | 6.29e-01 | 7.27e-01 | 1.47e+00 | 1.69e+00 | ms |![aligned-types](SVGs_builtin/aligned-types.svg) |
| all-pairs-distance | 2.12e+02 | 2.54e+02 | 2.16e+02 | 1.87e+03 | us |![all-pairs-distance](SVGs_builtin/all-pairs-distance.svg) |
| amgmk | 2.45e+00 | 2.39e+00 | 2.37e+00 | 2.38e+00 | s |![amgmk](SVGs_builtin/amgmk.svg) |
| ans | 5.14e+00 | 5.36e+00 | | | s |![ans](SVGs_builtin/ans.svg) |
| aobench | 8.90e+01 | 1.02e+02 | 7.88e+01 | 1.06e+02 | us |![aobench](SVGs_builtin/aobench.svg) |
| aop | 4.00e+00 | 4.01e+00 | | | ms |![aop](SVGs_builtin/aop.svg) |
| asmooth | 5.16e-03 | 5.67e-03 | 5.65e-03 | 5.80e-03 | s |![asmooth](SVGs_builtin/asmooth.svg) |
| assert | 2.43e+00 | 3.25e-02 | | 5.02e+00 | ns |![assert](SVGs_builtin/assert.svg) |
| asta | 1.53e-02 | | | 7.29e-02 | s |![asta](SVGs_builtin/asta.svg) |
| atan2 | 1.38e+02 | 1.41e+02 | 1.70e+02 | 1.89e+02 | us |![atan2](SVGs_builtin/atan2.svg) |
| atomicCost | 1.09e+04 | 1.08e+04 | 1.14e+04 | 1.42e+04 | us |![atomicCost](SVGs_builtin/atomicCost.svg) |
| atomicIntrinsics | 4.23e+05 | | 3.11e+04 | 3.38e+04 | us |![atomicIntrinsics](SVGs_builtin/atomicIntrinsics.svg) |
| atomicPerf | 6.61e+03 | 6.61e+03 | 6.69e+03 | 7.20e+03 | us |![atomicPerf](SVGs_builtin/atomicPerf.svg) |
| atomicReduction | 5.12e+04 | 4.71e+04 | 4.56e+04 | 4.59e+04 | GB/s |![atomicReduction](SVGs_builtin/atomicReduction.svg) |
| attention | 4.31e-01 | 4.31e-01 | 4.32e-01 | 9.30e-01 | ms |![attention](SVGs_builtin/attention.svg) |
| axhelm | 2.22e+07 | 2.17e+07 | | 2.18e+07 | ns |![axhelm](SVGs_builtin/axhelm.svg) |
| babelstream | 6.69e+06 | 6.66e+06 | 6.60e+06 | 6.04e+06 | s |![babelstream](SVGs_builtin/babelstream.svg) |
| background-subtract | 1.03e+02 | 1.06e+02 | 1.14e+02 | 1.63e+02 | us |![background-subtract](SVGs_builtin/background-subtract.svg) |
| backprop | 2.45e+00 | 2.37e+00 | 2.36e+00 | 2.39e+00 | s |![backprop](SVGs_builtin/backprop.svg) |
| bezier-surface | 8.80e+01 | 1.13e+02 | 3.19e+03 | 9.70e+01 | ms |![bezier-surface](SVGs_builtin/bezier-surface.svg) |
| bfs | 3.84e+02 | 5.31e+02 | 4.35e+02 | 1.27e+03 | us |![bfs](SVGs_builtin/bfs.svg) |
| bilateral | 4.86e+00 | 4.90e+00 | 2.03e+02 | 5.22e+00 | ms |![bilateral](SVGs_builtin/bilateral.svg) |
| binomial | 2.99e+03 | 2.78e+03 | | | ms |![binomial](SVGs_builtin/binomial.svg) |
| bitonic-sort | 3.29e+01 | 3.46e+01 | 3.74e+01 | 5.55e+01 | ms |![bitonic-sort](SVGs_builtin/bitonic-sort.svg) |
| black-scholes | 2.08e+03 | 2.24e+03 | 1.85e+03 | 2.25e+03 | ms |![black-scholes](SVGs_builtin/black-scholes.svg) |
| blas-gemm | 5.06e+03 | | | | us |![blas-gemm](SVGs_builtin/blas-gemm.svg) |
| bn | 1.62e-01 | 2.22e-01 | | | s |![bn](SVGs_builtin/bn.svg) |
| bonds | 3.76e+01 | 2.21e+01 | 5.30e+01 | 5.05e+01 | ms |![bonds](SVGs_builtin/bonds.svg) |
| boxfilter | 6.43e+02 | 5.97e+02 | 3.94e+02 | 7.06e+02 | us |![boxfilter](SVGs_builtin/boxfilter.svg) |
| bsearch | 3.12e+00 | 2.30e+00 | 2.73e-01 | 2.27e-01 | s |![bsearch](SVGs_builtin/bsearch.svg) |
| bspline-vgh | 1.32e-01 | 1.59e-01 | 1.41e-01 | 1.43e-01 | s |![bspline-vgh](SVGs_builtin/bspline-vgh.svg) |
| b+tree | 2.06e+02 | 3.34e+02 | 1.35e+03 | 1.39e+03 | us |![b+tree](SVGs_builtin/b+tree.svg) |
| burger | 5.62e-01 | 5.18e-01 | 6.35e-01 | 5.78e-01 | s |![burger](SVGs_builtin/burger.svg) |
| bwt | 1.58e+04 | 1.59e+04 | 1.21e+03 | | ms |![bwt](SVGs_builtin/bwt.svg) |
| car | 4.84e-03 | 4.55e-03 | 5.60e-03 | 5.65e-03 | s |![car](SVGs_builtin/car.svg) |
| cbsfil | 3.66e-02 | 3.43e-02 | 5.26e-02 | 3.64e-02 | s |![cbsfil](SVGs_builtin/cbsfil.svg) |
| ccsd-trpdrv | 4.20e-05 | 4.50e-05 | 4.90e-05 | 8.70e-05 | s |![ccsd-trpdrv](SVGs_builtin/ccsd-trpdrv.svg) |
| ccs | 1.09e-02 | 1.17e-02 | 8.89e-03 | 1.31e-02 | s |![ccs](SVGs_builtin/ccs.svg) |
| cfd | 8.72e-02 | 8.18e-02 | 3.01e-01 | 4.22e-01 | s |![cfd](SVGs_builtin/cfd.svg) |
| chacha20 | 2.25e+01 | 2.28e+01 | 6.75e+01 | 4.18e+01 | us |![chacha20](SVGs_builtin/chacha20.svg) |
| channelShuffle | 2.40e+01 | 2.43e+01 | 3.85e+01 | 3.83e+01 | ms |![channelShuffle](SVGs_builtin/channelShuffle.svg) |
| channelSum | 6.26e+01 | 6.27e+01 | 5.93e+01 | 8.76e+01 | ms |![channelSum](SVGs_builtin/channelSum.svg) |
| chemv | | 6.82e+02 | 3.50e+02 | 7.67e+02 | us |![chemv](SVGs_builtin/chemv.svg) |
| che | 7.66e-01 | 6.09e-01 | 8.56e-01 | 9.08e-01 | s |![che](SVGs_builtin/che.svg) |
| chi2 | | 1.47e-02 | 1.40e-02 | 1.40e-02 | s |![chi2](SVGs_builtin/chi2.svg) |
| clenergy | 2.10e-01 | 2.23e-01 | 7.12e-01 | 7.32e-01 | s |![clenergy](SVGs_builtin/clenergy.svg) |
| clink | 3.20e+01 | 3.58e+01 | 4.99e+01 | 4.28e+01 | ms |![clink](SVGs_builtin/clink.svg) |
| cmp | | | | |  |![cmp](SVGs_builtin/cmp.svg) |
| cm | | | | |  |![cm](SVGs_builtin/cm.svg) |
| cobahh | 4.61e+04 | 3.44e+04 | 4.96e+04 | 1.40e+04 | us |![cobahh](SVGs_builtin/cobahh.svg) |
| colorwheel | 1.25e+02 | 1.24e+02 | 1.23e+02 | 1.22e+02 | ms |![colorwheel](SVGs_builtin/colorwheel.svg) |
| columnarSolver | 7.30e+00 | 7.76e+00 | 1.47e+01 | 7.75e+00 | s |![columnarSolver](SVGs_builtin/columnarSolver.svg) |
| complex | 1.02e-02 | 1.04e-02 | 1.01e-02 | 1.04e-02 | s |![complex](SVGs_builtin/complex.svg) |
| compute-score | 6.12e-01 | 6.14e-01 | 7.00e-01 | 1.79e+00 | ms |![compute-score](SVGs_builtin/compute-score.svg) |
| concat | 2.98e+03 | 2.97e+03 | 3.06e+03 | 3.67e+03 | us |![concat](SVGs_builtin/concat.svg) |
| contract | 6.65e+00 | 8.82e+00 | 2.63e+01 | 8.29e+00 | s |![contract](SVGs_builtin/contract.svg) |
| conversion | 1.74e-01 | 1.69e-01 | 9.22e-02 | 7.96e-02 | , |![conversion](SVGs_builtin/conversion.svg) |
| convolution1D | 1.42e+05 | 1.72e+05 | 3.11e+05 | 1.58e+06 | us |![convolution1D](SVGs_builtin/convolution1D.svg) |
| convolution3D | 3.16e+01 | 3.53e+01 | 1.22e+03 | 2.76e+01 | us |![convolution3D](SVGs_builtin/convolution3D.svg) |
| convolutionSeparable | 2.30e-03 | 2.30e-03 | | 1.13e-02 | s |![convolutionSeparable](SVGs_builtin/convolutionSeparable.svg) |
| cooling | 5.13e+02 | 3.97e+02 | 4.19e+02 | 3.80e+02 | ms |![cooling](SVGs_builtin/cooling.svg) |
| crc64 | 4.77e+04 | 5.24e+04 | 5.54e+04 | 5.60e+04 | MB/s |![crc64](SVGs_builtin/crc64.svg) |
| cross | 6.96e+03 | 6.96e+03 | 7.00e+03 | 7.00e+03 | us |![cross](SVGs_builtin/cross.svg) |
| crs | 6.64e-02 | 7.09e-02 | | | s |![crs](SVGs_builtin/crs.svg) |
| d2q9-bgk | 5.83e+00 | 5.53e+00 | | 2.45e+01 | us |![d2q9-bgk](SVGs_builtin/d2q9-bgk.svg) |
| damage | 1.89e-02 | 1.99e-02 | 3.22e-02 | 2.39e-01 | s|ms|us |![damage](SVGs_builtin/damage.svg) |
| dct8x8 | 9.43e-03 | 9.43e-03 | | | s |![dct8x8](SVGs_builtin/dct8x8.svg) |
| ddbp | 3.42e-01 | | | 2.63e-01 | s |![ddbp](SVGs_builtin/ddbp.svg) |
| debayer | 3.62e-03 | 3.68e-03 | 7.34e-02 | 8.39e-02 | s |![debayer](SVGs_builtin/debayer.svg) |
| degrid | 1.29e-02 | 1.28e-02 | 1.01e-01 | 3.28e-02 | s |![degrid](SVGs_builtin/degrid.svg) |
| dense-embedding | 2.37e+04 | 2.37e+04 | 2.35e+04 | 3.88e+04 | us |![dense-embedding](SVGs_builtin/dense-embedding.svg) |
| depixel | 1.01e+02 | 1.52e+02 | 2.04e+02 | 1.63e+02 | s |![depixel](SVGs_builtin/depixel.svg) |
| deredundancy | 1.62e+01 | 1.55e+01 | 1.57e+01 | 1.57e+01 | s |![deredundancy](SVGs_builtin/deredundancy.svg) |
| diamond | 3.67e+01 | 1.61e+01 | | 3.72e+01 | s |![diamond](SVGs_builtin/diamond.svg) |
| distort | 1.24e-02 | 7.93e-03 | 2.20e-02 | 1.97e-02 | ms |![distort](SVGs_builtin/distort.svg) |
| divergence | 1.09e+03 | 1.44e+03 | 3.26e+02 | 3.39e+02 | ms |![divergence](SVGs_builtin/divergence.svg) |
| doh | 6.92e+00 | 7.75e+00 | 1.39e+01 | 2.32e+01 | us |![doh](SVGs_builtin/doh.svg) |
| dp | 2.11e+01 | | 2.02e+01 | 2.04e+01 | ms |![dp](SVGs_builtin/dp.svg) |
| dslash | 4.09e-02 | 5.08e-02 | 6.24e-02 | 6.45e-02 | s |![dslash](SVGs_builtin/dslash.svg) |
| dxtc2 | 3.16e+02 | 3.61e+02 | | | us |![dxtc2](SVGs_builtin/dxtc2.svg) |
| easyWave | 2.14e+04 | 2.05e+04 | 1.94e+04 | 2.07e+04 | ms |![easyWave](SVGs_builtin/easyWave.svg) |
| ecdh | 1.19e-01 | 8.15e-02 | 1.50e-01 | 1.64e-01 | s |![ecdh](SVGs_builtin/ecdh.svg) |
| eigenvalue | 2.04e+01 | 2.43e+01 | | 2.34e+01 | us |![eigenvalue](SVGs_builtin/eigenvalue.svg) |
| entropy | 4.86e-02 | 5.30e-02 | 5.17e-01 | 7.12e-02 | s |![entropy](SVGs_builtin/entropy.svg) |
| epistasis | 2.01e-02 | 1.99e-02 | 1.52e+00 | 2.47e-02 | s |![epistasis](SVGs_builtin/epistasis.svg) |
| expdist | 3.32e-03 | 3.29e-03 | | 8.57e-03 | s |![expdist](SVGs_builtin/expdist.svg) |
| extend2 | 2.60e+03 | 1.10e+03 | 2.05e+03 | 2.18e+03 | us |![extend2](SVGs_builtin/extend2.svg) |
| extrema | 2.54e-01 | 3.09e-01 | 9.44e-02 | 1.09e-01 | s |![extrema](SVGs_builtin/extrema.svg) |
| face | 2.64e+00 | 2.56e+00 | 2.37e+00 | 2.11e-01 | s |![face](SVGs_builtin/face.svg) |
| fdtd3d | 4.69e-04 | 3.96e-04 | | 4.78e-03 | s |![fdtd3d](SVGs_builtin/fdtd3d.svg) |
| feynman-kac | 7.87e+00 | 3.94e+00 | 4.58e+00 | 1.98e+00 | s |![feynman-kac](SVGs_builtin/feynman-kac.svg) |
| fft | 3.55e-04 | 6.14e-04 | | 1.15e-03 | s |![fft](SVGs_builtin/fft.svg) |
| fhd | | 1.79e+02 | | | s |![fhd](SVGs_builtin/fhd.svg) |
| filter | 6.17e+00 | | | 7.69e+01 | ms |![filter](SVGs_builtin/filter.svg) |
| flip | 4.12e+01 | 4.72e+01 | 6.36e+01 | 7.60e+01 | ms |![flip](SVGs_builtin/flip.svg) |
| floydwarshall | 2.02e-01 | 2.08e-01 | 2.39e-01 | 3.47e-01 | s |![floydwarshall](SVGs_builtin/floydwarshall.svg) |
| fluidSim | 1.20e-05 | 8.00e-06 | 2.20e-05 | 2.90e-05 | s |![fluidSim](SVGs_builtin/fluidSim.svg) |
| fpc | 1.19e-02 | 1.21e-02 | | | s |![fpc](SVGs_builtin/fpc.svg) |
| fpdc | 6.77e-04 | 6.42e-04 | | | s |![fpdc](SVGs_builtin/fpdc.svg) |
| fresnel | 3.13e-03 | 1.49e-03 | 8.27e-03 | 8.23e-03 | s |![fresnel](SVGs_builtin/fresnel.svg) |
| frna | 7.78e+01 | 8.04e+01 | | | s |![frna](SVGs_builtin/frna.svg) |
| fwt | 4.81e-04 | 4.84e-04 | 7.83e-04 | 1.56e-03 | s |![fwt](SVGs_builtin/fwt.svg) |
| gabor | 7.95e+03 | 8.18e+03 | 1.01e+04 | 1.11e+04 | us |![gabor](SVGs_builtin/gabor.svg) |
| gamma-correction | 5.21e-04 | 6.17e-04 | 4.96e-04 | 1.79e-03 | s |![gamma-correction](SVGs_builtin/gamma-correction.svg) |
| ga | 2.36e+00 | 2.81e+00 | 2.85e+00 | 2.68e+00 | s |![ga](SVGs_builtin/ga.svg) |
| gaussian | 4.61e+05 | 3.96e+05 | 3.72e+05 | 3.82e+05 | us |![gaussian](SVGs_builtin/gaussian.svg) |
| gc | 6.20e-05 | | | | s |![gc](SVGs_builtin/gc.svg) |
| gd | 3.99e-01 | 5.10e-01 | 5.42e-01 | 7.87e-01 | s |![gd](SVGs_builtin/gd.svg) |
| geglu | 7.58e+03 | 1.83e+04 | 1.97e+04 | 2.02e+04 | us |![geglu](SVGs_builtin/geglu.svg) |
| geodesic | 2.89e+02 | 2.60e+02 | 2.86e+02 | 2.96e+02 | us |![geodesic](SVGs_builtin/geodesic.svg) |
| glu | 1.86e+04 | 1.52e+04 | 1.90e+04 | 2.06e+04 | us |![glu](SVGs_builtin/glu.svg) |
| gmm | 3.26e+00 | 3.58e+00 | | 4.23e+00 | s |![gmm](SVGs_builtin/gmm.svg) |
| goulash | 1.25e+00 | 1.25e+00 | 1.26e+00 | 1.75e+00 | s |![goulash](SVGs_builtin/goulash.svg) |
| gpp | 2.42e-01 | 1.83e-01 | 1.67e-01 | 1.72e-01 | s |![gpp](SVGs_builtin/gpp.svg) |
| grep | | | | |  |![grep](SVGs_builtin/grep.svg) |
| grrt | 1.01e+01 | 1.05e+01 | 9.26e+00 | 8.74e+00 | s |![grrt](SVGs_builtin/grrt.svg) |
| haccmk | 2.15e-03 | 1.91e-03 | 4.60e-05 | 1.94e-03 | s |![haccmk](SVGs_builtin/haccmk.svg) |
| hausdorff | 6.49e+00 | 6.76e+00 | 1.01e+01 | 7.26e+00 | ms |![hausdorff](SVGs_builtin/hausdorff.svg) |
| haversine | 2.09e-04 | 2.35e-04 | | | s |![haversine](SVGs_builtin/haversine.svg) |
| heartwall | 7.42e+00 | | | | s |![heartwall](SVGs_builtin/heartwall.svg) |
| heat2d | 1.22e+03 | 1.18e+03 | 7.72e+02 | 6.89e+02 | GFLOPS |![heat2d](SVGs_builtin/heat2d.svg) |
| heat | 2.29e+01 | 2.20e+01 | 1.53e+01 | 1.61e+01 | s |![heat](SVGs_builtin/heat.svg) |
| hellinger | 2.87e-03 | 1.62e-03 | 2.73e-02 | 3.79e-03 | s |![hellinger](SVGs_builtin/hellinger.svg) |
| henry | 1.25e-03 | 1.14e-03 | 4.64e-04 | 4.28e-04 | s |![henry](SVGs_builtin/henry.svg) |
| hexciton | 1.38e+00 | 3.27e+00 | | 2.13e+00 | s |![hexciton](SVGs_builtin/hexciton.svg) |
| histogram | 3.32e+02 | 2.93e+02 | | 7.93e+02 | us  |![histogram](SVGs_builtin/histogram.svg) |
| hmm | 1.39e-01 | 1.44e-01 | 2.83e-01 | 1.90e-01 | s |![hmm](SVGs_builtin/hmm.svg) |
| hogbom | 2.60e-01 | 1.40e-01 | 1.60e-01 | 3.10e-01 | ms |![hogbom](SVGs_builtin/hogbom.svg) |
| hotspot3D | 7.44e+00 | 7.55e+00 | 1.73e+01 | 1.66e+01 | us |![hotspot3D](SVGs_builtin/hotspot3D.svg) |
| hwt1d | 6.11e-04 | 6.22e-04 | 6.34e-04 | 7.06e-04 | s |![hwt1d](SVGs_builtin/hwt1d.svg) |
| hybridsort | 2.22e+02 | 5.28e+01 | | 2.25e+02 | ms |![hybridsort](SVGs_builtin/hybridsort.svg) |
| hypterm | 2.30e+00 | 2.61e+00 | 2.39e+00 | 2.42e+00 | ms |![hypterm](SVGs_builtin/hypterm.svg) |
| idivide | 1.08e-02 | 4.31e-03 | 7.27e-03 | 1.82e-02 | s |![idivide](SVGs_builtin/idivide.svg) |
| interleave | 1.47e-02 | 1.47e-02 | 1.40e-02 | 7.96e-03 | s |![interleave](SVGs_builtin/interleave.svg) |
| interval | 4.33e+01 | 3.86e+01 | 1.02e+02 | 7.41e+01 | us |![interval](SVGs_builtin/interval.svg) |
| inversek2j | 2.71e+00 | 7.76e+00 | 2.25e+03 | 1.11e+03 | us |![inversek2j](SVGs_builtin/inversek2j.svg) |
| ising | 3.09e-01 | 3.50e-01 | 3.89e-01 | 4.86e-01 | s |![ising](SVGs_builtin/ising.svg) |
| iso2dfd | 2.40e+01 | 2.44e+01 | 2.24e+01 | 3.20e+01 | ms |![iso2dfd](SVGs_builtin/iso2dfd.svg) |
| jacobi | 2.74e+00 | 2.72e+00 | 3.51e+00 | 3.44e+00 | s |![jacobi](SVGs_builtin/jacobi.svg) |
| jenkins-hash | 3.10e-03 | 3.59e-03 | 2.34e-01 | 2.28e-01 | s |![jenkins-hash](SVGs_builtin/jenkins-hash.svg) |
| kalman | 3.22e-01 | | | | s |![kalman](SVGs_builtin/kalman.svg) |
| keccaktreehash | 2.91e+07 | 3.15e+07 | 7.36e+05 | 2.86e+07 | kB/s |![keccaktreehash](SVGs_builtin/keccaktreehash.svg) |
| keogh | 4.56e-03 | 5.05e-03 | 5.09e-02 | 1.52e-01 | s |![keogh](SVGs_builtin/keogh.svg) |
| kernelLaunch | 1.83e+00 | 2.07e+00 | 1.12e+02 | 1.24e+01 | us |![kernelLaunch](SVGs_builtin/kernelLaunch.svg) |
| kmeans | 2.79e+04 | 2.83e+04 | 2.64e+04 | 2.21e+04 | ms |![kmeans](SVGs_builtin/kmeans.svg) |
| knn | 6.45e-01 | 4.69e-01 | 5.92e-01 | 7.78e-01 | s |![knn](SVGs_builtin/knn.svg) |
| lanczos | 3.14e-01 | 3.12e-01 | | 4.95e-01 | s |![lanczos](SVGs_builtin/lanczos.svg) |
| langevin | 6.30e-03 | 6.19e-03 | 6.31e-03 | 9.40e-03 | s |![langevin](SVGs_builtin/langevin.svg) |
| langford | 1.24e-01 | 1.21e-01 | | | s |![langford](SVGs_builtin/langford.svg) |
| laplace3d | 1.01e-02 | 1.03e-02 | 1.80e-02 | 1.95e-02 | s |![laplace3d](SVGs_builtin/laplace3d.svg) |
| laplace | 1.16e+00 | 8.27e+00 | 4.94e+00 | 4.61e+00 | s |![laplace](SVGs_builtin/laplace.svg) |
| lavaMD | 4.99e+00 | 5.83e+00 | | 1.82e+01 | s |![lavaMD](SVGs_builtin/lavaMD.svg) |
| layout | 1.33e+02 | 1.41e+02 | 9.07e+01 | 8.85e+01 | us |![layout](SVGs_builtin/layout.svg) |
| lci | 1.53e+01 | 7.50e+00 | 4.28e+00 | 4.28e+00 | s |![lci](SVGs_builtin/lci.svg) |
| lda | 6.31e-03 | 6.53e-03 | | | s |![lda](SVGs_builtin/lda.svg) |
| ldpc | 1.19e+00 | 1.30e+00 | | 5.92e+00 | s |![ldpc](SVGs_builtin/ldpc.svg) |
| lebesgue | 2.66e+00 | 1.75e-01 | 2.53e+00 | 2.47e+00 | s |![lebesgue](SVGs_builtin/lebesgue.svg) |
| leukocyte | 5.89e-01 | | | 8.64e-01 | s |![leukocyte](SVGs_builtin/leukocyte.svg) |
| libor | 2.99e-03 | 3.21e-03 | 1.07e-01 | 3.30e-03 | s |![libor](SVGs_builtin/libor.svg) |
| lid-driven-cavity | 7.56e+00 | 8.37e+00 | | 1.98e+01 | s |![lid-driven-cavity](SVGs_builtin/lid-driven-cavity.svg) |
| lif | 1.37e+04 | 1.27e+04 | 1.70e+04 | 1.09e+04 | us |![lif](SVGs_builtin/lif.svg) |
| linearprobing | 1.76e-03 | 1.78e-03 | | | s |![linearprobing](SVGs_builtin/linearprobing.svg) |
| log2 | 2.02e+03 | 1.84e+03 | 2.41e+03 | 2.68e+03 | us |![log2](SVGs_builtin/log2.svg) |
| lombscargle | 3.05e+02 | 1.80e+02 | 2.02e+02 | 2.17e+02 | us |![lombscargle](SVGs_builtin/lombscargle.svg) |
| loopback | 3.26e-02 | 2.90e-02 | 2.98e-02 | 5.74e-02 | s |![loopback](SVGs_builtin/loopback.svg) |
| lrn | 4.28e-01 | 6.49e-01 | 5.27e-01 | 6.42e-01 | s |![lrn](SVGs_builtin/lrn.svg) |
| lr | 7.56e+00 | 7.51e+00 | 2.50e+01 | 5.51e+01 | us |![lr](SVGs_builtin/lr.svg) |
| lsqt | 1.30e+01 | 1.28e+01 | 1.77e+01 | 1.74e+01 | s |![lsqt](SVGs_builtin/lsqt.svg) |
| lud | 5.07e+00 | 5.72e+00 | 7.03e+00 | 4.36e+01 | s |![lud](SVGs_builtin/lud.svg) |
| lulesh | 4.59e+00 | 5.58e+00 | 8.48e+00 | 5.18e+00 | s |![lulesh](SVGs_builtin/lulesh.svg) |
| mallocFree | 2.53e+06 | 3.07e+03 | 2.43e+06 | 5.09e+02 | us |![mallocFree](SVGs_builtin/mallocFree.svg) |
| mandelbrot | 5.78e-02 | 6.43e-02 | 5.88e-02 | 6.16e-02 | ms |![mandelbrot](SVGs_builtin/mandelbrot.svg) |
| mask | 2.79e+04 | 2.75e+04 | 2.74e+04 | 2.78e+04 | us |![mask](SVGs_builtin/mask.svg) |
| match | 2.38e+02 | 2.51e+02 | | 3.57e+11 | ms |![match](SVGs_builtin/match.svg) |
| matern | 1.81e+04 | 1.92e+04 | 6.98e+03 | 1.45e+05 | us |![matern](SVGs_builtin/matern.svg) |
| matrix-rotate | 7.03e-02 | 7.02e-02 | 7.14e-02 | 6.75e-02 | s |![matrix-rotate](SVGs_builtin/matrix-rotate.svg) |
| maxFlops | 4.61e+00 | 4.62e+00 | 4.80e+00 | 4.59e+00 | s |![maxFlops](SVGs_builtin/maxFlops.svg) |
| maxpool3d | 3.94e-03 | 4.15e-03 | 5.74e-03 | 6.76e-03 | s |![maxpool3d](SVGs_builtin/maxpool3d.svg) |
| mcmd | 1.23e+01 | 8.96e+00 | 1.07e+01 | 1.10e+01 | s |![mcmd](SVGs_builtin/mcmd.svg) |
| mcpr | 2.93e-02 | 5.56e-03 | 1.28e-01 | 2.83e-02 | s |![mcpr](SVGs_builtin/mcpr.svg) |
| md5hash | 1.44e+04 | 1.39e+04 | 1.44e+04 | 1.45e+04 | ms |![md5hash](SVGs_builtin/md5hash.svg) |
| mdh | 8.79e-02 | 2.39e+00 | 1.74e+00 | 6.86e-02 |  |![mdh](SVGs_builtin/mdh.svg) |
| md | 6.56e-05 | 6.64e-05 | 7.49e-05 | 7.23e-05 | s |![md](SVGs_builtin/md.svg) |
| meanshift | 4.41e-01 | 8.77e-01 | 1.38e+00 | 9.38e-01 | ms |![meanshift](SVGs_builtin/meanshift.svg) |
| medianfilter | 7.60e-05 | | 8.50e-05 | 2.40e-04 | s |![medianfilter](SVGs_builtin/medianfilter.svg) |
| memcpy | 1.82e+02 | 2.00e+02 | 2.49e+02 | 2.49e+02 | us |![memcpy](SVGs_builtin/memcpy.svg) |
| memtest | 3.25e-02 | 3.46e-02 | 4.32e-02 | 4.31e-02 | s |![memtest](SVGs_builtin/memtest.svg) |
| merge | 6.73e+04 | 6.65e+04 | | 1.62e+05 | us |![merge](SVGs_builtin/merge.svg) |
| metropolis | 9.72e+00 | 1.11e+01 | | 6.14e+01 | s |![metropolis](SVGs_builtin/metropolis.svg) |
| michalewicz | 1.09e+05 | 1.05e+06 | 1.45e+05 | 1.90e+05 | us |![michalewicz](SVGs_builtin/michalewicz.svg) |
| minibude | 7.66e+02 | 9.10e+02 | 8.43e+02 | 1.25e+03 | ms |![minibude](SVGs_builtin/minibude.svg) |
| miniFE | | | | |  |![miniFE](SVGs_builtin/miniFE.svg) |
| minimap2 | 4.20e-02 | 5.82e-03 | 1.13e-02 | 1.09e-01 | s |![minimap2](SVGs_builtin/minimap2.svg) |
| minisweep | 5.03e+01 | 5.41e+01 | 3.29e+01 | 3.34e+01 | s |![minisweep](SVGs_builtin/minisweep.svg) |
| miniWeather | 6.96e-01 | 6.94e-01 | 3.00e+01 | 2.99e+01 | s |![miniWeather](SVGs_builtin/miniWeather.svg) |
| minkowski | 2.31e-03 | 2.65e-03 | 2.22e-03 | 2.20e-03 | s |![minkowski](SVGs_builtin/minkowski.svg) |
| mis | 2.17e-04 | 1.07e-04 | 0.00e+00 | 0.00e+00 | s |![mis](SVGs_builtin/mis.svg) |
| mixbench | 6.06e-01 | 5.55e-01 | 6.31e-01 | 6.67e-01 | s |![mixbench](SVGs_builtin/mixbench.svg) |
| morphology | 2.23e-01 | 2.24e-01 | | | s |![morphology](SVGs_builtin/morphology.svg) |
| mrc | 1.38e+04 | 1.38e+04 | 1.39e+04 | 1.41e+04 | us |![mrc](SVGs_builtin/mrc.svg) |
| mriQ | 2.60e+00 | 3.24e+00 | 3.29e+00 | 3.30e+00 | s |![mriQ](SVGs_builtin/mriQ.svg) |
| mr | 2.25e+00 | 1.12e+00 | | 2.30e+00 | ms |![mr](SVGs_builtin/mr.svg) |
| mt | 2.95e-03 | 1.88e-03 | 1.67e-03 | 1.68e-03 | s |![mt](SVGs_builtin/mt.svg) |
| multimaterial | 1.14e+02 | 2.46e+00 | 3.15e+00 | 4.37e+00 | ms |![multimaterial](SVGs_builtin/multimaterial.svg) |
| murmurhash3 | 3.97e-03 | 3.88e-03 | 3.98e-03 | 3.91e-03 | s |![murmurhash3](SVGs_builtin/murmurhash3.svg) |
| myocyte | 1.49e+01 | | 3.26e+00 | 4.00e+00 | s |![myocyte](SVGs_builtin/myocyte.svg) |
| nbody | 3.54e-02 | 3.35e-02 | 2.35e+00 | 2.40e+00 |  |![nbody](SVGs_builtin/nbody.svg) |
| ne | 9.06e-04 | 1.18e-03 | 1.33e-03 | 1.61e-03 | s |![ne](SVGs_builtin/ne.svg) |
| nlll | 8.41e+01 | 9.41e+01 | 1.33e+04 | 1.00e+02 | us |![nlll](SVGs_builtin/nlll.svg) |
| nms | 7.40e-05 | 1.36e-04 | | 2.18e-04 | s |![nms](SVGs_builtin/nms.svg) |
| nn | 2.43e+00 | 2.42e+00 | 7.62e+00 | 8.28e+00 | us |![nn](SVGs_builtin/nn.svg) |
| norm2 | 2.05e+03 | | 2.91e+03 | 1.59e+04 | us |![norm2](SVGs_builtin/norm2.svg) |
| nqueen | 3.85e-02 | 4.50e-02 | 2.14e-01 | 4.39e-02 | s |![nqueen](SVGs_builtin/nqueen.svg) |
| ntt | 3.62e+01 | 4.16e+01 | | 4.19e+01 | us |![ntt](SVGs_builtin/ntt.svg) |
| nw | 4.97e-02 | 5.02e-02 | | 1.23e-01 | s |![nw](SVGs_builtin/nw.svg) |
| openmp | 2.55e+00 | 9.79e-02 | 2.22e-01 | 2.29e-01 | s |![openmp](SVGs_builtin/openmp.svg) |
| overlay | 5.51e-01 | 5.85e-01 | 3.44e+00 | 4.89e+00 | s |![overlay](SVGs_builtin/overlay.svg) |
| p4 | 8.25e+01 | 8.52e+01 | | 1.39e+03 | us |![p4](SVGs_builtin/p4.svg) |
| page-rank | 3.93e+00 | 3.84e+00 | 4.09e+00 | 4.36e+00 | s |![page-rank](SVGs_builtin/page-rank.svg) |
| particle-diffusion | 2.20e-03 | 1.70e-03 | 2.29e-03 | 2.27e-03 | s |![particle-diffusion](SVGs_builtin/particle-diffusion.svg) |
| particlefilter | 2.22e+02 | 1.98e+02 | | 2.20e+02 | s |![particlefilter](SVGs_builtin/particlefilter.svg) |
| particles | 1.07e+00 | 9.94e-01 | | | s |![particles](SVGs_builtin/particles.svg) |
| pathfinder | 3.88e+01 | 3.88e+01 | 4.35e+01 | 7.99e+02 | s |![pathfinder](SVGs_builtin/pathfinder.svg) |
| permutate | 2.05e+00 | 2.42e+00 | | 2.26e+00 | s |![permutate](SVGs_builtin/permutate.svg) |
| permute | 2.00e+01 | 2.22e+01 | 2.23e+01 | 2.28e+01 | ms |![permute](SVGs_builtin/permute.svg) |
| perplexity | 1.64e+00 | 2.24e+00 | 1.53e+00 | 1.69e+00 | s |![perplexity](SVGs_builtin/perplexity.svg) |
| phmm | 1.16e+03 | 1.10e+03 | 4.62e+03 | 5.23e+03 | millis |![phmm](SVGs_builtin/phmm.svg) |
| pnpoly | 4.59e-02 | 5.17e-02 | | 7.17e-01 | s |![pnpoly](SVGs_builtin/pnpoly.svg) |
| pns | 5.44e+00 | 5.38e+00 | | | s |![pns](SVGs_builtin/pns.svg) |
| pointwise | 9.21e-02 | 1.14e-01 | 1.80e-01 | 1.66e-01 | s |![pointwise](SVGs_builtin/pointwise.svg) |
| pool | 8.88e-03 | 9.02e-03 | 1.83e-02 | 1.57e-02 | s |![pool](SVGs_builtin/pool.svg) |
| popcount | 1.71e+05 | 8.28e+04 | 2.00e+05 | 7.06e+04 | us |![popcount](SVGs_builtin/popcount.svg) |
| present | 3.81e+01 | 3.88e+01 | 3.70e+01 | 9.36e+01 | us |![present](SVGs_builtin/present.svg) |
| prna | | | | |  |![prna](SVGs_builtin/prna.svg) |
| projectile | 1.13e-04 | 1.13e-04 | 1.18e-04 | 1.19e-04 | s |![projectile](SVGs_builtin/projectile.svg) |
| pso | 2.55e+01 | 2.44e+01 | 2.48e+01 | 2.19e+01 | us |![pso](SVGs_builtin/pso.svg) |
| qrg | 4.48e+01 | 3.66e+01 | 5.93e+01 | 7.56e+01 | us |![qrg](SVGs_builtin/qrg.svg) |
| qtclustering | 3.31e+01 | 3.20e+01 | | | s |![qtclustering](SVGs_builtin/qtclustering.svg) |
| quantBnB | 6.09e+03 | 6.66e+03 | 2.74e+03 | 3.26e+03 | us |![quantBnB](SVGs_builtin/quantBnB.svg) |
| quicksort | 6.92e+03 | 6.87e+03 | | 2.53e+03 | ms |![quicksort](SVGs_builtin/quicksort.svg) |
| radixsort | 5.06e-04 | 5.06e-04 | | 5.71e-03 | s |![radixsort](SVGs_builtin/radixsort.svg) |
| rainflow | 1.71e+05 | 1.50e+05 | 1.69e+05 | 1.27e+05 | us |![rainflow](SVGs_builtin/rainflow.svg) |
| randomAccess | 1.58e-01 | 1.58e-01 | | 1.92e+00 | s |![randomAccess](SVGs_builtin/randomAccess.svg) |
| reaction | 1.63e+00 | 1.65e+00 | | 6.75e+00 | s |![reaction](SVGs_builtin/reaction.svg) |
| recursiveGaussian | 1.32e-03 | 1.48e-03 | | 1.67e-03 | s |![recursiveGaussian](SVGs_builtin/recursiveGaussian.svg) |
| resize | 6.98e+02 | 7.58e+02 | 7.14e+02 | 8.35e+02 | us |![resize](SVGs_builtin/resize.svg) |
| reverse | 9.29e-01 | 1.25e+00 | 4.25e+00 | 5.49e+00 | s |![reverse](SVGs_builtin/reverse.svg) |
| rfs | 8.78e-03 | 8.78e-03 | 8.79e-03 | 8.79e-03 | s |![rfs](SVGs_builtin/rfs.svg) |
| rng-wallace | 1.25e+02 | 1.18e+02 | | 1.12e+03 | us |![rng-wallace](SVGs_builtin/rng-wallace.svg) |
| rodrigues | 3.10e+04 | 3.50e+04 | 3.49e+04 | 2.01e+04 | us |![rodrigues](SVGs_builtin/rodrigues.svg) |
| romberg | 2.85e-04 | 2.92e-04 | 3.51e-04 | 3.26e-04 | s |![romberg](SVGs_builtin/romberg.svg) |
| rsbench | 2.94e+00 | 1.22e+00 | 9.66e+00 | 2.91e+00 | s |![rsbench](SVGs_builtin/rsbench.svg) |
| rsc | 1.55e+02 | 1.31e+02 | 2.17e+02 | 2.28e+02 | ms |![rsc](SVGs_builtin/rsc.svg) |
| rtm8 | 1.02e-03 | 1.06e-03 | 8.59e-04 | 8.90e-04 | s |![rtm8](SVGs_builtin/rtm8.svg) |
| rushlarsen | 2.29e-01 | 3.24e-01 | 2.32e-01 | 2.45e-01 | s |![rushlarsen](SVGs_builtin/rushlarsen.svg) |
| s3d | 2.86e+00 | 4.80e-01 | | 2.84e+00 | s |![s3d](SVGs_builtin/s3d.svg) |
| s8n | 1.16e+01 | 1.33e+01 | 4.06e+01 | 5.00e+01 | us |![s8n](SVGs_builtin/s8n.svg) |
| sad | 4.00e+01 | 4.20e+01 | 4.84e+02 | 8.18e+04 | ms |![sad](SVGs_builtin/sad.svg) |
| sampling | 2.39e+02 | 2.81e+02 | 3.67e+02 | 5.63e+02 | us |![sampling](SVGs_builtin/sampling.svg) |
| scan2 | 3.55e+02 | 4.19e+02 | | 4.32e+03 | us |![scan2](SVGs_builtin/scan2.svg) |
| scan | 1.75e+04 | 1.87e+04 | | 2.05e+05 | us |![scan](SVGs_builtin/scan.svg) |
| scatterAdd | 3.91e+03 | 4.85e+03 | 1.61e+05 | 1.61e+05 | us |![scatterAdd](SVGs_builtin/scatterAdd.svg) |
| scel | 2.92e+03 | 2.93e+03 | 4.27e+04 | 6.24e+03 | us |![scel](SVGs_builtin/scel.svg) |
| secp256k1 | 2.91e-03 | 2.77e-03 | 2.92e-03 | 2.87e-03 | s |![secp256k1](SVGs_builtin/secp256k1.svg) |
| sheath | 3.18e+00 | 3.42e+00 | 3.36e+00 | 3.40e+00 | ms |![sheath](SVGs_builtin/sheath.svg) |
| shmembench | 2.70e+00 | 6.74e+00 | | | ms |![shmembench](SVGs_builtin/shmembench.svg) |
| simplemoc | 1.76e+02 | 3.42e+02 | 2.35e+02 | 2.35e+02 | s |![simplemoc](SVGs_builtin/simplemoc.svg) |
| simpleSpmv | 4.46e+01 | 3.57e+01 | 2.28e+01 | 4.70e+01 | ms |![simpleSpmv](SVGs_builtin/simpleSpmv.svg) |
| slu | | | | | ms |![slu](SVGs_builtin/slu.svg) |
| snake | 4.39e+03 | 4.85e+03 | 7.15e+03 | 7.15e+03 | us |![snake](SVGs_builtin/snake.svg) |
| sobel | 4.80e+00 | 4.85e+00 | 1.01e+01 | 1.11e+01 | us |![sobel](SVGs_builtin/sobel.svg) |
| sobol | 1.22e-03 | 1.28e-03 | | 3.35e-03 | s |![sobol](SVGs_builtin/sobol.svg) |
| softmax-online | 1.44e+01 | 1.45e+01 | | 2.92e+01 | ms |![softmax-online](SVGs_builtin/softmax-online.svg) |
| softmax | 1.12e+01 | 1.12e+01 | 3.43e+03 | 5.39e+01 | ms |![softmax](SVGs_builtin/softmax.svg) |
| sort | 2.13e-02 | | | 3.44e-01 | s |![sort](SVGs_builtin/sort.svg) |
| sosfil | 2.02e-02 | 2.29e-02 | 3.14e-02 | 1.05e-01 | s |![sosfil](SVGs_builtin/sosfil.svg) |
| sph | 5.81e+00 | 5.84e+00 | 4.01e+01 | 5.91e+00 | ms |![sph](SVGs_builtin/sph.svg) |
| split | 1.27e+04 | 1.28e+04 | | 1.49e+05 | us |![split](SVGs_builtin/split.svg) |
| spm | 7.59e+01 | 6.59e+01 | 6.33e+01 | 7.27e+01 | ms |![spm](SVGs_builtin/spm.svg) |
| sptrsv | 1.18e+02 | 1.37e+02 | | 1.17e+02 | us |![sptrsv](SVGs_builtin/sptrsv.svg) |
| srad | 3.67e+01 | 3.74e+01 | | | s |![srad](SVGs_builtin/srad.svg) |
| ss | 2.49e+02 | 2.37e+02 | 3.80e+02 | 7.90e+02 | us |![ss](SVGs_builtin/ss.svg) |
| stddev | 3.77e-03 | 5.05e-03 | 5.56e-03 | 6.51e-02 | s |![stddev](SVGs_builtin/stddev.svg) |
| stencil1d | 2.25e-03 | 2.25e-03 | 3.00e-01 | 3.50e-01 | s |![stencil1d](SVGs_builtin/stencil1d.svg) |
| stencil3d | 6.27e-03 | 6.31e-03 | 6.40e-03 | 1.74e-02 | s |![stencil3d](SVGs_builtin/stencil3d.svg) |
| streamcluster | 5.84e+00 | 1.15e+01 | 9.53e+00 | 7.69e+00 | s |![streamcluster](SVGs_builtin/streamcluster.svg) |
| su3 | 1.92e+00 | 2.32e+00 | 2.38e+00 | 2.35e+00 | s |![su3](SVGs_builtin/su3.svg) |
| surfel | | | 3.07e+02 | 3.59e+02 | ms |![surfel](SVGs_builtin/surfel.svg) |
| svd3x3 | 4.60e+01 | 4.84e+01 | 6.81e+01 | 7.09e+01 | us |![svd3x3](SVGs_builtin/svd3x3.svg) |
| sw4ck | 8.78e+00 | 1.02e+01 | 2.48e+02 | 2.36e+02 | ms |![sw4ck](SVGs_builtin/sw4ck.svg) |
| swish | 4.13e+03 | 4.35e+03 | 4.58e+03 | 6.21e+03 | us |![swish](SVGs_builtin/swish.svg) |
| tensorT | 1.41e+00 | 1.28e+00 | 1.36e+00 | 4.44e+00 | ms |![tensorT](SVGs_builtin/tensorT.svg) |
| testSNAP | 1.11e+01 | 1.16e+01 | 6.35e+02 | 1.48e+01 | ms |![testSNAP](SVGs_builtin/testSNAP.svg) |
| thomas | 1.44e+04 | 1.44e+04 | 1.24e+04 | 1.24e+04 | ms |![thomas](SVGs_builtin/thomas.svg) |
| threadfence | 2.30e+02 | 2.34e+02 | 1.50e+02 | 3.27e+02 | ms |![threadfence](SVGs_builtin/threadfence.svg) |
| tissue | 7.68e-03 | 7.99e-03 | 1.07e-02 | 8.15e-03 | s |![tissue](SVGs_builtin/tissue.svg) |
| tonemapping | 1.69e+01 | 1.62e+01 | 1.91e+01 | 1.82e+01 | us |![tonemapping](SVGs_builtin/tonemapping.svg) |
| tqs | 1.31e+02 | 7.99e+01 | | 1.42e+02 | ms |![tqs](SVGs_builtin/tqs.svg) |
| triad | 1.86e+03 | 1.23e+03 | 5.85e+02 | 6.10e+02 | GFLOPS |![triad](SVGs_builtin/triad.svg) |
| tridiagonal | 3.83e-02 | 3.83e-02 | | 1.55e-01 | s |![tridiagonal](SVGs_builtin/tridiagonal.svg) |
| tsa | 1.71e+04 | 1.76e+04 | | 1.70e+05 | us |![tsa](SVGs_builtin/tsa.svg) |
| tsp | 2.52e-02 | 5.34e-02 | | | s |![tsp](SVGs_builtin/tsp.svg) |
| urng | 4.87e+00 | 5.05e+00 | 1.17e+01 | 2.37e+01 | us |![urng](SVGs_builtin/urng.svg) |
| vanGenuchten | 7.29e-03 | 7.25e-03 | 7.45e-03 | 7.55e-03 | s |![vanGenuchten](SVGs_builtin/vanGenuchten.svg) |
| vmc | 2.52e-03 | 1.88e-03 | | | s |![vmc](SVGs_builtin/vmc.svg) |
| vol2col | 2.53e+03 | 2.55e+03 | 2.02e+03 | 2.73e+03 | us |![vol2col](SVGs_builtin/vol2col.svg) |
| wedford | | | | | ms |![wedford](SVGs_builtin/wedford.svg) |
| winograd | 7.85e-01 | 2.94e-01 | 3.19e+00 | 3.14e+00 | s |![winograd](SVGs_builtin/winograd.svg) |
| wlcpow | 2.03e+03 | 2.09e+03 | | 6.74e+03 | us |![wlcpow](SVGs_builtin/wlcpow.svg) |
| wordcount | 1.74e-03 | 3.75e-03 | 1.75e-03 | 2.73e-03 | s |![wordcount](SVGs_builtin/wordcount.svg) |
| wsm5 | 2.98e+02 | 2.32e+02 | 3.10e+02 | 3.28e+02 | ms |![wsm5](SVGs_builtin/wsm5.svg) |
| wyllie | 2.93e+01 | 2.95e+01 | 3.08e+01 | | ms |![wyllie](SVGs_builtin/wyllie.svg) |
| xlqc | 2.24e+06 | | | 3.03e+05 | us |![xlqc](SVGs_builtin/xlqc.svg) |
| xsbench | 4.45e+01 | 3.61e+01 | 3.57e+01 | 2.39e+00 | s |![xsbench](SVGs_builtin/xsbench.svg) |
| zeropoint | 1.39e+04 | 1.51e+04 | 1.72e+04 | 2.01e+04 | us |![zeropoint](SVGs_builtin/zeropoint.svg) |
| zmddft | 1.16e+00 | 1.40e+00 | | 2.01e+00 | ms |![zmddft](SVGs_builtin/zmddft.svg) |
