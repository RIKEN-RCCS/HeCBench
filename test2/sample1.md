# 実行時間 (sec.)

| name | sycl | acc | omp_nvc | plot |
| -- | -- | -- | -- | -- |
| accuracy | 2.49 | 2.48 | 2.53 |![accuracy](SVGs/accuracy.svg) |
| ace | 3.13 | 2.84 | 2.79 |![ace](SVGs/ace.svg) |
| adam | 5.24 | 6.72 | 6.70 |![adam](SVGs/adam.svg) |
| adamw | 12.32 | 20.30 | 20.34 |![adamw](SVGs/adamw.svg) |
| adjacent | 9.13 | 7.03 | 15.22 |![adjacent](SVGs/adjacent.svg) |
| adv | 3.35 | build err | 3.49 |![adv](SVGs/adv.svg) |
| aes | 2.79 | build err | 2.70 |![aes](SVGs/aes.svg) |
| affine | 5.42 | 12.78 | 12.37 |![affine](SVGs/affine.svg) |
| aidw | 12.87 | 5.42 | 6.21 |![aidw](SVGs/aidw.svg) |
| aligned-types | 3.68 | 4.40 | 4.71 |![aligned-types](SVGs/aligned-types.svg) |
| all-pairs-distance | 5.23 | 4.81 | 21.34 |![all-pairs-distance](SVGs/all-pairs-distance.svg) |
| amgmk | 2.45 | 2.42 | 2.44 |![amgmk](SVGs/amgmk.svg) |
| ans | 5.42 | exe err | exe err |![ans](SVGs/ans.svg) |
| aobench | 0.36 | 2.42 | 2.47 |![aobench](SVGs/aobench.svg) |
| aop | 7.02 | build err | build err |![aop](SVGs/aop.svg) |
| asmooth | 5.70 | 7.38 | 7.33 |![asmooth](SVGs/asmooth.svg) |
| assert | exe err | build err | 2.58 |![assert](SVGs/assert.svg) |
| asta | build err | build err | 10.71 |![asta](SVGs/asta.svg) |
| atan2 | 3.24 | 2.52 | 2.54 |![atan2](SVGs/atan2.svg) |
| atomicCost | 18.89 | 18.84 | 5.36 |![atomicCost](SVGs/atomicCost.svg) |
| atomicIntrinsics | over 600 | 3.18 | 3.21 |![atomicIntrinsics](SVGs/atomicIntrinsics.svg) |
| atomicPerf | 14.66 | 13.47 | 13.88 |![atomicPerf](SVGs/atomicPerf.svg) |
| atomicReduction | 3.44 | 2.65 | 2.63 |![atomicReduction](SVGs/atomicReduction.svg) |
| attention | 2.47 | 83.27 | 83.85 |![attention](SVGs/attention.svg) |
| axhelm | 2.41 | build err | 2.47 |![axhelm](SVGs/axhelm.svg) |
| babelstream | 0.56 | 2.72 | 2.85 |![babelstream](SVGs/babelstream.svg) |
| background-subtract | 8.07 | 8.66 | 8.33 |![background-subtract](SVGs/background-subtract.svg) |
| backprop | 2.45 | 2.44 | 2.47 |![backprop](SVGs/backprop.svg) |
| bezier-surface | 12.93 | 14.25 | 10.71 |![bezier-surface](SVGs/bezier-surface.svg) |
| bfs | 1.03 | 2.48 | 2.46 |![bfs](SVGs/bfs.svg) |
| bilateral | 15.09 | 205.59 | 7.85 |![bilateral](SVGs/bilateral.svg) |
| binomial | 3.10 | | -- |![binomial](SVGs/binomial.svg) |
| bitonic-sort | 10.81 | 7.80 | 8.49 |![bitonic-sort](SVGs/bitonic-sort.svg) |
| black-scholes | 5.00 | 39.24 | 4.88 |![black-scholes](SVGs/black-scholes.svg) |
| blas-gemm | build err | | -- |![blas-gemm](SVGs/blas-gemm.svg) |
| bn | 5.00 | build err | exe err |![bn](SVGs/bn.svg) |
| bonds | 7.66 | 11.50 | 11.19 |![bonds](SVGs/bonds.svg) |
| boxfilter | 8.46 | 6.47 | 9.55 |![boxfilter](SVGs/boxfilter.svg) |
| bsearch | 34.85 | 32.68 | 31.84 |![bsearch](SVGs/bsearch.svg) |
| bspline-vgh | 7.74 | 4.73 | 4.81 |![bspline-vgh](SVGs/bspline-vgh.svg) |
| b+tree | 2.46 | 2.46 | 2.46 |![b+tree](SVGs/b+tree.svg) |
| burger | 72.44 | 68.61 | 112.53 |![burger](SVGs/burger.svg) |
| bwt | 199.29 | 188.93 | exe err |![bwt](SVGs/bwt.svg) |
| car | 7.18 | 6.64 | 6.65 |![car](SVGs/car.svg) |
| cbsfil | 31.48 | 35.01 | 33.19 |![cbsfil](SVGs/cbsfil.svg) |
| ccsd-trpdrv | 2.67 | 2.55 | 2.63 |![ccsd-trpdrv](SVGs/ccsd-trpdrv.svg) |
| ccs | 3.57 | 2.97 | 3.65 |![ccs](SVGs/ccs.svg) |
| cfd | 0.58 | 3.02 | 3.08 |![cfd](SVGs/cfd.svg) |
| chacha20 | 4.68 | 9.24 | 6.66 |![chacha20](SVGs/chacha20.svg) |
| channelShuffle | 538.46 | 19.23 | 19.24 |![channelShuffle](SVGs/channelShuffle.svg) |
| channelSum | 64.97 | 65.55 | 68.17 |![channelSum](SVGs/channelSum.svg) |
| chemv | 3.20 | 2.85 | 3.21 |![chemv](SVGs/chemv.svg) |
| che | 3.05 | 3.27 | 3.38 |![che](SVGs/che.svg) |
| chi2 | 217.04 | 220.78 | 251.54 |![chi2](SVGs/chi2.svg) |
| clenergy | 2.66 | 3.13 | 3.16 |![clenergy](SVGs/clenergy.svg) |
| clink | 56.20 | 8.31 | 8.15 |![clink](SVGs/clink.svg) |
| cmp | exe err | build err | exe err |![cmp](SVGs/cmp.svg) |
| cm | exe err | build err | exe err |![cm](SVGs/cm.svg) |
| cobahh | 406.65 | 383.64 | 388.03 |![cobahh](SVGs/cobahh.svg) |
| colorwheel | 37.38 | 27.55 | 27.96 |![colorwheel](SVGs/colorwheel.svg) |
| columnarSolver | 10.16 | 16.87 | 10.22 |![columnarSolver](SVGs/columnarSolver.svg) |
| complex | 22.28 | 12.59 | 12.96 |![complex](SVGs/complex.svg) |
| compute-score | 3.24 | 3.27 | 4.34 |![compute-score](SVGs/compute-score.svg) |
| concat | 15.08 | 16.24 | 15.73 |![concat](SVGs/concat.svg) |
| contract | 11.40 | 28.95 | 10.82 |![contract](SVGs/contract.svg) |
| conversion | 4.98 | 3.56 | 3.44 |![conversion](SVGs/conversion.svg) |
| convolution1D | 380.96 | 333.42 | 383.27 |![convolution1D](SVGs/convolution1D.svg) |
| convolution3D | 2.48 | 3.17 | 2.51 |![convolution3D](SVGs/convolution3D.svg) |
| convolutionSeparable | 13.58 | build err | 22.05 |![convolutionSeparable](SVGs/convolutionSeparable.svg) |
| cooling | 154.60 | 452.57 | 454.09 |![cooling](SVGs/cooling.svg) |
| crc64 | 58.47 | 57.16 | 57.25 |![crc64](SVGs/crc64.svg) |
| cross | 14.58 | 7.67 | 8.04 |![cross](SVGs/cross.svg) |
| crs | 24.78 | build err | over 600 |![crs](SVGs/crs.svg) |
| d2q9-bgk | 2.85 | build err | 4.37 |![d2q9-bgk](SVGs/d2q9-bgk.svg) |
| damage | 24.45 | 36.68 | 243.28 |![damage](SVGs/damage.svg) |
| dct8x8 | 130.74 | build err | build err |![dct8x8](SVGs/dct8x8.svg) |
| ddbp | 17.41 | build err | 17.27 |![ddbp](SVGs/ddbp.svg) |
| debayer | 6.37 | 6.68 | 6.79 |![debayer](SVGs/debayer.svg) |
| degrid | 14.97 | 23.24 | 17.04 |![degrid](SVGs/degrid.svg) |
| dense-embedding | 142.22 | 99.47 | 184.08 |![dense-embedding](SVGs/dense-embedding.svg) |
| depixel | 157.40 | 208.93 | 168.04 |![depixel](SVGs/depixel.svg) |
| deredundancy | 17.89 | 15.82 | 15.76 |![deredundancy](SVGs/deredundancy.svg) |
| diamond | build err | build err | 37.30 |![diamond](SVGs/diamond.svg) |
| distort | 2.45 | 2.45 | 2.50 |![distort](SVGs/distort.svg) |
| divergence | 5.32 | 3.15 | 3.15 |![divergence](SVGs/divergence.svg) |
| doh | 14.97 | 2.39 | 2.41 |![doh](SVGs/doh.svg) |
| dp | build err | 25.04 | 25.25 |![dp](SVGs/dp.svg) |
| dslash | 2.70 | 2.68 | 2.66 |![dslash](SVGs/dslash.svg) |
| dxtc2 | 2.49 | build err | over 600 |![dxtc2](SVGs/dxtc2.svg) |
| easyWave | 2.97 | 2.79 | 3.08 |![easyWave](SVGs/easyWave.svg) |
| ecdh | 11.26 | 18.09 | 19.13 |![ecdh](SVGs/ecdh.svg) |
| eigenvalue | 19.92 | over 600 | 22.02 |![eigenvalue](SVGs/eigenvalue.svg) |
| entropy | 47.60 | 94.05 | 49.43 |![entropy](SVGs/entropy.svg) |
| epistasis | 53.82 | 259.15 | 110.41 |![epistasis](SVGs/epistasis.svg) |
| expdist | 7.45 | build err | 12.43 |![expdist](SVGs/expdist.svg) |
| extend2 | 5.17 | 4.68 | 4.94 |![extend2](SVGs/extend2.svg) |
| extrema | 6.75 | 6.99 | 7.10 |![extrema](SVGs/extrema.svg) |
| face | 2.64 | 2.45 | exe err |![face](SVGs/face.svg) |
| fdtd3d | 31.50 | build err | 43.29 |![fdtd3d](SVGs/fdtd3d.svg) |
| feynman-kac | 41.84 | 48.24 | 22.16 |![feynman-kac](SVGs/feynman-kac.svg) |
| fft | 3.19 | build err | 3.19 |![fft](SVGs/fft.svg) |
| fhd | 183.93 | over 600 | exe err |![fhd](SVGs/fhd.svg) |
| filter | build err | build err | 154.77 |![filter](SVGs/filter.svg) |
| flip | 37.32 | 14.45 | 15.50 |![flip](SVGs/flip.svg) |
| floydwarshall | 73.45 | 72.59 | 83.41 |![floydwarshall](SVGs/floydwarshall.svg) |
| fluidSim | 17.67 | 17.30 | 17.36 |![fluidSim](SVGs/fluidSim.svg) |
| fpc | 66.38 | build err | exe err |![fpc](SVGs/fpc.svg) |
| fpdc | 5.37 | build err | over 600 |![fpdc](SVGs/fpdc.svg) |
| fresnel | 6.00 | 6.61 | 6.61 |![fresnel](SVGs/fresnel.svg) |
| frna | 104.63 | build err | build err |![frna](SVGs/frna.svg) |
| fwt | 3.12 | 3.35 | 3.47 |![fwt](SVGs/fwt.svg) |
| gabor | 21.57 | 21.27 | 57.42 |![gabor](SVGs/gabor.svg) |
| gamma-correction | 63.44 | 105.30 | 104.46 |![gamma-correction](SVGs/gamma-correction.svg) |
| ga | 36.53 | 32.72 | 33.46 |![ga](SVGs/ga.svg) |
| gaussian | 4.12 | 3.96 | 4.07 |![gaussian](SVGs/gaussian.svg) |
| gc | 2.41 | build err | build err |![gc](SVGs/gc.svg) |
| gd | 13.02 | 12.66 | 12.83 |![gd](SVGs/gd.svg) |
| geglu | 94.40 | 22.13 | 22.07 |![geglu](SVGs/geglu.svg) |
| geodesic | 2.60 | 2.51 | 2.52 |![geodesic](SVGs/geodesic.svg) |
| glu | 121.58 | 124.43 | 120.93 |![glu](SVGs/glu.svg) |
| gmm | 35.97 | build err | 37.28 |![gmm](SVGs/gmm.svg) |
| goulash | 27.35 | 36.03 | 36.52 |![goulash](SVGs/goulash.svg) |
| gpp | 4.29 | 4.10 | 4.15 |![gpp](SVGs/gpp.svg) |
| grep | 29.24 | build err | 1.72 |![grep](SVGs/grep.svg) |
| grrt | 14.54 | 13.31 | 12.82 |![grrt](SVGs/grrt.svg) |
| haccmk | 4.33 | 2.57 | 4.42 |![haccmk](SVGs/haccmk.svg) |
| hausdorff | 30.37 | 32.21 | 31.92 |![hausdorff](SVGs/hausdorff.svg) |
| haversine | 2.18 | 2.49 | 2.43 |![haversine](SVGs/haversine.svg) |
| heartwall | exe err | build err | build err |![heartwall](SVGs/heartwall.svg) |
| heat2d | 61.50 | 25.22 | 5.92 |![heat2d](SVGs/heat2d.svg) |
| heat | 22.04 | 15.34 | 16.22 |![heat](SVGs/heat.svg) |
| hellinger | 4.80 | 6.88 | 4.52 |![hellinger](SVGs/hellinger.svg) |
| henry | 3.77 | 2.86 | 2.74 |![henry](SVGs/henry.svg) |
| hexciton | 6.03 | build err | 4.96 |![hexciton](SVGs/hexciton.svg) |
| histogram | 2.98 | build err | 3.30 |![histogram](SVGs/histogram.svg) |
| hmm | 8.56 | 14.71 | 2.81 |![hmm](SVGs/hmm.svg) |
| hogbom | 2.67 | 2.65 | 2.76 |![hogbom](SVGs/hogbom.svg) |
| hotspot3D | 16.21 | 18.11 | 18.00 |![hotspot3D](SVGs/hotspot3D.svg) |
| hwt1d | 3.13 | 3.27 | 3.23 |![hwt1d](SVGs/hwt1d.svg) |
| hybridsort | 15.27 | build err | 12.96 |![hybridsort](SVGs/hybridsort.svg) |
| hypterm | 5.86 | 13.14 | 13.12 |![hypterm](SVGs/hypterm.svg) |
| idivide | 8.31 | 8.02 | 9.39 |![idivide](SVGs/idivide.svg) |
| interleave | 4.13 | 4.07 | 3.44 |![interleave](SVGs/interleave.svg) |
| interval | 7.15 | 13.47 | 10.78 |![interval](SVGs/interval.svg) |
| inversek2j | 1.05 | 225.58 | 111.49 |![inversek2j](SVGs/inversek2j.svg) |
| ising | 258.64 | 13.36 | 13.31 |![ising](SVGs/ising.svg) |
| iso2dfd | 17.67 | 13.97 | 14.20 |![iso2dfd](SVGs/iso2dfd.svg) |
| jacobi | 2.77 | 3.56 | 3.50 |![jacobi](SVGs/jacobi.svg) |
| jenkins-hash | 4.98 | 5.14 | 5.17 |![jenkins-hash](SVGs/jenkins-hash.svg) |
| kalman | over 600 | build err | exe err |![kalman](SVGs/kalman.svg) |
| keccaktreehash | 8.26 | 11.46 | 10.08 |![keccaktreehash](SVGs/keccaktreehash.svg) |
| keogh | 7.03 | 5.80 | 5.89 |![keogh](SVGs/keogh.svg) |
| kernelLaunch | 14.84 | 281.99 | 73.99 |![kernelLaunch](SVGs/kernelLaunch.svg) |
| kmeans | 28.93 | 27.05 | 22.68 |![kmeans](SVGs/kmeans.svg) |
| knn | 4.48 | 4.28 | 4.53 |![knn](SVGs/knn.svg) |
| lanczos | 2.61 | build err | 2.81 |![lanczos](SVGs/lanczos.svg) |
| langevin | 17.73 | 6.26 | 31.31 |![langevin](SVGs/langevin.svg) |
| langford | 2.54 | build err | build err |![langford](SVGs/langford.svg) |
| laplace3d | 64.92 | 14.60 | 14.64 |![laplace3d](SVGs/laplace3d.svg) |
| laplace | 333.46 | 11.93 | 11.34 |![laplace](SVGs/laplace.svg) |
| lavaMD | 102.55 | build err | 116.55 |![lavaMD](SVGs/lavaMD.svg) |
| layout | 3.27 | 2.66 | 2.93 |![layout](SVGs/layout.svg) |
| lci | 10.00 | 4.36 | 4.36 |![lci](SVGs/lci.svg) |
| lda | 3.10 | build err | build err |![lda](SVGs/lda.svg) |
| ldpc | 4.91 | build err | 9.60 |![ldpc](SVGs/ldpc.svg) |
| lebesgue | 41.65 | 39.43 | 39.11 |![lebesgue](SVGs/lebesgue.svg) |
| leukocyte | build err | build err | 0.92 |![leukocyte](SVGs/leukocyte.svg) |
| libor | 2.87 | 13.25 | 2.84 |![libor](SVGs/libor.svg) |
| lid-driven-cavity | 10.91 | build err | 22.40 |![lid-driven-cavity](SVGs/lid-driven-cavity.svg) |
| lif | 116.49 | 125.87 | 124.42 |![lif](SVGs/lif.svg) |
| linearprobing | 102.65 | build err | build err |![linearprobing](SVGs/linearprobing.svg) |
| log2 | 0.74 | 2.77 | 2.81 |![log2](SVGs/log2.svg) |
| lombscargle | 2.99 | 3.06 | 3.05 |![lombscargle](SVGs/lombscargle.svg) |
| loopback | 5.33 | 5.38 | 8.16 |![loopback](SVGs/loopback.svg) |
| lrn | 105.73 | 111.48 | 78.63 |![lrn](SVGs/lrn.svg) |
| lr | 6.36 | build err | build err |![lr](SVGs/lr.svg) |
| lsqt | 19.99 | 32.63 | 32.86 |![lsqt](SVGs/lsqt.svg) |
| lud | 8.19 | 9.54 | 46.07 |![lud](SVGs/lud.svg) |
| lulesh | 11.28 | 11.70 | 8.41 |![lulesh](SVGs/lulesh.svg) |
| mallocFree | 3.69 | 2.68 | 2.67 |![mallocFree](SVGs/mallocFree.svg) |
| mandelbrot | 5.29 | 3.75 | 3.82 |![mandelbrot](SVGs/mandelbrot.svg) |
| mask | 55.31 | 21.60 | 29.68 |![mask](SVGs/mask.svg) |
| match | 42.51 | build err | 41.46 |![match](SVGs/match.svg) |
| matern | 6.24 | 3.85 | 31.37 |![matern](SVGs/matern.svg) |
| matrix-rotate | 35.98 | 35.18 | 35.00 |![matrix-rotate](SVGs/matrix-rotate.svg) |
| maxFlops | 25.56 | 26.46 | 25.46 |![maxFlops](SVGs/maxFlops.svg) |
| maxpool3d | 33.75 | 32.80 | 32.24 |![maxpool3d](SVGs/maxpool3d.svg) |
| mcmd | 9.03 | 10.79 | 11.06 |![mcmd](SVGs/mcmd.svg) |
| mcpr | 14.07 | 21.86 | 15.00 |![mcpr](SVGs/mcpr.svg) |
| md5hash | 15.14 | 15.35 | 15.50 |![md5hash](SVGs/md5hash.svg) |
| mdh | 48.13 | 35.15 | 2.67 |![mdh](SVGs/mdh.svg) |
| md | 14.68 | 14.24 | 14.22 |![md](SVGs/md.svg) |
| meanshift | 4.42 | 4.71 | 4.40 |![meanshift](SVGs/meanshift.svg) |
| medianfilter | 3.75 | build err | 4.68 |![medianfilter](SVGs/medianfilter.svg) |
| memcpy | 3.54 | 6.52 | 6.54 |![memcpy](SVGs/memcpy.svg) |
| memtest | 18.85 | 19.78 | 19.45 |![memtest](SVGs/memtest.svg) |
| merge | 442.02 | build err | 425.74 |![merge](SVGs/merge.svg) |
| metropolis | 14.03 | build err | 65.32 |![metropolis](SVGs/metropolis.svg) |
| michalewicz | 150.55 | over 600 | 63.30 |![michalewicz](SVGs/michalewicz.svg) |
| minibude | 3.31 | build err | 3.54 |![minibude](SVGs/minibude.svg) |
| miniFE | | | -- |![miniFE](SVGs/miniFE.svg) |
| minimap2 | 2.32 | | 2.19 |![minimap2](SVGs/minimap2.svg) |
| minisweep | 56.83 | 35.43 | 35.90 |![minisweep](SVGs/minisweep.svg) |
| miniWeather | 8.19 | 33.30 | 33.14 |![miniWeather](SVGs/miniWeather.svg) |
| minkowski | 24.25 | 22.27 | 22.36 |![minkowski](SVGs/minkowski.svg) |
| mis | 13.22 | build err | 2.60 |![mis](SVGs/mis.svg) |
| mixbench | 3.62 | build err | 3.80 |![mixbench](SVGs/mixbench.svg) |
| morphology | 4.81 | build err | over 600 |![morphology](SVGs/morphology.svg) |
| mrc | 83.40 | 17.88 | 17.83 |![mrc](SVGs/mrc.svg) |
| mriQ | 5.31 | 5.04 | 5.21 |![mriQ](SVGs/mriQ.svg) |
| mr | 3.33 | build err | 3.31 |![mr](SVGs/mr.svg) |
| mt | 4.65 | 4.29 | 4.08 |![mt](SVGs/mt.svg) |
| multimaterial | 14.67 | 12.11 | 21.91 |![multimaterial](SVGs/multimaterial.svg) |
| murmurhash3 | 2.80 | 2.67 | 2.69 |![murmurhash3](SVGs/murmurhash3.svg) |
| myocyte | 3.03 | build err | exe err |![myocyte](SVGs/myocyte.svg) |
| nbody | 25.24 | 19.63 | 19.61 |![nbody](SVGs/nbody.svg) |
| ne | 5.37 | 4.90 | 4.94 |![ne](SVGs/ne.svg) |
| nlll | 3.08 | build err | 3.06 |![nlll](SVGs/nlll.svg) |
| nms | 2.59 | build err | 2.73 |![nms](SVGs/nms.svg) |
| nn | 2.47 | 2.56 | 2.52 |![nn](SVGs/nn.svg) |
| norm2 | build err | 3.71 | 4.97 |![norm2](SVGs/norm2.svg) |
| nqueen | 6.94 | 23.85 | 6.87 |![nqueen](SVGs/nqueen.svg) |
| ntt | 6.63 | build err | 6.65 |![ntt](SVGs/ntt.svg) |
| nw | 26.82 | build err | 33.48 |![nw](SVGs/nw.svg) |
| openmp | 81.27 | build err | 13.65 |![openmp](SVGs/openmp.svg) |
| overlay | 5.33 | 9.20 | 10.50 |![overlay](SVGs/overlay.svg) |
| p4 | 3.77 | build err | 16.94 |![p4](SVGs/p4.svg) |
| page-rank | 375.35 | 46.20 | 46.72 |![page-rank](SVGs/page-rank.svg) |
| particle-diffusion | 11.94 | 13.04 | 12.23 |![particle-diffusion](SVGs/particle-diffusion.svg) |
| particlefilter | 198.39 | build err | 219.65 |![particlefilter](SVGs/particlefilter.svg) |
| particles | 3.46 | build err | over 600 |![particles](SVGs/particles.svg) |
| pathfinder | 64.27 | build err | exe err |![pathfinder](SVGs/pathfinder.svg) |
| permutate | 24.87 | build err | 25.87 |![permutate](SVGs/permutate.svg) |
| permute | 61.55 | 85.77 | 85.40 |![permute](SVGs/permute.svg) |
| perplexity | 192.24 | 97.16 | 96.53 |![perplexity](SVGs/perplexity.svg) |
| phmm | 3.53 | build err | 7.57 |![phmm](SVGs/phmm.svg) |
| pnpoly | 21.12 | build err | 74.17 |![pnpoly](SVGs/pnpoly.svg) |
| pns | 5.44 | build err | exe err |![pns](SVGs/pns.svg) |
| pointwise | 21.28 | 53.08 | 75.86 |![pointwise](SVGs/pointwise.svg) |
| pool | 51.58 | 41.75 | 41.00 |![pool](SVGs/pool.svg) |
| popcount | 178.30 | 279.36 | 134.99 |![popcount](SVGs/popcount.svg) |
| present | 2.67 | 3.04 | 3.05 |![present](SVGs/present.svg) |
| prna | build err | build err | exe err |![prna](SVGs/prna.svg) |
| projectile | 0.91 | 2.66 | 2.55 |![projectile](SVGs/projectile.svg) |
| pso | 2.83 | build err | 2.69 |![pso](SVGs/pso.svg) |
| qrg | 6.30 | 8.57 | 10.24 |![qrg](SVGs/qrg.svg) |
| qtclustering | 34.48 | build err | exe err |![qtclustering](SVGs/qtclustering.svg) |
| quantBnB | 121.16 | 125.38 | 125.68 |![quantBnB](SVGs/quantBnB.svg) |
| quicksort | 36.99 | build err | 33.98 |![quicksort](SVGs/quicksort.svg) |
| radixsort | 2.90 | build err | 8.11 |![radixsort](SVGs/radixsort.svg) |
| rainflow | 29.99 | 29.64 | 29.52 |![rainflow](SVGs/rainflow.svg) |
| randomAccess | 6.90 | build err | 24.54 |![randomAccess](SVGs/randomAccess.svg) |
| reaction | 4.08 | build err | 9.17 |![reaction](SVGs/reaction.svg) |
| recursiveGaussian | 4.04 | build err | 4.27 |![recursiveGaussian](SVGs/recursiveGaussian.svg) |
| resize | 5.98 | 6.03 | 6.01 |![resize](SVGs/resize.svg) |
| reverse | 3.67 | build err | 8.08 |![reverse](SVGs/reverse.svg) |
| rfs | 46.70 | 48.57 | 48.26 |![rfs](SVGs/rfs.svg) |
| rng-wallace | 2.56 | build err | 3.55 |![rng-wallace](SVGs/rng-wallace.svg) |
| rodrigues | 101.46 | 114.74 | 101.38 |![rodrigues](SVGs/rodrigues.svg) |
| romberg | 2.72 | build err | 2.81 |![romberg](SVGs/romberg.svg) |
| rsbench | 3.39 | build err | build err |![rsbench](SVGs/rsbench.svg) |
| rsc | 0.50 | build err | 2.73 |![rsc](SVGs/rsc.svg) |
| rtm8 | 17.67 | 15.76 | 4.22 |![rtm8](SVGs/rtm8.svg) |
| rushlarsen | 129.52 | 135.46 | 135.06 |![rushlarsen](SVGs/rushlarsen.svg) |
| s3d | 2.94 | build err | 2.90 |![s3d](SVGs/s3d.svg) |
| s8n | 2.49 | 2.47 | 2.43 |![s8n](SVGs/s8n.svg) |
| sad | 2.48 | 2.63 | 81.81 |![sad](SVGs/sad.svg) |
| sampling | 5.77 | build err | 5.65 |![sampling](SVGs/sampling.svg) |
| scan2 | 2.91 | build err | 6.76 |![scan2](SVGs/scan2.svg) |
| scan | 31.87 | build err | 33.05 |![scan](SVGs/scan.svg) |
| scatterAdd | 8.10 | 8.05 | 7.79 |![scatterAdd](SVGs/scatterAdd.svg) |
| scel | 322.81 | 11.91 | 11.29 |![scel](SVGs/scel.svg) |
| secp256k1 | 2.68 | 2.79 | 2.69 |![secp256k1](SVGs/secp256k1.svg) |
| sheath | 5.85 | 5.81 | 5.82 |![sheath](SVGs/sheath.svg) |
| shmembench | 9.16 | build err | exe err |![shmembench](SVGs/shmembench.svg) |
| simplemoc | 400.17 | 293.35 | 294.34 |![simplemoc](SVGs/simplemoc.svg) |
| simpleSpmv | 319.81 | 320.00 | 317.47 |![simpleSpmv](SVGs/simpleSpmv.svg) |
| slu | build err | build err | build err |![slu](SVGs/slu.svg) |
| snake | 8.43 | 10.57 | 10.42 |![snake](SVGs/snake.svg) |
| sobel | 2.89 | 3.18 | 3.32 |![sobel](SVGs/sobel.svg) |
| sobol | 3.89 | build err | 4.51 |![sobol](SVGs/sobol.svg) |
| softmax-online | 13.69 | build err | 15.10 |![softmax-online](SVGs/softmax-online.svg) |
| softmax | 39.36 | 40.91 | 36.96 |![softmax](SVGs/softmax.svg) |
| sort | build err | build err | 36.92 |![sort](SVGs/sort.svg) |
| sosfil | 4.75 | build err | 13.32 |![sosfil](SVGs/sosfil.svg) |
| sph | 5.41 | 22.53 | 5.40 |![sph](SVGs/sph.svg) |
| split | 35.78 | build err | 37.73 |![split](SVGs/split.svg) |
| spm | 104.39 | 98.00 | 98.41 |![spm](SVGs/spm.svg) |
| sptrsv | 2.70 | build err | 2.81 |![sptrsv](SVGs/sptrsv.svg) |
| srad | 37.49 | build err | exe err |![srad](SVGs/srad.svg) |
| ss | 7.23 | 9.95 | 18.10 |![ss](SVGs/ss.svg) |
| stddev | 46.55 | build err | 51.00 |![stddev](SVGs/stddev.svg) |
| stencil1d | 4.68 | 30.81 | 35.77 |![stencil1d](SVGs/stencil1d.svg) |
| stencil3d | 4.35 | build err | 4.24 |![stencil3d](SVGs/stencil3d.svg) |
| streamcluster | 11.84 | build err | 26.24 |![streamcluster](SVGs/streamcluster.svg) |
| su3 | 5.17 | 7.47 | 7.48 |![su3](SVGs/su3.svg) |
| surfel | 510.01 | 12.02 | 11.97 |![surfel](SVGs/surfel.svg) |
| svd3x3 | 2.00 | 2.86 | 2.89 |![svd3x3](SVGs/svd3x3.svg) |
| sw4ck | 6.40 | 29.28 | 28.11 |![sw4ck](SVGs/sw4ck.svg) |
| swish | 24.54 | 5.48 | 5.39 |![swish](SVGs/swish.svg) |
| tensorT | 2.56 | build err | 2.87 |![tensorT](SVGs/tensorT.svg) |
| testSNAP | 3.73 | 66.16 | 4.19 |![testSNAP](SVGs/testSNAP.svg) |
| thomas | 131.17 | 59.68 | 49.73 |![thomas](SVGs/thomas.svg) |
| threadfence | 26.09 | build err | 35.11 |![threadfence](SVGs/threadfence.svg) |
| tissue | 12.45 | 11.82 | 18.23 |![tissue](SVGs/tissue.svg) |
| tonemapping | 3.15 | 3.02 | 2.95 |![tonemapping](SVGs/tonemapping.svg) |
| tqs | 2.57 | build err | 2.59 |![tqs](SVGs/tqs.svg) |
| triad | 2.62 | 2.82 | 2.75 |![triad](SVGs/triad.svg) |
| tridiagonal | 8.80 | build err | 24.55 |![tridiagonal](SVGs/tridiagonal.svg) |
| tsa | 36.86 | build err | 29.09 |![tsa](SVGs/tsa.svg) |
| tsp | 7.82 | build err | build err |![tsp](SVGs/tsp.svg) |
| urng | 2.43 | 2.21 | 2.20 |![urng](SVGs/urng.svg) |
| vanGenuchten | 21.48 | 30.99 | 30.92 |![vanGenuchten](SVGs/vanGenuchten.svg) |
| vmc | 4.38 | build err | build err |![vmc](SVGs/vmc.svg) |
| vol2col | 5.26 | 7.87 | 8.13 |![vol2col](SVGs/vol2col.svg) |
| wedford | build err | build err | exe err |![wedford](SVGs/wedford.svg) |
| winograd | 0.65 | 3.25 | 3.20 |![winograd](SVGs/winograd.svg) |
| wlcpow | 4.50 | build err | 12.37 |![wlcpow](SVGs/wlcpow.svg) |
| wordcount | 7.98 | 8.18 | 7.93 |![wordcount](SVGs/wordcount.svg) |
| wsm5 | 4.78 | 5.59 | 5.73 |![wsm5](SVGs/wsm5.svg) |
| wyllie | 112.64 | 108.06 | over 600 |![wyllie](SVGs/wyllie.svg) |
| xlqc | build err | build err | build err |![xlqc](SVGs/xlqc.svg) |
| xsbench | 37.40 | 37.18 | 3.89 |![xsbench](SVGs/xsbench.svg) |
| zeropoint | 55.54 | 21.82 | 24.23 |![zeropoint](SVGs/zeropoint.svg) |
| zmddft | 2.63 | build err | 14.28 |![zmddft](SVGs/zmddft.svg) |
| | | | |
| completed | 306 | 214 | 285 ||


sycl と acc がともに完了した件数 211

