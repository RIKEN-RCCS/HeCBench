#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

// Constants from original
#define U133AArrayLength 22283
#define threadsPerBlock  128

KOKKOS_INLINE_FUNCTION double computeUCMax(int sigNGenes, int nGenesTotal) {
  return ((double)sigNGenes*nGenesTotal - (double)sigNGenes*(sigNGenes+1)/2 + sigNGenes);
}

/*
 * Compute dot product of two integer vectors using TeamPolicy reduction.
 * Each team handles a block of elements, outputting a partial sum.
 */
void computeDotProductHelper(
    Kokkos::View<int*> result,
    Kokkos::View<const int*> v1,
    Kokkos::View<const int*> v2,
    const int teams,
    const int vLength)
{
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchI = Kokkos::View<int*, ScratchSpace, Kokkos::MemoryUnmanaged>;
  const int scratchBytes = threadsPerBlock * sizeof(int);

  Kokkos::parallel_for("computeDotProduct",
    Kokkos::TeamPolicy<>(teams, threadsPerBlock).set_scratch_size(0, Kokkos::PerTeam(scratchBytes)),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
      ScratchI cache(team.team_scratch(0), threadsPerBlock);
      const int blockIdx_x = team.league_rank();
      const int cacheIndex = team.team_rank();
      int tid = cacheIndex + blockIdx_x * threadsPerBlock;
      int temp = 0;
      while(tid < vLength) {
        temp += v1[tid] * v2[tid];
        tid += threadsPerBlock * teams;
      }
      cache[cacheIndex] = temp;
      team.team_barrier();
      int i = threadsPerBlock/2;
      while(i != 0) {
        if(cacheIndex < i) cache[cacheIndex] += cache[cacheIndex+i];
        team.team_barrier();
        i /= 2;
      }
      if(cacheIndex == 0) result[blockIdx_x] = cache[0];
    });
}

/*
 * Count elements above threshold using TeamPolicy reduction.
 */
void countAboveThresholdHelper(
    Kokkos::View<const float*> array,
    const float threshold,
    Kokkos::View<int*> counter,
    const int teams,
    const int arrayLength)
{
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchI = Kokkos::View<int*, ScratchSpace, Kokkos::MemoryUnmanaged>;
  const int scratchBytes = threadsPerBlock * sizeof(int);

  Kokkos::parallel_for("countAboveThreshold",
    Kokkos::TeamPolicy<>(teams, threadsPerBlock).set_scratch_size(0, Kokkos::PerTeam(scratchBytes)),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
      ScratchI binomial(team.team_scratch(0), threadsPerBlock);
      const int blockIdx_x = team.league_rank();
      const int cacheIndex = team.team_rank();
      int tid = cacheIndex + blockIdx_x * threadsPerBlock;
      int ptemp = 0;
      while(tid < arrayLength) {
        if(array[tid] > threshold) ptemp++;
        tid += threadsPerBlock * teams;
      }
      binomial[cacheIndex] = ptemp;
      team.team_barrier();
      int i = threadsPerBlock/2;
      while(i != 0) {
        if(cacheIndex < i) binomial[cacheIndex] += binomial[cacheIndex+i];
        team.team_barrier();
        i /= 2;
      }
      if(cacheIndex == 0) counter[blockIdx_x] = binomial[0];
    });
}

/*
 * Compute connection scores for randomized signatures.
 */
void computeRandomConnectionScores(
    Kokkos::View<const float*> random,
    Kokkos::View<const int*> reffile,
    Kokkos::View<float*> output,
    const int M, const float UCmax, const int setSize,
    const int nRandomGenerations)
{
  Kokkos::parallel_for("randomConnectionScores", nRandomGenerations, KOKKOS_LAMBDA(int idx) {
    float temp = 0.0f;
    for(int col = idx; col < M; col += nRandomGenerations) {
      float n = 1.0f - random[col];
      float regulateFactor = 1.0f / setSize;
      if(n >= 0.5f) { regulateFactor = -regulateFactor; n -= 0.5f; }
      int rangeInArray = (int)(n * U133AArrayLength * 2);
      if(rangeInArray >= 0 && rangeInArray < M)
        temp += reffile[rangeInArray] * regulateFactor;
    }
    output[idx] = temp / UCmax;
  });
}

int main(int argc, char **argv) {
  const int nRandomGenerations = (argc>1) ? atoi(argv[1]) : 1000;
  const int repeat = (argc>2) ? atoi(argv[2]) : 10;

  const int nGenesTotal = U133AArrayLength;
  const int sigNGenes = 50;
  double UCmax = computeUCMax(sigNGenes, nGenesTotal);

  const int signatureByRNGs = nRandomGenerations * sigNGenes;
  const int blocksPerGrid_Gene = (int)ceil((float)nGenesTotal / threadsPerBlock);
  const int blocksPerGrid_Gen  = (int)ceil((float)nRandomGenerations / threadsPerBlock);

  printf("nRandomGenerations=%d, blocksPerGrid_Gene=%d, blocksPerGrid_Gen=%d\n",
    nRandomGenerations, blocksPerGrid_Gene, blocksPerGrid_Gen);

  // Host data
  float *h_random = (float*)malloc(signatureByRNGs*sizeof(float));
  float *h_arraysAdded = (float*)malloc(nRandomGenerations*sizeof(float));
  int   *h_refRegValues = (int*)malloc(nGenesTotal*sizeof(int));
  int   *h_qIndex = (int*)malloc(nGenesTotal*sizeof(int));
  int   *h_dotProductResult = (int*)malloc(blocksPerGrid_Gene*sizeof(int));
  int   *h_aboveThreshold = (int*)malloc(blocksPerGrid_Gen*sizeof(int));

  std::default_random_engine gen(123);
  std::uniform_real_distribution<float> dist(0.0f,1.0f);
  for(int i=0;i<signatureByRNGs;i++) h_random[i]=dist(gen);
  for(int i=0;i<nGenesTotal;i++) h_refRegValues[i]=(i%3==0)?1:(i%3==1)?-1:0;
  memset(h_qIndex,0,nGenesTotal*sizeof(int));
  for(int i=0;i<sigNGenes;i++) h_qIndex[i*400]=(i%2==0)?1:-1;

  Kokkos::initialize(argc, argv);
  {
    using ViewF  = Kokkos::View<float*>;
    using ViewI  = Kokkos::View<int*>;
    using ViewCF = Kokkos::View<const float*>;
    using ViewCI = Kokkos::View<const int*>;

    ViewF d_random("random", signatureByRNGs);
    ViewF d_arraysAdded("arraysAdded", nRandomGenerations);
    ViewI d_refRegValues("refRegValues", nGenesTotal);
    ViewI d_qIndex("qIndex", nGenesTotal);
    ViewI d_dotProductResult("dotProductResult", blocksPerGrid_Gene);
    ViewI d_aboveThreshold("aboveThreshold", blocksPerGrid_Gen);

    {
      auto hr=Kokkos::create_mirror_view(d_random);
      auto hv=Kokkos::create_mirror_view(d_refRegValues);
      auto hq=Kokkos::create_mirror_view(d_qIndex);
      for(int i=0;i<signatureByRNGs;i++) hr(i)=h_random[i];
      for(int i=0;i<nGenesTotal;i++) hv(i)=h_refRegValues[i];
      for(int i=0;i<nGenesTotal;i++) hq(i)=h_qIndex[i];
      Kokkos::deep_copy(d_random,hr);
      Kokkos::deep_copy(d_refRegValues,hv);
      Kokkos::deep_copy(d_qIndex,hq);
    }

    auto t0=std::chrono::steady_clock::now();
    for(int rep=0;rep<repeat;rep++) {
      // Dot product: qIndex . refRegValues
      computeDotProductHelper(d_dotProductResult,
        ViewCI(d_qIndex.data(), nGenesTotal),
        ViewCI(d_refRegValues.data(), nGenesTotal),
        blocksPerGrid_Gene, nGenesTotal);

      // Random connection scores
      computeRandomConnectionScores(
        ViewCF(d_random.data(), signatureByRNGs),
        ViewCI(d_refRegValues.data(), nGenesTotal),
        d_arraysAdded,
        signatureByRNGs, (float)UCmax, sigNGenes, nRandomGenerations);

      // Count above threshold
      countAboveThresholdHelper(
        ViewCF(d_arraysAdded.data(), nRandomGenerations),
        0.0f, d_aboveThreshold,
        blocksPerGrid_Gen, nRandomGenerations);
    }
    Kokkos::fence();
    auto t1=std::chrono::steady_clock::now();
    printf("Average kernel time: %f ms\n",
      std::chrono::duration<double,std::milli>(t1-t0).count()/repeat);

    // Copy results back
    {
      auto hd=Kokkos::create_mirror_view(d_dotProductResult);
      auto ha=Kokkos::create_mirror_view(d_aboveThreshold);
      Kokkos::deep_copy(hd,d_dotProductResult);
      Kokkos::deep_copy(ha,d_aboveThreshold);
      int dot=0; for(int i=0;i<blocksPerGrid_Gene;i++) dot+=hd(i);
      int above=0; for(int i=0;i<blocksPerGrid_Gen;i++) above+=ha(i);
      printf("Dot product result: %d\n", dot);
      printf("Count above threshold: %d\n", above);
    }
  }
  Kokkos::finalize();

  free(h_random); free(h_arraysAdded); free(h_refRegValues);
  free(h_qIndex); free(h_dotProductResult); free(h_aboveThreshold);
  return 0;
}
