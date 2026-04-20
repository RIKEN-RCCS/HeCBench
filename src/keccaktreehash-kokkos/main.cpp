/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010
   Kokkos port, 2024

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>

//==============================================================================
// Parameters
//==============================================================================
#define NB_THREADS              64
#define NB_THREADS_BLOCKS       64
#define NB_STREAMS              2
#define INPUT_BLOCK_SIZE_B      32
#define OUTPUT_BLOCK_SIZE_B     32
#define NB_INPUT_BLOCK          1024
#define NB_SCND_STAGE_THREADS   16
#define NB_INPUT_BLOCK_SNCD_STAGE (2*NB_THREADS/NB_SCND_STAGE_THREADS)

typedef unsigned int tKeccakLane;

#define cKeccakNumberOfRounds 22
#define ROL32(a, offset) (((a) << (offset)) ^ ((a) >> (32-(offset))))

//==============================================================================
// Round constants (host copy used by both CPU functions and View initialisation)
//==============================================================================
static const tKeccakLane KeccakF_RoundConstants_host[22] = {
  (tKeccakLane)0x00000001, (tKeccakLane)0x00008082,
  (tKeccakLane)0x0000808a, (tKeccakLane)0x80008000,
  (tKeccakLane)0x0000808b, (tKeccakLane)0x80000001,
  (tKeccakLane)0x80008081, (tKeccakLane)0x00008009,
  (tKeccakLane)0x0000008a, (tKeccakLane)0x00000088,
  (tKeccakLane)0x80008009, (tKeccakLane)0x8000000a,
  (tKeccakLane)0x8000808b, (tKeccakLane)0x0000008b,
  (tKeccakLane)0x00008089, (tKeccakLane)0x00008003,
  (tKeccakLane)0x00008002, (tKeccakLane)0x00000080,
  (tKeccakLane)0x0000800a, (tKeccakLane)0x8000000a,
  (tKeccakLane)0x80008081, (tKeccakLane)0x00008080
};

//==============================================================================
// Device Keccak-f permutation (unrolled)
//==============================================================================
KOKKOS_INLINE_FUNCTION
void KeccakFunr(tKeccakLane* state, const tKeccakLane* rc)
{
  tKeccakLane BC[5], temp;

  for (int round = 0; round < cKeccakNumberOfRounds; ++round) {
    // Theta
    BC[0] = state[0]^state[5]^state[10]^state[15]^state[20];
    BC[1] = state[1]^state[6]^state[11]^state[16]^state[21];
    BC[2] = state[2]^state[7]^state[12]^state[17]^state[22];
    BC[3] = state[3]^state[8]^state[13]^state[18]^state[23];
    BC[4] = state[4]^state[9]^state[14]^state[19]^state[24];
    temp = BC[4]^ROL32(BC[1],1);
    state[0]^=temp; state[5]^=temp; state[10]^=temp; state[15]^=temp; state[20]^=temp;
    temp = BC[0]^ROL32(BC[2],1);
    state[1]^=temp; state[6]^=temp; state[11]^=temp; state[16]^=temp; state[21]^=temp;
    temp = BC[1]^ROL32(BC[3],1);
    state[2]^=temp; state[7]^=temp; state[12]^=temp; state[17]^=temp; state[22]^=temp;
    temp = BC[2]^ROL32(BC[4],1);
    state[3]^=temp; state[8]^=temp; state[13]^=temp; state[18]^=temp; state[23]^=temp;
    temp = BC[3]^ROL32(BC[0],1);
    state[4]^=temp; state[9]^=temp; state[14]^=temp; state[19]^=temp; state[24]^=temp;

    // Rho Pi
    temp=state[1];  BC[0]=state[10]; state[10]=ROL32(temp, 1);
    temp=BC[0];     BC[0]=state[7];  state[7] =ROL32(temp, 3);
    temp=BC[0];     BC[0]=state[11]; state[11]=ROL32(temp, 6);
    temp=BC[0];     BC[0]=state[17]; state[17]=ROL32(temp,10);
    temp=BC[0];     BC[0]=state[18]; state[18]=ROL32(temp,15);
    temp=BC[0];     BC[0]=state[3];  state[3] =ROL32(temp,21);
    temp=BC[0];     BC[0]=state[5];  state[5] =ROL32(temp,28);
    temp=BC[0];     BC[0]=state[16]; state[16]=ROL32(temp, 4);
    temp=BC[0];     BC[0]=state[8];  state[8] =ROL32(temp,13);
    temp=BC[0];     BC[0]=state[21]; state[21]=ROL32(temp,23);
    temp=BC[0];     BC[0]=state[24]; state[24]=ROL32(temp, 2);
    temp=BC[0];     BC[0]=state[4];  state[4] =ROL32(temp,14);
    temp=BC[0];     BC[0]=state[15]; state[15]=ROL32(temp,27);
    temp=BC[0];     BC[0]=state[23]; state[23]=ROL32(temp, 9);
    temp=BC[0];     BC[0]=state[19]; state[19]=ROL32(temp,24);
    temp=BC[0];     BC[0]=state[13]; state[13]=ROL32(temp, 8);
    temp=BC[0];     BC[0]=state[12]; state[12]=ROL32(temp,25);
    temp=BC[0];     BC[0]=state[2];  state[2] =ROL32(temp,11);
    temp=BC[0];     BC[0]=state[20]; state[20]=ROL32(temp,30);
    temp=BC[0];     BC[0]=state[14]; state[14]=ROL32(temp,18);
    temp=BC[0];     BC[0]=state[22]; state[22]=ROL32(temp, 7);
    temp=BC[0];     BC[0]=state[9];  state[9] =ROL32(temp,29);
    temp=BC[0];     BC[0]=state[6];  state[6] =ROL32(temp,20);
    temp=BC[0];     BC[0]=state[1];  state[1] =ROL32(temp,12);
    temp=BC[0]; (void)temp;

    // Chi
    BC[0]=state[0];  BC[1]=state[1];  BC[2]=state[2];  BC[3]=state[3];  BC[4]=state[4];
    state[0]=BC[0]^((~BC[1])&BC[2]); state[1]=BC[1]^((~BC[2])&BC[3]);
    state[2]=BC[2]^((~BC[3])&BC[4]); state[3]=BC[3]^((~BC[4])&BC[0]); state[4]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[5];  BC[1]=state[6];  BC[2]=state[7];  BC[3]=state[8];  BC[4]=state[9];
    state[5]=BC[0]^((~BC[1])&BC[2]); state[6]=BC[1]^((~BC[2])&BC[3]);
    state[7]=BC[2]^((~BC[3])&BC[4]); state[8]=BC[3]^((~BC[4])&BC[0]); state[9]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[10]; BC[1]=state[11]; BC[2]=state[12]; BC[3]=state[13]; BC[4]=state[14];
    state[10]=BC[0]^((~BC[1])&BC[2]); state[11]=BC[1]^((~BC[2])&BC[3]);
    state[12]=BC[2]^((~BC[3])&BC[4]); state[13]=BC[3]^((~BC[4])&BC[0]); state[14]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[15]; BC[1]=state[16]; BC[2]=state[17]; BC[3]=state[18]; BC[4]=state[19];
    state[15]=BC[0]^((~BC[1])&BC[2]); state[16]=BC[1]^((~BC[2])&BC[3]);
    state[17]=BC[2]^((~BC[3])&BC[4]); state[18]=BC[3]^((~BC[4])&BC[0]); state[19]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[20]; BC[1]=state[21]; BC[2]=state[22]; BC[3]=state[23]; BC[4]=state[24];
    state[20]=BC[0]^((~BC[1])&BC[2]); state[21]=BC[1]^((~BC[2])&BC[3]);
    state[22]=BC[2]^((~BC[3])&BC[4]); state[23]=BC[3]^((~BC[4])&BC[0]); state[24]=BC[4]^((~BC[0])&BC[1]);

    // Iota
    state[0] ^= rc[round];
  }
}

//==============================================================================
// Host-only Keccak-f permutation (uses host constant array)
//==============================================================================
static void KeccakF_CPU(tKeccakLane* state)
{
  tKeccakLane BC[5], temp;

  for (int round = 0; round < cKeccakNumberOfRounds; ++round) {
    // Theta
    BC[0] = state[0]^state[5]^state[10]^state[15]^state[20];
    BC[1] = state[1]^state[6]^state[11]^state[16]^state[21];
    BC[2] = state[2]^state[7]^state[12]^state[17]^state[22];
    BC[3] = state[3]^state[8]^state[13]^state[18]^state[23];
    BC[4] = state[4]^state[9]^state[14]^state[19]^state[24];
    temp = BC[4]^ROL32(BC[1],1);
    state[0]^=temp; state[5]^=temp; state[10]^=temp; state[15]^=temp; state[20]^=temp;
    temp = BC[0]^ROL32(BC[2],1);
    state[1]^=temp; state[6]^=temp; state[11]^=temp; state[16]^=temp; state[21]^=temp;
    temp = BC[1]^ROL32(BC[3],1);
    state[2]^=temp; state[7]^=temp; state[12]^=temp; state[17]^=temp; state[22]^=temp;
    temp = BC[2]^ROL32(BC[4],1);
    state[3]^=temp; state[8]^=temp; state[13]^=temp; state[18]^=temp; state[23]^=temp;
    temp = BC[3]^ROL32(BC[0],1);
    state[4]^=temp; state[9]^=temp; state[14]^=temp; state[19]^=temp; state[24]^=temp;

    // Rho Pi
    temp=state[1];  BC[0]=state[10]; state[10]=ROL32(temp, 1);
    temp=BC[0];     BC[0]=state[7];  state[7] =ROL32(temp, 3);
    temp=BC[0];     BC[0]=state[11]; state[11]=ROL32(temp, 6);
    temp=BC[0];     BC[0]=state[17]; state[17]=ROL32(temp,10);
    temp=BC[0];     BC[0]=state[18]; state[18]=ROL32(temp,15);
    temp=BC[0];     BC[0]=state[3];  state[3] =ROL32(temp,21);
    temp=BC[0];     BC[0]=state[5];  state[5] =ROL32(temp,28);
    temp=BC[0];     BC[0]=state[16]; state[16]=ROL32(temp, 4);
    temp=BC[0];     BC[0]=state[8];  state[8] =ROL32(temp,13);
    temp=BC[0];     BC[0]=state[21]; state[21]=ROL32(temp,23);
    temp=BC[0];     BC[0]=state[24]; state[24]=ROL32(temp, 2);
    temp=BC[0];     BC[0]=state[4];  state[4] =ROL32(temp,14);
    temp=BC[0];     BC[0]=state[15]; state[15]=ROL32(temp,27);
    temp=BC[0];     BC[0]=state[23]; state[23]=ROL32(temp, 9);
    temp=BC[0];     BC[0]=state[19]; state[19]=ROL32(temp,24);
    temp=BC[0];     BC[0]=state[13]; state[13]=ROL32(temp, 8);
    temp=BC[0];     BC[0]=state[12]; state[12]=ROL32(temp,25);
    temp=BC[0];     BC[0]=state[2];  state[2] =ROL32(temp,11);
    temp=BC[0];     BC[0]=state[20]; state[20]=ROL32(temp,30);
    temp=BC[0];     BC[0]=state[14]; state[14]=ROL32(temp,18);
    temp=BC[0];     BC[0]=state[22]; state[22]=ROL32(temp, 7);
    temp=BC[0];     BC[0]=state[9];  state[9] =ROL32(temp,29);
    temp=BC[0];     BC[0]=state[6];  state[6] =ROL32(temp,20);
    temp=BC[0];     BC[0]=state[1];  state[1] =ROL32(temp,12);
    temp=BC[0]; (void)temp;

    // Chi
    BC[0]=state[0];  BC[1]=state[1];  BC[2]=state[2];  BC[3]=state[3];  BC[4]=state[4];
    state[0]=BC[0]^((~BC[1])&BC[2]); state[1]=BC[1]^((~BC[2])&BC[3]);
    state[2]=BC[2]^((~BC[3])&BC[4]); state[3]=BC[3]^((~BC[4])&BC[0]); state[4]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[5];  BC[1]=state[6];  BC[2]=state[7];  BC[3]=state[8];  BC[4]=state[9];
    state[5]=BC[0]^((~BC[1])&BC[2]); state[6]=BC[1]^((~BC[2])&BC[3]);
    state[7]=BC[2]^((~BC[3])&BC[4]); state[8]=BC[3]^((~BC[4])&BC[0]); state[9]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[10]; BC[1]=state[11]; BC[2]=state[12]; BC[3]=state[13]; BC[4]=state[14];
    state[10]=BC[0]^((~BC[1])&BC[2]); state[11]=BC[1]^((~BC[2])&BC[3]);
    state[12]=BC[2]^((~BC[3])&BC[4]); state[13]=BC[3]^((~BC[4])&BC[0]); state[14]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[15]; BC[1]=state[16]; BC[2]=state[17]; BC[3]=state[18]; BC[4]=state[19];
    state[15]=BC[0]^((~BC[1])&BC[2]); state[16]=BC[1]^((~BC[2])&BC[3]);
    state[17]=BC[2]^((~BC[3])&BC[4]); state[18]=BC[3]^((~BC[4])&BC[0]); state[19]=BC[4]^((~BC[0])&BC[1]);
    BC[0]=state[20]; BC[1]=state[21]; BC[2]=state[22]; BC[3]=state[23]; BC[4]=state[24];
    state[20]=BC[0]^((~BC[1])&BC[2]); state[21]=BC[1]^((~BC[2])&BC[3]);
    state[22]=BC[2]^((~BC[3])&BC[4]); state[23]=BC[3]^((~BC[4])&BC[0]); state[24]=BC[4]^((~BC[0])&BC[1]);

    // Iota
    state[0] ^= KeccakF_RoundConstants_host[round];
  }
}

//==============================================================================
// Top-node absorb: folds all first-stage outputs into a single Keccak state
//==============================================================================
static void Keccak_top(tKeccakLane* Kstate, tKeccakLane* inBuffer, int block_number)
{
  for (int k = 0; k < block_number; k++) {
    for (int ind_word = 0; ind_word < OUTPUT_BLOCK_SIZE_B/4; ind_word++)
      Kstate[ind_word] ^= inBuffer[ind_word + k * OUTPUT_BLOCK_SIZE_B/4];
    KeccakF_CPU(Kstate);
  }
}

//==============================================================================
// CPU reference: first-stage tree hash
//==============================================================================
static void KeccakTreeCPU(tKeccakLane* inBuffer, tKeccakLane* outBuffer)
{
  for (int blkIdx = 0; blkIdx < NB_THREADS_BLOCKS; blkIdx++) {
    for (int thrIdx = 0; thrIdx < NB_THREADS; thrIdx++) {
      tKeccakLane Kstate[25];
      memset(Kstate, 0, 25 * sizeof(tKeccakLane));

      for (int k = 0; k < NB_INPUT_BLOCK; k++) {
        for (int ind_word = 0; ind_word < INPUT_BLOCK_SIZE_B/4; ind_word++) {
          Kstate[ind_word] ^=
            inBuffer[thrIdx
              + ind_word  * NB_THREADS
              + k         * NB_THREADS * INPUT_BLOCK_SIZE_B/4
              + blkIdx    * NB_THREADS * INPUT_BLOCK_SIZE_B/4 * NB_INPUT_BLOCK];
        }
        KeccakF_CPU(Kstate);
      }

      for (int ind_word = 0; ind_word < OUTPUT_BLOCK_SIZE_B/4; ind_word++) {
        outBuffer[thrIdx
          + ind_word * NB_THREADS
          + blkIdx   * NB_THREADS * OUTPUT_BLOCK_SIZE_B/4] = Kstate[ind_word];
      }
    }
  }
}

//==============================================================================
// Helpers
//==============================================================================
static void print_KS_256(tKeccakLane* Kstate)
{
  printf("\n");
  for (int x = 0; x < 8; x++)
    printf("%08x ", Kstate[x]);
  printf("\n");
}

static int isEqual_KS(tKeccakLane* Ks1, tKeccakLane* Ks2)
{
  for (int x = 0; x < 5; x++)
    for (int y = 0; y < 5; y++)
      if (Ks1[x + 5*y] != Ks2[x + 5*y])
        return 0;
  return 1;
}

//==============================================================================
// Main
//==============================================================================
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    // Buffer sizes (in number of tKeccakLane elements)
    const int inBufSize  = NB_THREADS_BLOCKS * NB_INPUT_BLOCK * NB_THREADS * (INPUT_BLOCK_SIZE_B/4);
    const int outBufSize = NB_THREADS_BLOCKS * NB_THREADS * (OUTPUT_BLOCK_SIZE_B/4);

    printf("Numbers of Threads PER BLOCK            NB_THREADS           %u\n", NB_THREADS);
    printf("Numbers of Threads Blocks               NB_THREADS_BLOCKS    %u\n", NB_THREADS_BLOCKS);
    printf("Input block size of Keccak (in Byte)    INPUT_BLOCK_SIZE_B   %u\n", INPUT_BLOCK_SIZE_B);
    printf("Output block size of Keccak (in Byte)   OUTPUT_BLOCK_SIZE_B  %u\n", OUTPUT_BLOCK_SIZE_B);
    printf("NB of input blocks per thread           NB_INPUT_BLOCK       %u\n\n", NB_INPUT_BLOCK);

    // -------------------------------------------------------------------------
    // Allocate device Views
    // -------------------------------------------------------------------------
    Kokkos::View<tKeccakLane*> d_inBuffer("inBuffer",  inBufSize);
    Kokkos::View<tKeccakLane*> d_outBuffer("outBuffer", outBufSize);
    Kokkos::View<tKeccakLane*> d_rc("rc", cKeccakNumberOfRounds);

    // Host mirrors
    auto h_inBuffer  = Kokkos::create_mirror_view(d_inBuffer);
    auto h_outBuffer = Kokkos::create_mirror_view(d_outBuffer);
    auto h_rc        = Kokkos::create_mirror_view(d_rc);

    // Initialise input buffer: h_inBuffer[i] = i  (matches OMP TestGPU)
    for (int i = 0; i < inBufSize; i++)
      h_inBuffer(i) = static_cast<tKeccakLane>(i);

    // Initialise round constants
    for (int i = 0; i < cKeccakNumberOfRounds; i++)
      h_rc(i) = KeccakF_RoundConstants_host[i];

    Kokkos::deep_copy(d_inBuffer,  h_inBuffer);
    Kokkos::deep_copy(d_rc,        h_rc);

    // -------------------------------------------------------------------------
    // GPU kernel: first-stage tree hash
    // Each team corresponds to one block index; each team member to a thread.
    // -------------------------------------------------------------------------
    Kokkos::parallel_for(
      "KeccakTreeGPU",
      Kokkos::TeamPolicy<>(NB_THREADS_BLOCKS, NB_THREADS),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team)
      {
        const int blkIdx = team.league_rank();
        const int thrIdx = team.team_rank();

        tKeccakLane Kstate[25];
        for (int i = 0; i < 25; i++) Kstate[i] = 0;

        for (int k = 0; k < NB_INPUT_BLOCK; k++) {
          for (int ind_word = 0; ind_word < INPUT_BLOCK_SIZE_B/4; ind_word++) {
            Kstate[ind_word] ^=
              d_inBuffer(thrIdx
                + ind_word * NB_THREADS
                + k        * NB_THREADS * (INPUT_BLOCK_SIZE_B/4)
                + blkIdx   * NB_THREADS * (INPUT_BLOCK_SIZE_B/4) * NB_INPUT_BLOCK);
          }
          KeccakFunr(Kstate, d_rc.data());
        }

        for (int ind_word = 0; ind_word < OUTPUT_BLOCK_SIZE_B/4; ind_word++) {
          d_outBuffer(thrIdx
            + ind_word * NB_THREADS
            + blkIdx   * NB_THREADS * (OUTPUT_BLOCK_SIZE_B/4)) = Kstate[ind_word];
        }
      });

    Kokkos::fence();

    // Copy GPU output back to host
    Kokkos::deep_copy(h_outBuffer, d_outBuffer);

    // -------------------------------------------------------------------------
    // Top-level hash over GPU first-stage output
    // -------------------------------------------------------------------------
    tKeccakLane Kstate_GPU[25];
    memset(Kstate_GPU, 0, sizeof(Kstate_GPU));
    Keccak_top(Kstate_GPU, h_outBuffer.data(), NB_THREADS_BLOCKS * NB_THREADS);

    printf("GPU result:");
    print_KS_256(Kstate_GPU);

    // -------------------------------------------------------------------------
    // CPU reference
    // -------------------------------------------------------------------------
    // Allocate plain host arrays (same size) for the CPU path
    tKeccakLane* cpu_inBuffer  = (tKeccakLane*)malloc(inBufSize  * sizeof(tKeccakLane));
    tKeccakLane* cpu_outBuffer = (tKeccakLane*)malloc(outBufSize * sizeof(tKeccakLane));

    for (int i = 0; i < inBufSize; i++)
      cpu_inBuffer[i] = static_cast<tKeccakLane>(i);
    memset(cpu_outBuffer, 0, outBufSize * sizeof(tKeccakLane));

    KeccakTreeCPU(cpu_inBuffer, cpu_outBuffer);

    tKeccakLane Kstate_CPU[25];
    memset(Kstate_CPU, 0, sizeof(Kstate_CPU));
    Keccak_top(Kstate_CPU, cpu_outBuffer, NB_THREADS_BLOCKS * NB_THREADS);

    printf("CPU result:");
    print_KS_256(Kstate_CPU);

    free(cpu_inBuffer);
    free(cpu_outBuffer);

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------
    if (isEqual_KS(Kstate_GPU, Kstate_CPU))
      printf("Test Passed\n");
    else
      printf("Test Failed\n");
  }
  Kokkos::finalize();
  return 0;
}
