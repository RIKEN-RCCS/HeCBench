#include <iostream>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

#define p_IJWID 6
#define p_JID   4
#define p_JWID  5
#define p_Np    512
#define p_Nq    8
#define p_Nvgeo 12
#define p_RXID  0
#define p_RYID  1
#define p_RZID  7
#define p_SXID  2
#define p_SYID  3
#define p_SZID  8
#define p_TXID  9
#define p_TYID  10
#define p_TZID  11
#define p_cubNp 4096
#define p_cubNq 16

dfloat *drandAlloc(int N){
  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));
  for(int n = 0; n < N; ++n) v[n] = drand48();
  return v;
}

int main(int argc, char **argv) {

  if (argc < 4) {
    printf("Usage: ./adv N cubN numElements [nRepetitions]\n");
    exit(-1);
  }

  const int N = atoi(argv[1]);
  const int cubN = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  int Ntests = 1;

  if(argc >= 5) Ntests = atoi(argv[4]);

  const int Nq = N+1;
  const int cubNq = cubN+1;
  const int Np = Nq*Nq*Nq;
  const int cubNp = cubNq*cubNq*cubNq;
  const dlong offset = Nelements*Np;

  printf("Data type in bytes: %zu\n", sizeof(dfloat));

  srand48(123);
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);

  Kokkos::initialize(argc, argv);
  {
    // Create device views
    Kokkos::View<dfloat*> d_vgeo("vgeo", Np*Nelements*p_Nvgeo);
    Kokkos::View<dfloat*> d_cubvgeo("cubvgeo", cubNp*Nelements*p_Nvgeo);
    Kokkos::View<dfloat*> d_cubDiffInterpT("cubDiffInterpT", 3*cubNp*Nelements);
    Kokkos::View<dfloat*> d_cubInterpT("cubInterpT", Np*cubNp);
    Kokkos::View<dfloat*> d_u("u", 3*Np*Nelements);
    Kokkos::View<dfloat*> d_adv("adv", 3*Np*Nelements);

    // Create host mirrors and copy data to device
    auto h_vgeo = Kokkos::create_mirror_view(d_vgeo);
    auto h_cubvgeo = Kokkos::create_mirror_view(d_cubvgeo);
    auto h_cubDiffInterpT = Kokkos::create_mirror_view(d_cubDiffInterpT);
    auto h_cubInterpT = Kokkos::create_mirror_view(d_cubInterpT);
    auto h_u = Kokkos::create_mirror_view(d_u);

    for (int i = 0; i < Np*Nelements*p_Nvgeo; i++) h_vgeo(i) = vgeo[i];
    for (int i = 0; i < cubNp*Nelements*p_Nvgeo; i++) h_cubvgeo(i) = cubvgeo[i];
    for (int i = 0; i < 3*cubNp*Nelements; i++) h_cubDiffInterpT(i) = cubDiffInterpT[i];
    for (int i = 0; i < Np*cubNp; i++) h_cubInterpT(i) = cubInterpT[i];
    for (int i = 0; i < 3*Np*Nelements; i++) h_u(i) = u[i];

    Kokkos::deep_copy(d_vgeo, h_vgeo);
    Kokkos::deep_copy(d_cubvgeo, h_cubvgeo);
    Kokkos::deep_copy(d_cubDiffInterpT, h_cubDiffInterpT);
    Kokkos::deep_copy(d_cubInterpT, h_cubInterpT);
    Kokkos::deep_copy(d_u, h_u);

    // Scratch memory types
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchView = Kokkos::View<dfloat*, scratch_space, Kokkos::MemoryUnmanaged>;

    // Calculate total scratch memory needed
    size_t scratch_bytes = ScratchView::shmem_size(16*16)       // s_cubD[16][16]
                         + ScratchView::shmem_size(8*16)        // s_cubInterpT[8][16]
                         + ScratchView::shmem_size(8*8) * 3     // s_U, s_V, s_W [8][8]
                         + ScratchView::shmem_size(16*16) * 3;  // s_U1, s_V1, s_W1 [16][16]

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    auto policy = team_policy(Nelements, 256)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    auto start = std::chrono::high_resolution_clock::now();

    for (int test = 0; test < Ntests; ++test) {
      Kokkos::parallel_for("advection", policy,
        KOKKOS_LAMBDA(const member_type &team) {

          // Allocate scratch views (flattened 2D arrays)
          ScratchView s_cubD(team.team_scratch(0), 16*16);
          ScratchView s_cubInterpT(team.team_scratch(0), 8*16);
          ScratchView s_U(team.team_scratch(0), 8*8);
          ScratchView s_V(team.team_scratch(0), 8*8);
          ScratchView s_W(team.team_scratch(0), 8*8);
          ScratchView s_U1(team.team_scratch(0), 16*16);
          ScratchView s_V1(team.team_scratch(0), 16*16);
          ScratchView s_W1(team.team_scratch(0), 16*16);

          dfloat r_U[16], r_V[16], r_W[16];
          dfloat r_Ud[16], r_Vd[16], r_Wd[16];

          const int e = team.league_rank();
          const int i = team.team_rank() % 16;
          const int j = team.team_rank() / 16;
          const int id = j * 16 + i;

          if (id < 8 * 16) s_cubInterpT(id) = d_cubInterpT(id);
          s_cubD(id) = d_cubDiffInterpT(id);

          for (int k = 0; k < 16; ++k) {
            r_U[k] = 0;
            r_V[k] = 0;
            r_W[k] = 0;
            r_Ud[k] = 0;
            r_Vd[k] = 0;
            r_Wd[k] = 0;
          }

          for (int c = 0; c < 8; ++c) {
            if (j < 8 && i < 8) {
              const int lid = e * p_Np + c * 8 * 8 + j * 8 + i;
              s_U(j * 8 + i) = d_u(lid + 0 * offset);
              s_V(j * 8 + i) = d_u(lid + 1 * offset);
              s_W(j * 8 + i) = d_u(lid + 2 * offset);
            }

            team.team_barrier();

            if (j < 8) {
              dfloat U1 = 0, V1 = 0, W1 = 0;
              for (int a = 0; a < 8; ++a) {
                dfloat Iia = s_cubInterpT(a * 16 + i);
                U1 += Iia * s_U(j * 8 + a);
                V1 += Iia * s_V(j * 8 + a);
                W1 += Iia * s_W(j * 8 + a);
              }
              s_U1(j * 16 + i) = U1;
              s_V1(j * 16 + i) = V1;
              s_W1(j * 16 + i) = W1;
            } else {
              s_U1(j * 16 + i) = 0;
              s_V1(j * 16 + i) = 0;
              s_W1(j * 16 + i) = 0;
            }

            team.team_barrier();

            dfloat U2 = 0, V2 = 0, W2 = 0;
            for (int b = 0; b < 8; ++b) {
              dfloat Ijb = s_cubInterpT(b * 16 + j);
              U2 += Ijb * s_U1(b * 16 + i);
              V2 += Ijb * s_V1(b * 16 + i);
              W2 += Ijb * s_W1(b * 16 + i);
            }
            for (int k = 0; k < 16; ++k) {
              dfloat Ikc = s_cubInterpT(c * 16 + k);
              r_U[k] += Ikc * U2;
              r_V[k] += Ikc * V2;
              r_W[k] += Ikc * W2;
            }
            for (int k = 0; k < 16; ++k) {
              r_Ud[k] = r_U[k];
              r_Vd[k] = r_V[k];
              r_Wd[k] = r_W[k];
            }
          }

          for (int k = 0; k < 16; ++k) {
            s_U1(j * 16 + i) = r_Ud[k];
            s_V1(j * 16 + i) = r_Vd[k];
            s_W1(j * 16 + i) = r_Wd[k];

            team.team_barrier();

            dfloat Udr = 0, Uds = 0, Udt = 0;
            dfloat Vdr = 0, Vds = 0, Vdt = 0;
            dfloat Wdr = 0, Wds = 0, Wdt = 0;
            for (int n = 0; n < 16; ++n) {
              dfloat Din = s_cubD(i * 16 + n);
              Udr += Din * s_U1(j * 16 + n);
              Vdr += Din * s_V1(j * 16 + n);
              Wdr += Din * s_W1(j * 16 + n);
            }
            for (int n = 0; n < 16; ++n) {
              dfloat Djn = s_cubD(j * 16 + n);
              Uds += Djn * s_U1(n * 16 + i);
              Vds += Djn * s_V1(n * 16 + i);
              Wds += Djn * s_W1(n * 16 + i);
            }
            for (int n = 0; n < 16; ++n) {
              dfloat Dkn = s_cubD(k * 16 + n);
              Udt += Dkn * r_Ud[n];
              Vdt += Dkn * r_Vd[n];
              Wdt += Dkn * r_Wd[n];
            }

            const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
            const dfloat drdx = d_cubvgeo(gid + p_RXID * p_cubNp);
            const dfloat drdy = d_cubvgeo(gid + p_RYID * p_cubNp);
            const dfloat drdz = d_cubvgeo(gid + p_RZID * p_cubNp);
            const dfloat dsdx = d_cubvgeo(gid + p_SXID * p_cubNp);
            const dfloat dsdy = d_cubvgeo(gid + p_SYID * p_cubNp);
            const dfloat dsdz = d_cubvgeo(gid + p_SZID * p_cubNp);
            const dfloat dtdx = d_cubvgeo(gid + p_TXID * p_cubNp);
            const dfloat dtdy = d_cubvgeo(gid + p_TYID * p_cubNp);
            const dfloat dtdz = d_cubvgeo(gid + p_TZID * p_cubNp);
            const dfloat JW   = d_cubvgeo(gid + p_JWID * p_cubNp);
            const dfloat Un = r_U[k];
            const dfloat Vn = r_V[k];
            const dfloat Wn = r_W[k];
            const dfloat Uhat = JW * (Un * drdx + Vn * drdy + Wn * drdz);
            const dfloat Vhat = JW * (Un * dsdx + Vn * dsdy + Wn * dsdz);
            const dfloat What = JW * (Un * dtdx + Vn * dtdy + Wn * dtdz);
            r_U[k] = Uhat * Udr + Vhat * Uds + What * Udt;
            r_V[k] = Uhat * Vdr + Vhat * Vds + What * Vdt;
            r_W[k] = Uhat * Wdr + Vhat * Wds + What * Wdt;
          }

          for (int c = 0; c < 8; ++c) {
            dfloat rhsU = 0, rhsV = 0, rhsW = 0;
            for (int k = 0; k < 16; ++k) {
              dfloat Ikc = s_cubInterpT(c * 16 + k);
              rhsU += Ikc * r_U[k];
              rhsV += Ikc * r_V[k];
              rhsW += Ikc * r_W[k];
            }

            if (i < 8 && j < 8) {
              s_U(j * 8 + i) = rhsU;
              s_V(j * 8 + i) = rhsV;
              s_W(j * 8 + i) = rhsW;
            }

            team.team_barrier();

            if (j < 8) {
              dfloat rhsU2 = 0, rhsV2 = 0, rhsW2 = 0;
              for (int k = 0; k < 16; ++k) {
                dfloat Ijb = s_cubInterpT(j * 16 + k);
                if (k < 8 && i < 8) {
                  rhsU2 += Ijb * s_U(k * 8 + i);
                  rhsV2 += Ijb * s_V(k * 8 + i);
                  rhsW2 += Ijb * s_W(k * 8 + i);
                }
              }
              s_U1(j * 16 + i) = rhsU2;
              s_V1(j * 16 + i) = rhsV2;
              s_W1(j * 16 + i) = rhsW2;
            }

            team.team_barrier();

            if (i < 8 && j < 8) {
              dfloat rhsU3 = 0, rhsV3 = 0, rhsW3 = 0;
              for (int k = 0; k < 16; ++k) {
                dfloat Iia = s_cubInterpT(i * 16 + k);
                rhsU3 += Iia * s_U1(j * 16 + k);
                rhsV3 += Iia * s_V1(j * 16 + k);
                rhsW3 += Iia * s_W1(j * 16 + k);
              }
              const int gid = e * p_Np * p_Nvgeo + c * 8 * 8 + j * 8 + i;
              const dfloat IJW = d_vgeo(gid + p_IJWID * p_Np);
              const int lid = e * p_Np + c * 8 * 8 + j * 8 + i;
              d_adv(lid + 0 * offset) = IJW * rhsU3;
              d_adv(lid + 1 * offset) = IJW * rhsV3;
              d_adv(lid + 2 * offset) = IJW * rhsW3;
            }
          }
        }
      );
    }

    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

    // statistics
    const dfloat GDOFPerSecond = (N*N*N)*Nelements/elapsed;
    std::cout << " NRepetitions=" << Ntests
              << " N=" << N
              << " cubN=" << cubN
              << " Nelements=" << Nelements
              << " elapsed time=" << elapsed
              << " GDOF/s=" << GDOFPerSecond
              << "\n";

    // Copy result back to host
    auto h_adv = Kokkos::create_mirror_view(d_adv);
    Kokkos::deep_copy(h_adv, d_adv);
    for (int i = 0; i < 3*Np*Nelements; i++) adv[i] = h_adv(i);
  }
  Kokkos::finalize();

  double checksum = 0;
  for (int i = 0; i < 3*Np*Nelements; i++) {
    checksum += adv[i];
    #ifdef OUTPUT
    std::cout << adv[i] << "\n";
    #endif
  }
  std::cout << "Checksum=" << checksum << "\n";

  free(vgeo          );
  free(cubvgeo       );
  free(cubDiffInterpT);
  free(cubInterpT    );
  free(u             );
  free(adv           );
  return 0;
}
