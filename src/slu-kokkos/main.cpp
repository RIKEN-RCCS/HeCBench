#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>
#include "symbolic.h"
#include "Timer.h"

using namespace std;

#define TMPMEMNUM  10353
#define Nstreams   16

using team_policy = Kokkos::TeamPolicy<>;
using member_type = team_policy::member_type;

// Scratch view for per-team s[32]
using ScratchREAL = Kokkos::View<REAL*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                 Kokkos::MemoryUnmanaged>;

void RL(
    const int nteams,
    const int nthreads,
    Kokkos::View<const unsigned*> sym_c_ptr_dev,
    Kokkos::View<const unsigned*> sym_r_idx_dev,
    Kokkos::View<REAL*>           val_dev,
    Kokkos::View<const unsigned*> l_col_ptr_dev,
    Kokkos::View<const unsigned*> csr_r_ptr_dev,
    Kokkos::View<const unsigned*> csr_c_idx_dev,
    Kokkos::View<const unsigned*> csr_diag_ptr_dev,
    Kokkos::View<const int*>      level_idx_dev,
    Kokkos::View<REAL*>           tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos)
{
  size_t scratch_bytes = ScratchREAL::shmem_size(32);

  Kokkos::parallel_for("RL",
    team_policy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const member_type& team) {
      const int tid = team.team_rank();
      const int bid = team.league_rank();
      const int wid = tid / 32;
      const int tidInWarp = tid % 32;

      ScratchREAL s(team.team_scratch(0), 32);

      const unsigned currentCol = level_idx_dev(levelHead + inLevPos + bid);
      const unsigned currentLColSize = sym_c_ptr_dev(currentCol + 1) - l_col_ptr_dev(currentCol) - 1;
      const unsigned currentLPos = l_col_ptr_dev(currentCol) + tid + 1;

      // Update current col: divide by diagonal
      int offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          val_dev(currentLPos + offset) /= val_dev(l_col_ptr_dev(currentCol));
          tmpMem(bid * n + ridx) = val_dev(currentLPos + offset);
        }
        offset += team.team_size();
      }
      team.team_barrier();

      // Broadcast to submatrix
      const unsigned subColPos = csr_diag_ptr_dev(currentCol) + wid + 1;
      const unsigned subMatSize = csr_r_ptr_dev(currentCol + 1) - csr_diag_ptr_dev(currentCol) - 1;

      int woffset = 0;
      while (subMatSize > (unsigned)woffset) {
        if (wid + woffset < (int)subMatSize) {
          offset = 0;
          unsigned subCol = csr_c_idx_dev(subColPos + woffset);
          unsigned colSize = sym_c_ptr_dev(subCol + 1) - sym_c_ptr_dev(subCol);
          while ((unsigned)offset < colSize) {
            if (tidInWarp + offset < (int)colSize) {
              unsigned subColElem = sym_c_ptr_dev(subCol) + tidInWarp + offset;
              unsigned ridx = sym_r_idx_dev(subColElem);
              if (ridx == currentCol) {
                s(wid) = val_dev(subColElem);
              }
              if (ridx > currentCol) {
                Kokkos::atomic_add(&val_dev(subColElem),
                                   -tmpMem(ridx + n * bid) * s(wid));
              }
            }
            offset += 32;
          }
        }
        woffset += team.team_size() / 32;
      }

      team.team_barrier();
      // Clear tmpMem
      offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          tmpMem(bid * n + ridx) = 0;
        }
        offset += team.team_size();
      }
    });
}

void RL_perturb(
    const int nteams,
    const int nthreads,
    Kokkos::View<const unsigned*> sym_c_ptr_dev,
    Kokkos::View<const unsigned*> sym_r_idx_dev,
    Kokkos::View<REAL*>           val_dev,
    Kokkos::View<const unsigned*> l_col_ptr_dev,
    Kokkos::View<const unsigned*> csr_r_ptr_dev,
    Kokkos::View<const unsigned*> csr_c_idx_dev,
    Kokkos::View<const unsigned*> csr_diag_ptr_dev,
    Kokkos::View<const int*>      level_idx_dev,
    Kokkos::View<REAL*>           tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos,
    const float pert)
{
  size_t scratch_bytes = ScratchREAL::shmem_size(32);

  Kokkos::parallel_for("RL_perturb",
    team_policy(nteams, nthreads).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const member_type& team) {
      const int tid = team.team_rank();
      const int bid = team.league_rank();
      const int wid = tid / 32;
      const int tidInWarp = tid % 32;

      ScratchREAL s(team.team_scratch(0), 32);

      const unsigned currentCol = level_idx_dev(levelHead + inLevPos + bid);
      const unsigned currentLColSize = sym_c_ptr_dev(currentCol + 1) - l_col_ptr_dev(currentCol) - 1;
      const unsigned currentLPos = l_col_ptr_dev(currentCol) + tid + 1;

      int offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          if (fabs(val_dev(l_col_ptr_dev(currentCol))) < pert)
            val_dev(l_col_ptr_dev(currentCol)) = pert;
          val_dev(currentLPos + offset) /= val_dev(l_col_ptr_dev(currentCol));
          tmpMem(bid * n + ridx) = val_dev(currentLPos + offset);
        }
        offset += team.team_size();
      }
      team.team_barrier();

      const unsigned subColPos = csr_diag_ptr_dev(currentCol) + wid + 1;
      const unsigned subMatSize = csr_r_ptr_dev(currentCol + 1) - csr_diag_ptr_dev(currentCol) - 1;

      int woffset = 0;
      while (subMatSize > (unsigned)woffset) {
        if (wid + woffset < (int)subMatSize) {
          offset = 0;
          unsigned subCol = csr_c_idx_dev(subColPos + woffset);
          unsigned colSize = sym_c_ptr_dev(subCol + 1) - sym_c_ptr_dev(subCol);
          while ((unsigned)offset < colSize) {
            if (tidInWarp + offset < (int)colSize) {
              unsigned subColElem = sym_c_ptr_dev(subCol) + tidInWarp + offset;
              unsigned ridx = sym_r_idx_dev(subColElem);
              if (ridx == currentCol) {
                s(wid) = val_dev(subColElem);
              }
              if (ridx > currentCol) {
                Kokkos::atomic_add(&val_dev(subColElem),
                                   -tmpMem(ridx + n * bid) * s(wid));
              }
            }
            offset += 32;
          }
        }
        woffset += team.team_size() / 32;
      }

      team.team_barrier();
      offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          tmpMem(bid * n + ridx) = 0;
        }
        offset += team.team_size();
      }
    });
}

void RL_onecol_factorizeCurrentCol(
    const int nteams,
    const int nthreads,
    Kokkos::View<const unsigned*> sym_c_ptr_dev,
    Kokkos::View<const unsigned*> sym_r_idx_dev,
    Kokkos::View<REAL*>           val_dev,
    Kokkos::View<const unsigned*> l_col_ptr_dev,
    const unsigned currentCol,
    Kokkos::View<REAL*>           tmpMem,
    const int stream,
    const unsigned n)
{
  Kokkos::parallel_for("RL_onecol_factorize",
    team_policy(nteams, nthreads),
    KOKKOS_LAMBDA(const member_type& team) {
      const int tid = team.team_rank();
      const unsigned currentLColSize = sym_c_ptr_dev(currentCol + 1) - l_col_ptr_dev(currentCol) - 1;
      const unsigned currentLPos = l_col_ptr_dev(currentCol) + tid + 1;

      int offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          val_dev(currentLPos + offset) /= val_dev(l_col_ptr_dev(currentCol));
          tmpMem(stream * n + ridx) = val_dev(currentLPos + offset);
        }
        offset += team.team_size();
      }
    });
}

void RL_onecol_factorizeCurrentCol_perturb(
    const int nteams,
    const int nthreads,
    Kokkos::View<const unsigned*> sym_c_ptr_dev,
    Kokkos::View<const unsigned*> sym_r_idx_dev,
    Kokkos::View<REAL*>           val_dev,
    Kokkos::View<const unsigned*> l_col_ptr_dev,
    const unsigned currentCol,
    Kokkos::View<REAL*>           tmpMem,
    const int stream,
    const unsigned n,
    const float pert)
{
  Kokkos::parallel_for("RL_onecol_factorize_perturb",
    team_policy(nteams, nthreads),
    KOKKOS_LAMBDA(const member_type& team) {
      const int tid = team.team_rank();
      const unsigned currentLColSize = sym_c_ptr_dev(currentCol + 1) - l_col_ptr_dev(currentCol) - 1;
      const unsigned currentLPos = l_col_ptr_dev(currentCol) + tid + 1;

      int offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          if (fabs(val_dev(l_col_ptr_dev(currentCol))) < pert)
            val_dev(l_col_ptr_dev(currentCol)) = pert;
          val_dev(currentLPos + offset) /= val_dev(l_col_ptr_dev(currentCol));
          tmpMem(stream * n + ridx) = val_dev(currentLPos + offset);
        }
        offset += team.team_size();
      }
    });
}

void RL_onecol_updateSubmat(
    const int nteams,
    const int nthreads,
    Kokkos::View<const unsigned*> sym_c_ptr_dev,
    Kokkos::View<const unsigned*> sym_r_idx_dev,
    Kokkos::View<REAL*>           val_dev,
    Kokkos::View<const unsigned*> csr_c_idx_dev,
    Kokkos::View<const unsigned*> csr_diag_ptr_dev,
    const unsigned currentCol,
    Kokkos::View<REAL*>           tmpMem,
    const int stream,
    const unsigned n)
{
  Kokkos::parallel_for("RL_onecol_updateSubmat",
    team_policy(nteams, nthreads),
    KOKKOS_LAMBDA(const member_type& team) {
      const int bid = team.league_rank();
      const int tid = team.team_rank();
      const int wid = tid / 32;
      const int tidInWarp = tid % 32;

      const unsigned subColPos = csr_diag_ptr_dev(currentCol) + bid + 1;
      const unsigned subCol = csr_c_idx_dev(subColPos);
      const unsigned colSize = sym_c_ptr_dev(subCol + 1) - sym_c_ptr_dev(subCol);

      int offset = tidInWarp;
      while ((unsigned)offset < colSize) {
        unsigned subColElem = sym_c_ptr_dev(subCol) + offset;
        unsigned ridx = sym_r_idx_dev(subColElem);
        if (ridx > currentCol) {
          Kokkos::atomic_add(&val_dev(subColElem),
                             -tmpMem(ridx + n * stream) * val_dev(sym_c_ptr_dev(subCol)));
        }
        offset += 32;
      }
    });
}

void RL_onecol_cleartmpMem(
    const int nteams,
    const int nthreads,
    Kokkos::View<const unsigned*> sym_c_ptr_dev,
    Kokkos::View<const unsigned*> sym_r_idx_dev,
    Kokkos::View<const unsigned*> l_col_ptr_dev,
    const unsigned currentCol,
    Kokkos::View<REAL*>           tmpMem,
    const int stream,
    const unsigned n)
{
  Kokkos::parallel_for("RL_onecol_clear",
    team_policy(nteams, nthreads),
    KOKKOS_LAMBDA(const member_type& team) {
      const int tid = team.team_rank();
      const unsigned currentLColSize = sym_c_ptr_dev(currentCol + 1) - l_col_ptr_dev(currentCol) - 1;
      const unsigned currentLPos = l_col_ptr_dev(currentCol) + tid + 1;

      int offset = 0;
      while (currentLColSize > (unsigned)offset) {
        if (tid + offset < (int)currentLColSize) {
          unsigned ridx = sym_r_idx_dev(currentLPos + offset);
          tmpMem(stream * n + ridx) = 0;
        }
        offset += team.team_size();
      }
    });
}

// ---- main factorization routine using Kokkos views ----------------------

void factorize(Symbolic_Matrix& A_sym, const bool PERTURB,
               std::ostream& out, std::ostream& err)
{
  Kokkos::initialize();
  {
    Timer t;
    double utime;
    const unsigned n   = A_sym.n;
    const unsigned nnz = A_sym.sym_c_ptr[n];
    const float pert   = 1e-4f;

    // Upload all arrays to device
    Kokkos::View<unsigned*> sym_c_ptr_dev("sym_c_ptr", n + 1);
    Kokkos::View<unsigned*> sym_r_idx_dev("sym_r_idx", nnz);
    Kokkos::View<REAL*>     val_dev("val",       nnz);
    Kokkos::View<unsigned*> l_col_ptr_dev("l_col_ptr", n + 1);
    Kokkos::View<unsigned*> csr_r_ptr_dev("csr_r_ptr", n + 1);
    Kokkos::View<unsigned*> csr_c_idx_dev("csr_c_idx", A_sym.csr_c_idx.size());
    Kokkos::View<unsigned*> csr_diag_ptr_dev("csr_diag_ptr", n);
    Kokkos::View<int*>      level_idx_dev("level_idx", A_sym.level_idx.size());
    Kokkos::View<REAL*>     tmpMem("tmpMem", (size_t)TMPMEMNUM * n);

    {
      auto cp = Kokkos::View<unsigned*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.sym_c_ptr.data(), n + 1);
      auto ri = Kokkos::View<unsigned*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.sym_r_idx.data(), nnz);
      auto vl = Kokkos::View<REAL*,     Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.val.data(), nnz);
      auto lc = Kokkos::View<unsigned*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.l_col_ptr.data(), n + 1);
      auto rp = Kokkos::View<unsigned*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.csr_r_ptr.data(), n + 1);
      auto cc = Kokkos::View<unsigned*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.csr_c_idx.data(), A_sym.csr_c_idx.size());
      auto dp = Kokkos::View<unsigned*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.csr_diag_ptr.data(), n);
      auto li = Kokkos::View<int*,      Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(A_sym.level_idx.data(), A_sym.level_idx.size());
      Kokkos::deep_copy(sym_c_ptr_dev, cp); Kokkos::deep_copy(sym_r_idx_dev, ri);
      Kokkos::deep_copy(val_dev, vl);       Kokkos::deep_copy(l_col_ptr_dev, lc);
      Kokkos::deep_copy(csr_r_ptr_dev, rp); Kokkos::deep_copy(csr_c_idx_dev, cc);
      Kokkos::deep_copy(csr_diag_ptr_dev, dp); Kokkos::deep_copy(level_idx_dev, li);
    }
    Kokkos::deep_copy(tmpMem, (REAL)0);

    auto sym_c_const   = Kokkos::View<const unsigned*>(sym_c_ptr_dev);
    auto sym_r_const   = Kokkos::View<const unsigned*>(sym_r_idx_dev);
    auto l_col_const   = Kokkos::View<const unsigned*>(l_col_ptr_dev);
    auto csr_r_const   = Kokkos::View<const unsigned*>(csr_r_ptr_dev);
    auto csr_c_const   = Kokkos::View<const unsigned*>(csr_c_idx_dev);
    auto csr_d_const   = Kokkos::View<const unsigned*>(csr_diag_ptr_dev);
    auto level_const   = Kokkos::View<const int*>(level_idx_dev);

    t.start();

    for (int i = 0; i < A_sym.num_lev; i++) {
      int lev_start = A_sym.level_ptr[i];
      int lev_end   = A_sym.level_ptr[i + 1];
      int lev_size  = lev_end - lev_start;

      if (lev_size > 896) {
        int dimBlock = 64;
        int j = 0;
        while (lev_size > 0) {
          int restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
          if (!PERTURB)
            RL(restCol, dimBlock, sym_c_const, sym_r_const, val_dev,
               l_col_const, csr_r_const, csr_c_const, csr_d_const, level_const,
               tmpMem, n, lev_start, j * TMPMEMNUM);
          else
            RL_perturb(restCol, dimBlock, sym_c_const, sym_r_const, val_dev,
                       l_col_const, csr_r_const, csr_c_const, csr_d_const, level_const,
                       tmpMem, n, lev_start, j * TMPMEMNUM, pert);
          j++;
          lev_size -= TMPMEMNUM;
        }
      } else if (lev_size > 448) {
        int dimBlock = 128;
        int j = 0;
        while (lev_size > 0) {
          int restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
          if (!PERTURB)
            RL(restCol, dimBlock, sym_c_const, sym_r_const, val_dev,
               l_col_const, csr_r_const, csr_c_const, csr_d_const, level_const,
               tmpMem, n, lev_start, j * TMPMEMNUM);
          else
            RL_perturb(restCol, dimBlock, sym_c_const, sym_r_const, val_dev,
                       l_col_const, csr_r_const, csr_c_const, csr_d_const, level_const,
                       tmpMem, n, lev_start, j * TMPMEMNUM, pert);
          j++;
          lev_size -= TMPMEMNUM;
        }
      } else if (lev_size > Nstreams) {
        int dimBlock = 256;
        int j = 0;
        while (lev_size > 0) {
          int restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
          if (!PERTURB)
            RL(restCol, dimBlock, sym_c_const, sym_r_const, val_dev,
               l_col_const, csr_r_const, csr_c_const, csr_d_const, level_const,
               tmpMem, n, lev_start, j * TMPMEMNUM);
          else
            RL_perturb(restCol, dimBlock, sym_c_const, sym_r_const, val_dev,
                       l_col_const, csr_r_const, csr_c_const, csr_d_const, level_const,
                       tmpMem, n, lev_start, j * TMPMEMNUM, pert);
          j++;
          lev_size -= TMPMEMNUM;
        }
      } else {
        for (int offset = 0; offset < lev_size; offset += Nstreams) {
          for (int j = 0; j < Nstreams; j++) {
            if (j + offset < lev_size) {
              unsigned currentCol = A_sym.level_idx[lev_start + j + offset];
              unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                                  - A_sym.csr_diag_ptr[currentCol] - 1;
              if (!PERTURB)
                RL_onecol_factorizeCurrentCol(1, 256, sym_c_const, sym_r_const, val_dev,
                    l_col_const, currentCol, tmpMem, j, n);
              else
                RL_onecol_factorizeCurrentCol_perturb(1, 256, sym_c_const, sym_r_const,
                    val_dev, l_col_const, currentCol, tmpMem, j, n, pert);
              if (subMatSize > 0)
                RL_onecol_updateSubmat((int)subMatSize, 256, sym_c_const, sym_r_const,
                    val_dev, csr_c_const, csr_d_const, currentCol, tmpMem, j, n);
              RL_onecol_cleartmpMem(1, 256, sym_c_const, sym_r_const, l_col_const,
                                    currentCol, tmpMem, j, n);
            }
          }
        }
      }
    }

    Kokkos::fence();
    t.elapsedUserTime(utime);
    out << "Total LU kernel execution time: " << utime << " ms" << std::endl;

    // Copy val back to host
    {
      auto h_val = Kokkos::View<REAL*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                      A_sym.val.data(), nnz);
      Kokkos::deep_copy(h_val, val_dev);
    }

#ifdef VERIFY
    unsigned err_find = 0;
    for (unsigned i = 0; i < nnz; i++)
      if (isnan(A_sym.val[i]) || isinf(A_sym.val[i])) err_find++;
    if (err_find != 0)
      err << "LU data check: NaN found!!" << std::endl;
#endif
  }
  Kokkos::finalize();
}
