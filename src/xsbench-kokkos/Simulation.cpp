#include <Kokkos_Core.hpp>
#include "XSbench_header.h"
#include <chrono>

////////////////////////////////////////////////////////////////////////////////////
// DEVICE-CALLABLE HELPER FUNCTIONS
// These correspond to the #pragma omp declare target section in the OMP version.
////////////////////////////////////////////////////////////////////////////////////

// binary search for energy on unionized energy grid
// returns lower index
template <class T>
KOKKOS_INLINE_FUNCTION
long grid_search( long n, double quarry, T A)
{
  long lowerLimit = 0;
  long upperLimit = n-1;
  long examinationPoint;
  long length = upperLimit - lowerLimit;

  while( length > 1 )
  {
    examinationPoint = lowerLimit + ( length / 2 );

    if( A[examinationPoint] > quarry )
      upperLimit = examinationPoint;
    else
      lowerLimit = examinationPoint;

    length = upperLimit - lowerLimit;
  }

  return lowerLimit;
}

// Calculates the microscopic cross section for a given nuclide & energy
template <class Double_Type, class Int_Type, class NGP_Type>
KOKKOS_INLINE_FUNCTION
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
    long n_gridpoints,
    Double_Type  egrid, Int_Type  index_data,
    NGP_Type  nuclide_grids,
    long idx, double *  xs_vector, int grid_type, int hash_bins ){
  // Variables
  double f;
  NuclideGridPoint low, high;
  long low_idx, high_idx;

  // If using only the nuclide grid, we must perform a binary search
  // to find the energy location in this particular nuclide's grid.
  if( grid_type == NUCLIDE )
  {
    // Perform binary search on the Nuclide Grid to find the index
    long offset = nuc * n_gridpoints;
    idx = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset, offset + n_gridpoints-1);

    // pull ptr from nuclide grid and check to ensure that
    // we're not reading off the end of the nuclide's grid
    if( idx == n_gridpoints - 1 )
      low_idx = idx - 1;
    else
      low_idx = idx;
  }
  else if( grid_type == UNIONIZED) // Unionized Energy Grid - we already know the index, no binary search needed.
  {
    // pull ptr from energy grid and check to ensure that
    // we're not reading off the end of the nuclide's grid
    if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
      low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1;
    else
    {
      low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc];
    }
  }
  else // Hash grid
  {
    // load lower bounding index
    int u_low = index_data[idx * n_isotopes + nuc];

    // Determine higher bounding index
    int u_high;
    if( idx == hash_bins - 1 )
      u_high = n_gridpoints - 1;
    else
      u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

    // Check edge cases to make sure energy is actually between these
    // Then, if things look good, search for gridpoint in the nuclide grid
    // within the lower and higher limits we've calculated.
    double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
    double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
    long lower;
    if( p_energy <= e_low )
      lower = nuc*n_gridpoints;
    else if( p_energy >= e_high )
      lower = nuc*n_gridpoints + n_gridpoints - 1;
    else
    {
      long offset = nuc*n_gridpoints;
      lower = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset+u_low, offset+u_high);
    }

    if( (lower % n_gridpoints) == n_gridpoints - 1 )
      low_idx = lower - 1;
    else
      low_idx = lower;
  }

  high_idx = low_idx + 1;
  low = nuclide_grids[low_idx];
  high = nuclide_grids[high_idx];

  // calculate the re-useable interpolation factor
  f = (high.energy - p_energy) / (high.energy - low.energy);

  // Total XS
  xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);

  // Elastic XS
  xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);

  // Absorbtion XS
  xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);

  // Fission XS
  xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);

  // Nu Fission XS
  xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
}

// Calculates macroscopic cross section based on a given material & energy
template <class Double_Type, class Int_Type, class NGP_Type, class E_GRID_TYPE, class INDEX_TYPE>
KOKKOS_INLINE_FUNCTION
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
    long n_gridpoints, Int_Type  num_nucs,
    Double_Type  concs,
    E_GRID_TYPE  egrid, INDEX_TYPE  index_data,
    NGP_Type  nuclide_grids,
    Int_Type  mats,
    double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
  int p_nuc; // the nuclide we are looking up
  long idx = -1;
  double conc; // the concentration of the nuclide in the material

  // cleans out macro_xs_vector
  for( int k = 0; k < 5; k++ )
    macro_xs_vector[k] = 0;

  if( grid_type == UNIONIZED )
    idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);
  else if( grid_type == HASH )
  {
    double du = 1.0 / hash_bins;
    idx = p_energy / du;
  }

  for( int j = 0; j < num_nucs[mat]; j++ )
  {
    double xs_vector[5];
    p_nuc = mats[mat*max_num_nucs + j];
    conc = concs[mat*max_num_nucs + j];
    calculate_micro_xs( p_energy, p_nuc, n_isotopes,
        n_gridpoints, egrid, index_data,
        nuclide_grids, idx, xs_vector, grid_type, hash_bins );
    for( int k = 0; k < 5; k++ )
      macro_xs_vector[k] += xs_vector[k] * conc;
  }
}

KOKKOS_INLINE_FUNCTION
double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

KOKKOS_INLINE_FUNCTION
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;

  n = n % m;

  uint64_t a_new = 1;
  uint64_t c_new = 0;

  while(n > 0)
  {
    if(n & 1)
    {
      a_new *= a;
      c_new = c_new * a + c;
    }
    c *= (a + 1);
    a *= a;

    n >>= 1;
  }

  return (a_new * seed + c_new) % m;
}

// picks a material based on a probabilistic distribution
KOKKOS_INLINE_FUNCTION
int pick_mat( unsigned long * seed )
{
  double dist[12];
  dist[0]  = 0.140;  // fuel
  dist[1]  = 0.052;  // cladding
  dist[2]  = 0.275;  // cold, borated water
  dist[3]  = 0.134;  // hot, borated water
  dist[4]  = 0.154;  // RPV
  dist[5]  = 0.064;  // Lower, radial reflector
  dist[6]  = 0.066;  // Upper reflector / top plate
  dist[7]  = 0.055;  // bottom plate
  dist[8]  = 0.008;  // bottom nozzle
  dist[9]  = 0.015;  // top nozzle
  dist[10] = 0.025;  // top of fuel assemblies
  dist[11] = 0.013;  // bottom of fuel assemblies

  double roll = LCG_random_double(seed);

  for( int i = 0; i < 12; i++ )
  {
    double running = 0;
    for( int j = i; j > 0; j-- )
      running += dist[j];
    if( roll < running )
      return i;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////

// run the simulation on a host for validation
unsigned long long
run_event_based_simulation(Inputs in, SimulationData SD, int mype)
{
  if(mype==0) printf("Beginning event based simulation on the host for verification...\n");

  int * verification = (int *) malloc(in.lookups * sizeof(int));

  if( SD.length_unionized_energy_array == 0 )
  {
    SD.length_unionized_energy_array = 1;
    SD.unionized_energy_array = (double *) malloc(sizeof(double));
  }

  if( SD.length_index_grid == 0 )
  {
    SD.length_index_grid = 1;
    SD.index_grid = (int *) malloc(sizeof(int));
  }

  #pragma omp parallel for
  for( int i = 0; i < in.lookups; i++ )
  {
    uint64_t seed = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2*i);
    double p_energy = LCG_random_double(&seed);
    int mat         = pick_mat(&seed);

    double macro_xs_vector[5] = {0};

    calculate_macro_xs(
        p_energy, mat, in.n_isotopes, in.n_gridpoints,
        SD.num_nucs, SD.concs, SD.unionized_energy_array, SD.index_grid,
        SD.nuclide_grid, SD.mats, macro_xs_vector,
        in.grid_type, in.hash_bins, SD.max_num_nucs
    );

    double max = -1.0;
    int max_idx = 0;
    for(int j = 0; j < 5; j++ )
    {
      if( macro_xs_vector[j] > max )
      {
        max = macro_xs_vector[j];
        max_idx = j;
      }
    }
    verification[i] = max_idx+1;
  }

  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification[i];

  if( SD.length_unionized_energy_array == 0 ) free(SD.unionized_energy_array);
  if( SD.length_index_grid == 0 ) free(SD.index_grid);
  free(verification);

  return verification_scalar;
}

unsigned long long
run_event_based_simulation(Inputs in, SimulationData SD,
                           int mype, double *kernel_time)
{
  if( mype == 0)
    printf("Beginning event based simulation...\n");

  if( mype == 0 )
     printf("Allocating an additional %.1lf MB of memory for verification arrays...\n",
            in.lookups * sizeof(int) /1024.0/1024.0);

  if( SD.length_unionized_energy_array == 0 )
  {
    SD.length_unionized_energy_array = 1;
    SD.unionized_energy_array = (double *) malloc(sizeof(double));
  }

  if( SD.length_index_grid == 0 )
  {
    SD.length_index_grid = 1;
    SD.index_grid = (int *) malloc(sizeof(int));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Allocate Kokkos device views and copy simulation data to device
  ////////////////////////////////////////////////////////////////////////////////
  Kokkos::View<int*>             d_num_nucs("num_nucs", SD.length_num_nucs);
  Kokkos::View<double*>          d_concs("concs", SD.length_concs);
  Kokkos::View<int*>             d_mats("mats", SD.length_mats);
  Kokkos::View<double*>          d_unionized_energy_array("unionized_energy_array",
                                                           SD.length_unionized_energy_array);
  Kokkos::View<int*>             d_index_grid("index_grid", SD.length_index_grid);
  Kokkos::View<NuclideGridPoint*> d_nuclide_grid("nuclide_grid", SD.length_nuclide_grid);
  Kokkos::View<int*>             d_verification("verification", in.lookups);

  // Copy host data to device using unmanaged host views
  {
    auto h = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
               SD.num_nucs, SD.length_num_nucs);
    Kokkos::deep_copy(d_num_nucs, h);
  }
  {
    auto h = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
               SD.concs, SD.length_concs);
    Kokkos::deep_copy(d_concs, h);
  }
  {
    auto h = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
               SD.mats, SD.length_mats);
    Kokkos::deep_copy(d_mats, h);
  }
  {
    auto h = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
               SD.unionized_energy_array, SD.length_unionized_energy_array);
    Kokkos::deep_copy(d_unionized_energy_array, h);
  }
  {
    auto h = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
               SD.index_grid, SD.length_index_grid);
    Kokkos::deep_copy(d_index_grid, h);
  }
  {
    auto h = Kokkos::View<NuclideGridPoint*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
               SD.nuclide_grid, SD.length_nuclide_grid);
    Kokkos::deep_copy(d_nuclide_grid, h);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Capture device pointers for use inside KOKKOS_LAMBDA
  ////////////////////////////////////////////////////////////////////////////////
  const int*              p_num_nucs      = d_num_nucs.data();
  const double*           p_concs         = d_concs.data();
  const int*              p_mats          = d_mats.data();
  const double*           p_energy_array  = d_unionized_energy_array.data();
  const int*              p_index_grid    = d_index_grid.data();
  const NuclideGridPoint* p_nuclide_grid  = d_nuclide_grid.data();
  int*                    p_verification  = d_verification.data();

  // Capture Inputs scalars
  const long   n_isotopes   = in.n_isotopes;
  const long   n_gridpoints = in.n_gridpoints;
  const int    lookups      = in.lookups;
  const int    grid_type    = in.grid_type;
  const int    hash_bins    = in.hash_bins;
  const int    max_num_nucs = SD.max_num_nucs;

  ////////////////////////////////////////////////////////////////////////////////
  // Begin Actual Simulation Loop
  ////////////////////////////////////////////////////////////////////////////////
  auto kstart = std::chrono::steady_clock::now();

  for (int n = 0; n < in.kernel_repeat; n++) {
    Kokkos::parallel_for("XSBench_simulation",
      Kokkos::RangePolicy<>(0, lookups),
      KOKKOS_LAMBDA(const int i) {
        // Set the initial seed value
        uint64_t seed = STARTING_SEED;

        // Forward seed to lookup index (we need 2 samples per lookup)
        seed = fast_forward_LCG(seed, 2*i);

        // Randomly pick an energy and material for the particle
        double p_energy = LCG_random_double(&seed);
        int mat         = pick_mat(&seed);

        double macro_xs_vector[5] = {0};

        // Perform macroscopic Cross Section Lookup
        calculate_macro_xs(
            p_energy, mat, n_isotopes, n_gridpoints,
            p_num_nucs, p_concs,
            p_energy_array, p_index_grid,
            p_nuclide_grid, p_mats,
            macro_xs_vector, grid_type, hash_bins, max_num_nucs
        );

        // Find max xs_vector index for verification
        double max = -1.0;
        int max_idx = 0;
        for(int j = 0; j < 5; j++ )
        {
          if( macro_xs_vector[j] > max )
          {
            max = macro_xs_vector[j];
            max_idx = j;
          }
        }
        p_verification[i] = max_idx+1;
      });
  }

  Kokkos::fence();

  auto kstop = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kstop - kstart).count();
  *kernel_time = (ktime * 1e-9) / in.kernel_repeat;

  ////////////////////////////////////////////////////////////////////////////////
  // Copy verification array back to host and reduce
  ////////////////////////////////////////////////////////////////////////////////
  auto h_verification = Kokkos::create_mirror_view(d_verification);
  Kokkos::deep_copy(h_verification, d_verification);

  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += h_verification(i);

  if( SD.length_unionized_energy_array == 0 ) free(SD.unionized_energy_array);
  if( SD.length_index_grid == 0 ) free(SD.index_grid);

  return verification_scalar;
}
