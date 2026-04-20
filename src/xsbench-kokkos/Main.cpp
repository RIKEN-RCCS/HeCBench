#include <Kokkos_Core.hpp>
#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

int main( int argc, char* argv[] )
{
  // =====================================================================
  // Initialization & Command Line Read-In
  // =====================================================================
  int version = 19;
  int mype = 0;
  double omp_start, omp_end;
  int nprocs = 1;
  unsigned long long verification[2];

#ifdef MPI
  MPI_Status stat;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
#endif

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    // Process CLI Fields -- store in "Inputs" structure
    Inputs in = read_CLI( argc, argv );

    // Print-out of Input Summary
    if( mype == 0 )
      print_inputs( in, nprocs, version );

    // =====================================================================
    // Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
    // =====================================================================

    SimulationData SD;

    if( in.binary_mode == READ )
      SD = binary_read(in);
    else
      SD = grid_init_do_not_profile( in, mype );

    if( in.binary_mode == WRITE && mype == 0 )
      binary_write(in, SD);

    // =====================================================================
    // Cross Section (XS) Parallel Lookup Simulation
    // =====================================================================

    if( mype == 0 )
    {
      printf("\n");
      border_print();
      center_print("SIMULATION", 79);
      border_print();
    }

    // Start Simulation Timer
    omp_start = get_time();
    double kernel_time;

    // Run simulation
    if( in.simulation_method == EVENT_BASED )
    {
      if( in.kernel_id == 0 )
      {
        verification[0] = run_event_based_simulation(in, SD, mype, &kernel_time);
        verification[1] = run_event_based_simulation(in, SD, mype);
      }
      else
      {
        printf("Error: No kernel ID %d found!\n", in.kernel_id);
        exit(1);
      }
    }
    else
    {
      printf("History-based simulation not implemented. Instead,\nuse the event-based method with \"-m event\" argument.\n");
      exit(1);
    }

    if( mype == 0)
    {
      printf("\n" );
      printf("Simulation complete.\n" );
    }

    // End Simulation Timer
    omp_end = get_time();

    // =====================================================================
    // Output Results & Finalize
    // =====================================================================

    print_results( in, mype, omp_end-omp_start, nprocs, verification, kernel_time );
  }

  Kokkos::finalize();

#ifdef MPI
  MPI_Finalize();
#endif

  return 0;
}
