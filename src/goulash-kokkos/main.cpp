#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

// Inline secs_elapsed from goulash-cuda/utils.h
static double get_raw_secs() {
  struct timeval ts;
  if (gettimeofday(&ts, NULL) != 0) return -1.0;
  return (double)ts.tv_sec + (double)ts.tv_usec * 1e-6;
}

static double get_base_time(double new_time) {
  static double base_time = -1.0;
  if (new_time >= 0.0) base_time = new_time;
  return base_time;
}

static double secs_elapsed() {
  double new_time  = get_raw_secs();
  double base_time = get_base_time(-1.0);
  if (base_time < 0.0) base_time = get_base_time(new_time);
  return new_time - base_time;
}

// Reference implementation (CPU)
static void reference(double* __restrict m_gate, const long nCells,
                      const double* __restrict Vm)
{
  for (long ii = 0; ii < nCells; ii++) {
    double sum1, sum2;
    const double x = Vm[ii];
    const int Mhu_l = 10, Mhu_m = 5;
    const double Mhu_a[] = {
      9.9632117206253790e-01, 4.0825738726469545e-02, 6.3401613233199589e-04,
      4.4158436861700431e-06, 1.1622058324043520e-08, 1.0000000000000000e+00,
      4.0568375699663400e-02, 6.4216825832642788e-04, 4.2661664422410096e-06,
      1.3559930396321903e-08,-1.3573468728873069e-11,-4.2594802366702580e-13,
      7.6779952208246166e-15, 1.4260675804433780e-16,-2.6656212072499249e-18};

    sum1 = 0;
    for (int j = Mhu_m - 1; j >= 0; j--) sum1 = Mhu_a[j] + x * sum1;
    sum2 = 0;
    int k = Mhu_m + Mhu_l - 1;
    for (int j = k; j >= Mhu_m; j--) sum2 = Mhu_a[j] + x * sum2;
    double mhu = sum1 / sum2;

    const int Tau_m = 18;
    const double Tau_a[] = {
      1.7765862602413648e+01*0.02, 5.0010202770602419e-02*0.02,-7.8002064070783474e-04*0.02,
     -6.9399661775931530e-05*0.02, 1.6936588308244311e-06*0.02, 5.4629017090963798e-07*0.02,
     -1.3805420990037933e-08*0.02,-8.0678945216155694e-10*0.02, 1.6209833004622630e-11*0.02,
      6.5130101230170358e-13*0.02,-6.9931705949674988e-15*0.02,-3.1161210504114690e-16*0.02,
      5.0166191902609083e-19*0.02, 7.8608831661430381e-20*0.02, 4.3936315597226053e-22*0.02,
     -7.0535966258003289e-24*0.02,-9.0473475495087118e-26*0.02,-2.9878427692323621e-28*0.02,
      1.0000000000000000e+00};

    sum1 = 0;
    for (int j = Tau_m - 1; j >= 0; j--) sum1 = Tau_a[j] + x * sum1;
    double tauR = sum1;
    m_gate[ii] += (mhu - m_gate[ii]) * (1 - exp(-tauR));
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <Iterations> <Kernel_GBs_used>\n\n", argv[0]);
    return 1;
  }

  long iterations      = atol(argv[1]);
  double kernel_mem_used = atof(argv[2]);

  long nCells = (long)((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) /
                       (sizeof(double) * 2));
  printf("Number of cells: %ld\n", nCells);

  double* m_gate   = (double*)calloc(nCells, sizeof(double));
  double* m_gate_h = (double*)calloc(nCells, sizeof(double));
  double* Vm       = (double*)calloc(nCells, sizeof(double));

  if (!m_gate || !m_gate_h || !Vm) {
    printf("Allocation failed\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_m_gate("m_gate", nCells);
    Kokkos::View<double*> d_Vm("Vm", nCells);

    Kokkos::deep_copy(d_m_gate, 0.0);
    Kokkos::deep_copy(d_Vm,     0.0);

    double kernel_starttime = 0.0, kernel_endtime = 0.0;

    for (long itime = 0; itime <= iterations; itime++) {
      if (itime == 1) {
        // Mirror OMP behaviour: copy device->host after the warmup iteration,
        // before timing begins.  This is what the OMP target-data region
        // leaves in m_gate when it only has map(to:...).
        Kokkos::fence();
        auto h_snap = Kokkos::create_mirror_view(d_m_gate);
        Kokkos::deep_copy(h_snap, d_m_gate);
        for (long i = 0; i < nCells; i++) m_gate[i] = h_snap(i);
        kernel_starttime = secs_elapsed();
      }

      Kokkos::parallel_for("gate", nCells, KOKKOS_LAMBDA(long i) {
        double sum1, sum2;
        const double x = d_Vm(i);
        const int Mhu_l = 10, Mhu_m = 5;
        const double Mhu_a[] = {
          9.9632117206253790e-01, 4.0825738726469545e-02, 6.3401613233199589e-04,
          4.4158436861700431e-06, 1.1622058324043520e-08, 1.0000000000000000e+00,
          4.0568375699663400e-02, 6.4216825832642788e-04, 4.2661664422410096e-06,
          1.3559930396321903e-08,-1.3573468728873069e-11,-4.2594802366702580e-13,
          7.6779952208246166e-15, 1.4260675804433780e-16,-2.6656212072499249e-18};

        sum1 = 0;
        for (int j = Mhu_m - 1; j >= 0; j--) sum1 = Mhu_a[j] + x * sum1;
        sum2 = 0;
        int k = Mhu_m + Mhu_l - 1;
        for (int j = k; j >= Mhu_m; j--) sum2 = Mhu_a[j] + x * sum2;
        double mhu = sum1 / sum2;

        const int Tau_m = 18;
        const double Tau_a[] = {
          1.7765862602413648e+01*0.02, 5.0010202770602419e-02*0.02,-7.8002064070783474e-04*0.02,
         -6.9399661775931530e-05*0.02, 1.6936588308244311e-06*0.02, 5.4629017090963798e-07*0.02,
         -1.3805420990037933e-08*0.02,-8.0678945216155694e-10*0.02, 1.6209833004622630e-11*0.02,
          6.5130101230170358e-13*0.02,-6.9931705949674988e-15*0.02,-3.1161210504114690e-16*0.02,
          5.0166191902609083e-19*0.02, 7.8608831661430381e-20*0.02, 4.3936315597226053e-22*0.02,
         -7.0535966258003289e-24*0.02,-9.0473475495087118e-26*0.02,-2.9878427692323621e-28*0.02,
          1.0000000000000000e+00};

        sum1 = 0;
        for (int j = Tau_m - 1; j >= 0; j--) sum1 = Tau_a[j] + x * sum1;
        double tauR = sum1;
        d_m_gate(i) += (mhu - d_m_gate(i)) * (1.0 - exp(-tauR));
      });
      Kokkos::fence();
    }

    kernel_endtime = secs_elapsed();
    double kernel_runtime = kernel_endtime - kernel_starttime;
    printf("total kernel time %lf(s) for %ld iterations\n", kernel_runtime, iterations - 1);
  }
  Kokkos::finalize();

  // Reference computation (starts from m_gate_h = 0, Vm = 0)
  reference(m_gate_h, nCells, Vm);

  bool ok = true;
  for (long i = 0; i < nCells; i++) {
    if (fabs(m_gate[i] - m_gate_h[i]) > 1e-6) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(m_gate);
  free(m_gate_h);
  free(Vm);
  return 0;
}
