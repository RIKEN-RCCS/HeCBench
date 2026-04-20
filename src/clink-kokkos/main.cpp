#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>
#include <Kokkos_Core.hpp>

#define N 8192
#define WGS 256
#define SAMPLE_TEST_LEN 20000

KOKKOS_INLINE_FUNCTION
float sigmoid(float x) { return 1.f / (1.f + Kokkos::exp(-x)); }

void init(const char* work_path, const char* input_filename, const char* weight_filename,
          float* sample_input, float* inW, float* intW, float* intB, float* outW, float* outB)
{
  char file_name[256];
  FILE *fp;
  float weightVal;

  snprintf(file_name, sizeof(file_name), "%s/%s", work_path, input_filename);
  fp = fopen(file_name, "r");
  if (!fp) { printf("File %s cannot be opened.\n", input_filename); exit(-1); }
  for (int i = 0; i < SAMPLE_TEST_LEN; ++i) fscanf(fp, "%f", &sample_input[i]);
  fclose(fp);

  for (int i = 1; i < N; i++)
    memcpy(sample_input + i*SAMPLE_TEST_LEN, sample_input, SAMPLE_TEST_LEN*sizeof(float));

  snprintf(file_name, sizeof(file_name), "%s/%s", work_path, weight_filename);
  fp = fopen(file_name, "r");
  if (!fp) { printf("File %s cannot be opened.\n", weight_filename); exit(-1); }
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 5; ++i) { fscanf(fp,"%f",&weightVal); inW[j*5+i]=weightVal; }
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 5; ++j)
      for (int i = 0; i < 5; ++i) { fscanf(fp,"%f",&weightVal); intW[k*25+j*5+i]=weightVal; }
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 5; ++i) { fscanf(fp,"%f",&weightVal); intB[j*5+i]=weightVal; }
  for (int i = 0; i < 5; ++i) { fscanf(fp,"%f",&weightVal); outW[i]=weightVal; }
  fscanf(fp,"%f",&weightVal); *outB=weightVal;
  fclose(fp);
}

long lstm_n5(const float* x, const float* inW, const float* intW, const float* intB,
             const float* outW, const float* outB, float* y)
{
  Kokkos::View<float*> d_x("x", (size_t)N*SAMPLE_TEST_LEN);
  Kokkos::View<float*> d_inW("inW", 20);
  Kokkos::View<float*> d_intW("intW", 100);
  Kokkos::View<float*> d_intB("intB", 20);
  Kokkos::View<float*> d_outW("outW", 5);
  Kokkos::View<float[1]> d_outB("outB");
  Kokkos::View<float*> d_y("y", (size_t)N*SAMPLE_TEST_LEN);

  auto h_x    = Kokkos::create_mirror_view(d_x);
  auto h_inW  = Kokkos::create_mirror_view(d_inW);
  auto h_intW = Kokkos::create_mirror_view(d_intW);
  auto h_intB = Kokkos::create_mirror_view(d_intB);
  auto h_outW = Kokkos::create_mirror_view(d_outW);
  auto h_outB = Kokkos::create_mirror_view(d_outB);

  for (size_t i=0;i<(size_t)N*SAMPLE_TEST_LEN;i++) h_x(i)=x[i];
  for (int i=0;i<20;i++) h_inW(i)=inW[i];
  for (int i=0;i<100;i++) h_intW(i)=intW[i];
  for (int i=0;i<20;i++) h_intB(i)=intB[i];
  for (int i=0;i<5;i++) h_outW(i)=outW[i];
  h_outB(0)=*outB;

  Kokkos::deep_copy(d_x,h_x); Kokkos::deep_copy(d_inW,h_inW);
  Kokkos::deep_copy(d_intW,h_intW); Kokkos::deep_copy(d_intB,h_intB);
  Kokkos::deep_copy(d_outW,h_outW); Kokkos::deep_copy(d_outB,h_outB);

  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for("lstm", N, KOKKOS_LAMBDA(int gid) {
    float h_st[5]={0,0,0,0,0}, c_st[5]={0,0,0,0,0};
    float i_st[5], f_st[5], o_st[5], g_st[5];

    for (int t = 0; t < SAMPLE_TEST_LEN; ++t) {
      float v = d_x(gid*(int)SAMPLE_TEST_LEN + t);
      for (int j=0;j<5;j++){
        i_st[j]=d_inW(j)*v;
        for(int i=0;i<5;i++) i_st[j]+=h_st[i]*d_intW(j*5+i);
        i_st[j]+=d_intB(j); i_st[j]=sigmoid(i_st[j]);
      }
      for (int j=0;j<5;j++){
        f_st[j]=d_inW(5+j)*v;
        for(int i=0;i<5;i++) f_st[j]+=h_st[i]*d_intW(25+j*5+i);
        f_st[j]+=d_intB(5+j); f_st[j]=sigmoid(f_st[j]);
      }
      for (int j=0;j<5;j++){
        o_st[j]=d_inW(10+j)*v;
        for(int i=0;i<5;i++) o_st[j]+=h_st[i]*d_intW(50+j*5+i);
        o_st[j]+=d_intB(10+j); o_st[j]=sigmoid(o_st[j]);
      }
      for (int j=0;j<5;j++){
        g_st[j]=d_inW(15+j)*v;
        for(int i=0;i<5;i++) g_st[j]+=h_st[i]*d_intW(75+j*5+i);
        g_st[j]+=d_intB(15+j); g_st[j]=Kokkos::tanh(g_st[j]);
      }
      for (int j=0;j<5;j++){
        c_st[j]=c_st[j]*f_st[j]+g_st[j]*i_st[j];
        h_st[j]=Kokkos::tanh(c_st[j])*o_st[j];
      }
      float b=d_outB(0);
      for(int j=0;j<5;j++) b+=h_st[j]*d_outW(j);
      d_y(gid*(int)SAMPLE_TEST_LEN+t)=b;
    }
  });
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  long time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

  auto h_y = Kokkos::create_mirror_view(d_y);
  Kokkos::deep_copy(h_y, d_y);
  for (size_t i=0;i<(size_t)N*SAMPLE_TEST_LEN;i++) y[i]=h_y(i);

  return time;
}

int main(int argc, char* argv[])
{
  if (argc != 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int repeat = atoi(argv[1]);

  float* sample_input = (float*) malloc(sizeof(float)*(size_t)N*SAMPLE_TEST_LEN);
  float* infer1_out   = (float*) malloc(sizeof(float)*(size_t)N*SAMPLE_TEST_LEN);
  float* infer2_out   = (float*) malloc(sizeof(float)*(size_t)N*SAMPLE_TEST_LEN);
  float inW[20], intW[100], intB[20], outW[5], outB;

  Kokkos::initialize(argc, argv);
  {
    long kernel_time = 0;
    for (int n = 0; n < repeat; n++) {
      init("./", "input.hpp", "weight_1.hpp", sample_input, inW, intW, intB, outW, &outB);
      auto t0 = std::chrono::steady_clock::now();
      kernel_time += lstm_n5(sample_input, inW, intW, intB, outW, &outB, infer1_out);
      auto t1 = std::chrono::steady_clock::now();
      std::cout << "Device offload time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
                << " ms\n";

      init("./", "input.hpp", "weight_2.hpp", sample_input, inW, intW, intB, outW, &outB);
      t0 = std::chrono::steady_clock::now();
      kernel_time += lstm_n5(sample_input, inW, intW, intB, outW, &outB, infer2_out);
      t1 = std::chrono::steady_clock::now();
      std::cout << "Device offload time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
                << " ms\n";
    }
    std::cout << "Average kernel time: " << kernel_time * 1e-6 / (2 * repeat) << " ms\n";
  }
  Kokkos::finalize();

  free(sample_input); free(infer1_out); free(infer2_out);
  printf("Processing complete.\n");
  return 0;
}
