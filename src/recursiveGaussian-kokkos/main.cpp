#define CLAMP_TO_EDGE

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>

// ─── GaussParms ────────────────────────────────────────────────────────────
typedef struct {
  float nsigma, alpha, ema, ema2;
  float b1, b2, a0, a1, a2, a3;
  float coefp, coefn;
} GaussParms;

static GaussParms GP;

void PreProcessGaussParms(float fSigma, int iOrder, GaussParms *pGP) {
  pGP->nsigma = fSigma;
  pGP->alpha  = 1.695f / pGP->nsigma;
  pGP->ema    = expf(-pGP->alpha);
  pGP->ema2   = expf(-2.0f * pGP->alpha);
  pGP->b1     = -2.0f * pGP->ema;
  pGP->b2     = pGP->ema2;
  pGP->a0 = pGP->a1 = pGP->a2 = pGP->a3 = 0.0f;
  pGP->coefp  = 0.0f;
  pGP->coefn  = 0.0f;
  const float ea = pGP->ema;
  const float ea2 = pGP->ema2;
  switch (iOrder) {
  case 0: {
    const float k = (1.0f - ea) * (1.0f - ea) /
                    (1.0f + (2.0f * pGP->alpha * ea) - ea2);
    pGP->a0 = k;
    pGP->a1 = k * (pGP->alpha - 1.0f) * ea;
    pGP->a2 = k * (pGP->alpha + 1.0f) * ea;
    pGP->a3 = -k * ea2;
  } break;
  case 1:
    pGP->a0 = (1.0f - ea) * (1.0f - ea);
    pGP->a2 = -pGP->a0;
    break;
  case 2: {
    const float k  = -(ea2 - 1.0f) / (2.0f * pGP->alpha * ea);
    float kn = -2.0f * (-1.0f + 3*ea - 3*ea*ea + ea*ea*ea);
    kn /= (3*ea + 1.0f + 3*ea*ea + ea*ea*ea);
    pGP->a0 = kn;
    pGP->a1 = -kn * (1.0f + k * pGP->alpha) * ea;
    pGP->a2 =  kn * (1.0f - k * pGP->alpha) * ea;
    pGP->a3 = -kn * ea2;
  } break;
  }
  pGP->coefp = (pGP->a0 + pGP->a1) / (1.0f + pGP->b1 + pGP->b2);
  pGP->coefn = (pGP->a2 + pGP->a3) / (1.0f + pGP->b1 + pGP->b2);
}

// ─── Device float4 helpers ─────────────────────────────────────────────────
struct float4k { float x, y, z, w; };

KOKKOS_INLINE_FUNCTION float4k operator*(float4k a, float4k b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w};
}
KOKKOS_INLINE_FUNCTION float4k operator*(float4k a, float b) {
  return {a.x*b, a.y*b, a.z*b, a.w*b};
}
KOKKOS_INLINE_FUNCTION float4k operator+(float4k a, float4k b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}
KOKKOS_INLINE_FUNCTION float4k operator-(float4k a, float4k b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w};
}
KOKKOS_INLINE_FUNCTION float4k rgbaUintToFloat4(unsigned int p) {
  return { (float)(p & 0xff),
           (float)((p >> 8)  & 0xff),
           (float)((p >> 16) & 0xff),
           (float)((p >> 24) & 0xff) };
}
KOKKOS_INLINE_FUNCTION unsigned int rgbaFloat4ToUint(float4k c) {
  unsigned int r = 0;
  r |= 0x000000FFu & (unsigned int)c.x;
  r |= 0x0000FF00u & ((unsigned int)c.y << 8);
  r |= 0x00FF0000u & ((unsigned int)c.z << 16);
  r |= 0xFF000000u & ((unsigned int)c.w << 24);
  return r;
}

// ─── Transpose ─────────────────────────────────────────────────────────────
void Transpose(const Kokkos::View<unsigned int *> &d_in,
               Kokkos::View<unsigned int *>       &d_out,
               int iWidth, int iHeight)
{
  const int TILE = 16;
  const int numTeamsX = (iWidth  + TILE - 1) / TILE;
  const int numTeamsY = (iHeight + TILE - 1) / TILE;
  const int numTeams  = numTeamsX * numTeamsY;

  using ScratchPad = Kokkos::View<unsigned int *,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const int scratch_bytes = ScratchPad::shmem_size(TILE * (TILE + 1));

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(numTeams, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
      ScratchPad tile(team.team_scratch(0), TILE * (TILE + 1));

      const int tidX = team.league_rank() % numTeamsX;
      const int tidY = team.league_rank() / numTeamsX;

      // Phase 1: load TILE×TILE block from input into scratch
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TILE * TILE), [&](int idx) {
        const int lidX = idx % TILE;
        const int lidY = idx / TILE;
        unsigned int xIndex = tidX * TILE + lidX;
        unsigned int yIndex = tidY * TILE + lidY;
        if (xIndex < (unsigned)iWidth && yIndex < (unsigned)iHeight)
          tile[lidY * (TILE + 1) + lidX] = d_in[yIndex * iWidth + xIndex];
      });
      team.team_barrier();

      // Phase 2: write transposed block to output
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TILE * TILE), [&](int idx) {
        const int lidX = idx % TILE;
        const int lidY = idx / TILE;
        unsigned int xIndex = tidY * TILE + lidX;
        unsigned int yIndex = tidX * TILE + lidY;
        if (xIndex < (unsigned)iHeight && yIndex < (unsigned)iWidth)
          d_out[yIndex * iHeight + xIndex] = tile[lidX * (TILE + 1) + lidY];
      });
    });
}

// ─── SimpleRecursiveRGBA ───────────────────────────────────────────────────
void SimpleRecursiveRGBA(const Kokkos::View<unsigned int *> &d_in,
                         Kokkos::View<unsigned int *>       &d_out,
                         int iWidth, int iHeight, float a)
{
  Kokkos::parallel_for(iWidth, KOKKOS_LAMBDA(int X) {
    const unsigned int *col_in  = d_in.data()  + X;
    unsigned int       *col_out = d_out.data() + X;

    float4k yp = rgbaUintToFloat4(*col_in);
    for (int Y = 0; Y < iHeight; Y++) {
      float4k xc = rgbaUintToFloat4(*col_in);
      float4k yc = xc + (yp - xc) * a;
      *col_out = rgbaFloat4ToUint(yc);
      yp = yc;
      col_in  += iWidth;
      col_out += iWidth;
    }
    col_in  -= iWidth;
    col_out -= iWidth;

    yp = rgbaUintToFloat4(*col_in);
    for (int Y = iHeight - 1; Y >= 0; Y--) {
      float4k xc = rgbaUintToFloat4(*col_in);
      float4k yc = xc + (yp - xc) * a;
      *col_out = rgbaFloat4ToUint((rgbaUintToFloat4(*col_out) + yc) * 0.5f);
      yp = yc;
      col_in  -= iWidth;
      col_out -= iWidth;
    }
  });
}

// ─── RecursiveRGBA ─────────────────────────────────────────────────────────
void RecursiveRGBA(const Kokkos::View<unsigned int *> &d_in,
                   Kokkos::View<unsigned int *>       &d_out,
                   int iWidth, int iHeight,
                   float a0, float a1, float a2, float a3,
                   float b1, float b2, float coefp, float coefn)
{
  Kokkos::parallel_for(iWidth, KOKKOS_LAMBDA(int X) {
    const unsigned int *col_in  = d_in.data()  + X;
    unsigned int       *col_out = d_out.data() + X;

    float4k xp = {0,0,0,0}, yp = {0,0,0,0}, yb = {0,0,0,0};
#ifdef CLAMP_TO_EDGE
    xp = rgbaUintToFloat4(*col_in);
    yb = xp * coefp;
    yp = yb;
#endif

    for (int Y = 0; Y < iHeight; Y++) {
      float4k xc = rgbaUintToFloat4(*col_in);
      float4k yc = xc * a0 + xp * a1 - yp * b1 - yb * b2;
      *col_out = rgbaFloat4ToUint(yc);
      xp = xc; yb = yp; yp = yc;
      col_in  += iWidth;
      col_out += iWidth;
    }
    col_in  -= iWidth;
    col_out -= iWidth;

    float4k xn = {0,0,0,0}, xa = {0,0,0,0}, yn = {0,0,0,0}, ya = {0,0,0,0};
#ifdef CLAMP_TO_EDGE
    xn = rgbaUintToFloat4(*col_in);
    xa = xn;
    yn = xn * coefn;
    ya = yn;
#endif

    for (int Y = iHeight - 1; Y >= 0; Y--) {
      float4k xc = rgbaUintToFloat4(*col_in);
      float4k yc = xn * a2 + xa * a3 - yn * b1 - ya * b2;
      xa = xn; xn = xc; ya = yn; yn = yc;
      *col_out = rgbaFloat4ToUint(rgbaUintToFloat4(*col_out) + yc);
      col_in  -= iWidth;
      col_out -= iWidth;
    }
  });
}

// ─── GPU wrapper (matches original GPUGaussianFilterRGBA structure) ─────────
double GPUGaussianFilterRGBA(
    Kokkos::View<unsigned int *> &d_input,
    Kokkos::View<unsigned int *> &d_tmp,
    Kokkos::View<unsigned int *> &d_output,
    const Kokkos::View<unsigned int *>::HostMirror &h_input,
    int iWidth, int iHeight, const GaussParms *pGP)
{
  const unsigned int szBuff = iWidth * iHeight;

  // Upload input each call to match OMP benchmark semantics
  Kokkos::deep_copy(d_input, h_input);

  auto start = std::chrono::steady_clock::now();

  RecursiveRGBA(d_input, d_tmp, iWidth, iHeight,
                pGP->a0, pGP->a1, pGP->a2, pGP->a3,
                pGP->b1, pGP->b2, pGP->coefp, pGP->coefn);

  Transpose(d_tmp, d_output, iWidth, iHeight);

  RecursiveRGBA(d_output, d_tmp, iHeight, iWidth,
                pGP->a0, pGP->a1, pGP->a2, pGP->a3,
                pGP->b1, pGP->b2, pGP->coefp, pGP->coefn);

  Transpose(d_tmp, d_output, iHeight, iWidth);
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

// ─── Host reference (scalar) ───────────────────────────────────────────────
static void hostRecursiveColumnRGBA(const unsigned int *in, unsigned int *out,
                                    int iWidth, int iHeight,
                                    float a0, float a1, float a2, float a3,
                                    float b1, float b2, float coefp, float coefn)
{
  for (int X = 0; X < iWidth; X++) {
    float xp[4]={0},yp[4]={0},yb[4]={0};
#ifdef CLAMP_TO_EDGE
    unsigned int pv = in[X];
    for (int c=0;c<4;c++){xp[c]=(float)((pv>>(c*8))&0xff); yb[c]=xp[c]*coefp; yp[c]=yb[c];}
#endif
    for (int Y = 0; Y < iHeight; Y++) {
      unsigned int pix = in[Y*iWidth+X];
      float xc[4]; for(int c=0;c<4;c++) xc[c]=(float)((pix>>(c*8))&0xff);
      float yc[4];
      for(int c=0;c<4;c++) yc[c]=a0*xc[c]+a1*xp[c]-b1*yp[c]-b2*yb[c];
      unsigned int v=0;
      for(int c=0;c<4;c++) v |= ((unsigned int)yc[c])<<(c*8);
      out[Y*iWidth+X]=v;
      for(int c=0;c<4;c++){xp[c]=xc[c];yb[c]=yp[c];yp[c]=yc[c];}
    }
    float xn[4]={0},xa[4]={0},yn[4]={0},ya[4]={0};
#ifdef CLAMP_TO_EDGE
    {unsigned int pv=in[(iHeight-1)*iWidth+X]; for(int c=0;c<4;c++){xn[c]=(float)((pv>>(c*8))&0xff);xa[c]=xn[c];yn[c]=xn[c]*coefn;ya[c]=yn[c];}}
#endif
    for (int Y = iHeight-1; Y >= 0; Y--) {
      unsigned int pix = in[Y*iWidth+X];
      float xc[4]; for(int c=0;c<4;c++) xc[c]=(float)((pix>>(c*8))&0xff);
      float yc[4];
      for(int c=0;c<4;c++) yc[c]=a2*xn[c]+a3*xa[c]-b1*yn[c]-b2*ya[c];
      for(int c=0;c<4;c++){xa[c]=xn[c];xn[c]=xc[c];ya[c]=yn[c];yn[c]=yc[c];}
      unsigned int cur=out[Y*iWidth+X];
      unsigned int v=0;
      for(int c=0;c<4;c++) v |= (((unsigned int)((float)((cur>>(c*8))&0xff)+yc[c]))<<(c*8));
      out[Y*iWidth+X]=v;
    }
  }
}
static void hostTranspose(const unsigned int *in, unsigned int *out, int W, int H) {
  for(int Y=0;Y<H;Y++) for(int X=0;X<W;X++) out[X*H+Y]=in[Y*W+X];
}
static void HostRecursiveGaussianRGBA(const unsigned int *input, unsigned int *tmp,
                                      unsigned int *output, int W, int H, const GaussParms *pGP)
{
  hostRecursiveColumnRGBA(input,  tmp,    W, H, pGP->a0,pGP->a1,pGP->a2,pGP->a3,pGP->b1,pGP->b2,pGP->coefp,pGP->coefn);
  hostTranspose(tmp, output, W, H);
  hostRecursiveColumnRGBA(output, tmp,    H, W, pGP->a0,pGP->a1,pGP->a2,pGP->a3,pGP->b1,pGP->b2,pGP->coefp,pGP->coefn);
  hostTranspose(tmp, output, H, W);
}

// ─── Simple comparison ─────────────────────────────────────────────────────
static bool compareImages(const unsigned int *ref, const unsigned int *test,
                          unsigned int n, float tol)
{
  int mismatches = 0;
  for (unsigned int i = 0; i < n; i++) {
    for (int c = 0; c < 4; c++) {
      float r = (float)((ref[i]  >> (c*8)) & 0xff);
      float t = (float)((test[i] >> (c*8)) & 0xff);
      if (fabsf(r - t) > tol) { mismatches++; break; }
    }
  }
  return mismatches == 0;
}

// ─── main ──────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <image_width> <image_height> <repeat>\n", argv[0]);
    return 1;
  }
  const int iW      = atoi(argv[1]);
  const int iH      = atoi(argv[2]);
  const int iCycles = atoi(argv[3]);
  printf("Image Width = %d, Height = %d\n\n", iW, iH);

  const float fSigma = 10.0f;
  const int   iOrder = 0;
  PreProcessGaussParms(fSigma, iOrder, &GP);

  Kokkos::initialize(argc, argv);
  {
    const unsigned int szBuff = (unsigned int)(iW * iH);
    const unsigned int szBytes = szBuff * sizeof(unsigned int);

    // Allocate and fill synthetic RGBA host image (repeatable)
    unsigned int *h_input_raw = new unsigned int[szBuff];
    srand(42);
    for (unsigned int i = 0; i < szBuff; i++) h_input_raw[i] = rand();

    // Device views
    Kokkos::View<unsigned int *> d_input ("input",  szBuff);
    Kokkos::View<unsigned int *> d_tmp   ("tmp",    szBuff);
    Kokkos::View<unsigned int *> d_output("output", szBuff);

    // Host mirrors
    auto h_input  = Kokkos::create_mirror_view(d_input);
    auto h_output = Kokkos::create_mirror_view(d_output);

    for (unsigned int i = 0; i < szBuff; i++) h_input(i) = h_input_raw[i];

    // Warmup
    GPUGaussianFilterRGBA(d_input, d_tmp, d_output, h_input, iW, iH, &GP);

    printf("Running GPUGaussianFilterRGBA for %d cycles...\n\n", iCycles);
    double totalTime = 0.0;
    for (int i = 0; i < iCycles; i++) {
      totalTime += GPUGaussianFilterRGBA(d_input, d_tmp, d_output, h_input, iW, iH, &GP);
    }
    printf("Average execution time of kernels: %f (s)\n",
           (totalTime * 1e-9) / iCycles);

    // Copy output back for verification
    Kokkos::deep_copy(h_output, d_output);

    // CPU reference
    unsigned int *h_golden = new unsigned int[szBuff];
    unsigned int *h_tmp    = new unsigned int[szBuff];
    HostRecursiveGaussianRGBA(h_input_raw, h_tmp, h_golden, iW, iH, &GP);

    printf("Comparing GPU Result to CPU Result...\n");
    bool match = compareImages(h_golden, h_output.data(), szBuff, 1.0f);
    printf("\nGPU Result %s CPU Result within tolerance...\n",
           match ? "matches" : "DOESN'T match");

    delete[] h_input_raw;
    delete[] h_golden;
    delete[] h_tmp;
  }
  Kokkos::finalize();
  return 0;
}
