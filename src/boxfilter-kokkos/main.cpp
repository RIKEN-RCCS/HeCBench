#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef struct { unsigned char x,y,z,w; } uchar4;
typedef struct { float x,y,z,w; } float4;

KOKKOS_INLINE_FUNCTION float4 rgbaUintToFloat4(unsigned int c) {
  float4 r; r.x=c&0xff; r.y=(c>>8)&0xff; r.z=(c>>16)&0xff; r.w=(c>>24)&0xff; return r;
}
KOKKOS_INLINE_FUNCTION uchar4 rgbaUintToUchar4(unsigned int c) {
  uchar4 r; r.x=c&0xff; r.y=(c>>8)&0xff; r.z=(c>>16)&0xff; r.w=(c>>24)&0xff; return r;
}
KOKKOS_INLINE_FUNCTION unsigned int rgbaFloat4ToUint(float4 r, float fScale) {
  unsigned int p=0;
  p|=0x000000FFu&(unsigned int)(r.x*fScale);
  p|=0x0000FF00u&(((unsigned int)(r.y*fScale))<<8);
  p|=0x00FF0000u&(((unsigned int)(r.z*fScale))<<16);
  p|=0xFF000000u&(((unsigned int)(r.w*fScale))<<24);
  return p;
}

const unsigned int RADIUS = 10;
const float SCALE = 1.0f/(2.0f*RADIUS+1.0f);

// Simple PPM loader (stubs - generates synthetic data if no file)
static void loadImage(const char *fname, unsigned int **data,
                      unsigned int *width, unsigned int *height) {
  // Try to read PPM file
  FILE *f = fopen(fname, "rb");
  if (!f) {
    *width=512; *height=512;
    *data=(unsigned int*)malloc((*width)*(*height)*sizeof(unsigned int));
    for(unsigned int i=0;i<(*width)*(*height);i++) (*data)[i]=0xAABBCCFFu;
    return;
  }
  char magic[3]; fscanf(f,"%2s",magic);
  int w,h,maxv; fscanf(f,"%d %d %d ",&w,&h,&maxv);
  *width=w; *height=h;
  *data=(unsigned int*)malloc(w*h*sizeof(unsigned int));
  for(int i=0;i<w*h;i++) {
    unsigned char r=fgetc(f),g=fgetc(f),b=fgetc(f);
    (*data)[i]=(unsigned int)r|((unsigned int)g<<8)|((unsigned int)b<<16)|0xFF000000u;
  }
  fclose(f);
}

inline unsigned int DivUp(unsigned int a, unsigned int b){ return (a%b!=0)?(a/b+1):(a/b); }

int main(int argc, char **argv) {
  if (argc != 3) { printf("Usage %s <PPM image> <repeat>\n", argv[0]); return 1; }
  const int iCycles = atoi(argv[2]);

  unsigned int uiImageWidth=0, uiImageHeight=0;
  unsigned int *uiInput=nullptr;
  loadImage(argv[1], &uiInput, &uiImageWidth, &uiImageHeight);
  printf("Image Width = %u, Height = %u, Mask Radius = %u\n", uiImageWidth, uiImageHeight, RADIUS);

  size_t szBuff = uiImageWidth * uiImageHeight;
  unsigned int *uiDevOutput = (unsigned int*)malloc(szBuff*sizeof(unsigned int));
  unsigned int *uiHostOutput = (unsigned int*)malloc(szBuff*sizeof(unsigned int));

  Kokkos::initialize(argc, argv);
  {
    const int iRadius = RADIUS;
    const int iRadiusAligned = ((iRadius+15)/16)*16;
    unsigned int uiNumOutputPix = 64;
    const int szMaxWorkgroupSize = 256;
    if(szMaxWorkgroupSize < (iRadiusAligned+(int)uiNumOutputPix+iRadius))
      uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;
    const unsigned int uiBlockWidth = DivUp(uiImageWidth, uiNumOutputPix);
    const int numTeams = uiImageHeight * uiBlockWidth;
    const int blockSize = iRadiusAligned + uiNumOutputPix + iRadius;

    using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchView = Kokkos::View<uchar4*, ScratchSpace, Kokkos::MemoryUnmanaged>;
    const int scratchBytes = blockSize * sizeof(uchar4);

    using ViewUI = Kokkos::View<unsigned int*>;
    ViewUI d_in("in", szBuff), d_tmp("tmp", szBuff), d_out("out", szBuff);

    {
      auto hi=Kokkos::create_mirror_view(d_in);
      for(size_t i=0;i<szBuff;i++) hi(i)=uiInput[i];
      Kokkos::deep_copy(d_in,hi);
    }

    auto t0=std::chrono::steady_clock::now();

    for(int cycle=0;cycle<iCycles;cycle++) {
      // Row kernel - TeamPolicy with scratch memory
      Kokkos::parallel_for("boxfilter_row",
        Kokkos::TeamPolicy<>(numTeams, blockSize).set_scratch_size(0, Kokkos::PerTeam(scratchBytes)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
          ScratchView uc4LocalData(team.team_scratch(0), blockSize);
          const int lid = team.team_rank();
          const int gidx = team.league_rank() % uiBlockWidth;
          const int gidy = team.league_rank() / uiBlockWidth;
          const int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
          const int globalPosY = gidy;
          const int iGlobalOffset = globalPosY * uiImageWidth + globalPosX;

          if(globalPosX>=0 && globalPosX<(int)uiImageWidth)
            uc4LocalData(lid)=rgbaUintToUchar4(d_in(iGlobalOffset));
          else { uc4LocalData(lid)={0,0,0,0}; }

          team.team_barrier();

          if((globalPosX>=0)&&(globalPosX<(int)uiImageWidth)&&
             (lid>=iRadiusAligned)&&(lid<iRadiusAligned+(int)uiNumOutputPix)) {
            float4 f4Sum={0,0,0,0};
            int iOffsetX=lid-iRadius;
            int iLimit=iOffsetX+(2*iRadius)+1;
            for(;iOffsetX<iLimit;iOffsetX++) {
              f4Sum.x+=uc4LocalData(iOffsetX).x;
              f4Sum.y+=uc4LocalData(iOffsetX).y;
              f4Sum.z+=uc4LocalData(iOffsetX).z;
              f4Sum.w+=uc4LocalData(iOffsetX).w;
            }
            d_tmp(iGlobalOffset)=rgbaFloat4ToUint(f4Sum,SCALE);
          }
        });

      // Column kernel - simple parallel_for
      Kokkos::parallel_for("boxfilter_col", uiImageWidth, KOKKOS_LAMBDA(size_t globalPosX) {
        const unsigned int *uiInputImage = &d_tmp(globalPosX);
        unsigned int *uiOutputImage = &d_out(globalPosX);
        const int W = uiImageWidth;
        const int H = uiImageHeight;
        float4 top_color=rgbaUintToFloat4(uiInputImage[0]);
        float4 bot_color=rgbaUintToFloat4(uiInputImage[(H-1)*W]);
        float4 f4iRadius={float(iRadius),float(iRadius),float(iRadius),float(iRadius)};
        float4 f4Sum; f4Sum.x=top_color.x*f4iRadius.x; f4Sum.y=top_color.y*f4iRadius.y;
        f4Sum.z=top_color.z*f4iRadius.z; f4Sum.w=top_color.w*f4iRadius.w;
        for(int y=0;y<iRadius+1;y++) {
          float4 c=rgbaUintToFloat4(uiInputImage[y*W]);
          f4Sum.x+=c.x; f4Sum.y+=c.y; f4Sum.z+=c.z; f4Sum.w+=c.w;
        }
        uiOutputImage[0]=rgbaFloat4ToUint(f4Sum,SCALE);
        for(int y=1;y<iRadius+1;y++) {
          float4 a=rgbaUintToFloat4(uiInputImage[(y+iRadius)*W]);
          f4Sum.x+=a.x-top_color.x; f4Sum.y+=a.y-top_color.y;
          f4Sum.z+=a.z-top_color.z; f4Sum.w+=a.w-top_color.w;
          uiOutputImage[y*W]=rgbaFloat4ToUint(f4Sum,SCALE);
        }
        for(int y=iRadius+1;y<H-iRadius;y++) {
          float4 a=rgbaUintToFloat4(uiInputImage[(y+iRadius)*W]);
          float4 b=rgbaUintToFloat4(uiInputImage[(y-iRadius-1)*W]);
          f4Sum.x+=a.x-b.x; f4Sum.y+=a.y-b.y; f4Sum.z+=a.z-b.z; f4Sum.w+=a.w-b.w;
          uiOutputImage[y*W]=rgbaFloat4ToUint(f4Sum,SCALE);
        }
        for(int y=H-iRadius;y<H;y++) {
          float4 b=rgbaUintToFloat4(uiInputImage[(y-iRadius-1)*W]);
          f4Sum.x+=bot_color.x-b.x; f4Sum.y+=bot_color.y-b.y;
          f4Sum.z+=bot_color.z-b.z; f4Sum.w+=bot_color.w-b.w;
          uiOutputImage[y*W]=rgbaFloat4ToUint(f4Sum,SCALE);
        }
      });
    }
    Kokkos::fence();
    auto t1=std::chrono::steady_clock::now();
    auto time=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    printf("Average kernel execution time %f (us)\n", (time*1e-3f)/iCycles);

    auto ho=Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(ho,d_out);
    for(size_t i=0;i<szBuff;i++) uiDevOutput[i]=ho(i);
  }
  Kokkos::finalize();

  free(uiInput); free(uiDevOutput); free(uiHostOutput);
  return 0;
}
