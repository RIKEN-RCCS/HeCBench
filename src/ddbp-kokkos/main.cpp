/*
 * Distance-Driven Backprojection (DBT) – Kokkos port
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define integrateXcoord 1
#define integrateYcoord 0

using View1D = Kokkos::View<double*>;

KOKKOS_INLINE_FUNCTION void map_boudaries_dev(double* d_pBound, int nElem,
    double valueLeftBound, double sizeElem, double offset, int gid) {
  if (gid < nElem)
    d_pBound[gid] = (gid - valueLeftBound) * sizeElem + offset;
}

static void map_boudaries_kernel(View1D d_pBound, int nElem,
    double valueLeftBound, double sizeElem, double offset)
{
  Kokkos::parallel_for("map_boundaries", nElem,
    KOKKOS_LAMBDA(int gid) {
      d_pBound(gid) = (gid - valueLeftBound) * sizeElem + offset;
    });
}

static void rot_detector_kernel(View1D d_pRdetY, View1D d_pRdetZ,
    View1D d_pYcoord, View1D d_pZcoord,
    double yOffset, double zOffset, double phi, int nElem)
{
  Kokkos::parallel_for("rot_detector", nElem,
    KOKKOS_LAMBDA(int gid) {
      d_pRdetY(gid) = ((d_pYcoord(gid)-yOffset)*cos(phi)-(d_pZcoord(gid)-zOffset)*sin(phi))+yOffset;
      d_pRdetZ(gid) = ((d_pYcoord(gid)-yOffset)*sin(phi)+(d_pZcoord(gid)-zOffset)*cos(phi))+zOffset;
    });
}

static void mapDet2Slice_kernel(View1D d_pXmapp, View1D d_pYmapp,
    double tubeX, double tubeY, double tubeZ,
    View1D d_pXcoord, View1D d_pYcoord, View1D d_pZcoord, View1D d_pZSlicecoord,
    int nDetXMap, int nDetYMap, int nz)
{
  Kokkos::parallel_for("mapDet2Slice", nDetXMap * nDetYMap,
    KOKKOS_LAMBDA(int id) {
      int py = id / nDetYMap;
      int px = id % nDetYMap;
      int pos = py * nDetYMap + px;
      d_pXmapp(pos) = ((d_pXcoord(py)-tubeX)*(d_pZSlicecoord(nz)-d_pZcoord(px))
                      -(d_pXcoord(py)*tubeZ)+(d_pXcoord(py)*d_pZcoord(px)))
                      /(-tubeZ+d_pZcoord(px));
      if (py == 0)
        d_pYmapp(px) = ((d_pYcoord(px)-tubeY)*(d_pZSlicecoord(nz)-d_pZcoord(px))
                       -(d_pYcoord(px)*tubeZ)+(d_pYcoord(px)*d_pZcoord(px)))
                       /(-tubeZ+d_pZcoord(px));
    });
}

static void bilinear_interpolation_kernel(View1D d_sliceI, View1D d_pProj,
    View1D d_pObjX, View1D d_pObjY, View1D d_pDetmX, View1D d_pDetmY,
    int nPixXMap, int nPixYMap, int nDetXMap, int nDetYMap, int nDetX, int nDetY, int np)
{
  Kokkos::parallel_for("bilinear", nPixXMap * nPixYMap,
    KOKKOS_LAMBDA(int id) {
      int py = id / nPixYMap;
      int px = id % nPixYMap;
      double xNormData = nDetX - d_pObjX(py) / d_pDetmX(0);
      int    xData     = (int)floor(xNormData);
      double alpha     = xNormData - xData;
      double yNormData = (d_pObjY(px) / d_pDetmX(0)) - (d_pDetmY(0) / d_pDetmX(0));
      int    yData     = (int)floor(yNormData);
      double beta      = yNormData - yData;
      double d00=0, d10=0, d01=0, d11=0;
      int base = np * nDetYMap * nDetXMap;
      if (xNormData >= 0 && xNormData <= nDetX && yNormData >= 0 && yNormData <= nDetY)
        d00 = d_pProj(base + xData*nDetYMap + yData);
      if ((xData+1) > 0 && (xData+1) <= nDetX && yNormData >= 0 && yNormData <= nDetY)
        d10 = d_pProj(base + (xData+1)*nDetYMap + yData);
      if (xNormData >= 0 && xNormData <= nDetX && (yData+1) > 0 && (yData+1) <= nDetY)
        d01 = d_pProj(base + xData*nDetYMap + yData+1);
      if ((xData+1) > 0 && (xData+1) <= nDetX && (yData+1) > 0 && (yData+1) <= nDetY)
        d11 = d_pProj(base + (xData+1)*nDetYMap + yData+1);
      double t1 = alpha*d10 + (-d00*alpha + d00);
      double t2 = alpha*d11 + (-d01*alpha + d01);
      d_sliceI(py*nPixYMap + px) = beta*t2 + (-t1*beta + t1);
    });
}

static void differentiation_kernel(View1D d_pVolume, View1D d_sliceI,
    double tubeX, double rtubeY, double rtubeZ,
    View1D d_pObjX, View1D d_pObjY, View1D d_pObjZ,
    int nPixX, int nPixY, int nPixXMap, int nPixYMap,
    double du, double dv, double dx, double dy, double dz, int nz)
{
  Kokkos::parallel_for("differentiation", nPixX * nPixY,
    KOKKOS_LAMBDA(int id) {
      int py = id / nPixY;
      int px = id % nPixY;
      int pos = nPixX*nPixY*nz + py*nPixY + px;
      int coordA = py * nPixYMap + px;
      int coordB = (py+1) * nPixYMap + px;
      int coordC = coordA + 1;
      int coordD = coordB + 1;
      double gamma = atan((d_pObjX(py) + (dx/2.0) - tubeX) / (rtubeZ - d_pObjZ(nz)));
      double alpha = atan((d_pObjY(px) + (dy/2.0) - rtubeY) / (rtubeZ - d_pObjZ(nz)));
      double dA = d_sliceI(coordA), dB = d_sliceI(coordB);
      double dC = d_sliceI(coordC), dD = d_sliceI(coordD);
      if (dC == 0.0 && dD == 0.0) { dC = dA; dD = dB; }
      d_pVolume(pos) += (dD - dC - dB + dA) * (du*dv*dz / (cos(alpha)*cos(gamma)*dx*dy));
    });
}

static void division_kernel(View1D d_img, int nPixX, int nPixY, int nSlices, int nProj)
{
  Kokkos::parallel_for("division", nPixX * nPixY * nSlices,
    KOKKOS_LAMBDA(int id) { d_img(id) /= (double)nProj; });
}

// Serial cumulative sum along X (row) for each (column, slice)
static void integrate_X(View1D d_pProj, int nDetXMap, int nDetYMap, int nProj)
{
  Kokkos::parallel_for("integrateX", nDetYMap * nProj,
    KOKKOS_LAMBDA(int id) {
      int px = id % nDetYMap;
      int pz = id / nDetYMap;
      for (int py = 1; py < nDetXMap; py++)
        d_pProj(pz*nDetYMap*nDetXMap + py*nDetYMap + px) +=
          d_pProj(pz*nDetYMap*nDetXMap + (py-1)*nDetYMap + px);
    });
}

// Serial cumulative sum along Y (column) for each (row, slice)
static void integrate_Y(View1D d_pProj, int nDetXMap, int nDetYMap, int nProj)
{
  Kokkos::parallel_for("integrateY", nDetXMap * nProj,
    KOKKOS_LAMBDA(int id) {
      int py = id % nDetXMap;
      int pz = id / nDetXMap;
      for (int px = 1; px < nDetYMap; px++)
        d_pProj(pz*nDetYMap*nDetXMap + py*nDetYMap + px) +=
          d_pProj(pz*nDetYMap*nDetXMap + py*nDetYMap + px - 1);
    });
}

void backprojectionDDb(double* const h_pVolume, const double* const h_pProj,
    const double* const h_pTubeAngle, const double* const h_pDetAngle,
    int idXProj, int nProj, int nPixX, int nPixY, int nSlices,
    int nDetX, int nDetY, double dx, double dy, double dz,
    double du, double dv, double DSD, double DDR, double DAG)
{
  const int nDetXMap = nDetX + 1;
  const int nDetYMap = nDetY + 1;
  const int nPixXMap = nPixX + 1;
  const int nPixYMap = nPixY + 1;

  // Allocate and fill padded projection array on host
  std::vector<double> h_dProj(nDetXMap * nDetYMap * nProj, 0.0);
  for (int np = 0; np < nProj; np++)
    for (int c = 0; c < nDetX; c++) {
      const double* src = h_pProj + c*nDetY + nDetX*nDetY*np;
      double* dst = h_dProj.data() + ((c+1)*nDetYMap + 1) + nDetXMap*nDetYMap*np;
      memcpy(dst, src, nDetY * sizeof(double));
    }

  // Kokkos Views
  View1D d_pProj("proj", (size_t)nDetXMap*nDetYMap*nProj);
  View1D d_pVolume("vol", (size_t)nPixX*nPixY*nSlices);
  View1D d_sliceI("sliceI", (size_t)nPixXMap*nPixYMap);
  View1D d_pDetX("detX", nDetXMap), d_pDetY("detY", nDetYMap), d_pDetZ("detZ", nDetYMap);
  View1D d_pObjX("objX", nPixXMap),  d_pObjY("objY", nPixYMap),  d_pObjZ("objZ", nSlices);
  View1D d_pDetmY("detmY", nDetYMap), d_pDetmX("detmX", (size_t)nDetYMap*nDetXMap);
  View1D d_pRdetY("rdetY", nDetYMap), d_pRdetZ("rdetZ", nDetYMap);

  // Copy projection data to device
  {
    auto h_proj = Kokkos::create_mirror_view(d_pProj);
    for (size_t i = 0; i < h_dProj.size(); i++) h_proj(i) = h_dProj[i];
    Kokkos::deep_copy(d_pProj, h_proj);
  }

  // Map boundaries
  map_boudaries_kernel(d_pDetX, nDetXMap, (double)nDetX,   -du, 0.0);
  map_boudaries_kernel(d_pDetY, nDetYMap, nDetY / 2.0,      dv, 0.0);
  map_boudaries_kernel(d_pDetZ, nDetYMap, 0.0,              0.0, 0.0);
  map_boudaries_kernel(d_pObjX, nPixXMap, (double)nPixX,   -dx, 0.0);
  map_boudaries_kernel(d_pObjY, nPixYMap, nPixY / 2.0,      dy, 0.0);
  map_boudaries_kernel(d_pObjZ, nSlices,  0.0,               dz, DAG + dz/2.0);

  // Zero volume
  Kokkos::parallel_for("zero_vol", nPixX*nPixY*nSlices,
    KOKKOS_LAMBDA(int i) { d_pVolume(i) = 0.0; });

  // Integrate projection along X and Y (cumulative sum = integral image)
  integrate_X(d_pProj, nDetXMap, nDetYMap, nProj);
  Kokkos::fence();
  integrate_Y(d_pProj, nDetXMap, nDetYMap, nProj);
  Kokkos::fence();

  double tubeX = 0, tubeY = 0, tubeZ = DSD;
  double isoY = 0, isoZ = DDR;

  auto t_start = std::chrono::steady_clock::now();

  int projIni = (idXProj == -1) ? 0 : idXProj;
  int projEnd = (idXProj == -1) ? nProj : idXProj + 1;
  int nProj2Run = projEnd - projIni;

  // Subview into last two rows of detmX (nDetYMap elements from end of nDetXMap dimension)
  // This corresponds to d_pDetmX_tmp = d_pDetmX + nDetYMap*(nDetXMap-2)
  // We pass the offset as a parameter to the kernel

  for (int p = projIni; p < projEnd; p++) {
    double theta = h_pTubeAngle[p] * M_PI / 180.0;
    double phi   = h_pDetAngle[p]  * M_PI / 180.0;
    double rtubeY = ((tubeY-isoY)*cos(theta)-(tubeZ-isoZ)*sin(theta)) + isoY;
    double rtubeZ = ((tubeY-isoY)*sin(theta)+(tubeZ-isoZ)*cos(theta)) + isoZ;

    rot_detector_kernel(d_pRdetY, d_pRdetZ, d_pDetY, d_pDetZ, isoY, isoZ, phi, nDetYMap);
    Kokkos::fence();

    for (int nz = 0; nz < nSlices; nz++) {
      mapDet2Slice_kernel(d_pDetmX, d_pDetmY, tubeX, rtubeY, rtubeZ,
          d_pDetX, d_pRdetY, d_pRdetZ, d_pObjZ, nDetXMap, nDetYMap, nz);
      Kokkos::fence();

      // Subview for interpolation: last 2 nDetXMap rows -> offset nDetYMap*(nDetXMap-2)
      int detmX_offset = nDetYMap * (nDetXMap - 2);
      // Use a shifted view via a raw-pointer subview workaround
      Kokkos::parallel_for("bilinear", nPixXMap * nPixYMap,
        KOKKOS_LAMBDA(int id) {
          int py = id / nPixYMap;
          int px = id % nPixYMap;
          double xNormData = nDetX - d_pObjX(py) / d_pDetmX(detmX_offset + 0);
          int    xData     = (int)floor(xNormData);
          double alpha     = xNormData - xData;
          double yNormData = d_pObjY(px) / d_pDetmX(detmX_offset + 0)
                           - d_pDetmY(0) / d_pDetmX(detmX_offset + 0);
          int    yData     = (int)floor(yNormData);
          double beta      = yNormData - yData;
          double d00=0, d10=0, d01=0, d11=0;
          int base = p * nDetYMap * nDetXMap;
          if (xNormData >= 0 && xNormData <= nDetX && yNormData >= 0 && yNormData <= nDetY)
            d00 = d_pProj(base + xData*nDetYMap + yData);
          if ((xData+1) > 0 && (xData+1) <= nDetX && yNormData >= 0 && yNormData <= nDetY)
            d10 = d_pProj(base + (xData+1)*nDetYMap + yData);
          if (xNormData >= 0 && xNormData <= nDetX && (yData+1) > 0 && (yData+1) <= nDetY)
            d01 = d_pProj(base + xData*nDetYMap + yData+1);
          if ((xData+1) > 0 && (xData+1) <= nDetX && (yData+1) > 0 && (yData+1) <= nDetY)
            d11 = d_pProj(base + (xData+1)*nDetYMap + yData+1);
          double t1 = alpha*d10 + (-d00*alpha + d00);
          double t2 = alpha*d11 + (-d01*alpha + d01);
          d_sliceI(py*nPixYMap + px) = beta*t2 + (-t1*beta + t1);
        });
      Kokkos::fence();

      differentiation_kernel(d_pVolume, d_sliceI, tubeX, rtubeY, rtubeZ,
          d_pObjX, d_pObjY, d_pObjZ, nPixX, nPixY, nPixXMap, nPixYMap,
          du, dv, dx, dy, dz, nz);
      Kokkos::fence();
    }
  }

  division_kernel(d_pVolume, nPixX, nPixY, nSlices, nProj2Run);
  Kokkos::fence();

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
  printf("Total kernel execution %f (s)\n", elapsed * 1e-9f);

  // Copy volume back
  {
    auto h_vol = Kokkos::create_mirror_view(d_pVolume);
    Kokkos::deep_copy(h_vol, d_pVolume);
    for (int i = 0; i < nPixX*nPixY*nSlices; i++) h_pVolume[i] = h_vol(i);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) { printf("Usage: %s <nProj>\n", argv[0]); return 1; }
  const int nProj = atoi(argv[1]);

  const int nPixX=1996, nPixY=2457, nSlices=78;
  const int nDetX=1664, nDetY=2048;
  const int idXProj = -1;
  const double dx=0.112, dy=0.112, dz=1.0;
  const double du=0.14,  dv=0.14;
  const double DSD=700,  DDR=0.0, DAG=25.0;

  size_t pixVol = (size_t)nPixX*nPixY*nSlices;
  size_t detVol = (size_t)nDetX*nDetY*nProj;
  double *h_pVolume      = (double*)malloc(pixVol * sizeof(double));
  double *h_pProj        = (double*)malloc(detVol * sizeof(double));
  double *h_pTubeAngle   = (double*)malloc(nProj  * sizeof(double));
  double *h_pDetAngle    = (double*)malloc(nProj  * sizeof(double));

  for (int i = 0; i < nProj; i++) {
    h_pTubeAngle[i] = -7.5 + i * 15.0 / nProj;
    h_pDetAngle[i]  = -2.1 + i *  4.2 / nProj;
  }
  srand(123);
  for (size_t i = 0; i < pixVol; i++) h_pVolume[i] = (double)rand()/(double)RAND_MAX;
  for (size_t i = 0; i < detVol; i++) h_pProj[i]   = (double)rand()/(double)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    backprojectionDDb(h_pVolume, h_pProj, h_pTubeAngle, h_pDetAngle,
        idXProj, nProj, nPixX, nPixY, nSlices, nDetX, nDetY,
        dx, dy, dz, du, dv, DSD, DDR, DAG);
  }
  Kokkos::finalize();

  double checkSum = 0;
  for (size_t i = 0; i < pixVol; i++) checkSum += h_pVolume[i];
  printf("checksum = %lf\n", checkSum);

  free(h_pVolume); free(h_pProj); free(h_pTubeAngle); free(h_pDetAngle);
  return 0;
}
