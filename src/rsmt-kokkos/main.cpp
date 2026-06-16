// RSMT: Rectilinear Steiner Minimum Tree (Kokkos port)
// Copyright 2019-2022 Texas State University
// Ported from CUDA warp-level implementation to Kokkos parallel_for
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <climits>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>

static const int MaxPins = 256;

using ID    = short;
using ctype = int;

struct edge { ID src; ID dst; };

// Compute wire length of a Steiner tree
static ctype treeLength(const ID num, const ctype* x, const ctype* y, const edge* edges)
{
  ctype len = 0;
  for (ID i = 0; i < num - 1; i++) {
    ctype x1 = x[edges[i].src], y1 = y[edges[i].src];
    ctype x2 = x[edges[i].dst], y2 = y[edges[i].dst];
    len += std::abs(x1 - x2) + std::abs(y1 - y2);
  }
  return len;
}

struct grid {
  int g[3];
  int *vc = nullptr, *hc = nullptr;
  int *min_wid = nullptr, *min_space = nullptr, *via_space = nullptr;
  int llx, lly, tile_wid, tile_height;
};

struct net_list {
  int num_net = 0;
  std::vector<std::tuple<int,int>>* num_net_arr = nullptr;
  int *net_id = nullptr, *net_num_pins = nullptr, *net_min_wid = nullptr;
};

static void free_memory(const grid& g, const net_list& n)
{
  delete[] g.vc; delete[] g.hc;
  delete[] g.min_wid; delete[] g.min_space; delete[] g.via_space;
  delete[] n.net_id; delete[] n.net_num_pins; delete[] n.net_min_wid;
  delete[] n.num_net_arr;
}

static bool read_file(const char* file, grid& g, net_list& n)
{
  std::string line, text1, text2;
  std::fstream myfile(file);
  if (!myfile.is_open()) { std::cerr << "ERROR: Cannot open file: " << file << "\n"; return false; }
  
  getline(myfile, line);
  { std::stringstream s(line); s >> text1 >> g.g[0] >> g.g[1] >> g.g[2]; }
  g.vc = new int[g.g[2]+1];
  getline(myfile, line);
  { std::stringstream s(line); s >> text1 >> text2; for (int i=1;i<=g.g[2];i++) s>>g.vc[i]; }
  g.hc = new int[g.g[2]+1];
  getline(myfile, line);
  { std::stringstream s(line); s >> text1 >> text2; for (int i=1;i<=g.g[2];i++) s>>g.hc[i]; }
  // min width, spacing, via
  g.min_wid   = new int[g.g[2]+1];
  g.min_space = new int[g.g[2]+1];
  g.via_space = new int[g.g[2]+1];
  for (int grp = 0; grp < 3; grp++) {
    getline(myfile, line);
    std::stringstream ss(line); ss >> text1;
    if (grp==0) { for (int i=1;i<=g.g[2];i++) ss>>g.min_wid[i]; }
    else if (grp==1) { for (int i=1;i<=g.g[2];i++) ss>>g.min_space[i]; }
    else { for (int i=1;i<=g.g[2];i++) ss>>g.via_space[i]; }
  }
  // tile origin
  getline(myfile, line);
  { std::stringstream s(line); s >> text1 >> g.llx >> g.lly >> g.tile_wid >> g.tile_height; }
  // num nets
  getline(myfile, line);
  int num_net;
  { std::stringstream s(line); s >> text1 >> num_net; }
  n.num_net      = num_net;
  n.num_net_arr  = new std::vector<std::tuple<int,int>>[num_net];
  n.net_id       = new int[num_net];
  n.net_num_pins = new int[num_net];
  n.net_min_wid  = new int[num_net];
  for (int i = 0; i < num_net; i++) {
    int id, npins, mwid;
    getline(myfile, line);
    std::stringstream sn(line); sn >> text1 >> id >> npins >> mwid;
    n.net_id[i]       = id;
    n.net_num_pins[i] = npins;
    n.net_min_wid[i]  = mwid;
    n.num_net_arr[i].reserve(npins);
    for (int j = 0; j < npins; j++) {
      getline(myfile, line);
      int x, y, layer; std::string name;
      std::stringstream sp(line); sp >> x >> y >> layer >> name;
      n.num_net_arr[i].emplace_back(x, y);
    }
  }
  return true;
}

// Build MST with Prim's algorithm (sequential, per-net)
static void buildMST(ID num, ctype* x, ctype* y, edge* edges, ctype* dist, ID* source, ID* destin)
{
  ID numItems = num - 1;
  for (ID i = 0; i < numItems; i++) {
    dist[i]    = INT_MAX;
    destin[i]  = i + 1;
    source[i]  = 0;
  }
  ID src = 0;
  for (ID cnt = 0; cnt < num - 1; cnt++) {
    ctype mindj = INT_MAX;
    ID best = 0;
    for (ID j = 0; j < numItems; j++) {
      ctype dnew = std::abs(x[src] - x[destin[j]]) + std::abs(y[src] - y[destin[j]]);
      if (dist[j] > dnew) {
        dist[j]   = dnew;
        source[j] = src;
      }
      if (dist[j] < mindj) { mindj = dist[j]; best = j; }
    }
    edges[cnt].src = source[best];
    edges[cnt].dst = destin[best];
    src = destin[best];
    numItems--;
    dist[best]   = dist[numItems];
    source[best] = source[numItems];
    destin[best] = destin[numItems];
  }
}

// Insert Steiner points; returns true if any were added
static bool insertSteinerPoints(ID& num, ctype* x, ctype* y, edge* edges)
{
  // Build adjacency lists
  struct AdjList { int cnt; ID adj[8]; };
  std::vector<AdjList> alist(num);
  for (ID i = 0; i < num; i++) alist[i].cnt = 0;

  std::vector<ctype> dist(num - 1, -1);

  for (ID e = 0; e < num - 1; e++) {
    ID s = edges[e].src, d = edges[e].dst;
    if ((x[d] != x[s]) || (y[d] != y[s])) {
      alist[s].adj[alist[s].cnt++] = e;
      alist[d].adj[alist[d].cnt++] = e;
    }
  }

  for (ID s = 0; s < num; s++) {
    if (alist[s].cnt < 2) continue;
    ctype x0 = x[s], y0 = y[s];
    for (int j = 0; j < alist[s].cnt - 1; j++) {
      ID e1 = alist[s].adj[j];
      ID d1 = (s != edges[e1].src) ? edges[e1].src : edges[e1].dst;
      ctype x1 = x[d1], y1 = y[d1];
      for (int k = j + 1; k < alist[s].cnt; k++) {
        ID e2 = alist[s].adj[k];
        ID d2 = (s != edges[e2].src) ? edges[e2].src : edges[e2].dst;
        ctype stx = std::max(std::min(x0,x1), std::min(std::max(x0,x1), x[d2]));
        ctype sty = std::max(std::min(y0,y1), std::min(std::max(y0,y1), y[d2]));
        ctype rd  = std::abs(stx-x0) + std::abs(sty-y0);
        if (rd > 0) {
          ctype rd1 = rd * (MaxPins*2) + e1;
          ctype rd2 = rd * (MaxPins*2) + e2;
          if (dist[e1] < rd2) dist[e1] = rd2;
          if (dist[e2] < rd1) dist[e2] = rd1;
        }
      }
    }
  }

  bool updated = false;
  ID orig_num = num;
  for (ID e1 = 0; e1 < orig_num - 2; e1++) {
    ctype d1 = dist[e1];
    if (d1 <= 0) continue;
    ID e2 = (ID)(d1 % (MaxPins*2));
    if (e2 <= e1) continue;
    ctype d2 = (e2 < orig_num - 1) ? dist[e2] : -1;
    if (d2 <= 0) continue;
    if (e1 != (ID)(d2 % (MaxPins*2))) continue;
    ctype x0 = x[edges[e1].src], y0 = y[edges[e1].src];
    ctype x1 = x[edges[e1].dst], y1 = y[edges[e1].dst];
    ctype x2 = x[edges[e2].src], y2 = y[edges[e2].src];
    if (((x2==x0)&&(y2==y0)) || ((x2==x1)&&(y2==y1))) {
      x2 = x[edges[e2].dst]; y2 = y[edges[e2].dst];
    }
    ctype stx = std::max(std::min(x0,x1), std::min(std::max(x0,x1), x2));
    ctype sty = std::max(std::min(y0,y1), std::min(std::max(y0,y1), y2));
    if (num < 2 * (ID)orig_num) {
      x[num] = stx; y[num] = sty; num++;
    }
    updated = true;
  }
  return updated;
}

// Process one net: copy pins, run Prim+Steiner, fill idxout, edges
static void processNet(int net_i,
                       const int* idxin, const ctype* xin, const ctype* yin,
                       int* idxout, ctype* xout, ctype* yout, edge* edges_out)
{
  int pin  = idxin[net_i];
  ID  num  = (ID)(idxin[net_i+1] - pin);
  int pout = 2 * pin;
  idxout[net_i] = pout;

  // Copy pins to extended output buffer (x2 space for Steiner points)
  for (ID j = 0; j < num; j++) {
    xout[pout+j] = xin[pin+j];
    yout[pout+j] = yin[pin+j];
  }

  if (num < 2) return;

  std::vector<ctype> dist(num);
  std::vector<ID>    source(num), destin(num);
  std::vector<edge>  local_edges(2*num);

  ID cnt = num;
  do {
    buildMST(cnt, &xout[pout], &yout[pout], local_edges.data(), dist.data(), source.data(), destin.data());
    if ((int)local_edges.size() < 2*cnt) local_edges.resize(2*cnt);
    if ((int)dist.size()   < 2*cnt) { dist.resize(2*cnt); source.resize(2*cnt); destin.resize(2*cnt); }
  } while (insertSteinerPoints(cnt, &xout[pout], &yout[pout], local_edges.data()));

  // Copy edges to output
  for (ID j = 0; j < cnt - 1; j++) {
    edges_out[pout+j] = local_edges[j];
  }
}

int main(int argc, char* argv[])
{
  printf("A Simple, Fast, and GPU-friendly Steiner-Tree Heuristic\n");
  printf("Copyright 2019-2022 Texas State University\n\n");

  if (argc != 2) {
    printf("Usage: %s file_name\n", argv[0]);
    return 1;
  }

  printf("reading input file: %s\n", argv[1]);
  grid g; net_list n;
  if (!read_file(argv[1], g, n)) return 1;

  const int numnets = n.num_net;

  std::vector<int>  idxin(numnets + 1);
  idxin[0] = 0;
  ID hipin = 0;
  int pos = 0;
  for (int i = 0; i < numnets; i++) {
    ID num = (ID)std::min(n.net_num_pins[i], MaxPins);
    hipin   = std::max(hipin, num);
    pos    += num;
    idxin[i+1] = pos;
  }
  int total_pins = idxin[numnets];
  int trunc = 0;
  for (int i = 0; i < numnets; i++)
    if (n.net_num_pins[i] > MaxPins) trunc++;

  printf("number of nets: %d\n", numnets);
  printf("max pins per net: %d\n", hipin);
  printf("truncated nets: %d\n", trunc);
  if (hipin > MaxPins) { printf("ERROR: hi_pin_count > %d\n", MaxPins); return 1; }

  std::vector<ctype> xin(total_pins), yin(total_pins);
  pos = 0;
  for (int i = 0; i < numnets; i++) {
    ID num = (ID)(idxin[i+1] - idxin[i]);
    for (ID j = 0; j < num; j++) {
      xin[pos+j] = (ctype)std::get<0>(n.num_net_arr[i][j]);
      yin[pos+j] = (ctype)std::get<1>(n.num_net_arr[i][j]);
    }
    pos += num;
  }

  // Output buffers (2x size for Steiner points)
  std::vector<int>   idxout(numnets + 1, 0);
  std::vector<ctype> xout(2 * total_pins, -1);
  std::vector<ctype> yout(2 * total_pins, -1);
  std::vector<edge>  edges(2 * total_pins, {0, 0});

  Kokkos::initialize(argc, argv);
  {
    // Host views wrapping the std::vectors
    Kokkos::View<int*,  Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_idxin (idxin.data(),  numnets+1);
    Kokkos::View<ctype*,Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_xin   (xin.data(),    total_pins);
    Kokkos::View<ctype*,Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_yin   (yin.data(),    total_pins);
    Kokkos::View<int*,  Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_idxout(idxout.data(), numnets+1);
    Kokkos::View<ctype*,Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_xout  (xout.data(),   2*total_pins);
    Kokkos::View<ctype*,Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_yout  (yout.data(),   2*total_pins);
    Kokkos::View<edge*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_edges (edges.data(),  2*total_pins);

    auto t_start = std::chrono::steady_clock::now();

    // Parallel over nets
    Kokkos::parallel_for("RSMT_Nets",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numnets),
      [&](const int i) {
        processNet(i,
                   h_idxin.data(), h_xin.data(), h_yin.data(),
                   h_idxout.data(), h_xout.data(), h_yout.data(),
                   h_edges.data());
      });
    Kokkos::fence();

    // Set final index
    idxout[numnets] = 2 * idxin[numnets];

    auto t_end   = std::chrono::steady_clock::now();
    auto ns      = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    double runtime = ns * 1e-9;

    printf("compute time: %.6f s\n", runtime);
    printf("throughput: %.0f nets/sec\n", numnets / runtime);

    long total_len = 0, total_pin = 0;
    for (int i = 0; i < numnets; i++) {
      int seg_len = idxout[i+1] - idxout[i];
      if (seg_len > 1) {
        ctype len = treeLength((ID)seg_len,
                               &xout[idxout[i]], &yout[idxout[i]],
                               &edges[idxout[i]]);
        total_len += len;
      }
      total_pin += (ID)std::min(n.net_num_pins[i], MaxPins);
    }
    printf("total wirelength: %ld\n", total_len);
    printf("total pins: %ld\n", total_pin);
  }
  Kokkos::finalize();

  free_memory(g, n);
  return 0;
}
