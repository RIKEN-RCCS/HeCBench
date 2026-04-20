// Kokkos port of local-ht-cuda benchmark
// Local hash table for sequence assembly (LocalAssm)
// Each Kokkos thread handles one contig (replaces one CUDA warp).
#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Definitions (from kernel.hpp)
// ---------------------------------------------------------------------------
#define EMPTY        0xFFFFFFFF
#define LASSM_MIN_QUAL       10
#define LASSM_MIN_HI_QUAL    20
#define LASSM_MIN_VIABLE_DEPTH   0.2
#define LASSM_MIN_EXPECTED_DEPTH 0.3
#define LASSM_RATING_THRES   0
#define LASSM_MIN_KMER_LEN   21
#define LASSM_SHIFT_SIZE     8
#define LASSM_MAX_KMER_LEN   121

// ---------------------------------------------------------------------------
// cstr_type: lightweight string view (pointer + length)
// ---------------------------------------------------------------------------
struct cstr_type {
  char* start_ptr;
  int   length;

  KOKKOS_INLINE_FUNCTION cstr_type() : start_ptr(nullptr), length((int)EMPTY) {}
  KOKKOS_INLINE_FUNCTION cstr_type(char* ptr, int len)
    : start_ptr(ptr), length(len) {}

  KOKKOS_INLINE_FUNCTION bool operator==(const cstr_type& o) const {
    if (length != o.length) return false;
    if ((unsigned)length == EMPTY) return false;
    for (int i = 0; i < o.length; i++)
      if (start_ptr[i] != o.start_ptr[i]) return false;
    return true;
  }
};

KOKKOS_INLINE_FUNCTION void cstr_copy(cstr_type& dst, const cstr_type& src) {
  for (int i = 0; i < src.length; i++) dst.start_ptr[i] = src.start_ptr[i];
  dst.length = src.length;
}

// ---------------------------------------------------------------------------
// Extension counts
// ---------------------------------------------------------------------------
struct ExtCounts {
  uint32_t count_A, count_C, count_G, count_T;

  KOKKOS_INLINE_FUNCTION void inc(char ext, int cnt) {
    switch (ext) {
      case 'A': count_A += cnt; break;
      case 'C': count_C += cnt; break;
      case 'G': count_G += cnt; break;
      case 'T': count_T += cnt; break;
    }
  }
};

// ---------------------------------------------------------------------------
// MerBase and MerFreqs
// ---------------------------------------------------------------------------
struct MerBase {
  char     base;
  uint32_t nvotes_hi_q, nvotes, rating;

  KOKKOS_INLINE_FUNCTION uint16_t get_base_rating(int depth) {
    double min_viable  = Kokkos::max(LASSM_MIN_VIABLE_DEPTH  * depth, 2.0);
    double min_exp_dep = Kokkos::max(LASSM_MIN_EXPECTED_DEPTH * depth, 2.0);
    if (nvotes == 0) return 0;
    if (nvotes == 1) return 1;
    if (nvotes < (uint32_t)min_viable) return 2;
    if ((double)nvotes < min_exp_dep && (double)nvotes >= min_viable && nvotes_hi_q < (uint32_t)min_viable) return 3;
    if ((double)nvotes < min_exp_dep && (double)nvotes >= min_viable && nvotes_hi_q >= (uint32_t)min_viable) return 4;
    if ((double)nvotes >= min_exp_dep && nvotes_hi_q < (uint32_t)min_viable) return 5;
    if ((double)nvotes >= min_exp_dep && (double)nvotes_hi_q < min_exp_dep
        && nvotes_hi_q >= (uint32_t)min_viable) return 6;
    return 7;
  }
};

struct MerFreqs {
  ExtCounts hi_q_exts, low_q_exts;
  char ext;
  int  count;

  KOKKOS_INLINE_FUNCTION bool comp_merbase(const MerBase& a, const MerBase& b) const {
    if (a.rating != b.rating) return a.rating > b.rating;
    if (a.nvotes_hi_q != b.nvotes_hi_q) return a.nvotes_hi_q > b.nvotes_hi_q;
    return a.nvotes > b.nvotes;
  }

  KOKKOS_INLINE_FUNCTION void sort_merbase(MerBase (&mb)[4]) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (comp_merbase(mb[i], mb[j])) {
          MerBase tmp = mb[i]; mb[i] = mb[j]; mb[j] = tmp;
        }
  }

  KOKKOS_INLINE_FUNCTION void set_ext(int seq_depth) {
    MerBase mb[4] = {
      {'A', hi_q_exts.count_A, low_q_exts.count_A, 0},
      {'C', hi_q_exts.count_C, low_q_exts.count_C, 0},
      {'G', hi_q_exts.count_G, low_q_exts.count_G, 0},
      {'T', hi_q_exts.count_T, low_q_exts.count_T, 0}
    };
    for (int i = 0; i < 4; i++) mb[i].rating = mb[i].get_base_rating(seq_depth);
    sort_merbase(mb);

    int top    = mb[0].rating;
    int runner = mb[1].rating;
    char top_base = mb[0].base;
    ext = 'X'; count = 0;

    if (top > LASSM_RATING_THRES) {
      if (top <= 3) { if (runner == 0) ext = top_base; }
      else if (top < 6) { if (runner < 3) ext = top_base; }
      else if (top == 6) { if (runner < 4) ext = top_base; }
      else {
        if (runner < 7) {
          ext = top_base;
        } else {
          if (mb[2].rating == 7 || mb[0].nvotes == mb[1].nvotes) ext = 'F';
          else if (mb[0].nvotes > mb[1].nvotes) ext = mb[0].base;
          else                                   ext = mb[1].base;
        }
      }
    }
    for (int i = 0; i < 4; i++)
      if (mb[i].base == ext) { count = (int)mb[i].nvotes; break; }
  }
};

// ---------------------------------------------------------------------------
// Hash table entry types
// ---------------------------------------------------------------------------
struct loc_ht {
  cstr_type key;
  MerFreqs  val;
};

struct loc_ht_bool {
  cstr_type key;
  bool      val;
};

// ---------------------------------------------------------------------------
// MurmurHash (from kernel.cu)
// ---------------------------------------------------------------------------
#define MIX(h,k,m) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }

KOKKOS_INLINE_FUNCTION uint32_t MurmurHashAligned2(const cstr_type& key_in,
                                                     uint32_t max_size) {
  int len         = key_in.length;
  const char* key = key_in.start_ptr;
  const uint32_t m    = 0x5bd1e995u;
  const int      r    = 24;
  uint32_t       seed = 0x3FB0BB5Fu;

  const unsigned char* data = (const unsigned char*)key;
  uint32_t h = seed ^ (uint32_t)len;

  int align = (int)((uintptr_t)data & 3);

  if (align && len >= 4) {
    uint32_t t = 0, d = 0;
    switch (align) {
      case 1: t |= (uint32_t)data[2] << 16; // fallthrough
      case 2: t |= (uint32_t)data[1] << 8;  // fallthrough
      case 3: t |= (uint32_t)data[0];
    }
    t <<= (8 * align);
    data += 4 - align;
    len  -= 4 - align;
    int sl = 8 * (4 - align);
    int sr = 8 * align;

    while (len >= 4) {
      uint32_t dk;
      __builtin_memcpy(&dk, data, 4);
      t = (t >> sr) | (dk << sl);
      uint32_t k = t;
      MIX(h, k, m);
      t = dk; data += 4; len -= 4;
    }
    d = 0;
    if (len >= align) {
      switch (align) {
        case 3: d |= (uint32_t)data[2] << 16; // fallthrough
        case 2: d |= (uint32_t)data[1] << 8;  // fallthrough
        case 1: d |= (uint32_t)data[0];
      }
      uint32_t k = (t >> sr) | (d << sl);
      MIX(h, k, m);
      data += align; len -= align;
      switch (len) {
        case 3: h ^= (uint32_t)data[2] << 16; // fallthrough
        case 2: h ^= (uint32_t)data[1] << 8;  // fallthrough
        case 1: h ^= (uint32_t)data[0]; h *= m;
      }
    } else {
      switch (len) {
        case 3: d |= (uint32_t)data[2] << 16; // fallthrough
        case 2: d |= (uint32_t)data[1] << 8;  // fallthrough
        case 1: d |= (uint32_t)data[0];        // fallthrough
        case 0: h ^= (t >> sr) | (d << sl); h *= m;
      }
    }
    h ^= h >> 13; h *= m; h ^= h >> 15;
    return h % max_size;
  }

  while (len >= 4) {
    uint32_t k; __builtin_memcpy(&k, data, 4);
    MIX(h, k, m);
    data += 4; len -= 4;
  }
  switch (len) {
    case 3: h ^= (uint32_t)data[2] << 16; // fallthrough
    case 2: h ^= (uint32_t)data[1] << 8;  // fallthrough
    case 1: h ^= (uint32_t)data[0]; h *= m;
  }
  h ^= h >> 13; h *= m; h ^= h >> 15;
  return h % max_size;
}
#undef MIX

// ---------------------------------------------------------------------------
// Non-atomic hash table get/insert (single-threaded per contig)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION loc_ht& ht_get(loc_ht* ht, const cstr_type& key,
                                        uint32_t max_size) {
  uint32_t h    = MurmurHashAligned2(key, max_size);
  uint32_t orig = h;
  while (true) {
    if ((unsigned)ht[h].key.length == EMPTY) {
      ht[h].key = key;
      ht[h].val = {{0,0,0,0},{0,0,0,0},'X',0};
      return ht[h];
    }
    if (ht[h].key == key) return ht[h];
    h = (h + 1) % max_size;
    if (h == orig) break; // full
  }
  return ht[0];
}

KOKKOS_INLINE_FUNCTION loc_ht_bool& ht_get_bool(loc_ht_bool* ht,
                                                   const cstr_type& key,
                                                   uint32_t max_size) {
  uint32_t h    = MurmurHashAligned2(key, max_size);
  uint32_t orig = h;
  while (true) {
    if ((unsigned)ht[h].key.length == EMPTY) {
      ht[h].key = key;
      ht[h].val = false;
      return ht[h];
    }
    if (ht[h].key == key) return ht[h];
    h = (h + 1) % max_size;
    if (h == orig) break;
  }
  return ht[0];
}

// ---------------------------------------------------------------------------
// count_mers: populate hash table from reads (sequential per contig)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void count_mers(
    loc_ht*   loc_ht_arr, const char* loc_reads, uint32_t max_ht_size,
    const char* loc_quals,  const uint32_t* reads_offset,
    uint32_t  r_rds_cnt,    const uint32_t* rds_count_sum,
    double    loc_ctg_depth, int mer_len,
    uint32_t  qual_offset,  const long int idx)
{
  uint32_t running_sum = 0;
  for (uint32_t i = 0; i < r_rds_cnt; i++) {
    uint32_t read_len;
    if (i == 0) {
      if (idx == 0)
        read_len = reads_offset[rds_count_sum[idx] - r_rds_cnt];
      else {
        if (rds_count_sum[idx-1] == 0)
          read_len = reads_offset[rds_count_sum[idx] - r_rds_cnt];
        else
          read_len = reads_offset[rds_count_sum[idx] - r_rds_cnt]
                   - reads_offset[rds_count_sum[idx-1] - 1];
      }
    } else {
      read_len = reads_offset[rds_count_sum[idx] - r_rds_cnt + i]
               - reads_offset[rds_count_sum[idx] - r_rds_cnt + i - 1];
    }

    const char* read = loc_reads + running_sum;
    const char* qual = loc_quals + running_sum;

    if ((int)read_len <= mer_len) { running_sum += read_len; continue; }

    int num_mers = (int)read_len - mer_len;
    for (int start = 0; start < num_mers; start++) {
      cstr_type mer(const_cast<char*>(read + start), mer_len);
      loc_ht& slot = ht_get(loc_ht_arr, mer, max_ht_size);

      int ext_pos = start + mer_len;
      if (ext_pos >= (int)read_len) continue;
      char ext = read[ext_pos];
      if (ext == 'N') continue;

      int qual_diff = (int)(unsigned char)qual[ext_pos] - (int)qual_offset;
      if (qual_diff >= LASSM_MIN_QUAL)    slot.val.low_q_exts.inc(ext, 1);
      if (qual_diff >= LASSM_MIN_HI_QUAL) slot.val.hi_q_exts.inc(ext, 1);
    }
    running_sum += read_len;
  }

  // set_ext for all occupied slots
  for (uint32_t k = 0; k < max_ht_size; k++)
    if ((unsigned)loc_ht_arr[k].key.length != EMPTY)
      loc_ht_arr[k].val.set_ext((int)loc_ctg_depth);
}

// ---------------------------------------------------------------------------
// walk_mers: walk hash table from contig end to find longest extension
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION char walk_mers(
    loc_ht*      thrd_ht,      loc_ht_bool* thrd_ht_bool,
    uint32_t     max_ht_size,  int& mer_len,
    cstr_type&   mer_walk_temp, cstr_type& longest_walk,
    cstr_type&   walk,         int max_walk_len)
{
  char result = 'X';
  for (int nsteps = 0; nsteps < max_walk_len; nsteps++) {
    loc_ht_bool& bool_slot = ht_get_bool(thrd_ht_bool, mer_walk_temp,
                                          (uint32_t)max_walk_len);
    if ((unsigned)bool_slot.key.length == EMPTY) {
      // newly inserted
      bool_slot.val = true;
    } else {
      result = 'R'; // cycle detected
      break;
    }

    loc_ht& mer_slot = ht_get(thrd_ht, mer_walk_temp, max_ht_size);
    if ((unsigned)mer_slot.key.length == EMPTY) { result = 'X'; break; }

    char ext = mer_slot.val.ext;
    if (ext == 'F' || ext == 'X') { result = ext; break; }

    mer_walk_temp.start_ptr++;
    mer_walk_temp.start_ptr[mer_walk_temp.length - 1] = ext;
    if (ext != 0) walk.length++;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Per-contig iterative walks kernel body (replaces CUDA warp logic)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void iterative_walks_body(
    const long int warp_id,
    uint32_t* cid, uint32_t* ctg_offsets, char* contigs,
    char* reads_r, char* quals_r,
    uint32_t* reads_r_offset, uint32_t* rds_count_r_sum,
    double*   ctg_depth,      loc_ht* global_ht, uint32_t* prefix_ht,
    loc_ht_bool* global_ht_bool,
    int kmer_len, uint32_t max_mer_len_off, int max_walk_len,
    char* longest_walks, char* mer_walk_temp_buf,
    uint32_t* final_walk_lens, int tot_ctgs,
    uint32_t qual_offset)
{
  if (warp_id >= tot_ctgs) return;

  // --- Determine per-contig data pointers ---
  cstr_type loc_ctg;
  char*     loc_r_reads;
  char*     loc_r_quals;
  uint32_t  r_rds_cnt;
  loc_ht*   loc_mer_map;
  uint32_t  ht_loc_size;
  loc_ht_bool* loc_bool_map;
  double    loc_ctg_depth;

  if (warp_id == 0) {
    loc_ctg      = cstr_type(contigs, (int)ctg_offsets[0]);
    loc_bool_map = global_ht_bool + (size_t)warp_id * max_walk_len;
    r_rds_cnt    = rds_count_r_sum[0];
    loc_r_reads  = reads_r;
    loc_r_quals  = quals_r;
    loc_mer_map  = global_ht;
    ht_loc_size  = prefix_ht[0];
    loc_ctg_depth= ctg_depth[0];
  } else {
    loc_ctg      = cstr_type(contigs + ctg_offsets[warp_id-1],
                              (int)(ctg_offsets[warp_id] - ctg_offsets[warp_id-1]));
    loc_bool_map = global_ht_bool + (size_t)warp_id * max_walk_len;
    loc_ctg_depth= ctg_depth[warp_id];
    r_rds_cnt    = rds_count_r_sum[warp_id] - rds_count_r_sum[warp_id-1];
    if (rds_count_r_sum[warp_id-1] == 0) {
      loc_r_reads = reads_r;
      loc_r_quals = quals_r;
    } else {
      loc_r_reads = reads_r + reads_r_offset[rds_count_r_sum[warp_id-1] - 1];
      loc_r_quals = quals_r + reads_r_offset[rds_count_r_sum[warp_id-1] - 1];
    }
    loc_mer_map  = global_ht + prefix_ht[warp_id-1];
    ht_loc_size  = prefix_ht[warp_id] - prefix_ht[warp_id-1];
  }

  char*     longest_walk_loc  = longest_walks    + (size_t)warp_id * max_walk_len;
  char*     loc_mer_walk_buf  = mer_walk_temp_buf + (size_t)warp_id * (max_walk_len + max_mer_len_off);
  uint32_t  max_ht_size       = ht_loc_size;
  int       max_mer_len       = Kokkos::min((int)LASSM_MAX_KMER_LEN, loc_ctg.length);
  int       min_mer_len       = LASSM_MIN_KMER_LEN;

  cstr_type longest_walk_thread(longest_walk_loc, 0);
  int shift = 0;

  for (int mer_len = kmer_len;
       mer_len >= min_mer_len && mer_len <= max_mer_len;
       mer_len += shift) {
    // Reset hash table
    for (uint32_t k = 0; k < max_ht_size; k++) loc_mer_map[k].key.length = (int)EMPTY;
    // Reset bool table
    for (int k = 0; k < max_walk_len; k++) loc_bool_map[k].key.length = (int)EMPTY;

    count_mers(loc_mer_map, loc_r_reads, max_ht_size, loc_r_quals,
               reads_r_offset, r_rds_cnt, rds_count_r_sum,
               loc_ctg_depth, mer_len, qual_offset, warp_id);

    // Walk from the end of contig
    cstr_type ctg_mer(loc_ctg.start_ptr + (loc_ctg.length - mer_len), mer_len);
    cstr_type loc_mer_walk(loc_mer_walk_buf, 0);
    cstr_copy(loc_mer_walk, ctg_mer);
    cstr_type walk(loc_mer_walk.start_ptr + mer_len, 0);

    char walk_res = walk_mers(loc_mer_map, loc_bool_map, max_ht_size,
                               mer_len, loc_mer_walk,
                               longest_walk_thread, walk, max_walk_len);

    if (walk.length > longest_walk_thread.length)
      cstr_copy(longest_walk_thread, walk);

    if (walk_res == 'X') {
      if (shift == LASSM_SHIFT_SIZE) break;
      shift = -LASSM_SHIFT_SIZE;
    } else {
      if (shift == -LASSM_SHIFT_SIZE) break;
      if (mer_len > loc_ctg.length) break;
      shift = LASSM_SHIFT_SIZE;
    }
  }

  final_walk_lens[warp_id] = (uint32_t)longest_walk_thread.length;
}

// ---------------------------------------------------------------------------
// Helper types (from helper.hpp)
// ---------------------------------------------------------------------------
struct ReadSeq { std::string read_id, seq, quals; };
struct CtgWithReads {
  int32_t cid;
  std::string seq;
  double depth;
  int max_reads;
  std::vector<ReadSeq> reads_left, reads_right;
};

template<class T>
void print_vals(T v) { std::cout << v << "\n"; }
template<class T, class... Ts>
void print_vals(T v, Ts... vs) { std::cout << v << " "; print_vals(vs...); }

inline void revcomp(char* s, char* rc, int n) {
  for (int i = n-1; i >= 0; i--)
    switch (s[i]) {
      case 'A': *rc++ = 'T'; break; case 'C': *rc++ = 'G'; break;
      case 'G': *rc++ = 'C'; break; case 'T': *rc++ = 'A'; break;
      default:  *rc++ = 'N'; break;
    }
}
inline std::string revcomp(const std::string& in) {
  std::string rc;
  for (int i = (int)in.size()-1; i >= 0; i--)
    switch (in[i]) {
      case 'A': rc += 'T'; break; case 'C': rc += 'G'; break;
      case 'G': rc += 'C'; break; case 'T': rc += 'A'; break;
      default:  rc += 'N'; break;
    }
  return rc;
}

void read_locassm_data(std::vector<CtgWithReads>* data, const std::string& fname,
                       uint32_t& max_ctg, uint32_t& tot_r, uint32_t& tot_l,
                       uint32_t& max_rd, uint32_t& max_r_cnt, uint32_t& max_l_cnt) {
  std::ifstream f(fname);
  std::string line;
  max_ctg = tot_r = tot_l = max_rd = max_r_cnt = max_l_cnt = 0;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    CtgWithReads t; int ls = 0, rs = 0;
    ss >> t.cid >> t.seq >> t.depth >> ls >> rs;
    tot_l += ls; tot_r += rs;
    if ((uint32_t)rs > max_r_cnt) max_r_cnt = rs;
    if ((uint32_t)ls > max_l_cnt) max_l_cnt = ls;
    t.max_reads = std::max(ls, rs);
    if (t.seq.size() > max_ctg) max_ctg = (uint32_t)t.seq.size();
    for (int i = 0; i < ls; i++) {
      ReadSeq r; ss >> r.read_id >> r.seq >> r.quals;
      if (r.seq.size() > max_rd) max_rd = (uint32_t)r.seq.size();
      t.reads_left.push_back(r);
    }
    for (int i = 0; i < rs; i++) {
      ReadSeq r; ss >> r.read_id >> r.seq >> r.quals;
      if (r.seq.size() > max_rd) max_rd = (uint32_t)r.seq.size();
      t.reads_right.push_back(r);
    }
    data->push_back(t);
  }
  print_vals("inside max:", max_r_cnt);
}

struct accum_data {
  std::vector<uint32_t> ht_sizes, l_reads_count, r_reads_count, ctg_sizes;
};

// ---------------------------------------------------------------------------
// call_kernel: pack data and launch Kokkos parallel_for
// ---------------------------------------------------------------------------
void call_kernel(std::vector<CtgWithReads>& data_in,
                 uint32_t max_ctg_size, uint32_t max_read_size,
                 uint32_t max_r_count, uint32_t max_l_count,
                 int mer_len, int /*max_reads_count*/,
                 accum_data& sizes_vecs, std::ofstream& out_file_g)
{
  const int max_mer_len_val = LASSM_MAX_KMER_LEN;
  int insert_avg    = 121, insert_stddev = 246;
  int max_walk_len  = insert_avg + 2 * insert_stddev;
  uint32_t qual_offset = 33;  // Phred+33

  unsigned tot_extensions = (unsigned)data_in.size();
  if (tot_extensions == 0) return;

  uint32_t total_l_reads = std::accumulate(sizes_vecs.l_reads_count.begin(),
                                            sizes_vecs.l_reads_count.end(), 0u);
  uint32_t total_r_reads = std::accumulate(sizes_vecs.r_reads_count.begin(),
                                            sizes_vecs.r_reads_count.end(), 0u);
  uint32_t ht_tot_size   = std::accumulate(sizes_vecs.ht_sizes.begin(),
                                            sizes_vecs.ht_sizes.end(), 0u);

  print_vals("HT size (bytes):", ht_tot_size * sizeof(loc_ht));
  print_vals("tot_extensions:", tot_extensions);

  // ------------------------------------------------------------------
  // Allocate host CPU arrays for the full dataset
  // ------------------------------------------------------------------
  std::vector<char>     ctg_seqs_h(max_ctg_size * tot_extensions, 0);
  std::vector<char>     ctg_rc_h  (max_ctg_size * tot_extensions, 0);
  std::vector<uint32_t> ctg_offsets_h(tot_extensions, 0);
  std::vector<uint32_t> cid_h(tot_extensions, 0);
  std::vector<double>   depth_h(tot_extensions, 0);
  std::vector<char>     reads_left_h (max_l_count * max_read_size * tot_extensions, 0);
  std::vector<char>     reads_right_h(max_r_count * max_read_size * tot_extensions, 0);
  std::vector<char>     quals_left_h (max_l_count * max_read_size * tot_extensions, 0);
  std::vector<char>     quals_right_h(max_r_count * max_read_size * tot_extensions, 0);
  std::vector<uint32_t> reads_l_offset_h(total_l_reads, 0);
  std::vector<uint32_t> reads_r_offset_h(total_r_reads, 0);
  std::vector<uint32_t> rds_l_cnt_h(tot_extensions, 0);
  std::vector<uint32_t> rds_r_cnt_h(tot_extensions, 0);
  std::vector<uint32_t> prefix_ht_h(tot_extensions, 0);
  std::vector<char>     longest_walks_r(tot_extensions * max_walk_len, 0);
  std::vector<char>     longest_walks_l(tot_extensions * max_walk_len, 0);
  std::vector<uint32_t> final_walk_lens_r(tot_extensions, 0);
  std::vector<uint32_t> final_walk_lens_l(tot_extensions, 0);

  // ------------------------------------------------------------------
  // Pack data
  // ------------------------------------------------------------------
  {
    uint32_t ctg_off = 0, lr_off = 0, rr_off = 0;
    uint32_t l_read_flat = 0, r_read_flat = 0;

    for (uint32_t i = 0; i < tot_extensions; i++) {
      const auto& ctg = data_in[i];
      cid_h[i]   = (uint32_t)ctg.cid;
      depth_h[i] = ctg.depth;
      uint32_t csz = (uint32_t)ctg.seq.size();
      std::memcpy(ctg_seqs_h.data() + ctg_off, ctg.seq.data(), csz);
      ctg_off += csz;
      ctg_offsets_h[i] = ctg_off;
      prefix_ht_h[i]   = (i == 0 ? 0 : prefix_ht_h[i-1]) + sizes_vecs.ht_sizes[i];

      // Left reads
      uint32_t lc = (uint32_t)ctg.reads_left.size();
      rds_l_cnt_h[i] = (i == 0 ? 0 : rds_l_cnt_h[i-1]) + lc;
      for (auto& rd : ctg.reads_left) {
        uint32_t rsz = (uint32_t)rd.seq.size();
        std::memcpy(reads_left_h.data()  + lr_off, rd.seq.data(),  rsz);
        std::memcpy(quals_left_h.data()  + lr_off, rd.quals.data(), rsz);
        lr_off += rsz;
        reads_l_offset_h[l_read_flat++] = lr_off;
      }
      // Right reads
      uint32_t rc_cnt = (uint32_t)ctg.reads_right.size();
      rds_r_cnt_h[i] = (i == 0 ? 0 : rds_r_cnt_h[i-1]) + rc_cnt;
      for (auto& rd : ctg.reads_right) {
        uint32_t rsz = (uint32_t)rd.seq.size();
        std::memcpy(reads_right_h.data() + rr_off, rd.seq.data(),  rsz);
        std::memcpy(quals_right_h.data() + rr_off, rd.quals.data(), rsz);
        rr_off += rsz;
        reads_r_offset_h[r_read_flat++] = rr_off;
      }
    }
  }

  // ------------------------------------------------------------------
  // Kokkos Views for device data
  // ------------------------------------------------------------------
  using ViewChar = Kokkos::View<char*>;
  using ViewU32  = Kokkos::View<uint32_t*>;
  using ViewDbl  = Kokkos::View<double*>;
  using ViewHT   = Kokkos::View<loc_ht*>;
  using ViewBool = Kokkos::View<loc_ht_bool*>;

  uint32_t ctg_total_len = ctg_offsets_h[tot_extensions-1];
  uint32_t lr_total      = reads_l_offset_h.empty() ? 1 : reads_l_offset_h.back();
  uint32_t rr_total      = reads_r_offset_h.empty() ? 1 : reads_r_offset_h.back();

  ViewChar d_ctg_seqs   ("ctg_seqs",    ctg_total_len);
  ViewChar d_ctg_rc     ("ctg_rc",      ctg_total_len);
  ViewU32  d_ctg_offsets("ctg_offsets", tot_extensions);
  ViewU32  d_prefix_ht  ("prefix_ht",   tot_extensions);
  ViewDbl  d_depth      ("depth",       tot_extensions);
  ViewU32  d_cid        ("cid",         tot_extensions);
  ViewChar d_reads_r    ("reads_r",     rr_total);
  ViewChar d_quals_r    ("quals_r",     rr_total);
  ViewChar d_reads_l    ("reads_l",     lr_total);
  ViewChar d_quals_l    ("quals_l",     lr_total);
  ViewU32  d_rds_r_off  ("rds_r_off",   total_r_reads > 0 ? total_r_reads : 1);
  ViewU32  d_rds_l_off  ("rds_l_off",   total_l_reads > 0 ? total_l_reads : 1);
  ViewU32  d_rds_r_cnt  ("rds_r_cnt",   tot_extensions);
  ViewU32  d_rds_l_cnt  ("rds_l_cnt",   tot_extensions);
  ViewHT   d_ht         ("global_ht",   ht_tot_size > 0 ? ht_tot_size : 1);
  ViewChar d_longest    ("longest",     (size_t)tot_extensions * max_walk_len);
  ViewChar d_mer_walk   ("mer_walk",    (size_t)tot_extensions *
                                         (max_walk_len + max_mer_len_val));
  ViewBool d_ht_bool    ("ht_bool",     (size_t)tot_extensions * max_walk_len);
  ViewU32  d_walk_lens  ("walk_lens",   tot_extensions);

  // Helper to copy std::vector → Kokkos View
  auto vec_to_view = [](const auto& vec, auto& view, size_t n) {
    auto hm = Kokkos::create_mirror_view(view);
    for (size_t i = 0; i < n; i++) hm(i) = vec[i];
    Kokkos::deep_copy(view, hm);
  };

  vec_to_view(ctg_seqs_h,      d_ctg_seqs,    ctg_total_len);
  vec_to_view(ctg_offsets_h,   d_ctg_offsets, tot_extensions);
  vec_to_view(prefix_ht_h,     d_prefix_ht,   tot_extensions);
  vec_to_view(depth_h,         d_depth,        tot_extensions);
  vec_to_view(cid_h,           d_cid,         tot_extensions);
  if (total_r_reads > 0) {
    vec_to_view(reads_r_offset_h, d_rds_r_off, total_r_reads);
    vec_to_view(reads_right_h,    d_reads_r,   rr_total);
    vec_to_view(quals_right_h,    d_quals_r,   rr_total);
  }
  if (total_l_reads > 0) {
    vec_to_view(reads_l_offset_h, d_rds_l_off, total_l_reads);
    vec_to_view(reads_left_h,     d_reads_l,   lr_total);
    vec_to_view(quals_left_h,     d_quals_l,   lr_total);
  }
  vec_to_view(rds_r_cnt_h, d_rds_r_cnt, tot_extensions);
  vec_to_view(rds_l_cnt_h, d_rds_l_cnt, tot_extensions);

  // Init HT to EMPTY
  Kokkos::parallel_for("init_ht", d_ht.size(), KOKKOS_LAMBDA(int k) {
    d_ht(k).key.length = (int)EMPTY;
  });
  Kokkos::fence();

  // ---- RIGHT WALK ----
  {
    char*     ctg_ptr    = d_ctg_seqs.data();
    uint32_t* ctg_off    = d_ctg_offsets.data();
    char*     rr_ptr     = d_reads_r.data();
    char*     qr_ptr     = d_quals_r.data();
    uint32_t* rr_off_ptr = d_rds_r_off.data();
    uint32_t* rr_cnt_ptr = d_rds_r_cnt.data();
    double*   dep_ptr    = d_depth.data();
    loc_ht*   ht_ptr     = d_ht.data();
    uint32_t* pht_ptr    = d_prefix_ht.data();
    loc_ht_bool* htb_ptr = d_ht_bool.data();
    char*     lw_ptr     = d_longest.data();
    char*     mw_ptr     = d_mer_walk.data();
    uint32_t* wl_ptr     = d_walk_lens.data();
    uint32_t* cid_ptr    = d_cid.data();
    int       kml        = mer_len;
    int       mwl        = max_walk_len;
    uint32_t  mmlo       = (uint32_t)max_mer_len_val;
    uint32_t  qoff       = qual_offset;
    int       ntot       = (int)tot_extensions;

    Kokkos::parallel_for("locht_right",
      Kokkos::RangePolicy<>(0, tot_extensions),
      KOKKOS_LAMBDA(const int idx) {
        iterative_walks_body(
          idx, cid_ptr, ctg_off, ctg_ptr,
          rr_ptr, qr_ptr, rr_off_ptr, rr_cnt_ptr,
          dep_ptr, ht_ptr, pht_ptr, htb_ptr,
          kml, mmlo, mwl,
          lw_ptr, mw_ptr, wl_ptr, ntot, qoff);
      });
    Kokkos::fence();

    auto h_lens = Kokkos::create_mirror_view(d_walk_lens);
    auto h_lw   = Kokkos::create_mirror_view(d_longest);
    Kokkos::deep_copy(h_lens, d_walk_lens);
    Kokkos::deep_copy(h_lw,   d_longest);
    for (uint32_t i = 0; i < tot_extensions; i++) {
      final_walk_lens_r[i] = h_lens(i);
      for (int c = 0; c < max_walk_len; c++)
        longest_walks_r[i * max_walk_len + c] = h_lw(i * max_walk_len + c);
    }
  }

  // ---- Compute revcomp of contig sequences for LEFT walk ----
  {
    uint32_t off = 0;
    for (uint32_t i = 0; i < tot_extensions; i++) {
      uint32_t sz = ctg_offsets_h[i] - (i > 0 ? ctg_offsets_h[i-1] : 0);
      revcomp(ctg_seqs_h.data() + (i > 0 ? ctg_offsets_h[i-1] : 0),
              ctg_rc_h.data()   + (i > 0 ? ctg_offsets_h[i-1] : 0), sz);
      off += sz;
    }
    vec_to_view(ctg_rc_h, d_ctg_rc, ctg_total_len);
  }

  // Reset walk lens / longest walks
  Kokkos::deep_copy(d_walk_lens, (uint32_t)0);
  Kokkos::deep_copy(d_longest, (char)0);

  // ---- LEFT WALK ----
  {
    char*     ctg_ptr    = d_ctg_rc.data();
    uint32_t* ctg_off    = d_ctg_offsets.data();
    char*     lr_ptr     = d_reads_l.data();
    char*     ql_ptr     = d_quals_l.data();
    uint32_t* lr_off_ptr = d_rds_l_off.data();
    uint32_t* lr_cnt_ptr = d_rds_l_cnt.data();
    double*   dep_ptr    = d_depth.data();
    loc_ht*   ht_ptr     = d_ht.data();
    uint32_t* pht_ptr    = d_prefix_ht.data();
    loc_ht_bool* htb_ptr = d_ht_bool.data();
    char*     lw_ptr     = d_longest.data();
    char*     mw_ptr     = d_mer_walk.data();
    uint32_t* wl_ptr     = d_walk_lens.data();
    uint32_t* cid_ptr    = d_cid.data();
    int       kml        = mer_len;
    int       mwl        = max_walk_len;
    uint32_t  mmlo       = (uint32_t)max_mer_len_val;
    uint32_t  qoff       = qual_offset;
    int       ntot       = (int)tot_extensions;

    Kokkos::parallel_for("locht_left",
      Kokkos::RangePolicy<>(0, tot_extensions),
      KOKKOS_LAMBDA(const int idx) {
        iterative_walks_body(
          idx, cid_ptr, ctg_off, ctg_ptr,
          lr_ptr, ql_ptr, lr_off_ptr, lr_cnt_ptr,
          dep_ptr, ht_ptr, pht_ptr, htb_ptr,
          kml, mmlo, mwl,
          lw_ptr, mw_ptr, wl_ptr, ntot, qoff);
      });
    Kokkos::fence();

    auto h_lens = Kokkos::create_mirror_view(d_walk_lens);
    auto h_lw   = Kokkos::create_mirror_view(d_longest);
    Kokkos::deep_copy(h_lens, d_walk_lens);
    Kokkos::deep_copy(h_lw,   d_longest);
    for (uint32_t i = 0; i < tot_extensions; i++) {
      final_walk_lens_l[i] = h_lens(i);
      for (int c = 0; c < max_walk_len; c++)
        longest_walks_l[i * max_walk_len + c] = h_lw(i * max_walk_len + c);
    }
  }

  // ------------------------------------------------------------------
  // Write results to output file
  // ------------------------------------------------------------------
  for (uint32_t i = 0; i < tot_extensions; i++) {
    uint32_t right_len = final_walk_lens_r[i];
    uint32_t left_len  = final_walk_lens_l[i];

    std::string right_walk(longest_walks_r.data() + i * max_walk_len, right_len);
    std::string left_walk (longest_walks_l.data() + i * max_walk_len, left_len);

    std::string left_rc = revcomp(left_walk);
    std::string new_seq = left_rc + data_in[i].seq + right_walk;
    out_file_g << data_in[i].cid << " " << new_seq << "\n";
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4) {
      std::cout << "Usage:\n";
      std::cout << "./main ../locassm_data/localassm_extend_7-21.dat 21 ./out_file.dat\n";
      Kokkos::finalize();
      return 0;
    }

    std::string in_file   = argv[1];
    int         mer_len   = std::stoi(argv[2]);
    std::ofstream ofile(argv[3]);

    std::vector<CtgWithReads> data_in;
    uint32_t max_ctg_size, tot_r, tot_l, max_read_size, max_r_cnt, max_l_cnt;
    read_locassm_data(&data_in, in_file, max_ctg_size, tot_r, tot_l,
                      max_read_size, max_r_cnt, max_l_cnt);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Partition contigs into zero / mid (1-9 reads) / outlier (10+ reads)
    std::vector<CtgWithReads> zero_slice, mid_slice, outlier_slice;
    accum_data sizes_mid, sizes_outliers;

    for (auto& ctg : data_in) {
      if (ctg.max_reads == 0) {
        zero_slice.push_back(ctg);
      } else if (ctg.max_reads < 10) {
        mid_slice.push_back(ctg);
        sizes_mid.ht_sizes.push_back((uint32_t)(ctg.max_reads * max_read_size));
        sizes_mid.ctg_sizes.push_back((uint32_t)ctg.seq.size());
        sizes_mid.l_reads_count.push_back((uint32_t)ctg.reads_left.size());
        sizes_mid.r_reads_count.push_back((uint32_t)ctg.reads_right.size());
      } else {
        outlier_slice.push_back(ctg);
        sizes_outliers.ht_sizes.push_back((uint32_t)(ctg.max_reads * max_read_size));
        sizes_outliers.ctg_sizes.push_back((uint32_t)ctg.seq.size());
        sizes_outliers.l_reads_count.push_back((uint32_t)ctg.reads_left.size());
        sizes_outliers.r_reads_count.push_back((uint32_t)ctg.reads_right.size());
      }
    }

    print_vals("zeroes, count:", zero_slice.size());
    for (auto& z : zero_slice)
      ofile << z.cid << " " << z.seq << "\n";
    zero_slice.clear();
    data_in.clear();

    // Compute per-partition max values
    uint32_t mid_l_max = 0, mid_r_max = 0, mid_max_ctg = 0;
    for (auto& c : mid_slice) {
      if (c.reads_left.size()  > mid_l_max) mid_l_max = (uint32_t)c.reads_left.size();
      if (c.reads_right.size() > mid_r_max) mid_r_max = (uint32_t)c.reads_right.size();
      if (c.seq.size()         > mid_max_ctg) mid_max_ctg = (uint32_t)c.seq.size();
    }
    uint32_t out_l_max = 0, out_r_max = 0, out_max_ctg = 0;
    for (auto& c : outlier_slice) {
      if (c.reads_left.size()  > out_l_max) out_l_max = (uint32_t)c.reads_left.size();
      if (c.reads_right.size() > out_r_max) out_r_max = (uint32_t)c.reads_right.size();
      if (c.seq.size()         > out_max_ctg) out_max_ctg = (uint32_t)c.seq.size();
    }

    print_vals("mids calling", "mids count:", mid_slice.size());
    call_kernel(mid_slice, mid_max_ctg, max_read_size, mid_r_max, mid_l_max,
                mer_len, 10, sizes_mid, ofile);

    print_vals("outliers calling", "outliers count:", outlier_slice.size());
    call_kernel(outlier_slice, out_max_ctg, max_read_size, out_r_max, out_l_max,
                mer_len, 10, sizes_outliers, ofile);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = t_end - t_start;
    print_vals("Total Time including file write:", total.count());

    ofile.flush();
    ofile.close();
  }
  Kokkos::finalize();
  return 0;
}
