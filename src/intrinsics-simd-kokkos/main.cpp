#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ── Lane extraction helpers ───────────────────────────────────────────────────

// 16-bit lanes (2-lane packed in uint32)
KOKKOS_INLINE_FUNCTION int32_t  s16lo(uint32_t v) { return (int32_t)(int16_t)(v & 0xFFFFu); }
KOKKOS_INLINE_FUNCTION int32_t  s16hi(uint32_t v) { return (int32_t)(int16_t)(v >> 16); }
KOKKOS_INLINE_FUNCTION uint32_t u16lo(uint32_t v) { return v & 0xFFFFu; }
KOKKOS_INLINE_FUNCTION uint32_t u16hi(uint32_t v) { return v >> 16; }
KOKKOS_INLINE_FUNCTION uint32_t pack16(int32_t lo, int32_t hi)
{ return ((uint32_t)(uint16_t)(int16_t)lo) | ((uint32_t)(uint16_t)(int16_t)hi << 16); }
KOKKOS_INLINE_FUNCTION uint32_t upack16(uint32_t lo, uint32_t hi)
{ return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16); }

// 8-bit lanes (4-lane packed in uint32)
KOKKOS_INLINE_FUNCTION int32_t  s8b0(uint32_t v) { return (int32_t)(int8_t)(v & 0xFFu); }
KOKKOS_INLINE_FUNCTION int32_t  s8b1(uint32_t v) { return (int32_t)(int8_t)((v >> 8) & 0xFFu); }
KOKKOS_INLINE_FUNCTION int32_t  s8b2(uint32_t v) { return (int32_t)(int8_t)((v >> 16) & 0xFFu); }
KOKKOS_INLINE_FUNCTION int32_t  s8b3(uint32_t v) { return (int32_t)(int8_t)(v >> 24); }
KOKKOS_INLINE_FUNCTION uint32_t u8b0(uint32_t v) { return v & 0xFFu; }
KOKKOS_INLINE_FUNCTION uint32_t u8b1(uint32_t v) { return (v >> 8) & 0xFFu; }
KOKKOS_INLINE_FUNCTION uint32_t u8b2(uint32_t v) { return (v >> 16) & 0xFFu; }
KOKKOS_INLINE_FUNCTION uint32_t u8b3(uint32_t v) { return v >> 24; }
KOKKOS_INLINE_FUNCTION uint32_t pack8(int32_t b0, int32_t b1, int32_t b2, int32_t b3) {
  return ((uint32_t)(uint8_t)(int8_t)b0)        |
         ((uint32_t)(uint8_t)(int8_t)b1 << 8)   |
         ((uint32_t)(uint8_t)(int8_t)b2 << 16)  |
         ((uint32_t)(uint8_t)(int8_t)b3 << 24);
}
KOKKOS_INLINE_FUNCTION uint32_t upack8(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
  return (b0 & 0xFFu) | ((b1 & 0xFFu) << 8) | ((b2 & 0xFFu) << 16) | ((b3 & 0xFFu) << 24);
}

// Saturation helpers
KOKKOS_INLINE_FUNCTION int32_t clamp_s16(int32_t x) {
  return x < -32768 ? -32768 : (x > 32767 ? 32767 : x);
}
KOKKOS_INLINE_FUNCTION int32_t clamp_u16(int32_t x) {
  return x < 0 ? 0 : (x > 65535 ? 65535 : x);
}
KOKKOS_INLINE_FUNCTION int32_t clamp_s8(int32_t x) {
  return x < -128 ? -128 : (x > 127 ? 127 : x);
}
KOKKOS_INLINE_FUNCTION int32_t clamp_u8(int32_t x) {
  return x < 0 ? 0 : (x > 255 ? 255 : x);
}

// ── SIMD operation implementations ───────────────────────────────────────────

// --- vabs ---
KOKKOS_INLINE_FUNCTION uint32_t vabs2(uint32_t a) {
  int32_t l = s16lo(a), h = s16hi(a);
  return pack16(l < 0 ? -l : l, h < 0 ? -h : h);
}
KOKKOS_INLINE_FUNCTION uint32_t vabs4(uint32_t a) {
  int32_t b0 = s8b0(a), b1 = s8b1(a), b2 = s8b2(a), b3 = s8b3(a);
  return pack8(b0<0?-b0:b0, b1<0?-b1:b1, b2<0?-b2:b2, b3<0?-b3:b3);
}

// --- vabsdiffs (saturating signed subtraction then abs) ---
KOKKOS_INLINE_FUNCTION uint32_t vabsdiffs2(uint32_t a, uint32_t b) {
  int32_t dl = clamp_s16(s16lo(a) - s16lo(b));
  int32_t dh = clamp_s16(s16hi(a) - s16hi(b));
  return pack16(dl < 0 ? -dl : dl, dh < 0 ? -dh : dh);
}
KOKKOS_INLINE_FUNCTION uint32_t vabsdiffs4(uint32_t a, uint32_t b) {
  int32_t d0 = clamp_s8(s8b0(a)-s8b0(b)), d1 = clamp_s8(s8b1(a)-s8b1(b));
  int32_t d2 = clamp_s8(s8b2(a)-s8b2(b)), d3 = clamp_s8(s8b3(a)-s8b3(b));
  return pack8(d0<0?-d0:d0, d1<0?-d1:d1, d2<0?-d2:d2, d3<0?-d3:d3);
}

// --- vabsdiffu (unsigned abs difference) ---
KOKKOS_INLINE_FUNCTION uint32_t vabsdiffu2(uint32_t a, uint32_t b) {
  int32_t dl = (int32_t)u16lo(a) - (int32_t)u16lo(b);
  int32_t dh = (int32_t)u16hi(a) - (int32_t)u16hi(b);
  return upack16((uint32_t)(dl<0?-dl:dl), (uint32_t)(dh<0?-dh:dh));
}
KOKKOS_INLINE_FUNCTION uint32_t vabsdiffu4(uint32_t a, uint32_t b) {
  int32_t d0=(int32_t)u8b0(a)-(int32_t)u8b0(b), d1=(int32_t)u8b1(a)-(int32_t)u8b1(b);
  int32_t d2=(int32_t)u8b2(a)-(int32_t)u8b2(b), d3=(int32_t)u8b3(a)-(int32_t)u8b3(b);
  return upack8((uint32_t)(d0<0?-d0:d0),(uint32_t)(d1<0?-d1:d1),
                (uint32_t)(d2<0?-d2:d2),(uint32_t)(d3<0?-d3:d3));
}

// --- vabsss (saturating abs) ---
KOKKOS_INLINE_FUNCTION uint32_t vabsss2(uint32_t a) {
  int32_t l = s16lo(a), h = s16hi(a);
  if (l == -32768) l = 32767; else if (l < 0) l = -l;
  if (h == -32768) h = 32767; else if (h < 0) h = -h;
  return pack16(l, h);
}
KOKKOS_INLINE_FUNCTION uint32_t vabsss4(uint32_t a) {
  int32_t b0=s8b0(a),b1=s8b1(a),b2=s8b2(a),b3=s8b3(a);
  auto ss=[](int32_t x){return x==-128?127:(x<0?-x:x);};
  return pack8(ss(b0),ss(b1),ss(b2),ss(b3));
}

// --- vadd (wrapping) ---
KOKKOS_INLINE_FUNCTION uint32_t vadd2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)+u16lo(b), u16hi(a)+u16hi(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vadd4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)+u8b0(b), u8b1(a)+u8b1(b), u8b2(a)+u8b2(b), u8b3(a)+u8b3(b));
}

// --- vaddss (signed saturating add) ---
KOKKOS_INLINE_FUNCTION uint32_t vaddss2(uint32_t a, uint32_t b) {
  return pack16(clamp_s16(s16lo(a)+s16lo(b)), clamp_s16(s16hi(a)+s16hi(b)));
}
KOKKOS_INLINE_FUNCTION uint32_t vaddss4(uint32_t a, uint32_t b) {
  return pack8(clamp_s8(s8b0(a)+s8b0(b)), clamp_s8(s8b1(a)+s8b1(b)),
               clamp_s8(s8b2(a)+s8b2(b)), clamp_s8(s8b3(a)+s8b3(b)));
}

// --- vaddus (unsigned saturating add) ---
KOKKOS_INLINE_FUNCTION uint32_t vaddus2(uint32_t a, uint32_t b) {
  return upack16(clamp_u16((int32_t)u16lo(a)+(int32_t)u16lo(b)),
                 clamp_u16((int32_t)u16hi(a)+(int32_t)u16hi(b)));
}
KOKKOS_INLINE_FUNCTION uint32_t vaddus4(uint32_t a, uint32_t b) {
  return upack8(clamp_u8((int32_t)u8b0(a)+(int32_t)u8b0(b)),
                clamp_u8((int32_t)u8b1(a)+(int32_t)u8b1(b)),
                clamp_u8((int32_t)u8b2(a)+(int32_t)u8b2(b)),
                clamp_u8((int32_t)u8b3(a)+(int32_t)u8b3(b)));
}

// --- vavgs (signed average, round toward +inf) ---
KOKKOS_INLINE_FUNCTION uint32_t vavgs2(uint32_t a, uint32_t b) {
  // (x+y+1)>>1 with arithmetic shift gives round-to-+inf average
  int32_t l = s16lo(a)+s16lo(b), h = s16hi(a)+s16hi(b);
  // arithmetic right shift by 1: preserve sign
  return pack16((int16_t)((l+1) >> 1), (int16_t)((h+1) >> 1));
}
KOKKOS_INLINE_FUNCTION uint32_t vavgs4(uint32_t a, uint32_t b) {
  auto avg=[](int32_t x,int32_t y){return (int8_t)((x+y+1)>>1);};
  return pack8(avg(s8b0(a),s8b0(b)),avg(s8b1(a),s8b1(b)),
               avg(s8b2(a),s8b2(b)),avg(s8b3(a),s8b3(b)));
}

// --- vavgu (unsigned average, round toward +inf) ---
KOKKOS_INLINE_FUNCTION uint32_t vavgu2(uint32_t a, uint32_t b) {
  return upack16((u16lo(a)+u16lo(b)+1u)>>1, (u16hi(a)+u16hi(b)+1u)>>1);
}
KOKKOS_INLINE_FUNCTION uint32_t vavgu4(uint32_t a, uint32_t b) {
  return upack8((u8b0(a)+u8b0(b)+1u)>>1, (u8b1(a)+u8b1(b)+1u)>>1,
                (u8b2(a)+u8b2(b)+1u)>>1, (u8b3(a)+u8b3(b)+1u)>>1);
}

// --- vcmpeq (equal → 0xFFFF/0xFF or 0) ---
KOKKOS_INLINE_FUNCTION uint32_t vcmpeq2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)==u16lo(b)?0xFFFFu:0u, u16hi(a)==u16hi(b)?0xFFFFu:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpeq4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)==u8b0(b)?0xFFu:0u, u8b1(a)==u8b1(b)?0xFFu:0u,
                u8b2(a)==u8b2(b)?0xFFu:0u, u8b3(a)==u8b3(b)?0xFFu:0u);
}

// --- vcmpge signed ---
KOKKOS_INLINE_FUNCTION uint32_t vcmpges2(uint32_t a, uint32_t b) {
  return pack16(s16lo(a)>=s16lo(b)?(int32_t)0xFFFF:0, s16hi(a)>=s16hi(b)?(int32_t)0xFFFF:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpges4(uint32_t a, uint32_t b) {
  return pack8(s8b0(a)>=s8b0(b)?0x7F:0, s8b1(a)>=s8b1(b)?0x7F:0,
               s8b2(a)>=s8b2(b)?0x7F:0, s8b3(a)>=s8b3(b)?0x7F:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpgeu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)>=u16lo(b)?0xFFFFu:0u, u16hi(a)>=u16hi(b)?0xFFFFu:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpgeu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)>=u8b0(b)?0xFFu:0u, u8b1(a)>=u8b1(b)?0xFFu:0u,
                u8b2(a)>=u8b2(b)?0xFFu:0u, u8b3(a)>=u8b3(b)?0xFFu:0u);
}

// --- vcmpgt signed/unsigned ---
KOKKOS_INLINE_FUNCTION uint32_t vcmpgts2(uint32_t a, uint32_t b) {
  return pack16(s16lo(a)>s16lo(b)?(int32_t)0xFFFF:0, s16hi(a)>s16hi(b)?(int32_t)0xFFFF:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpgts4(uint32_t a, uint32_t b) {
  return pack8(s8b0(a)>s8b0(b)?0x7F:0, s8b1(a)>s8b1(b)?0x7F:0,
               s8b2(a)>s8b2(b)?0x7F:0, s8b3(a)>s8b3(b)?0x7F:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpgtu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)>u16lo(b)?0xFFFFu:0u, u16hi(a)>u16hi(b)?0xFFFFu:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpgtu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)>u8b0(b)?0xFFu:0u, u8b1(a)>u8b1(b)?0xFFu:0u,
                u8b2(a)>u8b2(b)?0xFFu:0u, u8b3(a)>u8b3(b)?0xFFu:0u);
}

// --- vcmple signed/unsigned ---
KOKKOS_INLINE_FUNCTION uint32_t vcmples2(uint32_t a, uint32_t b) {
  return pack16(s16lo(a)<=s16lo(b)?(int32_t)0xFFFF:0, s16hi(a)<=s16hi(b)?(int32_t)0xFFFF:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmples4(uint32_t a, uint32_t b) {
  return pack8(s8b0(a)<=s8b0(b)?0x7F:0, s8b1(a)<=s8b1(b)?0x7F:0,
               s8b2(a)<=s8b2(b)?0x7F:0, s8b3(a)<=s8b3(b)?0x7F:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpleu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)<=u16lo(b)?0xFFFFu:0u, u16hi(a)<=u16hi(b)?0xFFFFu:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpleu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)<=u8b0(b)?0xFFu:0u, u8b1(a)<=u8b1(b)?0xFFu:0u,
                u8b2(a)<=u8b2(b)?0xFFu:0u, u8b3(a)<=u8b3(b)?0xFFu:0u);
}

// --- vcmplt signed/unsigned ---
KOKKOS_INLINE_FUNCTION uint32_t vcmplts2(uint32_t a, uint32_t b) {
  return pack16(s16lo(a)<s16lo(b)?(int32_t)0xFFFF:0, s16hi(a)<s16hi(b)?(int32_t)0xFFFF:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmplts4(uint32_t a, uint32_t b) {
  return pack8(s8b0(a)<s8b0(b)?0x7F:0, s8b1(a)<s8b1(b)?0x7F:0,
               s8b2(a)<s8b2(b)?0x7F:0, s8b3(a)<s8b3(b)?0x7F:0);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpltu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)<u16lo(b)?0xFFFFu:0u, u16hi(a)<u16hi(b)?0xFFFFu:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpltu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)<u8b0(b)?0xFFu:0u, u8b1(a)<u8b1(b)?0xFFu:0u,
                u8b2(a)<u8b2(b)?0xFFu:0u, u8b3(a)<u8b3(b)?0xFFu:0u);
}

// --- vcmpne ---
KOKKOS_INLINE_FUNCTION uint32_t vcmpne2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)!=u16lo(b)?0xFFFFu:0u, u16hi(a)!=u16hi(b)?0xFFFFu:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vcmpne4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)!=u8b0(b)?0xFFu:0u, u8b1(a)!=u8b1(b)?0xFFu:0u,
                u8b2(a)!=u8b2(b)?0xFFu:0u, u8b3(a)!=u8b3(b)?0xFFu:0u);
}

// --- vhaddu (half unsigned add, truncated average) ---
KOKKOS_INLINE_FUNCTION uint32_t vhaddu2(uint32_t a, uint32_t b) {
  return upack16((u16lo(a)+u16lo(b))>>1, (u16hi(a)+u16hi(b))>>1);
}
KOKKOS_INLINE_FUNCTION uint32_t vhaddu4(uint32_t a, uint32_t b) {
  return upack8((u8b0(a)+u8b0(b))>>1, (u8b1(a)+u8b1(b))>>1,
                (u8b2(a)+u8b2(b))>>1, (u8b3(a)+u8b3(b))>>1);
}

// ── vi* operations (CUDA Volta+) ──────────────────────────────────────────────

// --- viaddmax (max(a+b, c)) ---
KOKKOS_INLINE_FUNCTION uint32_t viaddmax_s16x2(uint32_t a, uint32_t b, uint32_t c) {
  int32_t sl = (int16_t)((int16_t)(s16lo(a) + s16lo(b))); // wrapping add
  int32_t sh = (int16_t)((int16_t)(s16hi(a) + s16hi(b)));
  return pack16(sl > s16lo(c) ? sl : s16lo(c), sh > s16hi(c) ? sh : s16hi(c));
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmax_s16x2_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t sl = (int16_t)(s16lo(a) + s16lo(b));
  int32_t sh = (int16_t)(s16hi(a) + s16hi(b));
  sl = sl > s16lo(c) ? sl : s16lo(c); if (sl < 0) sl = 0;
  sh = sh > s16hi(c) ? sh : s16hi(c); if (sh < 0) sh = 0;
  return pack16(sl, sh);
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmax_s32(uint32_t a, uint32_t b, uint32_t c) {
  int32_t s = (int32_t)a + (int32_t)b;
  int32_t cc = (int32_t)c;
  return (uint32_t)(s > cc ? s : cc);
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmax_s32_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t s = (int32_t)a + (int32_t)b;
  int32_t cc = (int32_t)c;
  int32_t r = s > cc ? s : cc;
  return (uint32_t)(r < 0 ? 0 : r);
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmax_u16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t sl = (uint16_t)(u16lo(a) + u16lo(b));  // wrapping uint16 add
  uint32_t sh = (uint16_t)(u16hi(a) + u16hi(b));
  return upack16(sl > u16lo(c) ? sl : u16lo(c), sh > u16hi(c) ? sh : u16hi(c));
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmax_u32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t s = a + b;
  return s > c ? s : c;
}

// --- viaddmin (min(a+b, c)) ---
KOKKOS_INLINE_FUNCTION uint32_t viaddmin_s16x2(uint32_t a, uint32_t b, uint32_t c) {
  int32_t sl = (int16_t)(s16lo(a) + s16lo(b));
  int32_t sh = (int16_t)(s16hi(a) + s16hi(b));
  return pack16(sl < s16lo(c) ? sl : s16lo(c), sh < s16hi(c) ? sh : s16hi(c));
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmin_s16x2_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t sl = (int16_t)(s16lo(a) + s16lo(b));
  int32_t sh = (int16_t)(s16hi(a) + s16hi(b));
  sl = sl < s16lo(c) ? sl : s16lo(c); if (sl < 0) sl = 0;
  sh = sh < s16hi(c) ? sh : s16hi(c); if (sh < 0) sh = 0;
  return pack16(sl, sh);
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmin_s32(uint32_t a, uint32_t b, uint32_t c) {
  int32_t s = (int32_t)a + (int32_t)b;
  int32_t cc = (int32_t)c;
  return (uint32_t)(s < cc ? s : cc);
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmin_s32_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t s = (int32_t)a + (int32_t)b;
  int32_t cc = (int32_t)c;
  int32_t r = s < cc ? s : cc;
  return (uint32_t)(r < 0 ? 0 : r);
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmin_u16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t sl = (uint16_t)(u16lo(a) + u16lo(b));
  uint32_t sh = (uint16_t)(u16hi(a) + u16hi(b));
  return upack16(sl < u16lo(c) ? sl : u16lo(c), sh < u16hi(c) ? sh : u16hi(c));
}
KOKKOS_INLINE_FUNCTION uint32_t viaddmin_u32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t s = a + b;
  return s < c ? s : c;
}

// --- vibmax (max with pred output) ---
KOKKOS_INLINE_FUNCTION uint32_t vibmax_s16x2(uint32_t a, uint32_t b,
                                              bool *pred_hi, bool *pred_lo) {
  int32_t al = s16lo(a), bl = s16lo(b), ah = s16hi(a), bh = s16hi(b);
  *pred_lo = (al >= bl); *pred_hi = (ah >= bh);
  return pack16(al >= bl ? al : bl, ah >= bh ? ah : bh);
}
KOKKOS_INLINE_FUNCTION uint32_t vibmax_s32(uint32_t a, uint32_t b, bool *pred) {
  int32_t ia = (int32_t)a, ib = (int32_t)b;
  *pred = (ia >= ib);
  return (uint32_t)(ia >= ib ? ia : ib);
}
KOKKOS_INLINE_FUNCTION uint32_t vibmax_u16x2(uint32_t a, uint32_t b,
                                              bool *pred_hi, bool *pred_lo) {
  uint32_t al = u16lo(a), bl = u16lo(b), ah = u16hi(a), bh = u16hi(b);
  *pred_lo = (al >= bl); *pred_hi = (ah >= bh);
  return upack16(al >= bl ? al : bl, ah >= bh ? ah : bh);
}
KOKKOS_INLINE_FUNCTION uint32_t vibmax_u32(uint32_t a, uint32_t b, bool *pred) {
  *pred = (a >= b);
  return a >= b ? a : b;
}

// --- vibmin (min with pred output) ---
KOKKOS_INLINE_FUNCTION uint32_t vibmin_s16x2(uint32_t a, uint32_t b,
                                              bool *pred_hi, bool *pred_lo) {
  int32_t al = s16lo(a), bl = s16lo(b), ah = s16hi(a), bh = s16hi(b);
  *pred_lo = (al <= bl); *pred_hi = (ah <= bh);
  return pack16(al <= bl ? al : bl, ah <= bh ? ah : bh);
}
KOKKOS_INLINE_FUNCTION uint32_t vibmin_s32(uint32_t a, uint32_t b, bool *pred) {
  int32_t ia = (int32_t)a, ib = (int32_t)b;
  *pred = (ia <= ib);
  return (uint32_t)(ia <= ib ? ia : ib);
}
KOKKOS_INLINE_FUNCTION uint32_t vibmin_u16x2(uint32_t a, uint32_t b,
                                              bool *pred_hi, bool *pred_lo) {
  uint32_t al = u16lo(a), bl = u16lo(b), ah = u16hi(a), bh = u16hi(b);
  *pred_lo = (al <= bl); *pred_hi = (ah <= bh);
  return upack16(al <= bl ? al : bl, ah <= bh ? ah : bh);
}
KOKKOS_INLINE_FUNCTION uint32_t vibmin_u32(uint32_t a, uint32_t b, bool *pred) {
  *pred = (a <= b);
  return a <= b ? a : b;
}

// --- vimax3 / vimin3 (3-operand max/min) ---
KOKKOS_INLINE_FUNCTION uint32_t vimax3_s16x2(uint32_t a, uint32_t b, uint32_t c) {
  int32_t rl = s16lo(a), rh = s16hi(a);
  if (s16lo(b) > rl) rl = s16lo(b); if (s16lo(c) > rl) rl = s16lo(c);
  if (s16hi(b) > rh) rh = s16hi(b); if (s16hi(c) > rh) rh = s16hi(c);
  return pack16(rl, rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimax3_s16x2_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t rl = s16lo(a), rh = s16hi(a);
  if (s16lo(b) > rl) rl = s16lo(b); if (s16lo(c) > rl) rl = s16lo(c);
  if (s16hi(b) > rh) rh = s16hi(b); if (s16hi(c) > rh) rh = s16hi(c);
  return pack16(rl < 0 ? 0 : rl, rh < 0 ? 0 : rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimax3_s32(uint32_t a, uint32_t b, uint32_t c) {
  int32_t r = (int32_t)a;
  if ((int32_t)b > r) r = (int32_t)b;
  if ((int32_t)c > r) r = (int32_t)c;
  return (uint32_t)r;
}
KOKKOS_INLINE_FUNCTION uint32_t vimax3_s32_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t r = (int32_t)a;
  if ((int32_t)b > r) r = (int32_t)b;
  if ((int32_t)c > r) r = (int32_t)c;
  return (uint32_t)(r < 0 ? 0 : r);
}
KOKKOS_INLINE_FUNCTION uint32_t vimax3_u16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t rl = u16lo(a), rh = u16hi(a);
  if (u16lo(b) > rl) rl = u16lo(b); if (u16lo(c) > rl) rl = u16lo(c);
  if (u16hi(b) > rh) rh = u16hi(b); if (u16hi(c) > rh) rh = u16hi(c);
  return upack16(rl, rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimax3_u32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r = a;
  if (b > r) r = b; if (c > r) r = c;
  return r;
}

KOKKOS_INLINE_FUNCTION uint32_t vimin3_s16x2(uint32_t a, uint32_t b, uint32_t c) {
  int32_t rl = s16lo(a), rh = s16hi(a);
  if (s16lo(b) < rl) rl = s16lo(b); if (s16lo(c) < rl) rl = s16lo(c);
  if (s16hi(b) < rh) rh = s16hi(b); if (s16hi(c) < rh) rh = s16hi(c);
  return pack16(rl, rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimin3_s16x2_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t rl = s16lo(a), rh = s16hi(a);
  if (s16lo(b) < rl) rl = s16lo(b); if (s16lo(c) < rl) rl = s16lo(c);
  if (s16hi(b) < rh) rh = s16hi(b); if (s16hi(c) < rh) rh = s16hi(c);
  return pack16(rl < 0 ? 0 : rl, rh < 0 ? 0 : rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimin3_s32(uint32_t a, uint32_t b, uint32_t c) {
  int32_t r = (int32_t)a;
  if ((int32_t)b < r) r = (int32_t)b;
  if ((int32_t)c < r) r = (int32_t)c;
  return (uint32_t)r;
}
KOKKOS_INLINE_FUNCTION uint32_t vimin3_s32_relu(uint32_t a, uint32_t b, uint32_t c) {
  int32_t r = (int32_t)a;
  if ((int32_t)b < r) r = (int32_t)b;
  if ((int32_t)c < r) r = (int32_t)c;
  return (uint32_t)(r < 0 ? 0 : r);
}
KOKKOS_INLINE_FUNCTION uint32_t vimin3_u16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t rl = u16lo(a), rh = u16hi(a);
  if (u16lo(b) < rl) rl = u16lo(b); if (u16lo(c) < rl) rl = u16lo(c);
  if (u16hi(b) < rh) rh = u16hi(b); if (u16hi(c) < rh) rh = u16hi(c);
  return upack16(rl, rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimin3_u32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r = a;
  if (b < r) r = b; if (c < r) r = c;
  return r;
}

// --- vimax/vimin _relu (2-operand with ReLU) ---
KOKKOS_INLINE_FUNCTION uint32_t vimax_s16x2_relu(uint32_t a, uint32_t b) {
  int32_t rl = s16lo(a) >= s16lo(b) ? s16lo(a) : s16lo(b);
  int32_t rh = s16hi(a) >= s16hi(b) ? s16hi(a) : s16hi(b);
  return pack16(rl < 0 ? 0 : rl, rh < 0 ? 0 : rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimax_s32_relu(uint32_t a, uint32_t b) {
  int32_t r = (int32_t)a >= (int32_t)b ? (int32_t)a : (int32_t)b;
  return (uint32_t)(r < 0 ? 0 : r);
}
KOKKOS_INLINE_FUNCTION uint32_t vimin_s16x2_relu(uint32_t a, uint32_t b) {
  int32_t rl = s16lo(a) <= s16lo(b) ? s16lo(a) : s16lo(b);
  int32_t rh = s16hi(a) <= s16hi(b) ? s16hi(a) : s16hi(b);
  return pack16(rl < 0 ? 0 : rl, rh < 0 ? 0 : rh);
}
KOKKOS_INLINE_FUNCTION uint32_t vimin_s32_relu(uint32_t a, uint32_t b) {
  int32_t r = (int32_t)a <= (int32_t)b ? (int32_t)a : (int32_t)b;
  return (uint32_t)(r < 0 ? 0 : r);
}

// --- vmaxs / vmaxu / vmins / vminu ---
KOKKOS_INLINE_FUNCTION uint32_t vmaxs2(uint32_t a, uint32_t b) {
  return pack16(s16lo(a)>s16lo(b)?s16lo(a):s16lo(b), s16hi(a)>s16hi(b)?s16hi(a):s16hi(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vmaxs4(uint32_t a, uint32_t b) {
  return pack8(s8b0(a)>s8b0(b)?s8b0(a):s8b0(b), s8b1(a)>s8b1(b)?s8b1(a):s8b1(b),
               s8b2(a)>s8b2(b)?s8b2(a):s8b2(b), s8b3(a)>s8b3(b)?s8b3(a):s8b3(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vmaxu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)>u16lo(b)?u16lo(a):u16lo(b), u16hi(a)>u16hi(b)?u16hi(a):u16hi(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vmaxu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)>u8b0(b)?u8b0(a):u8b0(b), u8b1(a)>u8b1(b)?u8b1(a):u8b1(b),
                u8b2(a)>u8b2(b)?u8b2(a):u8b2(b), u8b3(a)>u8b3(b)?u8b3(a):u8b3(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vmins2(uint32_t a, uint32_t b) {
  return pack16(s16lo(a)<s16lo(b)?s16lo(a):s16lo(b), s16hi(a)<s16hi(b)?s16hi(a):s16hi(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vmins4(uint32_t a, uint32_t b) {
  return pack8(s8b0(a)<s8b0(b)?s8b0(a):s8b0(b), s8b1(a)<s8b1(b)?s8b1(a):s8b1(b),
               s8b2(a)<s8b2(b)?s8b2(a):s8b2(b), s8b3(a)<s8b3(b)?s8b3(a):s8b3(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vminu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)<u16lo(b)?u16lo(a):u16lo(b), u16hi(a)<u16hi(b)?u16hi(a):u16hi(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vminu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)<u8b0(b)?u8b0(a):u8b0(b), u8b1(a)<u8b1(b)?u8b1(a):u8b1(b),
                u8b2(a)<u8b2(b)?u8b2(a):u8b2(b), u8b3(a)<u8b3(b)?u8b3(a):u8b3(b));
}

// --- vneg / vnegss ---
KOKKOS_INLINE_FUNCTION uint32_t vneg2(uint32_t a) {
  return pack16(-s16lo(a), -s16hi(a));
}
KOKKOS_INLINE_FUNCTION uint32_t vneg4(uint32_t a) {
  return pack8(-s8b0(a), -s8b1(a), -s8b2(a), -s8b3(a));
}
KOKKOS_INLINE_FUNCTION uint32_t vnegss2(uint32_t a) {
  int32_t l = s16lo(a), h = s16hi(a);
  l = (l == -32768) ? 32767 : -l;
  h = (h == -32768) ? 32767 : -h;
  return pack16(l, h);
}
KOKKOS_INLINE_FUNCTION uint32_t vnegss4(uint32_t a) {
  auto ss=[](int32_t x){return x==-128?127:-x;};
  return pack8(ss(s8b0(a)),ss(s8b1(a)),ss(s8b2(a)),ss(s8b3(a)));
}

// --- vsad (sum of absolute differences) ---
KOKKOS_INLINE_FUNCTION uint32_t vsads2(uint32_t a, uint32_t b) {
  int32_t dl = s16lo(a)-s16lo(b), dh = s16hi(a)-s16hi(b);
  return (uint32_t)((dl<0?-dl:dl) + (dh<0?-dh:dh));
}
KOKKOS_INLINE_FUNCTION uint32_t vsads4(uint32_t a, uint32_t b) {
  int32_t d0=s8b0(a)-s8b0(b), d1=s8b1(a)-s8b1(b);
  int32_t d2=s8b2(a)-s8b2(b), d3=s8b3(a)-s8b3(b);
  return (uint32_t)((d0<0?-d0:d0)+(d1<0?-d1:d1)+(d2<0?-d2:d2)+(d3<0?-d3:d3));
}
KOKKOS_INLINE_FUNCTION uint32_t vsadu2(uint32_t a, uint32_t b) {
  int32_t dl=(int32_t)u16lo(a)-(int32_t)u16lo(b), dh=(int32_t)u16hi(a)-(int32_t)u16hi(b);
  return (uint32_t)((dl<0?-dl:dl)+(dh<0?-dh:dh));
}
KOKKOS_INLINE_FUNCTION uint32_t vsadu4(uint32_t a, uint32_t b) {
  int32_t d0=(int32_t)u8b0(a)-(int32_t)u8b0(b), d1=(int32_t)u8b1(a)-(int32_t)u8b1(b);
  int32_t d2=(int32_t)u8b2(a)-(int32_t)u8b2(b), d3=(int32_t)u8b3(a)-(int32_t)u8b3(b);
  return (uint32_t)((d0<0?-d0:d0)+(d1<0?-d1:d1)+(d2<0?-d2:d2)+(d3<0?-d3:d3));
}

// --- vset (compare → 0 or 1 per lane) ---
KOKKOS_INLINE_FUNCTION uint32_t vseteq2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)==u16lo(b)?1u:0u, u16hi(a)==u16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vseteq4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)==u8b0(b)?1u:0u, u8b1(a)==u8b1(b)?1u:0u,
                u8b2(a)==u8b2(b)?1u:0u, u8b3(a)==u8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetges2(uint32_t a, uint32_t b) {
  return upack16(s16lo(a)>=(int32_t)s16lo(b)?1u:0u, s16hi(a)>=(int32_t)s16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetges4(uint32_t a, uint32_t b) {
  return upack8(s8b0(a)>=s8b0(b)?1u:0u, s8b1(a)>=s8b1(b)?1u:0u,
                s8b2(a)>=s8b2(b)?1u:0u, s8b3(a)>=s8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetgeu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)>=u16lo(b)?1u:0u, u16hi(a)>=u16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetgeu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)>=u8b0(b)?1u:0u, u8b1(a)>=u8b1(b)?1u:0u,
                u8b2(a)>=u8b2(b)?1u:0u, u8b3(a)>=u8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetgts2(uint32_t a, uint32_t b) {
  return upack16(s16lo(a)>s16lo(b)?1u:0u, s16hi(a)>s16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetgts4(uint32_t a, uint32_t b) {
  return upack8(s8b0(a)>s8b0(b)?1u:0u, s8b1(a)>s8b1(b)?1u:0u,
                s8b2(a)>s8b2(b)?1u:0u, s8b3(a)>s8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetgtu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)>u16lo(b)?1u:0u, u16hi(a)>u16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetgtu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)>u8b0(b)?1u:0u, u8b1(a)>u8b1(b)?1u:0u,
                u8b2(a)>u8b2(b)?1u:0u, u8b3(a)>u8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetles2(uint32_t a, uint32_t b) {
  return upack16(s16lo(a)<=s16lo(b)?1u:0u, s16hi(a)<=s16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetles4(uint32_t a, uint32_t b) {
  return upack8(s8b0(a)<=s8b0(b)?1u:0u, s8b1(a)<=s8b1(b)?1u:0u,
                s8b2(a)<=s8b2(b)?1u:0u, s8b3(a)<=s8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetleu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)<=u16lo(b)?1u:0u, u16hi(a)<=u16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetleu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)<=u8b0(b)?1u:0u, u8b1(a)<=u8b1(b)?1u:0u,
                u8b2(a)<=u8b2(b)?1u:0u, u8b3(a)<=u8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetlts2(uint32_t a, uint32_t b) {
  return upack16(s16lo(a)<s16lo(b)?1u:0u, s16hi(a)<s16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetlts4(uint32_t a, uint32_t b) {
  return upack8(s8b0(a)<s8b0(b)?1u:0u, s8b1(a)<s8b1(b)?1u:0u,
                s8b2(a)<s8b2(b)?1u:0u, s8b3(a)<s8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetltu2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)<u16lo(b)?1u:0u, u16hi(a)<u16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetltu4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)<u8b0(b)?1u:0u, u8b1(a)<u8b1(b)?1u:0u,
                u8b2(a)<u8b2(b)?1u:0u, u8b3(a)<u8b3(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetne2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)!=u16lo(b)?1u:0u, u16hi(a)!=u16hi(b)?1u:0u);
}
KOKKOS_INLINE_FUNCTION uint32_t vsetne4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)!=u8b0(b)?1u:0u, u8b1(a)!=u8b1(b)?1u:0u,
                u8b2(a)!=u8b2(b)?1u:0u, u8b3(a)!=u8b3(b)?1u:0u);
}

// --- vsub (wrapping) ---
KOKKOS_INLINE_FUNCTION uint32_t vsub2(uint32_t a, uint32_t b) {
  return upack16(u16lo(a)-u16lo(b), u16hi(a)-u16hi(b));
}
KOKKOS_INLINE_FUNCTION uint32_t vsub4(uint32_t a, uint32_t b) {
  return upack8(u8b0(a)-u8b0(b), u8b1(a)-u8b1(b), u8b2(a)-u8b2(b), u8b3(a)-u8b3(b));
}

// --- vsubss (signed saturating subtract) ---
KOKKOS_INLINE_FUNCTION uint32_t vsubss2(uint32_t a, uint32_t b) {
  return pack16(clamp_s16(s16lo(a)-s16lo(b)), clamp_s16(s16hi(a)-s16hi(b)));
}
KOKKOS_INLINE_FUNCTION uint32_t vsubss4(uint32_t a, uint32_t b) {
  return pack8(clamp_s8(s8b0(a)-s8b0(b)), clamp_s8(s8b1(a)-s8b1(b)),
               clamp_s8(s8b2(a)-s8b2(b)), clamp_s8(s8b3(a)-s8b3(b)));
}

// --- vsubus (unsigned saturating subtract) ---
KOKKOS_INLINE_FUNCTION uint32_t vsubus2(uint32_t a, uint32_t b) {
  return upack16(clamp_u16((int32_t)u16lo(a)-(int32_t)u16lo(b)),
                 clamp_u16((int32_t)u16hi(a)-(int32_t)u16hi(b)));
}
KOKKOS_INLINE_FUNCTION uint32_t vsubus4(uint32_t a, uint32_t b) {
  return upack8(clamp_u8((int32_t)u8b0(a)-(int32_t)u8b0(b)),
                clamp_u8((int32_t)u8b1(a)-(int32_t)u8b1(b)),
                clamp_u8((int32_t)u8b2(a)-(int32_t)u8b2(b)),
                clamp_u8((int32_t)u8b3(a)-(int32_t)u8b3(b)));
}

// ── Kernel: applies all SIMD operations and XORs results ─────────────────────

KOKKOS_INLINE_FUNCTION uint32_t simd_compute(int i, uint32_t a)
{
  uint32_t b = a + ((i % 2) ? 1u : (uint32_t)(-1));
  uint32_t c = a ^ b;
  uint32_t r;

  r  = vabs2(a);
  r ^= vabs4(a);

  r ^= vabsdiffs2(a, b);
  r ^= vabsdiffs4(a, b);
  r ^= vabsdiffu2(a, b);
  r ^= vabsdiffu4(a, b);

  r ^= vabsss2(a);
  r ^= vabsss4(a);

  r ^= vadd2(a, b);
  r ^= vadd4(a, b);

  r ^= vaddss2(a, b);
  r ^= vaddss4(a, b);
  r ^= vaddus2(a, b);
  r ^= vaddus4(a, b);

  r ^= vavgs2(a, b);
  r ^= vavgs4(a, b);
  r ^= vavgu2(a, b);
  r ^= vavgu4(a, b);

  r ^= vcmpeq2(a, b);
  r ^= vcmpeq4(a, b);

  r ^= vcmpges2(a, b);
  r ^= vcmpges4(a, b);
  r ^= vcmpgeu2(a, b);
  r ^= vcmpgeu4(a, b);

  r ^= vcmpgts2(a, b);
  r ^= vcmpgts4(a, b);
  r ^= vcmpgtu2(a, b);
  r ^= vcmpgtu4(a, b);

  r ^= vcmples2(a, b);
  r ^= vcmples4(a, b);
  r ^= vcmpleu2(a, b);
  r ^= vcmpleu4(a, b);

  r ^= vcmplts2(a, b);
  r ^= vcmplts4(a, b);
  r ^= vcmpltu2(a, b);
  r ^= vcmpltu4(a, b);

  r ^= vcmpne2(a, b);
  r ^= vcmpne4(a, b);

  r ^= vhaddu2(a, b);
  r ^= vhaddu4(a, b);

  r ^= viaddmax_s16x2(a, b, c);
  r ^= viaddmax_s16x2_relu(a, b, c);
  r ^= viaddmax_s32(a, b, c);
  r ^= viaddmax_s32_relu(a, b, c);
  r ^= viaddmax_u16x2(a, b, c);
  r ^= viaddmax_u32(a, b, c);

  r ^= viaddmin_s16x2(a, b, c);
  r ^= viaddmin_s16x2_relu(a, b, c);
  r ^= viaddmin_s32(a, b, c);
  r ^= viaddmin_s32_relu(a, b, c);
  r ^= viaddmin_u16x2(a, b, c);
  r ^= viaddmin_u32(a, b, c);

  bool pred, pred_hi, pred_lo;

  r ^= vibmax_s16x2(a, b, &pred_hi, &pred_lo);
  r ^= vibmax_s32(a, b, &pred);
  r ^= vibmax_u16x2(a, b, &pred_hi, &pred_lo);
  r ^= vibmax_u32(a, b, &pred);

  r ^= vibmin_s16x2(a, b, &pred_hi, &pred_lo);
  r ^= vibmin_s32(a, b, &pred);
  r ^= vibmin_u16x2(a, b, &pred_hi, &pred_lo);
  r ^= vibmin_u32(a, b, &pred);

  r ^= vimax3_s16x2(a, b, c);
  r ^= vimax3_s16x2_relu(a, b, c);
  r ^= vimax3_s32(a, b, c);
  r ^= vimax3_s32_relu(a, b, c);
  r ^= vimax3_u16x2(a, b, c);
  r ^= vimax3_u32(a, b, c);

  r ^= vimax_s16x2_relu(a, b);
  r ^= vimax_s32_relu(a, b);

  r ^= vimin3_s16x2(a, b, c);
  r ^= vimin3_s16x2_relu(a, b, c);
  r ^= vimin3_s32(a, b, c);
  r ^= vimin3_s32_relu(a, b, c);
  r ^= vimin3_u16x2(a, b, c);
  r ^= vimin3_u32(a, b, c);

  r ^= vimin_s16x2_relu(a, b);
  r ^= vimin_s32_relu(a, b);

  r ^= vmaxs2(a, b);
  r ^= vmaxs4(a, b);
  r ^= vmaxu2(a, b);
  r ^= vmaxu4(a, b);

  r ^= vmins2(a, b);
  r ^= vmins4(a, b);
  r ^= vminu2(a, b);
  r ^= vminu4(a, b);

  r ^= vneg2(a);
  r ^= vneg4(a);
  r ^= vnegss2(a);
  r ^= vnegss4(a);

  r ^= vsads2(a, b);
  r ^= vsads4(a, b);
  r ^= vsadu2(a, b);
  r ^= vsadu4(a, b);

  r ^= vseteq2(a, b);
  r ^= vseteq4(a, b);

  r ^= vsetges2(a, b);
  r ^= vsetges4(a, b);
  r ^= vsetgeu2(a, b);
  r ^= vsetgeu4(a, b);

  r ^= vsetgts2(a, b);
  r ^= vsetgts4(a, b);
  r ^= vsetgtu2(a, b);
  r ^= vsetgtu4(a, b);

  r ^= vsetles2(a, b);
  r ^= vsetles4(a, b);
  r ^= vsetleu2(a, b);
  r ^= vsetleu4(a, b);

  r ^= vsetlts2(a, b);
  r ^= vsetlts4(a, b);
  r ^= vsetltu2(a, b);
  r ^= vsetltu4(a, b);

  r ^= vsetne2(a, b);
  r ^= vsetne4(a, b);

  r ^= vsub2(a, b);
  r ^= vsub4(a, b);
  r ^= vsubss2(a, b);
  r ^= vsubss4(a, b);
  r ^= vsubus2(a, b);
  r ^= vsubus4(a, b);

  return r;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  uint32_t *h_input  = new uint32_t[n];
  uint32_t *h_output = new uint32_t[n];

  for (int i = 0; i < n; i++)
    h_input[i] = 0x1234aba5u ^ (uint32_t)((i < n / 2) ? (-i) : i);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<uint32_t *> d_input ("input",  n);
    Kokkos::View<uint32_t *> d_output("output", n);

    {
      auto m_in = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < n; i++) m_in(i) = h_input[i];
      Kokkos::deep_copy(d_input, m_in);
    }
    Kokkos::View<uint32_t *>::value_type zero = 0;
    Kokkos::deep_copy(d_output, zero);

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++)
      Kokkos::parallel_for("simd", n, KOKKOS_LAMBDA(int i) {
        d_output(i) = simd_compute(i, d_input(i));
      });

    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("Average execution time of the SIMD intrinsics kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    auto m_out = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(m_out, d_output);

    uint32_t checksum = 0;
    for (int i = 0; i < n; i++) checksum ^= m_out(i);
    printf("Checksum = %x\n", checksum);
  }
  Kokkos::finalize();

  delete[] h_input;
  delete[] h_output;
  return 0;
}
