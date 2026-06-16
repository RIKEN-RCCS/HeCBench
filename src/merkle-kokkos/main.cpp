// Kokkos port of merkle-cuda benchmark
// Merkle tree using Rescue Prime hash over F(2^64 - 2^32 + 1)
#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>
#include <climits>

typedef unsigned long ulong;

// ---------------------------------------------------------------------------
// ulong4: 4-element vector of uint64 (replaces CUDA's ulong4)
// ---------------------------------------------------------------------------
struct ulong4 {
  uint64_t x, y, z, w;
};

KOKKOS_INLINE_FUNCTION ulong4 make_ulong4(uint64_t a, uint64_t b,
                                           uint64_t c, uint64_t d) {
  return {a, b, c, d};
}

// ---------------------------------------------------------------------------
// Prime field constants: p = 2^64 - 2^32 + 1
// ---------------------------------------------------------------------------
inline constexpr uint64_t MOD =
  ((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;

inline constexpr uint64_t STATE_WIDTH = 12;
inline constexpr uint64_t RATE_WIDTH  = 8;
inline constexpr uint64_t DIGEST_SIZE = 4;
inline constexpr uint64_t NUM_ROUNDS  = 7;
inline constexpr uint64_t MAX_UINT    = 0xFFFFFFFFULL;

// ---------------------------------------------------------------------------
// Portable 64-bit high multiply
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION uint64_t umul64hi(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#elif defined(__SIZEOF_INT128__)
  return (uint64_t)(((unsigned __int128)a * b) >> 64);
#else
  // Fallback: split into 32-bit halves
  uint64_t lo_a = a & 0xFFFFFFFFULL, hi_a = a >> 32;
  uint64_t lo_b = b & 0xFFFFFFFFULL, hi_b = b >> 32;
  uint64_t mid1 = lo_a * hi_b, mid2 = hi_a * lo_b;
  uint64_t lo = lo_a * lo_b;
  uint64_t carry = ((lo >> 32) + (mid1 & 0xFFFFFFFFULL) + (mid2 & 0xFFFFFFFFULL)) >> 32;
  return hi_a * hi_b + (mid1 >> 32) + (mid2 >> 32) + carry;
#endif
}

// ---------------------------------------------------------------------------
// ulong4 operator overloads (device + host)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION ulong4 operator*(const ulong4& a, const ulong4& b) {
  return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w};
}
KOKKOS_INLINE_FUNCTION ulong4 operator+(const ulong4& a, const ulong4& b) {
  return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}
KOKKOS_INLINE_FUNCTION ulong4 operator-(const ulong4& a, const ulong4& b) {
  return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w};
}
KOKKOS_INLINE_FUNCTION ulong4 operator-(uint64_t a, const ulong4& b) {
  return {a-b.x, a-b.y, a-b.z, a-b.w};
}
KOKKOS_INLINE_FUNCTION ulong4 operator&(const ulong4& a, uint64_t b) {
  return {a.x&b, a.y&b, a.z&b, a.w&b};
}
KOKKOS_INLINE_FUNCTION ulong4 operator>>(const ulong4& a, int b) {
  return {a.x>>b, a.y>>b, a.z>>b, a.w>>b};
}
KOKKOS_INLINE_FUNCTION ulong4 operator<<(const ulong4& a, int b) {
  return {a.x<<b, a.y<<b, a.z<<b, a.w<<b};
}
KOKKOS_INLINE_FUNCTION ulong4 cmp_lt(const ulong4& a, const ulong4& b) {
  return {(a.x<b.x)?ULONG_MAX:0, (a.y<b.y)?ULONG_MAX:0,
          (a.z<b.z)?ULONG_MAX:0, (a.w<b.w)?ULONG_MAX:0};
}
KOKKOS_INLINE_FUNCTION ulong4 cmp_gt(const ulong4& a, const ulong4& b) {
  return {(a.x>b.x)?ULONG_MAX:0, (a.y>b.y)?ULONG_MAX:0,
          (a.z>b.z)?ULONG_MAX:0, (a.w>b.w)?ULONG_MAX:0};
}
KOKKOS_INLINE_FUNCTION ulong4 cmp_ge(const ulong4& a, const ulong4& b) {
  return {(a.x>=b.x)?ULONG_MAX:0, (a.y>=b.y)?ULONG_MAX:0,
          (a.z>=b.z)?ULONG_MAX:0, (a.w>=b.w)?ULONG_MAX:0};
}

KOKKOS_INLINE_FUNCTION ulong4 mul_hi(const ulong4& a, const ulong4& b) {
  return {umul64hi(a.x,b.x), umul64hi(a.y,b.y),
          umul64hi(a.z,b.z), umul64hi(a.w,b.w)};
}

// ---------------------------------------------------------------------------
// Field arithmetic (ff_p)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION uint64_t ff_p_add(uint64_t a, uint64_t b) {
  if (b >= MOD) b -= MOD;
  uint64_t res = a + b;
  bool ov = (a > UINT64_MAX - b);
  uint64_t t = (uint64_t)(-(uint32_t)(ov ? 1 : 0));
  res += t;
  bool ov2 = (res < t && ov);
  res += (uint64_t)(-(uint32_t)(ov2 ? 1 : 0));
  return res;
}

KOKKOS_INLINE_FUNCTION uint64_t ff_p_sub(uint64_t a, uint64_t b) {
  if (b >= MOD) b -= MOD;
  uint64_t res = a - b;
  bool un = (a < b);
  uint64_t t = (uint64_t)(-(uint32_t)(un ? 1 : 0));
  res -= t;
  bool un2 = (res > a && un);
  res += (uint64_t)(-(uint32_t)(un2 ? 1 : 0));
  return res;
}

KOKKOS_INLINE_FUNCTION uint64_t ff_p_mult(uint64_t a, uint64_t b) {
  if (b >= MOD) b -= MOD;
  uint64_t ab = a * b;
  uint64_t cd_hi = umul64hi(a, b);
  uint64_t c = cd_hi & 0x00000000ffffffffULL;
  uint64_t d = cd_hi >> 32;

  uint64_t res = ab - d;
  bool un = (ab < d);
  uint64_t t = (uint64_t)(-(uint32_t)(un ? 1 : 0));
  res -= t;

  uint64_t t1 = (c << 32) - c;
  uint64_t res2 = res + t1;
  bool ov = (res > UINT64_MAX - t1);
  res2 += (uint64_t)(-(uint32_t)(ov ? 1 : 0));
  return res2;
}

KOKKOS_INLINE_FUNCTION uint64_t ff_p_pow(uint64_t a, uint64_t b) {
  if (b == 0) return 1;
  if (b == 1) return a;
  if (a == 0) return 0;

  // count bits in b
  int bits = 64;
  while (bits > 0 && !((b >> (bits-1)) & 1)) bits--;

  uint64_t r = (b & 1) ? a : 1;
  for (int i = 1; i < bits; i++) {
    a = ff_p_mult(a, a);
    if ((b >> i) & 1) r = ff_p_mult(r, a);
  }
  return r;
}

KOKKOS_INLINE_FUNCTION uint64_t ff_p_inv(uint64_t a) {
  if (a >= MOD) a -= MOD;
  if (a == 0) return 0;
  return ff_p_pow(a, MOD - 2);
}

// ---------------------------------------------------------------------------
// ulong4 field arithmetic (vectorized)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION ulong4 ff_p_vec_mul_(ulong4 a, ulong4 b) {
  if (b.x >= MOD) b.x -= MOD;
  if (b.y >= MOD) b.y -= MOD;
  if (b.z >= MOD) b.z -= MOD;
  if (b.w >= MOD) b.w -= MOD;

  ulong4 ab = a * b;
  ulong4 cd = mul_hi(a, b);
  ulong4 c = cd & 0x00000000ffffffffULL;
  ulong4 d = cd >> 32;

  ulong4 res0 = ab - d;
  ulong4 un0 = cmp_lt(ab, d);
  res0 = res0 - (un0 & 1ULL);

  ulong4 t1 = (c << 32) - c;
  ulong4 res1 = res0 + t1;
  ulong4 ov0 = cmp_gt(res0, make_ulong4(UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX) - t1);
  res1 = res1 + (ov0 & 1ULL);
  return res1;
}

KOKKOS_INLINE_FUNCTION void ff_p_vec_mul(const ulong4* a, const ulong4* b,
                                          ulong4* c) {
  c[0] = ff_p_vec_mul_(a[0], b[0]);
  c[1] = ff_p_vec_mul_(a[1], b[1]);
  c[2] = ff_p_vec_mul_(a[2], b[2]);
}

KOKKOS_INLINE_FUNCTION ulong4 ff_p_vec_add_(const ulong4& a, const ulong4& b) {
  ulong4 res;
  res.x = ff_p_add(a.x, b.x);
  res.y = ff_p_add(a.y, b.y);
  res.z = ff_p_add(a.z, b.z);
  res.w = ff_p_add(a.w, b.w);
  return res;
}

KOKKOS_INLINE_FUNCTION void ff_p_vec_add(const ulong4* a, const ulong4* b,
                                          ulong4* c) {
  c[0] = ff_p_vec_add_(a[0], b[0]);
  c[1] = ff_p_vec_add_(a[1], b[1]);
  c[2] = ff_p_vec_add_(a[2], b[2]);
}

// ---------------------------------------------------------------------------
// Rescue Prime constants
// ---------------------------------------------------------------------------
inline constexpr ulong MDS_DATA[144] = {
  2108866337646019936ull,  11223275256334781131ull, 2318414738826783588ull,
  11240468238955543594ull, 8007389560317667115ull,  11080831380224887131ull,
  3922954383102346493ull,  17194066286743901609ull, 152620255842323114ull,
  7203302445933022224ull,  17781531460838764471ull, 2306881200ull,

  3368836954250922620ull,  5531382716338105518ull,  7747104620279034727ull,
  14164487169476525880ull, 4653455932372793639ull,  5504123103633670518ull,
  3376629427948045767ull,  1687083899297674997ull,  8324288417826065247ull,
  17651364087632826504ull, 15568475755679636039ull, 4656488262337620150ull,

  2560535215714666606ull,  10793518538122219186ull, 408467828146985886ull,
  13894393744319723897ull, 17856013635663093677ull, 14510101432365346218ull,
  12175743201430386993ull, 12012700097100374591ull, 976880602086740182ull,
  3187015135043748111ull,  4630899319883688283ull,  17674195666610532297ull,

  10940635879119829731ull, 9126204055164541072ull,  13441880452578323624ull,
  13828699194559433302ull, 6245685172712904082ull,  3117562785727957263ull,
  17389107632996288753ull, 3643151412418457029ull,  10484080975961167028ull,
  4066673631745731889ull,  8847974898748751041ull,  9548808324754121113ull,

  15656099696515372126ull, 309741777966979967ull,   16075523529922094036ull,
  5384192144218250710ull,  15171244241641106028ull, 6660319859038124593ull,
  6595450094003204814ull,  15330207556174961057ull, 2687301105226976975ull,
  15907414358067140389ull, 2767130804164179683ull,  8135839249549115549ull,

  14687393836444508153ull, 8122848807512458890ull,  16998154830503301252ull,
  2904046703764323264ull,  11170142989407566484ull, 5448553946207765015ull,
  9766047029091333225ull,  3852354853341479440ull,  14577128274897891003ull,
  11994931371916133447ull, 8299269445020599466ull,  2859592328380146288ull,

  4920761474064525703ull,  13379538658122003618ull, 3169184545474588182ull,
  15753261541491539618ull, 622292315133191494ull,   14052907820095169428ull,
  5159844729950547044ull,  17439978194716087321ull, 9945483003842285313ull,
  13647273880020281344ull, 14750994260825376ull,    12575187259316461486ull,

  3371852905554824605ull,  8886257005679683950ull,  15677115160380392279ull,
  13242906482047961505ull, 12149996307978507817ull, 1427861135554592284ull,
  4033726302273030373ull,  14761176804905342155ull, 11465247508084706095ull,
  12112647677590318112ull, 17343938135425110721ull, 14654483060427620352ull,

  5421794552262605237ull,  14201164512563303484ull, 5290621264363227639ull,
  1020180205893205576ull,  14311345105258400438ull, 7828111500457301560ull,
  9436759291445548340ull,  5716067521736967068ull,  15357555109169671716ull,
  4131452666376493252ull,  16785275933585465720ull, 11180136753375315897ull,

  10451661389735482801ull, 12128852772276583847ull, 10630876800354432923ull,
  6884824371838330777ull,  16413552665026570512ull, 13637837753341196082ull,
  2558124068257217718ull,  4327919242598628564ull,  4236040195908057312ull,
  2081029262044280559ull,  2047510589162918469ull,  6835491236529222042ull,

  5675273097893923172ull,  8120839782755215647ull,  9856415804450870143ull,
  1960632704307471239ull,  15279057263127523057ull, 17999325337309257121ull,
  72970456904683065ull,    8899624805082057509ull,  16980481565524365258ull,
  6412696708929498357ull,  13917768671775544479ull, 5505378218427096880ull,

  10318314766641004576ull, 17320192463105632563ull, 11540812969169097044ull,
  7270556942018024148ull,  4755326086930560682ull,  2193604418377108959ull,
  11681945506511803967ull, 8000243866012209465ull,  6746478642521594042ull,
  12096331252283646217ull, 13208137848575217268ull, 5548519654341606996ull,
};

inline constexpr ulong ARK1_DATA[84] = {
  13917550007135091859ull, 16002276252647722320ull, 4729924423368391595ull,
  10059693067827680263ull, 9804807372516189948ull,  15666751576116384237ull,
  10150587679474953119ull, 13627942357577414247ull, 2323786301545403792ull,
  615170742765998613ull,   8870655212817778103ull,  10534167191270683080ull,

  14572151513649018290ull, 9445470642301863087ull,  6565801926598404534ull,
  12667566692985038975ull, 7193782419267459720ull,  11874811971940314298ull,
  17906868010477466257ull, 1237247437760523561ull,  6829882458376718831ull,
  2140011966759485221ull,  1624379354686052121ull,  50954653459374206ull,

  16288075653722020941ull, 13294924199301620952ull, 13370596140726871456ull,
  611533288599636281ull,   12865221627554828747ull, 12269498015480242943ull,
  8230863118714645896ull,  13466591048726906480ull, 10176988631229240256ull,
  14951460136371189405ull, 5882405912332577353ull,  18125144098115032453ull,

  6076976409066920174ull,  7466617867456719866ull,  5509452692963105675ull,
  14692460717212261752ull, 12980373618703329746ull, 1361187191725412610ull,
  6093955025012408881ull,  5110883082899748359ull,  8578179704817414083ull,
  9311749071195681469ull,  16965242536774914613ull, 5747454353875601040ull,

  13684212076160345083ull, 19445754899749561ull,    16618768069125744845ull,
  278225951958825090ull,   4997246680116830377ull,  782614868534172852ull,
  16423767594935000044ull, 9990984633405879434ull,  16757120847103156641ull,
  2103861168279461168ull,  16018697163142305052ull, 6479823382130993799ull,

  13957683526597936825ull, 9702819874074407511ull,  18357323897135139931ull,
  3029452444431245019ull,  1809322684009991117ull,  12459356450895788575ull,
  11985094908667810946ull, 12868806590346066108ull, 7872185587893926881ull,
  10694372443883124306ull, 8644995046789277522ull,  1422920069067375692ull,

  17619517835351328008ull, 6173683530634627901ull,  15061027706054897896ull,
  4503753322633415655ull,  11538516425871008333ull, 12777459872202073891ull,
  17842814708228807409ull, 13441695826912633916ull, 5950710620243434509ull,
  17040450522225825296ull, 8787650312632423701ull,  7431110942091427450ull,
};

inline constexpr ulong ARK2_DATA[84] = {
  7989257206380839449ull,  8639509123020237648ull,  6488561830509603695ull,
  5519169995467998761ull,  2972173318556248829ull,  14899875358187389787ull,
  14160104549881494022ull, 5969738169680657501ull,  5116050734813646528ull,
  12120002089437618419ull, 17404470791907152876ull, 2718166276419445724ull,
  2485377440770793394ull,  14358936485713564605ull, 3327012975585973824ull,
  6001912612374303716ull,  17419159457659073951ull, 11810720562576658327ull,
  14802512641816370470ull, 751963320628219432ull,   9410455736958787393ull,
  16405548341306967018ull, 6867376949398252373ull,  13982182448213113532ull,
  10436926105997283389ull, 13237521312283579132ull, 668335841375552722ull,
  2385521647573044240ull,  3874694023045931809ull,  12952434030222726182ull,
  1972984540857058687ull,  14000313505684510403ull, 976377933822676506ull,
  8407002393718726702ull,  338785660775650958ull,   4208211193539481671ull,
  2284392243703840734ull,  4500504737691218932ull,  3976085877224857941ull,
  2603294837319327956ull,  5760259105023371034ull,  2911579958858769248ull,
  18415938932239013434ull, 7063156700464743997ull,  16626114991069403630ull,
  163485390956217960ull,   11596043559919659130ull, 2976841507452846995ull,
  15090073748392700862ull, 3496786927732034743ull,  8646735362535504000ull,
  2460088694130347125ull,  3944675034557577794ull,  14781700518249159275ull,
  2857749437648203959ull,  8505429584078195973ull,  18008150643764164736ull,
  720176627102578275ull,   7038653538629322181ull,  8849746187975356582ull,
  17427790390280348710ull, 1159544160012040055ull,  17946663256456930598ull,
  6338793524502945410ull,  17715539080731926288ull, 4208940652334891422ull,
  12386490721239135719ull, 10010817080957769535ull, 5566101162185411405ull,
  12520146553271266365ull, 4972547404153988943ull,  5597076522138709717ull,
  18338863478027005376ull, 115128380230345639ull,   4427489889653730058ull,
  10890727269603281956ull, 7094492770210294530ull,  7345573238864544283ull,
  6834103517673002336ull,  14002814950696095900ull, 15939230865809555943ull,
  12717309295554119359ull, 4130723396860574906ull,  7706153020203677238ull,
};

// ---------------------------------------------------------------------------
// Rescue Prime hash functions  (KOKKOS_INLINE_FUNCTION = device+host)
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION void apply_sbox(const ulong4* state_in,
                                        ulong4* state_out) {
  ulong4 t2[3], t4[3];
  ff_p_vec_mul(state_in, state_in, t2);
  ff_p_vec_mul(t2, t2, t4);
  for (int i = 0; i < 3; i++) {
    ulong4 tmp;
    tmp.x = ff_p_mult(t4[i].x % MOD, t2[i].x % MOD);
    tmp.y = ff_p_mult(t4[i].y % MOD, t2[i].y % MOD);
    tmp.z = ff_p_mult(t4[i].z % MOD, t2[i].z % MOD);
    tmp.w = ff_p_mult(t4[i].w % MOD, t2[i].w % MOD);
    // t6 = t2*t4
    state_out[i].x = ff_p_mult(tmp.x % MOD, state_in[i].x % MOD);
    state_out[i].y = ff_p_mult(tmp.y % MOD, state_in[i].y % MOD);
    state_out[i].z = ff_p_mult(tmp.z % MOD, state_in[i].z % MOD);
    state_out[i].w = ff_p_mult(tmp.w % MOD, state_in[i].w % MOD);
  }
}

KOKKOS_INLINE_FUNCTION void apply_constants(const ulong4* state_in,
                                             const ulong4* cnst,
                                             ulong4* state_out) {
  ff_p_vec_add(state_in, cnst, state_out);
}

KOKKOS_INLINE_FUNCTION uint64_t accumulate_vec4(ulong4 a) {
  uint64_t v0 = ff_p_add(a.x, a.y);
  uint64_t v1 = ff_p_add(a.z, a.w);
  return ff_p_add(v0, v1);
}

KOKKOS_INLINE_FUNCTION uint64_t accumulate_state(const ulong4* state) {
  uint64_t v0 = accumulate_vec4(state[0]);
  uint64_t v1 = accumulate_vec4(state[1]);
  uint64_t v2 = accumulate_vec4(state[2]);
  return ff_p_add(v2, ff_p_add(v0, v1));
}

KOKKOS_INLINE_FUNCTION void apply_mds(const ulong4* state_in,
                                       const ulong4* mds,
                                       ulong4* state_out) {
  ulong4 scratch[3];
  uint64_t vals[STATE_WIDTH];
  for (uint64_t row = 0; row < STATE_WIDTH; row++) {
    ff_p_vec_mul(state_in, mds + row * 3, scratch);
    vals[row] = accumulate_state(scratch);
  }
  state_out[0] = make_ulong4(vals[0],  vals[1],  vals[2],  vals[3]);
  state_out[1] = make_ulong4(vals[4],  vals[5],  vals[6],  vals[7]);
  state_out[2] = make_ulong4(vals[8],  vals[9],  vals[10], vals[11]);
}

KOKKOS_INLINE_FUNCTION void exp_acc(uint64_t m, const ulong4* base,
                                     const ulong4* tail, ulong4* out) {
  out[0] = base[0]; out[1] = base[1]; out[2] = base[2];
  for (uint64_t i = 0; i < m; i++) {
    ulong4 sc[3];
    ff_p_vec_mul(out, out, sc);
    out[0] = sc[0]; out[1] = sc[1]; out[2] = sc[2];
  }
  ulong4 sc[3];
  ff_p_vec_mul(out, tail, sc);
  out[0] = sc[0]; out[1] = sc[1]; out[2] = sc[2];
}

KOKKOS_INLINE_FUNCTION void apply_inv_sbox(const ulong4* state_in,
                                            ulong4* state_out) {
  ulong4 t1[3], t2[3], t3[3], t4[3], t5[3], t6[3], t7[3];
  ff_p_vec_mul(state_in, state_in, t1);
  ff_p_vec_mul(t1, t1, t2);
  exp_acc(3,  t2, t2, t3);
  exp_acc(6,  t3, t3, t4);
  exp_acc(12, t4, t4, t5);
  exp_acc(6,  t5, t3, t6);
  exp_acc(31, t6, t6, t7);

  ulong4 a[3], b[3], sc[3];
  ff_p_vec_mul(t7, t7, sc);
  ff_p_vec_mul(t6, sc, a);
  ff_p_vec_mul(a,  a,  sc);
  ff_p_vec_mul(sc, sc, a);

  ff_p_vec_mul(t1, t2, sc);
  ff_p_vec_mul(sc, state_in, b);

  ff_p_vec_mul(a, b, state_out);
}

KOKKOS_INLINE_FUNCTION void apply_permutation_round(const ulong4* state_in,
                                                      const ulong4* mds,
                                                      const ulong4* ark1,
                                                      const ulong4* ark2,
                                                      ulong4* state_out) {
  ulong4 s0[3], s1[3], s2[3];
  apply_sbox(state_in, s0);
  apply_mds(s0, mds, s1);
  apply_constants(s1, ark1, s2);
  apply_inv_sbox(s2, s0);
  apply_mds(s0, mds, s1);
  apply_constants(s1, ark2, state_out);
}

KOKKOS_INLINE_FUNCTION void apply_rescue_permutation(const ulong4* state_in,
                                                       const ulong4* mds,
                                                       const ulong4* ark1,
                                                       const ulong4* ark2,
                                                       ulong4* state_out) {
  ulong4 s0[3], s1[3], s2[3];
  apply_permutation_round(state_in, mds, ark1+0,  ark2+0,  s0);
  apply_permutation_round(s0,       mds, ark1+3,  ark2+3,  s1);
  apply_permutation_round(s1,       mds, ark1+6,  ark2+6,  s2);
  apply_permutation_round(s2,       mds, ark1+9,  ark2+9,  s0);
  apply_permutation_round(s0,       mds, ark1+12, ark2+12, s1);
  apply_permutation_round(s1,       mds, ark1+15, ark2+15, s2);
  apply_permutation_round(s2,       mds, ark1+18, ark2+18, state_out);
}

// Merges two DIGEST_SIZE-element digests into one DIGEST_SIZE-element digest.
// input_hashes points to 2*DIGEST_SIZE = 8 consecutive ulong values.
KOKKOS_INLINE_FUNCTION void merge(const ulong* input_hashes,
                                   ulong* merged_hash,
                                   const ulong4* mds,
                                   const ulong4* ark1,
                                   const ulong4* ark2) {
  ulong4 state[3] = {
    make_ulong4(input_hashes[0], input_hashes[1],
                input_hashes[2], input_hashes[3]),
    make_ulong4(input_hashes[4], input_hashes[5],
                input_hashes[6], input_hashes[7]),
    make_ulong4(0, 0, 0, RATE_WIDTH)
  };
  ulong4 scratch[3];
  apply_rescue_permutation(state, mds, ark1, ark2, scratch);
  merged_hash[0] = scratch[0].x;
  merged_hash[1] = scratch[0].y;
  merged_hash[2] = scratch[0].z;
  merged_hash[3] = scratch[0].w;
}

// ---------------------------------------------------------------------------
// Prepare MDS and ARK constants into packed ulong4 arrays
// ---------------------------------------------------------------------------
void prepare_mds(ulong4* mds) {
  // STATE_WIDTH * 3 rows, each row = 4 consecutive elements packed into ulong4
  for (size_t i = 0; i < STATE_WIDTH * 3; i++) {
    mds[i] = make_ulong4(MDS_DATA[i*4+0], MDS_DATA[i*4+1],
                          MDS_DATA[i*4+2], MDS_DATA[i*4+3]);
  }
}
void prepare_ark1(ulong4* ark1) {
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    ark1[i] = make_ulong4(ARK1_DATA[i*4+0], ARK1_DATA[i*4+1],
                           ARK1_DATA[i*4+2], ARK1_DATA[i*4+3]);
  }
}
void prepare_ark2(ulong4* ark2) {
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    ark2[i] = make_ulong4(ARK2_DATA[i*4+0], ARK2_DATA[i*4+1],
                           ARK2_DATA[i*4+2], ARK2_DATA[i*4+3]);
  }
}

// ---------------------------------------------------------------------------
// Merkle tree construction using Kokkos
// ---------------------------------------------------------------------------
uint64_t merklize_approach_1_kokkos(const ulong* leaves_h,
                                     ulong* intermediates_h,
                                     size_t leaf_count,
                                     size_t /*wg_size unused*/,
                                     const ulong4* mds_h,
                                     const ulong4* ark1_h,
                                     const ulong4* ark2_h) {
  assert((leaf_count & (leaf_count - 1)) == 0);

  const size_t leaves_elems  = leaf_count * DIGEST_SIZE;
  const size_t mds_elems     = STATE_WIDTH * 3;
  const size_t ark_elems     = NUM_ROUNDS * 3;

  // Allocate device views
  Kokkos::View<ulong*>  d_leaves("leaves",        leaves_elems);
  Kokkos::View<ulong*>  d_inter ("intermediates", leaves_elems);
  Kokkos::View<ulong4*> d_mds   ("mds",  mds_elems);
  Kokkos::View<ulong4*> d_ark1  ("ark1", ark_elems);
  Kokkos::View<ulong4*> d_ark2  ("ark2", ark_elems);

  // Host mirrors for initial copy
  auto h_leaves = Kokkos::create_mirror_view(d_leaves);
  auto h_mds    = Kokkos::create_mirror_view(d_mds);
  auto h_ark1   = Kokkos::create_mirror_view(d_ark1);
  auto h_ark2   = Kokkos::create_mirror_view(d_ark2);

  for (size_t i = 0; i < leaves_elems; i++) h_leaves(i) = leaves_h[i];
  for (size_t i = 0; i < mds_elems;    i++) h_mds(i)    = mds_h[i];
  for (size_t i = 0; i < ark_elems;    i++) h_ark1(i)   = ark1_h[i];
  for (size_t i = 0; i < ark_elems;    i++) h_ark2(i)   = ark2_h[i];

  Kokkos::deep_copy(d_leaves, h_leaves);
  Kokkos::deep_copy(d_mds,    h_mds);
  Kokkos::deep_copy(d_ark1,   h_ark1);
  Kokkos::deep_copy(d_ark2,   h_ark2);
  Kokkos::fence();

  auto start = std::chrono::high_resolution_clock::now();

  const size_t output_offset = leaf_count >> 1;

  // Phase 0: pair up leaf digests → first level of intermediate nodes
  {
    ulong* leaves_ptr = d_leaves.data();
    ulong* inter_ptr  = d_inter.data();
    ulong4* mds_ptr   = d_mds.data();
    ulong4* ark1_ptr  = d_ark1.data();
    ulong4* ark2_ptr  = d_ark2.data();
    size_t out_off    = output_offset;

    Kokkos::parallel_for("phase0", output_offset,
      KOKKOS_LAMBDA(const size_t idx) {
        merge(leaves_ptr + idx * (DIGEST_SIZE >> 1),
              inter_ptr  + (out_off + idx) * DIGEST_SIZE,
              mds_ptr, ark1_ptr, ark2_ptr);
      });
    Kokkos::fence();
  }

  // Phase 1..rounds: interior levels
  const size_t rounds =
    static_cast<size_t>(std::log2(static_cast<double>(leaf_count >> 1)));

  for (size_t r = 0; r < rounds; r++) {
    const size_t offset = leaf_count >> (r + 2);

    ulong* inter_ptr = d_inter.data();
    ulong4* mds_ptr  = d_mds.data();
    ulong4* ark1_ptr = d_ark1.data();
    ulong4* ark2_ptr = d_ark2.data();

    Kokkos::parallel_for("phase1", offset,
      KOKKOS_LAMBDA(const size_t idx) {
        merge(inter_ptr + (offset << 1) * DIGEST_SIZE + idx * (DIGEST_SIZE >> 1),
              inter_ptr + (offset + idx) * DIGEST_SIZE,
              mds_ptr, ark1_ptr, ark2_ptr);
      });
    Kokkos::fence();
  }

  auto end = std::chrono::high_resolution_clock::now();
  uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();

  // Copy back
  auto h_inter = Kokkos::create_mirror_view(d_inter);
  Kokkos::deep_copy(h_inter, d_inter);
  for (size_t i = 0; i < leaves_elems; i++) intermediates_h[i] = h_inter(i);

  return ns;
}

// ---------------------------------------------------------------------------
// Benchmark wrapper
// ---------------------------------------------------------------------------
uint64_t benchmark_merklize_approach_1(size_t leaf_count, size_t wg_size) {
  const size_t leaves_elems = leaf_count * DIGEST_SIZE;
  const size_t mds_elems    = STATE_WIDTH * 3;
  const size_t ark_elems    = NUM_ROUNDS * 3;

  std::vector<ulong>  leaves_h(leaves_elems);
  std::vector<ulong>  inter_h(leaves_elems, 0);
  std::vector<ulong4> mds_h(mds_elems);
  std::vector<ulong4> ark1_h(ark_elems);
  std::vector<ulong4> ark2_h(ark_elems);

  {
    std::mt19937_64 gen(19937);
    std::uniform_int_distribution<uint64_t> dis(1ull, MOD);
    for (auto& v : leaves_h) v = dis(gen);
  }

  prepare_mds (mds_h.data());
  prepare_ark1(ark1_h.data());
  prepare_ark2(ark2_h.data());

  return merklize_approach_1_kokkos(
    leaves_h.data(), inter_h.data(), leaf_count, wg_size,
    mds_h.data(), ark1_h.data(), ark2_h.data());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nMerklize ( approach 1 ) using Rescue Prime on "
                 "F(2**64 - 2**32 + 1) elements\n\n";
    std::cout << std::setw(11) << "leaves"
              << "\t\t" << std::setw(15) << "total" << std::endl;

    const size_t BENCH_ROUND = 4;
    for (uint32_t dim = (1ul << 20); dim <= (1ul << 24); dim <<= 1) {
      double tm = 0;
      for (size_t i = 0; i < BENCH_ROUND; i++)
        tm += static_cast<double>(benchmark_merklize_approach_1(dim, 1ul << 5));
      tm /= static_cast<double>(BENCH_ROUND);

      std::cout << std::setw(11) << std::right << dim
                << "\t\t" << std::setw(15) << std::right
                << tm * 1e-6 << " ms" << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
