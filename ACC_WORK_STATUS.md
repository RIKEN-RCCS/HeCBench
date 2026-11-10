# acc ブランチ 作業状況メモ

> **注意: このファイルは作業引き継ぎ用の一時ファイルです。upstream に PR を出す前に必ず削除してください。**

- 最終更新: 2026-11-10（GPU マシン（GB200, cc100）で3章の確認を終えた時点）
- 対象ブランチ: `acc`（`origin` = `github.com:RIKEN-RCCS/HeCBench`）

## 1. 目的と方針

- **目的**: upstream（ORNL/HeCBench `master`）に、OpenACC 版 `-acc` を追加する PR を出す。
- **方針**: acc ブランチと master の差分は「`-acc` ディレクトリと、それに最低限必要なもの」だけにする。
- main ブランチは近く削除するので、比較の対象にも作業の対象にもしない。

決定済みのこと:

| 項目 | 決定 | コミット・タグ |
|---|---|---|
| `src2/`, `status2/` | acc から削除。タグで保存 | `82329bd55`, タグ `archive/src2-final-20261109`（タグは `3e0a8367a` を指す） |
| `-omp_nvc` | acc から削除（`-omp` のコピーなので不要） | `9fac85448`（削除前の状態は `4359d4730`） |
| ans-acc, leukocyte-acc, shmembench-acc | 「OpenACC 版は Not yet implemented」として削除 | `73c766130` |
| `src/include/*-NVIDIA` | **すべて残す**。`-acc` がないベンチマーク向けの約170件も、将来 `-acc` を追加するかもしれないので残す | — |
| `src/include/SDKBitMap.h`, `src/hausdorff-cuda/reference.h` | `#if defined(_OPENACC)` の追加だけ。`-acc` に必要なので残す | PR #3 |

## 2. これまでの作業（PR #3 マージ後、`f1902eb1a` 以降）

```
b496229af [diamond-acc] map each sequence to one thread and use the upstream input
c2d5f1558 [lr-acc] read the assets from lr-sycl like the OpenMP version
80d76967b [sad-acc] extract data.tar.gz at build time like the other ports
19c512f8b [diamond-acc] fix the data clauses and the host CPU loop pragma
6df834e2d [divergence-acc] read input.txt from divergence-cuda
1ffa6c53e [hausdorff-acc] keep the point coordinates in registers
3b14849e1 [pathfinder-acc] launch the pyramid kernels asynchronously
ae3af0359 Add ACC_WORK_STATUS.md (temporary hand-off notes; remove before the upstream PR)
73c766130 Remove ans-acc, leukocyte-acc and shmembench-acc (not yet implemented)
9fac85448 Remove the -omp_nvc ports
4359d4730 [pathfinder-acc] fix the launch geometry and verify results against a host checker
c2df6b098 [stencil1d-acc] keep warmup launches out of the timed loop
8fc7afa2e [gamma-correction-acc] reuse the image utilities; validate the user-input values; ...
cb5cea630 [sw4ck-acc] exit cleanly when the input is missing and run every case in the file
acd6a1d76 [bitonic-sort-acc] reduce kernel launches and verify results against std::sort
0b50eecba [hausdorff-acc] fail verification when the distance reset is set only once
afe12889b [lr-acc] validate the parallelized regression against the iterative one
b0c4179cb [sad-acc] verify the SAD array against a host reference
f39575303 [dct8x8-acc] include <stddef.h> for size_t
aa8acef40 [ccs-acc] index s_vect writes with MAXSAMPLE to match the reads
f7d7d449b [cfd-acc] correct the block-length warning
91a550fa1 [sad-acc] skip partial windows and index sad_array by window position
6994c7246 [tqs-acc] fix out-of-bounds task queue access
ddd4f1447 [divergence-acc] drop unused vectorization pragma macros
d8a8840c8 [fpdc-acc] address uninitialised variable warning
3d2b480a9 [ising-acc] prefer strrchr() to rindex()
9ae8ebb2a [mis-acc] validate an input graph with more constraints
23cea92b7 [gc-acc] validate undirected graph inputs
2ac3450a7 [bn-acc] initialize LG[0] and parent like the other ports
8660fae8d [b+tree-acc] keep print_leaves walking when verbose output is off
7f9643574 [su3-acc] align site on the struct type
9f904ae79 [hexciton-acc] index ham_imag with both arguments
1f9386ba7 [wsm5] initialize fallc_k and fallc_kp1 to zero in -acc and -omp_nvc
e8a3e305a Merge origin/master (upstream sync) into acc   ← master 8f44ff3e6 をマージ。diamond の unique_ptr 修正を含む
82329bd55 Remove src2/ and status2/ (archived at tag archive/src2-final-20261109)
3e0a8367a Use gang vector in black-scholes-acc and clenergy-acc single-gang loops
```

- `3e0a8367a`: AMB-work ブランチの `-acc_gv` から取り込んだもの。gv のほかの変更は、PR #3 の内容のほうが適切なので取り込まない（比較済み）。
- `e8a3e305a`: master（`8f44ff3e6`）をマージ。git のコンフリクトはなし。master が `diamond-sycl/src/basic/masking.h` を `unique_ptr` に変えたため、diamond-acc の `masking.cpp` も合わせて修正した。
- `9f904ae79`〜`4359d4730`（22コミット）: master が `-omp` などで直したバグや追加した検証を、`-acc` に移植したもの。各コミットのメッセージに、元になった upstream のコミットを書いてある。
  - pathfinder, bitonic-sort, gamma-correction, hausdorff, stencil1d, sw4ck は、master の新しい `-omp` 版をもとに ACC で書き直した。共有メモリの処理は、gang ごとの `private` 配列と、続けて並べた `loop vector` で表している（PR #3 の fwt-acc と同じ書き方）。
- `3b14849e1`〜`b496229af`（7コミット）: GPU での確認（3章）で見つかった問題の修正。

## 3. GPU での確認結果（2026-11-10, GB200 / cc100, nvc++ 26.3）

2章の変更を GPU で確認した。**ビルドできるものは、すべて正しい結果になることを確認済み。**

手順:
```bash
module load nvhpc
cd src/<name>-acc
make -f Makefile.NVD clean
make -f Makefile.NVD SM=cc100 run     # GB200 は cc100
cat log.std log.err log.time          # 実行コマンドは src/include/<name>-NVIDIA の COMMAND
```
比較には `-cuda`（`make ARCH=sm_100`）と `-omp`（`make -f Makefile.nvc SM=cc100`）を、同じ引数で実行した。

### 結果（カーネル時間）

| ベンチマーク | acc | cuda | omp | 判定 |
|---|---|---|---|---|
| pathfinder | 0.042s（修正前 0.159s） | 0.042s | 0.418s | PASS |
| bitonic-sort | 20.8ms | 25.5ms | 203ms | PASS |
| gamma-correction | 0.56ms | 0.46ms | 0.52ms | PASS |
| hausdorff | 8.8ms（修正前 27.2ms） | 7.0ms | 8.9ms | PASS |
| stencil1d | 1.66ms | 2.09ms | 9.21ms | PASS |
| sw4ck | 1.47 / 6.53ms | 1.21 / 5.40ms | 15.5 / 284ms | 誤差は omp と同じ程度 |
| black-scholes | 0.76ms | 0.70ms | 2.85ms | 出力の合計が omp と一致。gang の修正の効果を確認 |
| clenergy | 2.42ms/回 | 1.59ms/回 | 2.57ms/回 | 出力が omp・cuda と一致 |
| wsm5 | 290ms | 303ms | 310ms | チェックサムが omp と一致 |
| divergence | — | — | — | 修正後、omp と同じ値 |
| ccs | 10.9ms | — | 15.6ms | `Output.txt` が omp と完全に一致 |
| b+tree | 113µs | — | 1201µs | `output.txt` が omp と完全に一致 |
| cfd | 0.21s | — | 0.55s | 出力3ファイルが omp と完全に一致 |
| diamond | 2.60s | — | 2.73s | 検証エラー0件。`long_cur.m8` が omp と完全に一致 |
| sad, lr, su3, ising | — | — | — | PASS / SUCCESS |
| mis | — | — | — | 検証エラーなし（時間の表示は下記の注意を参照） |
| hexciton, bn, gc, tqs, fpdc, dct8x8 | — | — | — | 予想どおりビルドエラー（OpenMP の名残。4章 A） |

- 実行時間（real）が長いもの（gamma-correction 約110秒、hausdorff 約60秒、pathfinder 約30秒）は、ホスト側で逐次に検証しているため。GPU の問題ではない。

### この確認で直したもの（7コミット）
- **pathfinder**：10000回のカーネル起動のたびにホストと同期していた。`async(1)` を付け、最後に `wait(1)` を入れた。
- **hausdorff**：`float2` の点がローカルメモリに置かれていた。スカラーに分けて計算するようにした。
- **divergence**：`divergence-acc` に `input.txt` がなく、CPU・GPU ともに NaN になっていた。`../divergence-cuda/input.txt` を読むようにした。
- **diamond**：
  - `masking.cpp` の誤った2つの pragma を直した。データ句は `copy`/`copyin` にし、ホスト側の CPU 基準ループは omp 版と同じ `omp parallel for` に戻した。
  - マスク処理のループが gang だけになっていて、作業配列が共有メモリに置かれて競合し、約2250件の検証エラーが出ていた。`gang vector` に直した。
  - 入力を、外部の gerbil データから、omp と同じ DVC のデータ（`diamond-sycl/long.fastq.gz`）に変えた。
- **sad**：upstream `2c09431bc` を移植した。リポジトリに入っていた `bitmap_image.hpp` のコピーを削除し、ビルド時に `sad-cuda/data.tar.gz` を展開するようにした。それまでは、存在しないファイルに依存していたため、make の組み込みルールで CFLAGS なしにビルドされて失敗していた。
- **lr**：アセットを、omp と同じ `../lr-sycl/assets/` から読むようにした。

### 入力データ
- dvc はこのマシンに入っていない。確認には、scratchpad の venv に `pip install 'dvc[s3]'` で入れたものを使った。
- 取得したもの: `src/sad-cuda/data.tar.gz`, `src/ccs-cuda/data.tar.gz`, `src/lr-sycl/assets.tar.gz`, `src/data/b+tree/b+tree.tar.bz`, `src/data/cfd/cfd.tar.bz`, `src/diamond-sycl/data.tar.bz`（658MB）。
- 展開は、CMake の `UNZIPFILE` と同じく、アーカイブがあるディレクトリで `tar x` した。sad だけは Makefile が自動で展開する。

### 注意
- **mis**：`compute time: 0.000000 s` と表示される。ループ回数は100で固定なのに、時間を `repeat`（100000）で割っているため。upstream の mis-omp にも同じ不具合があり、acc 版はそれに合わせた状態（直していない）。
- **diamond-acc と diamond-omp** は、同じ `../diamond-sycl/src/*.o` にオブジェクトを出力する。並べてビルドすると互いに壊れるので、順番にクリーンビルドすること。
- `run_mem` ターゲットは `../scripts/m.sh` を呼ぶが、このファイルは `src/scripts/` にない（使わなければ問題ない）。

## 4. 未解決の課題：`-acc` に残る OpenMP の名残

`-acc` 322件を調べた結果、Makefile.NVD がビルドするコードに OpenMP の名残があるものが **96件** あった（コメント、`#if 0` の中、ビルドされないファイルは除外済み）。
- 名残の種類: CUDA 風のカーネル（`#pragma acc parallel teams num_teams() thread_limit()` の中に `#pragma omp parallel`、`omp_get_thread_num()`、`omp_get_team_num()`、`#pragma omp barrier`）、`#pragma acc data to:/from:` や `map(...)` のような OpenMP の書き方のデータ句、`#pragma omp declare target`、置換に失敗した存在しない関数（`omp_get_vector()`、`omp_get_gang()`）など。
- main ブランチ時代の status2（`archive/src2-final-20261109:status2/results_memory.md`）で ACC 版が build err/exe err だった82件のうち、78件がこの96件に含まれる。残りの4件は ans, leukocyte, shmembench（削除済み）と heartwall（名残以外が原因。未調査）。
- status2 で「正常」だったのに名残があるものは、`-mp=gpu` 付きでビルドしていたので OpenMP として通っていたと見られる。PR #3 などで `-acc=gpu` に変えると、通らなくなる可能性がある。

### A. カーネルの書き直しが必要（77件・263カーネル）

gang のループと `loop vector` で組み直し、`barrier` で区切られた部分は別々の vector ループに分ける必要がある。括弧内はカーネル数（`acc parallel teams` の数）。

adv(1) aes(2) aop(5) assert(2) asta(1) axhelm(2) binomial(1) bn(1) cm(3) cmp(2) convolutionSeparable(2) crs(20) d2q9-bgk(1) dct8x8(2) ddbp(8) deredundancy(1) dxtc2(1) expdist(1) fdtd3d(1) fft(2) filter(1) fpc(2) fpdc(1) frna(10) gc(2) gmm(10) grep(1) hexciton(9) histogram(0) hybridsort(6) lanczos(3) langford(1) lavaMD(1) lda(1) ldpc(4) lid-driven-cavity(11) match(10) merge(2) metropolis(4) morphology(2) nms(2) ntt(1) nw(2) openmp(0) p4(1) particlefilter(3) particles(9) permutate(0) pns(2) prna(8) qtclustering(4) quicksort(2) radixsort(6) randomAccess(1) reaction(6) recursiveGaussian(3) rng-wallace(1) s3d(27) scan(2) scan2(3) slu(6) sobol(1) softmax-online(2) sort(3) split(1) sptrsv(1) srad(1) tqs(1) tridiagonal(8) tsa(1) tsp(1) vmc(5) wedford(1) winograd(0) wlcpow(1) xlqc(4) zmddft(3)

- カーネル数 0 の histogram, openmp, permutate, winograd は、`teams` はないが `#pragma omp parallel` や `omp_get_*()` などが残っているもの。
- status2 で正常だったもの: deredundancy, openmp, winograd。記載がないもの: binomial。ほかはすべて build err。

### B. pragma の置き換えで直る（12件）

`#pragma omp declare target` → `#pragma acc routine seq`、`#pragma acc target update` → `#pragma acc update`、データ句を `copyin`/`copyout`/`copy` に直す、`omp atomic` → `acc atomic` など。

| ベンチマーク | status2 | 名残 |
|---|---|---|
| blas-gemm | 記載なし | データ句1、`omp dispatch` 3（MKL 呼び出し。置き換えだけでは済まない可能性あり） |
| chacha20 | 正常 | declare target 2 |
| complex | 正常 | declare target 4 |
| debayer | 正常 | declare target 6 |
| diamond | build err | **修正済み**（`19c512f8b`, `b496229af`） |
| ecdh | 正常 | declare target 2 |
| kalman | build err | データ句1、declare target 2 |
| lci | 正常 | declare target 2 |
| linearprobing | build err | データ句1、declare target 4 |
| lulesh | 正常 | `#pragma acc target update host(...)` 18 |
| multimaterial | 正常 | `omp_get_wtime` 6、そのほかの omp 2 |
| pnpoly | build err | データ句1、declare target 2 |

### C. ホスト側だけの名残（7件）

`#include <omp.h>`、`omp_get_wtime()`、`omp_get_num_procs()` など。chrono などへの置き換えで済む。

backprop（`omp_get_num_procs` など）, mr（データ句1。build err）, myocyte, rsbench, rsc, simplemoc, su3（いずれも `omp.h` などで、status2 は正常）

### `-mp=gpu` について（GPU マシンで確認）
Makefile.NVD の約250件が `-mp=gpu` を指定している（`-acc=gpu` は約60件）。多くは `CC = nvc++ -acc` も指定しているので、OpenACC と OpenMP のオフロードが両方とも有効になり、OpenMP の名残があってもビルドが通ってしまう。名残をなくした後に、`-acc=gpu` にそろえるかを決める必要がある。

### 決める必要があること
1. B と C の18件（diamond は修正済み）を直すか（比較的短時間で済む。GPU で確認しながら進められる）。
2. A の77件の扱い:
   - (a) ans などと同じく「Not yet implemented」として PR から外す（ディレクトリを削除）
   - (b) 書き直す（263カーネル分。1カーネルのものから順に、など）
   - (c) 両方：まず (a) で PR を出し、書き直しは後の別 PR にする

## 5. そのほかの未解決の課題

- **Makefile.NVD がない `-acc`**: binomial-acc, blas-gemm-acc, miniFE-acc
- **CMake**: `-acc` の CMakeLists.txt（324件）は、すべて `-omp` のコピー（`MODEL omp`、先頭のコメントも `# <name>-omp/CMakeLists.txt`）。
  - upstream の CMake は OpenACC を扱えない: `src/CMakeLists.txt:594` の `foreach(MODEL cuda hip sycl omp)` に acc がなく、`cmake/modules/BenchmarkMacros.cmake` の MODEL も cuda/hip/sycl/omp だけ。
  - そのため今は読み込まれていないが、594行目に acc を足すだけだと、ターゲット名 `<name>-omp` が `-omp` 版と重複してエラーになる。
  - 選択肢: (a) CMake には対応しない（CMakeLists.txt を削除し、Makefile だけにする）、(b) マクロに acc モデル（`HECBENCH_ENABLE_OPENACC`、nvc++ の `-acc` など）を足し、324件を `MODEL acc` に直す。
- **通常の Makefile**: `-acc/Makefile` はほぼすべて `CC = icpx` で、`-fopenacc -fopenacc-targets=spir64` を指定している（`-omp` の Makefile を機械的に置き換えたものと見られる）。実際に使えるのは Makefile.NVD（nvc++）だけ。upstream に合わせて `Makefile` をどうするか（Makefile.NVD を Makefile に統合するかなど）を決める必要がある。
- **heartwall-acc**: status2 では build err だが、OpenMP の名残はない。原因は未調査。
- **`prepare_data.sh`**: `src/include/*-NVIDIA` のうち55件が、COMMAND の先頭で `../<name>-cuda/prepare_data.sh` を呼んでいるが、このスクリプトは src2 の時代のもので、src にはない（sad と lr は直した）。`;` で区切られているので実行は止まらないが、データは展開されない。upstream の CMake `UNZIPFILE` や Makefile のやり方に合わせて直す必要がある。
  - ほかにも lr-acc のように、`prepare_data.sh` がコピーしたファイルを、自分のディレクトリから読もうとするものがあるかもしれない。
- **dct8x8-acc**: ACC のカーネルの中で `omp_get_thread_num()` を使っている（A に含まれる）。

## 6. 参考

- 保存してあるもの: `src2/`、`status2/` → タグ `archive/src2-final-20261109`。`-omp_nvc` → `4359d4730`。削除した3件 → `9fac85448`。
  - 取り出し方の例: `git worktree add ../hecbench-src2 archive/src2-final-20261109`
- 関連ブランチ: `AMB-work`（`src3/` に `-acc_gv` 版がある。上の比較で、取り込むものは black-scholes と clenergy だけと判断済み）。
- 3章の確認には、scratchpad に書いた一時的なスクリプトを使った（ベンチマークごとに clean、build、run を行い、`log.*` を集めるもの）。リポジトリには入っていない。
- 4章の分類は、Mac 上のスクリプトで出したもの（コメントを除き、`unifdef -k -D_OPENACC -U_OPENMP` で使われない分岐を取り除き、Makefile.NVD がビルドするソースとそこから `#include` されるファイルだけを対象にした）。スクリプトはリポジトリに入っていない。必要なら作り直す。
