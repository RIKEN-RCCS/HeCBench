# acc ブランチ 作業状況メモ

> **注意: このファイルは作業引き継ぎ用の一時ファイルです。upstream に PR を出す前に必ず削除してください。**

- 最終更新: 2026-11-26（prepare_data.sh の29件の対応を終えた時点）
- 対象ブランチ: `acc`（`origin` = `github.com:RIKEN-RCCS/HeCBench`）

## 0. 作業中（セッションは2時間で強制終了されるので、ここを随時更新してコミットする）

- **現在の状態（2026-11-26）**: `prepare_data.sh` を呼んでいた29件の対応を終えた（29件とも `make run` が最後まで通ることを確認し、ベンチマークごとにコミット済み。結果は4章「`prepare_data.sh` の29件」）。コミットされていない変更はない。
  - **コミットの作者を直した（2026-11-26）**: GPU マシンの git に `user.name` / `user.email` がなく、このマシンでのコミット100件が `rku00196 <rku00196@cXXX.rikyu.r-ccs.riken.jp>` で記録されていた。ユーザーが `git config --global` で `Hitoshi Murai <h-murai@riken.jp>` を設定し（ホームディレクトリは共通なので、どの GPU マシンでも使われるはず）、`ae3af0359` より後の134件を `git filter-branch` で書き換えた（作者の日時は元のまま、ツリーは不変、マージは範囲内になし）。ハッシュが変わったので、コミットのメッセージの中の参照（3か所）とこのファイルの中の参照も、新しいハッシュに直した。
    - 2026-11-26 にユーザーが `git push --force-with-lease origin acc` で force push した（`b74e6bb7d`）。`origin/acc` に自動設定の作者は残っていない。
    - 書き換え前の状態は、ローカルのブランチ `backup/acc-before-reauthor`、`backup/acc-before-reauthor2`、`backup/acc-before-msgfix` と `refs/original/` に残してある。force push の後、問題がなければ消してよい。
- **前の状態（2026-11-26）**: 区切りのよいところ。コミットされていない変更はない。残る `-acc` は247件。push はユーザーが行う（このマシンには GitHub の認証情報がない）。
  - 直前に終えたこと: ビルドできなかった9件の対応（4章「ビルドできなかった9件の対応」）、その前は31件への `default(present)` の追加（4章）。
- **次の候補**（どれから進めるかは未定）:
  1. 残した deredundancy, openmp, winograd の名残の確認（4章 A）
  2. （完了）`prepare_data.sh` の29件
  3. もともとの不具合: eigenvalue（`-O2` 以上で止まる）、boxfilter（FAIL）、lulesh（結果が誤り）、kalman（upstream の判定）
  4. gang だけのループの候補33件の性能、blas-gemm の書き直し
- **最終テストは nvhpc 26.11 で行う**（`module load nvhpc/26.11`）。それまでの作業は 26.3（`module load nvhpc` の既定）。26.11 では、全247件のビルドと実行に加えて、26.3 の不具合の回避（pnpoly `74cfca2e3`、linearprobing `d0872d728`、`default(present)` の31件）が必要か、`-acc=gpu` だけのビルドが遅くなる問題（car, channelShuffle）が起きるかを確かめ直す。
- **確認の手順**（scratchpad は日をまたぐと消える）:
  - **セッションの最初に GPU の数を確かめる**（`nvidia-smi --query-gpu=index,name,compute_cap --format=csv`）。数はセッションごとに変わる（4基、1基、2基を経験）。存在しない GPU を `CUDA_VISIBLE_DEVICES` で指定すると、GPU なしで動いて結果が誤る。
  - `module load nvhpc` の後、`cd src/<name>-acc && make -f Makefile.NVD clean && make -f Makefile.NVD SM=cc100 run`（結果は `log.std` / `log.err` / `log.time`）。`module load` をパイプにつなぐと反映されない。
  - OpenMP なしの確認は、`-mp=gpu` を `-acc=gpu` に置き換えた一時的な Makefile で行う。比較は `-omp`（`make -f Makefile.nvc SM=cc100`）や `-cuda`（`make ARCH=sm_100`）を同じ引数で実行する。
  - ビルドのログで `-acc` 付きでコンパイルされたかを確認する（依存先がないと、組み込みルールでフラグなしにビルドされる。4章 B）。
  - 変更の後は、クリーンビルドと実行をしてからコミットする。5分以上かかる実行はバックグラウンドで行う。

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
| 最終的なテストのコンパイラ | **nvhpc の最新版 26.11 で行う**（2026-11-25 に決定）。それまでの作業と確認は 26.3 で行っている | — |
| heartwall-acc | 「Not yet implemented」として PR から外す（2026-11-26 に決定）。`main.cpp` に OpenACC の指示文がなく、`kernel/`, `util/` が一度も追加されていない | `1127101fe`（削除前の状態は `13dbbb84c`） |
| 4章 A のうちビルドできない73件と binomial（計74件） | 「Not yet implemented」として PR から外す（2026-11-19 に決定）。書き直しは後の別 PR で行う | `27d343993`（削除前の状態は `9e1d5df4f`） |
| `src/include/*-NVIDIA` | **すべて残す**。`-acc` がないベンチマーク向けの約170件も、将来 `-acc` を追加するかもしれないので残す | — |
| `src/include/SDKBitMap.h`, `src/hausdorff-cuda/reference.h` | `#if defined(_OPENACC)` の追加だけ。`-acc` に必要なので残す | PR #3 |

## 2. これまでの作業（PR #3 マージ後、`f1902eb1a` 以降）

```
74cfca2e3 [pnpoly-acc] port the data clauses and work around two nvc++ issues
d0872d728 [linearprobing-acc] port the data clauses, kernels and atomic compare to OpenACC
fee77cfb4 [kalman-acc] use OpenACC data clauses and map each series to one thread
461e892a1 [lci-acc] build with the OpenACC flags and fix the gang clause
8337638ae [ecdh-acc] mark the device functions with acc routine seq
9f64bc3dd [debayer-acc] mark the device functions with acc routine seq
60c3f042b [complex-acc] mark the device functions with acc routine seq
25141f5d3 [chacha20-acc] mark the device functions with acc routine seq
36717dadd Update ACC_WORK_STATUS.md after fixing the host-only OpenMP remnants (temporary; remove before the upstream PR)
613079143 [su3-acc] drop the unused omp.h include
b4adbd3d0 [rsc-acc] drop the unused omp.h includes
f942cf05d [rsbench-acc] time with clock_gettime instead of omp_get_wtime
1520707d9 [myocyte-acc] drop the unused omp.h include
1b9967d3c [mr-acc] use OpenACC data clauses and fix the atomic updates
40a2fc25a [backprop-acc] restore the host OpenMP pragmas in the #ifdef OPEN blocks
e7b77358c Update ACC_WORK_STATUS.md with the GPU check results (temporary; remove before the upstream PR)
c3d7ba4c4 [diamond-acc] map each sequence to one thread and use the upstream input
0535bfd62 [lr-acc] read the assets from lr-sycl like the OpenMP version
e9395c364 [sad-acc] extract data.tar.gz at build time like the other ports
c30419011 [diamond-acc] fix the data clauses and the host CPU loop pragma
e9e40fca5 [divergence-acc] read input.txt from divergence-cuda
f97c6388d [hausdorff-acc] keep the point coordinates in registers
6d4a47984 [pathfinder-acc] launch the pyramid kernels asynchronously
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
- `6d4a47984`〜`c3d7ba4c4`（7コミット）: GPU での確認（3章）で見つかった問題の修正。
- `40a2fc25a`〜`613079143`（6コミット）: 4章 C の修正（2026-11-17）。
- `25141f5d3`〜`74cfca2e3`（8コミット）: 4章 B の修正（2026-11-19）。

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

### 全件ビルドの結果（2026-11-19）

全322件を、OpenMP なし（`-mp=*` を `-acc=*` に置き換えた一時的な Makefile）でビルドした（実行はしない）。失敗したものは、通常の Makefile.NVD でも試した。

| 結果 | 件数 |
|---|---|
| OpenMP なしで通る | 205 |
| `-mp=gpu` があるときだけ通る | 34 |
| どちらでも通らない | 80 |
| Makefile.NVD がない（binomial, blas-gemm, miniFE） | 3 |

- **A の77件**: 73件がどちらでも通らない。通るのは deredundancy, openmp, winograd の3件だけで、残りの1件は Makefile.NVD のない binomial。最初の分類は、A についてはおおむね正しかった。
- **`-mp=gpu` があるときだけ通る34件**:
  - 31件が "Could not find allocated-variable index for symbol"。関数の引数のポインタを、データ句なしでカーネルの中で使っている（呼び出し元の `acc data` にある配列）。OpenACC としては正しいコードで、nvc++ 26.3 が OpenMP を無効にしたときだけ失敗する（`-std` や `-Minfo` の有無には関係しない）。`default(present)` で回避できる（kalman, linearprobing で確認済み）。対象: adamw atomicPerf boxfilter bwt car channelShuffle eigenvalue face flip ga glu grrt heat2d hogbom hypterm ising iso2dfd keccaktreehash lif lsqt mask md memtest permute perplexity resize scatterAdd surfel tonemapping vol2col wsm5
  - haversine: `../haversine-cuda/distance.h` が `sycl/sycl.hpp` を include している。
  - rainflow: `../rainflow-cuda/reference.h` の `double3` が未定義。
  - kmeans: `rmse.c:79` で "Cannot determine bounds for array"。
  - haversine と rainflow は、cuda 版の共有ヘッダーの `_OPENMP` の分岐に頼っていると見られる（PR #3 の SDKBitMap.h のように `_OPENACC` の分岐が必要かもしれない）。
- **A 以外でどちらでも通らないもの**: upstream のファイル構成の変更に、acc 版が追従していないものと見られる。
  - lud: `stopwatch` が未定義（`-I../lud-cuda/common`）
  - gd: `../gd-cuda/utils.h` で宣言されている `get_time()` の定義をビルドしていない
  - gpp: `gettimeofday` が未定義
  - goulash: `secs_elapsed` が未定義
  - heartwall: omp 版にある `kernel/` と `util/` がない
  - extend2: 依存先の `filelist.txt` がない
- **組み込みルールへの切り替わり**: `-acc` なしでコンパイルされた GPU のコードは、ない（lci は直した。b+tree の timer.c はホスト側のコード）。
- **gang だけのループの候補**: ソースで gang/vector/worker/seq のどれも指定していない `acc parallel loop` が、202件・493か所ある。そのうち、`-Minfo` で `loop gang /* blockIdx.x */`（gang だけ）と報告されたものが33件:
  - accuracy adamw adjacent aidw amgmk attention bonds bspline-vgh channelSum chi2 dslash face fhd gd haccmk kmeans layout libor lsqt mcmd mdh meanshift multimaterial murmurhash3 nbody page-rank rfs s8n secp256k1 simpleSpmv snake softmax xsbench
  - 同じ領域で、共有メモリに配列が置かれている強い候補は5件。adamw と chi2 は PASS、libor は期待値どおり。meanshift と snake は入力データがなく、確認できなかった。
  - 正しさに問題がなくても、gang の中で1スレッドしか働かないので、性能は落ちている可能性が高い（kalman では gang vector にして 24倍速くなった）。
- **全件ビルドで見つけて直したもの**:
  - grrt, iso2dfd, tridiagonal の `make clean` が、`*.txt` の指定で追跡中の `CMakeLists.txt` を消していた。cuda・omp 版と同じく、ファイル名を個別に書くようにした（`d9fe5ac4b`, `35d096f2f`, `6b26486d0`）。
  - **私のコミット `f97c6388d` で hausdorff のビルドが壊れていた**。`hd()` を使われていないと判断して消したが、`../hausdorff-cuda/reference.h` が使っていた。消した後にビルドし直していなかった。`1dafdedf9` で戻した（8.8 ms、PASS）。教訓: 変更の後は、必ずクリーンビルドと実行をしてからコミットする。

### `prepare_data.sh` の29件（2026-11-26）

`include/*-NVIDIA` の実行コマンドの多くが、src2 の時代の `prepare_data.sh`（今はどこにもない）を呼んでいた。PR に残す `-acc` に関係する29件を直した（ほかの25件は、`-acc` を外したものか、もともと `-acc` がないもの）。

- **方針**: acc 版も最終的には CMake でビルドする。upstream の CMake は、設定のときに各版の CMakeLists.txt の `UNZIPFILE` でデータを展開する（有効にしたモデルに関係なく、全版の CMakeLists.txt を読み込む）。そこで、Makefile.NVD 用の展開の仕組みは作らず、`prepare_data.sh` の呼び出しを外して、入力のパスを upstream の `-omp` 版（＝CMake の展開先）に合わせた。Makefile.NVD を使う間は、データの展開を使う人に任せる（PR の説明に書く）。
- **パスも直したもの**: face（`../face-cuda/`）、lsqt（`../lsqt-cuda/examples/`）、medianfilter（`../medianfilter-sycl/data/`）、mriQ（`../mriQ-cuda/datasets/`）、geodesic（upstream `8060b37bc` を移植し、入力を引数で受け取る）、mcmd（`../mcmd-cuda/dataset` で実行。mcmd-acc にコミットされていた `dataset/`（実行の出力まで含む）を削除）。gd は `../gd-cuda/gisette_scale` のまま（DVC になく、README が gd-cuda で案内している）。
- **確認**: 24個のアーカイブを DVC で取得し、`UNZIPFILE` と同じ場所に展開して（scratchpad の `pd/prep_data.sh`）、29件とも `make run` が最後まで通ることを確かめた。PASS/FAIL を出すものはすべて PASS。geodesic と mcmd は出力が omp と一致（geodesic の「最大誤差 31.0」は omp も同じ）。
- **upstream の CMake でも用意されないデータ**（`UNZIPFILE` がない）: columnarSolver-cuda/data.tar.gz, deredundancy-sycl/testData.tar.gz, henry-cuda/data.tar.gz, snake-cuda/Datasets.tar.gz, tonemapping-sycl/input.tar.gz。ほかに multimaterial（実行するディレクトリの `volfrac.dat`。`multimaterial-cuda/volfrac.dat.tgz`）と gd（外部）。CMake に移っても手で用意する必要がある。
- 実行コマンドの引数の大きさ（mcpr, multimaterial, medianfilter など upstream と違うもの）は変えていない。

### ビルドできなかった9件の対応（2026-11-26）

全件ビルドで、どちらのビルドでも通らなかった6件と、`-mp=gpu` があるときだけ通った3件。8件を直し、heartwall は PR から外した。直した8件は、通常のビルドと OpenMP なしのビルドの両方で、ビルドと実行ができることを確認した。

| ベンチマーク | 原因 | 対応（コミット） | 確認 |
|---|---|---|---|
| lud, gd, gpp, goulash | upstream が時間計測を chrono に変えたとき、共有ヘッダー（`../X-cuda/`）から古い関数（`stopwatch`, `get_time()` など）を消したが、acc 版が追従していなかった | upstream の変更（`c6589a71c`, `5f366260f`, `a04e3a290`, `6c899be7f`）を移植（`6af18f876`, `36276c754`, `602c2d435`, `4047eb78f`） | lud は検証で不一致なし、gd と goulash は PASS、gpp は結果が omp と一致 |
| extend2, nn | `filelist.txt` がない | src2 から戻した（`16214140e`, `0757c4c5b`） | 出力が omp と一致 |
| haversine, rainflow | cuda 版の共有ヘッダーが `_OPENMP` のときだけ型を定義していた | PR #3 と同じく `_OPENACC` の分岐を加えた。haversine は入力も upstream と同じ `../geodesic-cuda/locations.txt` に直した（`7f9d6576a`, `e4da713a7`） | haversine は omp と一致、rainflow は PASS |
| kmeans | ホスト側の `omp parallel for` が、誤った OpenACC の指示文に置き換えられていた | upstream と同じ指示文に戻した（`29acd7bfa`） | `OMP_NUM_THREADS=1` で omp と一致 |
| heartwall | OpenACC 版が作られていない（指示文がなく、`kernel/`, `util/` が一度も追加されていない） | PR から外した（`1127101fe`） | — |

- **`.txt` の抜け落ち**: ルートの `.gitignore` に `*.txt` があるため（upstream も同じ。追跡中の `.txt` は `git add -f` で追加されている）、src2 から src へのコピー（`c578731c9`）で `.txt` が抜け落ちていた。src2 と比べて洗い出し、extend2 と nn の `filelist.txt`、**ライセンスの minibude-acc/LICENSE.txt, nbody-acc/License.txt, su3-acc/License.txt**（`2fcbf3b34`）を戻した。omp 版にあって acc 版にないライセンスは、ほかにない（miniFE-acc の README だけ）。
- **入力データ**: gd の gisette_scale は LIBSVM のサイトから、nn, kmeans, haversine は DVC（`data/nn`, `data/kmeans`, `geodesic-sycl/locations.tar.gz`）から取得した（リポジトリには置いていない）。
- **upstream 側の問題**: extend2 は omp 版も "Error:" の不一致を10704行表示する。kmeans は、`rms_err()` の `omp parallel for` に `sum_euclid` の reduction がなく、通常のビルドでは結果が実行ごとに変わる（omp 版も同じ）。kmeans の OpenMP なしのビルドは 92秒かかる（通常は 21秒）。

### `default(present)` の追加（31件）（2026-11-25〜26）

全件ビルドで「`-mp=gpu` があるときだけ通る」と分かった34件のうち、"Could not find allocated-variable index" の31件に `default(present)` を付けた。ベンチマークごとにコミットした（flip `cfd1f6052` 〜 eigenvalue `e5b5db9eb`）。

- **確認の方法**: OpenMP なし（`-acc=gpu` だけ）でビルドできることを確かめたうえで、通常の Makefile.NVD（`-mp=gpu`）で、変更前（HEAD の git worktree）と変更後を同じ引数で実行し、時間・速度以外の出力が同じことを確かめた。通常のビルドの `-Minfo` も比べ、ループの割り当てと `memcpy` への置き換えが変わらないことを確かめた。
- **`default(present)` だけでは済まなかったもの**: カーネルがグローバル変数の配列を使っていると、実行時に "data in PRESENT clause was not found" になる。adamw（量子化の表）と grrt（`K2_tab`）は、その配列を `copyin(...)` で明示して直した。
- **入力データ**: face, hogbom, lsqt, tonemapping は DVC のデータが必要だった（`prepare_data.sh` がないので、比較用の worktree の中に手で展開した）。置き場所: face は `face-cuda/dataset.tar.gz` を実行するディレクトリ（face-acc）に、hogbom は `hogbom-cuda/data.tar.bz` を hogbom-cuda に、tonemapping は `tonemapping-sycl/input.tar.gz` を tonemapping-sycl に、lsqt は `lsqt-cuda/examples.tar.gz` を lsqt-hip と lsqt-cuda の両方に展開する（acc 版の実行コマンドは `../lsqt-hip/examples/input.txt` を読み、その中で `../lsqt-cuda/examples/...` を参照している）。
- **もともとの不具合（変更前から同じ）**: boxfilter は FAIL。eigenvalue は `-O2` 以上でビルドすると止まる（`-O1` なら PASS。eigenvalue-omp は `-O3` で PASS）。
- **重要な発見（2026-11-25）: OpenMP なしのビルドの性能**: nvc++ 26.3 は `-mp=gpu` の有無でコード生成そのものが変わる。OpenMP なし（`-acc=gpu` だけ）のビルドは、`default(present)` でビルドできるようになっても、大幅に遅くなることがある。
  - car: 外側の collapse(4) のループが `gang, vector(256)` から `gang` だけになり、内側が `vector(256)` になった。カーネル 0.0047s → 0.94s（200倍）。RMSE も 0.100244 → 0.100481 と変わった（和の順番が変わったためと見られる）。
  - channelShuffle: ループの割り当ては同じだが、OpenMP なしのときだけ、カーネルの中のループが "Recognized memory copy idiom"（`memcpy` への置き換え）になり、実行が15分近く終わらない（通常のビルドでは各カーネル 0.02〜0.04 ms）。
  - そのため、**Makefile.NVD を `-acc=gpu` にそろえるのは、今の nvc++ では勧められない**。`default(present)` の追加は、通常のビルド（`-mp=gpu`）で結果が変わらないことを確認したうえでコミットする。

### A. カーネルの書き直しが必要（77件・263カーネル）：**74件を PR から外した**（2026-11-19, `27d343993`）

全件ビルドで、73件が `-mp=gpu` があってもビルドできないことを確認したので、binomial（Makefile.NVD なし）と合わせて74件を「Not yet implemented」として外した。OpenMP なしでビルドできた deredundancy, openmp, winograd の3件は残した（ただし名残がある）。以下は、外す前の記録。

gang のループと `loop vector` で組み直し、`barrier` で区切られた部分は別々の vector ループに分ける必要がある。括弧内はカーネル数（`acc parallel teams` の数）。

adv(1) aes(2) aop(5) assert(2) asta(1) axhelm(2) binomial(1) bn(1) cm(3) cmp(2) convolutionSeparable(2) crs(20) d2q9-bgk(1) dct8x8(2) ddbp(8) deredundancy(1) dxtc2(1) expdist(1) fdtd3d(1) fft(2) filter(1) fpc(2) fpdc(1) frna(10) gc(2) gmm(10) grep(1) hexciton(9) histogram(0) hybridsort(6) lanczos(3) langford(1) lavaMD(1) lda(1) ldpc(4) lid-driven-cavity(11) match(10) merge(2) metropolis(4) morphology(2) nms(2) ntt(1) nw(2) openmp(0) p4(1) particlefilter(3) particles(9) permutate(0) pns(2) prna(8) qtclustering(4) quicksort(2) radixsort(6) randomAccess(1) reaction(6) recursiveGaussian(3) rng-wallace(1) s3d(27) scan(2) scan2(3) slu(6) sobol(1) softmax-online(2) sort(3) split(1) sptrsv(1) srad(1) tqs(1) tridiagonal(8) tsa(1) tsp(1) vmc(5) wedford(1) winograd(0) wlcpow(1) xlqc(4) zmddft(3)

- カーネル数 0 の histogram, openmp, permutate, winograd は、`teams` はないが `#pragma omp parallel` や `omp_get_*()` などが残っているもの。
- status2 で正常だったもの: deredundancy, openmp, winograd。記載がないもの: binomial。ほかはすべて build err。

### B. pragma の置き換えで直る（12件）：**8件完了、残り2件**（2026-11-19）

数え直し（OpenMP なし、つまり `-mp=gpu` を `-acc=gpu` に置き換えた一時的な Makefile で、実際にビルドして実行した）の結果と対応:

| ベンチマーク | 実際の状況 | 対応（コミット） | 結果 |
|---|---|---|---|
| chacha20, complex, debayer, ecdh | `omp declare target` が残っていた（nvc++ は自動でデバイス向けにするので動いていた） | 関数ごとの `acc routine seq` に置き換えた（`25141f5d3`, `60c3f042b`, `9f64bc3dd`, `8337638ae`） | PASS（debayer はチェックサムが omp と一致） |
| lci | Makefile.NVD の依存先 `tables.h` が存在せず、組み込みルールで `-acc` なしにビルドされていた（**CPU で実行されていた**）。依存先を直すと `gang(1)` がエラーになった | 依存先を `../lci-cuda/tables.h` に直し（Makefile も）、`gang vector num_gangs(1) vector_length(96)` にした。routine も置き換えた（`461e892a1`） | `-DDUMP` の出力が CPU のビルドと一致 |
| kalman | OpenMP の書き方のデータ句で、ビルドできなかった | `copyin/create`、`update self` に直した。カーネルを `gang vector` にした（gang だけだと作業配列が共有メモリに置かれ、内側のループが vector に割り当てられていた。カーネル 1.9s → 0.08s）。OpenMP なしでもビルドできるよう `default(present)` を付けた（`fee77cfb4`） | **FAIL のまま**（下記） |
| linearprobing | データ句と `num_teams/thread_limit` が OpenMP の書き方。OpenACC にない `acc atomic compare capture` を使っていた | `copyin/create`、`update device/self`、`gang vector num_gangs/vector_length` に直した。compare-and-swap は C++20 の `std::atomic_ref::compare_exchange_strong` にした（`-std=c++20`）。`default(present)` も付けた（`d0872d728`） | PASS |
| pnpoly | OpenMP の書き方のデータ句で、ビルドできなかった | `copyin/copyout` に直し、カーネルを `gang vector` にし、範囲を明示した `present` を付けた。nvc++ の不具合を2つ回避した（下記）（`74cfca2e3`） | PASS。チェックサムが multicore のビルドとホストでの検算に一致 |
| diamond | — | **修正済み**（`c30419011`, `c3d7ba4c4`） | — |
| lulesh | 名残（18件の `acc target update host`）は、すべて `#ifdef VERIFY`（既定はオフ）の中で、使われない | 変更なし | **結果が誤っている**（下記。名残とは無関係） |
| multimaterial | `omp_get_wtime` は `#ifdef DEBUG` の中、`omp simd` は CPU で実行する基準の実装（upstream と同じ） | 変更なし | 2000×2000 で All tests passed（下記） |
| **blas-gemm（未着手）** | Makefile.NVD がない。MKL 向けの `omp dispatch` を使っている | 置き換えでは直らない。cuBLAS などで書き直す必要があり、A に近い | — |

kalman の FAIL について:
- 予測値が1億程度まで膨らむのに、upstream の reference（`kalman-cuda/reference.h`）は絶対誤差 1e-3 で判定している。kalman-cuda も8回中7回 FAIL する。acc 版の相対誤差は 1e-6 未満（n_diff=0 で 2.4e-8）で、`-gpu=nofma` にしてもほぼ変わらない。判定は upstream の共有コードなので変えていない。

この作業で分かった、ほかにも当てはまりそうなこと:
- **gang だけのループの競合**：外側のループに `gang vector` を書かないと、nvc++ は内側にループがあるときに gang だけを割り当てる。すると、反復ごとの作業配列が gang の共有メモリに置かれ、内側のループが vector に割り当てられて競合する（diamond, kalman, pnpoly で発生）。`-Minfo` で `CUDA shared memory used for ...` と `loop gang /* blockIdx.x */` が出ていたら疑う。ほかの `-acc` にも残っている可能性がある。
- **組み込みルールへの切り替わり**：パターンルールの依存先が存在しないと、make は組み込みルールで CFLAGS なしにビルドする（sad, lci で発生）。全322件を `make -n` で調べた結果、`-acc` なしでビルドされていた GPU のコードは lci だけ。extend2, heartwall, lavaMD, srad は依存先がなく、make が始まらない。
- **nvc++ 26.3 の不具合**（pnpoly で回避）：ループの増分の式の `k = j++` が変換に失敗する。`pnpoly_base` の `c = !c` が `-O2` 以上で誤ってコンパイルされる（`c ^= 1` で回避）。
- **compare-and-swap**：nvc++ の OpenACC では、`__atomic_compare_exchange_n` が逐次実行になり、結果も誤る。`std::atomic_ref` は使えるが、`compare_exchange_strong` の戻り値を使わないと、デバイス向けのコンストラクタが作られずエラーになる。
- **`-mp=gpu` なしで出るエラー**：関数の引数のポインタを、データ句なしでカーネルの中で使うと、`-acc=gpu` だけのとき "Could not find allocated-variable index" になる（kalman, linearprobing）。`default(present)` で直る。

### C. ホスト側だけの名残（7件）：**完了**（2026-11-17）

調べ直すと、実際の状況は次のとおりだった。7件とも、`-mp=gpu` を `-acc=gpu` に置き換えた一時的な Makefile（OpenMP なし）でビルドでき、実行して結果を確認した。

| ベンチマーク | 実際の状況 | 対応 | 結果 |
|---|---|---|---|
| backprop | `#ifdef OPEN`（未定義でビルドされない）の中で、`omp parallel for` が誤って `acc loop ... schedule(static)` に置き換えられていた。このブロックは cuda・omp 版にもある upstream 共通のホスト側コード | ほかの版と同じ `omp parallel for` に戻した（`40a2fc25a`） | PASS |
| mr | `acc parallel data map(...)`、`acc update to/from` が残っていた。また、`acc atomic update` の式の中で関数を呼んでいて、もともとビルドできなかった | `acc data copyin/create`、`update device/self` に直し、関数の結果をローカル変数に入れてから atomic で加えるようにした（`1b9967d3c`） | PASS（初めてビルドが通った） |
| rsbench | `rsbench.h` に `#define OPENMP` があり、`omp_get_wtime()` を実際に使っていた | `clock_gettime(CLOCK_MONOTONIC)` に置き換えた（ソースが .c なので chrono ではなくこれにした）。表示も「OpenACC Init」に直した（`f942cf05d`） | Valid |
| myocyte, rsc, su3 | 使っていない `#include <omp.h>` があった | 削除した（`1520707d9`, `b4adbd3d0`, `613079143`） | rsc：Test Passed、su3：SUCCESS、myocyte：5章を参照 |
| simplemoc | `OPENMP` が未定義なので、OpenMP のコードは使われていない。cuda 版と同じ書き方 | 変更なし | 動作を確認 |

- 当初の分類（Mac 上のスクリプト）は `_OPENMP` だけを未定義として扱っていたため、`OPEN` や `OPENMP` のような独自のマクロで囲まれた、ビルドされないコードも「名残」として数えていた。A と B の件数にも、同じように数えすぎているものがあるかもしれない。

### `-mp=gpu` について（GPU マシンで確認）
Makefile.NVD の約250件が `-mp=gpu` を指定している（`-acc=gpu` は約60件）。多くは `CC = nvc++ -acc` も指定しているので、OpenACC と OpenMP のオフロードが両方とも有効になり、OpenMP の名残があってもビルドが通ってしまう。名残をなくした後に、`-acc=gpu` にそろえるかを決める必要がある。

### 決める必要があること
1. B は8件を直した（2026-11-19）。残りは lulesh（結果が誤っている。名残とは無関係）と blas-gemm（書き直しが必要）。C の7件は完了。
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
- **myocyte-acc の出力が `nan`**: `output.txt` の約92万行のうち87万行が `nan`。`1520707d9` より前の状態（`omp.h` あり、`-mp=gpu`）でも同じなので、もともとの不具合。omp 版には `nan` がないが、時刻 0 の値（`y[0][0][0]=1.13e30`）が入力 `data/myocyte/y.txt` の先頭（1.37e-3）と合っていないので、upstream の実装も含めて調べる必要がある。実行コマンドは Makefile.NVD の50行目の COMMAND（`-time 10000`）で、`include/myocyte-NVIDIA` の COMMAND は上書きされている。
- **upstream の `myocyte-omp/Makefile.nvc` が壊れている**: リンクのルールの行の続け方が誤っていて、ビルドできない（比較のときは手でビルドした）。
- **simplemoc**: カーネル時間が約220秒、実行全体で約5分かかる（コードは変えていない）。
- **eigenvalue-acc は `-O2` 以上でビルドすると止まる**: 引数が `16 1` でも、"Executing kernel for 1 iterations" と表示した後で終わらない。`-O1` なら PASS（`2048 10000` で 662µs）。eigenvalue-omp は `-O3` で PASS。nvc++ 26.3 の最適化の問題の可能性があるが、GPU 向けのコードかホスト側かは未確認。upstream の eigenvalue-omp の Makefile.nvc も壊れている（存在しない `main2.o` を要求する）。
- **boxfilter-acc は FAIL**（`default(present)` の追加の前から同じ値）。
- **lulesh-acc の結果が誤っている**: Final Origin Energy が 2.453665e+10 なのに対し、lulesh-cuda と lulesh-omp はどちらも 1.850962e+10（`-i 5 -s 384 -r 11 -b 1 -c 1`）。`-mp=gpu` を加えても変わらないので、OpenMP の名残とは関係のない、もともとの不具合。
- **multimaterial の実行コマンドが大きすぎる**: `include/multimaterial-NVIDIA` の 6000×6000 では、omp 版も約66GB 使ったところで強制終了された（signal 9）。upstream の run ターゲットは 2000×2000。入力 `volfrac.dat` は `multimaterial-cuda/volfrac.dat.tgz` を実行するディレクトリに展開する必要がある（`prepare_data.sh` の問題の1件）。
- **upstream 側の問題（直していない）**:
  - kalman-omp は、系列数 100000 で何も出力せずに異常終了する（2500000 では動く）。
  - pnpoly-omp は PASS だが、チェックサム（1297432）がホストでの検算の結果（3938029）と合わない。基準と最適化版が同じように誤っている可能性がある。
  - lci-omp の Makefile.nvc にも、依存先 `tables.h` の同じ誤りがある。
- **`prepare_data.sh`**: `src/include/*-NVIDIA` のうち55件が、COMMAND の先頭で `../<name>-cuda/prepare_data.sh` を呼んでいるが、このスクリプトは src2 の時代のもので、src にはない（sad と lr は直した）。`;` で区切られているので実行は止まらないが、データは展開されない。upstream の CMake `UNZIPFILE` や Makefile のやり方に合わせて直す必要がある。
  - ほかにも lr-acc のように、`prepare_data.sh` がコピーしたファイルを、自分のディレクトリから読もうとするものがあるかもしれない。
- **dct8x8-acc**: ACC のカーネルの中で `omp_get_thread_num()` を使っている（A に含まれる）。

## 6. 参考

- 保存してあるもの: `src2/`、`status2/` → タグ `archive/src2-final-20261109`。`-omp_nvc` → `4359d4730`。削除した3件 → `9fac85448`。
  - 取り出し方の例: `git worktree add ../hecbench-src2 archive/src2-final-20261109`
- 関連ブランチ: `AMB-work`（`src3/` に `-acc_gv` 版がある。上の比較で、取り込むものは black-scholes と clenergy だけと判断済み）。
- 3章の確認には、scratchpad に書いた一時的なスクリプトを使った（ベンチマークごとに clean、build、run を行い、`log.*` を集めるもの）。リポジトリには入っていない。
- 4章の分類は、Mac 上のスクリプトで出したもの（コメントを除き、`unifdef -k -D_OPENACC -U_OPENMP` で使われない分岐を取り除き、Makefile.NVD がビルドするソースとそこから `#include` されるファイルだけを対象にした）。スクリプトはリポジトリに入っていない。必要なら作り直す。
