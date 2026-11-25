# acc ブランチ 作業状況メモ

> **注意: このファイルは作業引き継ぎ用の一時ファイルです。upstream に PR を出す前に必ず削除してください。**

- 最終更新: 2026-11-25（31件への default(present) の追加を、7件コミットしたところで中断した時点）
- 対象ブランチ: `acc`（`origin` = `github.com:RIKEN-RCCS/HeCBench`）

## 0. 作業中（セッションは2時間で強制終了されるので、ここを随時更新してコミットする）

- **現在の作業（2026-11-25 開始）**: `-mp=gpu` があるときだけ通る31件（4章「全件ビルドの結果」）に `default(present)` を付け、OpenMP なしでビルドでき、実行して正しい結果になることを1件ずつ確認する。ベンチマークごとにコミットする。
  - **中断（2026-11-25、ユーザーの指示）**: 比較を途中で止めた。**作業ツリーに、確認の済んでいない24件の変更（25ファイル）がコミットされずに残っている**: eigenvalue face ga glu grrt heat2d hogbom hypterm ising iso2dfd keccaktreehash lif lsqt mask md memtest permute perplexity resize scatterAdd surfel tonemapping vol2col wsm5。中身は、どれも指示文の行末への ` default(present)` の追加だけ（`git diff` で確かめられる）。OpenMP なしでビルドできることは確認済み。
    - 再開の手順: ①GPU の数を確かめる。②scratchpad が残っていれば `dp/pair.sh <name>` で1件ずつ比較する（変更前 `dp/wt`、変更後 `dp/wt2`）。scratchpad が消えていたら、`git worktree prune` で古い worktree の登録を消してから、HEAD の worktree（変更前）と、そこに `git diff -- src` を当てた worktree（変更後）を作り直し、通常の Makefile.NVD で両方を実行して、時間以外の出力を比べる。③出力が同じものから、ベンチマークごとにコミットする。
    - 比較中だった eigenvalue は、実行の途中で止めたので、最初からやり直す必要がある（変更前と変更後を合わせて7分以上かかる）。
  - 進捗（2026-11-25、更新）: コミット済み 7件（flip `3060bfaf7`, car `9f1adfdc9`, adamw `b7181eee0`, atomicPerf `3ac44b56d`, boxfilter `a58335353`, bwt `60947f93a`, channelShuffle `7a9d5c2b4`）。どれも、通常のビルドで変更前と出力が同じことを確認した。残り24件は `dp/pair.sh` で比較中（結果は `dp/pair28.txt`）。
  - 以前の進捗: flip は完了（`3060bfaf7`）。残り30件は `default(present)` を付けて OpenMP なしでビルドできることを確認済み（57か所、すべて行末に ` default(present)` を足しただけ。**作業ツリーにあり、まだコミットしていない**）。次は、30件を通常の Makefile.NVD と OpenMP なしの両方で実行して確認し、問題のないものからコミットする。
  - **重要な発見（2026-11-25）**: nvc++ 26.3 は `-mp=gpu` の有無でコード生成そのものが変わる。OpenMP なし（`-acc=gpu` だけ）のビルドは、`default(present)` でビルドできるようになっても、大幅に遅くなることがある。
    - car: 外側の collapse(4) のループが `gang, vector(256)` から `gang` だけになり、内側が `vector(256)` になった。カーネル 0.0047s → 0.94s（200倍）。RMSE も 0.100244 → 0.100481 と変わった（和の順番が変わったためと見られる）。
    - channelShuffle: ループの割り当ては同じだが、OpenMP なしのときだけ、カーネルの中のループが "Recognized memory copy idiom"（`memcpy` への置き換え）になり、実行が15分近く終わらない（通常のビルドでは各カーネル 0.02〜0.04 ms）。
    - そのため、**Makefile.NVD を `-acc=gpu` にそろえるのは、今の nvc++ では勧められない**。`default(present)` の追加は、通常のビルド（`-mp=gpu`）で結果が変わらないことを確認したうえでコミットする。
    - `default(present)` の注意: カーネルがグローバル変数の配列を使っていると、実行時に "data in PRESENT clause was not found" になる（adamw。定数表を `copyin(...)` で明示して直した）。
    - boxfilter は、変更前のコードでも同じ値で FAIL する（もともとの不具合）。
  - 比較用に、HEAD（変更前）を scratchpad の `dp/wt`、HEAD に作業ツリーの差分を当てたもの（変更後）を `dp/wt2` に、git worktree として取り出してある（作業が終わったら `git worktree remove` する）。`dp/pair.sh <name>` で、通常のビルドの変更前と変更後を GPU 0 で順番に実行し、時間以外の出力を比べる。
  - 通常のビルド（`-mp=gpu`）の `-Minfo` を変更前後で比べると、30件ともループの割り当てと `memcpy` への置き換えはまったく同じ（adamw は行番号が1つずれただけ）。
  - **注意: 2026-11-25 のセッションでは GPU が1基（GPU 0）しかない**（11-10〜11-19 は4基だった）。`CUDA_VISIBLE_DEVICES=1` を指定すると、GPU なしで動いて、遅く、結果も誤る。実行の前に `nvidia-smi` で GPU の数を確かめること。
  - 手順: scratchpad の `dp/fix_dp.py <name>`（OpenMP なしでビルドし、"Could not find allocated-variable index" の出たカーネルの指示文に `default(present)` を付けて、通るまで繰り返す）と `dp/run.sh <name> <tag> [makefile]`。セッションが切れて作業ツリーの変更が残っていたら、`git diff` で中身（`default(present)` の追加だけ）を確かめてから実行の確認に進めばよい。
- **前回の状態**: 4章 A の74件を PR から外した（`e56416827`）。残る `-acc` は248件。A のうち OpenMP なしでビルドできる deredundancy, openmp, winograd は残したが、OpenMP の名残があるので、中身の確認が必要。push はユーザーが行う（このマシンには GitHub の認証情報がない）。
- **次の候補**（どれから進めるかは未定）:
  1. 残した deredundancy, openmp, winograd の名残を確認する（4章 A）
  2. `-mp=gpu` があるときだけ通る31件に `default(present)` を付ける（`-acc=gpu` にそろえるなら必要。機械的に直せるが、1件ずつ実行して確認する）
  3. upstream のファイル構成の変更に追従していない6件（lud, gd, gpp, goulash, heartwall, extend2）と、haversine, rainflow, kmeans
  4. gang だけのループの候補33件の性能の確認（正しさは、確認できた3件では問題なし）
  5. 存在しない `prepare_data.sh` を呼んでいる `include/*-NVIDIA` 55件、lulesh-acc の結果の誤り、blas-gemm の書き直し
- **確認の手順**（scratchpad は日をまたぐと消えるので、ここに書いておく）:
  - **セッションの最初に、必ず GPU の数を確かめる**（`module load nvhpc` の後に `nvidia-smi --query-gpu=index,name,compute_cap --format=csv`）。GPU の数はセッションごとに変わる（11-10〜11-19 は4基、11-25 は1基）。存在しない GPU を `CUDA_VISIBLE_DEVICES` で指定すると、GPU なしで動いて、遅く、結果も誤る。
  - `module load nvhpc` をしてから、`cd src/<name>-acc && make -f Makefile.NVD clean && make -f Makefile.NVD SM=cc100 run`。結果は `log.std` / `log.err` / `log.time`。
  - OpenMP なしの確認: `sed 's/-mp=gpu/-acc=gpu/' Makefile.NVD > /tmp/…/x.mk` を作り、`make -f /tmp/…/x.mk SM=cc100 run`。
  - 比較: `-omp` は `make -f Makefile.nvc SM=cc100`、`-cuda` は `make ARCH=sm_100`。同じ引数で実行する。
  - ビルドのログで、パターンルールが使われたか（`-acc` 付きでコンパイルされたか）を必ず確認する（組み込みルールに切り替わる落とし穴がある。4章 B）。
  - 実行に5分以上かかるもの（simplemoc、kalman の既定サイズなど）はバックグラウンドで実行する。
  - 全件ビルド: 1件ごとに、`-mp=multicore,gpu`→`-acc=gpu`、`-mp=multicore`→`-acc=multicore`、`-mp=gpu`→`-acc=gpu` に置き換えた Makefile でビルドし、失敗したら通常の Makefile.NVD でも試す。`xargs -P 12` で全322件が約2分。

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
| 4章 A のうちビルドできない73件と binomial（計74件） | 「Not yet implemented」として PR から外す（2026-11-19 に決定）。書き直しは後の別 PR で行う | `e56416827`（削除前の状態は `63f12161d`） |
| `src/include/*-NVIDIA` | **すべて残す**。`-acc` がないベンチマーク向けの約170件も、将来 `-acc` を追加するかもしれないので残す | — |
| `src/include/SDKBitMap.h`, `src/hausdorff-cuda/reference.h` | `#if defined(_OPENACC)` の追加だけ。`-acc` に必要なので残す | PR #3 |

## 2. これまでの作業（PR #3 マージ後、`f1902eb1a` 以降）

```
9c2924105 [pnpoly-acc] port the data clauses and work around two nvc++ issues
e26dadabf [linearprobing-acc] port the data clauses, kernels and atomic compare to OpenACC
2c1db65a5 [kalman-acc] use OpenACC data clauses and map each series to one thread
f340613f7 [lci-acc] build with the OpenACC flags and fix the gang clause
61465e885 [ecdh-acc] mark the device functions with acc routine seq
017b12205 [debayer-acc] mark the device functions with acc routine seq
a28316ae7 [complex-acc] mark the device functions with acc routine seq
6cef2c456 [chacha20-acc] mark the device functions with acc routine seq
7be2fee27 Update ACC_WORK_STATUS.md after fixing the host-only OpenMP remnants (temporary; remove before the upstream PR)
c8120d96d [su3-acc] drop the unused omp.h include
0b030785f [rsc-acc] drop the unused omp.h includes
8d0b5406c [rsbench-acc] time with clock_gettime instead of omp_get_wtime
9fd25651c [myocyte-acc] drop the unused omp.h include
c5bac52ba [mr-acc] use OpenACC data clauses and fix the atomic updates
e0c31c939 [backprop-acc] restore the host OpenMP pragmas in the #ifdef OPEN blocks
b76e89284 Update ACC_WORK_STATUS.md with the GPU check results (temporary; remove before the upstream PR)
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
- `e0c31c939`〜`c8120d96d`（6コミット）: 4章 C の修正（2026-11-17）。
- `6cef2c456`〜`9c2924105`（8コミット）: 4章 B の修正（2026-11-19）。

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
  - grrt, iso2dfd, tridiagonal の `make clean` が、`*.txt` の指定で追跡中の `CMakeLists.txt` を消していた。cuda・omp 版と同じく、ファイル名を個別に書くようにした（`035ec4cea`, `44238d593`, `7b2d58ff7`）。
  - **私のコミット `1ffa6c53e` で hausdorff のビルドが壊れていた**。`hd()` を使われていないと判断して消したが、`../hausdorff-cuda/reference.h` が使っていた。消した後にビルドし直していなかった。`c38d652bb` で戻した（8.8 ms、PASS）。教訓: 変更の後は、必ずクリーンビルドと実行をしてからコミットする。

### A. カーネルの書き直しが必要（77件・263カーネル）：**74件を PR から外した**（2026-11-19, `e56416827`）

全件ビルドで、73件が `-mp=gpu` があってもビルドできないことを確認したので、binomial（Makefile.NVD なし）と合わせて74件を「Not yet implemented」として外した。OpenMP なしでビルドできた deredundancy, openmp, winograd の3件は残した（ただし名残がある）。以下は、外す前の記録。

gang のループと `loop vector` で組み直し、`barrier` で区切られた部分は別々の vector ループに分ける必要がある。括弧内はカーネル数（`acc parallel teams` の数）。

adv(1) aes(2) aop(5) assert(2) asta(1) axhelm(2) binomial(1) bn(1) cm(3) cmp(2) convolutionSeparable(2) crs(20) d2q9-bgk(1) dct8x8(2) ddbp(8) deredundancy(1) dxtc2(1) expdist(1) fdtd3d(1) fft(2) filter(1) fpc(2) fpdc(1) frna(10) gc(2) gmm(10) grep(1) hexciton(9) histogram(0) hybridsort(6) lanczos(3) langford(1) lavaMD(1) lda(1) ldpc(4) lid-driven-cavity(11) match(10) merge(2) metropolis(4) morphology(2) nms(2) ntt(1) nw(2) openmp(0) p4(1) particlefilter(3) particles(9) permutate(0) pns(2) prna(8) qtclustering(4) quicksort(2) radixsort(6) randomAccess(1) reaction(6) recursiveGaussian(3) rng-wallace(1) s3d(27) scan(2) scan2(3) slu(6) sobol(1) softmax-online(2) sort(3) split(1) sptrsv(1) srad(1) tqs(1) tridiagonal(8) tsa(1) tsp(1) vmc(5) wedford(1) winograd(0) wlcpow(1) xlqc(4) zmddft(3)

- カーネル数 0 の histogram, openmp, permutate, winograd は、`teams` はないが `#pragma omp parallel` や `omp_get_*()` などが残っているもの。
- status2 で正常だったもの: deredundancy, openmp, winograd。記載がないもの: binomial。ほかはすべて build err。

### B. pragma の置き換えで直る（12件）：**8件完了、残り2件**（2026-11-19）

数え直し（OpenMP なし、つまり `-mp=gpu` を `-acc=gpu` に置き換えた一時的な Makefile で、実際にビルドして実行した）の結果と対応:

| ベンチマーク | 実際の状況 | 対応（コミット） | 結果 |
|---|---|---|---|
| chacha20, complex, debayer, ecdh | `omp declare target` が残っていた（nvc++ は自動でデバイス向けにするので動いていた） | 関数ごとの `acc routine seq` に置き換えた（`6cef2c456`, `a28316ae7`, `017b12205`, `61465e885`） | PASS（debayer はチェックサムが omp と一致） |
| lci | Makefile.NVD の依存先 `tables.h` が存在せず、組み込みルールで `-acc` なしにビルドされていた（**CPU で実行されていた**）。依存先を直すと `gang(1)` がエラーになった | 依存先を `../lci-cuda/tables.h` に直し（Makefile も）、`gang vector num_gangs(1) vector_length(96)` にした。routine も置き換えた（`f340613f7`） | `-DDUMP` の出力が CPU のビルドと一致 |
| kalman | OpenMP の書き方のデータ句で、ビルドできなかった | `copyin/create`、`update self` に直した。カーネルを `gang vector` にした（gang だけだと作業配列が共有メモリに置かれ、内側のループが vector に割り当てられていた。カーネル 1.9s → 0.08s）。OpenMP なしでもビルドできるよう `default(present)` を付けた（`2c1db65a5`） | **FAIL のまま**（下記） |
| linearprobing | データ句と `num_teams/thread_limit` が OpenMP の書き方。OpenACC にない `acc atomic compare capture` を使っていた | `copyin/create`、`update device/self`、`gang vector num_gangs/vector_length` に直した。compare-and-swap は C++20 の `std::atomic_ref::compare_exchange_strong` にした（`-std=c++20`）。`default(present)` も付けた（`e26dadabf`） | PASS |
| pnpoly | OpenMP の書き方のデータ句で、ビルドできなかった | `copyin/copyout` に直し、カーネルを `gang vector` にし、範囲を明示した `present` を付けた。nvc++ の不具合を2つ回避した（下記）（`9c2924105`） | PASS。チェックサムが multicore のビルドとホストでの検算に一致 |
| diamond | — | **修正済み**（`19c512f8b`, `b496229af`） | — |
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
| backprop | `#ifdef OPEN`（未定義でビルドされない）の中で、`omp parallel for` が誤って `acc loop ... schedule(static)` に置き換えられていた。このブロックは cuda・omp 版にもある upstream 共通のホスト側コード | ほかの版と同じ `omp parallel for` に戻した（`e0c31c939`） | PASS |
| mr | `acc parallel data map(...)`、`acc update to/from` が残っていた。また、`acc atomic update` の式の中で関数を呼んでいて、もともとビルドできなかった | `acc data copyin/create`、`update device/self` に直し、関数の結果をローカル変数に入れてから atomic で加えるようにした（`c5bac52ba`） | PASS（初めてビルドが通った） |
| rsbench | `rsbench.h` に `#define OPENMP` があり、`omp_get_wtime()` を実際に使っていた | `clock_gettime(CLOCK_MONOTONIC)` に置き換えた（ソースが .c なので chrono ではなくこれにした）。表示も「OpenACC Init」に直した（`8d0b5406c`） | Valid |
| myocyte, rsc, su3 | 使っていない `#include <omp.h>` があった | 削除した（`9fd25651c`, `0b030785f`, `c8120d96d`） | rsc：Test Passed、su3：SUCCESS、myocyte：5章を参照 |
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
- **myocyte-acc の出力が `nan`**: `output.txt` の約92万行のうち87万行が `nan`。`9fd25651c` より前の状態（`omp.h` あり、`-mp=gpu`）でも同じなので、もともとの不具合。omp 版には `nan` がないが、時刻 0 の値（`y[0][0][0]=1.13e30`）が入力 `data/myocyte/y.txt` の先頭（1.37e-3）と合っていないので、upstream の実装も含めて調べる必要がある。実行コマンドは Makefile.NVD の50行目の COMMAND（`-time 10000`）で、`include/myocyte-NVIDIA` の COMMAND は上書きされている。
- **upstream の `myocyte-omp/Makefile.nvc` が壊れている**: リンクのルールの行の続け方が誤っていて、ビルドできない（比較のときは手でビルドした）。
- **simplemoc**: カーネル時間が約220秒、実行全体で約5分かかる（コードは変えていない）。
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
