# Spiral Monolith NEAT NumPy / スパイラル・モノリス NEAT NumPy

**EN:** A single-file research playground blending NEAT evolution, NumPy-powered backprop fine-tuning, interactive visual analytics, and a batteries-included CLI.
**JP:** NEAT 進化と NumPy バックプロパゲーション微調整、可視化分析、充実した CLI を 1 ファイルに凝縮した研究用プレイグラウンドです。

## Overview / 概要
This project provides an immediately runnable environment for experimenting with neuroevolution techniques that bridge symbolic topology search and gradient-based refinement. Everything ships in a single Python script so you can read, modify, and execute the full stack without hunting through modules.

本プロジェクトは、トポロジー探索と勾配ベース微調整を横断するニューロ進化手法をすぐに試せる実行環境です。ソースは 1 つの Python スクリプトにまとまっており、参照・改造・実行を一気通貫で行えます。

## Quickstart / クイックスタート
```bash
python spiral_monolith_neat_numpy.py --help
```
This command lists every experiment preset, export toggle, and advanced flag available from the CLI so you can discover features before running long jobs.

上記コマンドで利用可能な実験プリセット、エクスポートオプション、詳細フラグを確認し、長時間ジョブを回す前に機能を把握できます。

## Key Innovations / 主要な革新的機能
- **Monodromy-aware evolution** — Environment difficulty follows population topology changes, enabling path-dependent curricula for richer dynamics.<br>**モノドロミー対応進化** — 環境難度が個体群トポロジーの変化に追随し、軌跡依存のカリキュラムで多彩なダイナミクスを実現します。
- **Hybrid NEAT × backprop loop** — Winning genomes receive NumPy fine-tuning to squeeze extra accuracy without abandoning structural innovation.<br>**NEAT とバックプロップのハイブリッドループ** — 優勝ゲノムに NumPy 微調整を適用し、構造的革新を保ったまま精度を引き上げます。
- **Collective cognition signals** — Shared "altruism", "solidarity", and stress metrics feed back into learning, letting you prototype cooperative evolutionary dynamics.<br>**集団認知シグナル** — 「利他性」「連帯」「ストレス」などの共有メトリクスが学習へフィードバックし、協調的な進化ダイナミクスを試作できます。
- **Rich artefact exports** — Built-in exporters deliver lineage graphs, morph GIFs, regeneration timelines, and LCS ribbon plots for publication-ready storytelling.<br>**多彩な成果物エクスポート** — 系統グラフ、形態 GIF、再生タイムライン、LCS リボン図など、論文レベルのビジュアルを標準機能で生成します。
- **Headless-friendly stack** — Matplotlib is pre-configured for CPU-only, display-less environments so remote servers and CI can render assets without tweaks.<br>**ヘッドレス対応スタック** — Matplotlib を CPU/ヘッドレス向けに事前調整し、リモートサーバーや CI でも設定不要で描画できます。

## Signature CLI Recipes / 代表的な CLI レシピ
### Spiral benchmark explorer / スパイラル課題の徹底解析
```bash
python spiral_monolith_neat_numpy.py \
  --task spiral --gens 60 --pop 96 --steps 60 \
  --make-gifs --make-lineage --report --out out/spiral_bold
```
Get full spiral classification diagnostics with population lineage graphs, training GIFs, and a structured HTML report in `out/spiral_bold`.

スパイラル分類の診断情報（系統グラフ、学習 GIF、HTML レポート）を一括生成し、`out/spiral_bold` に保存します。

### Monodromy mode spotlight / モノドロミーモード解説
```bash
python spiral_monolith_neat_numpy.py \
  --task spiral --gens 60 --pop 96 --steps 60 \
  --monodromy --make-gifs --make-lineage --report --out out/spiral_monodromy
```
Activates the topology-aware curriculum where structural shifts drive environment pressure, ideal for studying punctuated equilibria effects.

トポロジー変化に応じて環境圧を調整するカリキュラムを有効化し、断続平衡の挙動を観察できます。

### Gym integration baseline / Gym 連携ベースライン
```bash
python - <<'PY'
from spiral_monolith_neat_numpy import run_gym_neat_experiment
run_gym_neat_experiment(
    "CartPole-v1", gens=30, pop=64, episodes=3, max_steps=500,
    stochastic=True, temp=0.8, out_prefix="out/cartpole"
)
PY
```
Launches a NEAT + backprop hybrid baseline on CartPole, recording reward curves and GIFs under `out/cartpole` for downstream analysis.

CartPole で NEAT × バックプロップのベースライン実験を走らせ、報酬カーブや GIF を `out/cartpole` に保存します。

## Artefact Outputs / 生成される成果物
All PNG figures, GIF animations, and optional HTML reports are written beneath the directory passed to `--out` (or `out_prefix`). The exporters rely on the hardened `_savefig` pipeline so permissions, fonts, and transparency settings are consistent across platforms.

PNG 図版、GIF アニメーション、HTML レポートは `--out`（または `out_prefix`）で指定したディレクトリ以下に整理されます。エクスポーターは堅牢化した `_savefig` パイプラインを使用し、権限やフォント、透過設定をプラットフォーム間で揃えます。

## Monodromy Mode / モノドロミーモード
The `--monodromy` flag enables a topology-aware environment scheduler that creates a feedback loop between network structure evolution and environmental difficulty. Instead of a fixed generation-based curriculum, the environment dynamically adapts based on population-wide structural metrics such as:

`--monodromy` フラグは、ネットワーク構造の進化と環境難度が双方向に作用するスケジューラを有効化します。従来の固定カリキュラムではなく、以下のような集団トポロジー指標に基づき動的に調整されます。

- **Average topology complexity / 平均トポロジー複雑度**（隠れノードやエッジ数）
- **Topology change velocity / トポロジー変化速度**（構造進化のテンポ）
- **Population diversity / 個体群多様性**（個体間の構造分散）

This creates a path-dependent transformation—"monodromy"—where the environment follows and responds to the population's evolutionary trajectory through topology space. Periodic oscillations triggered by cumulative structural changes help simulate biological punctuated equilibria.

この仕組みにより、トポロジー空間での進化的軌跡に環境が追随・応答する「モノドロミー（単回帰）」的な変換が成立します。構造変化の累積に応じた周期振動が導入され、生物学的な断続平衡を想起させるダイナミクスを再現します。
