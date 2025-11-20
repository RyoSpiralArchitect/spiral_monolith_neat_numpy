# Spiral Monolith NEAT NumPy

A single-file research playground that blends NEAT-style topology evolution with NumPy-powered backprop refinement, rich artefact exports, and a batteries-included CLI. Everything lives in one script so you can inspect, tweak, and rerun the full stack without hunting through packages.

## Overview
* **Monodromy-aware evolution** – Environment difficulty follows population topology changes, creating path-dependent curricula and punctuated-equilibrium dynamics.
* **Hybrid NEAT × backprop loop** – Winning genomes receive NumPy fine-tuning to squeeze extra accuracy without abandoning structural innovation.
* **Collective cognition signals** – Shared “altruism”, “solidarity”, and stress metrics feed back into learning, supporting cooperative evolutionary experiments.
* **RL defences and DSL accelerations** – Replay, scheduler, and diversity scaling all run through compiled cpp-style DSL kernels to reduce Python overhead while evolving hyperparameters such as learning rate, entropy bonus, discount, and GAE lambda.
* **Artefact-rich exports** – Built-in exporters generate topology PNGs, lineage graphs, regeneration timelines, morph GIFs, and Lazy Council telemetry ribbons.
* **Headless-friendly stack** – Matplotlib is pre-configured for CPU-only environments so remote servers and CI can render assets without extra flags.

## Quickstart
List every experiment preset, exporter, and advanced flag:
```bash
python spiral_monolith_neat_numpy.py --help
```

Run the default spinor governance demo and emit artefacts under `out/monolith`:
```bash
python spiral_monolith_neat_numpy.py --out out/monolith
```

## Signature CLI Recipes
**Spiral benchmark explorer**
```bash
python spiral_monolith_neat_numpy.py \
  --task spiral --gens 60 --pop 96 --steps 60 \
  --make-gifs --make-lineage --report --out out/spiral_bold
```
Generates spiral classification diagnostics with lineage graphs, training GIFs, and an HTML report in `out/spiral_bold`.

**Monodromy spotlight**
```bash
python spiral_monolith_neat_numpy.py \
  --task spiral --gens 60 --pop 96 --steps 60 \
  --monodromy --make-gifs --make-lineage --report --out out/spiral_monodromy
```
Enables the topology-aware curriculum where structural shifts drive environment pressure.

**Gym integration baseline**
```bash
python - <<'PY'
from spiral_monolith_neat_numpy import run_gym_neat_experiment
run_gym_neat_experiment(
    "CartPole-v1", gens=30, pop=64, episodes=3, max_steps=500,
    stochastic=True, temp=0.8, out_prefix="out/cartpole",
)
PY
```
Launches a NEAT + backprop hybrid baseline on CartPole, recording reward curves and GIFs under `out/cartpole`.

## Artefact Outputs
PNG figures, GIF animations, and optional HTML reports are written beneath the directory passed to `--out` (or `out_prefix`). The exporters use the hardened `_savefig` pipeline so permissions, fonts, and transparency settings are consistent across platforms.

## Monodromy Mode
Passing `--monodromy` enables a topology-aware environment scheduler that creates a feedback loop between network structure evolution and difficulty. Instead of a fixed generation-based curriculum, the environment adapts to population-wide structural metrics such as:
- **Average topology complexity** (hidden nodes and edges)
- **Topology change velocity** (pace of structural evolution)
- **Population diversity** (structural dispersion between genomes)

Cumulative structural changes trigger periodic oscillations so the environment follows the population’s trajectory through topology space, yielding path-dependent “monodromy” effects.
