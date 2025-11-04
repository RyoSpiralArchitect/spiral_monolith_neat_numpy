# Spiral Monolith NEAT NumPy

This repository contains a single-file research playground that fuses NEAT evolution,
NumPy backpropagation fine-tuning, rich visualisations, reinforcement learning probes,
and a self-contained CLI for exporting publication-ready artefacts.

## Quickstart

Create a virtual environment with the runtime dependencies and run the script directly:

```bash
python spiral_monolith_neat_numpy.py --help
```

## Suggested experiments

The following commands highlight the most useful demos.

```bash
# Deep dive on the spiral classification task with visual artefacts and report output
python spiral_monolith_neat_numpy.py \
  --task spiral --gens 60 --pop 96 --steps 60 \
  --make-gifs --make-lineage --report --out out/spiral_bold

# Topology-aware monodromy mode: environment adapts to population topology evolution
python spiral_monolith_neat_numpy.py \
  --task spiral --gens 60 --pop 96 --steps 60 \
  --monodromy --make-gifs --make-lineage --report --out out/spiral_monodromy

# Reinforcement-learning baseline on CartPole with reward curve export
python - <<'PY'
from spiral_monolith_neat_numpy import run_gym_neat_experiment
run_gym_neat_experiment(
    "CartPole-v1", gens=30, pop=64, episodes=3, max_steps=500,
    stochastic=True, temp=0.8, out_prefix="out/cartpole"
)
PY
```

All outputs (PNG figures, GIFs, and optional HTML reports) are written beneath the
`--out` directory you choose. The CLI defaults to a CPU-friendly, backend-agnostic
matplotlib configuration so the commands above can run headlessly on macOS or Linux.

## Monodromy Mode

The `--monodromy` flag enables a topology-aware environment scheduler that creates a feedback
loop between network structure evolution and environmental difficulty. Instead of a fixed
generation-based curriculum, the environment dynamically adapts based on:

- **Average topology complexity** (hidden nodes and edges in the population)
- **Topology change velocity** (rate of structural evolution)
- **Population diversity** (structural variance across individuals)

This creates a "monodromy" - a path-dependent transformation where the environment follows
and responds to the population's evolutionary trajectory through topology space. The monodromy
schedule includes periodic oscillations that depend on cumulative topological changes, creating
evolutionary dynamics that resemble biological punctuated equilibria.
