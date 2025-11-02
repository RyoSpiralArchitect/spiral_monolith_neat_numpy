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
