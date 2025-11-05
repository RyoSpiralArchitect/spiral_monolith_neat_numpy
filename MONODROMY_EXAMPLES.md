# Monodromy Environment Scheduler - Examples

## What is Monodromy?

In the context of NEAT evolution, **monodromy** refers to a path-dependent environmental transformation that "follows" the population's topological evolution. Instead of a fixed generation-based curriculum, the environment dynamically adapts based on:

- **Topology Complexity**: Average hidden nodes and edges in the population
- **Evolution Velocity**: Rate of structural change
- **Population Diversity**: Variance in network structures

This creates a co-evolutionary feedback loop where structural evolution drives environmental changes, leading to punctuated equilibria dynamics similar to biological evolution.

## Basic Usage

```bash
# Enable monodromy mode with the --monodromy flag
python spiral_monolith_neat_numpy.py --task spiral --gens 60 --pop 96 --monodromy

# Compare with default (generation-based) schedule
python spiral_monolith_neat_numpy.py --task spiral --gens 60 --pop 96
```

## Example Evolution Trajectories

### Scenario 1: Early Exploration Phase

**Population State:**
- Average Hidden Nodes: 0-2
- Average Edges: 3-10
- Diversity: Low (0.1-1.0)

**Monodromy Response:**
- Difficulty: 0.35-0.45 (gentle)
- Noise: 0.02-0.03 (low)
- Regeneration: Disabled

**Why?** Early populations need stable conditions for initial structure building.

### Scenario 2: Growth Phase

**Population State:**
- Average Hidden Nodes: 5-10
- Average Edges: 25-50
- Diversity: Medium (2.0-4.0)
- Velocity: High (rapid structural changes)

**Monodromy Response:**
- Difficulty: 0.60-0.85 (challenging)
- Noise: 0.03-0.05 (moderate)
- Regeneration: Enabled

**Why?** Growing complexity triggers increased pressure to test emerging structures.

### Scenario 3: Plateau Phase

**Population State:**
- Average Hidden Nodes: 8-9 (stable)
- Average Edges: 35-40 (stable)
- Diversity: Low (1.0-2.0)
- Velocity: Near zero (little change)

**Monodromy Response:**
- Difficulty: 0.70-0.75 (steady)
- Noise: 0.04-0.05 (moderate)
- Regeneration: Enabled

**Why?** Stable topology indicates convergence; maintain pressure without disrupting.

### Scenario 4: Second Growth Burst

**Population State:**
- Average Hidden Nodes: 12-20
- Average Edges: 60-100
- Diversity: High (4.0-6.0)
- Velocity: Moderate (renewed exploration)

**Monodromy Response:**
- Difficulty: 0.80-0.95 (high)
- Noise: 0.05-0.07 (high)
- Regeneration: Enabled

**Why?** Complex, diverse topologies need strong pressure to maintain fitness.

## Comparison with Default Schedule

| Generation | Default Difficulty | Monodromy (Low Complexity) | Monodromy (High Complexity) |
|------------|-------------------|----------------------------|----------------------------|
| 0-14       | 0.40              | 0.35-0.45                  | 0.35-0.50                  |
| 15-29      | 0.50-0.80         | 0.50-0.65                  | 0.65-0.85                  |
| 30-49      | 1.00-1.80         | 0.60-0.75                  | 0.80-1.00                  |
| 50+        | 1.80-2.50+        | 0.70-0.85                  | 0.85-1.20                  |

**Key Difference:** Default schedule increases indefinitely with generation. Monodromy adapts to actual population state.

## Advanced Configuration

You can customize normalization parameters via the fitness function context:

```python
from spiral_monolith_neat_numpy import ReproPlanaNEATPlus, _topology_monodromy_schedule

# Create custom schedule with different normalization ranges
def custom_monodromy(gen, ctx):
    # For problems expected to evolve larger topologies
    ctx['max_hidden_norm'] = 100.0  # Expect up to 100 hidden nodes
    ctx['max_edges_norm'] = 500.0   # Expect up to 500 edges
    return _topology_monodromy_schedule(gen, ctx)

# Use in evolution
neat = ReproPlanaNEATPlus(...)
neat.evolve(fitness_fn, env_schedule=custom_monodromy)
```

## Mathematical Model

The monodromy difficulty is computed as:

```
difficulty = (base + monodromy + velocity) × diversity_factor

where:
  base = 0.3 + 1.5 × normalized_complexity
  monodromy = amplitude × sin(phase)
  velocity = 0.15 × tanh(|structural_change| / 2)
  diversity_factor = 1 - 0.2 × tanh(diversity / 3)
```

The periodic monodromy component creates oscillations based on cumulative topology:
```
phase = (avg_hidden × 0.1 + avg_edges × 0.02) mod 2π
amplitude = 0.2 + 0.3 × normalized_complexity
```

This ensures the environment "remembers" the evolutionary path, creating history-dependent dynamics.

## When to Use Monodromy

**Use Monodromy When:**
- You want evolution to adapt organically to population dynamics
- Exploring open-ended evolution scenarios
- Seeking punctuated equilibria dynamics
- Population structure varies significantly across runs

**Use Default Schedule When:**
- You need reproducible, deterministic difficulty curves
- Comparing results across different experiments
- Fixed curriculum is important for your research
- Benchmarking against established baselines

## Expected Benefits

1. **Organic Adaptation**: Environment responds to actual evolutionary progress
2. **Punctuated Equilibria**: Periods of rapid change followed by stability
3. **Diverse Solutions**: Different runs explore different regions of topology space
4. **Reduced Overfitting**: Varying difficulty prevents convergence to local optima
5. **Biological Realism**: Mirrors real-world co-evolutionary dynamics

## Implementation Details

The monodromy scheduler is implemented in `_topology_monodromy_schedule()` and integrates seamlessly with the existing NEAT framework:

1. Each generation, `_compute_topology_metrics()` calculates population statistics
2. Metrics are passed to the environment schedule function
3. Schedule returns updated difficulty, noise, and regeneration settings
4. Environment applies these settings for fitness evaluation
5. Metrics are stored in environment history for next generation

No changes to genome structure or fitness functions are required - monodromy is purely an environmental scheduling mechanism.
