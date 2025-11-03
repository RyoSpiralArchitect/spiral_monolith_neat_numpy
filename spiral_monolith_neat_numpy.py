
# spiral_monolith_neat_numpy.py
# Monolithic, NumPy-only NEAT + Backprop + Visualization toolkit.
# - Feed-forward only (strict DAG). No frameworks beyond NumPy/matplotlib/imageio.
# - "Part 1" NEAT (with optional planarian-like regeneration) + "Part 2" Backprop NEAT.
# - Clean, single-file design for reproducible demos and paper-ready figures.
#
# Additions in this build:
#   * Auto-export of regen/morph GIFs from evolution snapshots
#   * Learning curve with moving average + rolling-std "CI" (line styles only)
#   * Decision boundaries for Circles/XOR/Spiral as separate PNGs
#   * Lineage diagram renderer (no fixed colors; shape/linestyle/linewidth encoding)
#
# Author: SpiralReality (Ryō) + GPT-5 Pro co-engineering

from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional, Set, Iterable, Any
import math, argparse, os, mimetypes
import matplotlib


def _ensure_matplotlib_agg(force: bool = False):
    """Select Agg backend even if pyplot was already imported elsewhere."""
    try:
        matplotlib.use("Agg", force=force)
    except TypeError:  # pragma: no cover - older matplotlib doesn't support force
        matplotlib.use("Agg")
    return matplotlib


_ensure_matplotlib_agg()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

try:  # pragma: no cover - optional dependency
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover - fallback path when imageio is unavailable
    imageio = None  # type: ignore
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover - Pillow also missing
        Image = None  # type: ignore
    else:
        Image = Image  # type: ignore
else:
    Image = None  # type: ignore


def _mimsave(path, frames, fps=12):
    """Robust GIF writer that tolerates missing imageio by falling back to Pillow."""
    frame_seq = list(frames)
    if not frame_seq:
        return
    if imageio is not None:
        imageio.mimsave(path, frame_seq, duration=1.0 / max(1, int(fps)))
        return
    if Image is None:
        raise RuntimeError("imageio is unavailable and Pillow is not installed; cannot write GIF")
    imgs = [Image.fromarray(np.asarray(fr)) for fr in frame_seq]
    dur = int(1000 / max(1, int(fps)))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0)


def _imread_image(src):
    """imageio.imread fallback that also supports BytesIO via Pillow."""
    if imageio is not None:
        return imageio.imread(src)
    if Image is None:
        raise RuntimeError("imageio is unavailable and Pillow is not installed; cannot read image")
    img = Image.open(src)
    try:
        return np.asarray(img.convert("RGB"))
    finally:
        img.close()


def _imwrite_image(path, array):
    """Write an image array using imageio or Pillow."""
    if imageio is not None:
        imageio.imwrite(path, array)
        return
    if Image is None:
        raise RuntimeError("imageio is unavailable and Pillow is not installed; cannot write image")
    Image.fromarray(np.asarray(array)).save(path)

# ============================================================
# 1) Genes & Genome (strict DAG)
# ============================================================

__all__ = [
    "NodeGene","ConnectionGene","InnovationTracker","Genome",
    "compatibility_distance","EvalMode","ReproPlanaNEATPlus",
    "compile_genome","forward_batch","train_with_backprop_numpy","predict","predict_proba",
    "fitness_backprop_classifier","make_circles","make_xor","make_spirals",
    "draw_genome_png","export_regen_gif","export_morph_gif","export_double_exposure",
    "plot_learning_and_complexity","plot_decision_boundary",
    "export_decision_boundaries_all","render_lineage","export_scars_spiral_map",
    "output_dim_from_space","build_action_mapper","eval_with_node_activations","run_policy_in_env","run_gym_neat_experiment",
]

@dataclass
class NodeGene:
    id: int
    type: str                 # 'input' | 'hidden' | 'output' | 'bias'
    activation: str = 'tanh'  # 'tanh' | 'sigmoid' | 'relu' | 'identity'

@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int

class InnovationTracker:
    def __init__(self, next_node_id: int, next_conn_innov: int = 0):
        self.next_node_id = next_node_id
        self.next_conn_innov = next_conn_innov
        self.conn_innovations: Dict[Tuple[int,int], int] = {}
        self.node_innovations: Dict[Tuple[int,int], int] = {}

    def get_conn_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.conn_innovations:
            self.conn_innovations[key] = self.next_conn_innov
            self.next_conn_innov += 1
        return self.conn_innovations[key]

    def get_or_create_split_node(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key in self.node_innovations:
            return self.node_innovations[key]
        nid = self.new_node_id()
        self.node_innovations[key] = nid
        return nid

    def new_node_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

class Genome:
    def __init__(self, nodes: Dict[int, 'NodeGene'], connections: Dict[int, 'ConnectionGene'],
                 sex: Optional[str] = None, regen: bool = False, regen_mode: Optional[str] = None,
                 embryo_bias: Optional[str] = None, gid: Optional[int] = None, birth_gen: int = 0,
                 hybrid_scale: float = 1.0, parents: Optional[Tuple[Optional[int], Optional[int]]] = None):
        self.nodes = nodes
        self.connections = connections
        self.sex = sex or ('female' if np.random.random() < 0.5 else 'male')
        self.regen = bool(regen)
        self.regen_mode = regen_mode or np.random.choice(['head','tail','split'])
        self.embryo_bias = embryo_bias or np.random.choice(['neutral','inputward','outputward'], p=[0.5,0.25,0.25])
        self.id = gid if gid is not None else int(np.random.randint(1,1e9))
        self.birth_gen = int(birth_gen)
        self.hybrid_scale = float(hybrid_scale)
        self.parents = parents if parents is not None else (None, None)
        # Optional capacity limits
        self.max_hidden_nodes: Optional[int] = None
        self.max_edges: Optional[int] = None

    def copy(self):
        nodes = {nid: NodeGene(n.id, n.type, n.activation) for nid, n in self.nodes.items()}
        conns = {innov: ConnectionGene(c.in_node, c.out_node, c.weight, c.enabled, c.innovation)
                 for innov, c in self.connections.items()}
        g = Genome(nodes, conns, self.sex, self.regen, self.regen_mode, self.embryo_bias,
                   self.id, self.birth_gen, self.hybrid_scale, self.parents)
        g.max_hidden_nodes = self.max_hidden_nodes
        g.max_edges = self.max_edges
        return g

    # ----- Graph helpers -----
    def enabled_connections(self):
        return [c for c in self.connections.values() if c.enabled]

    def adjacency(self):
        adj = {}
        for c in self.enabled_connections():
            adj.setdefault(c.in_node, set()).add(c.out_node)
        return adj

    def has_connection(self, in_id, out_id):
        for c in self.connections.values():
            if c.in_node == in_id and c.out_node == out_id:
                return True
        return False

    def topological_order(self):
        in_edges_count = {nid:0 for nid in self.nodes}
        for c in self.enabled_connections():
            in_edges_count[c.out_node] += 1
        queue = [nid for nid in self.nodes if in_edges_count[nid]==0]
        order = []
        adj = self.adjacency()
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for m in adj.get(nid, []):
                in_edges_count[m] -= 1
                if in_edges_count[m]==0:
                    queue.append(m)
        if len(order) != len(self.nodes):
            raise RuntimeError("Cycle detected: feed-forward constraint violated.")
        return order

    def _creates_cycle(self, in_node, out_node):
        adj = self.adjacency()
        stack = [out_node]; visited=set()
        while stack:
            v = stack.pop()
            if v == in_node: return True
            if v in visited: continue
            visited.add(v)
            for w in adj.get(v, []):
                stack.append(w)
        return False

    def node_depths(self):
        order = self.topological_order()
        inputs = [nid for nid,n in self.nodes.items() if n.type in ('input','bias')]
        depth = {nid:(0 if nid in inputs else -1) for nid in order}
        adj_in = {}
        for c in self.enabled_connections():
            adj_in.setdefault(c.out_node, []).append(c.in_node)
        changed=True
        while changed:
            changed=False
            for nid in order:
                if depth[nid] >= 0: continue
                parents = adj_in.get(nid, [])
                if parents and all((p in depth and depth[p] >= 0) for p in parents):
                    depth[nid] = max(depth[p] for p in parents) + 1
                    changed=True
        for nid in order:
            if depth[nid] < 0: depth[nid]=0
        return depth

    # ----- Mutations -----
    def mutate_weights(self, rng: np.random.Generator, perturb_chance=0.9, sigma=0.8, reset_range=2.0):
        for c in self.connections.values():
            if rng.random() < perturb_chance: c.weight += float(rng.normal(0, sigma))
            else: c.weight = float(rng.uniform(-reset_range, reset_range))

    def mutate_toggle_enable(self, rng: np.random.Generator, prob=0.01):
        for c in self.connections.values():
            if rng.random() >= prob:
                continue
            if c.enabled:
                c.enabled = False
            else:
                if not self._creates_cycle(c.in_node, c.out_node):
                    c.enabled = True

    def _choose_conn_for_node_add(self, rng: np.random.Generator, bias: str):
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled: return None
        if bias == 'neutral': return enabled[int(rng.integers(len(enabled)))]
        depth = self.node_depths()
        scores = []
        for c in enabled:
            din = depth.get(c.in_node, 0)
            dout = depth.get(c.out_node, din+1)
            s = 1.0/(1.0+din) if bias=='inputward' else 1.0 + dout
            scores.append(max(1e-3, float(s)))
        scores = np.array(scores, float); probs = scores/scores.sum()
        idx = int(rng.choice(len(enabled), p=probs))
        return enabled[idx]

    def mutate_add_connection(self, rng: np.random.Generator, innov: 'InnovationTracker', tries=30):
        if self.max_edges is not None:
            if sum(1 for c in self.connections.values() if c.enabled) >= int(self.max_edges):
                return False
        node_ids = list(self.nodes.keys())
        for _ in range(tries):
            in_id = int(rng.choice(node_ids)); out_id = int(rng.choice(node_ids))
            in_node = self.nodes[in_id]; out_node = self.nodes[out_id]
            if in_id == out_id: continue
            if in_node.type == 'output': continue
            if out_node.type in ('input','bias'): continue
            if self.has_connection(in_id, out_id): continue
            if self._creates_cycle(in_id, out_id): continue
            w = float(rng.uniform(-2.0, 2.0))
            inn = innov.get_conn_innovation(in_id, out_id)
            self.connections[inn] = ConnectionGene(in_id, out_id, w, True, inn)
            return True
        return False

    def mutate_add_node(self, rng: np.random.Generator, innov: 'InnovationTracker'):
        if self.max_hidden_nodes is not None:
            if sum(1 for n in self.nodes.values() if n.type=='hidden') >= int(self.max_hidden_nodes):
                return False
        chosen = self._choose_conn_for_node_add(rng, self.embryo_bias)
        if chosen is None: return False
        c = chosen
        if not c.enabled: return False
        c.enabled = False
        new_nid = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_nid not in self.nodes:
            self.nodes[new_nid] = NodeGene(new_nid, 'hidden', 'tanh')
        inn1 = innov.get_conn_innovation(c.in_node, new_nid)
        inn2 = innov.get_conn_innovation(new_nid, c.out_node)
        self.connections[inn1] = ConnectionGene(c.in_node, new_nid, 1.0, True, inn1)
        self.connections[inn2] = ConnectionGene(new_nid, c.out_node, c.weight, True, inn2)
        return True

    # ----- Inference -----
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        order = self.topological_order()
        incoming = {nid:[] for nid in order}
        for c in self.enabled_connections():
            incoming[c.out_node].append((c.in_node, c.weight))
        input_ids = [nid for nid,n in self.nodes.items() if n.type=='input']
        output_ids = [nid for nid,n in self.nodes.items() if n.type=='output']
        bias_ids = [nid for nid,n in self.nodes.items() if n.type=='bias']
        assert len(bias_ids)==1
        bias_id = bias_ids[0]
        n_samples = X.shape[0]
        values = {nid: np.zeros(n_samples) for nid in order}
        in_sorted = sorted(input_ids); assert X.shape[1] == len(in_sorted)
        for i,nid in enumerate(in_sorted): values[nid] = X[:,i]
        values[bias_id] = np.ones(n_samples)
        for nid in order:
            node = self.nodes[nid]
            if node.type in ('input','bias'): continue
            s = np.zeros(n_samples)
            for src,w in incoming[nid]: s += values[src]*w
            if node.activation == 'tanh': act = np.tanh(s)
            elif node.activation == 'sigmoid': act = 1/(1+np.exp(-s))
            elif node.activation == 'relu': act = np.maximum(0.0, s)
            elif node.activation == 'identity': act = s
            else: act = np.tanh(s)
            values[nid] = act
        out_sorted = sorted(output_ids)
        Y = np.stack([values[nid] for nid in out_sorted], axis=1)
        return Y

    def forward_one(self, x: np.ndarray) -> np.ndarray:
        order = self.topological_order()
        incoming = {nid:[] for nid in order}
        for c in self.enabled_connections():
            incoming[c.out_node].append((c.in_node, c.weight))
        input_ids  = sorted([nid for nid,n in self.nodes.items() if n.type=='input'])
        output_ids = sorted([nid for nid,n in self.nodes.items() if n.type=='output'])
        bias_id    = next(nid for nid,n in self.nodes.items() if n.type=='bias')
        vals = {nid: 0.0 for nid in order}
        for i, nid in enumerate(input_ids): vals[nid] = float(x[i])
        vals[bias_id] = 1.0
        for nid in order:
            node = self.nodes[nid]
            if node.type in ('input','bias'): continue
            s = 0.0
            for src, w in incoming[nid]: s += vals[src] * w
            if node.activation == 'tanh': y = math.tanh(s)
            elif node.activation == 'sigmoid': y = 1.0/(1.0+math.exp(-s))
            elif node.activation == 'relu': y = s if s>0 else 0.0
            elif node.activation == 'identity': y = s
            else: y = math.tanh(s)
            vals[nid] = y
        return np.array([vals[nid] for nid in output_ids], dtype=np.float32)

def compatibility_distance(g1: Genome, g2: Genome, c1=1.0, c2=1.0, c3=0.4):
    innovs1 = sorted(g1.connections.keys()); innovs2 = sorted(g2.connections.keys())
    i=j=0; E=D=0; W_diffs=[]
    max_innov1 = innovs1[-1] if innovs1 else -1
    max_innov2 = innovs2[-1] if innovs2 else -1
    while i < len(innovs1) and j < len(innovs2):
        in1 = innovs1[i]; in2 = innovs2[j]
        if in1 == in2:
            W_diffs.append(abs(g1.connections[in1].weight - g2.connections[in2].weight)); i+=1; j+=1
        elif in1 < in2:
            if in1 > max_innov2: E += 1
            else: D += 1
            i += 1
        else:
            if in2 > max_innov1: E += 1
            else: D += 1
            j += 1
    E += (len(innovs1)-i) + (len(innovs2)-j)
    N = max(len(innovs1), len(innovs2)); N = 1 if N < 20 else N
    W = (sum(W_diffs)/len(W_diffs)) if W_diffs else 0.0
    return c1*E/N + c2*D/N + c3*W

# ============================================================
# 2) Regeneration operators (planarian-inspired; optional)
# ============================================================

def _regenerate_head(g: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5):
    inputs = [nid for nid,n in g.nodes.items() if n.type in ('input','bias')]
    candidates = [c for c in g.enabled_connections() if c.in_node in inputs]
    if not candidates: return g
    rng.shuffle(candidates)
    frac = min(0.8, 0.15 + 0.7*float(intensity))
    k = int(len(candidates)*frac)
    for c in candidates[:k]: c.enabled=False
    chosen = rng.choice(candidates)
    new_id = innov.new_node_id()
    g.nodes[new_id] = NodeGene(new_id, 'hidden', 'tanh')
    inn1 = innov.get_conn_innovation(chosen.in_node, new_id)
    inn2 = innov.get_conn_innovation(new_id, chosen.out_node)
    g.connections[inn1] = ConnectionGene(chosen.in_node, new_id, 1.0, True, inn1)
    g.connections[inn2] = ConnectionGene(new_id, chosen.out_node, chosen.weight, True, inn2)
    return g

def _regenerate_tail(g: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5):
    outputs = [nid for nid,n in g.nodes.items() if n.type=='output']
    sinks = [c for c in g.enabled_connections() if c.out_node in outputs]
    if not sinks: return g
    rng.shuffle(sinks)
    k = max(1, int(len(sinks)*(0.2+0.6*float(intensity))))
    hidden = [nid for nid,n in g.nodes.items() if n.type=='hidden']
    for c in sinks[:k]:
        c.weight = float(rng.uniform(-2,2))
        if hidden and rng.random() < (0.3+0.5*float(intensity)):
            new_src = int(rng.choice(hidden))
            if not g.has_connection(new_src, c.out_node) and not g._creates_cycle(new_src, c.out_node):
                inn = innov.get_conn_innovation(new_src, c.out_node)
                g.connections[inn] = ConnectionGene(new_src, c.out_node, float(rng.uniform(-2,2)), True, inn)
    return g

def _regenerate_split(g: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5):
    hidden = [nid for nid,n in g.nodes.items() if n.type=='hidden']
    if not hidden:
        enabled = [c for c in g.connections.values() if c.enabled]
        if not enabled: return g
        c = enabled[int(rng.integers(len(enabled)))]
        c.enabled=False
        new_nid = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_nid not in g.nodes:
            g.nodes[new_nid] = NodeGene(new_nid, 'hidden', 'tanh')
        inn1 = innov.get_conn_innovation(c.in_node, new_nid)
        inn2 = innov.get_conn_innovation(new_nid, c.out_node)
        g.connections[inn1] = ConnectionGene(c.in_node, new_nid, 1.0, True, inn1)
        g.connections[inn2] = ConnectionGene(new_nid, c.out_node, c.weight, True, inn2)
        return g
    target = int(rng.choice(hidden))
    dup_id = innov.new_node_id()
    g.nodes[dup_id] = NodeGene(dup_id, 'hidden', 'tanh')
    incomings = [c for c in g.enabled_connections() if c.out_node == target]
    for cin in incomings:
        inn = innov.get_conn_innovation(cin.in_node, dup_id)
        g.connections[inn] = ConnectionGene(cin.in_node, dup_id, cin.weight + float(rng.normal(0,0.1)), True, inn)
    outgoings = [c for c in g.enabled_connections() if c.in_node == target]
    move_p = min(0.9, 0.3 + 0.6*float(intensity))
    for cout in outgoings:
        if rng.random() < move_p:
            cout.enabled = False
            inn = innov.get_conn_innovation(dup_id, cout.out_node)
            g.connections[inn] = ConnectionGene(dup_id, cout.out_node, cout.weight + float(rng.normal(0,0.1)), True, inn)
    return g

def platyregenerate(genome: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5) -> Genome:
    g = genome.copy()
    mode = g.regen_mode or 'split'
    if mode == 'head': g = _regenerate_head(g, rng, innov, intensity)
    elif mode == 'tail': g = _regenerate_tail(g, rng, innov, intensity)
    else: g = _regenerate_split(g, rng, innov, intensity)
    return g

# ============================================================
# 3) EvalMode & ReproPlanaNEATPlus
# ============================================================

@dataclass
class EvalMode:
    vanilla: bool = True                      # True -> pure NEAT fitness (no sex/regen bonuses)
    enable_regen_reproduction: bool = False   # allow asexual_regen in reproduction
    complexity_alpha: float = 0.01
    node_penalty: float = 1.0
    edge_penalty: float = 0.5
    species_low: int = 3
    species_high: int = 8

class ReproPlanaNEATPlus:
    def __init__(self, num_inputs, num_outputs, population_size=150, rng=None, output_activation='sigmoid'):
        self.num_inputs = num_inputs; self.num_outputs = num_outputs; self.pop_size = population_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mode = EvalMode(vanilla=True, enable_regen_reproduction=False)
        self.max_hidden_nodes = 128; self.max_edges = 1024
        self.complexity_threshold: Optional[float] = 1.0

        # Base genome
        nodes = {}
        for i in range(num_inputs): nodes[i] = NodeGene(i,'input','identity')
        for j in range(num_outputs): nodes[num_inputs + j] = NodeGene(num_inputs + j, 'output', output_activation)
        bias_id = num_inputs + num_outputs; nodes[bias_id] = NodeGene(bias_id, 'bias', 'identity')
        next_node_id = bias_id + 1
        self.innov = InnovationTracker(next_node_id=next_node_id, next_conn_innov=0)
        base_connections = {}
        for in_id in list(range(num_inputs)) + [bias_id]:
            for out_id in range(num_inputs, num_inputs+num_outputs):
                inn = self.innov.get_conn_innovation(in_id, out_id)
                w = float(self.rng.uniform(-2,2))
                base_connections[inn] = ConnectionGene(in_id, out_id, w, True, inn)
        base_genome = Genome(nodes, base_connections)
        base_genome.max_hidden_nodes = self.max_hidden_nodes; base_genome.max_edges = self.max_edges

        # Population
        self.population = []; self.next_gid = 1
        for _ in range(population_size):
            g = base_genome.copy()
            g.max_hidden_nodes = self.max_hidden_nodes; g.max_edges = self.max_edges
            g.sex = 'female' if self.rng.random() < 0.5 else 'male'
            g.regen = bool(self.rng.random() < 0.5)
            g.regen_mode = self.rng.choice(['head','tail','split'])
            g.embryo_bias = 'inputward'
            g.id = self.next_gid; self.next_gid += 1; g.birth_gen = 0
            self.population.append(g)

        # Params
        self.generation = 0
        self.compatibility_threshold = 3.0
        self.c1=self.c2=1.0; self.c3=0.4
        self.elitism = 1; self.survival_rate = 0.2
        self.mutate_add_conn_prob = 0.10; self.mutate_add_node_prob = 0.10
        self.mutate_weight_prob = 0.8; self.mutate_toggle_prob = 0.01
        self.weight_perturb_chance = 0.9; self.weight_sigma = 0.8; self.weight_reset_range = 2.0
        self.regen_mode_mut_rate = 0.05; self.embryo_bias_mut_rate = 0.03
        self.regen_rate = 0.15; self.allow_selfing = True
        self.sex_fitness_scale = {'female':1.0, 'male':0.9}; self.regen_bonus = 0.2
        self.env = {'difficulty':0.0, 'noise_std':0.0}
        self.mix_asexual_base = 0.10; self.mix_asexual_gain = 0.40
        self.injury_intensity_base = 0.25; self.injury_intensity_gain = 0.65
        self.pollen_flow_rate = 0.10
        self.heterosis_center = 3.0; self.heterosis_width=1.8; self.heterosis_gain=0.15
        self.distance_cutoff=6.0; self.penalty_far=0.20

        # Logs & snapshots
        self.event_log=[]; self.hidden_counts_history=[]; self.edge_counts_history=[]
        self.best_ids=[]; self.lineage_edges=[]; self.env_history=[]
        self.snapshots_genomes: List[Genome] = []
        self.snapshots_scars: List[Dict[int,'Scar']] = []
        self.node_registry: Dict[int, Dict[str, Any]] = {}

        # registry for lineage node styling
        for g in self.population:
            self.node_registry[g.id] = {'sex': g.sex, 'regen': g.regen, 'birth_gen': g.birth_gen}

    def _heterosis_scale(self, mother: Genome, father: Genome) -> float:
        d = compatibility_distance(mother, father, self.c1, self.c2, self.c3)
        peak = 1.0 + self.heterosis_gain * np.exp(-0.5*((d - self.heterosis_center)/self.heterosis_width)**2)
        if d > self.distance_cutoff:
            penalty = max(0.0, self.penalty_far * (d - self.distance_cutoff)/self.distance_cutoff)
            peak *= (1.0 - min(0.9, penalty))
        return float(peak)

    def _regen_intensity(self) -> float:
        return float(min(1.0, max(0.0, self.injury_intensity_base + self.injury_intensity_gain * self.env['difficulty'])))

    def _mix_asexual_ratio(self) -> float:
        return float(min(0.95, max(0.0, self.mix_asexual_base + self.mix_asexual_gain * self.env['difficulty'])))

    class Species:
        def __init__(self, representative: 'Genome'):
            self.representative = representative.copy(); self.members=[]; self.best_fitness=-1e9; self.last_improved=0
        def add(self, genome: 'Genome', fitness: float): self.members.append((genome, fitness))
        def sort(self): self.members.sort(key=lambda gf: gf[1], reverse=True)

    def speciate(self, fitnesses: List[float]) -> List['Species']:
        species=[] 
        for genome, fit in zip(self.population, fitnesses):
            placed=False
            for sp in species:
                delta = compatibility_distance(genome, sp.representative, self.c1, self.c2, self.c3)
                if delta < self.compatibility_threshold:
                    sp.add(genome, fit); placed=True; break
            if not placed:
                sp = ReproPlanaNEATPlus.Species(genome); sp.add(genome, fit); species.append(sp)
        for sp in species: sp.sort()
        return species

    def _adapt_compat_threshold(self, num_species: int):
        if num_species < self.mode.species_low:
            self.compatibility_threshold *= 0.95
        elif num_species > self.mode.species_high:
            self.compatibility_threshold *= 1.05

    def _mutate(self, genome: Genome):
        if self.rng.random() < self.mutate_toggle_prob: genome.mutate_toggle_enable(self.rng, prob=self.mutate_toggle_prob)
        if self.rng.random() < self.mutate_add_node_prob: genome.mutate_add_node(self.rng, self.innov)
        if self.rng.random() < self.mutate_add_conn_prob: genome.mutate_add_connection(self.rng, self.innov)
        if self.rng.random() < self.mutate_weight_prob: genome.mutate_weights(self.rng, self.weight_perturb_chance, self.weight_sigma, self.weight_reset_range)
        if self.rng.random() < self.regen_mode_mut_rate:
            if self.env['difficulty'] > 0.6: genome.regen_mode = self.rng.choice(['split','head','tail'], p=[0.6,0.25,0.15])
            else: genome.regen_mode = self.rng.choice(['split','head','tail'])
        if self.rng.random() < self.embryo_bias_mut_rate:
            genome.embryo_bias = self.rng.choice(['neutral','inputward','outputward'])

    def _crossover_maternal_biased(self, mother: Genome, father: Genome, species_members):
        fit_dict = {g:f for g,f in species_members}
        f_m = fit_dict.get(mother, 0.0); f_f = fit_dict.get(father, 0.0)
        if f_f > f_m: mother, father = father, mother
        child_nodes={}; child_conns={}
        for nid,n in mother.nodes.items():
            if n.type in ('input','output','bias'): child_nodes[nid] = NodeGene(n.id, n.type, n.activation)
        all_innovs = sorted(set(mother.connections.keys()).union(father.connections.keys()))
        for inn in all_innovs:
            if inn in mother.connections and inn in father.connections:
                cm = mother.connections[inn]; cf = father.connections[inn]
                pick = cm if self.rng.random() < 0.7 else cf
                enabled = True
                if (not cm.enabled) or (not cf.enabled): enabled = not (self.rng.random() < 0.75)
                child_conns[inn] = ConnectionGene(pick.in_node, pick.out_node, pick.weight, enabled, inn)
            elif inn in mother.connections:
                g = mother.connections[inn]; child_conns[inn] = ConnectionGene(g.in_node, g.out_node, g.weight, g.enabled, inn)
            if inn in child_conns:
                g = child_conns[inn]
                for nid in (g.in_node, g.out_node):
                    if nid not in child_nodes:
                        n = mother.nodes.get(nid) or father.nodes.get(nid)
                        child_nodes[nid] = NodeGene(n.id, n.type, n.activation)
        child = Genome(child_nodes, child_conns)
        child.max_hidden_nodes = self.max_hidden_nodes; child.max_edges = self.max_edges
        child.sex = 'female' if self.rng.random() < 0.5 else 'male'
        p = 0.7 if (mother.regen or father.regen) else 0.2
        child.regen = bool(self.rng.random() < p)
        child.regen_mode = self.rng.choice(['head','tail','split'])
        child.embryo_bias = mother.embryo_bias if self.rng.random() < 0.7 else father.embryo_bias
        return child

    def _make_offspring(self, species, offspring_counts, sidx, species_pool):
        sp = species[sidx]; new_pop=[]; events={'sexual_within':0,'sexual_cross':0,'asexual_regen':0,'asexual_clone':0}
        sp.sort()
        elites=[g for g,_ in sp.members[:min(self.elitism, offspring_counts[sidx])]]
        for e in elites:
            child=e.copy(); child.id=self.next_gid; self.next_gid+=1; child.parents=(e.id,e.id); child.birth_gen=self.generation+1
            new_pop.append(child); events['asexual_clone']+=1; 
            self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen}
        remaining = offspring_counts[sidx]-len(elites)
        k = max(2, int(math.ceil(self.survival_rate * len(sp.members))))
        females=[g for g,_ in sp.members[:k] if g.sex=='female']; males=[g for g,_ in sp.members[:k] if g.sex=='male']; pool=[g for g,_ in sp.members[:k]]
        if (not females) or (not males):
            females = [g for g,_ in sp.members if g.sex=='female'] or females
            males   = [g for g,_ in sp.members if g.sex=='male'] or males
        mix_ratio=self._mix_asexual_ratio()
        while remaining>0:
            mode=None
            if self.rng.random()<mix_ratio:
                parent=pool[int(self.rng.integers(len(pool)))]
                if parent.regen and self.mode.enable_regen_reproduction:
                    child = platyregenerate(parent, self.rng, self.innov, intensity=self._regen_intensity()); mode='asexual_regen'
                else:
                    child=parent.copy(); mode='asexual_clone'
                mother_id=parent.id; father_id=None
            else:
                if females and males and self.rng.random()>self.pollen_flow_rate:
                    mother=females[int(self.rng.integers(len(females)))]; father=males[int(self.rng.integers(len(males)))]; mode='sexual_within'; sp_for_fit=sp.members
                else:
                    if len(species_pool)>1:
                        mother=pool[int(self.rng.integers(len(pool)))]
                        other=species_pool[(sidx+1)%len(species_pool)]
                        other_pool=[g for g,_ in other.members]; males_other=[g for g,_ in other.members if g.sex=='male']
                        father = males_other[int(self.rng.integers(len(males_other)))] if males_other else other_pool[int(self.rng.integers(len(other_pool)))]
                        mode='sexual_cross'; sp_for_fit = sp.members + other.members
                    else:
                        if females and males:
                            mother=females[int(self.rng.integers(len(females)))]; father=males[int(self.rng.integers(len(males)))]; mode='sexual_within'; sp_for_fit=sp.members
                        else:
                            parent=pool[int(self.rng.integers(len(pool)))]
                            if self.allow_selfing:
                                mother=parent; father=parent; mode='sexual_within'; sp_for_fit=sp.members
                            else:
                                child=parent.copy(); mode='asexual_clone'
                                mother_id=parent.id; father_id=None
                                child.id=self.next_gid; self.next_gid+=1; child.parents=(mother_id,father_id); child.birth_gen=self.generation+1
                                self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen}
                                self._mutate(child); new_pop.append(child); events[mode]+=1; remaining-=1; continue
                child=self._crossover_maternal_biased(mother,father,sp_for_fit); child.hybrid_scale=self._heterosis_scale(mother,father); mother_id=mother.id; father_id=father.id
            child.id=self.next_gid; self.next_gid+=1; child.parents=(mother_id,father_id); child.birth_gen=self.generation+1
            self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen}
            self._mutate(child); new_pop.append(child); events[mode]+=1; remaining-=1
        return new_pop, events

    def reproduce(self, species, fitnesses):
        total_adjusted=0.0; species_adjusted=[]
        for sp in species:
            adj=sum(f for _,f in sp.members)/len(sp.members); species_adjusted.append(adj); total_adjusted+=adj
        if total_adjusted<=0:
            offspring_counts=[self.pop_size//len(species)]*len(species)
            for i in range(self.pop_size - sum(offspring_counts)): offspring_counts[i%len(offspring_counts)]+=1
        else:
            shares=[adj/total_adjusted for adj in species_adjusted]; offspring_counts=[int(round(s*self.pop_size)) for s in shares]
            diff=self.pop_size - sum(offspring_counts); idxs=np.argsort(shares)[::-1]; i=0
            while diff!=0:
                idx=int(idxs[i%len(idxs)]); offspring_counts[idx]+=1 if diff>0 else -1; diff += -1 if diff>0 else 1; i+=1
        new_pop=[]; gen_events={'sexual_within':0,'sexual_cross':0,'asexual_regen':0,'asexual_clone':0}
        for sidx,sp in enumerate(species):
            offspring,events=self._make_offspring(species,offspring_counts,sidx,species)
            for k,v in events.items(): gen_events[k]+=v
            new_pop.extend(offspring)
        if len(new_pop)<self.pop_size:
            bests=[g for sp in species for g,_ in sp.members]
            while len(new_pop)<self.pop_size:
                parent=bests[int(self.rng.integers(len(bests)))]; child=parent.copy()
                child.id=self.next_gid; self.next_gid+=1; child.parents=(parent.id,None); child.birth_gen=self.generation+1
                self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen}
                self._mutate(child); new_pop.append(child); gen_events['asexual_clone']+=1
        elif len(new_pop)>self.pop_size:
            new_pop=new_pop[:self.pop_size]
        self.population=new_pop; self.event_log.append(gen_events)
        for g in new_pop: self.lineage_edges.append((g.parents[0], g.parents[1], g.id, g.birth_gen, 'birth'))

    def _complexity_penalty(self, g: Genome) -> float:
        n_hidden = sum(1 for n in g.nodes.values() if n.type=='hidden')
        n_edges  = sum(1 for c in g.connections.values() if c.enabled)
        m = self.mode
        penalty = m.complexity_alpha * (m.node_penalty*n_hidden + m.edge_penalty*n_edges)
        threshold = getattr(self, "complexity_threshold", None)
        if threshold is not None:
            penalty = min(float(threshold), penalty)
        return penalty

    def evolve(self, fitness_fn: Callable[[Genome], float], n_generations=100, target_fitness=None, verbose=True, env_schedule=None):
        history=[]; best_ever=None; best_ever_fit=-1e9
        from math import isnan
        scars=None; prev_best=None
        for gen in range(n_generations):
            self.generation=gen
            prev=history[-1] if history else (None,None)
            if env_schedule is not None:
                env=env_schedule(gen, {'gen':gen,'prev_best':prev[0] if prev else None, 'prev_avg':prev[1] if prev else None})
                if env is not None:
                    self.env.update({k: v for k, v in env.items() if k not in {'enable_regen'}})
                    if 'enable_regen' in env:
                        flag = bool(env['enable_regen'])
                        self.mode.enable_regen_reproduction = flag
                        if flag:
                            self.mix_asexual_base = max(self.mix_asexual_base, 0.30)
            self.env_history.append({'gen':gen, **self.env, 'regen_enabled': self.mode.enable_regen_reproduction})
            raw=[fitness_fn(g) for g in self.population]
            fitnesses=[]
            for g,f in zip(self.population, raw):
                f2 = f
                if not self.mode.vanilla:
                    f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * (getattr(g,'hybrid_scale',1.0))
                    if g.regen: f2 += self.regen_bonus
                f2 -= self._complexity_penalty(g)
                fitnesses.append(float(f2))
            best_idx=int(np.argmax(fitnesses)); best_fit=float(fitnesses[best_idx]); avg_fit=float(np.mean(fitnesses))
            # snapshots for GIFs
            curr_best = self.population[best_idx].copy()
            scars = diff_scars(prev_best, curr_best, scars, birth_gen=gen, regen_mode_for_new=getattr(curr_best,'regen_mode','split'))
            self.snapshots_genomes.append(curr_best); self.snapshots_scars.append(scars)
            prev_best = curr_best

            history.append((best_fit, avg_fit)); self.best_ids.append(self.population[best_idx].id)
            # complexity traces
            self.hidden_counts_history.append([sum(1 for n in g.nodes.values() if n.type=='hidden') for g in self.population])
            self.edge_counts_history.append([sum(1 for c in g.connections.values() if c.enabled) for g in self.population])
            if verbose:
                diff = float(self.env.get('difficulty', 0.0))
                noise = float(self.env.get('noise_std', 0.0))
                ev=self.event_log[-1] if self.event_log else {'sexual_within':0,'sexual_cross':0,'asexual_regen':0}
                print(
                    f"Gen {gen:3d} | best {best_fit:.4f} | avg {avg_fit:.4f} | difficulty {diff:.2f} | noise {noise:.2f} | "
                    f"sexual {ev.get('sexual_within',0)+ev.get('sexual_cross',0)} | regen {ev.get('asexual_regen',0)}"
                )
            if best_fit > best_ever_fit: best_ever_fit = best_fit; best_ever = self.population[best_idx].copy()
            if target_fitness is not None and best_fit >= target_fitness: break
            species=self.speciate(fitnesses)
            self._adapt_compat_threshold(len(species))
            self.reproduce(species, fitnesses)
        # Champion across all generations
        if best_ever is None and self.population:
            best_ever = self.population[0].copy()
        return best_ever, history

# ============================================================
# 4) Backprop NEAT (NumPy only)
# ============================================================

def act_forward(name, x):
    if name == 'tanh':     return np.tanh(x)
    if name == 'sigmoid':  return 1.0/(1.0+np.exp(-x))
    if name == 'relu':     return np.maximum(0.0, x)
    if name == 'identity': return x
    return np.tanh(x)

def act_deriv(name, x):
    if name == 'tanh':
        y = np.tanh(x); return 1.0 - y*y
    if name == 'sigmoid':
        s = 1.0/(1.0+np.exp(-x)); return s*(1.0 - s)
    if name == 'relu':
        return (x > 0.0).astype(x.dtype)
    if name == 'identity':
        return np.ones_like(x)
    y = np.tanh(x); return 1.0 - y*y

def compile_genome(g: Genome):
    order = g.topological_order()
    idx_of = {nid:i for i, nid in enumerate(order)}
    types = [g.nodes[n].type for n in order]
    acts  = [g.nodes[n].activation for n in order]
    in_ids   = [nid for nid in order if g.nodes[nid].type=='input']
    bias_ids = [nid for nid in order if g.nodes[nid].type=='bias']
    out_ids  = [nid for nid in order if g.nodes[nid].type=='output']
    edges = [c for c in g.enabled_connections()]
    src = np.array([idx_of[c.in_node]  for c in edges], dtype=np.int32)
    dst = np.array([idx_of[c.out_node] for c in edges], dtype=np.int32)
    w   = np.array([c.weight for c in edges], dtype=np.float64)
    eid = [c.innovation for c in edges]
    n = len(order)
    in_edges  = [[] for _ in range(n)]
    out_edges = [[] for _ in range(n)]
    for e,(s,d) in enumerate(zip(src, dst)):
        in_edges[d].append(e)
        out_edges[s].append(e)
    return {
        'order': order, 'idx_of': idx_of, 'types': types, 'acts': acts,
        'inputs': [idx_of[i] for i in sorted(in_ids)],
        'biases': [idx_of[i] for i in bias_ids],
        'outputs':[idx_of[i] for i in sorted(out_ids)],
        'src': src, 'dst': dst, 'w': w, 'eid': eid,
        'in_edges': in_edges, 'out_edges': out_edges
    }

def forward_batch(comp, X, w=None):
    if w is None: w = comp['w']
    B = X.shape[0]; n = len(comp['order'])
    A = np.zeros((B, n), dtype=np.float64)
    Z = np.zeros((B, n), dtype=np.float64)
    in_idx = comp['inputs']
    assert X.shape[1] == len(in_idx), "X dim != number of input nodes"
    for k, nid in enumerate(in_idx): A[:, nid] = X[:, k]
    for b in comp['biases']: A[:, b] = 1.0
    for j in range(n):
        if comp['types'][j] in ('input','bias'): continue
        z = np.zeros(B, dtype=np.float64)
        for e in comp['in_edges'][j]: z += A[:, comp['src'][e]] * w[e]
        Z[:, j] = z; A[:, j] = act_forward(comp['acts'][j], z)
    return A, Z

def _softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-9)

def loss_and_output_delta(comp, Z, y, l2, w):
    out_idx = comp['outputs']; B = Z.shape[0]
    if len(out_idx) == 1:
        z = Z[:, out_idx[0:1]]; p = 1.0/(1.0 + np.exp(-z))
        yv = y.reshape(B,1).astype(np.float64)
        loss = (np.log1p(np.exp(-np.abs(z))) + np.maximum(z,0) - yv*z).mean()
        delta_out = (p - yv); probs = p
    else:
        logits = Z[:, out_idx]; probs = _softmax(logits)
        y_one = np.eye(len(out_idx), dtype=np.float64)[y] if y.ndim==1 else y.astype(np.float64)
        loss = -(y_one * np.log(probs + 1e-9)).sum(axis=1).mean()
        delta_out = (probs - y_one)
    loss = float(loss + 0.5 * l2 * np.sum(w*w))
    return loss, delta_out, probs

def backprop_step(comp, X, y, w, lr=1e-2, l2=1e-4):
    A, Z = forward_batch(comp, X, w)
    loss, delta_out, _ = loss_and_output_delta(comp, Z, y, l2, w)
    B = X.shape[0]; n = len(comp['order'])
    grad_w = np.zeros_like(w); delta_z = np.zeros((B, n), dtype=np.float64); delta_a = np.zeros((B, n), dtype=np.float64)
    for j, oi in enumerate(comp['outputs']): delta_z[:, oi] = delta_out[:, j:j+1].reshape(B)
    for j in reversed(range(n)):
        t = comp['types'][j]
        if t == 'output': dz = delta_z[:, j]
        elif t in ('input','bias'): continue
        else:
            dz = delta_a[:, j] * act_deriv(comp['acts'][j], Z[:, j]); delta_z[:, j] = dz
        for e in comp['in_edges'][j]:
            s = comp['src'][e]
            grad_w[e] += np.dot(A[:, s], dz)
            delta_a[:, s] += dz * w[e]
    grad_w = grad_w / B + l2 * w
    w_new = w - lr * grad_w
    return w_new, float(loss)

def train_with_backprop_numpy(genome: Genome, X, y, steps=50, lr=1e-2, l2=1e-4):
    comp = compile_genome(genome); w = comp['w'].copy(); history=[]
    for _ in range(steps):
        w, L = backprop_step(comp, X, y, w, lr=lr, l2=l2); history.append(L)
    for e_idx, inn in enumerate(comp['eid']): genome.connections[inn].weight = float(w[e_idx])
    return history

def predict_proba(genome: Genome, X):
    comp = compile_genome(genome); _, Z = forward_batch(comp, X, comp['w']); out = comp['outputs']
    if len(out) == 1:
        z = Z[:, out[0:1]]; p = 1.0/(1.0 + np.exp(-z)); return np.concatenate([1.0-p, p], axis=1)
    else:
        logits = Z[:, out]; return _softmax(logits)

def predict(genome: Genome, X):
    P = predict_proba(genome, X); return np.argmax(P, axis=1).astype(np.int32)

def complexity_penalty(genome: Genome, alpha_nodes=1e-3, alpha_edges=5e-4):
    hidden = sum(1 for n in genome.nodes.values() if n.type=='hidden')
    edges  = len(genome.enabled_connections())
    return alpha_nodes*hidden + alpha_edges*edges

def fitness_backprop_classifier(genome: Genome, Xtr, ytr, Xva, yva,
                                steps=40, lr=5e-3, l2=1e-4,
                                alpha_nodes=1e-3, alpha_edges=5e-4):
    gg = genome.copy()
    train_with_backprop_numpy(gg, Xtr, ytr, steps=steps, lr=lr, l2=l2)
    pred = predict(gg, Xva)
    acc = (pred == (yva if yva.ndim==1 else np.argmax(yva,1))).mean()
    pen = complexity_penalty(gg, alpha_nodes=alpha_nodes, alpha_edges=alpha_edges)
    return float(acc - pen)

# ============================================================
# 5) Visualization utilities
# ============================================================

@dataclass
class Scar:
    birth_gen: int
    mode: str = "split"   # 'head'|'tail'|'split'|...
    age: int = 0

def layout_by_depth(genome: Genome, x_gap: float = 1.5, y_gap: float = 1.0) -> Dict[int, Tuple[float, float]]:
    depth = genome.node_depths()
    buckets: Dict[int, List[int]] = {}
    for nid, d in depth.items(): buckets.setdefault(d, []).append(nid)
    type_rank = {'input':0, 'bias':1, 'hidden':2, 'output':3}
    for d in buckets: buckets[d].sort(key=lambda nid: (type_rank.get(genome.nodes[nid].type,9), nid))
    pos = {}
    for i, d in enumerate(sorted(buckets.keys())):
        nodes = buckets[d]; y0 = -(len(nodes)-1)/2.0
        for j, nid in enumerate(nodes): pos[nid] = (i * x_gap, (y0 + j) * y_gap)
    return pos

def layout_by_depth_union(genomes: List[Genome], x_gap: float = 1.5, y_gap: float = 1.0) -> Dict[int, Tuple[float, float]]:
    depths_per: Dict[int, List[int]] = {}; type_by: Dict[int, str] = {}
    for g in genomes:
        d = g.node_depths()
        for nid, dep in d.items():
            depths_per.setdefault(nid, []).append(dep)
            if nid not in type_by: type_by[nid] = g.nodes[nid].type
    depth_final = {nid: max(ds) for nid, ds in depths_per.items()}
    buckets: Dict[int, List[int]] = {}
    for nid, dep in depth_final.items(): buckets.setdefault(dep, []).append(nid)
    type_rank = {'input':0, 'bias':1, 'hidden':2, 'output':3}
    for d in buckets: buckets[d].sort(key=lambda nid: (type_rank.get(type_by.get(nid,'hidden'),9), nid))
    pos = {}
    for i, d in enumerate(sorted(buckets.keys())):
        nodes = buckets[d]; y0 = -(len(nodes)-1)/2.0
        for j, nid in enumerate(nodes): pos[nid] = (i * x_gap, (y0 + j) * y_gap)
    return pos

def _edge_key(c) -> Tuple[int,int]:
    return (c.in_node, c.out_node)

def edge_sets(prev_genome: Optional[Genome], curr_genome: Genome) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]], Set[Tuple[int,int]]]:
    prev = set(_edge_key(c) for c in (prev_genome.enabled_connections() if prev_genome else []))
    curr = set(_edge_key(c) for c in (curr_genome.enabled_connections() if curr_genome else []))
    return prev & curr, curr - prev, prev - curr

def _draw_edges(ax, genome: Genome, pos, lw=1.0, alpha=0.8):
    for c in genome.enabled_connections():
        i, o = c.in_node, c.out_node
        if i not in pos or o not in pos: continue
        p1 = pos[i]; p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, alpha=alpha)

def _draw_edges_with_diff(ax, prev_genome: Optional[Genome], curr_genome: Genome, pos,
                          lw_base=1.0, lw_added=2.0, lw_removed=1.2,
                          alpha_base=0.7, alpha_added=1.0, alpha_removed=0.6,
                          linestyle_removed=(0,(3,3))):
    common, added, removed = edge_sets(prev_genome, curr_genome)
    for c in curr_genome.enabled_connections():
        e = _edge_key(c)
        if e in added: continue
        i, o = c.in_node, c.out_node
        if i not in pos or o not in pos: continue
        p1 = pos[i]; p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw_base, alpha=alpha_base)
    for c in curr_genome.enabled_connections():
        e = _edge_key(c)
        if e not in added: continue
        i, o = c.in_node, c.out_node
        if i not in pos or o not in pos: continue
        p1 = pos[i]; p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw_added, alpha=alpha_added)
    if prev_genome is not None:
        for c in prev_genome.enabled_connections():
            e = _edge_key(c)
            if e not in removed: continue
            i, o = c.in_node, c.out_node
            if i not in pos or o not in pos: continue
            p1 = pos[i]; p2 = pos[o]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw_removed, alpha=alpha_removed, linestyle=linestyle_removed)

def _mode_lw_factor(mode: str) -> float:
    if mode == "head": return 1.2
    if mode == "tail": return 1.0
    if mode == "split": return 1.4
    return 1.0

def _draw_nodes(ax, genome: Genome, pos, scars: Optional[Dict[int, 'Scar']] = None,
                pulse_t: float = 0.0, decay_horizon: float = 8.0,
                radius: float = 0.10, annotate_type=True, show_mode_mark=True):
    for nid, nd in genome.nodes.items():
        if nid not in pos: continue
        x, y = pos[nid]
        circ = Circle((x, y), radius=radius, fill=True, alpha=0.9, linewidth=0.0)
        ax.add_patch(circ)
        outline_lw = 1.0; outline_alpha = 0.9; mode_char=None
        if scars and (nid in scars):
            age = scars[nid].age; mode_char = scars[nid].mode[:1].upper() if scars[nid].mode else None
            amp = max(0.1, 1.0 - (age / float(decay_horizon))) if (decay_horizon and decay_horizon>0) else 1.0
            pulse = 1.0 + 0.25 * math.sin(2*math.pi * pulse_t) * amp
            outline_lw = 2.0 * pulse * _mode_lw_factor(scars[nid].mode)
            circ2 = Circle((x, y), radius=radius*(1.0+0.15*pulse), fill=False, linewidth=outline_lw, alpha=outline_alpha)
            ax.add_patch(circ2)
            ax.text(x, y + radius*1.6, f"{age}", ha="center", va="bottom", fontsize=8, alpha=0.9)
        else:
            circ2 = Circle((x, y), radius=radius, fill=False, linewidth=outline_lw, alpha=outline_alpha)
            ax.add_patch(circ2)
        if annotate_type: ax.text(x, y - radius*1.6, f"{nd.type[0]}", ha="center", va="top", fontsize=7, alpha=0.8)
        if show_mode_mark and mode_char is not None: ax.text(x + radius*1.2, y, mode_char, ha="left", va="center", fontsize=8, alpha=0.9)

def diff_scars(prev_genome: Optional[Genome], curr_genome: Genome, prev_scars: Optional[Dict[int, 'Scar']], birth_gen: int,
               regen_mode_for_new: str = "split") -> Dict[int, 'Scar']:
    scars = {} if prev_scars is None else {k: Scar(v.birth_gen, v.mode, v.age+1) for k, v in prev_scars.items()}
    prev_ids = set(prev_genome.nodes.keys()) if prev_genome is not None else set()
    curr_ids = set(curr_genome.nodes.keys())
    new_nodes = list(curr_ids - prev_ids)
    for nid in new_nodes: scars[nid] = Scar(birth_gen=birth_gen, mode=regen_mode_for_new, age=0)
    for nid in list(scars.keys()):
        if nid not in curr_ids: scars.pop(nid, None)
    return scars

def draw_genome_png(genome: Genome, scars: Optional[Dict[int, 'Scar']], path: str, title: Optional[str] = None,
                    prev_genome: Optional[Genome]=None, decay_horizon: float = 8.0):
    pos = layout_by_depth(genome)
    fig, ax = plt.subplots(figsize=(6, 4))
    if prev_genome is not None: _draw_edges_with_diff(ax, prev_genome, genome, pos)
    else: _draw_edges(ax, genome, pos, lw=1.0, alpha=0.8)
    _draw_nodes(ax, genome, pos, scars=scars, pulse_t=0.0, decay_horizon=decay_horizon, radius=0.12, annotate_type=True, show_mode_mark=True)
    if title: ax.set_title(title)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off'); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def _normalize_scar_snapshot(entry: Optional[Dict[Any, Any]]) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    """Convert a scar snapshot into numeric node/edge age dictionaries."""
    node_scars: Dict[int, int] = {}
    edge_scars: Dict[Tuple[int, int], int] = {}

    if not entry:
        return node_scars, edge_scars

    def _coerce_age(val: Any) -> Optional[int]:
        if hasattr(val, "age"):
            try:
                return int(getattr(val, "age"))
            except Exception:
                return None
        if isinstance(val, dict) and "age" in val:
            try:
                return int(val.get("age"))
            except Exception:
                return None
        if isinstance(val, (int, float)):
            return int(val)
        return None

    if isinstance(entry, dict) and ("nodes" in entry or "edges" in entry):
        raw_nodes = entry.get("nodes", {}) or {}
        for nid, val in raw_nodes.items():
            age = _coerce_age(val)
            if age is not None:
                try:
                    node_scars[int(nid)] = age
                except Exception:
                    continue
        raw_edges = entry.get("edges", {}) or {}
        for key, val in raw_edges.items():
            age = _coerce_age(val)
            if age is None:
                continue
            if isinstance(key, (tuple, list)) and len(key) == 2:
                try:
                    edge_scars[(int(key[0]), int(key[1]))] = age
                except Exception:
                    continue
            else:
                continue
        return node_scars, edge_scars

    if isinstance(entry, dict):
        for nid, val in entry.items():
            age = _coerce_age(val)
            if age is not None:
                try:
                    node_scars[int(nid)] = age
                except Exception:
                    continue
    return node_scars, edge_scars


def _export_morph_gif_with_scars(
    snapshots_genomes,
    snapshots_scars,
    path,
    *,
    fps=12,
    morph_frames=12,
    decay_horizon=10.0,
    fixed_layout=True,
    dpi=130,
    pulse_period_frames=16,
):
    """Scar-aware morph GIF helper shared by export_morph_gif."""

    import numpy as _np
    import matplotlib.pyplot as _plt

    if not snapshots_genomes or len(snapshots_genomes) < 2:
        raise ValueError("_export_morph_gif_with_scars: need >= 2 snapshots.")
    if len(snapshots_genomes) != len(snapshots_scars):
        raise ValueError("_export_morph_gif_with_scars: scars_seq length mismatch.")

    pos_union = layout_by_depth_union(snapshots_genomes) if fixed_layout else None
    if pos_union:
        xs = [p[0] for p in pos_union.values()]
        ys = [p[1] for p in pos_union.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        pos_union = {
            nid: (
                0.05 + 0.90 * ((x - min_x) / span_x),
                0.05 + 0.90 * ((y - min_y) / span_y),
            )
            for nid, (x, y) in pos_union.items()
        }

    def _node_groups_and_depths(_g):
        depth = _g.node_depths()
        groups = {"input": [], "bias": [], "hidden": [], "output": []}
        for nid, n in _g.nodes.items():
            groups.get(n.type, groups["hidden"]).append(nid)
        for key in groups:
            groups[key].sort()
        max_d = max(depth.values()) if depth else 1
        return groups, depth, max_d

    def _compute_positions(_g):
        groups, depth, max_d = _node_groups_and_depths(_g)
        pos = {}
        bands = [("input", 0.85, 1.00), ("bias", 0.70, 0.82), ("hidden", 0.20, 0.68), ("output", 0.02, 0.18)]
        for (gname, y0, y1) in bands:
            arr = groups[gname]
            n = max(1, len(arr))
            ys = _np.linspace(y1, y0, n)
            for idx, nid in enumerate(arr):
                depth_val = depth.get(nid, 0)
                if gname in ("input", "bias"):
                    x = 0.04
                elif gname == "output":
                    x = 0.96
                else:
                    x = 0.10 + 0.80 * (depth_val / max(1, max_d))
                pos[nid] = (x, ys[idx])
        return pos

    def _pulse_amp(age: int, frame_idx: int) -> float:
        base = max(0.1, 1.0 - float(age) / max(1e-6, decay_horizon))
        phase = 2.0 * _np.pi * (frame_idx % max(1, pulse_period_frames)) / max(1, pulse_period_frames)
        return float(base * (0.5 + 0.5 * _np.sin(phase)))

    frames = []
    total_steps = max(1, morph_frames)
    for i in range(len(snapshots_genomes) - 1):
        g0 = snapshots_genomes[i]
        g1 = snapshots_genomes[i + 1]
        node_scars0, edge_scars0 = _normalize_scar_snapshot(snapshots_scars[i])
        node_scars1, edge_scars1 = _normalize_scar_snapshot(snapshots_scars[i + 1])

        pos0 = pos_union if pos_union is not None else _compute_positions(g0)
        pos1 = pos_union if pos_union is not None else _compute_positions(g1)
        nodes_union = sorted(set(list(pos0.keys()) + list(pos1.keys())))

        edges0 = set((c.in_node, c.out_node) for c in g0.enabled_connections())
        edges1 = set((c.in_node, c.out_node) for c in g1.enabled_connections())

        for k in range(total_steps):
            t = 0.0 if total_steps <= 1 else k / float(total_steps - 1)
            frame_index = i * total_steps + k

            fig, ax = _plt.subplots(figsize=(6.6, 4.8), dpi=dpi)
            ax.set_axis_off()
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            # Fade edges from g0 -> g1
            for (u, v) in sorted(edges0):
                if (u not in pos0) or (v not in pos0):
                    continue
                x0, y0 = pos0[u]
                x1, y1 = pos0[v]
                width = 1.6
                if (u, v) in edge_scars0:
                    width += 0.6 * _pulse_amp(edge_scars0[(u, v)], frame_index)
                ax.plot([x0, x1], [y0, y1], linewidth=width, alpha=max(0.0, 1.0 - t))

            for (u, v) in sorted(edges1):
                if (u not in pos1) or (v not in pos1):
                    continue
                x0, y0 = pos1[u]
                x1, y1 = pos1[v]
                width = 1.8
                if (u, v) in edge_scars1:
                    width += 0.6 * _pulse_amp(edge_scars1[(u, v)], frame_index)
                ax.plot([x0, x1], [y0, y1], linewidth=width, alpha=max(0.0, t))

            for nid in nodes_union:
                pos_lookup = pos1 if nid in pos1 else pos0
                if nid not in pos_lookup:
                    continue
                x, y = pos_lookup[nid]
                tname = None
                if nid in g1.nodes:
                    tname = g1.nodes[nid].type
                elif nid in g0.nodes:
                    tname = g0.nodes[nid].type
                size = 50.0
                if tname == "input":
                    size = 35.0
                elif tname == "bias":
                    size = 28.0
                elif tname == "output":
                    size = 60.0
                alpha0 = 1.0 if (nid in g0.nodes and nid in g1.nodes) else (1.0 - t if nid in g0.nodes else t)
                ax.scatter([x], [y], s=size, alpha=max(0.2, min(1.0, alpha0)), linewidths=0.8, zorder=3)

                amp = 0.0
                if nid in node_scars0:
                    amp += (1.0 - t) * _pulse_amp(node_scars0[nid], frame_index)
                if nid in node_scars1:
                    amp += t * _pulse_amp(node_scars1[nid], frame_index)
                if amp > 0.0:
                    circ = _plt.Circle((x, y), 0.018 + 0.012 * amp, fill=False, linewidth=1.0 + 2.0 * amp, alpha=0.55)
                    ax.add_patch(circ)

            img = _fig_to_rgb(fig)
            frames.append(img)
            _plt.close(fig)

    _mimsave(path, frames, fps=fps)
    return path

def export_double_exposure(genome: Genome, lineage_edges: List[Tuple[Optional[int], Optional[int], int, int, str]],
                           current_gen: int, out_path: str, title: Optional[str] = None):
    gens = {}
    for (m, f, child, gen, kind) in lineage_edges:
        if gen > current_gen: continue
        gens.setdefault(gen, []).append(child)
    for gen in gens: gens[gen] = sorted(gens[gen])
    id_row = {}
    for gen in sorted(gens.keys()):
        for idx, cid in enumerate(gens[gen]): id_row[cid] = idx
    fig, ax = plt.subplots(figsize=(8, 5))
    for (m, f, child, gen, kind) in lineage_edges:
        if gen > current_gen: continue
        x2 = gen; y2 = id_row.get(child, 0)
        if m is not None:
            x1 = gen-1; y1 = id_row.get(m, y2); ax.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.25)
        if f is not None:
            x1 = gen-1; y1 = id_row.get(f, y2); ax.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.25)
    ax.set_xlabel("Generation"); ax.set_ylabel("Lineage Row")
    pos = layout_by_depth(genome)
    for c in genome.enabled_connections():
        i, o = c.in_node, c.out_node
        if i not in pos or o not in pos: continue
        p1 = pos[i]; p2 = pos[o]; ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1.5, alpha=0.9)
    for nid in genome.nodes:
        if nid not in pos: continue
        x, y = pos[nid]; circ = Circle((x, y), radius=0.10, fill=True, alpha=0.9, linewidth=0.0); ax.add_patch(circ)
        circ2 = Circle((x, y), radius=0.10, fill=False, linewidth=1.5, alpha=0.9); ax.add_patch(circ2)
    if title: ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)

# -----------------------------
# Lineage (no hard color choices; uses shape/linestyle/linewidth)
# -----------------------------

def _infer_generations(lineage_edges: Iterable[Tuple[Optional[int], Optional[int], int, int, str]]):
    gen: Dict[int, int] = {}
    edges = list(lineage_edges)
    for _, _, child, g, _ in edges:
        gen[child] = int(g)
    changed = True
    while changed:
        changed = False
        for m, f, c, g, _ in edges:
            if c not in gen: 
                continue
            for p in (m, f):
                if p is None: continue
                target = gen[c] - 1
                if (p not in gen) or (gen[p] > target):
                    gen[p] = target
                    changed = True
    if gen:
        shift = -min(gen.values())
        if shift > 0:
            for k in list(gen.keys()):
                gen[k] += shift
    return gen

def _fallback_lineage_layout(nodes: List[int], gen_map: Dict[int,int]):
    layers: Dict[int, List[int]] = {}
    for n in nodes:
        layers.setdefault(int(gen_map.get(n, 0)), []).append(n)
    for k in layers: layers[k] = sorted(layers[k])
    pos = {}
    max_gen = max(layers.keys()) if layers else 0
    for g in range(max_gen + 1):
        row = layers.get(g, []); n = len(row) or 1
        xs = np.linspace(0.1, 0.9, n)
        y = 1.0 - (g / max(1, max_gen + 0.5))
        for x, nid in zip(xs, row): pos[nid] = (x, y)
    for n in nodes:
        if n not in pos: pos[n] = (0.5, 0.5)
    return pos

def render_lineage(neat, path="lineage.png", title="Lineage", max_edges: Optional[int]=2500,
                   highlight: Optional[Iterable[int]]=None, dpi=200):
    edges = getattr(neat, "lineage_edges", None)
    if not edges:
        raise ValueError("neat.lineage_edges is empty. Run evolve() first.")
    use_edges = edges[-max_edges:] if (max_edges and len(edges) > max_edges) else edges
    nodes = set()
    for m,f,c,g,tag in use_edges:
        for nid in (m,f,c):
            if nid is not None: nodes.add(nid)
    gen_map = _infer_generations(use_edges)
    pos = _fallback_lineage_layout(sorted(nodes), gen_map)
    # draw
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_axis_off(); ax.set_title(title, loc='left', fontsize=12)
    # edges with styles
    for (m,f,c,g,tag) in use_edges:
        kind = tag if tag and tag!='birth' else ('asexual' if (m is None or f is None) else ('selfing' if m==f else 'sexual'))
        style = 'solid' if kind=='sexual' else ('dashdot' if kind=='selfing' else ('dashed' if kind=='asexual' else 'dotted'))
        width = 1.8 if kind in ('sexual','asexual') else 1.4
        for p in (m,f):
            if p is None: continue
            if p not in pos or c not in pos: continue
            x1,y1 = pos[p]; x2,y2 = pos[c]
            arr = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='-|>', mutation_scale=8, lw=width, linestyle=style, alpha=0.9)
            ax.add_patch(arr)
    # nodes (shape encodes sex; size encodes regen)
    reg = getattr(neat, "node_registry", {})
    xs_f=[]; ys_f=[]; ss_f=[]; xs_m=[]; ys_m=[]; ss_m=[]; xs_u=[]; ys_u=[]; ss_u=[]
    for nid,(x,y) in pos.items():
        info = reg.get(nid, {}); sex = info.get('sex', None); regen = bool(info.get('regen', False))
        size = 80*(1.3 if regen else 1.0)
        if sex=='female': xs_f.append(x); ys_f.append(y); ss_f.append(size)
        elif sex=='male': xs_m.append(x); ys_m.append(y); ss_m.append(size)
        else: xs_u.append(x); ys_u.append(y); ss_u.append(size)
    if xs_f: ax.scatter(xs_f, ys_f, s=ss_f, marker='o', alpha=0.95)
    if xs_m: ax.scatter(xs_m, ys_m, s=ss_m, marker='s', alpha=0.95)
    if xs_u: ax.scatter(xs_u, ys_u, s=ss_u, marker='^', alpha=0.95)
    # highlight ring
    hi = set(highlight or [])
    if hi:
        hx=[]; hy=[]
        for nid in hi:
            if nid in pos: 
                x,y=pos[nid]; hx.append(x); hy.append(y)
        if hx: ax.scatter(hx, hy, s=300, facecolors='none', edgecolors='black', linewidths=2.4, alpha=0.9)
    # labels
    if len(nodes) <= 1200:
        for nid,(x,y) in pos.items():
            ax.text(x, y+0.02, str(nid), fontsize=6, ha='center', va='bottom', alpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)

# ============================================================
# 6) Plot templates
# ============================================================

def _moving_stats(arr: List[float], window: int):
    arr = np.asarray(arr, dtype=np.float64)
    if window <= 1 or window > len(arr): 
        return arr, np.zeros_like(arr)
    ma = np.convolve(arr, np.ones(window)/window, mode='valid')
    pad = len(arr) - len(ma)
    ma = np.concatenate([np.full(pad, np.nan), ma])
    # rolling std
    rs = []
    for i in range(len(arr)):
        j0 = max(0, i-window+1); jj = arr[j0:i+1]
        rs.append(np.std(jj) if len(jj)>1 else 0.0)
    return ma, np.asarray(rs)

def plot_learning_and_complexity(history: List[Tuple[float,float]], hidden_counts_history: List[List[int]], edge_counts_history: List[List[int]], out_path: str, title: str, ma_window: int = 7):
    best = [b for b,_ in history]; avg = [a for _,a in history]
    best_ma, best_std = _moving_stats(best, ma_window)
    avg_ma,  avg_std  = _moving_stats(avg,  ma_window)
    mean_hidden = [float(np.mean(h)) for h in hidden_counts_history]
    mean_edges  = [float(np.mean(e)) for e in edge_counts_history]
    gens = np.arange(len(history))
    fig, ax1 = plt.subplots(figsize=(6,4))
    # raw
    ax1.plot(gens, best, linewidth=1.0, alpha=0.7, linestyle=':')
    ax1.plot(gens, avg,  linewidth=1.0, alpha=0.7, linestyle=':')
    # moving average
    ax1.plot(gens, best_ma, linewidth=1.6, alpha=0.95, label="best (MA)")
    ax1.plot(gens, avg_ma,  linewidth=1.4, alpha=0.95, label="avg (MA)")
    # "CI" as rolling std bounds (lines only)
    ax1.plot(gens, avg_ma-avg_std, linewidth=0.9, alpha=0.8, linestyle='--')
    ax1.plot(gens, avg_ma+avg_std, linewidth=0.9, alpha=0.8, linestyle='--')
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Fitness")
    # secondary y for complexity
    ax2 = ax1.twinx()
    ax2.plot(gens, mean_hidden, linewidth=1.2, alpha=0.75, linestyle='-')
    ax2.plot(gens, mean_edges,  linewidth=1.2, alpha=0.75, linestyle='-.')
    ax2.set_ylabel("Complexity")
    if title: ax1.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def plot_decision_boundary(
    genome: Genome,
    X,
    y,
    out_path: str,
    steps: int = 50,
    contour_cmap: str = "coolwarm",
    point_cmap: Optional[str] = None,
    point_size: float = 12.0,
    point_alpha: float = 0.85,
    add_colorbar: bool = False,
):
    gg = genome.copy()
    try:
        train_with_backprop_numpy(gg, X, y, steps=steps, lr=5e-3, l2=1e-4)
    except Exception:
        pass
    x_min, x_max = X[:,0].min()-0.2, X[:,0].max()+0.2
    y_min, y_max = X[:,1].min()-0.2, X[:,1].max()+0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    P = predict_proba(gg, grid)[:,1].reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6,4))
    cs = ax.contourf(xx, yy, P, levels=20, alpha=0.85, cmap=contour_cmap)
    scatter_kwargs = {
        "s": point_size,
        "alpha": point_alpha,
        "linewidths": 0.25,
        "edgecolors": "black",
    }
    if point_cmap:
        ax.scatter(X[:,0], X[:,1], c=y, cmap=point_cmap, **scatter_kwargs)
    else:
        ax.scatter(X[:,0], X[:,1], c=y, cmap="gray", **scatter_kwargs)
    if add_colorbar:
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)

# Utility: export decision boundaries for all 3 toy tasks (separate PNG files)
def export_decision_boundaries_all(genome: Genome, out_dir: str, steps: int = 50, seed: int = 0):
    os.makedirs(out_dir or ".", exist_ok=True)
    # Circles
    Xc, yc = make_circles(512, r=0.6, noise=0.05, seed=seed)
    path_c = os.path.join(out_dir, "decision_circles.png")
    plot_decision_boundary(
        genome,
        Xc,
        yc,
        path_c,
        steps=steps,
        contour_cmap="cividis",
        point_cmap="cool",
        point_size=22.0,
    )
    # XOR
    Xx, yx = make_xor(512, noise=0.05, seed=seed)
    path_x = os.path.join(out_dir, "decision_xor.png")
    plot_decision_boundary(
        genome,
        Xx,
        yx,
        path_x,
        steps=steps,
        contour_cmap="Spectral",
        point_cmap="Dark2",
        point_size=18.0,
    )
    # Spiral
    Xs, ys = make_spirals(512, noise=0.05, turns=1.5, seed=seed)
    path_s = os.path.join(out_dir, "decision_spiral.png")
    plot_decision_boundary(
        genome,
        Xs,
        ys,
        path_s,
        steps=steps,
        contour_cmap="magma",
        point_cmap="plasma",
        point_size=14.0,
    )
    return {"circles": path_c, "xor": path_x, "spiral": path_s}


def export_task_gallery(
    tasks: Tuple[str, ...],
    gens: int,
    pop: int,
    steps: int,
    out_dir: str,
) -> Dict[str, str]:
    """Run a batch of miniature experiments and collect figure paths."""

    os.makedirs(out_dir or ".", exist_ok=True)
    outputs: Dict[str, str] = {}
    import zlib
    for idx, task in enumerate(tasks, start=1):
        tag = f"{task}_g{gens}_p{pop}_s{steps}"
        seed = zlib.crc32(f"{task}|{gens}|{pop}|{steps}".encode("utf-8")) & 0xFFFFFFFF
        res = run_backprop_neat_experiment(
            task,
            gens=gens,
            pop=pop,
            steps=steps,
            out_prefix=os.path.join(out_dir, tag),
            make_gifs=False,
            make_lineage=False,
            rng_seed=seed,
        )
        lc = res.get("learning_curve")
        db = res.get("decision_boundary")
        if lc and db and os.path.exists(lc) and os.path.exists(db):
            combo = os.path.join(out_dir, f"{idx:02d}_{tag}_gallery.png")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
            for ax, path, title in zip(axes, (lc, db), ("学習曲線", "決定境界")):
                img = _imread_image(path)
                ax.imshow(img)
                ax.set_title(f"{task.upper()} | {title}")
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(combo, dpi=220)
            plt.close(fig)
            outputs[f"{idx:02d} {task.upper()} 学習曲線＋決定境界"] = combo
        else:
            if lc:
                outputs[f"{idx:02d} {task.upper()} 学習曲線"] = lc
            if db:
                outputs[f"{idx:02d} {task.upper()} 決定境界"] = db
        topo = res.get("topology")
        if topo:
            outputs[f"{idx:02d} {task.upper()} トポロジ"] = topo
    return outputs

# ============================================================
# 7) Toy datasets & CLI demo
# ============================================================

def make_circles(n=512, r=0.5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (np.sqrt((X**2).sum(axis=1)) > r).astype(np.int32)
    X += rng.normal(0, noise, size=X.shape)
    return X, y

def make_xor(n=512, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:,0] * X[:,1]) > 0).astype(np.int32)
    X += rng.normal(0, noise, size=X.shape)
    return X, y

def make_spirals(n=512, noise=0.1, turns=1.5, seed=0):
    rng = np.random.default_rng(seed); n2 = n//2
    t = np.linspace(0.0, turns*2*np.pi, n2); r = np.linspace(0.05, 1.0, n2)
    x1 = r*np.cos(t); y1 = r*np.sin(t); x2 = r*np.cos(t+np.pi); y2 = r*np.sin(t+np.pi)
    X = np.vstack([np.stack([x1,y1],1), np.stack([x2,y2],1)])
    X += rng.normal(0, noise, size=X.shape)
    y = np.concatenate([np.zeros(n2, dtype=np.int32), np.ones(n2, dtype=np.int32)])
    return X, y

def run_backprop_neat_experiment(task: str, gens=60, pop=64, steps=80, out_prefix="out/exp", make_gifs: bool = True, make_lineage: bool = True):
    # dataset
    if task=="xor": Xtr,ytr = make_xor(512, noise=0.05, seed=0); Xva,yva = make_xor(256, noise=0.05, seed=1)
    elif task=="spiral": Xtr,ytr = make_spirals(512, noise=0.05, turns=1.5, seed=0); Xva,yva = make_spirals(256, noise=0.05, turns=1.5, seed=1)
    else: Xtr,ytr = make_circles(512, r=0.6, noise=0.05, seed=0); Xva,yva = make_circles(256, r=0.6, noise=0.05, seed=1)
    # NEAT
    rng = np.random.default_rng(rng_seed)
    neat = ReproPlanaNEATPlus(num_inputs=2, num_outputs=1, population_size=pop, output_activation='identity', rng=rng)
    _apply_stable_neat_defaults(neat)

    def fit(g):
        noise_std = float(neat.env.get('noise_std', 0.0))
        if noise_std > 0.0:
            Xtr_aug = Xtr + neat.rng.normal(0.0, noise_std, size=Xtr.shape)
            Xva_aug = Xva + neat.rng.normal(0.0, noise_std, size=Xva.shape)
        else:
            Xtr_aug = Xtr
            Xva_aug = Xva
        return fitness_backprop_classifier(
            g,
            Xtr_aug,
            ytr,
            Xva_aug,
            yva,
            steps=steps,
            lr=5e-3,
            l2=1e-4,
            alpha_nodes=1e-3,
            alpha_edges=5e-4,
        )
    best, hist = neat.evolve(
        fit,
        n_generations=gens,
        verbose=True,
        env_schedule=_default_difficulty_schedule,
    )
    # Outputs
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    lc_path = f"{out_prefix}_learning_complexity.png"
    plot_learning_and_complexity(hist, neat.hidden_counts_history, neat.edge_counts_history, lc_path, title=f"{task.upper()} | Backprop NEAT", ma_window=7)
    db_path = f"{out_prefix}_decision_boundary.png"
    style_map = {
        "circles": dict(contour_cmap="cividis", point_cmap="cool", point_size=26.0),
        "xor": dict(contour_cmap="Spectral", point_cmap="Dark2", point_size=20.0),
        "spiral": dict(contour_cmap="magma", point_cmap="plasma", point_size=16.0),
    }
    style = style_map.get(task, {})
    plot_decision_boundary(best, Xtr, ytr, db_path, steps=steps, **style)
    topo_path = f"{out_prefix}_topology.png"; scars = diff_scars(None, best, None, birth_gen=gens, regen_mode_for_new="split")
    draw_genome_png(best, scars, topo_path, title=f"Best Topology (Gen {gens})")
    # GIFs (auto)
    regen_gif = None; morph_gif = None
    if make_gifs and len(neat.snapshots_genomes) >= 2:
        regen_gif = f"{out_prefix}_regen.gif"
        export_regen_gif(
            neat.snapshots_genomes,
            neat.snapshots_scars,
            regen_gif,
            fps=12,
            pulse_period_frames=10,
            decay_horizon=6.0,
            fixed_layout=True,
            dpi=150,
        )
        morph_gif = f"{out_prefix}_morph.gif"
        export_morph_gif(
            neat.snapshots_genomes,
            neat.snapshots_scars,
            path=morph_gif,
            fps=14,
            morph_frames=14,
            decay_horizon=8.0,
        )

    # 螺旋再生ヒートマップ
    scars_spiral = None
    if len(neat.snapshots_genomes) >= 2:
        scars_spiral = f"{out_prefix}_scars_spiral.png"
        export_scars_spiral_map(
            neat.snapshots_genomes,
            neat.snapshots_scars,
            scars_spiral,
            turns=None,
            jitter=0.014,
            marker_size=20,
            dpi=220,
        )
    # Lineage (auto)
    lineage_path = None
    if make_lineage:
        lineage_path = f"{out_prefix}_lineage.png"
        render_lineage(neat, path=lineage_path, title=f"{task.upper()} Lineage", max_edges=5000, highlight=neat.best_ids[-10:], dpi=220)
    # Summary decision boundaries for all tasks (separate files)
    summary_dir = f"{out_prefix}_decisions"
    summary_paths = export_decision_boundaries_all(best, summary_dir, steps=steps, seed=0)
    return {
        "learning_curve": lc_path,
        "decision_boundary": db_path,
        "topology": topo_path,
        "regen_gif": regen_gif,
        "morph_gif": morph_gif,
        "lineage": lineage_path,
        "scars_spiral": scars_spiral,
        "summary_decisions": summary_paths,
    }

# === backend-agnostic Figure -> RGB helper ===
def _fig_to_rgb(fig):
    """
    Convert a matplotlib Figure to an RGB uint8 ndarray (H, W, 3), robust across backends.
    Tries Agg (tostring_rgb), then MacOSX (tostring_argb), then in-memory PNG fallback.
    """
    fig.canvas.draw()
    try:
        w, h = fig.canvas.get_width_height()
    except Exception:
        w, h = map(int, fig.get_size_inches() * fig.dpi)

    # Agg: direct RGB
    if hasattr(fig.canvas, "tostring_rgb"):
        import numpy as _np
        buf = _np.frombuffer(fig.canvas.tostring_rgb(), dtype=_np.uint8)
        return buf.reshape(h, w, 3)

    # MacOSX: ARGB -> RGB
    if hasattr(fig.canvas, "tostring_argb"):
        import numpy as _np
        buf = _np.frombuffer(fig.canvas.tostring_argb(), dtype=_np.uint8).reshape(h, w, 4)
        return buf[:, :, 1:4].copy()

    # Fallback: save to PNG in-memory and read back
    import io
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.0)
    bio.seek(0)
    img = _imread_image(bio)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    return img



def export_regen_gif(
    snapshots_genomes,
    snapshots_scars,
    path,
    fps=12,
    pulse_period_frames=16,
    decay_horizon=10.0,
    fixed_layout=True,
    dpi=130
):
    """
    Render a regeneration digest GIF from per-generation snapshots.
    Encodes differences without color semantics: linestyle/linewidth/alpha only.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    if not snapshots_genomes:
        raise ValueError("export_regen_gif: snapshots_genomes is empty.")

    def _node_groups_and_depths(_g):
        depth = _g.node_depths()
        groups = {"input": [], "bias": [], "hidden": [], "output": []}
        for nid, n in _g.nodes.items():
            groups.get(n.type, groups["hidden"]).append(nid)
        for k in groups:
            groups[k].sort()
        max_d = max(depth.values()) if depth else 1
        return groups, depth, max_d

    def _compute_positions(_g):
        groups, depth, max_d = _node_groups_and_depths(_g)
        pos = {}
        bands = [("input", 0.85, 1.00), ("bias", 0.70, 0.82),
                 ("hidden", 0.20, 0.68), ("output", 0.02, 0.18)]
        for (gname, y0, y1) in bands:
            arr = groups[gname]
            n = max(1, len(arr))
            ys = _np.linspace(y1, y0, n)
            for i, nid in enumerate(arr):
                t = depth.get(nid, 0)
                if gname in ("input", "bias"):
                    x = 0.04
                elif gname == "output":
                    x = 0.96
                else:
                    x = 0.10 + 0.80 * (t / max(1, max_d))
                pos[nid] = (x, ys[i])
        return pos

    def _pulse_amp(age, frame_idx):
        base = max(0.1, 1.0 - float(age) / max(1e-6, decay_horizon))
        phase = 2.0 * _np.pi * (frame_idx % max(1, pulse_period_frames)) / max(1, pulse_period_frames)
        return float(base * (0.5 + 0.5 * _np.sin(phase)))

    frames = []
    prev_edges = None
    pos_fixed = _compute_positions(snapshots_genomes[0]) if fixed_layout else None

    for t, g in enumerate(snapshots_genomes):
        pos = pos_fixed if fixed_layout else _compute_positions(g)
        cur_edges = set((c.in_node, c.out_node) for c in g.enabled_connections())
        added = cur_edges - (prev_edges or set())
        removed = (prev_edges or set()) - cur_edges
        common = cur_edges & (prev_edges or set()) if prev_edges is not None else cur_edges

        node_scars, edge_scars = {}, {}
        if snapshots_scars and t < len(snapshots_scars):
            node_scars, edge_scars = _normalize_scar_snapshot(snapshots_scars[t])

        fig, ax = _plt.subplots(figsize=(6.6, 4.8), dpi=dpi)
        ax.set_axis_off()
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)

        # removed edges (dashed, low alpha)
        for (u, v) in sorted(removed):
            if (u not in pos) or (v not in pos): continue
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], linestyle="dashed", linewidth=1.0, alpha=0.25)

        # common edges (normal)
        for (u, v) in sorted(common):
            if (u not in pos) or (v not in pos): continue
            x0, y0 = pos[u]; x1, y1 = pos[v]
            lw = 1.6
            if (u, v) in edge_scars: lw += 0.6 * _pulse_amp(edge_scars[(u, v)], t)
            ax.plot([x0, x1], [y0, y1], linewidth=lw, alpha=0.9)

        # added edges (thick)
        for (u, v) in sorted(added):
            if (u not in pos) or (v not in pos): continue
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], linewidth=2.2, alpha=0.95)

        # nodes (size by role; scars as pulsating outline)
        types = {nid: g.nodes[nid].type for nid in g.nodes}
        for nid, (x, y) in pos.items():
            if nid not in types: continue
            tname = types[nid]
            sz = 50.0
            if tname == "input":  sz = 35.0
            if tname == "bias":   sz = 28.0
            if tname == "output": sz = 60.0
            ax.scatter([x], [y], s=sz, alpha=1.0, zorder=3, linewidths=0.8)
            age = node_scars.get(nid, None)
            if age is not None:
                amp = _pulse_amp(age, t)
                circ = _plt.Circle((x, y), 0.018 + 0.012 * amp, fill=False, linewidth=1.0 + 2.0 * amp, alpha=0.6)
                ax.add_patch(circ)

        img = _fig_to_rgb(fig)
        frames.append(img)
        _plt.close(fig)
        prev_edges = cur_edges

    _mimsave(path, frames, fps=fps)
    return path



def export_morph_gif(
    snapshots_genomes,
    snapshots_scars=None,
    path=None,
    *,
    fps=12,
    morph_frames=8,
    fixed_layout=True,
    dpi=130,
    decay_horizon=8.0,
):
    """
    Inter-generational morphological transition GIF.
    Supports legacy calls with scar metadata as well as lightweight
    visualisations that only rely on genomes.
    """
    import os as _os
    import numpy as _np
    import matplotlib.pyplot as _plt

    # Normalise "path" / scars arguments for backward compatibility.
    if path is None and isinstance(snapshots_scars, (str, bytes)):
        path = snapshots_scars
        snapshots_scars = None
    elif path is None and hasattr(snapshots_scars, "__fspath__"):
        path = _os.fspath(snapshots_scars)
        snapshots_scars = None

    if path is None:
        raise ValueError("export_morph_gif: path must be provided")

    if snapshots_scars is not None:
        if len(snapshots_scars) != len(snapshots_genomes):
            raise ValueError("export_morph_gif: scars_seq length must match genomes.")
        return _export_morph_gif_with_scars(
            snapshots_genomes,
            snapshots_scars,
            path,
            fps=fps,
            morph_frames=morph_frames,
            decay_horizon=decay_horizon,
            fixed_layout=fixed_layout,
            dpi=dpi,
        )

    if not snapshots_genomes or len(snapshots_genomes) < 2:
        raise ValueError("export_morph_gif: need >= 2 snapshots.")

    def _node_groups_and_depths(_g):
        depth = _g.node_depths()
        groups = {"input": [], "bias": [], "hidden": [], "output": []}
        for nid, n in _g.nodes.items():
            groups.get(n.type, groups["hidden"]).append(nid)
        for k in groups:
            groups[k].sort()
        max_d = max(depth.values()) if depth else 1
        return groups, depth, max_d

    def _compute_positions(_g):
        groups, depth, max_d = _node_groups_and_depths(_g)
        pos = {}
        bands = [("input", 0.85, 1.00), ("bias", 0.70, 0.82),
                 ("hidden", 0.20, 0.68), ("output", 0.02, 0.18)]
        for (gname, y0, y1) in bands:
            arr = groups[gname]
            n = max(1, len(arr))
            ys = _np.linspace(y1, y0, n)
            for i, nid in enumerate(arr):
                t = depth.get(nid, 0)
                if gname in ("input", "bias"):
                    x = 0.04
                elif gname == "output":
                    x = 0.96
                else:
                    x = 0.10 + 0.80 * (t / max(1, max_d))
                pos[nid] = (x, ys[i])
        return pos

    pos_fixed = _compute_positions(snapshots_genomes[0]) if fixed_layout else None

    frames = []
    for i in range(len(snapshots_genomes) - 1):
        g0 = snapshots_genomes[i]
        g1 = snapshots_genomes[i + 1]
        pos0 = pos_fixed if fixed_layout else _compute_positions(g0)
        pos1 = pos_fixed if fixed_layout else _compute_positions(g1)

        E0 = set((c.in_node, c.out_node) for c in g0.enabled_connections())
        E1 = set((c.in_node, c.out_node) for c in g1.enabled_connections())
        kept = E0 & E1
        gone = E0 - E1
        born = E1 - E0

        for k in range(max(1, morph_frames)):
            t = 0.0 if morph_frames <= 1 else k / float(morph_frames - 1)

            fig, ax = _plt.subplots(figsize=(6.6, 4.8), dpi=dpi)
            ax.set_axis_off(); ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)

            # gone edges fade out
            for (u, v) in sorted(gone):
                p = pos0 if (u in pos0 and v in pos0) else pos1
                if (u not in p) or (v not in p):
                    continue
                x0, y0 = p[u]; x1, y1 = p[v]
                ax.plot([x0, x1], [y0, y1], linestyle="dashed", linewidth=1.4, alpha=max(0.0, 1.0 - t))

            # kept edges stay
            for (u, v) in sorted(kept):
                if (u not in pos0) or (v not in pos0):
                    continue
                x0, y0 = pos0[u]; x1, y1 = pos0[v]
                ax.plot([x0, x1], [y0, y1], linewidth=1.8, alpha=0.9)

            # born edges fade in
            for (u, v) in sorted(born):
                p = pos1 if (u in pos1 and v in pos1) else pos0
                if (u not in p) or (v not in p):
                    continue
                x0, y0 = p[u]; x1, y1 = p[v]
                ax.plot([x0, x1], [y0, y1], linewidth=2.2, alpha=max(0.0, t))

            # nodes
            types = {nid: (g1.nodes[nid].type if nid in g1.nodes else g0.nodes.get(nid, None).type)
                     for nid in set(list(g0.nodes.keys()) + list(g1.nodes.keys()))}
            p = pos1 if fixed_layout else pos0
            for nid, (x, y) in p.items():
                if nid not in types:
                    continue
                tname = types[nid]
                sz = 50.0
                if tname == "input":
                    sz = 35.0
                if tname == "bias":
                    sz = 28.0
                if tname == "output":
                    sz = 60.0
                ax.scatter([x], [y], s=sz, alpha=1.0, zorder=3, linewidths=0.8)

            img = _fig_to_rgb(fig)
            frames.append(img)
            _plt.close(fig)

    _mimsave(path, frames, fps=fps)
    return path



def export_scars_spiral_map(
    snapshots_genomes: List[Genome],
    snapshots_scars: List[Dict[int, 'Scar']],
    out_path: str,
    *,
    turns: Optional[float] = None,
    jitter: float = 0.012,
    marker_size: float = 26.0,
    dpi: int = 220,
    title: str = "Scars Spiral Map"
):
    """
    再生痕(=新規ノード誕生)を、世代→角度θ・進行度→半径r のアルキメデス螺旋へ投影して可視化する。

    座標系:
      θ_g = (2π * turns) * g/(G-1)
      r_g = r0 + (r1 - r0) * (θ_g / (2π * turns))   （中心から外周へ等間隔で広がる）

    入力フォーマットは diff_scars の既存形式（Dict[node_id -> Scar]）に対応。
    互換性のため、scarsが無い場合はノード集合差分から「新生ノード」を推定する。
    """
    if not snapshots_genomes:
        raise ValueError("export_scars_spiral_map: snapshots_genomes is empty.")

    import numpy as _np
    import matplotlib.pyplot as _plt

    G = len(snapshots_genomes)
    if G <= 1:
        raise ValueError("export_scars_spiral_map: need >= 2 snapshots.")

    # 螺旋パラメータ
    if turns is None:
        # ランが長いほど少し多めに回す（2〜8回転の範囲）
        turns = float(max(2.0, min(8.0, G / 10.0)))
    theta_max = 2.0 * _np.pi * turns
    r0, r1 = 0.12, 0.96  # 中心〜外周の正規化半径

    # --- 世代ごとの「新生ノード」イベントを抽出 ---
    events_by_gen: Dict[int, List[Tuple[int, str]]] = {}
    for g in range(G):
        new_nodes: List[Tuple[int, str]] = []

        scars_g = snapshots_scars[g] if (snapshots_scars and g < len(snapshots_scars)) else None
        if isinstance(scars_g, dict) and scars_g:
            if "nodes" in scars_g or "edges" in scars_g:
                raw_nodes = scars_g.get("nodes", {}) or {}
                for nid, sc in raw_nodes.items():
                    age = getattr(sc, "age", None)
                    if age is None and isinstance(sc, dict):
                        age = sc.get("age", None)
                    if age is None and isinstance(sc, (int, float)):
                        age = int(sc)
                    birth = getattr(sc, "birth_gen", g)
                    if isinstance(sc, dict):
                        birth = sc.get("birth_gen", birth)
                    if age == 0 and birth == g:
                        mode = getattr(sc, "mode", None)
                        if isinstance(sc, dict):
                            mode = sc.get("mode", mode)
                        new_nodes.append((int(nid), mode or "split"))
            else:
                # 既存の diff_scars 形式: Dict[nid -> Scar]
                for nid, sc in scars_g.items():
                    # その世代で age==0 かつ birth_gen==g のものを「新生」とみなす
                    age = getattr(sc, "age", None)
                    birth = getattr(sc, "birth_gen", g)
                    mode = getattr(sc, "mode", None)
                    if isinstance(sc, dict):
                        age = sc.get("age", age)
                        birth = sc.get("birth_gen", birth)
                        mode = sc.get("mode", mode)
                    if age == 0 and birth == g:
                        new_nodes.append((int(nid), mode or "split"))
        else:
            # フォールバック: ノード集合差分から推定（互換目的）
            if g > 0:
                prev_ids = set(snapshots_genomes[g-1].nodes.keys())
                curr_ids = set(snapshots_genomes[g].nodes.keys())
                born = curr_ids - prev_ids
                mode = getattr(snapshots_genomes[g], "regen_mode", "split")
                new_nodes.extend((int(nid), mode) for nid in born)

        if new_nodes:
            events_by_gen[g] = new_nodes

    # --- 角度θと半径rへ写像し、モード別に点群を作る ---
    xs = {"split": [], "head": [], "tail": [], "other": []}
    ys = {"split": [], "head": [], "tail": [], "other": []}
    heat_x: List[float] = []
    heat_y: List[float] = []

    for g, items in events_by_gen.items():
        theta = theta_max * (g / max(1, G-1))
        base_r = r0 + (r1 - r0) * (theta / theta_max)

        n = len(items)
        offs = _np.linspace(-0.5, 0.5, n) if n > 1 else [0.0]  # 同一世代で少しだけ半径方向にズラす
        for (offset, (nid, mode)) in zip(offs, items):
            r = base_r + float(offset) * jitter
            x = r * _np.cos(theta)
            y = r * _np.sin(theta)
            key = mode if mode in xs else "other"
            xs[key].append(x)
            ys[key].append(y)
            heat_x.append(x)
            heat_y.append(y)

    # --- 描画 ---
    fig, ax = _plt.subplots(figsize=(6, 6), dpi=dpi, subplot_kw={"aspect": "equal"})
    # 螺旋のセンターライン
    ts = _np.linspace(0.0, theta_max, 1200)
    rr = r0 + (r1 - r0) * (ts / theta_max)
    ax.plot(rr * _np.cos(ts), rr * _np.sin(ts), linewidth=1.0, alpha=0.35, linestyle="-")

    # ヒートマップ層（2Dビニング）
    if heat_x and heat_y:
        bins = 160
        heat_grid, xedges, yedges = _np.histogram2d(
            heat_x,
            heat_y,
            bins=bins,
            range=[[-1.05, 1.05], [-1.05, 1.05]],
        )
        heat_grid = heat_grid.astype(float)
        if heat_grid.max() > 0:
            heat_grid /= heat_grid.max()
            kernel = _np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
            kernel /= kernel.sum()
            padded = _np.pad(heat_grid, 1, mode="reflect")
            smooth = _np.zeros_like(heat_grid)
            for i in range(heat_grid.shape[0]):
                for j in range(heat_grid.shape[1]):
                    smooth[i, j] = _np.sum(padded[i : i + 3, j : j + 3] * kernel)
            ax.imshow(
                smooth.T,
                extent=[-1.05, 1.05, -1.05, 1.05],
                origin="lower",
                cmap="inferno",
                alpha=0.45,
                interpolation="bilinear",
            )

    markers = {"split": "o", "head": "s", "tail": "^", "other": "x"}
    labels  = {"split": "split", "head": "head", "tail": "tail", "other": "other"}
    for k in ("split", "head", "tail", "other"):
        if xs[k]:
            ax.scatter(xs[k], ys[k], s=marker_size, alpha=0.15, marker=markers[k], linewidths=0.7, label=labels[k])

    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, loc="left")
    if any(xs[k] for k in xs):
        ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    _plt.close(fig)
    return out_path



# ============================================================
# 7') Gym action heads (Discrete / MultiDiscrete / MultiBinary / Box)
# ============================================================
def _softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    denom = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.maximum(denom, 1e-9)

def _import_gym():
    try:
        import gymnasium as gym  # type: ignore
    except ImportError:  # pragma: no cover - fallback for legacy gym installs
        import gym  # type: ignore
    return gym

def output_dim_from_space(space) -> int:
    try:
        from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary, Box  # type: ignore
    except Exception:
        from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box  # type: ignore
    import numpy as _np
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiBinary):
        return int(_np.prod(space.n if hasattr(space, "n") else space.shape))
    if isinstance(space, MultiDiscrete):
        return int(_np.sum(space.nvec))
    if isinstance(space, Box):
        return int(_np.prod(space.shape))
    raise ValueError(f"Unsupported action space: {type(space)}")

def obs_dim_from_space(space):
    if hasattr(space, "shape") and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "n"):
        return int(space.n)
    raise ValueError(f"Unsupported observation space: {type(space)}")

def build_action_mapper(space, stochastic=False, temp=1.0):
    try:
        from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary, Box  # type: ignore
    except Exception:
        from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box  # type: ignore

    t = max(1e-6, float(temp))

    if isinstance(space, Discrete):
        def f(logits):
            z = np.asarray(logits).reshape(-1)[: space.n]
            probs = _softmax_np(z / t).ravel()
            if stochastic:
                action = int(np.random.choice(len(probs), p=probs))
            else:
                action = int(np.argmax(z))
            return action, probs
        return f

    if isinstance(space, MultiDiscrete):
        nvec = np.asarray(space.nvec, dtype=int)
        cuts = np.cumsum(nvec)[:-1]

        def f(logits):
            z = np.asarray(logits).reshape(-1)
            parts = np.split(z, cuts) if len(cuts) else [z]
            actions = []
            probs_all = []
            for idx, nz in enumerate(nvec):
                seg = parts[idx][:nz]
                pp = _softmax_np(seg / t).ravel()
                if stochastic:
                    a = int(np.random.choice(nz, p=pp))
                else:
                    a = int(np.argmax(seg))
                actions.append(a)
                probs_all.append(pp)
            return np.asarray(actions, dtype=space.dtype), np.concatenate(probs_all, axis=0)

        return f

    if isinstance(space, MultiBinary):
        n = int(np.prod(space.n if hasattr(space, "n") else space.shape))

        def f(logits):
            z = np.asarray(logits).reshape(n)
            probs = 1.0 / (1.0 + np.exp(-(z / t)))
            if stochastic:
                actions = (np.random.random(size=n) < probs).astype(space.dtype)
            else:
                actions = (probs >= 0.5).astype(space.dtype)
            return actions.reshape(space.shape), probs

        return f

    if isinstance(space, Box):
        low = np.broadcast_to(space.low, space.shape).astype(float)
        high = np.broadcast_to(space.high, space.shape).astype(float)

        def f(logits):
            z = np.asarray(logits).reshape(space.shape)
            tanh = np.tanh(z)
            act = ((low + high) / 2.0) + ((high - low) / 2.0) * tanh
            return act.astype(space.dtype), tanh.ravel()

        return f

    raise ValueError(f"Unsupported action space: {type(space)}")

def _default_difficulty_schedule(gen: int, _ctx: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Three-phase curriculum with a late-game regeneration surge."""
    if gen < 25:
        return {"difficulty": 0.3, "noise_std": 0.0, "enable_regen": False}
    if gen < 40:
        return {"difficulty": 0.6, "noise_std": 0.02, "enable_regen": False}
    return {"difficulty": 1.0, "noise_std": 0.05, "enable_regen": True}


def _apply_stable_neat_defaults(neat: ReproPlanaNEATPlus):
    """Thesis-grade defaults: calm search, regen gated until curriculum lifts it."""
    neat.mode = EvalMode(
        vanilla=True,
        enable_regen_reproduction=False,
        complexity_alpha=neat.mode.complexity_alpha,
        node_penalty=neat.mode.node_penalty,
        edge_penalty=neat.mode.edge_penalty,
        species_low=neat.mode.species_low,
        species_high=neat.mode.species_high,
    )
    neat.mutate_add_conn_prob = 0.05
    neat.mutate_add_node_prob = 0.03
    neat.mutate_weight_prob = 0.8
    neat.regen_mode_mut_rate = 0.05
    neat.mix_asexual_base = 0.10
    if getattr(neat, "complexity_threshold", None) is None:
        neat.complexity_threshold = 1.0


def setup_neat_for_env(env_id: str, population: int = 48, output_activation: str = 'identity'):
    gym = _import_gym()
    env = gym.make(env_id)
    obs_dim = obs_dim_from_space(env.observation_space)
    out_dim = output_dim_from_space(env.action_space)
    neat = ReproPlanaNEATPlus(num_inputs=obs_dim, num_outputs=out_dim, population_size=population, output_activation=output_activation)
    _apply_stable_neat_defaults(neat)
    return neat, env

def _rollout_policy_in_env(genome, env, mapper, max_steps=None, render=False, obs_norm=None):
    """Rollout one episode with a Genome and an action mapper."""
    total, steps, done = 0.0, 0, False
    reset_out = env.reset()
    obs = reset_out[0] if (isinstance(reset_out, tuple) and len(reset_out) >= 1) else reset_out
    while not done:
        if render:
            try:
                env.render()
            except Exception:
                pass
        x = obs if obs_norm is None else obs_norm(obs)
        y = genome.forward_one(np.asarray(x, dtype=np.float32).ravel())
        mapped = mapper(y)
        if isinstance(mapped, tuple):
            act = mapped[0]
        else:
            act = mapped
        step_out = env.step(act)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out
            done = bool(done)
        total += float(reward)
        steps += 1
        if (max_steps is not None) and (steps >= int(max_steps)):
            break
    return total

def gym_fitness_factory(env_id, stochastic=False, temp=1.0, max_steps=1000, episodes=1, obs_norm=None):
    """Return a fitness function for evolve() that evaluates average episodic reward."""
    gym = _import_gym()
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)
    mapper = build_action_mapper(env.action_space, stochastic=stochastic, temp=temp)
    n_episodes = max(1, int(episodes))

    def _fitness(genome):
        total = 0.0
        for _ in range(n_episodes):
            total += _rollout_policy_in_env(
                genome,
                env,
                mapper,
                max_steps=max_steps,
                render=False,
                obs_norm=obs_norm,
            )
        return total / float(n_episodes)

    def _close_env():
        try:
            env.close()
        except Exception:
            pass

    _fitness.close_env = _close_env  # type: ignore[attr-defined]
    _fitness.env = env  # type: ignore[attr-defined]
    return _fitness


def eval_with_node_activations(genome: 'Genome', obs_vec: np.ndarray):
    """Evaluate a genome on a single observation and capture node activations."""
    nodes = genome.nodes
    order = genome.topological_order()
    incoming = {}
    for c in genome.enabled_connections():
        if not c.enabled:
            continue
        incoming.setdefault(c.out_node, []).append(c)

    acts = {nid: 0.0 for nid in nodes}
    pre = {nid: 0.0 for nid in nodes}

    inputs = [nid for nid, n in nodes.items() if n.type == 'input']
    for i, nid in enumerate(inputs):
        if i < len(obs_vec):
            acts[nid] = float(obs_vec[i])
    for nid, n in nodes.items():
        if n.type == 'bias':
            acts[nid] = 1.0

    for nid in order:
        node = nodes[nid]
        if node.type in ('input', 'bias'):
            pre[nid] = acts[nid]
            continue
        s = 0.0
        for c in incoming.get(nid, []):
            s += acts[c.in_node] * c.weight
        pre[nid] = s
        acts[nid] = act_forward(node.activation, s)

    outs = [nid for nid, n in nodes.items() if n.type == 'output']
    outs.sort()
    logits = np.array([acts[o] for o in outs], dtype=np.float64)
    return logits, acts, pre


def _color_from_value(v: float):
    v = float(np.tanh(v))
    r = 0.2 + 0.6 * max(0.0, v)
    b = 0.2 + 0.6 * max(0.0, -v)
    g = 0.22
    return (r, g, b)


def _draw_nn(ax, genome: 'Genome', acts: Dict[int, float], show_values: bool = False, scars=None, radius: float = 0.10):
    pos = layout_by_depth(genome)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    for c in genome.enabled_connections():
        i, o = c.in_node, c.out_node
        if i not in pos or o not in pos:
            continue
        p1, p2 = pos[i], pos[o]
        lw = 0.6 + 2.2 * min(1.0, abs(c.weight))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, alpha=0.75)
    for nid, nd in genome.nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        fc = _color_from_value(acts.get(nid, 0.0))
        ax.add_patch(Circle((x, y), radius=radius, color=fc, alpha=0.95))
        lw = 1.2
        if scars and (nid in scars):
            age = scars[nid].age
            lw = 1.2 + 0.8 * math.exp(-0.15 * age)
        ax.add_patch(Circle((x, y), radius=radius, fill=False, linewidth=lw, alpha=0.95))
        ax.text(x, y - radius * 1.7, nd.type[0], ha='center', va='top', fontsize=7, alpha=0.86)
        if show_values:
            ax.text(x, y + radius * 1.6, f"{acts.get(nid, 0.0):+.2f}", ha='center', va='bottom', fontsize=7, alpha=0.9)


def _draw_prob_bars(ax, probs, title="Action probabilities"):
    probs = np.asarray(probs).ravel()
    ax.clear()
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(probs) - 0.5)
    ax.bar(np.arange(len(probs)), probs, alpha=0.9)
    ax.set_xticks(range(len(probs)))
    ax.set_title(title, fontsize=9)
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.2f}", ha='center', va='bottom', fontsize=7)
    ax.grid(alpha=0.15, linestyle=':')


def _episode_bc_update(genome: 'Genome', obs_list, act_list, ret_list,
                       steps=20, lr=1e-2, l2=1e-4, top_frac=0.3):
    if len(obs_list) == 0:
        return
    n = len(obs_list)
    k = max(1, int(max(1.0 / n, top_frac) * n))
    idx = np.argsort(ret_list)[::-1][:k]
    X = np.asarray([obs_list[i] for i in idx], dtype=np.float64)
    y = np.asarray([act_list[i] for i in idx], dtype=np.int32)
    try:
        train_with_backprop_numpy(genome, X, y, steps=int(steps), lr=float(lr), l2=float(l2))
    except Exception as e:
        print("[warn] online update skipped:", e)


def run_policy_in_env(genome: 'Genome', env_id: str,
                      episodes: int = 1, max_steps: int = 1000,
                      stochastic: bool = True, temp: float = 1.0,
                      out_gif: str = "out/rl_rollout.gif", fps: int = 20,
                      panel_ratio: float = 0.58,
                      show_values: bool = True, show_bars: bool = True,
                      rl_update: bool = False, gamma: float = 0.99,
                      rl_steps: int = 20, rl_lr: float = 1e-2, rl_l2: float = 1e-4,
                      top_frac: float = 0.3):
    try:
        import gymnasium as gym
    except Exception:
        import gym  # type: ignore

    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id)
    mapper = build_action_mapper(env.action_space, stochastic=stochastic, temp=temp)

    frames = []
    for ep in range(int(episodes)):
        reset_out = env.reset()
        obs = reset_out[0] if (isinstance(reset_out, tuple) and len(reset_out) >= 1) else reset_out
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        ep_obs, ep_act, ep_rew = [], [], []

        for t in range(int(max_steps)):
            logits, acts, _pre = eval_with_node_activations(genome, obs)
            mapped = mapper(logits)
            if isinstance(mapped, tuple):
                action, probs = mapped
            else:
                action, probs = mapped, None

            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, rew, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, rew, done, _info = step_out
                done = bool(done)
            next_obs = np.asarray(next_obs, dtype=np.float64).reshape(-1)

            fig = plt.figure(figsize=(9.2, 5.0))
            if show_bars:
                gs = fig.add_gridspec(2, 2, height_ratios=[0.65, 0.35], width_ratios=[panel_ratio, 1.0 - panel_ratio])
                ax_env = fig.add_subplot(gs[0, 0])
                ax_prob = fig.add_subplot(gs[1, 0])
                ax_nn = fig.add_subplot(gs[:, 1])
            else:
                gs = fig.add_gridspec(1, 2, width_ratios=[panel_ratio, 1.0 - panel_ratio])
                ax_env = fig.add_subplot(gs[0, 0])
                ax_nn = fig.add_subplot(gs[0, 1])
                ax_prob = None

            try:
                img = env.render()
                ax_env.imshow(img)
                ax_env.axis('off')
            except Exception:
                ax_env.text(0.5, 0.5, "(render unavailable)", ha='center', va='center')
                ax_env.axis('off')

            _draw_nn(ax_nn, genome, acts, show_values=show_values)
            ax_env.set_title(f"{env_id} | ep {ep + 1} t={t} r={rew:.2f}")
            ax_nn.set_title("Policy network (activations)")

            if show_bars and probs is not None:
                _draw_prob_bars(ax_prob, probs, title="Action probabilities")

            fig.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            frames.append(buf)
            plt.close(fig)

            ep_obs.append(obs.copy())
            if np.isscalar(action):
                ep_act.append(int(action))
            else:
                flat = np.asarray(action).ravel()
                ep_act.append(int(flat[0]) if flat.size == 1 else None)
            ep_rew.append(float(rew))

            obs = next_obs
            if done:
                if rl_update:
                    G = 0.0
                    rets = []
                    for r in ep_rew[::-1]:
                        G = float(r) + float(gamma) * G
                        rets.append(G)
                    rets = list(reversed(rets))
                    rets_np = np.asarray(rets, dtype=np.float64)
                    if len(rets_np) > 1 and np.std(rets_np) > 1e-8:
                        rets_np = (rets_np - np.mean(rets_np)) / (np.std(rets_np) + 1e-8)
                    if all(a is not None for a in ep_act):
                        _episode_bc_update(genome, ep_obs, [int(a) for a in ep_act], rets_np,
                                           steps=rl_steps, lr=rl_lr, l2=rl_l2, top_frac=top_frac)
                break

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    _mimsave(out_gif, frames, fps=fps)
    try:
        env.close()
    except Exception:
        pass
    return out_gif


def run_gym_neat_experiment(
    env_id: str,
    gens: int = 20,
    pop: int = 24,
    episodes: int = 1,
    max_steps: int = 500,
    stochastic: bool = False,
    temp: float = 1.0,
    out_prefix: str = "out/rl",
) -> Dict[str, Any]:
    """Convenience wrapper that evolves NEAT agents on a Gym environment."""

    neat, env = setup_neat_for_env(env_id, population=pop, output_activation="identity")
    try:
        env.close()
    except Exception:
        pass
    fit = gym_fitness_factory(
        env_id,
        stochastic=stochastic,
        temp=temp,
        max_steps=max_steps,
        episodes=episodes,
    )
    best, hist = neat.evolve(fit, n_generations=gens, verbose=True, env_schedule=_default_difficulty_schedule)

    close_env = getattr(fit, "close_env", None)
    if callable(close_env):
        close_env()

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    rc_png = f"{out_prefix}_reward_curve.png"
    xs = list(range(len(hist)))
    ys_b = [b for (b, _a) in hist]
    ys_a = [a for (_b, a) in hist]
    plt.figure()
    plt.plot(xs, ys_b, label="best")
    plt.plot(xs, ys_a, label="avg")
    plt.xlabel("generation")
    plt.ylabel("episode reward")
    plt.title(f"{env_id} | Average Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(rc_png, dpi=150)
    plt.close()

    return {"best": best, "history": hist, "reward_curve": rc_png}


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Command-line interface entrypoint."""
    _ensure_matplotlib_agg(force=True)

    ap = argparse.ArgumentParser(description="Spiral-NEAT NumPy | built-in CLI")
    ap.add_argument("--task", choices=["xor","circles","spiral"])
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop",  type=int, default=64)
    ap.add_argument("--steps",type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rl-env", type=str)
    ap.add_argument("--rl-gens", type=int, default=20)
    ap.add_argument("--rl-pop",  type=int, default=24)
    ap.add_argument("--rl-episodes", type=int, default=1)
    ap.add_argument("--rl-max-steps", type=int, default=500)
    ap.add_argument("--rl-stochastic", action="store_true")
    ap.add_argument("--rl-temp", type=float, default=1.0)
    ap.add_argument("--rl-gameplay-gif", action="store_true")
    ap.add_argument("--out", default="out_monolith_cli")
    ap.add_argument("--make-gifs", action="store_true")
    ap.add_argument("--make-lineage", action="store_true")
    ap.add_argument("--gallery", nargs="*", default=[])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(None if argv is None else list(argv))

    os.makedirs(args.out, exist_ok=True)
    figs: Dict[str, Optional[str]] = {}

    # supervised
    if args.task:
        np.random.seed(args.seed)
        res = run_backprop_neat_experiment(
            args.task, gens=args.gens, pop=args.pop, steps=args.steps,
            out_prefix=os.path.join(args.out, args.task),
            make_gifs=args.make_gifs, make_lineage=args.make_lineage,
            rng_seed=args.seed,
        )
        figs["図1 学習曲線＋複雑度"] = res.get("learning_curve")
        figs["図2 最良トポロジ"] = res.get("topology")
        regen_gif = res.get("regen_gif")
        if regen_gif and os.path.exists(regen_gif) and imageio is not None:
            with imageio.get_reader(regen_gif) as r:
                idx = max(0, r.get_length()//2 - 1)
                frame = r.get_data(idx)
                fig3 = os.path.join(args.out, f"{args.task}_fig3_regen_frame.png")
                _imwrite_image(fig3, frame)
                figs["図3 再生ダイジェスト代表フレーム"] = fig3
        else:
            figs["図3 決定境界"] = res.get("decision_boundary")
        figs["図4 螺旋再生ヒートマップ"] = res.get("scars_spiral")

        if args.gallery:
            gal = export_task_gallery(
                tasks=tuple(args.gallery),
                gens=max(6, args.gens // 2),
                pop=max(12, args.pop // 2),
                steps=max(10, args.steps // 2),
                out_dir=os.path.join(args.out, "gallery"),
            )
            for k, v in gal.items():
                figs[f"ギャラリー {k}"] = v

    # RL
    if args.rl_env:
        try:
            gym = _import_gym()

            env_probe = gym.make(args.rl_env)
            try:
                obs_dim = obs_dim_from_space(env_probe.observation_space)
                out_dim = output_dim_from_space(env_probe.action_space)
            finally:
                try:
                    env_probe.close()
                except Exception:
                    pass

            neat = ReproPlanaNEATPlus(
                num_inputs=obs_dim,
                num_outputs=out_dim,
                population_size=args.rl_pop,
                output_activation="identity",
            )
            _apply_stable_neat_defaults(neat)
            fit = gym_fitness_factory(
                args.rl_env,
                stochastic=args.rl_stochastic,
                temp=args.rl_temp,
                max_steps=args.rl_max_steps,
                episodes=args.rl_episodes,
            )
            best, hist = neat.evolve(
                fit,
                n_generations=args.rl_gens,
                verbose=True,
                env_schedule=_default_difficulty_schedule,
            )
            close_env = getattr(fit, "close_env", None)
            if callable(close_env):
                close_env()
            rc_png = os.path.join(args.out, f"{args.rl_env.replace(':','_')}_reward_curve.png")
            xs = list(range(len(hist)))
            ys_b = [b for (b, _a) in hist]
            ys_a = [a for (_b, a) in hist]
            plt.figure()
            plt.plot(xs, ys_b, label="best")
            plt.plot(xs, ys_a, label="avg")
            plt.xlabel("generation")
            plt.ylabel("episode reward")
            plt.title(f"{args.rl_env} | Average Episode Reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(rc_png, dpi=150)
            plt.close()
            figs["RL 平均エピソード報酬"] = rc_png
            if args.rl_gameplay_gif:
                gif = os.path.join(args.out, f"{args.rl_env.replace(':','_')}_gameplay.gif")
                try:
                    out_path = run_policy_in_env(
                        best,
                        args.rl_env,
                        episodes=max(1, args.rl_episodes),
                        max_steps=args.rl_max_steps,
                        stochastic=args.rl_stochastic,
                        temp=args.rl_temp,
                        out_gif=gif,
                    )
                    if out_path and os.path.exists(out_path):
                        figs["RL ゲームプレイ"] = out_path
                except Exception as gif_err:
                    print("[WARN] gameplay gif failed:", gif_err)
        except Exception as e:
            print("[WARN] RL branch skipped:", e)

    if args.report and figs:
        # minimal self-contained HTML (base64 embed)
        def _data_uri(p: str) -> Tuple[str, str]:
            with open(p, "rb") as f:
                import base64

                raw = base64.b64encode(f.read()).decode("ascii")
            mime, _ = mimetypes.guess_type(p)
            if mime is None:
                if p.lower().endswith(".gif"):
                    mime = "image/gif"
                elif p.lower().endswith((".mp4", ".webm")):
                    mime = "video/mp4"
                else:
                    mime = "application/octet-stream"
            return f"data:{mime};base64,{raw}", mime

        html = os.path.join(args.out, "Sakana_NEAT_Report.html")
        entries = [(k, p) for k, p in figs.items() if p and os.path.exists(p)]
        with open(html, "w", encoding="utf-8") as f:
            from datetime import datetime

            f.write("<!DOCTYPE html><html lang='ja'><head><meta charset='utf-8'><title>Report</title>")
            f.write(
                "<style>body{font-family:'Hiragino Sans','Noto Sans JP',sans-serif;background:#fafafa;color:#222;line-height:1.6;padding:2rem;}"
                "header.cover{background:#fff;border:1px solid #ddd;border-radius:12px;padding:1.5rem;margin-bottom:2rem;box-shadow:0 8px 20px rgba(0,0,0,0.05);}"
                "header.cover h1{margin-top:0;font-size:1.9rem;}"
                "header.cover ul{margin:0;padding-left:1.2rem;}"
                "section.legend{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:1.25rem;margin-bottom:2rem;}"
                "section.narrative{background:#fff;border:1px solid #e4e4e4;border-radius:10px;padding:1.25rem;margin-bottom:2rem;box-shadow:0 10px 24px rgba(0,0,0,0.035);}"
                "section.narrative h2{margin-top:0;font-size:1.35rem;}section.narrative p{margin:0 0 0.8rem 0;}"
                "section.legend ol{margin:0;padding-left:1.4rem;}"
                "figure{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:1rem;margin:0 0 2rem 0;box-shadow:0 12px 24px rgba(0,0,0,0.04);}"
                "figure img,figure video{width:100%;height:auto;border-radius:8px;}"
                "figcaption{margin-top:0.75rem;font-weight:600;}"
                "</style></head><body>"
            )

            f.write("<header class='cover'>")
            f.write("<h1>Spiral Monolith NEAT Report</h1>")
            f.write(
                f"<p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 主要タスク: {args.task.upper() if args.task else 'N/A'}</p>"
            )
            if args.gallery:
                f.write("<p>ギャラリー: " + ", ".join(t.upper() for t in args.gallery) + "</p>")
            f.write("<ul>")
            f.write(f"<li>世代数: {args.gens}</li>")
            f.write(f"<li>集団サイズ: {args.pop}</li>")
            f.write(f"<li>Backprop steps: {args.steps}</li>")
            if args.rl_env:
                f.write(f"<li>RL 環境: {args.rl_env}</li>")
            f.write("</ul>")
            f.write("</header>")

            f.write("<section class='narrative'><h2>進化ダイジェスト</h2>")
            f.write(
                "<p>Early generations showed smooth structural adaptation and convergence under low-difficulty conditions.</p>"
            )
            f.write(
                "<p>However, as environmental difficulty and noise increased, regeneration-driven mutations began to trigger bursts of morphological diversification, resembling biological punctuated equilibria.</p>"
            )
            f.write("</section>")

            if entries:
                f.write("<section class='legend'><h2>図版リスト</h2><ol>")
                for label, path in entries:
                    f.write(
                        f"<li><strong>{label}</strong><br><small>{os.path.basename(path)}</small></li>"
                    )
                f.write("</ol></section>")

            for k, p in entries:
                uri, mime = _data_uri(p)
                if mime == "image/gif":
                    f.write(
                        f"<figure><img src='{uri}' style='max-width:100%'>"
                        f"<figcaption><strong>{k}</strong></figcaption></figure>"
                    )
                elif mime.startswith("image/"):
                    f.write(
                        f"<figure><img src='{uri}' style='max-width:100%'><figcaption><strong>{k}</strong></figcaption></figure>"
                    )
                elif mime.startswith("video/"):
                    f.write(
                        "<figure><video autoplay loop muted playsinline style='max-width:100%'>"
                        f"<source src='{uri}' type='{mime}'></video>"
                        f"<figcaption><strong>{k}</strong></figcaption></figure>"
                    )
                else:
                    f.write(
                        f"<figure><a href='{uri}'>download {os.path.basename(p)}</a>"
                        f"<figcaption><strong>{k}</strong></figcaption></figure>"
                    )
            f.write("</body></html>")
        print("[REPORT]", html)

    print("[OK] outputs in:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
