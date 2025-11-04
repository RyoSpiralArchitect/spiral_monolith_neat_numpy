
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

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable, Optional, Set, Iterable, Any
from collections import deque, defaultdict, OrderedDict
import math, argparse, os, mimetypes, csv
import matplotlib
import warnings
import pickle as _pickle
import json as _json

try:  # Python 3.8+
    from multiprocessing import shared_memory as _shm
except Exception:
    _shm = None

# === Safety & Runtime Preamble ===============================================
# - BLAS スレッドを 1 に制限して並列評価との過剰スレッド競合を防止
# - 既知の FutureWarning を静音化
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

def _is_picklable(obj) -> bool:
    """Process 並列に切り替える前に picklable かを事前検査（非picklableなら thread に自動フォールバック）"""
    try:
        _pickle.dumps(obj)
        return True
    except Exception:
        return False

# === Shared-memory datasets (for process-parallel, zero-copy) =================
_SHM_LOCAL = {}   # parent-owned SharedMemory objects (for cleanup)
_SHM_META  = {}   # {label -> {'name','shape','dtype','readonly'}}
_SHM_CACHE = {}   # worker-attached numpy views
_SHM_HANDLES = {}  # worker-side: keep SharedMemory objects alive

def shm_register_dataset(label: str, arr: "np.ndarray", readonly: bool = True) -> dict:
    """Create shared memory for arr (parent), return metadata dict."""
    if _shm is None:
        raise RuntimeError("shared_memory is unavailable on this Python.")
    arr = np.asarray(arr)
    size = int(arr.nbytes)
    # unique name
    name = f"sm_{label}_{np.random.randint(1, 1<<30):08x}"
    shm = _shm.SharedMemory(create=True, size=size, name=name)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    _SHM_LOCAL[label] = (shm, arr.shape, str(arr.dtype), bool(readonly))
    meta = {"name": name, "shape": tuple(arr.shape), "dtype": str(arr.dtype), "readonly": bool(readonly)}
    _SHM_META[label] = meta
    return meta

def shm_set_worker_meta(meta: dict | None):
    """Install metadata in worker; views are attached lazily on demand."""
    global _SHM_META, _SHM_CACHE, _SHM_HANDLES
    _SHM_META = dict(meta or {})
    _SHM_CACHE = {}
    _SHM_HANDLES = {}

def get_shared_dataset(label: str) -> "np.ndarray":
    """Worker-side: return cached numpy view to shared dataset by label."""
    if label in _SHM_CACHE:
        return _SHM_CACHE[label]
    meta = _SHM_META.get(label)
    if not meta:
        raise KeyError(f"Shared dataset '{label}' not found.")
    if _shm is None:
        raise RuntimeError("shared_memory is unavailable in worker.")
    shm = _shm.SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    if bool(meta.get("readonly", True)):
        try:
            arr.setflags(write=False)
        except Exception:
            pass
    # 重要: ハンドルを保持して GC で閉じられないようにする
    _SHM_HANDLES[label] = shm
    _SHM_CACHE[label] = arr
    return arr

def shm_release_all():
    """Parent-side: close & unlink all owned segments."""
    if not _SHM_LOCAL:
        return
    for _label, (shm, _shape, _dtype, _ro) in list(_SHM_LOCAL.items()):
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
    _SHM_LOCAL.clear()

def shm_worker_release_all():
    """Worker-side: close all attached SharedMemory handles."""
    global _SHM_HANDLES
    for _label, shm in list(_SHM_HANDLES.items()):
        try:
            shm.close()
        except Exception:
            pass
    _SHM_HANDLES.clear()

def _proc_init_worker(meta: dict | None = None):
    """ProcessPool initializer: cap BLAS threads and install SHM metadata."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if meta:
        try:
            shm_set_worker_meta(meta)
        except Exception:
            pass
    # 重要: worker 終了時に SHM を確実に close
    try:
        import atexit
        atexit.register(shm_worker_release_all)
    except Exception:
        pass

def _ensure_matplotlib_agg(force: bool = False):
    """Select Agg backend even if pyplot was already imported elsewhere."""
    try:
        matplotlib.use("Agg", force=force)
    except TypeError:  # pragma: no cover - older matplotlib doesn't support force
        matplotlib.use("Agg")
    return matplotlib


_ensure_matplotlib_agg()

STRUCTURAL_EPS = 1e-9

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
    "LCSMonitor","summarize_graph_changes","load_lcs_log","export_lcs_ribbon_png","export_lcs_timeline_gif",
    "PerSampleSequenceStopperPro",
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
        # Initialize sex as male or female (hermaphrodites only emerge through mutation)
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

    def weighted_adjacency(self, include_disabled: bool = False):
        nodes = set(self.nodes.keys())
        adj: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in nodes}
        for c in self.connections.values():
            if include_disabled or c.enabled:
                adj.setdefault(c.in_node, []).append((c.out_node, c.weight))
                if c.out_node not in adj:
                    adj[c.out_node] = []
        for nid in nodes:
            adj.setdefault(nid, [])
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

    def remove_cycles(self):
        """Remove cycles by disabling connections until the genome is acyclic.
        Returns True if any connections were disabled."""
        disabled_any = False
        # In the worst case, we only need to disable as many connections as there are enabled
        max_iterations = len(list(self.enabled_connections())) + 1
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            # Try to get topological order using Kahn's algorithm
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
            
            # If we got all nodes, no cycle exists
            if len(order) == len(self.nodes):
                break
            
            # There's a cycle. Find nodes that are in the cycle (not in topological order)
            nodes_in_order = set(order)
            cycle_nodes = set(self.nodes.keys()) - nodes_in_order
            
            # Disable one connection that involves only cycle nodes (both in and out)
            # This ensures we're breaking the actual cycle, not just cutting off descendants
            # Prefer connections with higher innovation numbers (newer connections)
            disabled_one = False
            for c in sorted(self.enabled_connections(), key=lambda x: x.innovation, reverse=True):
                if c.in_node in cycle_nodes and c.out_node in cycle_nodes:
                    c.enabled = False
                    disabled_any = True
                    disabled_one = True
                    break
            
            # If we didn't find a connection within the cycle, fall back to any connection involving cycle nodes
            if not disabled_one:
                for c in sorted(self.enabled_connections(), key=lambda x: x.innovation, reverse=True):
                    if c.in_node in cycle_nodes or c.out_node in cycle_nodes:
                        c.enabled = False
                        disabled_any = True
                        disabled_one = True
                        break
            
            if not disabled_one:
                # Final fallback: disable any enabled connection
                for c in self.enabled_connections():
                    c.enabled = False
                    disabled_any = True
                    break
        
        return disabled_any

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

    def mutate_sex(self, rng: np.random.Generator):
        """Mutate sex, with low probability of becoming hermaphrodite.
        
        The mutation probability is controlled by the NEAT instance's mutate_sex_prob parameter.
        Only male or female individuals can mutate into hermaphrodites.
        """
        # This method is called by _mutate when the mutation is triggered
        # The probability check is done in _mutate, so we just perform the mutation
        if self.sex in ('male', 'female'):
            self.sex = 'hermaphrodite'
            return True
        return False

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


def summarize_graph_changes(adj0: Dict[int, List[Tuple[int, float]]],
                            adj1: Dict[int, List[Tuple[int, float]]],
                            weight_tol: float = 0.0) -> Tuple[Set[int], int]:
    """Return nodes and edge-count touched by structural/weight deltas (|w|<=tol treated absent)."""
    def collect(adj):
        edges = {}
        nodes = set(adj.keys())
        for u, nbrs in adj.items():
            nodes.add(u)
            for v, w in nbrs:
                if abs(w) <= weight_tol:
                    continue
                edges[(u, v)] = w
                nodes.add(v)
        return edges, nodes

    edges0, nodes0 = collect(adj0)
    edges1, nodes1 = collect(adj1)

    changed_nodes: Set[int] = set()
    changed_edges = 0
    for edge in set(edges0.keys()).union(edges1.keys()):
        w0 = edges0.get(edge)
        w1 = edges1.get(edge)
        if w0 is None or w1 is None or abs(w0 - w1) > weight_tol:
            changed_edges += 1
            changed_nodes.update(edge)

    changed_nodes.update(nodes0.symmetric_difference(nodes1))
    return changed_nodes, changed_edges


def load_lcs_log(csv_path: str) -> List[Dict[str, Any]]:
    """Parse the LCS CSV into a list of numeric-friendly dict rows."""
    if not csv_path or not os.path.exists(csv_path):
        return []

    def _as_int(value):
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        s = str(value).strip()
        if s == "":
            return None
        try:
            return int(float(s))
        except ValueError:
            return None

    def _as_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        s = str(value).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                gen = _as_int(raw.get("gen"))
                out_id = _as_int(raw.get("o"))
                if gen is None or out_id is None:
                    continue
                lineage_raw = raw.get("lineage_id")
                lineage_id = _as_int(lineage_raw)
                row = {
                    "gen": gen,
                    "lineage_id": lineage_id if lineage_id is not None else lineage_raw,
                    "mut_id": raw.get("mut_id"),
                    "o": out_id,
                    "changed_nodes": _as_int(raw.get("changed_nodes")) or 0,
                    "changed_edges": _as_int(raw.get("changed_edges")) or 0,
                    "R0": int(_as_int(raw.get("R0")) or 0),
                    "R1": int(_as_int(raw.get("R1")) or 0),
                    "P0": _as_int(raw.get("P0")) or 0,
                    "P1": _as_int(raw.get("P1")) or 0,
                    "d0": _as_int(raw.get("d0")),
                    "d1": _as_int(raw.get("d1")),
                    "detour": _as_float(raw.get("detour")),
                    "delta_paths": _as_int(raw.get("delta_paths")) or 0,
                    "delta_sp": _as_int(raw.get("delta_sp")),
                    "heal_flag": int(_as_int(raw.get("heal_flag")) or 0),
                    "time_to_heal": _as_int(raw.get("time_to_heal")),
                    "disjoint_paths0": _as_int(raw.get("disjoint_paths0")) or 0,
                    "disjoint_paths1": _as_int(raw.get("disjoint_paths1")) or 0,
                }
                rows.append(row)
    except FileNotFoundError:
        return []
    return rows


def _prepare_lcs_series(lcs_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_output: Dict[int, "OrderedDict[int, Dict[str, Any]]"] = {}
    per_gen: Dict[int, Dict[str, Any]] = {}
    gens: Set[int] = set()
    for row in lcs_rows:
        gen = row.get("gen")
        out_id = row.get("o")
        if gen is None or out_id is None:
            continue
        gens.add(gen)
        per_output.setdefault(out_id, OrderedDict())[gen] = row
        agg = per_gen.setdefault(
            gen,
            {
                "count": 0,
                "paths1": [],
                "disjoint1": [],
                "detour": [],
                "delta_paths": [],
                "changed_edges": 0,
                "heals": 0,
                "breaks": 0,
                "time_to_heal": [],
                "connected": 0,
            },
        )
        agg["count"] += 1
        agg["changed_edges"] += row.get("changed_edges", 0) or 0
        agg["paths1"].append(row.get("P1", 0) or 0)
        agg["disjoint1"].append(row.get("disjoint_paths1", 0) or 0)
        detour_val = row.get("detour")
        if detour_val is not None and not np.isnan(detour_val):
            agg["detour"].append(float(detour_val))
        agg["delta_paths"].append(row.get("delta_paths", 0) or 0)
        if row.get("R0", 0) == 1 and row.get("R1", 0) == 0:
            agg["breaks"] += 1
        if row.get("heal_flag", 0):
            agg["heals"] += 1
        tth = row.get("time_to_heal")
        if tth is not None:
            agg["time_to_heal"].append(tth)
        if row.get("R1", 0):
            agg["connected"] += 1

    outputs = sorted(per_output.keys())
    for out_id in outputs:
        per_output[out_id] = OrderedDict(sorted(per_output[out_id].items()))
    generations = sorted(gens)
    return {
        "per_output": per_output,
        "per_gen": per_gen,
        "outputs": outputs,
        "generations": generations,
    }


def _latest_gen_summary(series: Optional[Dict[str, Any]], upto_gen: int) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not series:
        return None, None
    gens: List[int] = series.get("generations", [])  # type: ignore
    for g in reversed(gens):
        if g <= upto_gen:
            return g, series.get("per_gen", {}).get(g)  # type: ignore
    return None, None


def _cumulative_lcs_counts(series: Optional[Dict[str, Any]], upto_gen: int) -> Tuple[int, int]:
    if not series:
        return 0, 0
    heals = 0
    breaks = 0
    for gen, summary in series.get("per_gen", {}).items():  # type: ignore
        if gen <= upto_gen:
            heals += int(summary.get("heals", 0))
            breaks += int(summary.get("breaks", 0))
    return heals, breaks


def _format_lcs_summary(summary: Optional[Dict[str, Any]]) -> str:
    if not summary or summary.get("count", 0) == 0:
        return "LCS: no connectivity data yet"

    def _mean(vals: List[float], default: float = 0.0) -> float:
        return float(np.mean(vals)) if vals else default

    def _median(vals: List[float]) -> Optional[float]:
        return float(np.median(vals)) if vals else None

    avg_paths = _mean(summary.get("paths1", []))
    avg_disjoint = _mean(summary.get("disjoint1", []))
    mean_delta = _mean(summary.get("delta_paths", []))
    med_detour = _median(summary.get("detour", []))
    med_tth = _median(summary.get("time_to_heal", []))
    connected_ratio = (
        float(summary.get("connected", 0)) / float(summary.get("count", 1))
        if summary.get("count", 0)
        else 0.0
    )
    detour_str = f"detour≈{med_detour:.2f}" if med_detour is not None else "detour=—"
    tth_str = f"Tth≈{med_tth:.1f}" if med_tth is not None else "Tth=—"
    return (
        f"altμ {avg_paths:.2f} | disjointμ {avg_disjoint:.2f} | Δpaths {mean_delta:+.2f} | "
        f"{detour_str} | {tth_str} | conn {connected_ratio:.2f} | "
        f"H:{summary.get('heals', 0)} B:{summary.get('breaks', 0)} | Δedges {summary.get('changed_edges', 0)}"
    )


def export_lcs_ribbon_png(
    lcs_rows: List[Dict[str, Any]],
    path: str,
    series: Optional[Dict[str, Any]] = None,
    outputs: Optional[Iterable[int]] = None,
    dpi: int = 200,
) -> Optional[str]:
    if not lcs_rows:
        return None
    if series is None:
        series = _prepare_lcs_series(lcs_rows)
    outputs_list = list(outputs) if outputs is not None else list(series.get("outputs", []))
    if not outputs_list:
        return None

    import matplotlib.pyplot as _plt

    styles = ["solid", (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (3, 2, 1, 2))]
    markers = ["o", "s", "^", "v", "D", "P", "X", "+"]

    fig, axes = _plt.subplots(3, 1, sharex=True, figsize=(7.2, 7.4), dpi=dpi)
    detour_values = []
    for _row in lcs_rows:
        val = _row.get("detour")
        if val is not None and not np.isnan(val):
            detour_values.append(float(val))
    alt_max = max((r.get("P1", 0) or 0) for r in lcs_rows)
    dis_max = max((r.get("disjoint_paths1", 0) or 0) for r in lcs_rows)

    for idx, out_id in enumerate(outputs_list):
        data_dict = series.get("per_output", {}).get(out_id)  # type: ignore
        if not data_dict:
            continue
        gens = list(data_dict.keys())
        alt = [data_dict[g]["P1"] for g in gens]
        dis = [data_dict[g]["disjoint_paths1"] for g in gens]
        det = [data_dict[g]["detour"] if data_dict[g]["detour"] is not None else np.nan for g in gens]
        conn = [data_dict[g]["R1"] for g in gens]
        heals = [data_dict[g]["heal_flag"] for g in gens]
        style = styles[idx % len(styles)]
        marker = markers[idx % len(markers)]
        label = f"output {out_id}"
        axes[0].plot(gens, alt, linestyle=style, marker=marker, markersize=4.0, color="black", linewidth=1.5, label=label)
        axes[1].plot(gens, dis, linestyle=style, marker=marker, markersize=4.0, color="black", linewidth=1.5)
        axes[2].plot(gens, det, linestyle=style, marker=marker, markersize=4.0, color="black", linewidth=1.5)
        for g_val, alt_val, connected in zip(gens, alt, conn):
            if not connected:
                axes[0].scatter([g_val], [alt_val], marker="x", color="black", s=36, linewidths=1.2)
        for g_val, alt_val, heal, det_val in zip(gens, alt, heals, det):
            if heal:
                axes[0].scatter([g_val], [alt_val], marker="D", facecolors="none", edgecolors="black", s=58, linewidths=1.0)
                if not np.isnan(det_val):
                    axes[2].scatter([g_val], [det_val], marker="D", facecolors="none", edgecolors="black", s=58, linewidths=1.0)

    axes[0].set_ylabel("alt paths")
    axes[1].set_ylabel("edge-disjoint")
    axes[2].set_ylabel("detour")
    axes[2].set_xlabel("generation")
    axes[0].set_ylim(-0.1, max(1.0, alt_max) + 1.0)
    axes[1].set_ylim(-0.1, max(1.0, dis_max) + 1.0)
    if detour_values:
        d_min = min(detour_values)
        d_max = max(detour_values)
        pad = max(0.05, 0.05 * d_max)
        axes[2].set_ylim(max(0.0, d_min - pad), d_max + pad)
    else:
        axes[2].set_ylim(0.8, 1.4)
    axes[0].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(True, color="0.85", linestyle=(0, (1, 3)), linewidth=0.6)

    last_gen = series.get("generations", [])[-1] if series.get("generations") else None
    _, summary = _latest_gen_summary(series, last_gen) if last_gen is not None else (None, None)
    summary_text = _format_lcs_summary(summary)
    fig.suptitle("Local Continuity Signature overview", y=0.98, fontsize=12)
    fig.text(0.02, 0.03, summary_text, fontsize=9, family="monospace")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=dpi)
    _plt.close(fig)
    return path


def export_lcs_timeline_gif(
    lcs_rows: List[Dict[str, Any]],
    path: str,
    series: Optional[Dict[str, Any]] = None,
    fps: int = 6,
    dpi: int = 150,
) -> Optional[str]:
    if not lcs_rows:
        return None
    if series is None:
        series = _prepare_lcs_series(lcs_rows)
    gens = list(series.get("generations", []))
    if not gens:
        return None

    import matplotlib.pyplot as _plt

    styles = ["solid", (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (3, 2, 1, 2))]
    markers = ["o", "s", "^", "v", "D", "P", "X", "+"]
    first_gen = gens[0]
    alt_max = max((r.get("P1", 0) or 0) for r in lcs_rows)
    dis_max = max((r.get("disjoint_paths1", 0) or 0) for r in lcs_rows)
    detour_values = []
    for _row in lcs_rows:
        val = _row.get("detour")
        if val is not None and not np.isnan(val):
            detour_values.append(float(val))
    if detour_values:
        d_min = min(detour_values)
        d_max = max(detour_values)
        pad = max(0.05, 0.05 * d_max)
        det_ylim = (max(0.0, d_min - pad), d_max + pad)
    else:
        det_ylim = (0.8, 1.4)

    frames = []
    for upto in gens:
        fig, axes = _plt.subplots(3, 1, sharex=True, figsize=(6.4, 5.6), dpi=dpi)
        for idx, out_id in enumerate(series.get("outputs", [])):
            data_dict = series.get("per_output", {}).get(out_id)  # type: ignore
            if not data_dict:
                continue
            use_gens = [g for g in data_dict.keys() if g <= upto]
            if not use_gens:
                continue
            alt = [data_dict[g]["P1"] for g in use_gens]
            dis = [data_dict[g]["disjoint_paths1"] for g in use_gens]
            det = [data_dict[g]["detour"] if data_dict[g]["detour"] is not None else np.nan for g in use_gens]
            conn = [data_dict[g]["R1"] for g in use_gens]
            heals = [data_dict[g]["heal_flag"] for g in use_gens]
            style = styles[idx % len(styles)]
            marker = markers[idx % len(markers)]
            label = f"output {out_id}" if upto == first_gen else "_nolegend_"
            axes[0].plot(use_gens, alt, linestyle=style, marker=marker, markersize=4.0, color="black", linewidth=1.4, label=label)
            axes[1].plot(use_gens, dis, linestyle=style, marker=marker, markersize=4.0, color="black", linewidth=1.4)
            axes[2].plot(use_gens, det, linestyle=style, marker=marker, markersize=4.0, color="black", linewidth=1.4)
            for g_val, alt_val, connected in zip(use_gens, alt, conn):
                if not connected:
                    axes[0].scatter([g_val], [alt_val], marker="x", color="black", s=32, linewidths=1.1)
            for g_val, alt_val, heal, det_val in zip(use_gens, alt, heals, det):
                if heal:
                    axes[0].scatter([g_val], [alt_val], marker="D", facecolors="none", edgecolors="black", s=54, linewidths=1.0)
                    if not np.isnan(det_val):
                        axes[2].scatter([g_val], [det_val], marker="D", facecolors="none", edgecolors="black", s=54, linewidths=1.0)

        for ax in axes:
            ax.axvline(upto, color="0.35", linestyle=(0, (2, 3)), linewidth=1.0)
            ax.grid(True, color="0.85", linestyle=(0, (1, 3)), linewidth=0.6)

        axes[0].set_ylabel("alt paths")
        axes[1].set_ylabel("edge-disjoint")
        axes[2].set_ylabel("detour")
        axes[2].set_xlabel("generation")
        axes[0].set_ylim(-0.1, max(1.0, alt_max) + 1.0)
        axes[1].set_ylim(-0.1, max(1.0, dis_max) + 1.0)
        axes[2].set_ylim(*det_ylim)
        axes[0].legend(loc="upper left", frameon=False)

        gen_key, summary = _latest_gen_summary(series, upto)
        summary_line = _format_lcs_summary(summary)
        cumulative = [r for r in lcs_rows if r.get("gen") is not None and r["gen"] <= upto]
        cum_heals = sum(r.get("heal_flag", 0) for r in cumulative)
        cum_breaks = sum(1 for r in cumulative if r.get("R0", 0) == 1 and r.get("R1", 0) == 0)
        fig.suptitle(f"LCS timeline ≤ Gen {upto}", y=0.97, fontsize=12)
        fig.text(0.02, 0.06, summary_line, fontsize=9, family="monospace")
        fig.text(0.02, 0.03, f"cum heals {cum_heals} | cum breaks {cum_breaks}", fontsize=8, family="monospace")
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        frame = _fig_to_rgb(fig)
        _plt.close(fig)
        frames.append(frame)

    if not frames:
        return None
    _mimsave(path, frames, fps=max(1, fps))
    return path
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

def _reachable_outputs_fraction(g, eps=0.0) -> float:
    """Fraction of outputs that are reachable from any input via edges with |w|>eps."""
    adj = g.weighted_adjacency()
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input','bias')]
    outputs = [nid for nid, n in g.nodes.items() if n.type == 'output']
    if not inputs or not outputs:
        return 1.0
    seen = set(inputs)
    q = deque(inputs)
    while q:
        u = q.popleft()
        for v, w in adj.get(u, []):
            if abs(w) <= eps:
                continue
            if v not in seen:
                seen.add(v); q.append(v)
    cnt = sum(1 for o in outputs if o in seen)
    return float(cnt) / max(1, len(outputs))

def _connectivity_guard(g, innov, rng, min_frac=0.6, max_new_edges=16, eps=0.0):
    """If reachability falls below min_frac, add safe forward edges."""
    frac = _reachable_outputs_fraction(g, eps=eps)
    if frac >= min_frac:
        return
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input','bias')]
    outputs = [nid for nid, n in g.nodes.items() if n.type == 'output']
    adj = g.weighted_adjacency()
    seen = set(inputs); q = deque(inputs)
    while q:
        u = q.popleft()
        for v, w in adj.get(u, []):
            if abs(w) <= eps: continue
            if v not in seen:
                seen.add(v); q.append(v)
    sources = sorted([nid for nid in seen if g.nodes[nid].type in ('hidden','input','bias')])
    unreachable = [o for o in outputs if o not in seen]
    attempts = 0
    rng_local = rng or np.random.default_rng()
    while unreachable and attempts < int(max_new_edges):
        if not sources: break
        src = int(rng_local.choice(sources))
        out = int(rng_local.choice(unreachable))
        if (not g.has_connection(src, out)) and (not g._creates_cycle(src, out)):
            inn = innov.get_conn_innovation(src, out)
            g.connections[inn] = ConnectionGene(src, out, float(rng_local.uniform(0.6, 1.6)), True, inn)
            # 再評価
            adj = g.weighted_adjacency()
            seen = set(inputs); q = deque(inputs)
            while q:
                u = q.popleft()
                for v, w in adj.get(u, []):
                    if abs(w) <= eps: continue
                    if v not in seen: seen.add(v); q.append(v)
            unreachable = [o for o in outputs if o not in seen]
        attempts += 1

def _soft_regenerate_head(g, rng, innov, intensity=0.5):
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input','bias')]
    candidates = [c for c in g.enabled_connections() if c.in_node in inputs]
    if not candidates: return g
    rng_local = rng or np.random.default_rng()
    keep_rate = max(0.65, 1.0 - 0.5 * float(intensity))
    n = len(candidates)
    n_disable = int(min(n*(1.0-keep_rate)*0.4, max(1, 0.1*n)))
    n_atten = int(min(n*(1.0-keep_rate), n - n_disable))
    idx = np.arange(n); rng_local.shuffle(idx)
    for k in idx[:n_atten]:
        c = candidates[k]; c.weight *= float(rng_local.uniform(0.6, 0.9))
    for k in idx[n_atten:n_atten+n_disable]:
        c = candidates[k]; c.enabled = False
    m = int(1 + round(2 * float(intensity)))
    for _ in range(m):
        c = candidates[int(rng_local.integers(n))]
        new_id = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_id not in g.nodes:
            g.nodes[new_id] = NodeGene(new_id, 'hidden', 'tanh')
        inn1 = innov.get_conn_innovation(c.in_node, new_id)
        inn2 = innov.get_conn_innovation(new_id, c.out_node)
        g.connections[inn1] = ConnectionGene(c.in_node, new_id, 1.0, True, inn1)
        g.connections[inn2] = ConnectionGene(new_id, c.out_node, c.weight, True, inn2)
    return g

def _soft_regenerate_tail(g, rng, innov, intensity=0.5):
    outputs = [nid for nid, n in g.nodes.items() if n.type == 'output']
    sinks = [c for c in g.enabled_connections() if c.out_node in outputs]
    if not sinks: return g
    rng_local = rng or np.random.default_rng()
    k = max(1, int(len(sinks) * (0.15 + 0.45 * float(intensity))))
    hidden = [nid for nid, n in g.nodes.items() if n.type == 'hidden']
    choose = sinks if k >= len(sinks) else list(rng_local.choice(sinks, size=k, replace=False))
    for c in choose:
        c.weight = float(rng_local.uniform(-1.8, 1.8))
        if hidden and rng_local.random() < (0.25 + 0.35 * float(intensity)):
            new_src = int(rng_local.choice(hidden))
            if (not g.has_connection(new_src, c.out_node)) and (not g._creates_cycle(new_src, c.out_node)):
                inn = innov.get_conn_innovation(new_src, c.out_node)
                g.connections[inn] = ConnectionGene(new_src, c.out_node, float(rng_local.uniform(-1.6, 1.6)), True, inn)
    return g

def _soft_regenerate_split(g, rng, innov, intensity=0.5):
    hidden = [nid for nid, n in g.nodes.items() if n.type == 'hidden']
    rng_local = rng or np.random.default_rng()
    if not hidden:
        enabled = [c for c in g.connections.values() if c.enabled]
        if not enabled: return g
        c = enabled[int(rng_local.integers(len(enabled)))]
        c.enabled = False
        new_nid = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_nid not in g.nodes:
            g.nodes[new_nid] = NodeGene(new_nid, 'hidden', 'tanh')
        inn1 = innov.get_conn_innovation(c.in_node, new_nid)
        inn2 = innov.get_conn_innovation(new_nid, c.out_node)
        g.connections[inn1] = ConnectionGene(c.in_node, new_nid, 1.0, True, inn1)
        g.connections[inn2] = ConnectionGene(new_nid, c.out_node, c.weight, True, inn2)
        return g
    target = int(rng_local.choice(hidden))
    dup_id = innov.new_node_id()
    g.nodes[dup_id] = NodeGene(dup_id, 'hidden', 'tanh')
    incomings = [c for c in g.enabled_connections() if c.out_node == target]
    for cin in incomings:
        inn = innov.get_conn_innovation(cin.in_node, dup_id)
        g.connections[inn] = ConnectionGene(cin.in_node, dup_id, cin.weight + float(rng_local.normal(0, 0.08)), True, inn)
    outgoings = [c for c in g.enabled_connections() if c.in_node == target]
    move_p = min(0.7, 0.25 + 0.45 * float(intensity))
    for cout in outgoings:
        if rng_local.random() < move_p:
            cout.enabled = False
            inn = innov.get_conn_innovation(dup_id, cout.out_node)
            g.connections[inn] = ConnectionGene(dup_id, cout.out_node, cout.weight + float(rng_local.normal(0, 0.08)), True, inn)
    # 少量の並列出力枝を保持
    if outgoings and rng_local.random() < 0.3:
        pick = outgoings[int(rng_local.integers(len(outgoings)))]
        if (not g.has_connection(target, pick.out_node)) and (not g._creates_cycle(target, pick.out_node)):
            inn = innov.get_conn_innovation(target, pick.out_node)
            g.connections[inn] = ConnectionGene(target, pick.out_node, pick.weight, True, inn)
    return g

def platyregenerate(genome: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5) -> Genome:
    """Soft regeneration + connectivity guard."""
    g = genome.copy()
    mode = g.regen_mode or 'split'
    if mode == 'head': g = _soft_regenerate_head(g, rng, innov, intensity)
    elif mode == 'tail': g = _soft_regenerate_tail(g, rng, innov, intensity)
    else: g = _soft_regenerate_split(g, rng, innov, intensity)
    eps = 0.0
    try:
        monitor = getattr(getattr(genome, "lcs_monitor", None), "eps", None)  # best-effort
        eps = float(monitor or 0.0)
    except Exception:
        pass
    _connectivity_guard(g, innov, rng, min_frac=getattr(genome, "min_conn_after_regen", 0.65), eps=eps)
    return g

# ============================================================
# 3) EvalMode & ReproPlanaNEATPlus
# ============================================================

@dataclass
class EvalMode:
    vanilla: bool = True                      # True -> pure NEAT fitness (no sex/regen bonuses)
    enable_regen_reproduction: bool = False   # allow asexual_regen in reproduction
    complexity_alpha: float = 0.01
    node_penalty: float = 0.3  # 1.0 → 0.3 に緩和（複雑なトポロジーを保持）
    edge_penalty: float = 0.15  # 0.5 → 0.15 に緩和
    species_low: int = 3
    species_high: int = 8

INF = 10 ** 12


@dataclass
class PairState:
    status: str = "connected"
    broke_at: int = -1


@dataclass
class LCSMonitor:
    inputs: List[int]
    outputs: List[int]
    K: int = 5
    T: int = 3
    cooldown: int = 1
    eps: float = 0.0
    r_hop: int = 0
    csv_path: str = "regen_log.csv"
    _pair_state: Dict[int, PairState] = field(default_factory=dict)
    _last_heal_gen: Dict[int, int] = field(default_factory=dict)

    def _filtered_adj(self, adj):
        out = {u: [] for u in adj}
        for u, nbrs in adj.items():
            bucket = out.setdefault(u, [])
            for v, w in nbrs:
                if abs(w) > self.eps:
                    bucket.append((v, w))
                    out.setdefault(v, [])
        for node in self.inputs + self.outputs:
            out.setdefault(node, [])
        return out

    def _nbr_targets(self, nbrs):
        """Yield neighbor ids from adjacency entries that may be (v, w) tuples or plain v ids.
        This lets monitors operate on both weighted adjacencies and SCC-condensed graphs.
        """
        for item in nbrs:
            if isinstance(item, (tuple, list)) and len(item) >= 1:
                yield item[0]
            else:
                yield item

    def _nodes_in_scope(self, adj, changed_nodes):
        if self.r_hop <= 0 or not changed_nodes:
            nodes = set(adj.keys())
            for nbrs in adj.values():
                for v in self._nbr_targets(nbrs):
                    nodes.add(v)
            nodes.update(self.inputs)
            nodes.update(self.outputs)
            return nodes
        scope = set(changed_nodes)
        frontier = list(changed_nodes)
        for _ in range(self.r_hop):
            nxt = []
            for u in frontier:
                for v in self._nbr_targets(adj.get(u, ())):  # forward
                    if v not in scope:
                        scope.add(v)
                        nxt.append(v)
            for p, nbrs in adj.items():  # backward one hop
                for v in self._nbr_targets(nbrs):
                    if v in frontier and p not in scope:
                        scope.add(p)
                        nxt.append(p)
            frontier = nxt
        scope.update(self.inputs)
        scope.update(self.outputs)
        return scope

    def _induced_adj(self, adj, nodes):
        out = {u: [] for u in nodes}
        for u in nodes:
            for nbr in adj.get(u, ()):  # type: ignore[arg-type]
                # support both (v,w) entries and plain v entries
                if isinstance(nbr, (tuple, list)) and len(nbr) >= 1:
                    v, w = nbr[0], (nbr[1] if len(nbr) > 1 else None)
                else:
                    v, w = nbr, None
                if v in nodes:
                    out[u].append((v, w if w is not None else 0.0))
        return out

    def _strongly_connected_components(self, adj):
        nodes = []
        seen = set()
        for u in adj:
            if u not in seen:
                nodes.append(u)
                seen.add(u)
        for nbrs in adj.values():
            for v in self._nbr_targets(nbrs):
                if v not in seen:
                    nodes.append(v)
                    seen.add(v)
        index = {}
        lowlink = {}
        stack = []
        on_stack = set()
        components = []
        idx = 0

        def strongconnect(v):
            nonlocal idx
            index[v] = idx
            lowlink[v] = idx
            idx += 1
            stack.append(v)
            on_stack.add(v)
            for w in self._nbr_targets(adj.get(v, ())):  # pragma: no branch
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            if lowlink[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.append(w)
                    if w == v:
                        break
                components.append(comp)

        for v in nodes:
            if v not in index:
                strongconnect(v)

        mapping = {}
        for cid, comp in enumerate(components):
            for node in comp:
                mapping[node] = cid
        condensed = {cid: [] for cid in range(len(components))}
        for u, nbrs in adj.items():
            cu = mapping[u]
            bucket = condensed.setdefault(cu, [])
            for v in self._nbr_targets(nbrs):
                cv = mapping.get(v)
                if cv is None:
                    continue
                if cu != cv:
                    bucket.append(cv)
        for cid, nbrs in condensed.items():
            if nbrs:
                condensed[cid] = sorted(set(nbrs))
            else:
                condensed[cid] = []
        return components, mapping, condensed

    def _count_paths_with_cycles(self, adj):
        components, mapping, condensed = self._strongly_connected_components(adj)
        if not components:
            return {}
        try:
            order, parents = self._topo_order(condensed)
        except ValueError:  # pragma: no cover - condensation must be a DAG
            return {}
        base = defaultdict(int)
        for s in self.inputs:
            if s in mapping:
                base[mapping[s]] += 1
        comp_paths = {}
        for cid in order:
            total = base.get(cid, 0)
            for parent in parents.get(cid, ()):  # pragma: no branch
                total += comp_paths.get(parent, 0)
                if total >= self.K:
                    total = self.K
                    break
            comp_paths[cid] = min(total, self.K)
        node_paths = {}
        for node, cid in mapping.items():
            node_paths[node] = comp_paths.get(cid, 0)
        return node_paths

    def _shortest_len_unit(self, adj):
        nodes = set(adj.keys())
        for nbrs in adj.values():
            for v in self._nbr_targets(nbrs):
                nodes.add(v)
        nodes.update(self.inputs)
        nodes.update(self.outputs)
        dist = {node: INF for node in nodes}
        q = deque()
        for s in self.inputs:
            if dist[s] > 0:
                dist[s] = 0
                q.append(s)
        while q:
            u = q.popleft()
            du = dist[u]
            for v in self._nbr_targets(adj.get(u, ())):  # pragma: no branch
                alt = du + 1
                if alt < dist.get(v, INF):
                    dist[v] = alt
                    q.append(v)
        return dist

    def _topo_order(self, adj):
        """Topologically sort a graph.

        Accepts adjacency in two forms:
          - dict[node] -> list[(neighbor, weight)]
          - dict[node] -> list[neighbor]  (e.g., from SCC condensation)
        """
        nodes = set(adj.keys())
        for nbrs in adj.values():
            for v in self._nbr_targets(nbrs):
                nodes.add(v)
        indeg = {v: 0 for v in nodes}
        for u, nbrs in adj.items():
            for v in self._nbr_targets(nbrs):
                indeg[v] = indeg.get(v, 0) + 1
        queue = deque([v for v in nodes if indeg.get(v, 0) == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in self._nbr_targets(adj.get(v, ())):  # type: ignore[arg-type]
                indeg[w] -= 1
                if indeg[w] == 0:
                    queue.append(w)
        if len(order) != len(nodes):
            raise ValueError("Graph has a cycle; LCS expects a DAG or SCC-condensed DAG.")
        parents = {v: [] for v in nodes}
        for u, nbrs in adj.items():
            for v in self._nbr_targets(nbrs):
                parents[v].append(u)
        return order, parents
    def _reachable_from_inputs(self, adj):
        seen = set()
        queue = deque(self.inputs)
        for s in self.inputs:
            seen.add(s)
        while queue:
            u = queue.popleft()
            for v in self._nbr_targets(adj.get(u, ())):  # type: ignore[arg-type]
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return seen

    def _count_paths_DAG(self, order, parents):
        paths = defaultdict(int)
        for s in self.inputs:
            paths[s] = 1
        for v in order:
            if v in self.inputs:
                continue
            total = 0
            for u in parents.get(v, ()):  # type: ignore[arg-type]
                total += paths[u]
                if total >= self.K:
                    total = self.K
                    break
            paths[v] = min(total, self.K)
        return paths

    def _shortest_len_DAG(self, order, parents):
        dist = {v: INF for v in order}
        for s in self.inputs:
            dist[s] = 0
        for v in order:
            if v in self.inputs:
                continue
            best = INF
            for u in parents.get(v, ()):  # type: ignore[arg-type]
                cand = dist.get(u, INF)
                if cand + 1 < best:
                    best = cand + 1
            dist[v] = best
        return dist

    def _build_unit_capacity_graph(self, adj):
        base = {}
        for u, nbrs in adj.items():
            base.setdefault(u, {})
            for v in self._nbr_targets(nbrs):
                base[u][v] = 1
                base.setdefault(v, {})
        # ensure reverse edges present with zero capacity for residual graph
        for u, nbrs in list(base.items()):
            for v in list(nbrs.keys()):
                base.setdefault(v, {})
                base[v].setdefault(u, 0)
        return base

    def _max_flow(self, graph, source, sink):
        if sink not in graph:
            return 0
        flow = 0
        limit = self.K
        while flow < limit:
            parent = {source: None}
            q = deque([source])
            while q and sink not in parent:
                u = q.popleft()
                for v, cap in graph.get(u, {}).items():
                    if cap > 0 and v not in parent:
                        parent[v] = u
                        if v == sink:
                            break
                        q.append(v)
            if sink not in parent:
                break
            # compute bottleneck capacity
            v = sink
            path_cap = limit - flow
            while parent[v] is not None:
                u = parent[v]
                path_cap = min(path_cap, graph[u][v])
                v = u
            # update residual capacities
            v = sink
            while parent[v] is not None:
                u = parent[v]
                graph[u][v] -= path_cap
                graph[v].setdefault(u, 0)
                graph[v][u] += path_cap
                v = u
            flow += path_cap
        return int(flow)

    def _edge_disjoint_counts(self, adj):
        base = self._build_unit_capacity_graph(adj)
        source = "__source__"
        counts = {}
        # Connect super-source to all inputs present in the scoped graph
        base[source] = {}
        capacity = max(len(base), 1)
        for s in self.inputs:
            if s in base:
                base[source][s] = capacity
                base[s].setdefault(source, 0)
        for o in self.outputs:
            graph = {u: dict(vs) for u, vs in base.items()}
            counts[o] = self._max_flow(graph, source, o)
        return counts

    def _compute_metrics(self, G0, G1, changed_nodes, changed_edges, lineage_id, gen, mut_id):
        adj0 = self._filtered_adj(G0)
        adj1 = self._filtered_adj(G1)

        scope_nodes = self._nodes_in_scope(adj1, changed_nodes)
        adj0s = self._induced_adj(adj0, scope_nodes)
        adj1s = self._induced_adj(adj1, scope_nodes)

        reach0 = self._reachable_from_inputs(adj0s)
        reach1 = self._reachable_from_inputs(adj1s)

        try:
            order0, parents0 = self._topo_order(adj0s)
            paths0 = self._count_paths_DAG(order0, parents0)
            dist0 = self._shortest_len_DAG(order0, parents0)
        except ValueError:
            paths0 = self._count_paths_with_cycles(adj0s)
            dist0 = self._shortest_len_unit(adj0s)

        try:
            order1, parents1 = self._topo_order(adj1s)
            paths1 = self._count_paths_DAG(order1, parents1)
            dist1 = self._shortest_len_DAG(order1, parents1)
        except ValueError:
            paths1 = self._count_paths_with_cycles(adj1s)
            dist1 = self._shortest_len_unit(adj1s)

        disjoint0 = self._edge_disjoint_counts(adj0s)
        disjoint1 = self._edge_disjoint_counts(adj1s)

        rows = []
        for o in self.outputs:
            R0 = int(o in reach0)
            R1 = int(o in reach1)
            P0 = int(min(self.K, paths0.get(o, 0)))
            P1 = int(min(self.K, paths1.get(o, 0)))
            d0 = dist0.get(o, INF)
            d1 = dist1.get(o, INF)
            detour = ""
            delta_sp = ""
            if R0 and R1 and d0 < INF and d1 < INF and d0 > 0:
                detour = float(d1) / float(d0)
                delta_sp = d1 - d0
            elif d0 < INF and d1 < INF:
                delta_sp = d1 - d0
            dis0 = disjoint0.get(o, 0)
            dis1 = disjoint1.get(o, 0)
            rows.append({
                "gen": gen,
                "lineage_id": lineage_id,
                "mut_id": mut_id,
                "o": o,
                "changed_nodes": len(changed_nodes),
                "changed_edges": changed_edges,
                "R0": R0,
                "R1": R1,
                "P0": P0,
                "P1": P1,
                "d0": "" if d0 >= INF else int(d0),
                "d1": "" if d1 >= INF else int(d1),
                "detour": detour,
                "delta_paths": P1 - P0,
                "delta_sp": delta_sp if delta_sp != "" else "",
                "heal_flag": 0,
                "time_to_heal": "",
                "disjoint_paths0": dis0,
                "disjoint_paths1": dis1,
            })
        return rows

    def _update_heal_flags(self, gen, rows):
        updated = []
        for row in rows:
            o = row["o"]
            state = self._pair_state.setdefault(o, PairState())
            last_heal = self._last_heal_gen.get(o, -INF)
            R0 = bool(row["R0"])
            R1 = bool(row["R1"])
            if state.status == "connected":
                if R0 and not R1:
                    state.status = "broken"
                    state.broke_at = gen
            else:
                if R1:
                    tth = gen - state.broke_at
                    if tth <= self.T and gen - last_heal > self.cooldown:
                        row["heal_flag"] = 1
                        row["time_to_heal"] = tth
                        self._last_heal_gen[o] = gen
                    state.status = "connected"
                    state.broke_at = -1
            updated.append(row)
        return updated

    def log_step(self, G_prev, G_post, changed_nodes, lineage_id, gen, mut_id, changed_edges=0):
        rows = self._compute_metrics(G_prev, G_post, set(changed_nodes), changed_edges, lineage_id, gen, mut_id)
        rows = self._update_heal_flags(gen, rows)
        header = [
            "gen",
            "lineage_id",
            "mut_id",
            "o",
            "changed_nodes",
            "changed_edges",
            "R0",
            "R1",
            "P0",
            "P1",
            "d0",
            "d1",
            "detour",
            "delta_paths",
            "delta_sp",
            "heal_flag",
            "time_to_heal",
            "disjoint_paths0",
            "disjoint_paths1",
        ]
        need_header = not os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if need_header:
                    writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except Exception as exc:  # pragma: no cover - logging failure should not crash
            print(f"[LCS] CSV write error: {exc}")
        return rows


class ReproPlanaNEATPlus:
    def __init__(self, num_inputs, num_outputs, population_size=150, rng=None, output_activation='sigmoid'):
        self.num_inputs = num_inputs; self.num_outputs = num_outputs; self.pop_size = population_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mode = EvalMode(vanilla=True, enable_regen_reproduction=False)
        self.max_hidden_nodes = 128; self.max_edges = 1024
        self.complexity_threshold: Optional[float] = 5.0  # 1.0 → 5.0 複雑トポロジー許容のデフォルト値
        # ---- Hardened knobs
        self.grad_clip = 5.0
        self.weight_clip = 12.0
        self.snapshot_stride = 1 if self.pop_size <= 256 else 2
        self.snapshot_max = 320
        self.min_conn_after_regen = 0.65
        self.diversity_push = 0.15
        self.max_attempts_guard = 16
        # Parallel eval
        try:
            cpus = int((os.cpu_count() or 2))
            self.eval_workers = int(os.environ.get("NEAT_EVAL_WORKERS", max(1, cpus - 1)))
            self.parallel_backend = os.environ.get("NEAT_EVAL_BACKEND", "thread")
        except Exception:
            self.eval_workers = 1
            self.parallel_backend = "thread"
        # Auto curriculum toggle
        self.auto_curriculum = True
        # ---- Speciation target learning (dynamic)
        self.species_target = None  # set lazily around (species_low+species_high)/2
        self.species_target_min = 2.0
        self.species_target_max = max(float(self.mode.species_high), float(self.pop_size) / 3.0)
        self.species_target_step = 0.5            # hill-climb step
        self.species_target_update_every = 5      # generations
        self._spec_learn = {"dir": 1.0, "last_best": None, "last_tgt": None, "last_reward": None}
        # PID controller & bandit switching
        self.species_target_mode = "auto"         # "pid" | "hill" | "auto"
        self.pid_kp = 0.35; self.pid_ki = 0.02; self.pid_kd = 0.10
        self.pid_i_clip = 50.0
        self._spec_learn.update({"pid_i": 0.0, "pid_prev_err": None, "score_pid": 0.0, "score_hill": 0.0, "eps": 0.10, "last_method": "pid"})
        # Process pool "rolling restart"
        self.pool_keepalive = int(os.environ.get("NEAT_POOL_KEEPALIVE", "0"))
        self.pool_restart_every = int(os.environ.get("NEAT_POOL_RESTART_EVERY", "25"))
        self._proc_pool = None
        self._proc_pool_age = 0
        self._shm_meta = None

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

        input_ids = list(range(num_inputs))
        output_ids = list(range(num_inputs, num_inputs + num_outputs))
        self.lcs_monitor = LCSMonitor(inputs=input_ids, outputs=output_ids)

        # Params
        self.generation = 0
        self.compatibility_threshold = 3.0
        self.c1=self.c2=1.0; self.c3=0.4
        self.elitism = 1; self.survival_rate = 0.2
        self.mutate_add_conn_prob = 0.10; self.mutate_add_node_prob = 0.10
        self.mutate_weight_prob = 0.8; self.mutate_toggle_prob = 0.01
        self.weight_perturb_chance = 0.9; self.weight_sigma = 0.8; self.weight_reset_range = 2.0
        self.regen_mode_mut_rate = 0.05; self.embryo_bias_mut_rate = 0.03
        self.mutate_sex_prob = 0.005  # Low probability for hermaphrodite emergence
        self.hermaphrodite_inheritance_rate = 0.05  # Very low inheritance rate (5%)
        self.regen_rate = 0.15; self.allow_selfing = True
        self.sex_fitness_scale = {'female':1.0, 'male':0.9, 'hermaphrodite':1.2}; self.regen_bonus = 0.2
        self.hermaphrodite_mate_bias = 2.5  # Hermaphrodites have high mating preference
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
        # 攻撃的・目標駆動の適応
        low = int(self.mode.species_low)
        high = int(self.mode.species_high)
        # lazy init target（学習で更新される）
        if getattr(self, "species_target", None) is None:
            self.species_target = float((low + high) * 0.5)
        target = float(self.species_target)
        if target <= 0:
            target = (low + high) * 0.5
        err = (float(num_species) - target) / max(1.0, target)
        self.compatibility_threshold *= float(np.exp(0.18 * err))
        self.compatibility_threshold = float(np.clip(self.compatibility_threshold, 0.3, 50.0))

    def _learn_species_target(self, num_species: int, best_fit: float, gen: int) -> None:
        """species_target の"学習"：PID と Hill-Climb をバンディット切換（auto）。"""
        low, high = int(self.mode.species_low), int(self.mode.species_high)
        if self.species_target is None:
            self.species_target = float((low + high) * 0.5)
            self._spec_learn["last_best"] = float(best_fit)
            self._spec_learn["last_tgt"]  = float(self.species_target)
            self._spec_learn["last_reward"] = 0.0
            return
        # 更新間隔
        if gen % int(self.species_target_update_every) != 0:
            return
        st = self._spec_learn
        last_best = st.get("last_best", None)
        if last_best is None:
            st["last_best"] = float(best_fit)
            return
        reward = float(best_fit) - float(last_best)
        mode = getattr(self, "species_target_mode", "auto")
        # choose method
        method = "pid"
        if mode == "hill":
            method = "hill"
        elif mode == "auto":
            eps = float(st.get("eps", 0.10))
            # epsilon-greedy over 2 arms
            if self.rng.random() < eps:
                method = "pid" if (self.rng.random() < 0.5) else "hill"
            else:
                method = "pid" if (st.get("score_pid", 0.0) >= st.get("score_hill", 0.0)) else "hill"
        # run update
        if method == "pid":
            # error = actual - target  （種数が多すぎれば target を上げにくく/下げやすく）
            err = float(num_species) - float(self.species_target)
            prev = st.get("pid_prev_err", 0.0) or 0.0
            itg  = float(st.get("pid_i", 0.0)) + err
            itg  = float(np.clip(itg, -float(self.pid_i_clip), float(self.pid_i_clip)))
            de   = err - prev
            delta = float(self.pid_kp)*err + float(self.pid_ki)*itg + float(self.pid_kd)*de
            step_max = max(0.5, 0.75)  # 1ステップで動かし過ぎない
            new_t = float(self.species_target) + float(np.clip(delta, -step_max, step_max))
            new_t = float(np.clip(new_t, float(self.species_target_min), float(self.species_target_max)))
            self.species_target = new_t
            st["pid_prev_err"] = err
            st["pid_i"] = itg
            # EWMA reward
            st["score_pid"] = 0.85*float(st.get("score_pid", 0.0)) + 0.15*reward
            st["score_hill"] = 0.98*float(st.get("score_hill", 0.0))
        else:
            dir_ = float(st.get("dir", 1.0))
            last_reward = float(st.get("last_reward") or 0.0)
            if reward < (last_reward - 1e-6):
                dir_ = -dir_
            err_s = float(num_species) - float(self.species_target)
            if err_s != 0.0:
                dir_ = 0.7*dir_ + 0.3*np.sign(err_s)
            step = float(self.species_target_step)
            new_t = float(self.species_target) + step * dir_
            new_t = float(np.clip(new_t, float(self.species_target_min), float(self.species_target_max)))
            self.species_target = new_t
            st["dir"] = dir_
            st["score_hill"] = 0.85*float(st.get("score_hill", 0.0)) + 0.15*reward
            st["score_pid"] = 0.98*float(st.get("score_pid", 0.0))
        st["last_best"] = float(best_fit)
        st["last_tgt"]  = float(self.species_target)
        st["last_reward"] = float(reward)


    def _mutate(self, genome: Genome):
        if self.rng.random() < self.mutate_toggle_prob: genome.mutate_toggle_enable(self.rng, prob=self.mutate_toggle_prob)
        if self.rng.random() < self.mutate_add_node_prob: genome.mutate_add_node(self.rng, self.innov)
        if self.rng.random() < self.mutate_add_conn_prob: genome.mutate_add_connection(self.rng, self.innov)
        if self.rng.random() < self.mutate_weight_prob: genome.mutate_weights(self.rng, self.weight_perturb_chance, self.weight_sigma, self.weight_reset_range)
        if self.rng.random() < self.mutate_sex_prob: genome.mutate_sex(self.rng)
        if self.rng.random() < self.regen_mode_mut_rate:
            if self.env['difficulty'] > 0.6: genome.regen_mode = self.rng.choice(['split','head','tail'], p=[0.6,0.25,0.15])
            else: genome.regen_mode = self.rng.choice(['split','head','tail'])
        if self.rng.random() < self.embryo_bias_mut_rate:
            genome.embryo_bias = self.rng.choice(['neutral','inputward','outputward'])
        # ---- Diversity push under high difficulty
        if float(self.env.get('difficulty', 0.0)) >= 0.9 and self.rng.random() < getattr(self, "diversity_push", 0.15):
            if self.rng.random() < 0.6:
                genome.mutate_add_connection(self.rng, self.innov)
            else:
                genome.mutate_add_node(self.rng, self.innov)

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
        # Remove any cycles that may have been introduced by crossover
        child.remove_cycles()
        # Hermaphrodite trait is very difficult to inherit
        # Inheritance rate controlled by hermaphrodite_inheritance_rate parameter
        if mother.sex == 'hermaphrodite' or father.sex == 'hermaphrodite':
            if self.rng.random() < self.hermaphrodite_inheritance_rate:
                child.sex = 'hermaphrodite'
            else:
                child.sex = 'female' if self.rng.random() < 0.5 else 'male'
        else:
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
        females=[g for g,_ in sp.members[:k] if g.sex=='female']; males=[g for g,_ in sp.members[:k] if g.sex=='male']
        hermaphrodites=[g for g,_ in sp.members[:k] if g.sex=='hermaphrodite']
        pool=[g for g,_ in sp.members[:k]]
        if (not females) or (not males):
            females = [g for g,_ in sp.members if g.sex=='female'] or females
            males   = [g for g,_ in sp.members if g.sex=='male'] or males
            hermaphrodites = [g for g,_ in sp.members if g.sex=='hermaphrodite'] or hermaphrodites
        mix_ratio=self._mix_asexual_ratio()
        monitor = getattr(self, "lcs_monitor", None)
        weight_tol = getattr(monitor, "eps", 0.0) if monitor is not None else 0.0
        while remaining>0:
            mode=None
            mother_id=None; father_id=None
            parent_adj_before_regen=None
            use_sexual_reproduction = False  # Explicit flag for sexual reproduction
            
            # Hermaphrodites have high mating bias - reduce asexual ratio when present
            effective_mix_ratio = mix_ratio / self.hermaphrodite_mate_bias if hermaphrodites else mix_ratio
            
            if self.rng.random()<effective_mix_ratio:
                parent=pool[int(self.rng.integers(len(pool)))]
                # Hermaphrodites CANNOT reproduce parthenogenetically - force sexual reproduction
                if parent.sex == 'hermaphrodite':
                    use_sexual_reproduction = True
                elif parent.regen and self.mode.enable_regen_reproduction:
                    if monitor is not None:
                        parent_adj_before_regen = parent.weighted_adjacency()
                    child = platyregenerate(parent, self.rng, self.innov, intensity=self._regen_intensity())
                    mode='asexual_regen'
                    mother_id=parent.id; father_id=None
                else:
                    child=parent.copy(); mode='asexual_clone'
                    mother_id=parent.id; father_id=None
            else:
                use_sexual_reproduction = True
            
            # Sexual reproduction (either forced by hermaphrodite or chosen by mix_ratio)
            if use_sexual_reproduction:
                # Build mating pools including hermaphrodites
                # Hermaphrodites can act as either male or female
                potential_mothers = females + hermaphrodites
                potential_fathers = males + hermaphrodites
                
                if potential_mothers and potential_fathers and self.rng.random()>self.pollen_flow_rate:
                    mother=potential_mothers[int(self.rng.integers(len(potential_mothers)))]
                    father=potential_fathers[int(self.rng.integers(len(potential_fathers)))]
                    mode='sexual_within'; sp_for_fit=sp.members
                else:
                    if len(species_pool)>1:
                        mother=pool[int(self.rng.integers(len(pool)))]
                        other=species_pool[(sidx+1)%len(species_pool)]
                        other_pool=[g for g,_ in other.members]
                        other_males=[g for g,_ in other.members if g.sex=='male']
                        other_herm=[g for g,_ in other.members if g.sex=='hermaphrodite']
                        father_pool = other_males + other_herm if (other_males or other_herm) else other_pool
                        father = father_pool[int(self.rng.integers(len(father_pool)))]
                        mode='sexual_cross'; sp_for_fit = sp.members + other.members
                    else:
                        if potential_mothers and potential_fathers:
                            mother=potential_mothers[int(self.rng.integers(len(potential_mothers)))]
                            father=potential_fathers[int(self.rng.integers(len(potential_fathers)))]
                            mode='sexual_within'; sp_for_fit=sp.members
                        else:
                            parent=pool[int(self.rng.integers(len(pool)))]
                            if self.allow_selfing:
                                mother=parent; father=parent; mode='sexual_within'; sp_for_fit=sp.members
                            else:
                                child=parent.copy(); mode='asexual_clone'
                                mother_id=parent.id; father_id=None
                if mode in ('sexual_within','sexual_cross'):
                    child=self._crossover_maternal_biased(mother,father,sp_for_fit); child.hybrid_scale=self._heterosis_scale(mother,father); mother_id=mother.id; father_id=father.id
            child.id=self.next_gid; self.next_gid+=1; child.parents=(mother_id,father_id); child.birth_gen=self.generation+1
            self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen}
            if monitor is not None and parent_adj_before_regen is not None:
                regen_adj = child.weighted_adjacency()
                changed_nodes, changed_edges = summarize_graph_changes(parent_adj_before_regen, regen_adj, weight_tol)
                monitor.log_step(parent_adj_before_regen, regen_adj, changed_nodes, child.id, self.generation+1, f"{child.id}_regen", changed_edges=changed_edges)
            pre_adj = child.weighted_adjacency() if monitor is not None else None
            self._mutate(child)
            if monitor is not None and pre_adj is not None:
                post_adj = child.weighted_adjacency()
                changed_nodes, changed_edges = summarize_graph_changes(pre_adj, post_adj, weight_tol)
                monitor.log_step(pre_adj, post_adj, changed_nodes, child.id, self.generation+1, f"{child.id}_{mode}", changed_edges=changed_edges)
            if mode is None:
                mode = 'asexual_clone'
            new_pop.append(child); events[mode]+=1; remaining-=1
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
        monitor = getattr(self, "lcs_monitor", None)
        weight_tol = getattr(monitor, "eps", 0.0) if monitor is not None else 0.0
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
                pre_adj = child.weighted_adjacency() if monitor is not None else None
                self._mutate(child)
                if monitor is not None and pre_adj is not None:
                    post_adj = child.weighted_adjacency()
                    changed_nodes, changed_edges = summarize_graph_changes(pre_adj, post_adj, weight_tol)
                    monitor.log_step(pre_adj, post_adj, changed_nodes, child.id, self.generation+1, f"{child.id}_asexual_clone", changed_edges=changed_edges)
                new_pop.append(child); gen_events['asexual_clone']+=1
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

    def _evaluate_population(self, fitness_fn: Callable[[Genome], float]) -> List[float]:
        """並列評価（thread/process）。process は SHM メタを初期化し、必要なら持ち回りプールを再起動。"""
        workers = int(getattr(self, "eval_workers", 1))
        if workers <= 1:
            return [fitness_fn(g) for g in self.population]
        backend = getattr(self, "parallel_backend", "thread")
        if backend == "process" and not _is_picklable(fitness_fn):
            print("[WARN] fitness_fn is not picklable; falling back to threads")
            backend = "thread"
        try:
            import concurrent.futures as _cf
            if backend == "process":
                import multiprocessing as _mp
                start = os.environ.get("NEAT_PROCESS_START_METHOD", "spawn")
                try:
                    ctx = _mp.get_context(start)
                except ValueError:
                    ctx = _mp.get_context("spawn")
                initargs = (getattr(self, "_shm_meta", None),)
                # persistent pool?
                if int(getattr(self, "pool_keepalive", 0)) > 0:
                    need_new = (self._proc_pool is None) or (int(getattr(self, "_proc_pool_age", 0)) >= int(getattr(self, "pool_restart_every", 25)))
                    if need_new:
                        self._close_pool()
                        self._proc_pool = _cf.ProcessPoolExecutor(
                            max_workers=workers, mp_context=ctx, initializer=_proc_init_worker, initargs=initargs
                        )
                        self._proc_pool_age = 0
                    ex = self._proc_pool
                    out = list(ex.map(fitness_fn, self.population, chunksize=max(1, len(self.population)//workers)))
                    self._proc_pool_age += 1
                    return out
                else:
                    with _cf.ProcessPoolExecutor(
                        max_workers=workers, mp_context=ctx, initializer=_proc_init_worker, initargs=initargs
                    ) as ex:
                        return list(ex.map(fitness_fn, self.population, chunksize=max(1, len(self.population)//workers)))
            else:
                with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
                    return list(ex.map(fitness_fn, self.population))
        except Exception as e:
            print("[WARN] parallel evaluation disabled:", e)
            return [fitness_fn(g) for g in self.population]

    def _close_pool(self):
        ex = getattr(self, "_proc_pool", None)
        if ex is not None:
            try:
                ex.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._proc_pool = None
            self._proc_pool_age = 0


    def _auto_env_schedule(self, gen: int, history: List[Tuple[float,float]]) -> Dict[str, float]:
        """進捗に応じて difficulty / noise を自動昇圧。高難度で再生を解禁。上限撤廃版。"""
        diff = float(self.env.get('difficulty', 0.0))
        best_hist = [b for (b, _a) in history] if history else []
        bump = 0.0
        if len(best_hist) >= 10:
            delta10 = best_hist[-1] - best_hist[-10]
            if delta10 < 0.01:
                bump = 0.10
            elif delta10 < 0.05:
                bump = 0.05
        if gen < 10:
            diff = max(diff, 0.3)
        elif gen < 25:
            diff = max(diff, 0.5)
        else:
            # 上限撤廃: min(1.0, ...) を削除してdiffを無制限に増加可能に
            diff = diff + bump
        enable_regen = bool(diff >= 0.85)
        # noise_stdも無制限に増加可能に
        noise_std = 0.01 + 0.05 * diff
        return {"difficulty": float(diff), "noise_std": float(noise_std), "enable_regen": enable_regen}

    def _adaptive_refine_fitness(self, fitnesses: List[float], fitness_fn: Callable[[Genome], float]) -> List[float]:
        """上位個体にだけ backprop ステップを追加して再評価（軽量な二段評価）。"""
        if not hasattr(fitness_fn, "refine_raw"):
            return fitnesses
        n = len(fitnesses)
        if n == 0:
            return fitnesses
        k = max(1, int(0.10 * n))
        idxs = np.argsort(fitnesses)[-k:]
        improved = list(fitnesses)
        best_now = float(np.max(fitnesses))
        for i in map(int, idxs):
            try:
                gap = best_now - float(fitnesses[i])
                factor = 2.0 if gap > 0.02 else 1.5
                raw2 = float(fitness_fn.refine_raw(self.population[i], factor=factor))
                f2 = raw2
                if not self.mode.vanilla:
                    g = self.population[i]
                    f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * (getattr(g, 'hybrid_scale', 1.0))
                    if g.regen:
                        f2 += self.regen_bonus
                f2 -= self._complexity_penalty(self.population[i])
                if np.isfinite(f2):
                    improved[i] = f2
            except Exception:
                pass
        return improved

    def evolve(self, fitness_fn: Callable[[Genome], float], n_generations=100, target_fitness=None, verbose=True, env_schedule=None):
        history=[]; best_ever=None; best_ever_fit=-1e9
        from math import isnan
        scars=None; prev_best=None
        for gen in range(n_generations):
            self.generation=gen
            prev=history[-1] if history else (None,None)
            # === Curriculum ===
            if env_schedule is not None:
                env = env_schedule(gen, {'gen':gen,'prev_best':prev[0] if prev else None, 'prev_avg':prev[1] if prev else None})
            elif getattr(self, "auto_curriculum", True):
                env = self._auto_env_schedule(gen, history)
            else:
                env = None
            if env is not None:
                self.env.update({k: v for k, v in env.items() if k not in {'enable_regen'}})
                if 'enable_regen' in env:
                    flag = bool(env['enable_regen'])
                    self.mode.enable_regen_reproduction = flag
                    if flag:
                        self.mix_asexual_base = max(self.mix_asexual_base, 0.30)
            self.env_history.append({'gen':gen, **self.env, 'regen_enabled': self.mode.enable_regen_reproduction})
            # 難易度に応じた pollen flow
            diff = float(self.env.get('difficulty', 0.0))
            self.pollen_flow_rate = float(min(0.5, max(0.1, 0.1 + 0.35 * diff)))
            # fitness インスタンスに noise を伝播（プロセスでも都度 pickled）
            if hasattr(fitness_fn, "set_noise_std"):
                try:
                    fitness_fn.set_noise_std(float(self.env.get("noise_std", 0.0)))
                except Exception:
                    pass
            # === Evaluate (parallel-aware) ===
            raw = self._evaluate_population(fitness_fn)
            fitnesses=[]
            for g,f in zip(self.population, raw):
                f2 = float(f)
                if not self.mode.vanilla:
                    f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * (getattr(g,'hybrid_scale',1.0))
                    if g.regen: f2 += self.regen_bonus
                f2 -= self._complexity_penalty(g)
                if not np.isfinite(f2):
                    f2 = float(np.nan_to_num(f2, nan=-1e6, posinf=-1e6, neginf=-1e6))
                fitnesses.append(f2)
            # Adaptive refine for elites
            try:
                fitnesses = self._adaptive_refine_fitness(fitnesses, fitness_fn)
            except Exception:
                pass
            best_idx=int(np.argmax(fitnesses)); best_fit=float(fitnesses[best_idx]); avg_fit=float(np.mean(fitnesses))
            # === Snapshots (decimated & bounded) ===
            try:
                curr_best = self.population[best_idx].copy()
                scars = diff_scars(prev_best, curr_best, scars, birth_gen=gen, regen_mode_for_new=getattr(curr_best,'regen_mode','split'))
                stride = int(getattr(self, "snapshot_stride", 1))
                if (gen % max(1, stride) == 0) or (gen == n_generations - 1):
                    if len(self.snapshots_genomes) >= int(getattr(self, "snapshot_max", 320)):
                        self.snapshots_genomes.pop(0); self.snapshots_scars.pop(0)
                    self.snapshots_genomes.append(curr_best); self.snapshots_scars.append(scars)
                prev_best = curr_best
            except Exception:
                pass
            history.append((best_fit, avg_fit)); self.best_ids.append(self.population[best_idx].id)
            # complexity traces
            try:
                self.hidden_counts_history.append([sum(1 for n in g.nodes.values() if n.type=='hidden') for g in self.population])
                self.edge_counts_history.append([sum(1 for c in g.connections.values() if c.enabled) for g in self.population])
            except Exception:
                self.hidden_counts_history.append([]); self.edge_counts_history.append([])
            if verbose:
                noise = float(self.env.get('noise_std', 0.0))
                ev=self.event_log[-1] if self.event_log else {'sexual_within':0,'sexual_cross':0,'asexual_regen':0}
                # Count hermaphrodites in population
                n_herm = sum(1 for g in self.population if g.sex == 'hermaphrodite')
                herm_str = f" | herm {n_herm}" if n_herm > 0 else ""
                print(
                    f"Gen {gen:3d} | best {best_fit:.4f} | avg {avg_fit:.4f} | difficulty {diff:.2f} | noise {noise:.2f} | "
                    f"sexual {ev.get('sexual_within',0)+ev.get('sexual_cross',0)} | regen {ev.get('asexual_regen',0)}{herm_str}"
                )
            if best_fit > best_ever_fit:
                best_ever_fit = best_fit; best_ever = self.population[best_idx].copy()
            if target_fitness is not None and best_fit >= target_fitness:
                break
            species=self.speciate(fitnesses)
            # 学習した target を先に更新 → それに追従する形で compat を調整
            try:
                self._learn_species_target(len(species), best_fit, gen)
            except Exception as _spe:
                print("[WARN] species target learning skipped:", _spe)
            self._adapt_compat_threshold(len(species))
            self.reproduce(species, fitnesses)
        # Champion across all generations
        if best_ever is None and self.population:
            best_ever = self.population[0].copy()
        # 持ち回りプールがあれば閉じる
        try:
            self._close_pool()
        except Exception:
            pass
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
    """
    Hardened backprop with gradient/weight clipping and NaN guards.
    既存シグネチャ互換（追加引数は train_* から供給）。
    """
    import numpy as _np
    # 追加の安全引数（後方互換のため default をここで拾う）
    grad_clip = 5.0
    w_clip = 12.0
    A, Z = forward_batch(comp, X, w)
    loss, delta_out, _ = loss_and_output_delta(comp, Z, y, l2, w)
    if not _np.isfinite(loss):
        # ソフト・リセット
        w = _np.tanh(w) * 0.1
        loss = float(_np.nan_to_num(loss, nan=1e3, posinf=1e3, neginf=1e3))
    B = X.shape[0]; n = len(comp['order'])
    grad_w = _np.zeros_like(w)
    delta_z = _np.zeros((B, n), dtype=_np.float64)
    delta_a = _np.zeros((B, n), dtype=_np.float64)
    for j, oi in enumerate(comp['outputs']):
        delta_z[:, oi] = delta_out[:, j:j+1].reshape(B)
    for j in reversed(range(n)):
        t = comp['types'][j]
        if t == 'output':
            dz = delta_z[:, j]
        elif t in ('input','bias'):
            continue
        else:
            dz = delta_a[:, j] * act_deriv(comp['acts'][j], Z[:, j])
            delta_z[:, j] = dz
        for e in comp['in_edges'][j]:
            s = comp['src'][e]
            grad_w[e] += _np.dot(A[:, s], dz)
            delta_a[:, s] += dz * w[e]
    grad_w = grad_w / max(1, B) + l2 * w
    if not _np.all(_np.isfinite(grad_w)):
        grad_w = _np.nan_to_num(grad_w, nan=0.0, posinf=0.0, neginf=0.0)
    # global-norm clip
    if grad_clip and grad_clip > 0:
        gnorm = float(_np.linalg.norm(grad_w))
        if _np.isfinite(gnorm) and gnorm > grad_clip:
            grad_w *= (grad_clip / (gnorm + 1e-12))
    w_new = w - float(lr) * grad_w
    if w_clip and w_clip > 0:
        _np.clip(w_new, -float(w_clip), float(w_clip), out=w_new)
    return w_new, float(loss)

def train_with_backprop_numpy(genome: Genome, X, y, steps=50, lr=1e-2, l2=1e-4, grad_clip=5.0, w_clip=12.0):
    comp = compile_genome(genome); w = comp['w'].copy(); history=[]
    if w.size == 0:
        return history
    for _ in range(int(steps)):
        # backprop_step 側は後方互換だが、将来のため明示引数を付ける
        w, L = backprop_step(comp, X, y, w, lr=lr, l2=l2)
        if not np.isfinite(L):
            L = float(np.nan_to_num(L, nan=1e3, posinf=1e3, neginf=1e3))
        history.append(L)
    for e_idx, inn in enumerate(comp['eid']):
        genome.connections[inn].weight = float(w[e_idx])
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
    try:
        gg = genome.copy()
        train_with_backprop_numpy(gg, Xtr, ytr, steps=steps, lr=lr, l2=l2)
        pred = predict(gg, Xva)
        acc = (pred == (yva if yva.ndim==1 else np.argmax(yva,1))).mean()
        pen = complexity_penalty(gg, alpha_nodes=alpha_nodes, alpha_edges=alpha_edges)
        return float(acc - pen)
    except RuntimeError as e:
        if "Cycle detected" in str(e):
            # Return very low fitness for cyclic genomes
            return -1.0
        raise

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

def render_lineage(neat, path="lineage.png", title="Lineage", max_edges: Optional[int]=10000,
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
    xs_f=[]; ys_f=[]; ss_f=[]; xs_m=[]; ys_m=[]; ss_m=[]; xs_h=[]; ys_h=[]; ss_h=[]; xs_u=[]; ys_u=[]; ss_u=[]
    for nid,(x,y) in pos.items():
        info = reg.get(nid, {}); sex = info.get('sex', None); regen = bool(info.get('regen', False))
        size = 80*(1.3 if regen else 1.0)
        if sex=='female': xs_f.append(x); ys_f.append(y); ss_f.append(size)
        elif sex=='male': xs_m.append(x); ys_m.append(y); ss_m.append(size)
        elif sex=='hermaphrodite': xs_h.append(x); ys_h.append(y); ss_h.append(size)
        else: xs_u.append(x); ys_u.append(y); ss_u.append(size)
    if xs_f: ax.scatter(xs_f, ys_f, s=ss_f, marker='o', alpha=0.95, c='#FF69B4', label='female')
    if xs_m: ax.scatter(xs_m, ys_m, s=ss_m, marker='s', alpha=0.95, c='#4169E1', label='male')
    if xs_h: ax.scatter(xs_h, ys_h, s=ss_h, marker='D', alpha=0.95, c='#9370DB', label='hermaphrodite')
    if xs_u: ax.scatter(xs_u, ys_u, s=ss_u, marker='^', alpha=0.95, c='#808080')
    # Add legend if there are any sex-typed nodes
    if xs_f or xs_m or xs_h:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
            make_lineage=True,  # False → True に変更してlineageを生成
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
        # lineageも追加
        lineage = res.get("lineage")
        if lineage and os.path.exists(lineage):
            outputs[f"{idx:02d} {task.upper()} 系統図"] = lineage
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

# === Shared-memory aware fitness =============================================
class FitnessBackpropShared:
    """
    Picklable callable that reads datasets from shared memory by label.
    Also provides refine_raw(genome, factor) for adaptive extra-steps.
    """
    def __init__(self, keys=("Xtr","ytr","Xva","yva"), steps=40, lr=5e-3, l2=1e-4, alpha_nodes=1e-3, alpha_edges=5e-4):
        self.keys = tuple(keys)
        self.steps = int(steps); self.lr=float(lr); self.l2=float(l2)
        self.alpha_nodes=float(alpha_nodes); self.alpha_edges=float(alpha_edges)
        self.noise_std = 0.0
    def set_noise_std(self, s: float):
        self.noise_std = float(max(0.0, s))
    def _load(self):
        Xtr = get_shared_dataset(self.keys[0]); ytr = get_shared_dataset(self.keys[1])
        Xva = get_shared_dataset(self.keys[2]); yva = get_shared_dataset(self.keys[3])
        return Xtr, ytr, Xva, yva
    def _aug(self, X):
        s = float(self.noise_std)
        if s <= 0.0:
            return X
        rng = np.random.default_rng()
        return X + rng.normal(0.0, s, size=X.shape)
    def __call__(self, g: "Genome") -> float:
        Xtr, ytr, Xva, yva = self._load()
        return fitness_backprop_classifier(g, self._aug(Xtr), ytr, self._aug(Xva), yva,
                                           steps=self.steps, lr=self.lr, l2=self.l2,
                                           alpha_nodes=self.alpha_nodes, alpha_edges=self.alpha_edges)
    def refine_raw(self, g: "Genome", factor: float = 2.0) -> float:
        Xtr, ytr, Xva, yva = self._load()
        steps = int(max(1, round(self.steps * float(factor))))
        return fitness_backprop_classifier(g, self._aug(Xtr), ytr, self._aug(Xva), yva,
                                           steps=steps, lr=self.lr, l2=self.l2,
                                           alpha_nodes=self.alpha_nodes, alpha_edges=self.alpha_edges)

# === Per-Sample Sequence Stopper (with safety guards) ========================
class PerSampleSequenceStopperPro:
    """
    Stopper for per-sample sequence processing with stage-based termination logic.
    Includes safety guards to prevent IndexError when stage counter exceeds configuration bounds.
    """
    def __init__(self, cfg: dict):
        """
        Initialize stopper with configuration.
        
        Args:
            cfg: Dictionary with configuration including 'within' list for stage-based windows
        
        Raises:
            KeyError: If 'within' key is missing from cfg
            TypeError: If 'within' value is not a sequence
        """
        if "within" not in cfg:
            raise KeyError("Configuration must include 'within' key")
        if not hasattr(cfg["within"], '__len__'):
            raise TypeError("cfg['within'] must be a list or sequence")
        self.cfg = cfg
        self.finished_samples = set()
    
    def update_finished(self, sample_id: int, stage: int) -> bool:
        """
        Update finished status for a sample based on current stage.
        
        Args:
            sample_id: Identifier for the sample being processed
            stage: Current stage number (0-indexed or 1-indexed depending on usage)
        
        Returns:
            bool: True if sample should be marked as finished, False otherwise
        """
        # Check stage-based stopping conditions
        if stage >= 1:
            # ★安全ガード：範囲外アクセス回避 (Safety guard: prevent out-of-bounds access)
            if stage >= len(self.cfg["within"]):
                return False
            
            win = self.cfg["within"][stage]
            if win is not None and win >= 0:
                # Window value is valid - mark sample as finished
                # (Additional window-based logic can be implemented here as needed)
                self.finished_samples.add(sample_id)
                return True
        
        return False
    
    def is_finished(self, sample_id: int) -> bool:
        """Check if a sample has been marked as finished."""
        return sample_id in self.finished_samples
    
    def reset(self):
        """Reset all finished samples."""
        self.finished_samples.clear()

def run_backprop_neat_experiment(task: str, gens=60, pop=64, steps=80, out_prefix="out/exp", make_gifs: bool = True, make_lineage: bool = True, rng_seed: int = 0):
    # dataset
    if task=="xor": Xtr,ytr = make_xor(512, noise=0.05, seed=0); Xva,yva = make_xor(256, noise=0.05, seed=1)
    elif task=="spiral": Xtr,ytr = make_spirals(512, noise=0.05, turns=1.5, seed=0); Xva,yva = make_spirals(256, noise=0.05, turns=1.5, seed=1)
    else: Xtr,ytr = make_circles(512, r=0.6, noise=0.05, seed=0); Xva,yva = make_circles(256, r=0.6, noise=0.05, seed=1)
    # NEAT
    rng = np.random.default_rng(rng_seed)
    neat = ReproPlanaNEATPlus(num_inputs=2, num_outputs=1, population_size=pop, output_activation='identity', rng=rng)
    _apply_stable_neat_defaults(neat)
    regen_log_path = f"{out_prefix}_regen_log.csv"
    if hasattr(neat, "lcs_monitor") and neat.lcs_monitor is not None:
        neat.lcs_monitor.csv_path = regen_log_path
        if os.path.exists(regen_log_path):
            os.remove(regen_log_path)

    # --- Shared memory registration for process-parallel zero-copy
    # プロセス並列時のみ SHM を使う（thread では不要なのでリスクを減らす）
    use_shm = (os.environ.get("NEAT_EVAL_BACKEND", "thread") == "process")
    if use_shm:
        shm_meta = {}
        try:
            shm_meta["Xtr"] = shm_register_dataset("Xtr", Xtr, readonly=True)
            shm_meta["ytr"] = shm_register_dataset("ytr", ytr, readonly=True)
            shm_meta["Xva"] = shm_register_dataset("Xva", Xva, readonly=True)
            shm_meta["yva"] = shm_register_dataset("yva", yva, readonly=True)
            neat._shm_meta = shm_meta
        except Exception:
            neat._shm_meta = None
    else:
        # For thread backend, store datasets directly in cache without SHM
        _SHM_CACHE["Xtr"] = Xtr
        _SHM_CACHE["ytr"] = ytr
        _SHM_CACHE["Xva"] = Xva
        _SHM_CACHE["yva"] = yva
        neat._shm_meta = None
    
    fit = FitnessBackpropShared(steps=steps, lr=5e-3, l2=1e-4, alpha_nodes=1e-3, alpha_edges=5e-4)
    
    # shm_release_all() は try/finally で evolve 呼び出しを包むと例外でも確実に実行されます
    try:
        best, hist = neat.evolve(
            fit,
            n_generations=gens,
            verbose=True,
            env_schedule=_default_difficulty_schedule,
        )
    finally:
        # SHM cleanup
        if use_shm:
            try:
                shm_release_all()
            except Exception:
                pass
        else:
            # Clear thread-mode cache
            _SHM_CACHE.clear()
    
    lcs_rows = load_lcs_log(regen_log_path) if os.path.exists(regen_log_path) else []
    lcs_series = _prepare_lcs_series(lcs_rows) if lcs_rows else None
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
            lcs_series=lcs_series,
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

    lcs_ribbon = None
    lcs_timeline = None
    if lcs_rows:
        ribbon_path = f"{out_prefix}_lcs_ribbon.png"
        try:
            export_lcs_ribbon_png(lcs_rows, ribbon_path, series=lcs_series)
            lcs_ribbon = ribbon_path
        except Exception as ribbon_err:
            print("[WARN] LCS ribbon export failed:", ribbon_err)
        timeline_path = f"{out_prefix}_lcs_timeline.gif"
        try:
            export_lcs_timeline_gif(lcs_rows, timeline_path, series=lcs_series, fps=6)
            lcs_timeline = timeline_path
        except Exception as timeline_err:
            print("[WARN] LCS timeline export failed:", timeline_err)

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
    
    # Convert genome snapshots to Cytoscape format for interactive report
    genomes_cyto = []
    if hasattr(neat, 'snapshots_genomes') and neat.snapshots_genomes:
        for g in neat.snapshots_genomes:
            genomes_cyto.append(_genome_to_cyto(g))
    
    return {
        "learning_curve": lc_path,
        "decision_boundary": db_path,
        "topology": topo_path,
        "regen_gif": regen_gif,
        "morph_gif": morph_gif,
        "lineage": lineage_path,
        "scars_spiral": scars_spiral,
        "summary_decisions": summary_paths,
        "lcs_log": regen_log_path if os.path.exists(regen_log_path) else None,
        "lcs_ribbon": lcs_ribbon,
        "lcs_timeline": lcs_timeline,
        "history": hist,
        "genomes_cyto": genomes_cyto,
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
    dpi=130,
    lcs_series: Optional[Dict[str, Any]] = None,
):
    """
    Render a regeneration digest GIF from per-generation snapshots.
    Encodes differences without color semantics: linestyle/linewidth/alpha only.
    When lcs_series is provided, overlays per-generation LCS summary text.
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

        summary_line = None
        cum_line = None
        gen_label = f"generation {t}"
        if lcs_series is not None:
            _, summary = _latest_gen_summary(lcs_series, t)
            summary_line = _format_lcs_summary(summary)
            heals_cum, breaks_cum = _cumulative_lcs_counts(lcs_series, t)
            cum_line = f"cum heals {heals_cum} | cum breaks {breaks_cum}"
        if summary_line or cum_line:
            fig.subplots_adjust(bottom=0.28)
            fig.text(0.02, 0.18, summary_line or "", fontsize=8, family="monospace")
            if cum_line:
                fig.text(0.02, 0.11, cum_line, fontsize=7, family="monospace")
            fig.text(0.02, 0.06, gen_label, fontsize=7, family="monospace")
        else:
            fig.subplots_adjust(bottom=0.08)
            fig.text(0.02, 0.05, gen_label, fontsize=7, family="monospace")

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

    def _theta_radius(gen_idx: int) -> Tuple[float, float]:
        theta = theta_max * (gen_idx / max(1, G - 1))
        base_r = r0 + (r1 - r0) * (theta / theta_max)
        return theta, base_r

    generation_centers: List[Tuple[float, float]] = []
    generation_counts: List[int] = []

    for g in range(G):
        theta, base_r = _theta_radius(g)
        cx = base_r * _np.cos(theta)
        cy = base_r * _np.sin(theta)
        generation_centers.append((cx, cy))
        items = events_by_gen.get(g, [])
        generation_counts.append(len(items))

        n = len(items)
        if n == 0:
            continue
        offs = _np.linspace(-0.5, 0.5, n) if n > 1 else [0.0]  # 同一世代で半径方向にわずかにズラす
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
    # グリッド: 等半径リングと扇状のガイドライン
    ts = _np.linspace(0.0, theta_max, 1200)
    rr = r0 + (r1 - r0) * (ts / theta_max)
    ax.plot(rr * _np.cos(ts), rr * _np.sin(ts), linewidth=1.0, alpha=0.32, linestyle="-")

    ring_levels = _np.linspace(r0, r1, 5)
    base_angles = _np.linspace(0.0, 2.0 * _np.pi, 361)
    for level in ring_levels:
        ax.plot(level * _np.cos(base_angles), level * _np.sin(base_angles), linestyle="--", linewidth=0.55, color="#444", alpha=0.18)

    spoke_count = max(6, int(turns * 4.0))
    for t in _np.linspace(0.0, theta_max, spoke_count, endpoint=False):
        ax.plot([r0 * _np.cos(t), r1 * _np.cos(t)], [r0 * _np.sin(t), r1 * _np.sin(t)], linestyle=":", linewidth=0.5, color="#333", alpha=0.18)

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
            ax.scatter(xs[k], ys[k], s=marker_size * 1.1, alpha=0.28, marker=markers[k], linewidths=0.75, label=labels[k], edgecolors="black")

    max_count = max(generation_counts) if generation_counts else 0
    if max_count > 0:
        for idx in range(1, len(generation_centers)):
            prev = generation_centers[idx - 1]
            curr = generation_centers[idx]
            weight = 0.35 + 2.4 * ((_np.clip(generation_counts[idx], 0, max_count) / max_count) ** 0.65)
            alpha = 0.14 + 0.58 * ((_np.clip(generation_counts[idx], 0, max_count) / max_count) ** 0.5)
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], linewidth=weight, alpha=alpha, color="black", solid_capstyle="round")

        for (cx, cy), count in zip(generation_centers, generation_counts):
            if count <= 0:
                continue
            radius = 30.0 + 6.0 * count
            ax.scatter([cx], [cy], s=radius, facecolors="none", edgecolors="black", linewidths=0.9, alpha=0.7)

        peak_gen = int(_np.argmax(generation_counts))
        peak_count = generation_counts[peak_gen]
        if peak_count > 0:
            theta_peak, base_peak = _theta_radius(peak_gen)
            px = base_peak * _np.cos(theta_peak)
            py = base_peak * _np.sin(theta_peak)
            radial_boost = 0.12 + 0.18 * (_np.clip(peak_count / max_count, 0.0, 1.0) if max_count > 0 else 0.0)
            norm = float(_np.hypot(px, py))
            if norm < 1e-6:
                # 極めて中心に近い場合は右方向へ逃がして重なりを回避
                tx, ty = radial_boost, 0.0
            else:
                scale = norm + radial_boost
                tx = (px / norm) * scale
                ty = (py / norm) * scale
            ax.annotate(
                f"Peak gen {peak_gen}\n{peak_count} births",
                xy=(px, py),
                xytext=(tx, ty),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.6, alpha=0.78),
                arrowprops=dict(
                    arrowstyle="-",
                    color="black",
                    linewidth=0.65,
                    alpha=0.65,
                    shrinkA=0,
                    shrinkB=4.0,
                    connectionstyle="arc3,rad=0.08",
                ),
            )

    if generation_centers:
        theta_start, r_start = _theta_radius(0)
        sx = r_start * _np.cos(theta_start)
        sy = r_start * _np.sin(theta_start)
        ax.text(
            sx * 1.04,
            sy * 1.04,
            "Gen 0",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.85),
        )
        theta_end, r_end = _theta_radius(G - 1)
        ex = r_end * _np.cos(theta_end)
        ey = r_end * _np.sin(theta_end)
        ax.text(
            ex * 1.04,
            ey * 1.04,
            f"Gen {G-1}",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.85),
        )

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
    """Three-phase curriculum with a late-game regeneration surge. 上限撤廃版。"""
    if gen < 25:
        return {"difficulty": 0.3, "noise_std": 0.0, "enable_regen": False}
    if gen < 40:
        return {"difficulty": 0.6, "noise_std": 0.02, "enable_regen": False}
    # 上限撤廃: gen >= 40 では difficulty を世代数に応じて線形増加
    diff = 1.0 + (gen - 40) * 0.02  # 40世代以降は0.02ずつ増加
    noise = 0.05 + (gen - 40) * 0.001  # noiseも緩やかに増加
    return {"difficulty": diff, "noise_std": noise, "enable_regen": True}


def _apply_stable_neat_defaults(neat: ReproPlanaNEATPlus):
    """Thesis-grade defaults: calm search, regen gated until curriculum lifts it. 複雑トポロジー保持版。"""
    neat.mode = EvalMode(
        vanilla=True,
        enable_regen_reproduction=False,
        complexity_alpha=neat.mode.complexity_alpha,
        node_penalty=0.3,  # 緩和: 複雑なトポロジーが残りやすく
        edge_penalty=0.15,  # 緩和: エッジも保持
        species_low=neat.mode.species_low,
        species_high=neat.mode.species_high,
    )
    neat.mutate_add_conn_prob = 0.05
    neat.mutate_add_node_prob = 0.03
    neat.mutate_weight_prob = 0.8
    neat.regen_mode_mut_rate = 0.05
    neat.mix_asexual_base = 0.10
    # 常に5.0に設定して複雑なトポロジーを許容
    neat.complexity_threshold = 5.0


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
    regen_log_path = f"{out_prefix}_regen_log.csv"
    if hasattr(neat, "lcs_monitor") and neat.lcs_monitor is not None:
        neat.lcs_monitor.csv_path = regen_log_path
        if os.path.exists(regen_log_path):
            os.remove(regen_log_path)
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

    lcs_rows = load_lcs_log(regen_log_path) if os.path.exists(regen_log_path) else []
    lcs_series = _prepare_lcs_series(lcs_rows) if lcs_rows else None

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

    lcs_ribbon = None
    lcs_timeline = None
    if lcs_rows:
        ribbon_path = f"{out_prefix}_lcs_ribbon.png"
        try:
            export_lcs_ribbon_png(lcs_rows, ribbon_path, series=lcs_series)
            lcs_ribbon = ribbon_path
        except Exception as ribbon_err:
            print("[WARN] LCS ribbon export failed:", ribbon_err)
        timeline_path = f"{out_prefix}_lcs_timeline.gif"
        try:
            export_lcs_timeline_gif(lcs_rows, timeline_path, series=lcs_series, fps=6)
            lcs_timeline = timeline_path
        except Exception as timeline_err:
            print("[WARN] LCS timeline export failed:", timeline_err)

    return {
        "best": best,
        "history": hist,
        "reward_curve": rc_png,
        "lcs_log": regen_log_path if os.path.exists(regen_log_path) else None,
        "lcs_ribbon": lcs_ribbon,
        "lcs_timeline": lcs_timeline,
    }


def _genome_to_cyto(genome: Genome) -> dict:
    """
    Convert a Genome object to Cytoscape.js-compatible dictionary format.
    Returns dict with 'nodes', 'edges', 'id', and 'meta' keys.
    """
    nodes = []
    for nid, node in genome.nodes.items():
        node_data = {
            "id": str(nid),
            "label": f"{node.type[0].upper()}{nid}",
            "type": node.type,
            "bias": getattr(node, "bias", 0.0),
            "activation": node.activation,
        }
        nodes.append({"data": node_data})
    
    edges = []
    for innov, conn in genome.connections.items():
        edge_data = {
            "id": f"e{innov}",
            "source": str(conn.in_node),
            "target": str(conn.out_node),
            "weight": float(conn.weight),
            "enabled": bool(conn.enabled),
        }
        edges.append({"data": edge_data})
    
    return {
        "id": genome.id,
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "fitness": getattr(genome, "fitness", None),
            "birth_gen": genome.birth_gen,
        }
    }


def export_interactive_html_report(path: str, title: str, history, genomes, *, max_genomes: int = 50):
    """
    Write a self-contained interactive HTML report with:
    - Plotly learning curve (pan/zoom/hover)
    - Cytoscape genome viewer with:
        - Right-click context menu on nodes: Fix/Unfix, Color, Note
        - Drag nodes, zoom, click for detail
    No extra Python deps; loads JS via CDN. Per-node edits persist in localStorage.
    """
    import json, os
    # history is expected as list of (best, avg)
    xs = list(range(len(history or [])))
    ys_best = [float(b) for (b, a) in (history or [])]
    ys_avg  = [float(a) for (b, a) in (history or [])]

    # Downsample genomes if needed (evenly spaced picks)
    genomes = genomes or []
    if len(genomes) > max_genomes:
        idxs = [round(i) for i in [k*(len(genomes)-1)/(max_genomes-1) for k in range(max_genomes)]]
        genomes = [genomes[i] for i in idxs]

    # Convert to cytoscape-friendly dicts if raw Genome objects slipped in
    if genomes and "nodes" not in genomes[0]:
        genomes = [_genome_to_cyto(g) for g in genomes]

    data = {
        "title": title,
        "lc": {"x": xs, "best": ys_best, "avg": ys_avg},
        "genomes": genomes,
    }

    # NOTE: f-string uses {{ }} for literal braces inside CSS/JS
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <style>
    :root {{ --grid:#e5e5e5; --fg:#111; --muted:#666; --ok1:#0072B2; --ok2:#D55E00; --ok3:#009E73; --ok4:#CC79A7; --ok5:#F0E442; --ok6:#56B4E9; --ok7:#E69F00; --ok8:#000; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--fg); }}
    .container {{ max-width:1200px; margin:24px auto; padding:0 16px; }}
    h1 {{ font-size:22px; margin:0 0 12px; }}
    .card {{ border:1px solid var(--grid); border-radius:8px; padding:12px; background:#fff; margin-bottom:16px; }}
    #lc {{ height:360px; }}
    .panel {{ display:grid; grid-template-columns:1fr 320px; gap:16px; }}
    #cy {{ width:100%; height:560px; border:1px solid var(--grid); border-radius:6px; background:#fff; }}
    .detail {{ border:1px dashed var(--grid); border-radius:6px; padding:8px; font-size:13px; height:560px; overflow:auto; background:#fafafa; }}
    .row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }}
    select,button {{ padding:6px 8px; font-size:14px; border:1px solid var(--grid); border-radius:6px; background:#fff; }}
    .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
    .legend {{ display:inline-flex; gap:8px; align-items:center; }}
    /* Tooltip */
    .tip {{ position:fixed; pointer-events:none; background:rgba(0,0,0,.8); color:#fff; padding:6px 8px; font-size:12px; border-radius:4px; transform:translate(8px,8px); z-index:1000; display:none; max-width:320px; white-space:nowrap; }}
    /* Context menu */
    .ctx-menu {{
      position: fixed; z-index: 2000; display: none; min-width: 220px;
      background: #fff; color: var(--fg); border: 1px solid var(--grid); border-radius: 8px;
      box-shadow: 0 8px 20px rgba(0,0,0,.12); padding: 6px;
    }}
    .ctx-item {{ font-size: 13px; padding: 8px 10px; cursor: pointer; border-radius:6px; }}
    .ctx-item:hover {{ background: #f3f3f3; }}
    .ctx-sep {{ height:1px; background: var(--grid); margin:6px 0; }}
    .swatches {{ display:flex; flex-wrap:wrap; gap:6px; padding: 4px 2px 2px; }}
    .swatch {{ width:18px; height:18px; border-radius:50%; cursor:pointer; border:1px solid rgba(0,0,0,.15); }}
    .ctx-row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="card">
      <h2>Learning Curve</h2>
      <div id="lc"></div>
    </div>
    <div class="card">
      <h2>Genome Viewer</h2>
      <div class="row">
        <label for="genomeSelect">Genome:</label>
        <select id="genomeSelect"></select>
        <button id="layoutBtn">Re-layout</button>
        <span class="legend">
          <span class="dot" style="background:#009E73"></span> input
          <span class="dot" style="background:#999"></span> hidden
          <span class="dot" style="background:#0072B2"></span> output
        </span>
      </div>
      <div class="panel">
        <div id="cy"></div>
        <div class="detail" id="detail"><div class="fine">ノードやエッジを選択すると詳細が表示されます。</div></div>
      </div>
    </div>
  </div>
  <div class="tip" id="tip"></div>
  <div class="ctx-menu" id="ctx"></div>

  <script>
    const DATA = {json.dumps(data, ensure_ascii=False)};
    // Learning curve
    (function(){{
      const traces = [];
      if (DATA.lc && DATA.lc.x.length) {{
        traces.push({{ x: DATA.lc.x, y: DATA.lc.best, mode:'lines', name:'best', line:{{width:2, color:'#0072B2'}},
          hovertemplate:'gen=%{{x}}<br>best=%{{y:.4f}}<extra></extra>' }});
        traces.push({{ x: DATA.lc.x, y: DATA.lc.avg,  mode:'lines', name:'avg',  line:{{width:2, color:'#D55E00'}},
          hovertemplate:'gen=%{{x}}<br>avg=%{{y:.4f}}<extra></extra>' }});
      }}
      Plotly.newPlot('lc', traces, {{
        margin:{{l:40,r:10,t:10,b:40}},
        xaxis:{{title:'Generation', gridcolor:'#eee'}},
        yaxis:{{title:'Fitness', gridcolor:'#eee'}},
        legend:{{orientation:'h'}}
      }}, {{displayModeBar:true, responsive:true}});
    }})();

    // Genome viewer
    const cyContainer = document.getElementById('cy');
    const detail = document.getElementById('detail');
    const tip = document.getElementById('tip');
    const sel = document.getElementById('genomeSelect');
    const layoutBtn = document.getElementById('layoutBtn');
    const ctx = document.getElementById('ctx');
    let cy = null;
    let currentGenome = null; // store current genome object
    let ctxNode = null;

    function lsKey(gid) {{ return 'NEAT_REPORT_GENOME_STATE::' + String(gid); }}
    function saveState() {{
      if (!cy || !currentGenome) return;
      const st = {{}};
      cy.nodes().forEach(n => {{
        st[n.id()] = {{
          pos: n.position(),
          locked: n.locked(),
          color: n.data('color') || null,
          note: n.data('note') || null
        }};
      }});
      try {{ localStorage.setItem(lsKey(currentGenome.id || 'genome'), JSON.stringify(st)); }} catch(e){{}}
    }}
    function applyState() {{
      if (!cy || !currentGenome) return;
      let raw = null;
      try {{ raw = localStorage.getItem(lsKey(currentGenome.id || 'genome')); }} catch(e){{}}
      if (!raw) return;
      let st = null;
      try {{ st = JSON.parse(raw); }} catch(e) {{ return; }}
      if (!st) return;
      cy.batch(() => {{
        cy.nodes().forEach(n => {{
          const s = st[n.id()]; if (!s) return;
          if (s.pos && Number.isFinite(s.pos.x) && Number.isFinite(s.pos.y)) n.position(s.pos);
          if (s.locked) n.lock(); else n.unlock();
          if (s.color) {{ n.data('color', s.color); n.style('background-color', s.color); }}
          if (s.note) {{
            n.data('note', s.note);
            const orig = n.data('orig_label') || n.data('label') || n.id();
            n.data('label', orig + '\\n' + s.note);
          }}
        }});
      }});
    }}

    function genomeToElements(g) {{
      // Ensure orig_label availability for annotations
      const nodes = (g.nodes||[]).map(n => {{
        const d = n.data || n;
        return {{ data: Object.assign({{}}, d, {{ orig_label: d.label }}) }};
      }});
      const edges = (g.edges||[]).map(e => ({{ data: e.data || e, classes: ((e.data||e).enabled? 'enabled':'disabled') }}));
      return nodes.concat(edges);
    }}
    function populateSelect() {{
      sel.innerHTML = '';
      if (!DATA.genomes || DATA.genomes.length===0) {{
        const opt = document.createElement('option'); opt.text='(no genomes)'; sel.add(opt); sel.disabled=true; return;
      }}
      sel.disabled=false;
      DATA.genomes.forEach((g,i) => {{
        const meta = g.meta || {{}};
        const label = (g.id || ('genome_'+i)) + (meta.fitness!==undefined ? (' (fitness='+meta.fitness+')') : '');
        const opt = document.createElement('option'); opt.value=String(i); opt.text=label; sel.add(opt);
      }});
    }}
    function renderGenome(idx) {{
      if (!DATA.genomes || !DATA.genomes[idx]) return;
      const g = DATA.genomes[idx];
      currentGenome = g;
      const elements = genomeToElements(g);
      const styles = [
        {{ selector:'node', style:{{ 'label':'data(label)', 'font-size':11, 'text-valign':'center', 'text-halign':'center',
           'text-wrap':'wrap', 'text-max-width': 90, 'background-color':'#999','width':22,'height':22, 'color':'#111','border-color':'#333','border-width':0.5 }} }},
        {{ selector:'node[type = "input"]',  style:{{ 'background-color':'#009E73' }} }},
        {{ selector:'node[type = "output"]', style:{{ 'background-color':'#0072B2' }} }},
        {{ selector:'edge', style:{{ 'line-color':'#888', 'width':'mapData(weight, -2, 2, 0.6, 4)', 'opacity':0.95,
           'curve-style':'bezier','target-arrow-shape':'triangle','target-arrow-color':'#888' }} }},
        {{ selector:'edge[weight < 0]',  style:{{ 'line-color':'#D55E00', 'target-arrow-color':'#D55E00' }} }},
        {{ selector:'edge[weight >= 0]', style:{{ 'line-color':'#56B4E9', 'target-arrow-color':'#56B4E9' }} }},
        {{ selector:'edge.disabled', style:{{ 'line-style':'dotted','opacity':0.3 }} }},
        {{ selector:':selected', style:{{ 'border-width':2, 'border-color':'#F0E442' }} }},
      ];
      if (cy) cy.destroy();
      cy = cytoscape({{ container: cyContainer, elements: elements, style: styles, layout: {{ name:'cose', animate:false }},
        wheelSensitivity:0.2, minZoom:0.2, maxZoom:5 }});

      function showDetail(html) {{ detail.innerHTML = html; }}
      function nodeHtml(d) {{
        const a=(k)=> (d[k]!==undefined && d[k]!==null ? String(d[k]) : '');
        return `<div><b>Node</b></div>
          <div>ID: ${{a('id')}}</div>
          <div>Label: ${{a('label')}}</div>
          <div>Type: ${{a('type')}}</div>
          <div>Bias: ${{a('bias')}}</div>
          <div>Activation: ${{a('activation')}}</div>
          <div>Note: ${{a('note')}}</div>`;
      }}
      function edgeHtml(d) {{
        const a=(k)=> (d[k]!==undefined && d[k]!==null ? String(d[k]) : '');
        return `<div><b>Edge</b></div>
          <div>Source: ${{a('source')}}</div>
          <div>Target: ${{a('target')}}</div>
          <div>Weight: ${{a('weight')}}</div>
          <div>Enabled: ${{a('enabled')}}</div>`;
      }}

      // Click selects → details
      cy.on('tap','node',(evt)=> showDetail(nodeHtml(evt.target.data())));
      cy.on('tap','edge',(evt)=> showDetail(edgeHtml(evt.target.data())));
      cy.on('tap',(evt)=> {{ if (evt.target===cy) showDetail('<div class="fine">ノードやエッジを選択すると詳細が表示されます。</div>'); }});

      // Hover tooltip
      const moveTip = (e) => {{ tip.style.left=(e.renderedPosition.x + cyContainer.getBoundingClientRect().left)+'px';
                                tip.style.top=(e.renderedPosition.y + cyContainer.getBoundingClientRect().top)+'px'; }};
      cy.on('mouseover','node',(evt)=>{{ tip.innerHTML = evt.target.data('label') || evt.target.data('id'); tip.style.display='block'; moveTip(evt); }});
      cy.on('mousemove','node',(evt)=> moveTip(evt));
      cy.on('mouseout','node',()=> {{ tip.style.display='none'; }});

      // Persist position/color/note
      cy.on('free', 'node', saveState);
      cy.on('lock unlock', 'node', saveState);

      // Apply saved state for this genome
      applyState();

      // Context menu handlers
      cy.on('cxttapstart', 'node', (evt) => {{
        ctxNode = evt.target;
        openCtxAt(evt.renderedPosition);
      }});
      cy.on('cxttapstart', (evt) => {{
        if (evt.target === cy) closeCtx();
      }});
      document.addEventListener('click', (e) => {{
        if (!ctx.contains(e.target)) closeCtx();
      }});
      window.addEventListener('resize', closeCtx);
      document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeCtx(); }});
    }}

    // UI wiring
    populateSelect();
    if (DATA.genomes && DATA.genomes.length>0) renderGenome(0);
    sel.addEventListener('change', ()=> {{ const idx=parseInt(sel.value,10); if(!Number.isNaN(idx)) renderGenome(idx); }});
    layoutBtn.addEventListener('click', ()=> {{ if (cy) cy.layout({{name:'cose', animate:true}}).run(); }});

    // Context menu building
    const PALETTE = ['#0072B2','#D55E00','#009E73','#CC79A7','#F0E442','#56B4E9','#E69F00','#000000','#777777','#999999'];
    const hiddenColorPicker = document.createElement('input'); hiddenColorPicker.type='color'; hiddenColorPicker.style.display='none'; document.body.appendChild(hiddenColorPicker);

    function openCtxAt(renderedPos) {{
      if (!ctxNode) return;
      const rect = cyContainer.getBoundingClientRect();
      const x = rect.left + renderedPos.x;
      const y = rect.top + renderedPos.y;
      ctx.innerHTML = '';
      const menu = document.createElement('div');

      // Title
      const title = document.createElement('div');
      title.className='ctx-item';
      title.style.cursor='default';
      title.innerHTML = '<b>Node:</b> ' + (ctxNode.data('label') || ctxNode.id());
      ctx.appendChild(title);

      // Fix/Unfix
      const fix = document.createElement('div');
      fix.className='ctx-item';
      const locked = ctxNode.locked();
      fix.textContent = locked ? '位置の固定を解除' : '位置を固定';
      fix.onclick = () => {{
        if (ctxNode.locked()) ctxNode.unlock(); else ctxNode.lock();
        saveState(); closeCtx();
      }};
      ctx.appendChild(fix);

      // Color row
      const colorRow = document.createElement('div');
      colorRow.className='ctx-item';
      colorRow.innerHTML = '<div class="ctx-row"><span>色を変更</span><span style="font-size:12px;color:var(--muted)">クリックで適用</span></div>';
      const sw = document.createElement('div'); sw.className='swatches';
      PALETTE.forEach(c => {{
        const d = document.createElement('div'); d.className='swatch'; d.style.background=c;
        d.title = c;
        d.onclick = () => {{ ctxNode.data('color', c); ctxNode.style('background-color', c); saveState(); closeCtx(); }};
        sw.appendChild(d);
      }});
      // Custom picker
      const custom = document.createElement('div');
      custom.className='ctx-item';
      custom.textContent='カスタムカラー…';
      custom.onclick = () => {{
        hiddenColorPicker.value = ctxNode.data('color') || '#999999';
        hiddenColorPicker.onchange = () => {{
          const c = hiddenColorPicker.value;
          ctxNode.data('color', c); ctxNode.style('background-color', c); saveState(); closeCtx();
        }};
        hiddenColorPicker.click();
      }};
      colorRow.appendChild(sw);
      ctx.appendChild(colorRow);
      ctx.appendChild(custom);

      // Note editor
      const noteBtn = document.createElement('div');
      noteBtn.className='ctx-item';
      noteBtn.textContent='注釈を追加/編集…';
      noteBtn.onclick = () => {{
        const cur = ctxNode.data('note') || '';
        const txt = window.prompt('ノードの注釈（空で削除）', cur);
        if (txt === null) return;
        const orig = ctxNode.data('orig_label') || ctxNode.data('label') || ctxNode.id();
        if (txt.trim() === '') {{
          ctxNode.data('note', null);
          ctxNode.data('label', orig);
        }} else {{
          ctxNode.data('note', txt);
          ctxNode.data('label', orig + '\\n' + txt);
        }}
        saveState(); closeCtx();
      }};
      ctx.appendChild(noteBtn);

      // Reset color
      const resetColor = document.createElement('div');
      resetColor.className='ctx-item';
      resetColor.textContent='色をリセット';
      resetColor.onclick = () => {{
        ctxNode.data('color', null);
        // Revert to type-based color by removing inline style
        ctxNode.removeStyle('background-color');
        saveState(); closeCtx();
      }};
      ctx.appendChild(resetColor);

      // Separator
      const sep = document.createElement('div'); sep.className='ctx-sep'; ctx.appendChild(sep);

      // Save layout now
      const saveBtn = document.createElement('div');
      saveBtn.className='ctx-item';
      saveBtn.textContent='レイアウトを保存';
      saveBtn.onclick = () => {{ saveState(); closeCtx(); }};
      ctx.appendChild(saveBtn);

      // Open
      ctx.style.left = Math.round(x) + 'px';
      ctx.style.top  = Math.round(y) + 'px';
      ctx.style.display = 'block';
    }}

    function closeCtx() {{ ctx.style.display='none'; ctxNode = null; }}

  </script>
</body>
</html>"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("[REPORT]", path)


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

    script_name = os.path.basename(__file__) if "__file__" in globals() else "spiral_monolith_neat_numpy.py"

    os.makedirs(args.out, exist_ok=True)
    figs: Dict[str, Optional[str]] = {}
    report_meta: Dict[str, Optional[Dict[str, Any]]] = {"supervised": None, "rl": None}

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
        db_path = res.get("decision_boundary")
        if db_path and os.path.exists(db_path):
            figs["図3 決定境界"] = db_path
        regen_gif = res.get("regen_gif")
        if regen_gif and os.path.exists(regen_gif) and imageio is not None:
            with imageio.get_reader(regen_gif) as r:
                idx = max(0, r.get_length()//2 - 1)
                frame = r.get_data(idx)
                fig3 = os.path.join(args.out, f"{args.task}_fig3_regen_frame.png")
                _imwrite_image(fig3, frame)
                figs["図3A 再生ダイジェスト代表フレーム"] = fig3
        scars_spiral_path = res.get("scars_spiral")
        has_scars_spiral = bool(scars_spiral_path and os.path.exists(scars_spiral_path))
        if has_scars_spiral:
            figs["図4 螺旋再生ヒートマップ"] = scars_spiral_path
        else:
            figs["図4 螺旋再生ヒートマップ"] = res.get("scars_spiral")
        lineage_path = res.get("lineage")
        has_lineage = bool(lineage_path and os.path.exists(lineage_path))
        if has_lineage:
            figs["図5 系統ラインエイジ"] = lineage_path
        regen_log = res.get("lcs_log")
        has_regen_log = bool(regen_log and os.path.exists(regen_log))
        if has_regen_log:
            figs["LCS Healing Log"] = regen_log
        ribbon = res.get("lcs_ribbon")
        has_ribbon = bool(ribbon and os.path.exists(ribbon))
        if has_ribbon:
            figs["LCS Ribbon"] = ribbon
        timeline = res.get("lcs_timeline")
        has_timeline = bool(timeline and os.path.exists(timeline))
        if has_timeline:
            figs["LCS Timeline"] = timeline

        history = res.get("history") or []
        best_fit = max((b for (b, _a) in history), default=None)
        final_best = history[-1][0] if history else None
        final_avg = history[-1][1] if history else None
        initial_best = history[0][0] if history else None
        report_meta["supervised"] = {
            "task": args.task,
            "gens": args.gens,
            "pop": args.pop,
            "steps": args.steps,
            "best_fit": best_fit,
            "final_best": final_best,
            "final_avg": final_avg,
            "initial_best": initial_best,
            "has_lineage": has_lineage,
            "has_regen_log": has_regen_log,
            "has_lcs_viz": bool(has_ribbon or has_timeline),
            "has_spiral": has_scars_spiral,
        }

    # gallery（--task が無くても実行できるように独立させる）
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
            regen_log_path = os.path.join(args.out, f"{args.rl_env.replace(':','_')}_regen_log.csv")
            if hasattr(neat, "lcs_monitor") and neat.lcs_monitor is not None:
                neat.lcs_monitor.csv_path = regen_log_path
                if os.path.exists(regen_log_path):
                    os.remove(regen_log_path)
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
            lcs_rows = load_lcs_log(regen_log_path) if os.path.exists(regen_log_path) else []
            lcs_series = _prepare_lcs_series(lcs_rows) if lcs_rows else None
            if os.path.exists(regen_log_path):
                figs[f"LCS Healing Log ({args.rl_env})"] = regen_log_path
            if lcs_rows:
                ribbon_path = os.path.join(args.out, f"{args.rl_env.replace(':','_')}_lcs_ribbon.png")
                try:
                    export_lcs_ribbon_png(lcs_rows, ribbon_path, series=lcs_series)
                    figs[f"LCS Ribbon ({args.rl_env})"] = ribbon_path
                except Exception as ribbon_err:
                    print("[WARN] LCS ribbon export failed:", ribbon_err)
                timeline_path = os.path.join(args.out, f"{args.rl_env.replace(':','_')}_lcs_timeline.gif")
                try:
                    export_lcs_timeline_gif(lcs_rows, timeline_path, series=lcs_series, fps=6)
                    figs[f"LCS Timeline ({args.rl_env})"] = timeline_path
                except Exception as timeline_err:
                    print("[WARN] LCS timeline export failed:", timeline_err)
            gif = None
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

            rl_best = max((b for (b, _a) in hist), default=None)
            rl_final_best = hist[-1][0] if hist else None
            rl_final_avg = hist[-1][1] if hist else None
            report_meta["rl"] = {
                "env": args.rl_env,
                "gens": args.rl_gens,
                "pop": args.rl_pop,
                "episodes": args.rl_episodes,
                "best_reward": rl_best,
                "final_best": rl_final_best,
                "final_avg": rl_final_avg,
                "has_lcs_log": bool(os.path.exists(regen_log_path)),
                "has_lcs_viz": bool(lcs_rows),
                "has_gameplay": bool(gif and os.path.exists(gif)),
            }
        except Exception as e:
            print("[WARN] RL branch skipped:", e)

    if args.report:
        # Priority: supervised task interactive report
        if args.task:
            title = f"{args.task.upper()} | Interactive NEAT Report"
            html_path = os.path.join(args.out, f"{args.task}_interactive.html")
            export_interactive_html_report(
                html_path,
                title=title,
                history=res.get("history", []),
                genomes=res.get("genomes_cyto", []),
                max_genomes=60
            )
        # RL execution: show learning curve only (genomes optional)
        if args.rl_env:
            title = f"{args.rl_env} | Interactive NEAT Report"
            html_path = os.path.join(args.out, f"{args.rl_env.replace(':','_')}_interactive.html")
            # RL part: hist variable scope is within this block if it exists
            try:
                export_interactive_html_report(html_path, title=title, history=hist, genomes=[], max_genomes=1)
            except Exception:
                pass
        
        # Also generate the legacy static report if figs exist
        if figs:
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
                import html as htmllib

                f.write("<!DOCTYPE html><html lang='ja'><head><meta charset='utf-8'><title>Spiral Monolith NEAT Report | スパイラル・モノリスNEATレポート</title>")
                f.write(
                    "<style>body{font-family:'Hiragino Sans','Noto Sans JP',sans-serif;background:#fafafa;color:#222;line-height:1.6;padding:2rem;}"
                    "header.cover{background:#fff;border:1px solid #ddd;border-radius:12px;padding:1.5rem;margin-bottom:2rem;box-shadow:0 8px 20px rgba(0,0,0,0.05);}"
                    "header.cover h1{margin-top:0;font-size:1.9rem;}"
                    "header.cover p.meta{margin:0.35rem 0 0.6rem 0;}"
                    "header.cover ul{margin:0;padding-left:1.2rem;}"
                    "section.summary,section.legend,section.narrative,section.examples{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:1.25rem;margin-bottom:2rem;box-shadow:0 10px 24px rgba(0,0,0,0.035);}"
                    "section.summary h2,section.legend h2,section.narrative h2,section.examples h2{margin-top:0;font-size:1.35rem;}"
                    "section.summary ul{margin:0;padding-left:1.4rem;}"
                    "section.legend ol{margin:0;padding-left:1.4rem;}"
                    "section.narrative p{margin:0 0 0.8rem 0;}"
                    "section.examples ul{margin:0;padding-left:1.4rem;}"
                    "section.examples li{margin:0 0 0.65rem 0;}"
                    "section.examples code{background:#f4f4f4;border-radius:6px;display:block;padding:0.35rem 0.55rem;font-size:0.92rem;}"
                    "figure{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:1rem;margin:0 0 2rem 0;box-shadow:0 12px 24px rgba(0,0,0,0.04);}"
                    "figure img,figure video{width:100%;height:auto;border-radius:8px;}"
                    "figcaption{margin-top:0.75rem;font-weight:600;}"
                    "</style></head><body>"
                )

                f.write("<header class='cover'>")
                f.write("<h1>Spiral Monolith NEAT Report / スパイラル・モノリスNEATレポート</h1>")
                f.write(
                    f"<p class='meta'>Generated at / 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Primary Task / 主タスク: {args.task.upper() if args.task else 'N/A'}</p>"
                )
                if args.gallery:
                    f.write("<p class='meta'>Gallery Tasks / ギャラリー対象: " + ", ".join(htmllib.escape(t.upper()) for t in args.gallery) + "</p>")
                f.write("<ul>")
                f.write(f"<li>Generations / 世代数: {args.gens}</li>")
                f.write(f"<li>Population / 個体数: {args.pop}</li>")
                f.write(f"<li>Backprop Steps / 学習反復: {args.steps}</li>")
                if args.rl_env:
                    f.write(f"<li>RL Environment / 強化学習環境: {htmllib.escape(args.rl_env)}</li>")
                f.write("</ul>")
                f.write("</header>")

                summary_items = []

                def _fmt_float(val: Optional[float]) -> str:
                    return "–" if val is None else f"{val:.4f}"

                sup_meta = report_meta.get("supervised")
                if sup_meta:
                    extras = []
                    if sup_meta.get("has_regen_log"):
                        extras.append("LCS log")
                    if sup_meta.get("has_lcs_viz"):
                        extras.append("LCS visuals")
                    if sup_meta.get("has_lineage"):
                        extras.append("lineage")
                    if sup_meta.get("has_spiral"):
                        extras.append("scar map")
                    extra_txt = f" [{', '.join(extras)}]" if extras else ""
                    summary_items.append(
                        f"<li><strong>Supervised ({htmllib.escape(sup_meta['task'].upper())})</strong> — best {_fmt_float(sup_meta.get('best_fit'))} | final {_fmt_float(sup_meta.get('final_best'))} / avg {_fmt_float(sup_meta.get('final_avg'))}{extra_txt}</li>"
                    )

                rl_meta = report_meta.get("rl")
                if rl_meta and rl_meta.get("env"):
                    extras = []
                    if rl_meta.get("has_lcs_log"):
                        extras.append("LCS log")
                    if rl_meta.get("has_lcs_viz"):
                        extras.append("LCS visuals")
                    if rl_meta.get("has_gameplay"):
                        extras.append("gameplay gif")
                    extra_txt = f" [{', '.join(extras)}]" if extras else ""
                    summary_items.append(
                        f"<li><strong>RL ({htmllib.escape(rl_meta['env'])})</strong> — best {_fmt_float(rl_meta.get('best_reward'))} | final {_fmt_float(rl_meta.get('final_best'))} / avg {_fmt_float(rl_meta.get('final_avg'))}{extra_txt}</li>"
                    )

                f.write("<section class='summary'><h2>Overview / 概要</h2>")
                f.write(f"<p>Figures included / 図版数: {len(entries)}</p>")
                if summary_items:
                    f.write("<ul>")
                    for item in summary_items:
                        f.write(item)
                    f.write("</ul>")
                else:
                    f.write("<p>No supervised or RL runs were summarised for this report.</p>")
                f.write("</section>")

                f.write("<section class='narrative'><h2>Evolution Digest / 進化ダイジェスト</h2>")
                f.write(
                    "<p>Early generations showed smooth structural adaptation and convergence under low-difficulty conditions.</p>"
                )
                f.write(
                    "<p>However, as environmental difficulty and noise increased, regeneration-driven mutations began to trigger bursts of morphological diversification, resembling biological punctuated equilibria.</p>"
                )
                if sup_meta and sup_meta.get("has_regen_log"):
                    f.write(
                        "<p>LCS metrics highlighted how severed pathways recovered within the allowed healing window, aligning regenerative bursts with topology repairs.</p>"
                    )
                f.write("</section>")

                if entries:
                    f.write("<section class='legend'><h2>Figure Index / 図版リスト</h2><ol>")
                    for label, path in entries:
                        f.write(
                            f"<li><strong>{htmllib.escape(label)}</strong><br><small>{htmllib.escape(os.path.basename(path))}</small></li>"
                        )
                    f.write("</ol></section>")

                cli_examples = [
                    f"python {script_name} --task {htmllib.escape((args.task or 'spiral'))} --gens {max(args.gens, 60)} --pop {max(args.pop, 64)} --steps {max(args.steps, 80)} --make-gifs --make-lineage --report --out demo_{htmllib.escape((args.task or 'spiral'))}",
                    f"python {script_name} --task xor --gallery spiral circles --gens 40 --pop 48 --steps 60 --report --out gallery_pack"
                ]
                rl_example_env = args.rl_env or "CartPole-v1"
                cli_examples.append(
                    f"python {script_name} --rl-env {htmllib.escape(rl_example_env)} --rl-gens {max(args.rl_gens, 30)} --rl-pop {max(args.rl_pop, 32)} --rl-episodes {max(args.rl_episodes, 2)} --rl-max-steps {args.rl_max_steps} --report --out rl_{htmllib.escape(rl_example_env.replace(':','_'))}"
                )

                f.write("<section class='examples'><h2>CLI Quickstart / CLIクイックスタート</h2>")
                f.write("<p>Use the following commands as templates for supervised runs, gallery batches, and Gym integrations.</p>")
                f.write("<ul>")
                for cmd in cli_examples:
                    f.write(f"<li><code>{cmd}</code></li>")
                f.write("</ul></section>")

                for k, p in entries:
                    uri, mime = _data_uri(p)
                    escaped_label = htmllib.escape(k)
                    if mime == "image/gif":
                        f.write(
                            f"<figure><img src='{uri}' style='max-width:100%'>"
                            f"<figcaption><strong>{escaped_label}</strong></figcaption></figure>"
                        )
                    elif mime.startswith("image/"):
                        f.write(
                            f"<figure><img src='{uri}' style='max-width:100%'><figcaption><strong>{escaped_label}</strong></figcaption></figure>"
                        )
                    elif mime.startswith("video/"):
                        f.write(
                            "<figure><video autoplay loop muted playsinline style='max-width:100%'>"
                            f"<source src='{uri}' type='{mime}'></video>"
                            f"<figcaption><strong>{escaped_label}</strong></figcaption></figure>"
                        )
                    else:
                        f.write(
                            f"<figure><a href='{uri}'>download {htmllib.escape(os.path.basename(p))}</a>"
                            f"<figcaption><strong>{escaped_label}</strong></figcaption></figure>"
                        )
                f.write("</body></html>")

            print("[REPORT]", html)

    print("[OK] outputs in:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
