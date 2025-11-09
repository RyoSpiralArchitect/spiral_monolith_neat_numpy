# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ spiral_monolith_neat_numpy.py                                         ┃
# ┃ Monolithic NEAT × NumPy playground with the Lazy Council ecosystem.   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# EN: Single-file research lab for cooperative neuroevolution, backprop refinement,
#     and artefact-rich storytelling. Designed around the six-seat Lazy Council
#     steering model with mandatory governance enabled by default.
# JP: 協調的ニューロ進化とバックプロップ微調整を 1 ファイルで体験できる研究ラボ。
#     6 名の Lazy Council 合議制とデフォルト有効の mandatory ガバナンス設計を基軸としています。
#
# Why a monolith? / なぜ単一ファイルなのか？
# • Inspectable: read every subsystem—evolution loop, NumPy optimiser, exporters—without jumping packages.
#   参照性: 進化ループや NumPy 最適化、エクスポート機構をパッケージ横断なしで把握可能。
# • Portable: drop into a notebook, server, or CI job; no optional deps beyond Matplotlib and NumPy.
#   可搬性: ノートブック／サーバー／CI に即投入でき、NumPy と Matplotlib 以外の必須依存がありません。
# • Tunable: tweak council weights, lazy pressure, or mandatory policies inline and re-run instantly.
#   調整容易: 合議制ウェイトや怠惰個体圧、mandatory ポリシーをその場で書き換えすぐ検証できます。
#
# Quickstart / クイックスタート
#   python spiral_monolith_neat_numpy.py --help
# Discover CLI presets, exporters, and governance toggles—`--no-mandatory` flips the council back to
# soft advisory mode for exploratory runs.
# CLI でプリセットやエクスポート、ガバナンス設定を確認できます。`--no-mandatory` で mandatory を無効化し、
# 柔軟な実験モードに切り替えられます。
#
# Pillars / コア機能
# • Monodromy-aware curricula: environment difficulty follows topology shifts for path-dependent regimes.
#   モノドロミー対応カリキュラム: トポロジー変化に応じた環境難度制御で軌跡依存の学習を実現。
# • Lazy Council orchestration: top performers + stochastic delegates cast equal votes to steer dynamics,
#   while lazy-lineage amplification prevents overfitting to any diversity axis.
#   Lazy Council オーケストレーション: トップ個体とランダム代表が同票で舵取りし、多様性軸の過適応を怠惰系統強化で抑制。
# • Hybrid NEAT × backprop loop: champion genomes receive NumPy-based fine-tuning without losing structural novelty.
#   NEAT とバックプロップのハイブリッド: 勝者ゲノムに NumPy 微調整を適用し構造革新を維持。
# • Artefact exports: lineage graphs, morph GIFs, regeneration timelines, LCS ribbons—curated via a headless-ready
#   Matplotlib pipeline.
#   成果物エクスポート: 系統グラフや形態 GIF、再生タイムライン、LCS リボンをヘッドレス環境対応の Matplotlib で出力。
#
# Featured recipes / 代表的レシピ
# • Spiral benchmark explorer / スパイラル課題徹底解析
#     python spiral_monolith_neat_numpy.py \
#         --task spiral --gens 60 --pop 96 --steps 60 \
#         --make-gifs --make-lineage --report --out out/spiral_bold
# • Monodromy spotlight / モノドロミーモード詳解
#     python spiral_monolith_neat_numpy.py \
#         --task spiral --gens 60 --pop 96 --steps 60 \
#         --monodromy --make-gifs --make-lineage --report --out out/spiral_monodromy
# • Gym baseline / Gym 連携ベースライン
#     python - <<'PY'
#     from spiral_monolith_neat_numpy import run_gym_neat_experiment
#     run_gym_neat_experiment(
#         "CartPole-v1", gens=30, pop=64, episodes=3, max_steps=500,
#         stochastic=True, temp=0.8, out_prefix="out/cartpole"
#     )
#     PY
#
# Outputs / 成果物
# All PNG, GIF, and optional HTML artefacts land under `--out` (or `out_prefix`) using a hardened `_savefig`
# pipeline so fonts, transparency, and permissions stay consistent across platforms.
# PNG / GIF / HTML 成果物は強化済み `_savefig` パイプラインを介して `--out`（または `out_prefix`）以下に整理保存され、
# フォントや透過設定、パーミッションが環境間で揃います。
#
# Author: Ryo ∴ SpiralArcitect & AIs from SpiralReality

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable, Optional, Set, Iterable, Any, Sequence
from collections import deque, defaultdict, OrderedDict, Counter
import math, argparse, os, mimetypes, csv
import sys
import time
import subprocess
import matplotlib
import warnings
import pickle as _pickle
import json as _json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import font_manager as _font_manager
from matplotlib import gridspec as _gridspec
from matplotlib.lines import Line2D
import csv
import json
import math
import os
from typing import Dict, Optional, Tuple
import importlib.util
from datetime import datetime, timezone

_BUILD_INFO_CACHE: Optional[Dict[str, Any]] = None
_SPINOR_BOUND_SEED: Optional[int] = None
_COMPILE_TICK: int = 0
_STRUCTURE_CACHE_LIMIT: int = 512
_STRUCT_COMPILED_CACHE: 'OrderedDict[Any, Dict[str, Any]]' = OrderedDict()


def _structure_cache_key(g: 'Genome', order: Sequence[int]) -> Optional[Tuple[Any, ...]]:
    try:
        signature = g.structural_signature()
    except Exception:
        return None
    try:
        return (signature, tuple(order))
    except Exception:
        return None


def _structure_cache_get(key: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    if key is None:
        return None
    entry = _STRUCT_COMPILED_CACHE.get(key)
    if entry is None:
        return None
    entry['_tick'] = _COMPILE_TICK
    _STRUCT_COMPILED_CACHE.move_to_end(key)
    return entry.get('payload')


def _structure_cache_store(key: Optional[Tuple[Any, ...]], payload: Dict[str, Any]) -> None:
    if key is None:
        return
    _STRUCT_COMPILED_CACHE[key] = {'payload': dict(payload), '_tick': _COMPILE_TICK}
    _STRUCT_COMPILED_CACHE.move_to_end(key)
    _structure_cache_trim()


def _structure_cache_trim(max_entries: Optional[int]=None, min_tick: Optional[int]=None) -> None:
    if min_tick is not None:
        stale_keys = [
            key for key, entry in list(_STRUCT_COMPILED_CACHE.items())
            if entry.get('_tick', 0) < min_tick
        ]
        for key in stale_keys:
            _STRUCT_COMPILED_CACHE.pop(key, None)
    limit = max_entries if max_entries is not None else _STRUCTURE_CACHE_LIMIT
    if limit <= 0:
        limit = 1
    while len(_STRUCT_COMPILED_CACHE) > limit:
        _STRUCT_COMPILED_CACHE.popitem(last=False)


def clear_compiled_structure_cache() -> None:
    _STRUCT_COMPILED_CACHE.clear()


_NOISE_STYLE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'white': {'label': 'White', 'symbol': 'W', 'color': '#f6f7fb', 'index': 0, 'bias': 0.0},
    'alpha': {'label': 'Alpha', 'symbol': 'α', 'color': '#8ecae6', 'index': 1, 'bias': -0.06},
    'beta': {'label': 'Beta', 'symbol': 'β', 'color': '#ffb703', 'index': 2, 'bias': 0.04},
    'black': {'label': 'Black', 'symbol': '♭', 'color': '#1f1f1f', 'index': 3, 'bias': -0.12},
}
_NOISE_STYLE_FALLBACK: Dict[str, Any] = {'label': 'Unknown', 'symbol': '?', 'color': '#9e9e9e', 'index': -1, 'bias': 0.0}


def _resolve_noise_style(kind: Optional[str], overrides: Optional[Dict[str, Dict[str, Any]]]=None) -> Dict[str, Any]:
    key = (kind or '').strip().lower()
    base = dict(_NOISE_STYLE_DEFAULTS.get(key, _NOISE_STYLE_FALLBACK))
    if overrides:
        override = overrides.get(key) or overrides.get(kind or '')
        if override:
            base.update({k: v for k, v in override.items() if v is not None})
    label_source = kind or base.get('label') or 'Unknown'
    if not base.get('label'):
        base['label'] = str(label_source).title()
    if not base.get('symbol'):
        base['symbol'] = (label_source[:1].upper() if label_source else '?')
    base['index'] = int(base.get('index', _NOISE_STYLE_FALLBACK['index']))
    base['color'] = str(base.get('color', _NOISE_STYLE_FALLBACK['color']))
    base['bias'] = float(base.get('bias', _NOISE_STYLE_FALLBACK['bias']))
    base['kind'] = kind or ''
    return base


def _resolve_build_info() -> Dict[str, Any]:
    """Return cached build metadata including git short hash and UTC timestamp."""

    global _BUILD_INFO_CACHE
    if _BUILD_INFO_CACHE is not None:
        return _BUILD_INFO_CACHE
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    stamp_ts = now_utc.strftime('%Y%m%dT%H%M%SZ')
    short_hash = 'nogit'
    repo_root = None
    try:
        here = os.path.abspath(__file__)
        repo_root = os.path.dirname(here)
    except Exception:
        repo_root = os.getcwd()
    probe = repo_root
    while probe and not os.path.isdir(os.path.join(probe, '.git')):
        parent = os.path.dirname(probe)
        if parent == probe:
            probe = None
            break
        probe = parent
    if probe:
        try:
            raw = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=probe,
                stderr=subprocess.DEVNULL,
            )
            short_hash = raw.decode('ascii', errors='ignore').strip() or 'nogit'
        except Exception:
            short_hash = 'nogit'
    build_id = f'{short_hash}-{stamp_ts}'
    _BUILD_INFO_CACHE = {
        'hash': short_hash,
        'timestamp': stamp_ts,
        'datetime': now_utc,
        'id': build_id,
    }
    return _BUILD_INFO_CACHE


def _build_stamp_text(prefix: str='Spiral Monolith NEAT') -> str:
    info = _resolve_build_info()
    return f"{prefix} build {info['id']}"


def _build_stamp_short() -> str:
    info = _resolve_build_info()
    return info['id']

def _is_picklable(obj) -> bool:
    """Process 並列に切り替える前に picklable かを事前検査（非picklableなら thread に自動フォールバック）"""
    try:
        _pickle.dumps(obj)
        return True
    except Exception:
        return False

def shm_register_dataset(label: str, arr: 'np.ndarray', readonly: bool=True) -> dict:
    """Create shared memory for arr (parent), return metadata dict."""
    if _shm is None:
        raise RuntimeError('shared_memory is unavailable on this Python.')
    arr = np.asarray(arr)
    size = int(arr.nbytes)
    name = f'sm_{label}_{np.random.randint(1, 1 << 30):08x}'
    shm = _shm.SharedMemory(create=True, size=size, name=name)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    _SHM_LOCAL[label] = (shm, arr.shape, str(arr.dtype), bool(readonly))
    meta = {'name': name, 'shape': tuple(arr.shape), 'dtype': str(arr.dtype), 'readonly': bool(readonly)}
    _SHM_META[label] = meta
    return meta

def shm_set_worker_meta(meta: dict | None):
    """Install metadata in worker; views are attached lazily on demand."""
    global _SHM_META, _SHM_CACHE, _SHM_HANDLES
    _SHM_META = dict(meta or {})
    _SHM_CACHE = {}
    _SHM_HANDLES = {}

def get_shared_dataset(label: str) -> 'np.ndarray':
    """Worker-side: return cached numpy view to shared dataset by label."""
    if label in _SHM_CACHE:
        return _SHM_CACHE[label]
    meta = _SHM_META.get(label)
    if not meta:
        raise KeyError(f"Shared dataset '{label}' not found.")
    if _shm is None:
        raise RuntimeError('shared_memory is unavailable in worker.')
    shm = _shm.SharedMemory(name=meta['name'])
    arr = np.ndarray(tuple(meta['shape']), dtype=np.dtype(meta['dtype']), buffer=shm.buf)
    if bool(meta.get('readonly', True)):
        try:
            arr.setflags(write=False)
        except Exception:
            pass
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

def _proc_init_worker(meta: dict | None=None):
    """ProcessPool initializer: cap BLAS threads and install SHM metadata."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    if meta:
        try:
            shm_set_worker_meta(meta)
        except Exception:
            pass
    try:
        import atexit
        atexit.register(shm_worker_release_all)
    except Exception:
        pass

def _ensure_matplotlib_agg(force: bool=False):
    """Select Agg backend even if pyplot was already imported elsewhere."""
    try:
        matplotlib.use('Agg', force=force)
    except TypeError:
        matplotlib.use('Agg')
    return matplotlib


def _install_cjk_font() -> Optional['matplotlib.font_manager.FontProperties']:
    """Try to locate a system font that can render Japanese glyphs."""
    candidates = [
        'Hiragino Sans',
        'Hiragino Kaku Gothic ProN',
        'Yu Gothic',
        'YuGothic',
        'Meiryo',
        'MS Gothic',
        'MS Mincho',
        'Noto Sans CJK JP',
        'Source Han Sans JP',
        'Source Han Sans HW',
        'IPAPGothic',
        'IPAexGothic',
        'TakaoGothic',
        'AppleGothic',
    ]
    for family in candidates:
        try:
            prop = _font_manager.FontProperties(family=family)
            _font_manager.findfont(prop, fontext='ttf', fallback_to_default=False)
        except Exception:
            continue
        matplotlib.rcParams['font.family'] = [family]
        plt.rcParams['font.family'] = [family]
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.unicode_minus'] = False
        return prop
    try:
        for path in _font_manager.findSystemFonts(fontext='ttf'):
            try:
                prop = _font_manager.FontProperties(fname=path)
                name = prop.get_name()
            except Exception:
                continue
            if any(tag in name for tag in ('Gothic', 'Mincho', 'Hiragino', 'Yu', 'Meiryo', 'Noto', 'Source Han', 'IPA')):
                try:
                    _font_manager.fontManager.addfont(path)
                except Exception:
                    pass
                matplotlib.rcParams['font.family'] = [name]
                plt.rcParams['font.family'] = [name]
                matplotlib.rcParams['axes.unicode_minus'] = False
                plt.rcParams['axes.unicode_minus'] = False
                return prop
    except Exception:
        pass
    warnings.filterwarnings(
        'ignore',
        message='Glyph .* missing from font',
        category=UserWarning,
        module='matplotlib'
    )
    return None


_CJK_FONT_PROP = _install_cjk_font()

def _mimsave(path, frames, fps=12):
    """Robust GIF writer that tolerates missing imageio by falling back to Pillow."""
    frame_seq = list(frames)
    if not frame_seq:
        return
    if imageio is not None:
        imageio.mimsave(path, frame_seq, duration=1.0 / max(1, int(fps)))
        return
    if Image is None:
        raise RuntimeError('imageio is unavailable and Pillow is not installed; cannot write GIF')
    imgs = [Image.fromarray(np.asarray(fr)) for fr in frame_seq]
    dur = int(1000 / max(1, int(fps)))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0)

def _imread_image(src):
    """imageio.imread fallback that also supports BytesIO via Pillow."""
    if imageio is not None:
        return imageio.imread(src)
    if Image is None:
        raise RuntimeError('imageio is unavailable and Pillow is not installed; cannot read image')
    img = Image.open(src)
    try:
        return np.asarray(img.convert('RGB'))
    finally:
        img.close()

def _imwrite_image(path, array):
    """Write an image array using imageio or Pillow."""
    stamped = _stamp_image_array(np.asarray(array))
    if imageio is not None:
        imageio.imwrite(path, stamped)
        return
    if Image is None:
        raise RuntimeError('imageio is unavailable and Pillow is not installed; cannot write image')
    Image.fromarray(stamped).save(path)


def _stamp_figure(fig: 'plt.Figure') -> 'plt.Figure':
    """Watermark a matplotlib figure with the current build stamp."""

    if fig is None:
        return fig
    if getattr(fig, '_build_stamp_applied', False):
        return fig
    text = _build_stamp_text()
    try:
        fig.text(
            0.995,
            0.01,
            text,
            ha='right',
            va='bottom',
            fontsize=6,
            color='#4a4a4a',
            alpha=0.78,
        )
    except Exception:
        pass
    else:
        setattr(fig, '_build_stamp_applied', True)
    return fig


def _savefig(fig: 'plt.Figure', path: str, **kwargs) -> None:
    """Save a matplotlib figure with a build watermark."""

    _stamp_figure(fig)
    fig.savefig(path, **kwargs)


def _stamp_image_array(arr: np.ndarray) -> np.ndarray:
    """Overlay the build stamp onto a numpy image array when Pillow is available."""

    try:
        from PIL import Image as _PILImage, ImageDraw
    except Exception:
        return np.asarray(arr)
    if arr.ndim < 2:
        return np.asarray(arr)
    try:
        img = _PILImage.fromarray(np.asarray(arr))
    except Exception:
        return np.asarray(arr)
    draw = ImageDraw.Draw(img)
    text = _build_stamp_text()
    w, h = img.size
    margin = max(3, int(min(w, h) * 0.01))
    anchor_point = (w - margin, h - margin)
    try:
        draw.text(anchor_point, text, anchor='rd', fill=(68, 68, 68))
    except Exception:
        try:
            x, y = anchor_point
            draw.text((x - margin, y - margin), text, fill=(68, 68, 68))
        except Exception:
            pass
    return np.asarray(img)

@dataclass
class NodeGene:
    id: int
    type: str
    activation: str = 'tanh'
    backprop_sensitivity: float = 1.0
    sensitivity_jitter: float = 0.0
    sensitivity_momentum: float = 0.0
    sensitivity_variance: float = 0.0
    altruism: float = 0.5
    altruism_memory: float = 0.0
    altruism_span: float = 0.0


def _clone_node(
    template: NodeGene,
    new_id: Optional[int]=None,
    new_type: Optional[str]=None,
    activation: Optional[str]=None,
) -> NodeGene:
    return NodeGene(
        template.id if new_id is None else int(new_id),
        new_type or template.type,
        activation or template.activation,
        getattr(template, 'backprop_sensitivity', 1.0),
        getattr(template, 'sensitivity_jitter', 0.0),
        getattr(template, 'sensitivity_momentum', 0.0),
        getattr(template, 'sensitivity_variance', 0.0),
        float(np.clip(getattr(template, 'altruism', 0.5), 0.0, 1.0)),
        float(np.clip(getattr(template, 'altruism_memory', 0.0), -1.5, 1.5)),
        float(np.clip(getattr(template, 'altruism_span', 0.0), 0.0, 4.0)),
    )


def _random_hidden_node(node_id: int, rng: np.random.Generator, activation: str='tanh') -> NodeGene:
    return NodeGene(
        node_id,
        'hidden',
        activation,
        float(rng.uniform(0.9, 1.1)),
        float(np.clip(rng.normal(0.0, 0.05), -0.18, 0.18)),
        0.0,
        0.0,
        float(np.clip(rng.normal(0.5, 0.08), 0.0, 1.0)),
        float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0)),
        float(np.clip(rng.gamma(2.0, 0.2), 0.0, 3.0)),
    )


def _node_trait_array(
    genome: 'Genome',
    order: Sequence[int],
    attr: str,
    default: float,
    *,
    low: Optional[float]=None,
    high: Optional[float]=None,
) -> np.ndarray:
    vals: List[float] = []
    for nid in order:
        node = genome.nodes[nid]
        val = getattr(node, attr, default)
        if low is not None and val < low:
            val = low
        if high is not None and val > high:
            val = high
        vals.append(float(val))
    return np.asarray(vals, dtype=np.float64)

@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int

class InnovationTracker:

    def __init__(self, next_node_id: int, next_conn_innov: int=0):
        self.next_node_id = next_node_id
        self.next_conn_innov = next_conn_innov
        self.conn_innovations: Dict[Tuple[int, int], int] = {}
        self.node_innovations: Dict[Tuple[int, int], int] = {}

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

class _GenomeNodeDict(dict):
    """Dictionary that notifies the owning genome when its structure changes."""

    def __init__(self, initial: Dict[int, 'NodeGene'], owner: 'Genome'):
        self._owner = owner
        self._suspend = True
        super().__init__(initial)
        self._suspend = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)

    def __delitem__(self, key):
        super().__delitem__(key)
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)

    def clear(self):
        super().clear()
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)

    def pop(self, key, default=None):
        value = super().pop(key, default)
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)
        return value

    def popitem(self):
        item = super().popitem()
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)
        return item

    def setdefault(self, key, default=None):
        if key in self:
            return super().setdefault(key, default)
        self._suspend = True
        try:
            value = super().setdefault(key, default)
        finally:
            self._suspend = False
        self._owner.invalidate_caches(structure=True)
        return value

    def update(self, *args, **kwargs):
        self._suspend = True
        try:
            super().update(*args, **kwargs)
        finally:
            self._suspend = False
        self._owner.invalidate_caches(structure=True)


class _GenomeConnDict(dict):
    """Dictionary that invalidates caches when connections mutate."""

    def __init__(self, initial: Dict[int, 'ConnectionGene'], owner: 'Genome'):
        self._owner = owner
        self._suspend = True
        super().__init__(initial)
        self._suspend = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)

    def __delitem__(self, key):
        super().__delitem__(key)
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)

    def clear(self):
        super().clear()
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)

    def pop(self, key, default=None):
        value = super().pop(key, default)
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)
        return value

    def popitem(self):
        item = super().popitem()
        if not self._suspend:
            self._owner.invalidate_caches(structure=True)
        return item

    def setdefault(self, key, default=None):
        if key in self:
            return super().setdefault(key, default)
        self._suspend = True
        try:
            value = super().setdefault(key, default)
        finally:
            self._suspend = False
        self._owner.invalidate_caches(structure=True)
        return value

    def update(self, *args, **kwargs):
        self._suspend = True
        try:
            super().update(*args, **kwargs)
        finally:
            self._suspend = False
        self._owner.invalidate_caches(structure=True)


class Genome:

    def __init__(
        self,
        nodes: Dict[int, 'NodeGene'],
        connections: Dict[int, 'ConnectionGene'],
        sex: Optional[str]=None,
        regen: bool=False,
        regen_mode: Optional[str]=None,
        embryo_bias: Optional[str]=None,
        gid: Optional[int]=None,
        birth_gen: int=0,
        hybrid_scale: float=1.0,
        parents: Optional[Tuple[Optional[int], Optional[int]]]=None,
        mutation_will: Optional[float]=None,
        cooperative: bool=True,
        family_id: Optional[int]=None,
    ):
        self.origin_mode = 'initial'
        self._structure_rev = 0
        self._weights_rev = 0
        self._topo_cache: Optional[List[int]] = None
        self._topo_cache_rev = -1
        self._sorted_innovs_cache: Optional[List[int]] = None
        self._sorted_innovs_rev = -1
        self._max_innov_cache: int = -1
        self._conn_index: Optional[Set[Tuple[int, int]]] = None
        self._conn_index_rev = -1
        self._struct_sig_cache: Optional[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[int, int, int], ...]]] = None
        self._struct_sig_rev = -1
        self._complexity_cache: Optional[Tuple[int, int, float, float, float]] = None
        self._complexity_rev = -1
        self._compiled_cache: Optional[Dict[str, Any]] = None
        self._compiled_cache_rev: Tuple[int, int] = (-1, -1)
        self._compiled_cache_tick: int = -1
        self._compat_token: Optional[Tuple[int, int, int]] = None
        self._compat_token_rev: Tuple[int, int] = (-1, -1)
        self.nodes = _GenomeNodeDict(nodes, self)
        self.connections = _GenomeConnDict(connections, self)
        self.sex = sex or ('female' if np.random.random() < 0.5 else 'male')
        self.regen = bool(regen)
        self.regen_mode = regen_mode or np.random.choice(['head', 'tail', 'split'])
        self.embryo_bias = embryo_bias or np.random.choice(['neutral', 'inputward', 'outputward'], p=[0.5, 0.25, 0.25])
        self.id = gid if gid is not None else int(np.random.randint(1, 1000000000.0))
        self.birth_gen = int(birth_gen)
        self.hybrid_scale = float(hybrid_scale)
        self.parents = parents if parents is not None else (None, None)
        self.family_id = int(family_id) if family_id is not None else int(self.id)
        self.origin_mode = getattr(self, 'origin_mode', 'initial')
        self.max_hidden_nodes: Optional[int] = None
        self.max_edges: Optional[int] = None
        self.mutation_will = float(mutation_will) if mutation_will is not None else float(np.random.uniform(0.0, 1.0))
        self.cooperative = bool(cooperative)
        self.meta_reflections: List[Dict[str, Any]] = []
        self._meta_revision = 0

    def meta_reflect(self, event: str, payload: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        info = dict(payload or {})
        comp_before = self.structural_complexity_stats()
        info.setdefault('structure_rev', self._structure_rev)
        info.setdefault('weights_rev', self._weights_rev)
        info['complexity'] = self._complexity_dict(comp_before)
        try:
            info['signature'] = self.structural_signature()
        except Exception:
            info['signature'] = None
        try:
            info['family_id'] = int(getattr(self, 'family_id', 0))
        except Exception:
            info['family_id'] = None
        info['event'] = event
        info['timestamp'] = time.time()
        self.meta_reflections.append(info)
        self._meta_revision += 1
        return info

    @staticmethod
    def _complexity_dict(stats: Tuple[int, int, float, float, float]) -> Dict[str, float]:
        hidden, edges, branching, depth_spread, score = stats
        return {
            'hidden': int(hidden),
            'edges': int(edges),
            'branching': float(branching),
            'depth_spread': float(depth_spread),
            'score': float(score),
        }

    def copy(self):
        nodes = {nid: _clone_node(n, nid) for nid, n in self.nodes.items()}
        conns = {innov: ConnectionGene(c.in_node, c.out_node, c.weight, c.enabled, c.innovation) for innov, c in self.connections.items()}
        g = Genome(
            nodes,
            conns,
            self.sex,
            self.regen,
            self.regen_mode,
            self.embryo_bias,
            self.id,
            self.birth_gen,
            self.hybrid_scale,
            self.parents,
            mutation_will=self.mutation_will,
            cooperative=self.cooperative,
            family_id=getattr(self, 'family_id', None),
        )
        g.origin_mode = getattr(self, 'origin_mode', 'initial')
        g.max_hidden_nodes = self.max_hidden_nodes
        g.max_edges = self.max_edges
        g._compat_cache = None
        return g

    def invalidate_caches(self, structure: bool=False, weights: bool=False):
        if structure:
            self._structure_rev += 1
            self._topo_cache = None
            self._sorted_innovs_cache = None
            self._conn_index = None
            self._topo_cache_rev = -1
            self._sorted_innovs_rev = -1
            self._conn_index_rev = -1
            self._max_innov_cache = -1
            self._struct_sig_cache = None
            self._struct_sig_rev = -1
            self._complexity_cache = None
            self._complexity_rev = -1
            self._weights_rev += 1
        elif weights:
            self._weights_rev += 1
        self._compiled_cache = None
        self._compiled_cache_rev = (-1, -1)
        self._compiled_cache_tick = -1
        self._compat_token = None
        self._compat_token_rev = (-1, -1)

    def trim_runtime_caches(self, compiled: bool=True, compat: bool=True, topo: bool=False) -> None:
        if compiled:
            self._compiled_cache = None
            self._compiled_cache_rev = (-1, -1)
            self._compiled_cache_tick = -1
        if compat:
            self._compat_token = None
            self._compat_token_rev = (-1, -1)
        if topo:
            self._topo_cache = None
            self._topo_cache_rev = -1

    def _invalidate_cache(self):
        self.invalidate_caches(structure=True)

    def enabled_connections(self):
        return [c for c in self.connections.values() if c.enabled]

    def adjacency(self):
        adj = {}
        for c in self.enabled_connections():
            adj.setdefault(c.in_node, set()).add(c.out_node)
        return adj

    def weighted_adjacency(self, include_disabled: bool=False):
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
        if self._conn_index is None or self._conn_index_rev != self._structure_rev:
            self._conn_index = {(c.in_node, c.out_node) for c in self.connections.values()}
            self._conn_index_rev = self._structure_rev
        return (in_id, out_id) in self._conn_index

    def topological_order(self):
        if self._topo_cache is not None and self._topo_cache_rev == self._structure_rev:
            return list(self._topo_cache)
        in_edges_count = {nid: 0 for nid in self.nodes}
        for c in self.enabled_connections():
            in_edges_count[c.out_node] = in_edges_count.get(c.out_node, 0) + 1
        queue = deque([nid for nid in self.nodes if in_edges_count.get(nid, 0) == 0])
        order: List[int] = []
        adj = self.adjacency()
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for m in adj.get(nid, []):
                in_edges_count[m] -= 1
                if in_edges_count[m] == 0:
                    queue.append(m)
        if len(order) != len(self.nodes):
            raise RuntimeError('Cycle detected: feed-forward constraint violated.')
        self._topo_cache = list(order)
        self._topo_cache_rev = self._structure_rev
        return list(order)

    def sorted_innovations(self) -> List[int]:
        if self._sorted_innovs_cache is not None and self._sorted_innovs_rev == self._structure_rev:
            return self._sorted_innovs_cache
        innovs = sorted(self.connections.keys())
        self._sorted_innovs_cache = list(innovs)
        self._sorted_innovs_rev = self._structure_rev
        self._max_innov_cache = innovs[-1] if innovs else -1
        return self._sorted_innovs_cache

    def max_innovation(self) -> int:
        if self._sorted_innovs_cache is None or self._sorted_innovs_rev != self._structure_rev:
            self.sorted_innovations()
        return int(self._max_innov_cache)

    def _creates_cycle(self, in_node, out_node):
        adj = self.adjacency()
        stack = [out_node]
        visited = set()
        while stack:
            v = stack.pop()
            if v == in_node:
                return True
            if v in visited:
                continue
            visited.add(v)
            for w in adj.get(v, []):
                stack.append(w)
        return False

    def remove_cycles(self):
        """Remove cycles by disabling connections until the genome is acyclic.
        Returns True if any connections were disabled."""
        disabled_any = False
        max_iterations = len(list(self.enabled_connections())) + 1
        iterations = 0
        changed = False
        while iterations < max_iterations:
            iterations += 1
            in_edges_count = {nid: 0 for nid in self.nodes}
            for c in self.enabled_connections():
                in_edges_count[c.out_node] += 1
            queue = deque([nid for nid in self.nodes if in_edges_count[nid] == 0])
            order = []
            adj = self.adjacency()
            while queue:
                nid = queue.popleft()
                order.append(nid)
                for m in adj.get(nid, []):
                    in_edges_count[m] -= 1
                    if in_edges_count[m] == 0:
                        queue.append(m)
            if len(order) == len(self.nodes):
                break
            nodes_in_order = set(order)
            cycle_nodes = set(self.nodes.keys()) - nodes_in_order
            disabled_one = False
            for c in sorted(self.enabled_connections(), key=lambda x: x.innovation, reverse=True):
                if c.in_node in cycle_nodes and c.out_node in cycle_nodes:
                    c.enabled = False
                    disabled_any = True
                    disabled_one = True
                    changed = True
                    break
            if not disabled_one:
                for c in sorted(self.enabled_connections(), key=lambda x: x.innovation, reverse=True):
                    if c.in_node in cycle_nodes or c.out_node in cycle_nodes:
                        c.enabled = False
                        disabled_any = True
                        disabled_one = True
                        changed = True
                        break
            if not disabled_one:
                for c in self.enabled_connections():
                    c.enabled = False
                    disabled_any = True
                    changed = True
                    break
        if changed:
            self.invalidate_caches(structure=True)
        return disabled_any

    def node_depths(self):
        order = self.topological_order()
        inputs = [nid for nid, n in self.nodes.items() if n.type in ('input', 'bias')]
        depth = {nid: 0 if nid in inputs else -1 for nid in order}
        adj_in = {}
        for c in self.enabled_connections():
            adj_in.setdefault(c.out_node, []).append(c.in_node)
        changed = True
        while changed:
            changed = False
            for nid in order:
                if depth[nid] >= 0:
                    continue
                parents = adj_in.get(nid, [])
                if parents and all((p in depth and depth[p] >= 0 for p in parents)):
                    depth[nid] = max((depth[p] for p in parents)) + 1
                    changed = True
        for nid in order:
            if depth[nid] < 0:
                depth[nid] = 0
        return depth

    def structural_signature(self) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[int, int, int], ...]]:
        if self._struct_sig_cache is not None and self._struct_sig_rev == self._structure_rev:
            return self._struct_sig_cache
        node_part = tuple(sorted((n.type, n.activation) for n in self.nodes.values()))
        def _sign_bucket(w: float) -> int:
            if w > 1e-6:
                return 1
            if w < -1e-6:
                return -1
            return 0
        edge_part = tuple(sorted((c.in_node, c.out_node, _sign_bucket(c.weight)) for c in self.connections.values() if c.enabled))
        self._struct_sig_cache = (node_part, edge_part)
        self._struct_sig_rev = self._structure_rev
        return self._struct_sig_cache

    def structural_complexity_stats(self) -> Tuple[int, int, float, float, float]:
        if self._complexity_cache is not None and self._complexity_rev == self._structure_rev:
            return self._complexity_cache
        hidden = sum((1 for n in self.nodes.values() if n.type == 'hidden'))
        enabled_edges = [c for c in self.connections.values() if c.enabled]
        edges = len(enabled_edges)
        branch = 0.0
        out_deg: Dict[int, int] = {}
        for c in enabled_edges:
            out_deg[c.in_node] = out_deg.get(c.in_node, 0) + 1
        for deg in out_deg.values():
            if deg > 1:
                branch += float(deg - 1)
        depth_spread = 0.0
        try:
            depth = self.node_depths()
            if depth:
                vals = list(depth.values())
                depth_spread = float(max(vals) - min(vals))
        except RuntimeError:
            depth_spread = 0.0
        score = float(hidden) + 0.5 * float(edges) + 0.35 * branch + 0.2 * depth_spread
        self._complexity_cache = (hidden, edges, branch, depth_spread, score)
        self._complexity_rev = self._structure_rev
        return self._complexity_cache

    def compat_token(self) -> Tuple[int, int, int]:
        rev = (self._structure_rev, self._weights_rev)
        if self._compat_token is not None and self._compat_token_rev == rev:
            return self._compat_token
        sig = self.structural_signature()
        weights = tuple(sorted((inn, float(round(conn.weight, 6)), bool(conn.enabled)) for inn, conn in self.connections.items()))
        token = (hash(sig), hash(weights), len(weights))
        self._compat_token = token
        self._compat_token_rev = rev
        return token

    def structural_complexity_score(self) -> float:
        return float(self.structural_complexity_stats()[-1])

    def mutate_duplicate_node(self, rng: np.random.Generator, innov: 'InnovationTracker', weight_scale: float=0.85) -> bool:
        hidden_ids = [nid for nid, n in self.nodes.items() if n.type == 'hidden']
        if not hidden_ids:
            return False
        if self.max_hidden_nodes is not None:
            if len(hidden_ids) >= int(self.max_hidden_nodes):
                return False
        template_id = int(rng.choice(hidden_ids))
        baseline = self.structural_complexity_stats()
        incoming = [c for c in self.connections.values() if c.enabled and c.out_node == template_id]
        outgoing = [c for c in self.connections.values() if c.enabled and c.in_node == template_id]
        if not incoming and not outgoing:
            return False
        if self.max_edges is not None:
            enabled_now = sum((1 for c in self.connections.values() if c.enabled))
            budget = int(self.max_edges) - int(enabled_now)
            if budget <= 0:
                return False
        else:
            budget = None
        new_node_id = innov.new_node_id()
        proposals: List[Tuple[int, int, float]] = []
        for c in incoming:
            proposals.append((c.in_node, new_node_id, float(c.weight)))
        for c in outgoing:
            proposals.append((new_node_id, c.out_node, float(c.weight * weight_scale)))
        proposals = [p for p in proposals if not self.has_connection(p[0], p[1]) and not self._creates_cycle(p[0], p[1])]
        if not proposals:
            return False
        if budget is not None and len(proposals) > budget:
            proposals = list(proposals)
            order = rng.permutation(len(proposals))[:budget]
            proposals = [proposals[int(i)] for i in order]
        template = self.nodes[template_id]
        self.nodes[new_node_id] = _clone_node(template, new_node_id, 'hidden', template.activation)
        added = 0
        for src, dst, weight in proposals:
            if budget is not None and added >= budget:
                break
            inn = innov.get_conn_innovation(src, dst)
            self.connections[inn] = ConnectionGene(src, dst, float(weight), True, inn)
            added += 1
        if added == 0:
            self.nodes.pop(new_node_id, None)
            return False
        if budget is None or added < budget:
            if not self.has_connection(template_id, new_node_id) and not self._creates_cycle(template_id, new_node_id):
                try:
                    bridge_inn = innov.get_conn_innovation(template_id, new_node_id)
                    bridge_w = float(rng.normal(1.0, abs(weight_scale)))
                    self.connections[bridge_inn] = ConnectionGene(template_id, new_node_id, bridge_w, True, bridge_inn)
                except Exception:
                    pass
        self.invalidate_caches(structure=True)
        after = self.structural_complexity_stats()
        payload = {
            'template': template_id,
            'new_node': new_node_id,
            'before': self._complexity_dict(baseline),
            'after': self._complexity_dict(after),
            'delta_score': float(after[-1] - baseline[-1]),
        }
        self.meta_reflect('duplicate_node', payload)
        return True

    def mutate_weights(self, rng: np.random.Generator, perturb_chance=0.9, sigma=0.8, reset_range=2.0):
        before = {inn: c.weight for inn, c in self.connections.items()}
        for c in self.connections.values():
            if rng.random() < perturb_chance:
                c.weight += float(rng.normal(0, sigma))
            else:
                c.weight = float(rng.uniform(-reset_range, reset_range))
        self.invalidate_caches(weights=True)
        deltas = [abs(self.connections[inn].weight - w) for inn, w in before.items()]
        payload = {
            'mean_abs_delta': float(np.mean(deltas)) if deltas else 0.0,
            'max_abs_delta': float(np.max(deltas)) if deltas else 0.0,
            'changed': len(deltas),
        }
        self.meta_reflect('mutate_weights', payload)

    def mutate_toggle_enable(self, rng: np.random.Generator, prob=0.01):
        changed = False
        toggled = 0
        for c in self.connections.values():
            if rng.random() >= prob:
                continue
            if c.enabled:
                c.enabled = False
                changed = True
                toggled += 1
            elif not self._creates_cycle(c.in_node, c.out_node):
                c.enabled = True
                changed = True
                toggled += 1
        if changed:
            self.invalidate_caches(structure=True)
            payload = {'toggled': toggled, 'prob': prob}
            self.meta_reflect('toggle_enable', payload)

    def _choose_conn_for_node_add(self, rng: np.random.Generator, bias: str):
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return None
        if bias == 'neutral':
            return enabled[int(rng.integers(len(enabled)))]
        self.remove_cycles()
        try:
            depth = self.node_depths()
        except RuntimeError:
            return enabled[int(rng.integers(len(enabled)))]
        scores = []
        for c in enabled:
            din = depth.get(c.in_node, 0)
            dout = depth.get(c.out_node, din + 1)
            s = 1.0 / (1.0 + din) if bias == 'inputward' else 1.0 + dout
            scores.append(max(0.001, float(s)))
        scores = np.array(scores, float)
        probs = scores / scores.sum()
        idx = int(rng.choice(len(enabled), p=probs))
        return enabled[idx]

    def mutate_add_connection(self, rng: np.random.Generator, innov: 'InnovationTracker', tries=30):
        if self.max_edges is not None:
            if sum((1 for c in self.connections.values() if c.enabled)) >= int(self.max_edges):
                return False
        node_ids = list(self.nodes.keys())
        baseline = self.structural_complexity_stats()
        for _ in range(tries):
            in_id = int(rng.choice(node_ids))
            out_id = int(rng.choice(node_ids))
            in_node = self.nodes[in_id]
            out_node = self.nodes[out_id]
            if in_id == out_id:
                continue
            if in_node.type == 'output':
                continue
            if out_node.type in ('input', 'bias'):
                continue
            if self.has_connection(in_id, out_id):
                continue
            if self._creates_cycle(in_id, out_id):
                continue
            w = float(rng.uniform(-2.0, 2.0))
            inn = innov.get_conn_innovation(in_id, out_id)
            self.connections[inn] = ConnectionGene(in_id, out_id, w, True, inn)
            self._invalidate_cache()
            after = self.structural_complexity_stats()
            payload = {
                'in': in_id,
                'out': out_id,
                'weight': w,
                'before': self._complexity_dict(baseline),
                'after': self._complexity_dict(after),
                'delta_score': float(after[-1] - baseline[-1]),
            }
            self.meta_reflect('add_connection', payload)
            return True
        return False

    def mutate_add_node(self, rng: np.random.Generator, innov: 'InnovationTracker'):
        if self.max_hidden_nodes is not None:
            if sum((1 for n in self.nodes.values() if n.type == 'hidden')) >= int(self.max_hidden_nodes):
                return False
        chosen = self._choose_conn_for_node_add(rng, self.embryo_bias)
        if chosen is None:
            return False
        c = chosen
        if not c.enabled:
            return False
        baseline = self.structural_complexity_stats()
        split_edge = (c.in_node, c.out_node)
        c.enabled = False
        new_nid = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_nid not in self.nodes:
            self.nodes[new_nid] = NodeGene(
                new_nid,
                'hidden',
                'tanh',
                float(rng.uniform(0.9, 1.1)),
                float(np.clip(rng.normal(0.0, 0.04), -0.15, 0.15)),
                0.0,
                0.0,
                float(np.clip(rng.normal(0.5, 0.08), 0.0, 1.0)),
                float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0)),
                float(np.clip(rng.gamma(2.0, 0.2), 0.0, 3.0)),
            )
        inn1 = innov.get_conn_innovation(c.in_node, new_nid)
        inn2 = innov.get_conn_innovation(new_nid, c.out_node)
        self.connections[inn1] = ConnectionGene(c.in_node, new_nid, 1.0, True, inn1)
        self.connections[inn2] = ConnectionGene(new_nid, c.out_node, c.weight, True, inn2)
        self.invalidate_caches(structure=True)
        after = self.structural_complexity_stats()
        payload = {
            'new_node': new_nid,
            'split_edge': split_edge,
            'before': self._complexity_dict(baseline),
            'after': self._complexity_dict(after),
            'delta_score': float(after[-1] - baseline[-1]),
        }
        self.meta_reflect('add_node', payload)
        return True

    def mutate_sex(self, rng: np.random.Generator):
        """Mutate sex, with low probability of becoming hermaphrodite.
        
        The mutation probability is controlled by the NEAT instance's mutate_sex_prob parameter.
        Only male or female individuals can mutate into hermaphrodites.
        """
        if self.sex in ('male', 'female'):
            self.sex = 'hermaphrodite'
            return True
        return False

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        order = self.topological_order()
        incoming = {nid: [] for nid in order}
        for c in self.enabled_connections():
            incoming[c.out_node].append((c.in_node, c.weight))
        input_ids = [nid for nid, n in self.nodes.items() if n.type == 'input']
        output_ids = [nid for nid, n in self.nodes.items() if n.type == 'output']
        bias_ids = [nid for nid, n in self.nodes.items() if n.type == 'bias']
        assert len(bias_ids) == 1
        bias_id = bias_ids[0]
        n_samples = X.shape[0]
        values = {nid: np.zeros(n_samples) for nid in order}
        in_sorted = sorted(input_ids)
        assert X.shape[1] == len(in_sorted)
        for i, nid in enumerate(in_sorted):
            values[nid] = X[:, i]
        values[bias_id] = np.ones(n_samples)
        for nid in order:
            node = self.nodes[nid]
            if node.type in ('input', 'bias'):
                continue
            s = np.zeros(n_samples)
            for src, w in incoming[nid]:
                s += values[src] * w
            if node.activation == 'tanh':
                act = np.tanh(s)
            elif node.activation == 'sigmoid':
                act = 1 / (1 + np.exp(-s))
            elif node.activation == 'relu':
                act = np.maximum(0.0, s)
            elif node.activation == 'identity':
                act = s
            else:
                act = np.tanh(s)
            values[nid] = act
        out_sorted = sorted(output_ids)
        Y = np.stack([values[nid] for nid in out_sorted], axis=1)
        return Y

    def forward_one(self, x: np.ndarray) -> np.ndarray:
        order = self.topological_order()
        incoming = {nid: [] for nid in order}
        for c in self.enabled_connections():
            incoming[c.out_node].append((c.in_node, c.weight))
        input_ids = sorted([nid for nid, n in self.nodes.items() if n.type == 'input'])
        output_ids = sorted([nid for nid, n in self.nodes.items() if n.type == 'output'])
        bias_id = next((nid for nid, n in self.nodes.items() if n.type == 'bias'))
        vals = {nid: 0.0 for nid in order}
        for i, nid in enumerate(input_ids):
            vals[nid] = float(x[i])
        vals[bias_id] = 1.0
        for nid in order:
            node = self.nodes[nid]
            if node.type in ('input', 'bias'):
                continue
            s = 0.0
            for src, w in incoming[nid]:
                s += vals[src] * w
            if node.activation == 'tanh':
                y = math.tanh(s)
            elif node.activation == 'sigmoid':
                y = 1.0 / (1.0 + math.exp(-s))
            elif node.activation == 'relu':
                y = s if s > 0 else 0.0
            elif node.activation == 'identity':
                y = s
            else:
                y = math.tanh(s)
            vals[nid] = y
        return np.array([vals[nid] for nid in output_ids], dtype=np.float32)

def summarize_graph_changes(adj0: Dict[int, List[Tuple[int, float]]], adj1: Dict[int, List[Tuple[int, float]]], weight_tol: float=0.0) -> Tuple[Set[int], int]:
    """Return nodes and edge-count touched by structural/weight deltas (|w|<=tol treated absent)."""

    def collect(adj):
        edges = {}
        nodes = set(adj.keys())
        for u, nbrs in adj.items():
            nodes.add(u)
            for v, w in nbrs:
                if abs(w) <= weight_tol:
                    continue
                edges[u, v] = w
                nodes.add(v)
        return (edges, nodes)
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
    return (changed_nodes, changed_edges)

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
        if s == '':
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
        if s == '':
            return None
        try:
            return float(s)
        except ValueError:
            return None
    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                gen = _as_int(raw.get('gen'))
                out_id = _as_int(raw.get('o'))
                if gen is None or out_id is None:
                    continue
                lineage_raw = raw.get('lineage_id')
                lineage_id = _as_int(lineage_raw)
                row = {'gen': gen, 'lineage_id': lineage_id if lineage_id is not None else lineage_raw, 'mut_id': raw.get('mut_id'), 'o': out_id, 'changed_nodes': _as_int(raw.get('changed_nodes')) or 0, 'changed_edges': _as_int(raw.get('changed_edges')) or 0, 'R0': int(_as_int(raw.get('R0')) or 0), 'R1': int(_as_int(raw.get('R1')) or 0), 'P0': _as_int(raw.get('P0')) or 0, 'P1': _as_int(raw.get('P1')) or 0, 'd0': _as_int(raw.get('d0')), 'd1': _as_int(raw.get('d1')), 'detour': _as_float(raw.get('detour')), 'delta_paths': _as_int(raw.get('delta_paths')) or 0, 'delta_sp': _as_int(raw.get('delta_sp')), 'heal_flag': int(_as_int(raw.get('heal_flag')) or 0), 'time_to_heal': _as_int(raw.get('time_to_heal')), 'disjoint_paths0': _as_int(raw.get('disjoint_paths0')) or 0, 'disjoint_paths1': _as_int(raw.get('disjoint_paths1')) or 0}
                rows.append(row)
    except FileNotFoundError:
        return []
    return rows

def _prepare_lcs_series(lcs_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_output: Dict[int, 'OrderedDict[int, Dict[str, Any]]'] = {}
    per_gen: Dict[int, Dict[str, Any]] = {}
    gens: Set[int] = set()
    for row in lcs_rows:
        gen = row.get('gen')
        out_id = row.get('o')
        if gen is None or out_id is None:
            continue
        gens.add(gen)
        per_output.setdefault(out_id, OrderedDict())[gen] = row
        agg = per_gen.setdefault(gen, {'count': 0, 'paths1': [], 'disjoint1': [], 'detour': [], 'delta_paths': [], 'changed_edges': 0, 'heals': 0, 'breaks': 0, 'time_to_heal': [], 'connected': 0})
        agg['count'] += 1
        agg['changed_edges'] += row.get('changed_edges', 0) or 0
        agg['paths1'].append(row.get('P1', 0) or 0)
        agg['disjoint1'].append(row.get('disjoint_paths1', 0) or 0)
        detour_val = row.get('detour')
        if detour_val is not None and (not np.isnan(detour_val)):
            agg['detour'].append(float(detour_val))
        agg['delta_paths'].append(row.get('delta_paths', 0) or 0)
        if row.get('R0', 0) == 1 and row.get('R1', 0) == 0:
            agg['breaks'] += 1
        if row.get('heal_flag', 0):
            agg['heals'] += 1
        tth = row.get('time_to_heal')
        if tth is not None:
            agg['time_to_heal'].append(tth)
        if row.get('R1', 0):
            agg['connected'] += 1
    outputs = sorted(per_output.keys())
    for out_id in outputs:
        per_output[out_id] = OrderedDict(sorted(per_output[out_id].items()))
    generations = sorted(gens)
    return {'per_output': per_output, 'per_gen': per_gen, 'outputs': outputs, 'generations': generations}

def _latest_gen_summary(series: Optional[Dict[str, Any]], upto_gen: int) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not series:
        return (None, None)
    gens: List[int] = series.get('generations', [])
    for g in reversed(gens):
        if g <= upto_gen:
            return (g, series.get('per_gen', {}).get(g))
    return (None, None)

def _cumulative_lcs_counts(series: Optional[Dict[str, Any]], upto_gen: int) -> Tuple[int, int]:
    if not series:
        return (0, 0)
    heals = 0
    breaks = 0
    for gen, summary in series.get('per_gen', {}).items():
        if gen <= upto_gen:
            heals += int(summary.get('heals', 0))
            breaks += int(summary.get('breaks', 0))
    return (heals, breaks)

def _format_lcs_summary(summary: Optional[Dict[str, Any]]) -> str:
    if not summary or summary.get('count', 0) == 0:
        return 'LCS: no connectivity data yet'

    def _mean(vals: List[float], default: float=0.0) -> float:
        return float(np.mean(vals)) if vals else default

    def _median(vals: List[float]) -> Optional[float]:
        return float(np.median(vals)) if vals else None
    avg_paths = _mean(summary.get('paths1', []))
    avg_disjoint = _mean(summary.get('disjoint1', []))
    mean_delta = _mean(summary.get('delta_paths', []))
    med_detour = _median(summary.get('detour', []))
    med_tth = _median(summary.get('time_to_heal', []))
    connected_ratio = float(summary.get('connected', 0)) / float(summary.get('count', 1)) if summary.get('count', 0) else 0.0
    detour_str = f'detour≈{med_detour:.2f}' if med_detour is not None else 'detour=—'
    tth_str = f'Tth≈{med_tth:.1f}' if med_tth is not None else 'Tth=—'
    return f"altμ {avg_paths:.2f} | disjointμ {avg_disjoint:.2f} | Δpaths {mean_delta:+.2f} | {detour_str} | {tth_str} | conn {connected_ratio:.2f} | H:{summary.get('heals', 0)} B:{summary.get('breaks', 0)} | Δedges {summary.get('changed_edges', 0)}"

def export_lcs_ribbon_png(lcs_rows: List[Dict[str, Any]], path: str, series: Optional[Dict[str, Any]]=None, outputs: Optional[Iterable[int]]=None, dpi: int=200) -> Optional[str]:
    """Render the canonical LCS ribbon PNG using the hardened _savefig helper."""
    if not lcs_rows:
        return None
    if series is None:
        series = _prepare_lcs_series(lcs_rows)
    outputs_list = list(outputs) if outputs is not None else list(series.get('outputs', []))
    if not outputs_list:
        return None
    import matplotlib.pyplot as _plt
    styles = ['solid', (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (3, 2, 1, 2))]
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '+']
    fig, axes = _plt.subplots(3, 1, sharex=True, figsize=(7.2, 7.4), dpi=dpi)
    detour_values = []
    for _row in lcs_rows:
        val = _row.get('detour')
        if val is not None and (not np.isnan(val)):
            detour_values.append(float(val))
    alt_max = max((r.get('P1', 0) or 0 for r in lcs_rows))
    dis_max = max((r.get('disjoint_paths1', 0) or 0 for r in lcs_rows))
    for idx, out_id in enumerate(outputs_list):
        data_dict = series.get('per_output', {}).get(out_id)
        if not data_dict:
            continue
        gens = list(data_dict.keys())
        alt = [data_dict[g]['P1'] for g in gens]
        dis = [data_dict[g]['disjoint_paths1'] for g in gens]
        det = [data_dict[g]['detour'] if data_dict[g]['detour'] is not None else np.nan for g in gens]
        conn = [data_dict[g]['R1'] for g in gens]
        heals = [data_dict[g]['heal_flag'] for g in gens]
        style = styles[idx % len(styles)]
        marker = markers[idx % len(markers)]
        label = f'output {out_id}'
        axes[0].plot(gens, alt, linestyle=style, marker=marker, markersize=4.0, color='black', linewidth=1.5, label=label)
        axes[1].plot(gens, dis, linestyle=style, marker=marker, markersize=4.0, color='black', linewidth=1.5)
        axes[2].plot(gens, det, linestyle=style, marker=marker, markersize=4.0, color='black', linewidth=1.5)
        for g_val, alt_val, connected in zip(gens, alt, conn):
            if not connected:
                axes[0].scatter([g_val], [alt_val], marker='x', color='black', s=36, linewidths=1.2)
        for g_val, alt_val, heal, det_val in zip(gens, alt, heals, det):
            if heal:
                axes[0].scatter([g_val], [alt_val], marker='D', facecolors='none', edgecolors='black', s=58, linewidths=1.0)
                if not np.isnan(det_val):
                    axes[2].scatter([g_val], [det_val], marker='D', facecolors='none', edgecolors='black', s=58, linewidths=1.0)
    axes[0].set_ylabel('alt paths')
    axes[1].set_ylabel('edge-disjoint')
    axes[2].set_ylabel('detour')
    axes[2].set_xlabel('generation')
    axes[0].set_ylim(-0.1, max(1.0, alt_max) + 1.0)
    axes[1].set_ylim(-0.1, max(1.0, dis_max) + 1.0)
    if detour_values:
        d_min = min(detour_values)
        d_max = max(detour_values)
        pad = max(0.05, 0.05 * d_max)
        axes[2].set_ylim(max(0.0, d_min - pad), d_max + pad)
    else:
        axes[2].set_ylim(0.8, 1.4)
    axes[0].legend(loc='upper left', frameon=False)
    for ax in axes:
        ax.grid(True, color='0.85', linestyle=(0, (1, 3)), linewidth=0.6)
    last_gen = series.get('generations', [])[-1] if series.get('generations') else None
    _, summary = _latest_gen_summary(series, last_gen) if last_gen is not None else (None, None)
    summary_text = _format_lcs_summary(summary)
    fig.suptitle('Local Continuity Signature overview', y=0.98, fontsize=12)
    fig.text(0.02, 0.03, summary_text, fontsize=9, family='monospace')
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    _savefig(fig, path, dpi=dpi)
    _plt.close(fig)
    return path

def export_lcs_timeline_gif(lcs_rows: List[Dict[str, Any]], path: str, series: Optional[Dict[str, Any]]=None, fps: int=6, dpi: int=150) -> Optional[str]:
    if not lcs_rows:
        return None
    if series is None:
        series = _prepare_lcs_series(lcs_rows)
    gens = list(series.get('generations', []))
    if not gens:
        return None
    import matplotlib.pyplot as _plt
    styles = ['solid', (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (3, 2, 1, 2))]
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '+']
    first_gen = gens[0]
    alt_max = max((r.get('P1', 0) or 0 for r in lcs_rows))
    dis_max = max((r.get('disjoint_paths1', 0) or 0 for r in lcs_rows))
    detour_values = []
    for _row in lcs_rows:
        val = _row.get('detour')
        if val is not None and (not np.isnan(val)):
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
        for idx, out_id in enumerate(series.get('outputs', [])):
            data_dict = series.get('per_output', {}).get(out_id)
            if not data_dict:
                continue
            use_gens = [g for g in data_dict.keys() if g <= upto]
            if not use_gens:
                continue
            alt = [data_dict[g]['P1'] for g in use_gens]
            dis = [data_dict[g]['disjoint_paths1'] for g in use_gens]
            det = [data_dict[g]['detour'] if data_dict[g]['detour'] is not None else np.nan for g in use_gens]
            conn = [data_dict[g]['R1'] for g in use_gens]
            heals = [data_dict[g]['heal_flag'] for g in use_gens]
            style = styles[idx % len(styles)]
            marker = markers[idx % len(markers)]
            label = f'output {out_id}' if upto == first_gen else '_nolegend_'
            axes[0].plot(use_gens, alt, linestyle=style, marker=marker, markersize=4.0, color='black', linewidth=1.4, label=label)
            axes[1].plot(use_gens, dis, linestyle=style, marker=marker, markersize=4.0, color='black', linewidth=1.4)
            axes[2].plot(use_gens, det, linestyle=style, marker=marker, markersize=4.0, color='black', linewidth=1.4)
            for g_val, alt_val, connected in zip(use_gens, alt, conn):
                if not connected:
                    axes[0].scatter([g_val], [alt_val], marker='x', color='black', s=32, linewidths=1.1)
            for g_val, alt_val, heal, det_val in zip(use_gens, alt, heals, det):
                if heal:
                    axes[0].scatter([g_val], [alt_val], marker='D', facecolors='none', edgecolors='black', s=54, linewidths=1.0)
                    if not np.isnan(det_val):
                        axes[2].scatter([g_val], [det_val], marker='D', facecolors='none', edgecolors='black', s=54, linewidths=1.0)
        for ax in axes:
            ax.axvline(upto, color='0.35', linestyle=(0, (2, 3)), linewidth=1.0)
            ax.grid(True, color='0.85', linestyle=(0, (1, 3)), linewidth=0.6)
        axes[0].set_ylabel('alt paths')
        axes[1].set_ylabel('edge-disjoint')
        axes[2].set_ylabel('detour')
        axes[2].set_xlabel('generation')
        axes[0].set_ylim(-0.1, max(1.0, alt_max) + 1.0)
        axes[1].set_ylim(-0.1, max(1.0, dis_max) + 1.0)
        axes[2].set_ylim(*det_ylim)
        axes[0].legend(loc='upper left', frameon=False)
        gen_key, summary = _latest_gen_summary(series, upto)
        summary_line = _format_lcs_summary(summary)
        cumulative = [r for r in lcs_rows if r.get('gen') is not None and r['gen'] <= upto]
        cum_heals = sum((r.get('heal_flag', 0) for r in cumulative))
        cum_breaks = sum((1 for r in cumulative if r.get('R0', 0) == 1 and r.get('R1', 0) == 0))
        fig.suptitle(f'LCS timeline ≤ Gen {upto}', y=0.97, fontsize=12)
        fig.text(0.02, 0.06, summary_line, fontsize=9, family='monospace')
        fig.text(0.02, 0.03, f'cum heals {cum_heals} | cum breaks {cum_breaks}', fontsize=8, family='monospace')
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        frame = _fig_to_rgb(fig)
        _plt.close(fig)
        frames.append(frame)
    if not frames:
        return None
    _mimsave(path, frames, fps=max(1, fps))
    return path

def compatibility_distance(g1: Genome, g2: Genome, c1=1.0, c2=1.0, c3=0.4):
    innovs1 = g1.sorted_innovations()
    innovs2 = g2.sorted_innovations()
    i = j = 0
    E = D = 0
    W_diffs = []
    max_innov1 = g1.max_innovation()
    max_innov2 = g2.max_innovation()
    while i < len(innovs1) and j < len(innovs2):
        in1 = innovs1[i]
        in2 = innovs2[j]
        if in1 == in2:
            W_diffs.append(abs(g1.connections[in1].weight - g2.connections[in2].weight))
            i += 1
            j += 1
        elif in1 < in2:
            if in1 > max_innov2:
                E += 1
            else:
                D += 1
            i += 1
        else:
            if in2 > max_innov1:
                E += 1
            else:
                D += 1
            j += 1
    E += len(innovs1) - i + (len(innovs2) - j)
    N = max(len(innovs1), len(innovs2))
    N = 1 if N < 20 else N
    W = sum(W_diffs) / len(W_diffs) if W_diffs else 0.0
    return c1 * E / N + c2 * D / N + c3 * W


class HouseholdManager:

    """Track structural "households" to drive environment diversity and reuse stats.

    The manager clusters genomes by a hashed structural signature, remembers their
    rolling fitness and complexity, and provides difficulty nudges per household.
    """

    def __init__(self, max_households: int=256, smoothing: float=0.6, difficulty_push: float=0.12):
        self.max_households = int(max(1, max_households))
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.difficulty_push = float(difficulty_push)
        self._stats: "OrderedDict[Tuple[int, int, int, str], Dict[str, float]]" = OrderedDict()

    def _key(self, genome: Genome) -> Tuple[int, int, int, str]:
        token = genome.compat_token()
        comp = genome.structural_complexity_stats()[-1]
        band = int(comp // 1.5)
        regen = 1 if getattr(genome, 'regen', False) else 0
        sex = str(getattr(genome, 'sex', 'unknown'))
        return (token[0], token[1], band * 2 + regen, sex)

    def _prune(self):
        while len(self._stats) > self.max_households:
            oldest = min(self._stats.items(), key=lambda kv: kv[1].get('last_seen', 0.0))[0]
            self._stats.pop(oldest, None)

    def update(self, genomes: List[Genome], fitnesses: List[float], env_difficulty: float) -> None:
        if not genomes:
            return
        now = time.time()
        smooth = self.smoothing
        for g, fit in zip(genomes, fitnesses):
            if not np.isfinite(fit):
                continue
            key = self._key(g)
            entry = self._stats.get(key)
            weight = 0.6 if not getattr(g, 'cooperative', True) else 1.0
            comp = float(g.structural_complexity_stats()[-1])
            if entry is None:
                entry = {
                    'mean_fit': float(fit),
                    'trend': 0.0,
                    'complexity': comp,
                    'count': weight,
                    'last_seen': now,
                    'last_fit': float(fit),
                    'env_difficulty': float(env_difficulty),
                }
            else:
                entry['mean_fit'] = entry['mean_fit'] * smooth + float(fit) * (1.0 - smooth) * weight
                delta = float(fit) - float(entry.get('last_fit', fit))
                entry['trend'] = entry['trend'] * smooth + delta * (1.0 - smooth)
                entry['complexity'] = entry['complexity'] * smooth + comp * (1.0 - smooth)
                entry['count'] = entry.get('count', 0.0) * smooth + weight * (1.0 - smooth)
                entry['last_seen'] = now
                entry['last_fit'] = float(fit)
                entry['env_difficulty'] = float(env_difficulty)
            self._stats[key] = entry
        self._prune()

    def environment_adjustments(self, genomes: List[Genome], fitnesses: List[float], env: Dict[str, Any]) -> Dict[int, float]:
        if not self._stats:
            return {}
        means = np.array([entry['mean_fit'] for entry in self._stats.values()], dtype=float)
        if means.size == 0:
            return {}
        base_mean = float(means.mean())
        spread = float(np.std(means)) if means.size > 1 else 0.0
        env_diff = 0.0
        if isinstance(env, dict):
            env_diff = float(env.get('difficulty', 0.0))
        else:
            env_diff = float(env)
        adjustments: Dict[int, float] = {}
        push = self.difficulty_push * (1.0 + 0.25 * np.tanh(spread))
        for g, fit in zip(genomes, fitnesses):
            key = self._key(g)
            entry = self._stats.get(key)
            if entry is None:
                continue
            comp = float(entry.get('complexity', 0.0))
            trend = float(entry.get('trend', 0.0))
            normalized = 0.0
            if spread > 1e-09:
                normalized = (entry['mean_fit'] - base_mean) / spread
            else:
                normalized = entry['mean_fit'] - base_mean
            boost = push * np.tanh(normalized + 0.2 * trend)
            boost *= 1.0 + 0.15 * np.tanh((comp - (2.0 + env_diff)) / 3.0)
            if not getattr(g, 'cooperative', True):
                boost *= 0.5
            adjustments[g.id] = float(boost)
        return adjustments

    def global_pressure(self) -> float:
        if not self._stats:
            return 0.0
        means = np.array([entry['mean_fit'] for entry in self._stats.values()], dtype=float)
        if means.size <= 1:
            return 0.0
        spread = float(np.std(means))
        return float(np.tanh(spread * 1.2))

def _regenerate_head(g: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5):
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input', 'bias')]
    candidates = [c for c in g.enabled_connections() if c.in_node in inputs]
    if not candidates:
        return g
    rng.shuffle(candidates)
    frac = min(0.8, 0.15 + 0.7 * float(intensity))
    k = int(len(candidates) * frac)
    for c in candidates[:k]:
        c.enabled = False
    chosen = rng.choice(candidates)
    new_id = innov.new_node_id()
    g.nodes[new_id] = _random_hidden_node(new_id, rng)
    inn1 = innov.get_conn_innovation(chosen.in_node, new_id)
    inn2 = innov.get_conn_innovation(new_id, chosen.out_node)
    g.connections[inn1] = ConnectionGene(chosen.in_node, new_id, 1.0, True, inn1)
    g.connections[inn2] = ConnectionGene(new_id, chosen.out_node, chosen.weight, True, inn2)
    g.invalidate_caches(structure=True)
    return g

def _regenerate_tail(g: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5):
    outputs = [nid for nid, n in g.nodes.items() if n.type == 'output']
    sinks = [c for c in g.enabled_connections() if c.out_node in outputs]
    if not sinks:
        return g
    rng.shuffle(sinks)
    k = max(1, int(len(sinks) * (0.2 + 0.6 * float(intensity))))
    hidden = [nid for nid, n in g.nodes.items() if n.type == 'hidden']
    changed = False
    for c in sinks[:k]:
        c.weight = float(rng.uniform(-2, 2))
        changed = True
        if hidden and rng.random() < 0.3 + 0.5 * float(intensity):
            new_src = int(rng.choice(hidden))
            if not g.has_connection(new_src, c.out_node) and (not g._creates_cycle(new_src, c.out_node)):
                inn = innov.get_conn_innovation(new_src, c.out_node)
                g.connections[inn] = ConnectionGene(new_src, c.out_node, float(rng.uniform(-2, 2)), True, inn)
                changed = True
    if changed:
        g.invalidate_caches(structure=True)
    return g

def _regenerate_split(g: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5):
    hidden = [nid for nid, n in g.nodes.items() if n.type == 'hidden']
    if not hidden:
        enabled = [c for c in g.connections.values() if c.enabled]
        if not enabled:
            return g
        c = enabled[int(rng.integers(len(enabled)))]
        c.enabled = False
        new_nid = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_nid not in g.nodes:
            g.nodes[new_nid] = _random_hidden_node(new_nid, rng)
        inn1 = innov.get_conn_innovation(c.in_node, new_nid)
        inn2 = innov.get_conn_innovation(new_nid, c.out_node)
        g.connections[inn1] = ConnectionGene(c.in_node, new_nid, 1.0, True, inn1)
        g.connections[inn2] = ConnectionGene(new_nid, c.out_node, c.weight, True, inn2)
        g.invalidate_caches(structure=True)
        return g
    target = int(rng.choice(hidden))
    dup_id = innov.new_node_id()
    g.nodes[dup_id] = _random_hidden_node(dup_id, rng)
    incomings = [c for c in g.enabled_connections() if c.out_node == target]
    for cin in incomings:
        inn = innov.get_conn_innovation(cin.in_node, dup_id)
        g.connections[inn] = ConnectionGene(cin.in_node, dup_id, cin.weight + float(rng.normal(0, 0.1)), True, inn)
    outgoings = [c for c in g.enabled_connections() if c.in_node == target]
    move_p = min(0.9, 0.3 + 0.6 * float(intensity))
    changed = bool(incomings)
    for cout in outgoings:
        if rng.random() < move_p:
            cout.enabled = False
            inn = innov.get_conn_innovation(dup_id, cout.out_node)
            g.connections[inn] = ConnectionGene(dup_id, cout.out_node, cout.weight + float(rng.normal(0, 0.1)), True, inn)
            changed = True
        else:
            changed = True
    if changed:
        g.invalidate_caches(structure=True)
    return g

def _reachable_outputs_fraction(g, eps=0.0) -> float:
    """Fraction of outputs that are reachable from any input via edges with |w|>eps."""
    adj = g.weighted_adjacency()
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input', 'bias')]
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
                seen.add(v)
                q.append(v)
    cnt = sum((1 for o in outputs if o in seen))
    return float(cnt) / max(1, len(outputs))

def _connectivity_guard(g, innov, rng, min_frac=0.6, max_new_edges=16, eps=0.0):
    """If reachability falls below min_frac, add safe forward edges."""
    frac = _reachable_outputs_fraction(g, eps=eps)
    if frac >= min_frac:
        return
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input', 'bias')]
    outputs = [nid for nid, n in g.nodes.items() if n.type == 'output']
    adj = g.weighted_adjacency()
    seen = set(inputs)
    q = deque(inputs)
    while q:
        u = q.popleft()
        for v, w in adj.get(u, []):
            if abs(w) <= eps:
                continue
            if v not in seen:
                seen.add(v)
                q.append(v)
    sources = sorted([nid for nid in seen if g.nodes[nid].type in ('hidden', 'input', 'bias')])
    unreachable = [o for o in outputs if o not in seen]
    attempts = 0
    rng_local = rng or np.random.default_rng()
    while unreachable and attempts < int(max_new_edges):
        if not sources:
            break
        src = int(rng_local.choice(sources))
        out = int(rng_local.choice(unreachable))
        if not g.has_connection(src, out) and (not g._creates_cycle(src, out)):
            inn = innov.get_conn_innovation(src, out)
            g.connections[inn] = ConnectionGene(src, out, float(rng_local.uniform(0.6, 1.6)), True, inn)
            adj = g.weighted_adjacency()
            seen = set(inputs)
            q = deque(inputs)
            while q:
                u = q.popleft()
                for v, w in adj.get(u, []):
                    if abs(w) <= eps:
                        continue
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            unreachable = [o for o in outputs if o not in seen]
        attempts += 1
    if attempts:
        g.invalidate_caches(structure=True)

def _soft_regenerate_head(g, rng, innov, intensity=0.5):
    inputs = [nid for nid, n in g.nodes.items() if n.type in ('input', 'bias')]
    candidates = [c for c in g.enabled_connections() if c.in_node in inputs]
    if not candidates:
        return g
    rng_local = rng or np.random.default_rng()
    keep_rate = max(0.65, 1.0 - 0.5 * float(intensity))
    n = len(candidates)
    n_disable = int(min(n * (1.0 - keep_rate) * 0.4, max(1, 0.1 * n)))
    n_atten = int(min(n * (1.0 - keep_rate), n - n_disable))
    idx = np.arange(n)
    rng_local.shuffle(idx)
    changed = False
    for k in idx[:n_atten]:
        c = candidates[k]
        c.weight *= float(rng_local.uniform(0.6, 0.9))
        changed = True
    for k in idx[n_atten:n_atten + n_disable]:
        c = candidates[k]
        c.enabled = False
        changed = True
    m = int(1 + round(2 * float(intensity)))
    for _ in range(m):
        c = candidates[int(rng_local.integers(n))]
        new_id = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_id not in g.nodes:
            g.nodes[new_id] = _random_hidden_node(new_id, rng_local)
        inn1 = innov.get_conn_innovation(c.in_node, new_id)
        inn2 = innov.get_conn_innovation(new_id, c.out_node)
        g.connections[inn1] = ConnectionGene(c.in_node, new_id, 1.0, True, inn1)
        g.connections[inn2] = ConnectionGene(new_id, c.out_node, c.weight, True, inn2)
        changed = True
    if changed:
        g.invalidate_caches(structure=True)
    return g

def _soft_regenerate_tail(g, rng, innov, intensity=0.5):
    outputs = [nid for nid, n in g.nodes.items() if n.type == 'output']
    sinks = [c for c in g.enabled_connections() if c.out_node in outputs]
    if not sinks:
        return g
    rng_local = rng or np.random.default_rng()
    k = max(1, int(len(sinks) * (0.15 + 0.45 * float(intensity))))
    hidden = [nid for nid, n in g.nodes.items() if n.type == 'hidden']
    choose = sinks if k >= len(sinks) else list(rng_local.choice(sinks, size=k, replace=False))
    changed = False
    for c in choose:
        c.weight = float(rng_local.uniform(-1.8, 1.8))
        changed = True
        if hidden and rng_local.random() < 0.25 + 0.35 * float(intensity):
            new_src = int(rng_local.choice(hidden))
            if not g.has_connection(new_src, c.out_node) and (not g._creates_cycle(new_src, c.out_node)):
                inn = innov.get_conn_innovation(new_src, c.out_node)
                g.connections[inn] = ConnectionGene(new_src, c.out_node, float(rng_local.uniform(-1.6, 1.6)), True, inn)
                changed = True
    if changed:
        g.invalidate_caches(structure=True)
    return g

def _soft_regenerate_split(g, rng, innov, intensity=0.5):
    hidden = [nid for nid, n in g.nodes.items() if n.type == 'hidden']
    rng_local = rng or np.random.default_rng()
    if not hidden:
        enabled = [c for c in g.connections.values() if c.enabled]
        if not enabled:
            return g
        c = enabled[int(rng_local.integers(len(enabled)))]
        c.enabled = False
        new_nid = innov.get_or_create_split_node(c.in_node, c.out_node)
        if new_nid not in g.nodes:
            g.nodes[new_nid] = _random_hidden_node(new_nid, rng_local)
        inn1 = innov.get_conn_innovation(c.in_node, new_nid)
        inn2 = innov.get_conn_innovation(new_nid, c.out_node)
        g.connections[inn1] = ConnectionGene(c.in_node, new_nid, 1.0, True, inn1)
        g.connections[inn2] = ConnectionGene(new_nid, c.out_node, c.weight, True, inn2)
        g.invalidate_caches(structure=True)
        return g
    target = int(rng_local.choice(hidden))
    dup_id = innov.new_node_id()
    g.nodes[dup_id] = _random_hidden_node(dup_id, rng_local)
    incomings = [c for c in g.enabled_connections() if c.out_node == target]
    changed = False
    for cin in incomings:
        inn = innov.get_conn_innovation(cin.in_node, dup_id)
        g.connections[inn] = ConnectionGene(cin.in_node, dup_id, cin.weight + float(rng_local.normal(0, 0.08)), True, inn)
        changed = True
    outgoings = [c for c in g.enabled_connections() if c.in_node == target]
    move_p = min(0.7, 0.25 + 0.45 * float(intensity))
    for cout in outgoings:
        if rng_local.random() < move_p:
            cout.enabled = False
            inn = innov.get_conn_innovation(dup_id, cout.out_node)
            g.connections[inn] = ConnectionGene(dup_id, cout.out_node, cout.weight + float(rng_local.normal(0, 0.08)), True, inn)
            changed = True
        else:
            changed = True
    if outgoings and rng_local.random() < 0.3:
        pick = outgoings[int(rng_local.integers(len(outgoings)))]
        if not g.has_connection(target, pick.out_node) and (not g._creates_cycle(target, pick.out_node)):
            inn = innov.get_conn_innovation(target, pick.out_node)
            g.connections[inn] = ConnectionGene(target, pick.out_node, pick.weight, True, inn)
            changed = True
    if changed:
        g.invalidate_caches(structure=True)
    return g

def platyregenerate(genome: Genome, rng: np.random.Generator, innov: InnovationTracker, intensity=0.5) -> Genome:
    """Soft regeneration + connectivity guard."""
    g = genome.copy()
    mode = g.regen_mode or 'split'
    if mode == 'head':
        g = _soft_regenerate_head(g, rng, innov, intensity)
    elif mode == 'tail':
        g = _soft_regenerate_tail(g, rng, innov, intensity)
    else:
        g = _soft_regenerate_split(g, rng, innov, intensity)
    eps = 0.0
    try:
        monitor = getattr(getattr(genome, 'lcs_monitor', None), 'eps', None)
        eps = float(monitor or 0.0)
    except Exception:
        pass
    _connectivity_guard(g, innov, rng, min_frac=getattr(genome, 'min_conn_after_regen', 0.65), eps=eps)
    if hasattr(g, '_invalidate_cache'):
        g._invalidate_cache()
    return g

@dataclass
class EvalMode:
    vanilla: bool = True
    enable_regen_reproduction: bool = False
    complexity_alpha: float = 0.01
    node_penalty: float = 0.3
    edge_penalty: float = 0.15
    species_low: int = 3
    species_high: int = 8

@dataclass
class PairState:
    status: str = 'connected'
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
    csv_path: str = 'regen_log.csv'
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
                for v in self._nbr_targets(adj.get(u, ())):
                    if v not in scope:
                        scope.add(v)
                        nxt.append(v)
            for p, nbrs in adj.items():
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
            for nbr in adj.get(u, ()):
                if isinstance(nbr, (tuple, list)) and len(nbr) >= 1:
                    v, w = (nbr[0], nbr[1] if len(nbr) > 1 else None)
                else:
                    v, w = (nbr, None)
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
            for w in self._nbr_targets(adj.get(v, ())):
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
        return (components, mapping, condensed)

    def _count_paths_with_cycles(self, adj):
        components, mapping, condensed = self._strongly_connected_components(adj)
        if not components:
            return {}
        try:
            order, parents = self._topo_order(condensed)
        except ValueError:
            return {}
        base = defaultdict(int)
        for s in self.inputs:
            if s in mapping:
                base[mapping[s]] += 1
        comp_paths = {}
        for cid in order:
            total = base.get(cid, 0)
            for parent in parents.get(cid, ()):
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
            for v in self._nbr_targets(adj.get(u, ())):
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
            for w in self._nbr_targets(adj.get(v, ())):
                indeg[w] -= 1
                if indeg[w] == 0:
                    queue.append(w)
        if len(order) != len(nodes):
            raise ValueError('Graph has a cycle; LCS expects a DAG or SCC-condensed DAG.')
        parents = {v: [] for v in nodes}
        for u, nbrs in adj.items():
            for v in self._nbr_targets(nbrs):
                parents[v].append(u)
        return (order, parents)

    def _reachable_from_inputs(self, adj):
        seen = set()
        queue = deque(self.inputs)
        for s in self.inputs:
            seen.add(s)
        while queue:
            u = queue.popleft()
            for v in self._nbr_targets(adj.get(u, ())):
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
            for u in parents.get(v, ()):
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
            for u in parents.get(v, ()):
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
            v = sink
            path_cap = limit - flow
            while parent[v] is not None:
                u = parent[v]
                path_cap = min(path_cap, graph[u][v])
                v = u
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
        source = '__source__'
        counts = {}
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
            detour = ''
            delta_sp = ''
            if R0 and R1 and (d0 < INF) and (d1 < INF) and (d0 > 0):
                detour = float(d1) / float(d0)
                delta_sp = d1 - d0
            elif d0 < INF and d1 < INF:
                delta_sp = d1 - d0
            dis0 = disjoint0.get(o, 0)
            dis1 = disjoint1.get(o, 0)
            rows.append({'gen': gen, 'lineage_id': lineage_id, 'mut_id': mut_id, 'o': o, 'changed_nodes': len(changed_nodes), 'changed_edges': changed_edges, 'R0': R0, 'R1': R1, 'P0': P0, 'P1': P1, 'd0': '' if d0 >= INF else int(d0), 'd1': '' if d1 >= INF else int(d1), 'detour': detour, 'delta_paths': P1 - P0, 'delta_sp': delta_sp if delta_sp != '' else '', 'heal_flag': 0, 'time_to_heal': '', 'disjoint_paths0': dis0, 'disjoint_paths1': dis1})
        return rows

    def _update_heal_flags(self, gen, rows):
        updated = []
        for row in rows:
            o = row['o']
            state = self._pair_state.setdefault(o, PairState())
            last_heal = self._last_heal_gen.get(o, -INF)
            R0 = bool(row['R0'])
            R1 = bool(row['R1'])
            if state.status == 'connected':
                if R0 and (not R1):
                    state.status = 'broken'
                    state.broke_at = gen
            elif R1:
                tth = gen - state.broke_at
                if tth <= self.T and gen - last_heal > self.cooldown:
                    row['heal_flag'] = 1
                    row['time_to_heal'] = tth
                    self._last_heal_gen[o] = gen
                state.status = 'connected'
                state.broke_at = -1
            updated.append(row)
        return updated

    def log_step(self, G_prev, G_post, changed_nodes, lineage_id, gen, mut_id, changed_edges=0, birth_mode=None, is_regen_child=None):
        rows = self._compute_metrics(G_prev, G_post, set(changed_nodes), changed_edges, lineage_id, gen, mut_id)
        rows = self._update_heal_flags(gen, rows)
        for _r in rows:
            _r['birth_mode'] = birth_mode or ''
            _r['is_regen_child'] = int(bool(is_regen_child)) if is_regen_child is not None else 0
        header = ['birth_mode', 'is_regen_child', 'gen', 'lineage_id', 'mut_id', 'o', 'changed_nodes', 'changed_edges', 'R0', 'R1', 'P0', 'P1', 'd0', 'd1', 'detour', 'delta_paths', 'delta_sp', 'heal_flag', 'time_to_heal', 'disjoint_paths0', 'disjoint_paths1', 'build_id']
        need_header = not os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if need_header:
                    writer.writeheader()
                for row in rows:
                    row['build_id'] = _build_stamp_short()
                    writer.writerow(row)
        except Exception as exc:
            print(f'[LCS] CSV write error: {exc}')
        return rows

class ReproPlanaNEATPlus:

    def __init__(self, num_inputs, num_outputs, population_size=150, rng=None, output_activation='sigmoid'):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.pop_size = population_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mode = EvalMode(vanilla=True, enable_regen_reproduction=False)
        self.max_hidden_nodes = 192
        self.max_edges = 2048
        self.complexity_threshold: Optional[float] = 5.0
        self.grad_clip = 5.0
        self.weight_clip = 12.0
        self.snapshot_stride = 1 if self.pop_size <= 256 else 2
        self.snapshot_max = 320
        self.min_conn_after_regen = 0.65
        self.diversity_push = 0.15
        self.max_attempts_guard = 16
        self.mutate_duplicate_node_prob = 0.08
        self.duplicate_branch_weight_scale = 0.85
        self.structure_diversity_bonus = 0.08
        self.structure_diversity_power = 1.2
        self.complexity_survivor_bonus = 0.12
        self.complexity_survivor_exponent = 1.1
        self.complexity_survivor_cap = 1.6
        self.complexity_survivor_bonus_limit = 1.2
        self.complexity_bonus_baseline_quantile = 0.6
        self.complexity_bonus_span_quantile = 0.9
        try:
            cpus = int(os.cpu_count() or 2)
            self.eval_workers = int(os.environ.get('NEAT_EVAL_WORKERS', max(1, cpus - 1)))
            self.parallel_backend = os.environ.get('NEAT_EVAL_BACKEND', 'thread')
        except Exception:
            self.eval_workers = 1
            self.parallel_backend = 'thread'
        self.auto_curriculum = True
        self.top3_diversity_node_threshold = 2
        self.top3_diversity_edge_threshold = 4
        self.top3_candidate_pool_size = 24
        self.complexity_bonus_multiplier = -0.1
        self.complexity_bonus_threshold = 2.5
        self.species_target = None
        self.species_target_min = 2.0
        self.species_target_max = max(float(self.mode.species_high), float(self.pop_size) / 3.0)
        self.species_target_step = 0.5
        self.species_target_update_every = 5
        self._spec_learn = {'dir': 1.0, 'last_best': None, 'last_tgt': None, 'last_reward': None}
        self.species_target_mode = 'auto'
        self.pid_kp = 0.35
        self.pid_ki = 0.02
        self.pid_kd = 0.1
        self.pid_i_clip = 50.0
        self._spec_learn.update({'pid_i': 0.0, 'pid_prev_err': None, 'score_pid': 0.0, 'score_hill': 0.0, 'eps': 0.1, 'last_method': 'pid'})
        self.pool_keepalive = int(os.environ.get('NEAT_POOL_KEEPALIVE', '0'))
        self.pool_restart_every = int(os.environ.get('NEAT_POOL_RESTART_EVERY', '25'))
        self._proc_pool = None
        self._proc_pool_age = 0
        self._shm_meta = None
        self._resilience_failures = deque(maxlen=16)
        self._resilience_eval_guard = 0
        self._resilience_history = []
        self.diversity_history: List[Dict[str, float]] = []
        self.complexity_distribution_history: List[Dict[str, Any]] = []
        self.diversity_history_limit = 4096
        self.complexity_history_limit = 4096
        self._diversity_snapshot: Dict[str, Any] = {}
        self.refine_topk_ratio = float(os.environ.get('NEAT_REFINE_TOPK_RATIO', '0.08'))
        nodes = {}
        for i in range(num_inputs):
            nodes[i] = NodeGene(i, 'input', 'identity')
        for j in range(num_outputs):
            nodes[num_inputs + j] = NodeGene(num_inputs + j, 'output', output_activation)
        bias_id = num_inputs + num_outputs
        nodes[bias_id] = NodeGene(bias_id, 'bias', 'identity')
        next_node_id = bias_id + 1
        self.innov = InnovationTracker(next_node_id=next_node_id, next_conn_innov=0)
        base_connections = {}
        for in_id in list(range(num_inputs)) + [bias_id]:
            for out_id in range(num_inputs, num_inputs + num_outputs):
                inn = self.innov.get_conn_innovation(in_id, out_id)
                w = float(self.rng.uniform(-2, 2))
                base_connections[inn] = ConnectionGene(in_id, out_id, w, True, inn)
        base_genome = Genome(nodes, base_connections)
        base_genome.max_hidden_nodes = self.max_hidden_nodes
        base_genome.max_edges = self.max_edges
        self.population = []
        self.next_gid = 1
        self._family_seq = 1
        self._family_lineage_cache: 'OrderedDict[Tuple[int, ...], int]' = OrderedDict()
        self.family_lineage_cache_limit = max(4096, population_size * 8)
        self._family_parent_map: Dict[int, Tuple[int, ...]] = {}
        for _ in range(population_size):
            g = base_genome.copy()
            g.max_hidden_nodes = self.max_hidden_nodes
            g.max_edges = self.max_edges
            r0 = self.rng.random()
            if r0 < float(getattr(self, 'hermaphrodite_init_ratio', 0.0)):
                g.sex = 'hermaphrodite'
            else:
                g.sex = 'female' if self.rng.random() < 0.5 else 'male'
            g.regen = bool(self.rng.random() < 0.5)
            g.regen_mode = self.rng.choice(['head', 'tail', 'split'])
            g.embryo_bias = 'inputward'
            g.id = self.next_gid
            self.next_gid += 1
            g.birth_gen = 0
            g.mutation_will = float(np.clip(self.rng.uniform(0.0, 1.0), 0.0, 1.0))
            g.cooperative = True
            g.family_id = self._next_family_id()
            setattr(g, 'lazy_lineage_strength', 0.0)
            self.population.append(g)
        input_ids = list(range(num_inputs))
        output_ids = list(range(num_inputs, num_inputs + num_outputs))
        self.lcs_monitor = LCSMonitor(inputs=input_ids, outputs=output_ids)
        self.generation = 0
        self.compatibility_threshold = 3.0
        self.c1 = self.c2 = 1.0
        self.c3 = 0.4
        self.elitism = 1
        self.survival_rate = 0.2
        self.mutate_add_conn_prob = 0.12
        self.mutate_add_node_prob = 0.12
        self.mutate_weight_prob = 0.8
        self.mutate_toggle_prob = 0.01
        self.weight_perturb_chance = 0.9
        self.weight_sigma = 0.8
        self.weight_reset_range = 2.0
        self.regen_mode_mut_rate = 0.05
        self.embryo_bias_mut_rate = 0.03
        self.mutate_sex_prob = 0.005
        self.hermaphrodite_inheritance_rate = 0.05
        self.regen_rate = 0.15
        self.allow_selfing = True
        self.sex_fitness_scale = {'female': 1.0, 'male': 0.9, 'hermaphrodite': 1.2}
        self.regen_bonus = 0.2
        self.regen_mut_rate_boost = 1.8
        self.non_elite_mating_rate = 0.5
        self.lcs_reward_weight = 0.02
        self.diversity_weight = 0.02
        self.hermaphrodite_mate_bias = 2.5
        self.hermaphrodite_similarity_threshold = 0.08
        self.hemi_topology_inheritance_bias = 0.9
        self.mutation_will_child_noise = 0.05
        self.mutation_will_similarity_noise = 0.02
        self.mutation_will_mutation_rate = 0.1
        self.mutation_will_mutation_scale = 0.05
        self.lazy_fraction = 0.02
        self.lazy_fraction_max = 0.021
        self.lazy_individual_fitness = -1.0
        self.auto_complexity_controls = True
        self.auto_complexity_bonus_fraction = 0.18
        self.auto_complexity_smoothing = 0.35
        self.lazy_complexity_growth_steps = 4
        self.lazy_complexity_target_scale = 1.18
        self.lazy_complexity_duplicate_bias = 0.45
        self.lazy_complexity_min_growth = 0.0
        self.lazy_lineage_decay = 0.82
        self.lazy_lineage_strength_cap = 3.0
        self.lazy_lineage_inheritance_gain = 0.32
        self.lazy_lineage_inheritance_decay = 0.24
        self.lazy_lineage_persistence = 0.7
        self.env = {
            'difficulty': 0.0,
            'noise_std': 0.0,
            'noise_kind': 'white',
            'noise_profile': {
                'cycle_phase': 0.0,
                'envelope': 0.0,
                'jitter': 0.0,
                'spectral_bias': 0.0,
                'band_label': 'white',
            },
        }
        self.mix_asexual_base = 0.1
        self.mix_asexual_gain = 0.4
        self.injury_intensity_base = 0.25
        self.injury_intensity_gain = 0.65
        self.pollen_flow_rate = 0.1
        self.heterosis_center = 3.0
        self.heterosis_width = 1.8
        self.heterosis_gain = 0.15
        self.distance_cutoff = 6.0
        self.penalty_far = 0.2
        self.event_log = []
        self.hidden_counts_history = []
        self.edge_counts_history = []
        self.best_ids = []
        self.lineage_edges = []
        self.lineage_annotations: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        self.env_history = []
        self.snapshots_genomes: List[Genome] = []
        self.snapshots_scars: List[Dict[int, 'Scar']] = []
        self.node_registry: Dict[int, Dict[str, Any]] = {}
        self.top3_best_topologies = []
        self._compat_cache: 'OrderedDict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], float]' = OrderedDict()
        self._compat_cache_limit = max(1024, population_size * 8)
        try:
            struct_limit_env = int(os.environ.get('SMNN_STRUCTURE_CACHE_LIMIT', '0'))
        except Exception:
            struct_limit_env = 0
        self.structure_cache_limit = max(256, struct_limit_env or population_size * 6)
        self._households = HouseholdManager(max_households=max(population_size * 4, 256))
        self._last_household_adjustments: Dict[int, float] = {}
        for g in self.population:
            self.node_registry[g.id] = {'sex': g.sex, 'regen': g.regen, 'birth_gen': g.birth_gen, 'family_id': g.family_id}
        self._assign_lazy_individuals(self.population)
        self.monodromy_top_ratio = 0.12
        self.monodromy_span = 6.0
        self.monodromy_pressure_base = 0.018
        self.monodromy_pressure_range = 0.06
        self.monodromy_penalty_cap = 0.42
        self.monodromy_phase_step = 0.38196601125
        self.monodromy_smoothing = 0.4
        self.monodromy_decay = 0.55
        self.monodromy_release = 0.35
        self.monodromy_momentum_decay = 0.65
        self.monodromy_growth_weight = 0.85
        self.monodromy_slump_gain = 0.6
        self.monodromy_fast_release = 0.25
        self.monodromy_diversity_weight = 0.45
        self.monodromy_diversity_floor = 0.22
        self.monodromy_diversity_grace = 2.5
        self.monodromy_diversity_grace_decay = 0.6
        self.monodromy_diversity_grace_strength = 0.3
        self.monodromy_noise_weight = 0.25
        self.monodromy_family_weight = 0.35
        self.monodromy_noise_style_overrides: Dict[str, Dict[str, Any]] = {}
        self._monodromy_registry: Dict[int, Dict[str, float]] = {}
        self._monodromy_snapshot: Dict[str, float] = {
            'pressure_mean': 0.0,
            'pressure_max': 0.0,
            'active': 0,
            'families': 0,
            'release': 0,
            'relief_mean': 0.0,
            'momentum_mean': 0.0,
            'momentum_max': 0.0,
            'diversity_mean': 0.0,
            'diversity_max': 0.0,
            'grace_mean': 0.0,
            'noise_factor': 1.0,
            'noise_kind': '',
            'noise_focus': 0.0,
            'noise_entropy': 0.0,
            'family_factor_mean': 1.0,
            'family_factor_max': 1.0,
        }
        self._monodromy_noise_tag = ''
        self._auto_complexity_bonus_state = {
            'baseline': 0.0,
            'bonus_scale': float(self.complexity_survivor_bonus),
            'exponent': float(self.complexity_survivor_exponent),
            'cap': float(self.complexity_survivor_cap),
            'baseline_quantile': float(self.complexity_bonus_baseline_quantile),
            'span_quantile': float(self.complexity_bonus_span_quantile),
            'penalty_multiplier': 1.0,
            'penalty_threshold': float(self.complexity_threshold or 0.0) if self.complexity_threshold is not None else None,
            'bonus_multiplier': float(getattr(self, 'complexity_bonus_multiplier', -0.1)),
            'bonus_threshold': float(getattr(self, 'complexity_bonus_threshold', 2.5)),
            'bonus_limit': float(self.complexity_survivor_bonus_limit),
            'lazy_target': 0.0,
            'spread': 0.0,
            'span_value': 0.0,
        }
        self._lazy_fraction_carry = 0.0
        self.raw_best_history: List[Tuple[float, float]] = []
        self.context_best_history: List[Tuple[float, float]] = []
        self._lazy_env_feedback: Dict[str, Any] = {'generation': -1, 'share': 0.0, 'anchor': 0.0, 'gap': 0.0, 'stasis': 0.0}
        self.spinor_controller: Optional[Any] = None
        self._best_context_value = -1000000000.0
        self.monodromy_family_limit = max(self.pop_size * 6, 1536)
        self._collective_signal: Dict[str, float] = {
            'altruism_target': 0.5,
            'solidarity': 0.5,
            'stress': 0.0,
            'lazy_share': 0.0,
            'advantage': 0.0,
        }
        self.selfish_leader_threshold = 0.24
        self.compile_cache_prune_stride = 12
        self.compile_cache_recent_keep = 384
        self.compile_cache_top_keep = max(6, int(self.pop_size * 0.1))
        self.compile_cache_topo_stride = 48
        self.compile_cache_decay_start = max(120, int(self.pop_size * 0.6))
        self.compile_cache_decay_tau = 260.0

    def _heterosis_scale(self, mother: Genome, father: Genome) -> float:
        d = self._compat_distance(mother, father)
        peak = 1.0 + self.heterosis_gain * np.exp(-0.5 * ((d - self.heterosis_center) / self.heterosis_width) ** 2)
        if d > self.distance_cutoff:
            penalty = max(0.0, self.penalty_far * (d - self.distance_cutoff) / self.distance_cutoff)
            peak *= 1.0 - min(0.9, penalty)
        return float(peak)

    def _note_lineage(self, gid: int, gen: int, summary: str) -> None:
        if not summary:
            return
        notes = self.lineage_annotations.setdefault(gid, [])
        notes.append((gen, summary))
        if len(notes) > 6:
            self.lineage_annotations[gid] = notes[-6:]

    def _next_family_id(self) -> int:
        seq = int(getattr(self, '_family_seq', 1))
        self._family_seq = seq + 1
        return seq

    def _family_key_from_parents(self, parents: Sequence[Optional['Genome']]) -> Tuple[int, ...]:
        ids: Set[int] = set()
        for parent in parents:
            if parent is None:
                continue
            fid = getattr(parent, 'family_id', None)
            if fid is None:
                continue
            try:
                ids.add(int(fid))
            except Exception:
                continue
        if not ids:
            return tuple()
        return tuple(sorted(ids))

    def _resolve_family_from_key(self, key: Tuple[int, ...]) -> int:
        if not key:
            return self._next_family_id()
        if len(key) == 1:
            return int(key[0])
        cache = getattr(self, '_family_lineage_cache', None)
        if cache is None:
            cache = OrderedDict()
            self._family_lineage_cache = cache
        fid = cache.get(key)
        if fid is None:
            fid = self._next_family_id()
            cache[key] = fid
        else:
            cache.move_to_end(key)
        limit = int(getattr(self, 'family_lineage_cache_limit', max(4096, self.pop_size * 8)))
        while len(cache) > limit:
            cache.popitem(last=False)
        parent_map = getattr(self, '_family_parent_map', None)
        if isinstance(parent_map, dict):
            parent_map[fid] = key
            valid_ids = set(cache.values())
            if len(parent_map) > max(limit, len(valid_ids)):
                for stale in [k for k in list(parent_map.keys()) if k not in valid_ids]:
                    parent_map.pop(stale, None)
        return int(fid)

    def _assign_child_family(self, child: 'Genome', parents: Sequence[Optional['Genome']]) -> int:
        key = self._family_key_from_parents(parents)
        fid = getattr(child, 'family_id', None)
        if key:
            fid = self._resolve_family_from_key(key)
        elif fid is None:
            fid = self._next_family_id()
        child.family_id = int(fid)
        return int(fid)

    def _update_lazy_feedback(self, generation: int, fitnesses: Sequence[float], best_idx: int, best_fit: float, avg_fit: float) -> None:
        total = len(self.population)
        if total <= 0:
            return
        lazy_indices = [i for i, g in enumerate(self.population) if getattr(g, 'lazy_lineage', False)]
        share = float(len(lazy_indices)) / float(total)
        lazy_scores = [float(fitnesses[i]) for i in lazy_indices] if lazy_indices else []
        if lazy_scores:
            lazy_avg = float(np.mean(lazy_scores))
            lazy_best = float(np.max(lazy_scores))
            spread = float(np.std(lazy_scores))
        else:
            lazy_avg = float(avg_fit)
            lazy_best = float(best_fit)
            spread = 0.0
        scale = max(1.0, abs(best_fit) + abs(avg_fit) + abs(lazy_avg))
        anchor = float(np.clip(lazy_avg / scale, -1.0, 1.0))
        gap = float(np.clip((best_fit - lazy_avg) / scale, -1.0, 1.0))
        stasis = float(np.clip(1.0 - min(1.0, spread / scale), 0.0, 1.0))
        payload = {
            'generation': int(generation),
            'share': float(np.clip(share, 0.0, 1.0)),
            'anchor': anchor,
            'gap': gap,
            'stasis': stasis,
            'lazy_avg': float(lazy_avg),
            'lazy_best': float(lazy_best),
            'spread': float(spread),
            'raw_best': float(best_fit),
            'raw_avg': float(avg_fit),
            'count': int(len(lazy_indices)),
            'population': int(total),
        }
        self._lazy_env_feedback = payload
        controller = getattr(self, 'spinor_controller', None)
        if controller is not None and hasattr(controller, 'set_lazy_feedback'):
            try:
                controller.set_lazy_feedback(generation, payload)
            except Exception:
                pass

    def _contextual_best_axis(self, best_fit: float, avg_fit: float) -> float:
        env = getattr(self, 'env', {})
        if isinstance(env, dict):
            diff = float(env.get('difficulty', 0.0))
            noise = float(env.get('noise_std', 0.0))
        else:
            diff = 0.0
            noise = 0.0
        lazy = getattr(self, '_lazy_env_feedback', {}) or {}
        share = float(np.clip(lazy.get('share', 0.0), 0.0, 1.0))
        stasis = float(np.clip(lazy.get('stasis', 0.0), 0.0, 1.0))
        gap = float(np.clip(lazy.get('gap', 0.0), -1.0, 1.0))
        anchor = float(np.clip(lazy.get('anchor', 0.0), -1.0, 1.0))
        momentum = float(np.clip(lazy.get('spread', 0.0), 0.0, 5.0))
        difficulty_penalty = 0.07 * diff + 0.04 * noise
        cohesion_bonus = 0.05 * share * stasis
        anchor_bonus = 0.02 * anchor
        gap_penalty = 0.03 * abs(gap)
        variance_penalty = 0.01 * momentum
        return float(best_fit - difficulty_penalty - gap_penalty - variance_penalty + cohesion_bonus + anchor_bonus)

    def _trim_runtime_caches(self, generation: int) -> None:
        stride = int(getattr(self, 'compile_cache_prune_stride', 0))
        if stride <= 0 or generation % stride != 0:
            return
        keep_top = int(getattr(self, 'compile_cache_top_keep', 0))
        recent_keep = int(getattr(self, 'compile_cache_recent_keep', 0))
        decay_start = int(getattr(self, 'compile_cache_decay_start', 0))
        recent_window = recent_keep
        if recent_keep > 0 and decay_start > 0 and generation >= decay_start:
            tau = max(1.0, float(getattr(self, 'compile_cache_decay_tau', 240.0)))
            factor = math.exp(-(generation - decay_start) / tau)
            recent_window = max(4, int(round(recent_keep * factor)))
        try:
            current_tick = _COMPILE_TICK
        except Exception:
            current_tick = 0
        threshold = current_tick - recent_window if recent_window > 0 else None
        struct_limit = max(64, int(getattr(self, 'structure_cache_limit', _STRUCTURE_CACHE_LIMIT)))
        if threshold is not None:
            _structure_cache_trim(max_entries=struct_limit, min_tick=threshold)
        else:
            _structure_cache_trim(max_entries=struct_limit)
        topo_stride = int(getattr(self, 'compile_cache_topo_stride', 0))
        drop_topology = topo_stride > 0 and generation % topo_stride == 0
        for idx, g in enumerate(self.population):
            if keep_top and idx < keep_top:
                continue
            tick = int(getattr(g, '_compiled_cache_tick', -1))
            if threshold is not None and tick >= threshold:
                continue
            g.trim_runtime_caches(compiled=True, compat=True, topo=drop_topology)

    def _apply_selfish_leader_guard(
        self,
        fitnesses: List[float],
        baseline: List[float],
        generation: int,
    ) -> None:
        controller = getattr(self, 'spinor_controller', None)
        evaluator = getattr(controller, 'evaluator', None) if controller is not None else None
        if evaluator is None:
            return
        council_ids = getattr(evaluator, 'last_leader_council', None)
        if council_ids:
            try:
                ids_list = [int(x) for x in council_ids if x is not None]
            except Exception:
                ids_list = [x for x in council_ids if isinstance(x, int)]
        else:
            ids_list = []
        leader_id = getattr(evaluator, 'last_leader_id', None)
        if leader_id is not None:
            ids_list.append(int(leader_id))
        ids_unique: List[int] = list(dict.fromkeys(ids_list))
        advantage = float(getattr(evaluator, 'last_advantage_score', 0.0) or 0.0)
        threshold = float(getattr(self, 'selfish_leader_threshold', 0.22))
        if not ids_unique or advantage <= threshold:
            return
        penalty_floor = min(baseline) if baseline else -1.0
        applied = False
        for lid in ids_unique:
            try:
                idx = next(i for i, g in enumerate(self.population) if g.id == lid)
            except StopIteration:
                continue
            drop = (abs(baseline[idx]) + abs(penalty_floor) + 1.0) * (1.5 + 2.5 * advantage)
            drop /= max(1, len(ids_unique))
            killer = penalty_floor - drop
            baseline[idx] = killer
            fitnesses[idx] = killer
            try:
                self.population[idx].cooperative = False
                setattr(self.population[idx], 'selfish_culled', True)
            except Exception:
                pass
            note = 'selfish leader culled' if len(ids_unique) == 1 else 'selfish council culled'
            self._note_lineage(self.population[idx].id, generation, f'{note} ({advantage:.3f})')
            applied = True
        if applied:
            setattr(evaluator, 'last_advantage_penalty', True)

    def _update_collective_signal(
        self,
        diversity_entropy: float,
        diversity_scarcity: float,
        family_surplus_mean: float,
        generation: int,
    ) -> None:
        controller = getattr(self, 'spinor_controller', None)
        evaluator = getattr(controller, 'evaluator', None) if controller is not None else None
        lazy = getattr(self, '_lazy_env_feedback', {}) or {}
        lazy_share = float(np.clip(lazy.get('share', 0.0), 0.0, 1.0))
        advantage = float(getattr(evaluator, 'last_advantage_score', 0.0) or 0.0)
        altruism_hint = float(np.clip(getattr(evaluator, 'last_altruism_signal', 0.5) or 0.5, 0.0, 1.0))
        solidarity = float(np.clip(diversity_entropy, 0.0, 1.0))
        stress = float(np.clip(diversity_scarcity + max(0.0, family_surplus_mean), 0.0, 2.5))
        target = float(np.clip(0.6 * altruism_hint + 0.4 * (1.0 - advantage), 0.0, 1.0))
        self._collective_signal = {
            'altruism_target': target,
            'solidarity': solidarity,
            'stress': stress,
            'lazy_share': lazy_share,
            'advantage': advantage,
        }
        if controller is not None:
            try:
                controller.altruism_signal = dict(self._collective_signal)
            except Exception:
                pass

    def _imprint_population_altruism(self, fitnesses: Sequence[float]) -> None:
        if not fitnesses:
            return
        signal = getattr(self, '_collective_signal', {}) or {}
        target = float(signal.get('altruism_target', 0.5))
        solidarity = float(signal.get('solidarity', 0.5))
        stress = float(signal.get('stress', 0.0))
        lazy_share = float(signal.get('lazy_share', 0.0))
        advantage = float(signal.get('advantage', 0.0))
        smoothing = float(np.clip(getattr(self, 'altruism_imprint_smoothing', 0.72), 0.0, 0.99))
        depth_ratio = float(np.clip(getattr(self, 'altruism_imprint_ratio', 0.18), 0.05, 1.0))
        depth = max(1, int(len(self.population) * depth_ratio))
        order = np.argsort(fitnesses)[::-1][:depth]
        span_target = float(np.clip(stress + advantage, 0.0, 4.0))
        mem_target = float(np.clip(solidarity - stress, -1.5, 1.5))
        gain = float(np.clip(target + 0.1 * solidarity - 0.15 * advantage + 0.05 * lazy_share, 0.0, 1.0))
        for idx in map(int, order):
            g = self.population[idx]
            for node in g.nodes.values():
                if node.type == 'input':
                    continue
                prev_alt = float(np.clip(getattr(node, 'altruism', 0.5), 0.0, 1.0))
                prev_mem = float(np.clip(getattr(node, 'altruism_memory', 0.0), -1.5, 1.5))
                prev_span = float(np.clip(getattr(node, 'altruism_span', 0.0), 0.0, 4.0))
                node.altruism = float(np.clip(prev_alt * smoothing + gain * (1.0 - smoothing), 0.0, 1.0))
                node.altruism_memory = float(np.clip(0.6 * prev_mem + 0.4 * mem_target, -1.5, 1.5))
                node.altruism_span = float(np.clip(0.65 * prev_span + 0.35 * span_target, 0.0, 4.0))

    def _compat_distance(self, g1: Genome, g2: Genome) -> float:
        try:
            token1 = g1.compat_token()
            token2 = g2.compat_token()
            key = (token1, token2) if token1 <= token2 else (token2, token1)
        except Exception:
            return compatibility_distance(g1, g2, self.c1, self.c2, self.c3)
        cache = self._compat_cache
        dist = cache.get(key)
        if dist is not None:
            cache.move_to_end(key)
            return dist
        dist = compatibility_distance(g1, g2, self.c1, self.c2, self.c3)
        cache[key] = dist
        if len(cache) > self._compat_cache_limit:
            try:
                cache.popitem(last=False)
            except Exception:
                pass
        return dist

    def _regen_intensity(self) -> float:
        return float(min(1.0, max(0.0, self.injury_intensity_base + self.injury_intensity_gain * self.env['difficulty'])))

    def _mix_asexual_ratio(self) -> float:
        return float(min(0.95, max(0.0, self.mix_asexual_base + self.mix_asexual_gain * self.env['difficulty'])))

    class Species:

        def __init__(self, representative: 'Genome'):
            self.representative = representative.copy()
            self.members = []
            self.best_fitness = -1000000000.0
            self.last_improved = 0

        def add(self, genome: 'Genome', fitness: float):
            self.members.append((genome, fitness))

        def sort(self):
            self.members.sort(key=lambda gf: gf[1], reverse=True)

    def speciate(self, fitnesses: List[float]) -> List['Species']:
        species = []
        for genome, fit in zip(self.population, fitnesses):
            placed = False
            for sp in species:
                delta = self._compat_distance(genome, sp.representative)
                if delta < self.compatibility_threshold:
                    sp.add(genome, fit)
                    placed = True
                    break
            if not placed:
                sp = ReproPlanaNEATPlus.Species(genome)
                sp.add(genome, fit)
                species.append(sp)
        for sp in species:
            sp.sort()
        return species

    def _adapt_compat_threshold(self, num_species: int):
        low = int(self.mode.species_low)
        high = int(self.mode.species_high)
        if getattr(self, 'species_target', None) is None:
            self.species_target = float((low + high) * 0.5)
        target = float(self.species_target)
        if target <= 0:
            target = (low + high) * 0.5
        err = (float(num_species) - target) / max(1.0, target)
        self.compatibility_threshold *= float(np.exp(0.18 * err))
        self.compatibility_threshold = float(np.clip(self.compatibility_threshold, 0.3, 50.0))

    def _learn_species_target(self, num_species: int, best_fit: float, gen: int) -> None:
        """species_target の"学習"：PID と Hill-Climb をバンディット切換（auto）。"""
        low, high = (int(self.mode.species_low), int(self.mode.species_high))
        if self.species_target is None:
            self.species_target = float((low + high) * 0.5)
            self._spec_learn['last_best'] = float(best_fit)
            self._spec_learn['last_tgt'] = float(self.species_target)
            self._spec_learn['last_reward'] = 0.0
            return
        if gen % int(self.species_target_update_every) != 0:
            return
        st = self._spec_learn
        last_best = st.get('last_best', None)
        if last_best is None:
            st['last_best'] = float(best_fit)
            return
        reward = float(best_fit) - float(last_best)
        mode = getattr(self, 'species_target_mode', 'auto')
        method = 'pid'
        if mode == 'hill':
            method = 'hill'
        elif mode == 'auto':
            eps = float(st.get('eps', 0.1))
            if self.rng.random() < eps:
                method = 'pid' if self.rng.random() < 0.5 else 'hill'
            else:
                method = 'pid' if st.get('score_pid', 0.0) >= st.get('score_hill', 0.0) else 'hill'
        if method == 'pid':
            err = float(num_species) - float(self.species_target)
            prev = st.get('pid_prev_err', 0.0) or 0.0
            itg = float(st.get('pid_i', 0.0)) + err
            itg = float(np.clip(itg, -float(self.pid_i_clip), float(self.pid_i_clip)))
            de = err - prev
            delta = float(self.pid_kp) * err + float(self.pid_ki) * itg + float(self.pid_kd) * de
            step_max = max(0.5, 0.75)
            new_t = float(self.species_target) + float(np.clip(delta, -step_max, step_max))
            new_t = float(np.clip(new_t, float(self.species_target_min), float(self.species_target_max)))
            self.species_target = new_t
            st['pid_prev_err'] = err
            st['pid_i'] = itg
            st['score_pid'] = 0.85 * float(st.get('score_pid', 0.0)) + 0.15 * reward
            st['score_hill'] = 0.98 * float(st.get('score_hill', 0.0))
        else:
            dir_ = float(st.get('dir', 1.0))
            last_reward = float(st.get('last_reward') or 0.0)
            if reward < last_reward - 1e-06:
                dir_ = -dir_
            err_s = float(num_species) - float(self.species_target)
            if err_s != 0.0:
                dir_ = 0.7 * dir_ + 0.3 * np.sign(err_s)
            step = float(self.species_target_step)
            new_t = float(self.species_target) + step * dir_
            new_t = float(np.clip(new_t, float(self.species_target_min), float(self.species_target_max)))
            self.species_target = new_t
            st['dir'] = dir_
            st['score_hill'] = 0.85 * float(st.get('score_hill', 0.0)) + 0.15 * reward
            st['score_pid'] = 0.98 * float(st.get('score_pid', 0.0))
        st['last_best'] = float(best_fit)
        st['last_tgt'] = float(self.species_target)
        st['last_reward'] = float(reward)

    def _mutate(self, genome: Genome, context: str=None):
        boost = 1.0
        if context == 'regen':
            boost = float(getattr(self, 'regen_mut_rate_boost', 1.5))
        add_conn_prob = min(1.0, float(self.mutate_add_conn_prob) * boost)
        add_node_prob = min(1.0, float(self.mutate_add_node_prob) * boost)
        toggle_prob = float(getattr(self, 'mutate_toggle_prob', 0.05))
        weight_prob = float(getattr(self, 'mutate_weight_prob', 0.8))
        if self.rng.random() < toggle_prob:
            genome.mutate_toggle_enable(self.rng, prob=toggle_prob)
        if self.rng.random() < add_node_prob:
            genome.mutate_add_node(self.rng, self.innov)
        if self.rng.random() < add_conn_prob:
            genome.mutate_add_connection(self.rng, self.innov)
        dup_prob = min(1.0, float(getattr(self, 'mutate_duplicate_node_prob', 0.0)) * boost)
        if dup_prob > 0.0 and self.rng.random() < dup_prob:
            try:
                scale = float(getattr(self, 'duplicate_branch_weight_scale', 0.85))
            except Exception:
                scale = 0.85
            genome.mutate_duplicate_node(self.rng, self.innov, weight_scale=scale)
        if self.rng.random() < weight_prob:
            genome.mutate_weights(self.rng)

        # Occasional sex mutation to increase hermaphrodite emergence
        try:
            p_sex = float(getattr(self, 'mutate_sex_prob', 0.0))
        except Exception:
            p_sex = 0.0
        if p_sex > 0.0 and self.rng.random() < p_sex:
            try:
                genome.mutate_sex(self.rng)
            except Exception:
                pass
        if self.rng.random() < float(getattr(self, 'embryo_bias_mut_rate', 0.03)):
            genome.embryo_bias = self.rng.choice(['neutral', 'inputward', 'outputward'])
        if float(self.env.get('difficulty', 0.0)) >= 0.9 and self.rng.random() < getattr(self, 'diversity_push', 0.15):
            if self.rng.random() < 0.6:
                genome.mutate_add_connection(self.rng, self.innov)
            else:
                genome.mutate_add_node(self.rng, self.innov)
        will_rate = float(getattr(self, 'mutation_will_mutation_rate', 0.0))
        if will_rate > 0.0 and self.rng.random() < will_rate:
            scale = float(getattr(self, 'mutation_will_mutation_scale', 0.05))
            delta = float(self.rng.normal(0.0, scale))
            genome.mutation_will = float(np.clip(float(getattr(genome, 'mutation_will', 0.5)) + delta, 0.0, 1.0))

    def _crossover_maternal_biased(self, mother: Genome, father: Genome, species_members):
        fit_dict = {g: f for g, f in species_members}
        f_m = fit_dict.get(mother, 0.0)
        f_f = fit_dict.get(father, 0.0)
        if f_f > f_m:
            mother, father = (father, mother)
        will_m = float(getattr(mother, 'mutation_will', 0.5))
        will_f = float(getattr(father, 'mutation_will', 0.5))
        will_diff = abs(will_m - will_f)
        similarity_threshold = float(getattr(self, 'hermaphrodite_similarity_threshold', 0.08))
        similar_will = will_diff <= similarity_threshold
        topology_bias = 0.7
        preferred_parent = mother if will_m >= will_f else father
        if similar_will:
            topology_bias = float(getattr(self, 'hemi_topology_inheritance_bias', 0.9))
        child_nodes = {}
        child_conns = {}
        for nid, n in mother.nodes.items():
            if n.type in ('input', 'output', 'bias'):
                child_nodes[nid] = _clone_node(n, nid, n.type, n.activation)
        all_innovs = sorted(set(mother.connections.keys()).union(father.connections.keys()))
        for inn in all_innovs:
            if inn in mother.connections and inn in father.connections:
                cm = mother.connections[inn]
                cf = father.connections[inn]
                if similar_will:
                    if preferred_parent is mother:
                        pick = cm if self.rng.random() < topology_bias else cf
                    else:
                        pick = cf if self.rng.random() < topology_bias else cm
                else:
                    pick = cm if self.rng.random() < 0.7 else cf
                enabled = True
                if not cm.enabled or not cf.enabled:
                    enabled = not self.rng.random() < 0.75
                child_conns[inn] = ConnectionGene(pick.in_node, pick.out_node, pick.weight, enabled, inn)
            elif inn in mother.connections:
                g = mother.connections[inn]
                child_conns[inn] = ConnectionGene(g.in_node, g.out_node, g.weight, g.enabled, inn)
            if inn in child_conns:
                g = child_conns[inn]
                for nid in (g.in_node, g.out_node):
                    if nid not in child_nodes:
                        n = mother.nodes.get(nid) or father.nodes.get(nid)
                        if n is not None:
                            child_nodes[nid] = _clone_node(n, nid, n.type, n.activation)
        child = Genome(child_nodes, child_conns)
        child.max_hidden_nodes = self.max_hidden_nodes
        child.max_edges = self.max_edges
        child.remove_cycles()
        if similar_will:
            anchor_will = float(getattr(preferred_parent, 'mutation_will', 0.5))
            noise = float(self.rng.normal(0.0, getattr(self, 'mutation_will_similarity_noise', 0.02)))
            child.mutation_will = float(np.clip(anchor_will + noise, 0.0, 1.0))
        else:
            avg_will = 0.5 * (will_m + will_f)
            noise = float(self.rng.normal(0.0, getattr(self, 'mutation_will_child_noise', 0.05)))
            child.mutation_will = float(np.clip(avg_will + noise, 0.0, 1.0))
        inherits_hemi = False
        if mother.sex == 'hermaphrodite' or father.sex == 'hermaphrodite':
            if self.rng.random() < self.hermaphrodite_inheritance_rate:
                inherits_hemi = True
        if inherits_hemi or similar_will:
            child.sex = 'hermaphrodite'
        else:
            child.sex = 'female' if self.rng.random() < 0.5 else 'male'
        p = 0.7 if mother.regen or father.regen else 0.2
        child.regen = bool(self.rng.random() < p)
        child.regen_mode = self.rng.choice(['head', 'tail', 'split'])
        child.embryo_bias = mother.embryo_bias if self.rng.random() < 0.7 else father.embryo_bias
        return child

    def _make_offspring(self, species, offspring_counts, sidx, species_pool):
        sp = species[sidx]

        # Build fitness lookup for adaptive mutation
        fit_map = {g.id: f for (g, f) in sp.members}
        species_avg = (sum((f for _g, f in sp.members)) / max(1, len(sp.members))) if sp.members else 0.0
        new_pop = []
        events = {'sexual_within': 0, 'sexual_cross': 0, 'asexual_regen': 0, 'asexual_clone': 0}
        sp.sort()
        elites = [g for g, _ in sp.members[:min(self.elitism, offspring_counts[sidx])]]
        for e in elites:
            child = e.copy()
            child.cooperative = True
            child.id = self.next_gid
            self.next_gid += 1
            child.parents = (e.id, e.id)
            child.birth_gen = self.generation + 1
            self._assign_child_family(child, (e,))
            parent_strength = float(getattr(e, 'lazy_lineage_strength', 0.0) or 0.0)
            inheritance_decay = float(np.clip(getattr(self, 'lazy_lineage_inheritance_decay', 0.24), 0.0, 0.95))
            inheritance_gain = float(max(0.0, getattr(self, 'lazy_lineage_inheritance_gain', 0.0)))
            strength_cap = float(max(0.0, getattr(self, 'lazy_lineage_strength_cap', 3.0)))
            gain = inheritance_gain * (1.0 if bool(getattr(e, 'lazy_lineage', False)) else 0.5)
            new_strength = float(np.clip(parent_strength * (1.0 - inheritance_decay) + gain, 0.0, strength_cap))
            setattr(child, 'lazy_lineage_strength', new_strength)
            setattr(child, 'lazy_lineage', False)
            new_pop.append(child)
            events['asexual_clone'] += 1
            self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen, 'family_id': child.family_id}
        remaining = offspring_counts[sidx] - len(elites)
        k = max(2, int(math.ceil(self.survival_rate * len(sp.members))))
        females = [g for g, _ in sp.members[:k] if g.sex == 'female']
        males = [g for g, _ in sp.members[:k] if g.sex == 'male']
        hermaphrodites = [g for g, _ in sp.members[:k] if g.sex == 'hermaphrodite']
        pool = [g for g, _ in sp.members[:k]]
        non_elite_ids = set(getattr(self, '_last_top3_ids', set()))
        if not females or not males:
            females = [g for g, _ in sp.members if g.sex == 'female'] or females
            males = [g for g, _ in sp.members if g.sex == 'male'] or males
            hermaphrodites = [g for g, _ in sp.members if g.sex == 'hermaphrodite'] or hermaphrodites
        mix_ratio = self._mix_asexual_ratio()
        monitor = getattr(self, 'lcs_monitor', None)
        weight_tol = getattr(monitor, 'eps', 0.0) if monitor is not None else 0.0
        while remaining > 0:
            mode = None
            mother_id = None
            father_id = None
            parent_adj_before_regen = None
            use_sexual_reproduction = False
            parent = None
            mother = None
            father = None
            parent_candidate = pool[int(self.rng.integers(len(pool)))]
            effective_mix_ratio = mix_ratio
            if bool(getattr(self, 'adaptive_self_mutation', True)):
                f_par = float(fit_map.get(parent_candidate.id, species_avg))
                denom = (abs(species_avg) + 1e-9)
                rel = (species_avg - f_par) / denom
                try:
                    n_hidden = sum((1 for n in parent_candidate.nodes.values() if n.type == 'hidden'))
                    n_edges = sum((1 for c in parent_candidate.connections.values() if c.enabled))
                    comp = 0.5 * (n_hidden / max(1, self.max_hidden_nodes)) + 0.5 * (n_edges / max(1, self.max_edges))
                except Exception:
                    comp = 0.0
                gain = float(getattr(self, 'self_mutation_gain', 0.5))
                pen = float(getattr(self, 'self_mutation_complexity_penalty', 0.25))
                lim = float(getattr(self, 'self_mutation_limit', 0.5))
                delta = gain * rel - pen * comp
                if delta > lim:
                    delta = lim
                if delta < -lim:
                    delta = -lim
                effective_mix_ratio = min(0.95, max(0.0, effective_mix_ratio * (1.0 + delta)))
            if hermaphrodites:
                effective_mix_ratio = effective_mix_ratio / float(getattr(self, 'hermaphrodite_mate_bias', 2.5))
            if self.rng.random() < effective_mix_ratio:
                parent = parent_candidate
                if parent.sex == 'hermaphrodite':
                    use_sexual_reproduction = True
                elif parent.regen and self.mode.enable_regen_reproduction:
                    if monitor is not None:
                        parent_adj_before_regen = parent.weighted_adjacency()
                    child = platyregenerate(parent, self.rng, self.innov, intensity=self._regen_intensity())
                    child.cooperative = True
                    mode = 'asexual_regen'
                    mother_id = parent.id
                    father_id = None
                else:
                    child = parent.copy()
                    child.cooperative = True
                    mode = 'asexual_clone'
                    mother_id = parent.id
                    father_id = None
            else:
                use_sexual_reproduction = True
            if use_sexual_reproduction:
                potential_mothers = females + hermaphrodites
                potential_fathers = males + hermaphrodites
                if potential_mothers and potential_fathers and (self.rng.random() > self.pollen_flow_rate):
                    mother = potential_mothers[int(self.rng.integers(len(potential_mothers)))]
                    if potential_fathers:
                        if self.rng.random() < float(getattr(self, 'non_elite_mating_rate', 0.5)):
                            cand = [f for f in potential_fathers if f.id not in non_elite_ids] or potential_fathers
                            father = cand[int(self.rng.integers(len(cand)))]
                        else:
                            father = potential_fathers[int(self.rng.integers(len(potential_fathers)))]
                    else:
                        father = mother
                    mode = 'sexual_within'
                    sp_for_fit = sp.members
                elif len(species_pool) > 1:
                    mother = pool[int(self.rng.integers(len(pool)))]
                    other = species_pool[(sidx + 1) % len(species_pool)]
                    other_pool = [g for g, _ in other.members]
                    other_males = [g for g, _ in other.members if g.sex == 'male']
                    other_herm = [g for g, _ in other.members if g.sex == 'hermaphrodite']
                    father_pool = other_males + other_herm if other_males or other_herm else other_pool
                    father = father_pool[int(self.rng.integers(len(father_pool)))]
                    mode = 'sexual_cross'
                    sp_for_fit = sp.members + other.members
                elif potential_mothers and potential_fathers:
                    mother = potential_mothers[int(self.rng.integers(len(potential_mothers)))]
                    father = potential_fathers[int(self.rng.integers(len(potential_fathers)))]
                    mode = 'sexual_within'
                    sp_for_fit = sp.members
                else:
                    parent = pool[int(self.rng.integers(len(pool)))]
                    if self.allow_selfing:
                        mother = parent
                        father = parent
                        mode = 'sexual_within'
                        sp_for_fit = sp.members
                    else:
                        child = parent.copy()
                        child.cooperative = True
                        mode = 'asexual_clone'
                        mother_id = parent.id
                        father_id = None
                if mode in ('sexual_within', 'sexual_cross'):
                    child = self._crossover_maternal_biased(mother, father, sp_for_fit)
                    child.hybrid_scale = self._heterosis_scale(mother, father)
                    mother_id = mother.id
                    father_id = father.id
                child.cooperative = True
            child.id = self.next_gid
            self.next_gid += 1
            child.parents = (mother_id, father_id)
            child.birth_gen = self.generation + 1
            child.origin_mode = mode
            self._assign_child_family(child, (parent, mother, father))
            parent_strengths: List[float] = []
            parent_lazy_flags: List[bool] = []
            if parent is not None:
                strength = float(getattr(parent, 'lazy_lineage_strength', 0.0) or 0.0)
                parent_strengths.append(strength)
                parent_lazy_flags.append(bool(getattr(parent, 'lazy_lineage', False)))
            if mother is not None:
                parent_strengths.append(float(getattr(mother, 'lazy_lineage_strength', 0.0) or 0.0))
                parent_lazy_flags.append(bool(getattr(mother, 'lazy_lineage', False)))
            if father is not None and father is not mother:
                parent_strengths.append(float(getattr(father, 'lazy_lineage_strength', 0.0) or 0.0))
                parent_lazy_flags.append(bool(getattr(father, 'lazy_lineage', False)))
            base_strength = float(max(parent_strengths) if parent_strengths else 0.0)
            mix_strength = float(np.mean(parent_strengths)) if parent_strengths else 0.0
            inheritance_decay = float(np.clip(getattr(self, 'lazy_lineage_inheritance_decay', 0.24), 0.0, 0.95))
            inheritance_gain = float(max(0.0, getattr(self, 'lazy_lineage_inheritance_gain', 0.0)))
            strength_cap = float(max(0.0, getattr(self, 'lazy_lineage_strength_cap', 3.0)))
            active_lazy = any(parent_lazy_flags)
            carried = base_strength * (1.0 - inheritance_decay) + mix_strength * 0.35
            gain = inheritance_gain * (1.0 if active_lazy else 0.5)
            new_strength = float(np.clip(carried + gain, 0.0, strength_cap))
            setattr(child, 'lazy_lineage_strength', new_strength)
            setattr(child, 'lazy_lineage', False)
            self.node_registry[child.id] = {'sex': child.sex, 'regen': child.regen, 'birth_gen': child.birth_gen, 'family_id': child.family_id}
            if monitor is not None and parent_adj_before_regen is not None:
                regen_adj = child.weighted_adjacency()
                changed_nodes, changed_edges = summarize_graph_changes(parent_adj_before_regen, regen_adj, weight_tol)
                rows_regen = monitor.log_step(parent_adj_before_regen, regen_adj, changed_nodes, child.id, self.generation + 1, f'{child.id}_regen', changed_edges=changed_edges)
                regen_summary = _summarize_lineage_rows(rows_regen)
                if regen_summary:
                    self._note_lineage(child.id, child.birth_gen, regen_summary)
            pre_adj = child.weighted_adjacency() if monitor is not None else None
            self._mutate(child, context='regen' if mode == 'asexual_regen' else None)
            if monitor is not None and pre_adj is not None:
                post_adj = child.weighted_adjacency()
                changed_nodes, changed_edges = summarize_graph_changes(pre_adj, post_adj, weight_tol)
                rows_mut = monitor.log_step(pre_adj, post_adj, changed_nodes, child.id, self.generation + 1, f'{child.id}_{mode}', changed_edges=changed_edges)
                summary = _summarize_lineage_rows(rows_mut)
                summary = _merge_meta_summary(child, summary)
                self._note_lineage(child.id, child.birth_gen, summary)
            elif monitor is None:
                summary = _merge_meta_summary(child, '')
                self._note_lineage(child.id, child.birth_gen, summary)
            if mode is None:
                mode = 'asexual_clone'
            new_pop.append(child)
            events[mode] += 1
            remaining -= 1
        return (new_pop, events)

    def reproduce(self, species, fitnesses):
        cleaned_species = []
        for sp in species:
            members = [(g, f) for g, f in sp.members if not getattr(g, 'selfish_culled', False)]
            if members:
                sp.members = members
                cleaned_species.append(sp)
        if cleaned_species:
            species = cleaned_species
        total_adjusted = 0.0
        species_adjusted: List[float] = []
        for sp in species:
            adj = sum((f for _, f in sp.members)) / len(sp.members)
            species_adjusted.append(adj)
            total_adjusted += adj
        if total_adjusted <= 0.0:
            offspring_counts = [self.pop_size // len(species)] * len(species)
            for i in range(self.pop_size - sum(offspring_counts)):
                offspring_counts[i % len(offspring_counts)] += 1
        else:
            shares = [adj / total_adjusted for adj in species_adjusted]
            offspring_counts = [int(round(s * self.pop_size)) for s in shares]
            diff = self.pop_size - sum(offspring_counts)
            idxs = np.argsort(shares)[::-1]
            i = 0
            while diff != 0:
                idx = int(idxs[i % len(idxs)])
                offspring_counts[idx] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                i += 1
        new_pop: List[Genome] = []
        gen_events = {'sexual_within': 0, 'sexual_cross': 0, 'asexual_regen': 0, 'asexual_clone': 0}
        monitor = getattr(self, 'lcs_monitor', None)
        weight_tol = getattr(monitor, 'eps', 0.0) if monitor is not None else 0.0
        for sidx, sp in enumerate(species):
            offspring, events = self._make_offspring(species, offspring_counts, sidx, species)
            for k, v in events.items():
                gen_events[k] += v
            new_pop.extend(offspring)
        if len(new_pop) < self.pop_size:
            bests = [g for sp in species for g, _ in sp.members]
            while len(new_pop) < self.pop_size:
                parent = bests[int(self.rng.integers(len(bests)))]
                child = parent.copy()
                child.cooperative = True
                child.id = self.next_gid
                self.next_gid += 1
                child.parents = (parent.id, None)
                child.birth_gen = self.generation + 1
                self._assign_child_family(child, (parent,))
                self.node_registry[child.id] = {
                    'sex': child.sex,
                    'regen': child.regen,
                    'birth_gen': child.birth_gen,
                    'family_id': child.family_id,
                }
                parent_strength = float(getattr(parent, 'lazy_lineage_strength', 0.0) or 0.0)
                inheritance_decay = float(np.clip(getattr(self, 'lazy_lineage_inheritance_decay', 0.24), 0.0, 0.95))
                inheritance_gain = float(max(0.0, getattr(self, 'lazy_lineage_inheritance_gain', 0.0)))
                strength_cap = float(max(0.0, getattr(self, 'lazy_lineage_strength_cap', 3.0)))
                gain = inheritance_gain * (1.0 if bool(getattr(parent, 'lazy_lineage', False)) else 0.5)
                new_strength = float(np.clip(parent_strength * (1.0 - inheritance_decay) + gain, 0.0, strength_cap))
                setattr(child, 'lazy_lineage_strength', new_strength)
                setattr(child, 'lazy_lineage', False)
                pre_adj = child.weighted_adjacency() if monitor is not None else None
                self._mutate(child)
                if monitor is not None and pre_adj is not None:
                    post_adj = child.weighted_adjacency()
                    changed_nodes, changed_edges = summarize_graph_changes(pre_adj, post_adj, weight_tol)
                    rows_clone = monitor.log_step(
                        pre_adj,
                        post_adj,
                        changed_nodes,
                        child.id,
                        self.generation + 1,
                        f'{child.id}_asexual_clone',
                        changed_edges=changed_edges,
                    )
                    summary = _summarize_lineage_rows(rows_clone)
                    summary = _merge_meta_summary(child, summary)
                    self._note_lineage(child.id, child.birth_gen, summary)
                elif monitor is None:
                    summary = _merge_meta_summary(child, '')
                    self._note_lineage(child.id, child.birth_gen, summary)
                new_pop.append(child)
                gen_events['asexual_clone'] += 1
        elif len(new_pop) > self.pop_size:
            new_pop = new_pop[:self.pop_size]
        self._assign_lazy_individuals(new_pop)
        self.population = new_pop
        self.event_log.append(gen_events)
        for g in new_pop:
            self.lineage_edges.append((g.parents[0], g.parents[1], g.id, g.birth_gen, 'birth'))

    def _assign_lazy_individuals(self, population: List[Genome]):
        frac = float(getattr(self, 'lazy_fraction', 0.02))
        if not population:
            return
        max_frac = float(getattr(self, 'lazy_fraction_max', frac))
        max_frac = float(np.clip(max_frac, 0.0, max(0.0, frac * 1.1 + 1e-9)))
        decay = float(np.clip(getattr(self, 'lazy_lineage_decay', 0.8), 0.0, 1.0))
        strength_cap = float(max(0.0, getattr(self, 'lazy_lineage_strength_cap', 3.0)))
        for g in population:
            g.cooperative = True
            setattr(g, 'lazy_lineage', False)
            strength = float(getattr(g, 'lazy_lineage_strength', 0.0) or 0.0)
            if strength > 0.0:
                setattr(g, 'lazy_lineage_strength', float(np.clip(strength * decay, 0.0, strength_cap)))
        if frac <= 0.0:
            self._lazy_fraction_carry = 0.0
            return
        target = float(frac) * float(len(population))
        carry = float(getattr(self, '_lazy_fraction_carry', 0.0))
        desired = target + carry
        lazy_count = int(math.floor(desired + 1e-09))
        remainder = desired - lazy_count
        max_lazy = int(math.floor(max_frac * len(population) + 1e-09))
        if lazy_count > max_lazy:
            remainder += float(lazy_count - max_lazy)
            lazy_count = max_lazy
        elif remainder > 0.0 and lazy_count < max_lazy:
            if float(self.rng.random()) < remainder and (lazy_count + 1) <= max_lazy:
                lazy_count += 1
                remainder = max(0.0, remainder - 1.0)
        lazy_count = min(int(lazy_count), len(population))
        if lazy_count <= 0:
            self._lazy_fraction_carry = remainder
            return
        self._lazy_fraction_carry = remainder
        strengths = np.asarray([max(1e-09, 1.0 + float(getattr(g, 'lazy_lineage_strength', 0.0) or 0.0)) for g in population], dtype=np.float64)
        strengths_sum = float(strengths.sum())
        if not np.isfinite(strengths_sum) or strengths_sum <= 0.0:
            strengths = None
        else:
            strengths = strengths / strengths_sum
        indices = self.rng.choice(len(population), size=lazy_count, replace=False, p=strengths)
        for idx in np.atleast_1d(indices):
            g = population[int(idx)]
            g.cooperative = False
            setattr(g, 'lazy_lineage', True)
            try:
                self._promote_lazy_complexity(g)
            except Exception:
                pass

    def _promote_lazy_complexity(self, genome: Genome):
        steps = int(max(0, getattr(self, 'lazy_complexity_growth_steps', 0)))
        if steps <= 0:
            return
        target_scale = float(getattr(self, 'lazy_complexity_target_scale', 1.1))
        min_growth = float(getattr(self, 'lazy_complexity_min_growth', 0.0))
        auto_state = getattr(self, '_auto_complexity_bonus_state', None)
        baseline = 0.0
        spread = 0.0
        span_val = 0.0
        if auto_state:
            baseline = float(auto_state.get('baseline', 0.0) or 0.0)
            spread = float(auto_state.get('spread', 0.0) or 0.0)
            span_val = float(auto_state.get('span_value', 0.0) or 0.0)
        target_score = max(0.0, baseline * target_scale + min_growth)
        if spread > 0.0:
            target_score = max(target_score, baseline + spread * target_scale)
        if span_val > 0.0:
            target_score = max(target_score, span_val)
        lazy_target = float(auto_state.get('lazy_target', 0.0)) if auto_state else 0.0
        if lazy_target > 0.0:
            target_score = max(target_score, lazy_target)
        duplicate_bias = float(getattr(self, 'lazy_complexity_duplicate_bias', 0.4))
        current_score = float(genome.structural_complexity_score())
        if target_score > current_score and spread > 0.0:
            gap = float(max(0.0, target_score - current_score))
            approx_step_gain = max(1.0, spread)
            boost_steps = int(math.ceil(gap / approx_step_gain))
            steps = max(steps, boost_steps)
        for _ in range(steps):
            current = float(genome.structural_complexity_score())
            if current >= target_score:
                break
            roll = float(self.rng.random())
            try:
                if roll < 0.45:
                    genome.mutate_add_node(self.rng, self.innov)
                elif roll < 0.9:
                    genome.mutate_add_connection(self.rng, self.innov)
                else:
                    genome.mutate_weights(self.rng)
            except Exception:
                pass
            if duplicate_bias > 0.0 and float(self.rng.random()) < duplicate_bias:
                try:
                    scale = float(getattr(self, 'duplicate_branch_weight_scale', 0.85))
                except Exception:
                    scale = 0.85
                try:
                    genome.mutate_duplicate_node(self.rng, self.innov, weight_scale=scale)
                except Exception:
                    pass
        try:
            genome.cooperative = False
        except Exception:
            pass
        try:
            post_score = float(genome.structural_complexity_score())
        except Exception:
            post_score = 0.0
        baseline = max(1e-09, baseline)
        growth_ratio = float(max(0.0, post_score - baseline) / (baseline + spread + 1e-09))
        persistence = float(np.clip(getattr(self, 'lazy_lineage_persistence', 0.7), 0.0, 1.0))
        cap = float(max(0.0, getattr(self, 'lazy_lineage_strength_cap', 3.0)))
        prior = float(getattr(genome, 'lazy_lineage_strength', 0.0) or 0.0)
        gain = float(np.clip(growth_ratio * max(0.6, target_scale), 0.0, cap))
        new_strength = float(np.clip(prior * persistence + gain, 0.0, cap))
        setattr(genome, 'lazy_lineage_strength', new_strength)

    def _auto_tune_complexity_controls(self, scores: List[float], max_score: float) -> Optional[Dict[str, float]]:
        prev = getattr(self, '_auto_complexity_bonus_state', None)
        if not scores:
            return prev
        arr = np.asarray(scores, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return prev
        arr.sort()
        target_frac = float(getattr(self, 'auto_complexity_bonus_fraction', 0.18))
        target_frac = min(0.45, max(0.02, target_frac))
        baseline_q = float(np.clip(1.0 - target_frac, 0.55, 0.97))
        baseline = float(np.quantile(arr, baseline_q))
        median = float(np.quantile(arr, 0.5))
        span_quantile = float(np.clip(max(baseline_q + 0.12, baseline_q + 0.01), baseline_q + 0.01, 0.995))
        q90 = float(np.quantile(arr, min(0.95, baseline_q + 0.18)))
        span_val = float(np.quantile(arr, span_quantile))
        peak = float(max(float(max_score or 0.0), float(arr[-1])))
        spread = float(max(1e-09, q90 - median))
        intensity = float(min(1.0, spread / max(1e-09, baseline + median + 1e-09)))
        bonus_scale = float(np.clip(0.05 + 0.55 * intensity, 0.03, 0.85))
        exponent = float(np.clip(1.0 + 0.8 * intensity, 1.0, 1.8))
        cap_ratio = 0.0
        if peak > baseline:
            cap_ratio = float((peak - baseline) / max(1e-09, peak))
        cap = float(np.clip(0.8 + 1.4 * cap_ratio, 0.9, 2.2))
        bonus_limit = float(np.clip(0.35 + 1.35 * intensity, 0.4, 2.8))
        diff_pressure = float(min(1.0, max(0.0, float(self.env.get('difficulty', 0.0))) / 3.0))
        release = float(min(1.0, baseline / max(1e-09, peak)))
        penalty_multiplier = float(np.clip(1.0 - 0.6 * release * (0.4 + 0.6 * diff_pressure), 0.2, 1.1))
        penalty_threshold = float((baseline + spread) * (1.2 + 0.4 * release))
        bonus_multiplier = float(-0.04 - 0.12 * release)
        bonus_threshold = float(1.1 + 1.2 * (1.0 - release))
        lazy_target = float(baseline + spread * max(1.0, getattr(self, 'lazy_complexity_target_scale', 1.18)))
        smoothing = float(np.clip(getattr(self, 'auto_complexity_smoothing', 0.35), 0.0, 0.95))
        new_state = {
            'baseline': baseline,
            'bonus_scale': bonus_scale,
            'exponent': exponent,
            'cap': cap,
            'baseline_quantile': baseline_q,
            'span_quantile': span_quantile,
            'penalty_multiplier': penalty_multiplier,
            'penalty_threshold': penalty_threshold,
            'bonus_multiplier': bonus_multiplier,
            'bonus_threshold': bonus_threshold,
            'bonus_limit': bonus_limit,
            'lazy_target': lazy_target,
            'spread': spread,
            'span_value': span_val,
        }
        if prev:
            mixed = {}
            for key, val in new_state.items():
                old = prev.get(key)
                if isinstance(val, float) and np.isfinite(val):
                    if isinstance(old, float) and np.isfinite(old):
                        mixed[key] = float(old * smoothing + val * (1.0 - smoothing))
                    else:
                        mixed[key] = float(val)
                else:
                    mixed[key] = val if val is not None else old
            new_state = {**prev, **mixed}
        self._auto_complexity_bonus_state = new_state
        if getattr(self, 'auto_complexity_controls', False):
            self.complexity_survivor_bonus = new_state['bonus_scale']
            self.complexity_survivor_exponent = new_state['exponent']
            self.complexity_survivor_cap = new_state['cap']
            self.complexity_bonus_baseline_quantile = new_state['baseline_quantile']
            self.complexity_bonus_span_quantile = new_state['span_quantile']
            self.complexity_bonus_multiplier = new_state['bonus_multiplier']
            self.complexity_bonus_threshold = new_state['bonus_threshold']
            self.complexity_threshold = new_state['penalty_threshold']
            self.complexity_survivor_bonus_limit = new_state['bonus_limit']
        return new_state

    def _complexity_penalty(self, g: Genome) -> float:
        """Adaptive complexity penalty that encourages complex topologies under high difficulty."""
        n_hidden = sum((1 for n in g.nodes.values() if n.type == 'hidden'))
        n_edges = sum((1 for c in g.connections.values() if c.enabled))
        m = self.mode
        diff = float(self.env.get('difficulty', 0.0))
        bonus_threshold = getattr(self, 'complexity_bonus_threshold', 2.5)
        bonus_multiplier = getattr(self, 'complexity_bonus_multiplier', -0.1)
        auto_state = None
        if getattr(self, 'auto_complexity_controls', False):
            auto_state = getattr(self, '_auto_complexity_bonus_state', None)
            if auto_state:
                bonus_threshold = float(auto_state.get('bonus_threshold', bonus_threshold) or bonus_threshold)
                bonus_multiplier = float(auto_state.get('bonus_multiplier', bonus_multiplier) or bonus_multiplier)
        if diff < 0.5:
            multiplier = 1.0
        elif diff < 1.5:
            multiplier = 1.0 - 0.7 * (diff - 0.5) / 1.0
        elif diff < bonus_threshold:
            multiplier = 0.3 - 0.3 * (diff - 1.5) / (bonus_threshold - 1.5)
        else:
            multiplier = bonus_multiplier * (diff - bonus_threshold)
        if auto_state:
            mult_adj = float(auto_state.get('penalty_multiplier', 1.0) or 1.0)
            multiplier *= mult_adj
        penalty = multiplier * m.complexity_alpha * (m.node_penalty * n_hidden + m.edge_penalty * n_edges)
        threshold = getattr(self, 'complexity_threshold', None)
        if auto_state:
            auto_thresh = auto_state.get('penalty_threshold')
            if auto_thresh is not None:
                threshold = float(auto_thresh)
        if threshold is not None and penalty > 0:
            penalty = min(float(threshold), penalty)
        return penalty

    def _evaluate_population(self, fitness_fn: Callable[[Genome], float]) -> List[float]:
        """並列評価（thread/process）。process は SHM メタを初期化し、必要なら持ち回りプールを再起動。"""
        workers = int(getattr(self, 'eval_workers', 1))
        lazy_penalty = float(getattr(self, 'lazy_individual_fitness', 0.0))
        coop_pairs: List[Tuple[int, Genome]] = []
        out: List[Optional[float]] = [lazy_penalty] * len(self.population)
        for idx, g in enumerate(self.population):
            if getattr(g, 'cooperative', True):
                out[idx] = None
                coop_pairs.append((idx, g))
        if not coop_pairs:
            return [float(x) for x in out]
        genomes_to_eval = [g for _, g in coop_pairs]
        if workers <= 1:
            for idx, g in coop_pairs:
                try:
                    out[idx] = float(fitness_fn(g))
                except Exception as _e:
                    _tb = traceback.format_exc()
                    print(f"[ERROR] fitness exception gid={getattr(g, 'id', '?')}: {_e}\n{_tb}")
                    out[idx] = float(-1000000000.0)
            return [float(x if x is not None else lazy_penalty) for x in out]
        backend = getattr(self, 'parallel_backend', 'thread')
        if backend == 'process' and (not _is_picklable(fitness_fn)):
            print('[WARN] fitness_fn is not picklable; falling back to threads')
            backend = 'thread'
        try:
            import concurrent.futures as _cf
            if backend == 'process':
                import multiprocessing as _mp
                start = os.environ.get('NEAT_PROCESS_START_METHOD', 'spawn')
                try:
                    ctx = _mp.get_context(start)
                except ValueError:
                    ctx = _mp.get_context('spawn')
                initargs = (getattr(self, '_shm_meta', None),)
                if int(getattr(self, 'pool_keepalive', 0)) > 0:
                    need_new = self._proc_pool is None or int(getattr(self, '_proc_pool_age', 0)) >= int(getattr(self, 'pool_restart_every', 25))
                    if need_new:
                        self._close_pool()
                        self._proc_pool = _cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_proc_init_worker, initargs=initargs)
                        self._proc_pool_age = 0
                    ex = self._proc_pool
                    futs = [ex.submit(fitness_fn, g) for g in genomes_to_eval]
                    for (idx, g), fut in zip(coop_pairs, futs):
                        try:
                            out[idx] = float(fut.result())
                        except Exception as _e:
                            _tb = traceback.format_exc()
                            print(f"[ERROR] fitness exception (proc) gid={getattr(g, 'id', '?')}: {_e}\n{_tb}")
                            out[idx] = float(-1000000000.0)
                    self._proc_pool_age += 1
                    return [float(x if x is not None else lazy_penalty) for x in out]
                else:
                    with _cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_proc_init_worker, initargs=initargs) as ex:
                        futs = [ex.submit(fitness_fn, g) for g in genomes_to_eval]
                        for (idx, g), fut in zip(coop_pairs, futs):
                            try:
                                out[idx] = float(fut.result())
                            except Exception as _e:
                                _tb = traceback.format_exc()
                                print(f"[ERROR] fitness exception (proc) gid={getattr(g, 'id', '?')}: {_e}\n{_tb}")
                                out[idx] = float(-1000000000.0)
                    return [float(x if x is not None else lazy_penalty) for x in out]
            else:
                if backend != 'thread':
                    print(f"[WARN] Unknown backend '{backend}', defaulting to threads")
                with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(fitness_fn, g) for g in genomes_to_eval]
                    for (idx, g), fut in zip(coop_pairs, futs):
                        try:
                            out[idx] = float(fut.result())
                        except Exception as _e:
                            _tb = traceback.format_exc()
                            print(f"[ERROR] fitness exception (thread) gid={getattr(g, 'id', '?')}: {_e}\n{_tb}")
                            out[idx] = float(-1000000000.0)
                return [float(x if x is not None else lazy_penalty) for x in out]
        except Exception as e:
            print('[WARN] parallel evaluation disabled:', e)
            for idx, g in coop_pairs:
                try:
                    out[idx] = float(fitness_fn(g))
                except Exception as _e:
                    _tb = traceback.format_exc()
                    print(f"[ERROR] fitness exception gid={getattr(g, 'id', '?')}: {_e}\n{_tb}")
                    out[idx] = float(-1000000000.0)
            return [float(x if x is not None else lazy_penalty) for x in out]

    def _close_pool(self):
        ex = getattr(self, '_proc_pool', None)
        if ex is not None:
            try:
                ex.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._proc_pool = None
            self._proc_pool_age = 0

    def _auto_env_schedule(self, gen: int, history: List[Tuple[float, float]]) -> Dict[str, float]:
        """進捗に応じて difficulty / noise を自動昇圧。高難度で再生を解禁。上限撤廃版。"""
        diff = float(self.env.get('difficulty', 0.0))
        best_hist = [b for b, _a in history] if history else []
        bump = 0.0
        if len(best_hist) >= 10:
            delta10 = best_hist[-1] - best_hist[-10]
            if delta10 < 0.01:
                bump = 0.1
            elif delta10 < 0.05:
                bump = 0.05
        if gen < 10:
            diff = max(diff, 0.3)
        elif gen < 25:
            diff = max(diff, 0.5)
        else:
            diff = diff + bump
        diff += 0.08 * self._household_pressure()
        diff = float(np.clip(diff, 0.0, 5.0))
        enable_regen = bool(diff >= 0.85)
        noise_std = 0.01 + 0.05 * diff
        return {'difficulty': float(diff), 'noise_std': float(noise_std), 'enable_regen': enable_regen}

    def _household_pressure(self) -> float:
        manager = getattr(self, '_households', None)
        if manager is None:
            return 0.0
        try:
            return float(manager.global_pressure())
        except Exception:
            return 0.0

    def _adaptive_refine_fitness(self, fitnesses: List[float], fitness_fn: Callable[[Genome], float]) -> List[float]:
        """上位個体にだけ backprop ステップを追加して再評価（軽量な二段評価）。"""
        if not hasattr(fitness_fn, 'refine_raw'):
            return fitnesses
        n = len(fitnesses)
        if n == 0:
            return fitnesses
        k = max(1, int(float(getattr(self, 'refine_topk_ratio', 0.08)) * n))
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
                    f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * getattr(g, 'hybrid_scale', 1.0)
                    if g.regen:
                        f2 += self.regen_bonus
                f2 -= self._complexity_penalty(self.population[i])
                if np.isfinite(f2):
                    improved[i] = f2
            except Exception:
                pass
        return improved

    def _apply_monodromy_pressure(
        self,
        fitnesses: Sequence[float],
        baseline: Sequence[float],
        signature_map: Dict[int, Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[int, int, int], ...]]],
        generation: int,
        signature_counts: Optional[Dict[Any, int]]=None,
        family_counts: Optional[Dict[int, int]]=None,
        family_metrics: Optional[Dict[int, Dict[str, Any]]]=None,
    ) -> List[float]:
        def _reset_snapshot() -> None:
            self._monodromy_snapshot = {
                'pressure_mean': 0.0,
                'pressure_max': 0.0,
                'active': 0,
                'families': 0,
                'release': 0,
                'relief_mean': 0.0,
                'momentum_mean': 0.0,
                'momentum_max': 0.0,
                'diversity_mean': 0.0,
                'diversity_max': 0.0,
                'grace_mean': 0.0,
                'noise_factor': 1.0,
                'noise_kind': '',
                'noise_focus': 0.0,
                'noise_entropy': 0.0,
                'family_factor_mean': 1.0,
                'family_factor_max': 1.0,
            }

        n = len(fitnesses)
        if n == 0:
            _reset_snapshot()
            return list(fitnesses)
        base = float(getattr(self, 'monodromy_pressure_base', 0.0))
        rng = float(getattr(self, 'monodromy_pressure_range', 0.0))
        if base <= 0.0 and rng <= 0.0:
            _reset_snapshot()
            return list(fitnesses)
        try:
            baseline_arr = np.asarray(baseline, dtype=float)
        except Exception:
            baseline_arr = np.array(list(map(float, baseline)), dtype=float)
        if baseline_arr.size == 0:
            _reset_snapshot()
            return list(fitnesses)
        overrides = getattr(self, 'monodromy_noise_style_overrides', None)
        controller = getattr(self, 'spinor_controller', None)
        env_obj = getattr(controller, 'env', None) if controller is not None else None
        noise_kind = ''
        noise_focus = 0.0
        noise_entropy = 0.0
        if env_obj is not None:
            noise_kind = getattr(env_obj, 'noise_kind', '') or ''
            noise_focus = float(getattr(env_obj, 'noise_focus', 0.0) or 0.0)
            noise_entropy = float(getattr(env_obj, 'noise_entropy', 0.0) or 0.0)
        else:
            noise_kind = str(self.env.get('noise_kind', ''))
            noise_focus = float(self.env.get('noise_focus', 0.0) or 0.0)
            noise_entropy = float(self.env.get('noise_entropy', 0.0) or 0.0)
        style = _resolve_noise_style(noise_kind, overrides)
        noise_bias = float(style.get('bias', 0.0))
        noise_weight = float(np.clip(getattr(self, 'monodromy_noise_weight', 0.0), 0.0, 2.0))
        entropy_excess = max(0.0, float(noise_entropy) - float(noise_focus))
        noise_factor = float(np.clip(1.0 + noise_weight * (noise_bias - 0.3 * entropy_excess), 0.2, 1.6))
        self._monodromy_noise_tag = style.get('symbol', noise_kind or '')
        if signature_counts is None:
            signature_counts = Counter(signature_map.values()) if signature_map else Counter()
        else:
            signature_counts = Counter(signature_counts)
        div_weight = float(np.clip(getattr(self, 'monodromy_diversity_weight', 0.0), 0.0, 2.0))
        unique_relief = float(np.clip(getattr(self, 'monodromy_diversity_floor', 0.0), 0.0, 0.9))
        grace_init = float(max(0.0, getattr(self, 'monodromy_diversity_grace', 0.0)))
        grace_decay = float(np.clip(getattr(self, 'monodromy_diversity_grace_decay', 0.0), 0.05, 0.99))
        grace_strength = float(np.clip(getattr(self, 'monodromy_diversity_grace_strength', 0.0), 0.0, 1.0))
        try:
            top_ratio = float(getattr(self, 'monodromy_top_ratio', 0.1))
        except Exception:
            top_ratio = 0.1
        pop = getattr(self, 'population', [])
        family_members: Dict[int, Tuple[int, ...]] = {}
        family_best_idx: Dict[int, int] = {}
        fam_counts_local: Dict[int, int] = {}
        if family_metrics:
            fam_counts_local = {}
            for key, info in family_metrics.items():
                try:
                    fid = int(key)
                except Exception:
                    continue
                members_seq = info.get('members', ()) if isinstance(info, dict) else ()
                members_tuple = tuple(
                    int(m)
                    for m in members_seq
                    if isinstance(m, (int, np.integer)) and 0 <= int(m) < len(pop)
                )
                if not members_tuple and isinstance(info, dict):
                    # Fallback to recorded best index if provided
                    best_idx = int(info.get('best_idx', -1)) if 'best_idx' in info else -1
                    if 0 <= best_idx < len(pop):
                        members_tuple = (best_idx,)
                if not members_tuple:
                    continue
                fam_counts_local[fid] = int(info.get('size', len(members_tuple))) if isinstance(info, dict) else len(members_tuple)
                best_idx = int(info.get('best_idx', members_tuple[0])) if isinstance(info, dict) else members_tuple[0]
                if best_idx < 0 or best_idx >= len(pop):
                    best_idx = members_tuple[0]
                family_members[fid] = members_tuple
                family_best_idx[fid] = best_idx
        else:
            fallback_members: Dict[int, List[int]] = defaultdict(list)
            fam_counts_local = {}
            if family_counts:
                for key, val in family_counts.items():
                    try:
                        fam_counts_local[int(key)] = int(val)
                    except Exception:
                        continue
            for idx, genome in enumerate(pop):
                if idx >= baseline_arr.size:
                    break
                fid_raw = getattr(genome, 'family_id', None)
                if fid_raw is None:
                    parents = getattr(genome, 'parents', (None, None))
                    fid_raw = parents[0] if parents and parents[0] is not None else genome.id
                try:
                    fid = int(fid_raw)
                except Exception:
                    fid = int(getattr(genome, 'id', idx))
                fallback_members[fid].append(idx)
                prev_idx = family_best_idx.get(fid)
                if prev_idx is None or baseline_arr[idx] > baseline_arr[prev_idx]:
                    family_best_idx[fid] = idx
            family_members = {fid: tuple(members) for fid, members in fallback_members.items() if members}
            for fid, members in family_members.items():
                fam_counts_local.setdefault(fid, len(members))
        family_counts = fam_counts_local
        families = list(family_best_idx.keys())
        family_order: List[int] = []
        if families:
            fam_ids = np.asarray(families, dtype=np.int64)
            fam_scores = np.asarray([baseline_arr[family_best_idx[fid]] for fid in fam_ids], dtype=np.float64)
            fam_count = fam_ids.size
            fam_top = max(1, min(fam_count, int(math.ceil(top_ratio * fam_count))))
            if fam_top >= fam_count:
                fam_order_idx = np.argsort(fam_scores)[::-1]
            else:
                fam_part = np.argpartition(fam_scores, -fam_top)[-fam_top:]
                fam_order_idx = fam_part[np.argsort(fam_scores[fam_part])[::-1]]
            family_order = [int(fam_ids[i]) for i in fam_order_idx]
            top_indices = [int(family_best_idx[fid]) for fid in family_order]
        else:
            order = np.argsort(baseline_arr)[::-1]
            top_k = max(1, min(n, int(round(top_ratio * n))))
            top_indices = [int(i) for i in order[:top_k]]
            if pop:
                for idx in top_indices:
                    if 0 <= idx < len(pop):
                        fid_raw = getattr(pop[idx], 'family_id', None)
                        if fid_raw is None:
                            parents = getattr(pop[idx], 'parents', (None, None))
                            fid_raw = parents[0] if parents and parents[0] is not None else getattr(pop[idx], 'id', idx)
                        try:
                            family_order.append(int(fid_raw))
                        except Exception:
                            family_order.append(int(idx))
                    else:
                        family_order.append(int(idx))
            else:
                family_order = [int(i) for i in top_indices]
        if not top_indices:
            _reset_snapshot()
            return list(fitnesses)
        median = float(np.median(baseline_arr))
        if family_order:
            best_idx = family_best_idx.get(family_order[0], top_indices[0])
            best_val = float(baseline_arr[best_idx]) if baseline_arr.size else median
        else:
            best_val = float(baseline_arr[top_indices[0]]) if top_indices else median
        span_scale = abs(best_val - median)
        if not math.isfinite(span_scale) or span_scale < 1e-6:
            span_scale = max(1e-6, abs(best_val) if math.isfinite(best_val) else 1.0)
        phase_step = float(getattr(self, 'monodromy_phase_step', 0.38196601125))
        smoothing = float(np.clip(getattr(self, 'monodromy_smoothing', 0.4), 0.0, 1.0))
        span = float(max(1.0, getattr(self, 'monodromy_span', 4.0)))
        decay = float(np.clip(getattr(self, 'monodromy_decay', 0.5), 0.0, 1.0))
        release = float(np.clip(getattr(self, 'monodromy_release', 0.35), 0.0, 1.0))
        cap = float(max(0.0, getattr(self, 'monodromy_penalty_cap', 1.0)))
        momentum_decay = float(np.clip(getattr(self, 'monodromy_momentum_decay', 0.65), 0.0, 1.0))
        growth_weight = float(np.clip(getattr(self, 'monodromy_growth_weight', 0.0), 0.0, 1.5))
        slump_gain = float(np.clip(getattr(self, 'monodromy_slump_gain', 0.0), 0.0, 1.5))
        fast_release = float(np.clip(getattr(self, 'monodromy_fast_release', 0.0), 0.0, 1.0))
        registry = getattr(self, '_monodromy_registry', None)
        if registry is None:
            registry = {}
            self._monodromy_registry = registry
        adjusted = list(fitnesses)
        seen_families: Set[int] = set()
        total_penalty = 0.0
        spill_penalty_total = 0.0
        max_penalty = 0.0
        release_count = 0
        relief_total = 0.0
        momentum_total = 0.0
        momentum_max = 0.0
        diversity_total = 0.0
        diversity_max = 0.0
        grace_total = 0.0
        family_factor_total = 0.0
        family_factor_max = 1.0
        elite_seen = 0
        span_scale_safe = max(span_scale, 1e-6)
        family_weight = float(np.clip(getattr(self, 'monodromy_family_weight', 0.0), 0.0, 2.0))
        pop_size = max(1, len(pop))
        family_target_share = 1.0 / float(max(1, len(family_counts))) if family_counts else 0.0
        family_surplus_total = 0.0
        family_surplus_max = 0.0
        family_spread_total = 0.0
        family_trend_total = 0.0
        for fam_id, idx in zip(family_order, top_indices):
            if idx < 0 or idx >= n or idx >= len(self.population):
                continue
            genome = self.population[idx]
            gid = genome.id
            try:
                family_id = int(fam_id)
            except Exception:
                try:
                    family_id = int(getattr(genome, 'family_id', gid))
                except Exception:
                    family_id = int(gid)
            state = registry.get(family_id)
            sig = signature_map.get(gid)
            if state is None:
                phase = float(self.rng.random())
                state = {
                    'phase': phase,
                    'stasis': 0.0,
                    'signature': sig,
                    'pressure': 0.0,
                    'momentum': 0.0,
                    'diversity_grace': grace_init,
                    'family_id': family_id,
                }
            phase = (float(state.get('phase', 0.0)) + phase_step) % 1.0
            state['phase'] = phase
            prev_sig = state.get('signature')
            stasis = float(state.get('stasis', 0.0))
            if sig is not None and prev_sig is not None and sig != prev_sig:
                stasis = stasis * release
                state['signature'] = sig
                state['diversity_grace'] = grace_init
                release_count += 1
            elif sig is not None and prev_sig is None:
                state['signature'] = sig
                stasis += 1.0
                state['diversity_grace'] = grace_init
            else:
                stasis += 1.0
            current_fit = float(baseline_arr[idx])
            prev_baseline = float(state.get('last_baseline', current_fit))
            delta = current_fit - prev_baseline
            state['last_baseline'] = current_fit
            momentum = float(state.get('momentum', 0.0))
            momentum = momentum * momentum_decay + delta * (1.0 - momentum_decay)
            state['momentum'] = momentum
            relief_gain = 0.0
            if delta > 0.0 and fast_release > 0.0:
                relief_gain = fast_release * math.tanh(delta / span_scale_safe)
                stasis *= max(0.0, 1.0 - relief_gain)
            elif delta < 0.0 and slump_gain > 0.0:
                stasis += slump_gain * math.tanh(-delta / span_scale_safe)
            state['stasis'] = stasis
            crowd = 1
            if sig is not None:
                try:
                    crowd = int(signature_counts.get(sig, 1))
                except Exception:
                    crowd = 1
            crowd = max(1, crowd)
            div_log = math.log1p(max(0.0, crowd - 1.0))
            div_factor = (1.0 + div_weight * div_log) * max(0.1, 1.0 - unique_relief / max(1.0, float(crowd)))
            state['diversity_factor'] = div_factor
            state['diversity_crowd'] = float(crowd)
            grace_val = float(state.get('diversity_grace', 0.0))
            grace_factor = 1.0
            if grace_val > 0.0:
                grace_factor = max(0.2, 1.0 - grace_strength * min(1.0, grace_val))
                grace_val *= grace_decay
            state['diversity_grace'] = grace_val
            envelope = min(1.0, stasis / span)
            osc = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
            target = (base + rng * osc) * envelope
            if growth_weight > 0.0:
                grow = math.tanh(max(0.0, momentum) / span_scale_safe)
                target *= max(0.0, 1.0 - growth_weight * grow)
            if slump_gain > 0.0:
                slump = math.tanh(max(0.0, -momentum) / span_scale_safe)
                target *= 1.0 + slump_gain * slump
            if relief_gain > 0.0:
                target *= max(0.0, 1.0 - relief_gain)
            target *= div_factor
            target *= grace_factor
            target *= noise_factor
            info = family_metrics.get(family_id) if family_metrics else None
            members = family_members.get(family_id, tuple())
            if info:
                family_size = max(1, int(info.get('size', len(members) or 1)))
                family_share = float(info.get('share', family_size / float(pop_size)))
                family_surplus = float(max(0.0, info.get('surplus', family_share - family_target_share)))
                family_median = float(info.get('median', baseline_arr[idx]))
                family_spread = float(max(0.0, info.get('spread', 0.0)))
            else:
                family_size = max(1, int(family_counts.get(family_id, len(members) or 1)))
                family_share = float(family_size) / float(pop_size)
                family_surplus = max(0.0, family_share - family_target_share)
                if members:
                    try:
                        members_arr = np.asarray(members, dtype=np.int64)
                        local_scores = baseline_arr[members_arr]
                    except Exception:
                        local_scores = np.asarray([baseline_arr[m] for m in members], dtype=np.float64)
                    family_median = float(np.median(local_scores)) if local_scores.size else float(baseline_arr[idx])
                    family_spread = float(np.std(local_scores)) if local_scores.size else 0.0
                else:
                    family_median = float(baseline_arr[idx])
                    family_spread = 0.0
            prev_share = float(state.get('family_share', family_share))
            share_delta = float(family_share - prev_share)
            surplus_ratio = family_surplus / max(family_target_share, 1e-9) if family_target_share > 0.0 else 0.0
            family_surplus_total += surplus_ratio
            if surplus_ratio > family_surplus_max:
                family_surplus_max = surplus_ratio
            family_spread_total += family_spread
            family_trend_total += share_delta
            state['family_size'] = float(family_size)
            state['family_share'] = float(family_share)
            state['family_surplus'] = float(family_surplus)
            state['family_median'] = float(family_median)
            state['family_spread'] = float(family_spread)
            state['family_share_delta'] = float(share_delta)
            state['family_surplus_ratio'] = float(surplus_ratio)
            family_factor = 1.0
            if family_weight > 0.0:
                share_factor = 1.0 + family_weight * min(3.5, max(0.0, surplus_ratio))
                trend_factor = float(np.clip(1.0 + 0.5 * family_weight * share_delta * max(1, len(family_counts)), 0.5, 1.8))
                median_factor = 1.0 + 0.45 * family_weight * max(0.0, (family_median - median) / span_scale_safe)
                if span_scale_safe > 0.0:
                    spread_norm = float(np.clip(1.0 - min(1.0, family_spread / max(span_scale_safe, 1e-9)), 0.0, 1.0))
                else:
                    spread_norm = 0.0
                spread_factor = 1.0 + 0.25 * family_weight * spread_norm
                family_factor = float(np.clip(share_factor * trend_factor * median_factor * spread_factor, 1.0, 6.0))
            state['family_factor'] = family_factor
            family_factor_total += family_factor
            if family_factor > family_factor_max:
                family_factor_max = family_factor
            target *= family_factor
            pressure_prev = float(state.get('pressure', 0.0))
            pressure = pressure_prev * (1.0 - smoothing) + target * smoothing
            state['pressure'] = pressure
            penalty = min(cap * span_scale, pressure * span_scale)
            state['family_spill'] = 0.0
            if penalty > 0.0 and math.isfinite(penalty):
                adjusted[idx] = float(adjusted[idx] - penalty)
                total_penalty += penalty
                if penalty > max_penalty:
                    max_penalty = penalty
                state['last_penalty'] = penalty
                members = family_members.get(family_id, [])
                spill_total = 0.0
                if members and len(members) > 1:
                    others = [m for m in members if m != idx]
                    if others:
                        spill_total = float(penalty * min(0.35, 0.18 * math.log1p(len(others))))
                        if spill_total > 0.0:
                            per_other = spill_total / len(others)
                            for other_idx in others:
                                if 0 <= other_idx < len(adjusted):
                                    adjusted[other_idx] = float(adjusted[other_idx] - per_other)
                            spill_penalty_total += spill_total
                state['family_spill'] = float(spill_total)
            state['last_seen'] = int(generation)
            state['last_gid'] = int(gid)
            registry[family_id] = state
            seen_families.add(family_id)
            elite_seen += 1
            relief_total += relief_gain
            momentum_total += momentum
            if abs(momentum) > momentum_max:
                momentum_max = abs(momentum)
            diversity_total += float(div_factor)
            if div_factor > diversity_max:
                diversity_max = float(div_factor)
            grace_total += float(grace_factor)
        if registry:
            to_remove = []
            for fid, state in list(registry.items()):
                if int(fid) in seen_families:
                    continue
                state['stasis'] = float(state.get('stasis', 0.0)) * decay
                state['pressure'] = float(state.get('pressure', 0.0)) * decay
                state['momentum'] = float(state.get('momentum', 0.0)) * decay
                state['diversity_grace'] = float(state.get('diversity_grace', 0.0)) * grace_decay
                if state.get('stasis', 0.0) < 0.05:
                    to_remove.append(fid)
                else:
                    registry[fid] = state
            for fid in to_remove:
                registry.pop(fid, None)
            limit = int(getattr(self, 'monodromy_family_limit', max(self.pop_size * 6, 1536)))
            if len(registry) > limit:
                ordered = sorted(registry.items(), key=lambda kv: kv[1].get('last_seen', -1))
                for old_fid, _ in ordered[:-limit]:
                    if int(old_fid) in seen_families:
                        continue
                    registry.pop(old_fid, None)
        active = elite_seen
        total_penalty_all = total_penalty + spill_penalty_total
        mean_penalty = float(total_penalty_all / max(1, active)) if active else 0.0
        relief_mean = float(relief_total / max(1, active)) if active else 0.0
        momentum_mean = float(momentum_total / max(1, active)) if active else 0.0
        family_surplus_mean = float(family_surplus_total / max(1, active)) if active else 0.0
        family_spread_mean = float(family_spread_total / max(1, active)) if active else 0.0
        family_trend_mean = float(family_trend_total / max(1, active)) if active else 0.0
        self._monodromy_snapshot = {
            'pressure_mean': mean_penalty,
            'pressure_max': float(max_penalty),
            'active': int(active),
            'families': int(len(seen_families)),
            'release': int(release_count),
            'relief_mean': relief_mean,
            'momentum_mean': momentum_mean,
            'momentum_max': float(momentum_max),
            'diversity_mean': float(diversity_total / max(1, active)) if active else 0.0,
            'diversity_max': float(diversity_max),
            'grace_mean': float(grace_total / max(1, active)) if active else 0.0,
            'noise_factor': float(noise_factor),
            'noise_kind': style.get('symbol', noise_kind or ''),
            'noise_focus': float(noise_focus),
            'noise_entropy': float(noise_entropy),
            'family_factor_mean': float(family_factor_total / max(1, active)) if active else 1.0,
            'family_factor_max': float(family_factor_max),
            'family_surplus_mean': float(family_surplus_mean),
            'family_surplus_max': float(family_surplus_max),
            'family_spread_mean': float(family_spread_mean),
            'family_share_delta_mean': float(family_trend_mean),
            'family_target_share': float(family_target_share),
        }
        return adjusted

    def evolve(self, fitness_fn: Callable[[Genome], float], n_generations=100, target_fitness=None, verbose=True, env_schedule=None):
        history = []
        best_ever = None
        best_ever_fit = -1000000000.0
        top3_best = []
        from math import isnan
        scars = None
        prev_best = None
        start_gen = int(getattr(self, 'generation', 0))
        for step in range(n_generations):
            gen = start_gen + step
            self.generation = gen
            try:
                prev = history[-1] if history else (None, None)
                if env_schedule is not None:
                    env = env_schedule(gen, {'gen': gen, 'prev_best': prev[0] if prev else None, 'prev_avg': prev[1] if prev else None})
                elif getattr(self, 'auto_curriculum', True):
                    env = self._auto_env_schedule(gen, history)
                else:
                    env = None
                if env is not None:
                    self.env.update({k: v for k, v in env.items() if k not in {'enable_regen'}})
                    if 'enable_regen' in env:
                        flag = bool(env['enable_regen'])
                        self.mode.enable_regen_reproduction = flag
                        if flag:
                            self.mix_asexual_base = max(self.mix_asexual_base, 0.3)
                self.env_history.append({'gen': gen, **self.env, 'regen_enabled': self.mode.enable_regen_reproduction})
                diff = float(self.env.get('difficulty', 0.0))
                self.pollen_flow_rate = float(min(0.5, max(0.1, 0.1 + 0.35 * diff)))
                if hasattr(fitness_fn, 'set_noise_std'):
                    try:
                        fitness_fn.set_noise_std(float(self.env.get('noise_std', 0.0)))
                    except Exception:
                        pass
                if hasattr(fitness_fn, 'collective_signal'):
                    try:
                        fitness_fn.collective_signal = dict(getattr(self, '_collective_signal', {}))
                    except Exception:
                        fitness_fn.collective_signal = None
                raw = self._evaluate_population(fitness_fn)
                adjustments: Dict[int, float] = {}
                manager = getattr(self, '_households', None)
                if manager is not None:
                    try:
                        manager.update(self.population, raw, float(self.env.get('difficulty', 0.0)))
                        adjustments = manager.environment_adjustments(self.population, raw, self.env)
                    except Exception as _hm_err:
                        if getattr(self, 'debug_households', False):
                            print(f"[WARN] household manager failed: {_hm_err}")
                        adjustments = {}
                self._last_household_adjustments = dict(adjustments)
                pop_total = len(self.population)
                if adjustments:
                    raw = [float(r + adjustments.get(g.id, 0.0)) for g, r in zip(self.population, raw)]
                signature_map: Dict[int, Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[int, int, int], ...]]] = {}
                signature_counts: Dict[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[int, int, int], ...]], int] = {}
                family_counts: Dict[int, int] = {}
                family_members: Dict[int, List[int]] = defaultdict(list)
                complexity_stats: Dict[int, Tuple[int, int, float, float, float]] = {}
                complexity_scores: List[float] = []
                max_complexity_score = 0.0
                for idx, g in enumerate(self.population):
                    try:
                        sig = g.structural_signature()
                        signature_map[g.id] = sig
                        signature_counts[sig] = signature_counts.get(sig, 0) + 1
                    except Exception:
                        pass
                    try:
                        fid_raw = getattr(g, 'family_id', None)
                        if fid_raw is None:
                            parents = getattr(g, 'parents', (None, None))
                            fid_raw = parents[0] if parents and parents[0] is not None else g.id
                        fid = int(fid_raw)
                    except Exception:
                        fid = int(getattr(g, 'id', 0))
                    family_counts[fid] = family_counts.get(fid, 0) + 1
                    family_members[fid].append(idx)
                    try:
                        stats = g.structural_complexity_stats()
                        score = float(stats[-1])
                        complexity_stats[g.id] = stats
                        complexity_scores.append(score)
                        if score > max_complexity_score:
                            max_complexity_score = float(score)
                    except Exception:
                        pass
                base_div_bonus = float(getattr(self, 'structure_diversity_bonus', 0.0))
                base_div_power = float(getattr(self, 'structure_diversity_power', 1.0))
                diversity_counts = np.asarray(list(signature_counts.values()), dtype=np.float64) if signature_counts else np.zeros(0, dtype=np.float64)
                if diversity_counts.size:
                    freq = diversity_counts / max(1.0, diversity_counts.sum())
                    raw_entropy = float(-(freq * np.log(freq + 1e-12)).sum())
                    max_entropy = float(np.log(max(1.0, diversity_counts.size)))
                    diversity_entropy = raw_entropy / max(max_entropy, 1e-12) if max_entropy > 0 else 0.0
                else:
                    diversity_entropy = 0.0
                diversity_entropy = float(np.clip(diversity_entropy, 0.0, 1.0))
                diversity_scarcity = float(np.clip(1.0 - diversity_entropy, 0.0, 1.0))
                family_entropy = 0.0
                top_family_share = 0.0
                family_surplus_ratio_max = 0.0
                family_surplus_ratio_mean = 0.0
                family_count = len(family_counts)
                if family_count:
                    fam_arr = np.asarray(list(family_counts.values()), dtype=np.float64)
                    fam_freq = fam_arr / max(1.0, fam_arr.sum())
                    fam_entropy_raw = float(-(fam_freq * np.log(fam_freq + 1e-12)).sum())
                    fam_entropy_max = float(np.log(max(1.0, fam_arr.size)))
                    family_entropy = fam_entropy_raw / max(fam_entropy_max, 1e-12) if fam_entropy_max > 0 else 0.0
                    top_family_share = float(fam_freq.max())
                    if pop_total > 0:
                        target_share = 1.0 / float(family_count)
                        ratios = [
                            float(max(0.0, (count / float(pop_total)) - target_share)) / max(target_share, 1e-9)
                            for count in family_counts.values()
                        ]
                        if ratios:
                            family_surplus_ratio_max = float(max(ratios))
                            family_surplus_ratio_mean = float(sum(ratios) / len(ratios))
                family_entropy = float(np.clip(family_entropy, 0.0, 1.0))
                complexity_arr = np.asarray(complexity_scores, dtype=np.float64) if complexity_scores else np.zeros(0, dtype=np.float64)
                complexity_mean = float(complexity_arr.mean()) if complexity_arr.size else 0.0
                complexity_std = float(complexity_arr.std()) if complexity_arr.size else 0.0
                structural_spread = float(complexity_std / (abs(complexity_mean) + 1e-9)) if complexity_mean else 0.0
                env_noise = float(self.env.get('noise_std', 0.0))
                env_focus = float(self.env.get('noise_focus', 0.0) or 0.0)
                env_entropy = float(self.env.get('noise_entropy', 0.0) or 0.0)
                lazy_payload = getattr(self, '_lazy_env_feedback', {}) or {}
                lazy_share = float(lazy_payload.get('share', 0.0)) if isinstance(lazy_payload, dict) else 0.0
                adaptive_multiplier = 1.0 + 0.65 * diversity_scarcity + 0.25 * env_noise + 0.18 * env_focus + 0.35 * structural_spread
                adaptive_multiplier = float(np.clip(adaptive_multiplier, 0.5, 3.5))
                div_bonus_scale = float(base_div_bonus * adaptive_multiplier)
                div_power = float(np.clip(base_div_power * (1.0 + 0.5 * diversity_scarcity), 1.0, 3.5))
                household_pressure = float(self._household_pressure())
                diversity_snapshot = {
                    'gen': int(gen),
                    'entropy': float(diversity_entropy),
                    'scarcity': float(diversity_scarcity),
                    'complexity_mean': float(complexity_mean),
                    'complexity_std': float(complexity_std),
                    'structural_spread': float(structural_spread),
                    'diversity_bonus': float(div_bonus_scale),
                    'diversity_power': float(div_power),
                    'env_noise': float(env_noise),
                    'env_focus': float(env_focus),
                    'env_entropy': float(env_entropy),
                    'lazy_share': float(lazy_share),
                    'unique_signatures': int(len(signature_counts) or 0),
                    'family_entropy': float(family_entropy),
                    'top_family_share': float(top_family_share),
                    'family_count': int(family_count),
                    'family_surplus_ratio_max': float(family_surplus_ratio_max),
                    'family_surplus_ratio_mean': float(family_surplus_ratio_mean),
                    'household_pressure': float(household_pressure),
                }
                self._update_collective_signal(diversity_entropy, diversity_scarcity, family_surplus_ratio_mean, gen)
                self._diversity_snapshot = diversity_snapshot
                self.diversity_history.append(diversity_snapshot)
                if len(self.diversity_history) > int(getattr(self, 'diversity_history_limit', 4096)):
                    self.diversity_history = self.diversity_history[-int(self.diversity_history_limit):]
                controller = getattr(self, 'spinor_controller', None)
                auto_state = None
                if getattr(self, 'auto_complexity_controls', False):
                    try:
                        auto_state = self._auto_tune_complexity_controls(complexity_scores, max_complexity_score)
                    except Exception:
                        auto_state = getattr(self, '_auto_complexity_bonus_state', None)
                comp_bonus_scale = float(getattr(self, 'complexity_survivor_bonus', 0.0))
                comp_exp = float(getattr(self, 'complexity_survivor_exponent', 1.0))
                comp_cap = float(getattr(self, 'complexity_survivor_cap', 1.6))
                comp_bonus_limit = float(max(0.0, getattr(self, 'complexity_survivor_bonus_limit', 0.0)))
                baseline_q = float(getattr(self, 'complexity_bonus_baseline_quantile', 0.5))
                span_q = float(getattr(self, 'complexity_bonus_span_quantile', max(baseline_q + 0.1, 0.75)))
                if not math.isfinite(baseline_q) or baseline_q <= 0.0 or baseline_q >= 1.0:
                    comp_baseline = float(np.median(complexity_scores)) if complexity_scores else 0.0
                else:
                    try:
                        comp_baseline = float(np.quantile(complexity_scores, baseline_q)) if complexity_scores else 0.0
                    except Exception:
                        comp_baseline = float(np.median(complexity_scores)) if complexity_scores else 0.0
                comp_span_value = float(comp_baseline)
                if complexity_scores:
                    try:
                        span_q = float(np.clip(span_q, max(baseline_q + 0.01, 0.51), 0.995))
                        comp_span_value = float(np.quantile(complexity_scores, span_q))
                    except Exception:
                        comp_span_value = float(max_complexity_score)
                comp_span_value = float(max(comp_span_value, comp_baseline))
                if auto_state:
                    comp_bonus_scale = float(auto_state.get('bonus_scale', comp_bonus_scale))
                    comp_exp = float(auto_state.get('exponent', comp_exp))
                    comp_cap = float(auto_state.get('cap', comp_cap))
                    comp_baseline = float(auto_state.get('baseline', comp_baseline))
                    baseline_q = float(auto_state.get('baseline_quantile', baseline_q))
                    span_q = float(auto_state.get('span_quantile', span_q))
                    comp_span_value = float(max(comp_baseline, auto_state.get('span_value', comp_span_value)))
                    comp_bonus_limit = float(auto_state.get('bonus_limit', comp_bonus_limit))
                span_q = float(np.clip(span_q, min(0.995, max(baseline_q + 0.005, 0.5)), 0.995))
                comp_span_gap = float(max(1e-9, comp_span_value - comp_baseline))
                max_gap = float(max(1e-9, max_complexity_score - comp_baseline))
                bonus_denom = float(max(comp_span_gap, 0.25 * max_gap))
                comp_bonus_limit = float(max(0.0, comp_bonus_limit))
                diversity_snapshot.update({
                    'complexity_baseline': float(comp_baseline),
                    'complexity_span': float(comp_span_value),
                    'complexity_span_quantile': float(span_q),
                    'complexity_max': float(max_complexity_score),
                    'complexity_bonus_limit': float(comp_bonus_limit),
                })
                if complexity_scores:
                    comp_distribution_snapshot = {
                        'gen': int(gen),
                        'baseline': float(comp_baseline),
                        'baseline_quantile': float(baseline_q),
                        'span_quantile': float(span_q),
                        'span_value': float(comp_span_value),
                        'max': float(max_complexity_score),
                        'mean': float(complexity_mean),
                        'std': float(complexity_std),
                        'count': int(len(complexity_scores)),
                    }
                    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
                        try:
                            comp_distribution_snapshot[f'q{int(q * 100):02d}'] = float(np.quantile(complexity_scores, q))
                        except Exception:
                            continue
                    self.complexity_distribution_history.append(comp_distribution_snapshot)
                    if len(self.complexity_distribution_history) > int(getattr(self, 'complexity_history_limit', 4096)):
                        limit = int(getattr(self, 'complexity_history_limit', 4096))
                        self.complexity_distribution_history = self.complexity_distribution_history[-limit:]
                if auto_state:
                    auto_state['span_value'] = float(comp_span_value)
                base_state = getattr(self, '_auto_complexity_bonus_state', None)
                if isinstance(base_state, dict):
                    base_state['span_value'] = float(comp_span_value)
                fitnesses = []
                for g, f in zip(self.population, raw):
                    f2 = float(f)
                    if not self.mode.vanilla:
                        f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * getattr(g, 'hybrid_scale', 1.0)
                        if g.regen:
                            f2 += self.regen_bonus
                    f2 -= self._complexity_penalty(g)
                    if div_bonus_scale > 0.0:
                        sig = signature_map.get(g.id)
                        if sig is not None:
                            freq = float(signature_counts.get(sig, 1))
                            rarity = 1.0 / max(1.0, freq)
                            if div_power != 1.0:
                                rarity = float(rarity ** div_power)
                            f2 += div_bonus_scale * rarity
                    if (
                        comp_bonus_scale > 0.0
                        and max_complexity_score > 0.0
                        and max_complexity_score > comp_baseline
                    ):
                        stats = complexity_stats.get(g.id)
                        if stats is not None:
                            raw_score = float(stats[-1])
                            if raw_score > comp_baseline:
                                rel = (raw_score - comp_baseline) / bonus_denom
                                rel = max(0.0, min(comp_cap, rel))
                                if comp_exp != 1.0:
                                    rel = float(rel ** comp_exp)
                                bonus = comp_bonus_scale * rel
                                if comp_bonus_limit > 0.0:
                                    bonus = float(min(comp_bonus_limit, bonus))
                                f2 += bonus
                    if not np.isfinite(f2):
                        f2 = float(np.nan_to_num(f2, nan=-1000000.0, posinf=-1000000.0, neginf=-1000000.0))
                    fitnesses.append(f2)
                baseline_fitnesses = list(fitnesses)
                self._apply_selfish_leader_guard(fitnesses, baseline_fitnesses, gen)
                family_metrics: Dict[int, Dict[str, Any]] = {}
                if family_members:
                    try:
                        baseline_arr = np.asarray(baseline_fitnesses, dtype=np.float64)
                    except Exception:
                        baseline_arr = np.array([float(x) for x in baseline_fitnesses], dtype=np.float64)
                    target_share = 1.0 / float(family_count) if family_count else 0.0
                    for fid, members in family_members.items():
                        if not members:
                            continue
                        members_arr = np.asarray(members, dtype=np.int64)
                        try:
                            local_scores = baseline_arr[members_arr]
                        except Exception:
                            local_scores = np.asarray([baseline_fitnesses[i] for i in members], dtype=np.float64)
                        if local_scores.size == 0:
                            continue
                        best_local = int(np.argmax(local_scores))
                        best_idx = int(members_arr[best_local]) if best_local < members_arr.size else int(members[0])
                        share = float(len(members)) / float(max(1, pop_total))
                        surplus = max(0.0, share - target_share)
                        family_metrics[int(fid)] = {
                            'members': tuple(int(i) for i in members),
                            'size': int(len(members)),
                            'share': float(share),
                            'surplus': float(surplus),
                            'best_idx': int(best_idx),
                            'best_score': float(local_scores[best_local]),
                            'median': float(np.median(local_scores)),
                            'spread': float(np.std(local_scores)),
                        }
                if controller is not None and hasattr(controller, 'ingest_diversity_metrics'):
                    try:
                        controller.ingest_diversity_metrics(gen, diversity_snapshot)
                    except Exception:
                        pass
                try:
                    fitnesses = self._adaptive_refine_fitness(fitnesses, fitness_fn)
                except Exception:
                    pass
                try:
                    fitnesses = self._apply_monodromy_pressure(
                        fitnesses,
                        baseline_fitnesses,
                        signature_map,
                        gen,
                        signature_counts=signature_counts,
                        family_counts=family_counts,
                        family_metrics=family_metrics,
                    )
                except Exception as _mono_err:
                    if getattr(self, 'debug_monodromy', False):
                        print('[WARN] monodromy pressure skipped:', _mono_err)
                best_idx = int(np.argmax(fitnesses))
                best_fit = float(fitnesses[best_idx])
                avg_fit = float(np.mean(fitnesses))
                raw_best = float(np.max(baseline_fitnesses)) if baseline_fitnesses else best_fit
                raw_avg = float(np.mean(baseline_fitnesses)) if baseline_fitnesses else avg_fit
                self.raw_best_history.append((raw_best, raw_avg))
                self._update_lazy_feedback(gen, fitnesses, best_idx, best_fit, avg_fit)
                self._imprint_population_altruism(fitnesses)
                context_best = self._contextual_best_axis(best_fit, avg_fit)
                history.append((context_best, avg_fit))
                self.context_best_history.append((context_best, avg_fit))

                def genome_complexity(g):
                    n_hidden = sum((1 for n in g.nodes.values() if n.type == 'hidden'))
                    n_edges = sum((1 for c in g.connections.values() if c.enabled))
                    return (n_hidden, n_edges)
                sorted_indices = np.argsort(fitnesses)[::-1]
                pool_size = getattr(self, 'top3_candidate_pool_size', 10)
                node_threshold = getattr(self, 'top3_diversity_node_threshold', 1)
                edge_threshold = getattr(self, 'top3_diversity_edge_threshold', 2)
                for idx in sorted_indices[:pool_size]:
                    candidate = self.population[idx].copy()
                    candidate_fit = float(fitnesses[idx])
                    n_hidden_cand, n_edges_cand = genome_complexity(candidate)
                    is_diverse = True
                    for existing_genome, _, _ in top3_best:
                        n_hidden_exist, n_edges_exist = genome_complexity(existing_genome)
                        if abs(n_hidden_cand - n_hidden_exist) <= node_threshold and abs(n_edges_cand - n_edges_exist) <= edge_threshold:
                            is_diverse = False
                            break
                    if len(top3_best) < 3:
                        top3_best.append((candidate, candidate_fit, gen))
                    elif is_diverse:
                        top3_best.append((candidate, candidate_fit, gen))
                        top3_best.sort(key=lambda x: x[1], reverse=True)
                        top3_best = top3_best[:3]
                    if len(top3_best) >= 3 and (not is_diverse):
                        break
                self._last_top3_ids = [g.id for g, _fit, _gen in top3_best]
                self._last_top3_complexities = [genome_complexity(g) for g, _fit, _gen in top3_best]
                try:
                    curr_best = self.population[best_idx].copy()
                    scars = diff_scars(prev_best, curr_best, scars, birth_gen=gen, regen_mode_for_new=getattr(curr_best, 'regen_mode', 'split'))
                    stride = int(getattr(self, 'snapshot_stride', 1))
                    if gen % max(1, stride) == 0 or gen == n_generations - 1:
                        if len(self.snapshots_genomes) >= int(getattr(self, 'snapshot_max', 320)):
                            self.snapshots_genomes.pop(0)
                            self.snapshots_scars.pop(0)
                        self.snapshots_genomes.append(curr_best)
                        self.snapshots_scars.append(scars)
                    prev_best = curr_best
                except Exception:
                    pass
                self.best_ids.append(self.population[best_idx].id)
                try:
                    self.hidden_counts_history.append([sum((1 for n in g.nodes.values() if n.type == 'hidden')) for g in self.population])
                    self.edge_counts_history.append([sum((1 for c in g.connections.values() if c.enabled)) for g in self.population])
                except Exception:
                    self.hidden_counts_history.append([])
                    self.edge_counts_history.append([])
                if verbose:
                    noise = float(self.env.get('noise_std', 0.0))
                    ev = self.event_log[-1] if self.event_log else {'sexual_within': 0, 'sexual_cross': 0, 'asexual_regen': 0}
                    n_herm = sum((1 for g in self.population if g.sex == 'hermaphrodite'))
                    herm_str = f' | herm {n_herm}' if n_herm > 0 else ''
                    top3_str = ''
                    if len(top3_best) >= 3:
                        complexities = [(sum((1 for n in g.nodes.values() if n.type == 'hidden')), sum((1 for c in g.connections.values() if c.enabled))) for g, _, _ in top3_best]
                        top3_str = f' | top3: [{complexities[0][0]}n,{complexities[0][1]}e] [{complexities[1][0]}n,{complexities[1][1]}e] [{complexities[2][0]}n,{complexities[2][1]}e]'
                    mono = getattr(self, '_monodromy_snapshot', None)
                    mono_str = ''
                    if mono and mono.get('active'):
                        mono_str = (
                            f" | mono {mono.get('pressure_mean', 0.0):.3f}/{mono.get('pressure_max', 0.0):.3f}"
                            f"~{int(mono.get('release', 0))}"
                            f" Δ{mono.get('relief_mean', 0.0):.3f} μ{mono.get('momentum_mean', 0.0):.3f}"
                            f" div{mono.get('diversity_mean', 0.0):.2f} gr{mono.get('grace_mean', 0.0):.2f}"
                            f" nf{mono.get('noise_factor', 1.0):.2f} fam{mono.get('family_factor_mean', 1.0):.2f}@{int(mono.get('families', 0))}"
                        )
                        nk = mono.get('noise_kind')
                        if nk:
                            mono_str = f"{mono_str} {nk}"
                    div_str = ''
                    div_snap = getattr(self, '_diversity_snapshot', None)
                    if isinstance(div_snap, dict) and div_snap:
                        try:
                            div_str = (
                                f" | div H{float(div_snap.get('entropy', 0.0)):.2f}"
                                f" sc{float(div_snap.get('scarcity', 0.0)):.2f}"
                                f" κ{float(div_snap.get('structural_spread', 0.0)):.2f}"
                            )
                            fam_count = int(div_snap.get('family_count', 0) or 0)
                            if fam_count:
                                div_str += f" fam{float(div_snap.get('top_family_share', 0.0)):.2f}@{fam_count}"
                            div_str += f" hh{float(div_snap.get('household_pressure', 0.0)):.2f}"
                        except Exception:
                            div_str = ''
                    print(f"Gen {gen:3d} | best {best_fit:.4f} | axis {context_best:.4f} | avg {avg_fit:.4f} | difficulty {diff:.2f} | noise {noise:.2f} | sexual {ev.get('sexual_within', 0) + ev.get('sexual_cross', 0)} | regen {ev.get('asexual_regen', 0)}{herm_str}{top3_str}{mono_str}{div_str}")
                if context_best > best_ever_fit:
                    best_ever_fit = context_best
                    best_ever = self.population[best_idx].copy()
                if target_fitness is not None and best_fit >= target_fitness:
                    self._resilience_eval_guard = 0
                    self._resilience_history = list(history)
                    break
                species = self.speciate(fitnesses)
                try:
                    self._learn_species_target(len(species), context_best, gen)
                except Exception as _spe:
                    print('[WARN] species target learning skipped:', _spe)
                self._adapt_compat_threshold(len(species))
                self.reproduce(species, fitnesses)
                self._trim_runtime_caches(gen)
                self._resilience_eval_guard = 0
                self._resilience_history = list(history)
            except Exception as gen_err:
                _tb = traceback.format_exc()
                try:
                    self._resilience_failures.append({'gen': int(gen), 'error': repr(gen_err), 'traceback': _tb})
                except Exception:
                    pass
                self._resilience_eval_guard = getattr(self, '_resilience_eval_guard', 0) + 1
                if self._resilience_eval_guard == 1:
                    print(f"[WARN] Generation {gen} failed with {gen_err!r}; switching to single-thread fallback and continuing.")
                    self.parallel_backend = 'thread'
                    self.eval_workers = 1
                    self.pool_keepalive = 0
                elif self._resilience_eval_guard == 2:
                    print('[WARN] Multiple generation failures detected; disabling regenerative reproduction for stability.')
                    self.mode.enable_regen_reproduction = False
                else:
                    print(f"[WARN] Generation {gen} failed again ({gen_err!r}); continuing with resilience guard (attempt {self._resilience_eval_guard}).")
                try:
                    self._close_pool()
                except Exception:
                    pass
                self._resilience_history = list(history)
                continue
        if best_ever is None and self.population:
            best_ever = self.population[0].copy()
        self.top3_best_topologies = top3_best
        self._resilience_history = list(history)
        try:
            self._close_pool()
        except Exception:
            pass
        self._best_context_value = float(best_ever_fit)
        return (best_ever, history)

def act_forward(name, x):
    if name == 'tanh':
        return np.tanh(x)
    if name == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-x))
    if name == 'relu':
        return np.maximum(0.0, x)
    if name == 'identity':
        return x
    return np.tanh(x)

def act_deriv(name, x):
    if name == 'tanh':
        y = np.tanh(x)
        return 1.0 - y * y
    if name == 'sigmoid':
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s)
    if name == 'relu':
        return (x > 0.0).astype(x.dtype)
    if name == 'identity':
        return np.ones_like(x)
    y = np.tanh(x)
    return 1.0 - y * y

def compile_genome(g: Genome):
    global _COMPILE_TICK
    rev = (getattr(g, '_structure_rev', -1), getattr(g, '_weights_rev', -1))
    cached = getattr(g, '_compiled_cache', None)
    cached_rev = getattr(g, '_compiled_cache_rev', (-1, -1))
    _COMPILE_TICK += 1
    if cached is not None and cached_rev == rev:
        try:
            g._compiled_cache_tick = _COMPILE_TICK
        except Exception:
            pass
        return cached
    order = g.topological_order()
    idx_of = {nid: i for i, nid in enumerate(order)}
    type_sig = tuple(g.nodes[n].type for n in order)
    act_sig = tuple(g.nodes[n].activation for n in order)
    structure_key = _structure_cache_key(g, order)
    base = _structure_cache_get(structure_key)
    if (
        base is None
        or base.get('order_ref') != tuple(order)
        or base.get('type_sig') != type_sig
        or base.get('act_sig') != act_sig
    ):
        in_ids = [nid for nid in order if g.nodes[nid].type == 'input']
        bias_ids = [nid for nid in order if g.nodes[nid].type == 'bias']
        out_ids = [nid for nid in order if g.nodes[nid].type == 'output']
        edges = [c for c in g.enabled_connections()]
        src = np.array([idx_of[c.in_node] for c in edges], dtype=np.int32)
        dst = np.array([idx_of[c.out_node] for c in edges], dtype=np.int32)
        eid = tuple(c.innovation for c in edges)
        n = len(order)
        in_edges_lists = [[] for _ in range(n)]
        out_edges_lists = [[] for _ in range(n)]
        for e, (s, d) in enumerate(zip(src, dst)):
            in_edges_lists[d].append(e)
            out_edges_lists[s].append(e)
        if len(edges):
            in_sort = np.argsort(dst, kind='mergesort')
            out_sort = np.argsort(src, kind='mergesort')
            in_edges_flat = in_sort.astype(np.int32, copy=False)
            out_edges_flat = out_sort.astype(np.int32, copy=False)
            in_counts = np.bincount(dst[in_sort], minlength=n)
            out_counts = np.bincount(src[out_sort], minlength=n)
            in_edges_ptr = np.zeros(n + 1, dtype=np.int32)
            out_edges_ptr = np.zeros(n + 1, dtype=np.int32)
            np.cumsum(in_counts, out=in_edges_ptr[1:])
            np.cumsum(out_counts, out=out_edges_ptr[1:])
        else:
            in_edges_flat = np.zeros(0, dtype=np.int32)
            out_edges_flat = np.zeros(0, dtype=np.int32)
            in_edges_ptr = np.zeros(n + 1, dtype=np.int32)
            out_edges_ptr = np.zeros(n + 1, dtype=np.int32)
        base = {
            'order_ref': tuple(order),
            'type_sig': type_sig,
            'act_sig': act_sig,
            'inputs': tuple(idx_of[i] for i in sorted(in_ids)),
            'biases': tuple(idx_of[i] for i in bias_ids),
            'outputs': tuple(idx_of[i] for i in sorted(out_ids)),
            'src': src,
            'dst': dst,
            'eid': eid,
            'in_edges_flat': in_edges_flat,
            'out_edges_flat': out_edges_flat,
            'in_edges_ptr': in_edges_ptr,
            'out_edges_ptr': out_edges_ptr,
            'in_edges': tuple(tuple(row) for row in in_edges_lists),
            'out_edges': tuple(tuple(row) for row in out_edges_lists),
        }
        _structure_cache_store(structure_key, base)
    n = len(order)
    compiled = {
        'order': list(order),
        'idx_of': idx_of,
        'types': list(type_sig),
        'acts': list(act_sig),
        'inputs': list(base['inputs']),
        'biases': list(base['biases']),
        'outputs': list(base['outputs']),
        'src': base['src'],
        'dst': base['dst'],
        'eid': list(base['eid']),
        'in_edges': [list(row) for row in base['in_edges']],
        'out_edges': [list(row) for row in base['out_edges']],
        'in_edges_flat': base['in_edges_flat'],
        'in_edges_ptr': base['in_edges_ptr'],
        'out_edges_flat': base['out_edges_flat'],
        'out_edges_ptr': base['out_edges_ptr'],
    }
    compiled['w'] = np.array([g.connections[inn].weight for inn in compiled['eid']], dtype=np.float64)
    node_sensitivity = _node_trait_array(g, order, 'backprop_sensitivity', 1.0)
    node_jitter = _node_trait_array(g, order, 'sensitivity_jitter', 0.0, low=-0.25, high=0.25)
    node_momentum = _node_trait_array(g, order, 'sensitivity_momentum', 0.0)
    node_variance = _node_trait_array(g, order, 'sensitivity_variance', 0.0, low=0.0)
    node_altruism = _node_trait_array(g, order, 'altruism', 0.5, low=0.0, high=1.0)
    node_altruism_memory = _node_trait_array(g, order, 'altruism_memory', 0.0, low=-1.5, high=1.5)
    node_altruism_span = _node_trait_array(g, order, 'altruism_span', 0.0, low=0.0, high=4.0)
    compiled['node_sensitivity'] = node_sensitivity
    compiled['node_jitter'] = node_jitter
    compiled['node_momentum'] = node_momentum
    compiled['node_variance'] = node_variance
    compiled['node_altruism'] = node_altruism
    compiled['node_altruism_memory'] = node_altruism_memory
    compiled['node_altruism_span'] = node_altruism_span
    try:
        g._compiled_cache = compiled
        g._compiled_cache_rev = rev
        g._compiled_cache_tick = _COMPILE_TICK
    except Exception:
        pass
    return compiled

def forward_batch(comp, X, w=None):
    if w is None:
        w = comp['w']
    B = X.shape[0]
    n = len(comp['order'])
    A = np.zeros((B, n), dtype=np.float64)
    Z = np.zeros((B, n), dtype=np.float64)
    in_idx = comp['inputs']
    assert X.shape[1] == len(in_idx), 'X dim != number of input nodes'
    for k, nid in enumerate(in_idx):
        A[:, nid] = X[:, k]
    for b in comp['biases']:
        A[:, b] = 1.0
    for j in range(n):
        if comp['types'][j] in ('input', 'bias'):
            continue
        ptr = comp.get('in_edges_ptr')
        flat = comp.get('in_edges_flat')
        if ptr is not None and flat is not None and j < len(ptr) - 1:
            start = int(ptr[j])
            end = int(ptr[j + 1])
            if end > start:
                idx = flat[start:end]
                src_idx = comp['src'][idx]
                z = A[:, src_idx] @ w[idx]
            else:
                z = np.zeros(B, dtype=np.float64)
        else:
            z = np.zeros(B, dtype=np.float64)
            for e in comp['in_edges'][j]:
                z += A[:, comp['src'][e]] * w[e]
        Z[:, j] = z
        A[:, j] = act_forward(comp['acts'][j], z)
    return (A, Z)

def _softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-09)

def loss_and_output_delta(comp, Z, y, l2, w):
    out_idx = comp['outputs']
    B = Z.shape[0]
    if len(out_idx) == 1:
        z = Z[:, out_idx[0:1]]
        p = 1.0 / (1.0 + np.exp(-z))
        yv = y.reshape(B, 1).astype(np.float64)
        loss = (np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - yv * z).mean()
        delta_out = p - yv
        probs = p
    else:
        logits = Z[:, out_idx]
        probs = _softmax(logits)
        if y.ndim == 1:
            y_idx = np.asarray(y)
            if not np.issubdtype(y_idx.dtype, np.integer):
                y_idx = np.rint(y_idx).astype(np.int64, copy=False)
            else:
                y_idx = y_idx.astype(np.int64, copy=False)
            if y_idx.size:
                np.clip(y_idx, 0, len(out_idx) - 1, out=y_idx)
            y_one = np.zeros((B, len(out_idx)), dtype=np.float64)
            y_one[np.arange(B, dtype=np.int64), y_idx] = 1.0
        else:
            y_one = y.astype(np.float64)
        loss = -(y_one * np.log(probs + 1e-09)).sum(axis=1).mean()
        delta_out = probs - y_one
    loss = float(loss + 0.5 * l2 * np.sum(w * w))
    return (loss, delta_out, probs)

def backprop_step(comp, X, y, w, lr=0.01, l2=0.0001):
    """
    Hardened backprop with gradient/weight clipping and NaN guards.
    既存シグネチャ互換（追加引数は train_* から供給）。
    """
    import numpy as _np
    grad_clip = 5.0
    w_clip = 12.0
    A, Z = forward_batch(comp, X, w)
    loss, delta_out, _ = loss_and_output_delta(comp, Z, y, l2, w)
    if not _np.isfinite(loss):
        w = _np.tanh(w) * 0.1
        loss = float(_np.nan_to_num(loss, nan=1000.0, posinf=1000.0, neginf=1000.0))
    B = X.shape[0]
    n = len(comp['order'])
    grad_w = _np.zeros_like(w)
    delta_z = _np.zeros((B, n), dtype=_np.float64)
    delta_a = _np.zeros((B, n), dtype=_np.float64)
    node_scale = comp.get('node_sensitivity')
    if node_scale is None or getattr(node_scale, 'shape', (0,))[0] != n:
        node_scale = _np.ones(n, dtype=_np.float64)
    else:
        node_scale = _np.clip(_np.asarray(node_scale, dtype=_np.float64), 0.2, 5.0)
    node_jitter = comp.get('node_jitter')
    if node_jitter is None or getattr(node_jitter, 'shape', (0,))[0] != n:
        node_jitter = _np.zeros(n, dtype=_np.float64)
    else:
        node_jitter = _np.clip(_np.asarray(node_jitter, dtype=_np.float64), -0.3, 0.3)
    node_momentum = comp.get('node_momentum')
    if node_momentum is None or getattr(node_momentum, 'shape', (0,))[0] != n:
        node_momentum = _np.zeros(n, dtype=_np.float64)
    else:
        node_momentum = _np.asarray(node_momentum, dtype=_np.float64)
    node_variance = comp.get('node_variance')
    if node_variance is None or getattr(node_variance, 'shape', (0,))[0] != n:
        node_variance = _np.zeros(n, dtype=_np.float64)
    else:
        node_variance = _np.clip(_np.asarray(node_variance, dtype=_np.float64), 0.0, 5.0)
    node_altruism = comp.get('node_altruism')
    if node_altruism is None or getattr(node_altruism, 'shape', (0,))[0] != n:
        node_altruism = _np.full(n, 0.5, dtype=_np.float64)
    else:
        node_altruism = _np.clip(_np.asarray(node_altruism, dtype=_np.float64), 0.0, 1.0)
    node_altruism_memory = comp.get('node_altruism_memory')
    if node_altruism_memory is None or getattr(node_altruism_memory, 'shape', (0,))[0] != n:
        node_altruism_memory = _np.zeros(n, dtype=_np.float64)
    else:
        node_altruism_memory = _np.clip(_np.asarray(node_altruism_memory, dtype=_np.float64), -1.5, 1.5)
    node_altruism_span = comp.get('node_altruism_span')
    if node_altruism_span is None or getattr(node_altruism_span, 'shape', (0,))[0] != n:
        node_altruism_span = _np.zeros(n, dtype=_np.float64)
    else:
        node_altruism_span = _np.clip(_np.asarray(node_altruism_span, dtype=_np.float64), 0.0, 4.0)
    node_signal = _np.zeros(n, dtype=_np.float64)
    node_push = _np.zeros(n, dtype=_np.float64)
    for j, oi in enumerate(comp['outputs']):
        delta_z[:, oi] = delta_out[:, j:j + 1].reshape(B)
    for j in reversed(range(n)):
        t = comp['types'][j]
        if t in ('input', 'bias'):
            continue
        if t == 'output':
            dz_raw = delta_z[:, j]
        else:
            dz_raw = delta_a[:, j] * act_deriv(comp['acts'][j], Z[:, j])
        dest_mom = float(node_momentum[j])
        dest_var = float(node_variance[j])
        altruism_level = float(node_altruism[j])
        altruism_memory = float(node_altruism_memory[j])
        altruism_span = float(node_altruism_span[j])
        dest_scale = float(node_scale[j]) * (1.0 + 0.04 * _np.tanh(dest_mom))
        dest_jitter = 1.0 + 0.05 * float(node_jitter[j]) + 0.03 * _np.tanh(dest_var)
        social_gain = 1.0 + 0.18 * (altruism_level - 0.5) + 0.05 * altruism_memory
        social_damp = 1.0 / (1.0 + 0.25 * altruism_span)
        dest_mix = dest_scale * dest_jitter * social_gain * social_damp
        dz = dz_raw * dest_mix
        delta_z[:, j] = dz
        node_signal[j] += float(_np.mean(_np.abs(dz))) * (1.0 + 0.1 * _np.tanh(dest_var) + 0.06 * altruism_level)
        ptr = comp.get('in_edges_ptr')
        flat = comp.get('in_edges_flat')
        if ptr is not None and flat is not None and j < len(ptr) - 1:
            start = int(ptr[j])
            end = int(ptr[j + 1])
            if end > start:
                idx = flat[start:end]
                src_idx = comp['src'][idx]
                weights_local = w[idx]
                src_mom = node_momentum[src_idx]
                src_var = node_variance[src_idx]
                src_scale = node_scale[src_idx] * (1.0 + 0.04 * _np.tanh(src_mom))
                src_jitter = 1.0 + 0.05 * node_jitter[src_idx] + 0.03 * _np.tanh(src_var)
                src_mix = src_scale * src_jitter
                edge_scale = 0.5 * (dest_mix + src_mix)
                with _np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    contrib = A[:, src_idx].T @ dz
                contrib = _np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
                grad_w[idx] += edge_scale * contrib
                altruism_delta = altruism_level - node_altruism[src_idx]
                flow_bias = 1.0 + 0.15 * _np.tanh(dest_mom - src_mom) + 0.1 * altruism_delta
                mem_gain = 1.0 + 0.08 * node_altruism_memory[src_idx]
                with _np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    push_term = dz[:, None] * weights_local
                push_term = _np.nan_to_num(push_term, nan=0.0, posinf=0.0, neginf=0.0)
                node_push[src_idx] += _np.mean(_np.abs(push_term), axis=0) * edge_scale * flow_bias * mem_gain
                delta_update = push_term * (src_mix * mem_gain)
                delta_update = _np.nan_to_num(delta_update, nan=0.0, posinf=0.0, neginf=0.0)
                delta_a[:, src_idx] += delta_update
            else:
                for e in comp['in_edges'][j]:
                    s = comp['src'][e]
                    src_mom = float(node_momentum[s])
                    src_var = float(node_variance[s])
                    src_scale = float(node_scale[s]) * (1.0 + 0.04 * _np.tanh(src_mom))
                    src_jitter = 1.0 + 0.05 * float(node_jitter[s]) + 0.03 * _np.tanh(src_var)
                    src_mix = src_scale * src_jitter
                    edge_scale = 0.5 * (dest_mix + src_mix)
                    with _np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        contrib = _np.dot(A[:, s], dz)
                    contrib = float(_np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0))
                    grad_w[e] += edge_scale * contrib
                    altruism_delta = altruism_level - float(node_altruism[s])
                    flow_bias = 1.0 + 0.15 * _np.tanh(dest_mom - src_mom) + 0.1 * altruism_delta
                    mem_gain = 1.0 + 0.08 * float(node_altruism_memory[s])
                    with _np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        push_term = dz * w[e]
                    push_term = _np.nan_to_num(push_term, nan=0.0, posinf=0.0, neginf=0.0)
                    node_push[s] += float(_np.mean(_np.abs(push_term))) * edge_scale * flow_bias * mem_gain
                    delta_update = push_term * (src_mix * mem_gain)
                    delta_update = _np.nan_to_num(delta_update, nan=0.0, posinf=0.0, neginf=0.0)
                    delta_a[:, s] += delta_update
    grad_w = grad_w / max(1, B) + l2 * w
    if not _np.all(_np.isfinite(grad_w)):
        grad_w = _np.nan_to_num(grad_w, nan=0.0, posinf=0.0, neginf=0.0)
    if grad_clip and grad_clip > 0:
        gnorm = float(_np.linalg.norm(grad_w))
        if _np.isfinite(gnorm) and gnorm > grad_clip:
            grad_w *= grad_clip / (gnorm + 1e-12)
    w_new = w - float(lr) * grad_w
    if w_clip and w_clip > 0:
        _np.clip(w_new, -float(w_clip), float(w_clip), out=w_new)
    profile = node_signal + 0.5 * node_push
    profile = _np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    return (w_new, float(loss), profile)

def train_with_backprop_numpy(
    genome: Genome,
    X,
    y,
    steps=50,
    lr=0.01,
    l2=0.0001,
    grad_clip=5.0,
    w_clip=12.0,
    profile_out: Optional[Dict[str, Any]]=None,
    rng: Optional[np.random.Generator]=None,
    collective_signal: Optional[Dict[str, float]]=None,
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    np.nan_to_num(X, copy=False)
    np.nan_to_num(y, copy=False)
    collective_signal = dict(collective_signal or {})
    altruism_target = float(collective_signal.get('altruism_target', 0.5))
    solidarity = float(collective_signal.get('solidarity', 0.5))
    stress = float(collective_signal.get('stress', 0.0))
    lazy_share = float(collective_signal.get('lazy_share', 0.0))
    advantage = float(collective_signal.get('advantage', 0.0))
    comp = compile_genome(genome)
    w = comp['w'].copy()
    history = []
    node_profile_accum = np.zeros(len(comp['order']), dtype=np.float64)
    node_profile_sumsq = np.zeros(len(comp['order']), dtype=np.float64)
    step_profiles: List[np.ndarray] = [] if profile_out is not None else []
    if profile_out is not None:
        profile_out.clear()
        profile_out['node_order'] = list(comp['order'])
        profile_out['node_types'] = list(comp['types'])
        init_sens = comp['node_sensitivity'].copy()
        init_jit = comp['node_jitter'].copy()
        init_mom = comp['node_momentum'].copy()
        init_var = comp['node_variance'].copy()
        profile_out['initial_sensitivity'] = init_sens.copy()
        profile_out['initial_jitter'] = init_jit.copy()
        profile_out['initial_momentum'] = init_mom.copy()
        profile_out['initial_variance'] = init_var.copy()
        init_alt = comp['node_altruism'].copy()
        init_alt_mem = comp['node_altruism_memory'].copy()
        init_alt_span = comp['node_altruism_span'].copy()
        profile_out['initial_altruism'] = init_alt.copy()
        profile_out['initial_altruism_memory'] = init_alt_mem.copy()
        profile_out['initial_altruism_span'] = init_alt_span.copy()
        profile_out['collective_signal'] = dict(collective_signal)
    rng_local = rng or np.random.default_rng()
    if w.size == 0:
        return history
    for _ in range(int(steps)):
        w, L, profile = backprop_step(comp, X, y, w, lr=lr, l2=l2)
        if not np.isfinite(L):
            L = float(np.nan_to_num(L, nan=1000.0, posinf=1000.0, neginf=1000.0))
        history.append(L)
        if profile is not None and profile.shape[0] == node_profile_accum.shape[0]:
            node_profile_accum += profile
            node_profile_sumsq += profile * profile
            if profile_out is not None:
                step_profiles.append(np.asarray(profile, dtype=np.float64))
    for e_idx, inn in enumerate(comp['eid']):
        genome.connections[inn].weight = float(w[e_idx])
    if node_profile_accum.size:
        avg_profile = node_profile_accum / max(1, float(steps))
        mean_sq = node_profile_sumsq / max(1, float(steps))
        var_profile = np.maximum(0.0, mean_sq - avg_profile ** 2)
        global_mean = float(np.mean(avg_profile)) if avg_profile.size else 0.0
        global_rms = float(np.sqrt(np.maximum(1e-09, np.mean(var_profile)))) if var_profile.size else 0.0
        for idx, nid in enumerate(comp['order']):
            node = genome.nodes.get(nid)
            if node is None:
                continue
            baseline = float(getattr(node, 'backprop_sensitivity', 1.0))
            prev_momentum = float(getattr(node, 'sensitivity_momentum', 0.0))
            prev_variance = float(getattr(node, 'sensitivity_variance', 0.0))
            local = float(avg_profile[idx])
            centered = local - global_mean
            tone = float(np.tanh(centered / (global_rms + 1e-06))) if global_rms > 0 else float(np.tanh(centered))
            target = 1.0 + 0.24 * float(np.tanh(local)) + 0.08 * tone
            node.backprop_sensitivity = float(
                np.clip(
                    0.78 * baseline + 0.17 * target + 0.05 * (1.0 + np.tanh(prev_momentum)),
                    0.2,
                    5.0,
                )
            )
            var_local = float(np.sqrt(float(var_profile[idx]))) if idx < var_profile.size else 0.0
            node.sensitivity_momentum = float(
                np.clip(0.62 * prev_momentum + 0.38 * tone, -1.5, 1.5)
            )
            node.sensitivity_variance = float(
                np.clip(0.7 * prev_variance + 0.3 * var_local, 0.0, 3.0)
            )
            jitter_base = float(getattr(node, 'sensitivity_jitter', 0.0))
            jitter_scale = 0.02 + 0.018 * float(np.clip(node.sensitivity_variance, 0.0, 3.0))
            jitter_drive = 0.15 * node.sensitivity_momentum
            jitter_target = float(np.clip(jitter_drive + rng_local.normal(0.0, jitter_scale), -0.3, 0.3))
            node.sensitivity_jitter = float(np.clip(0.74 * jitter_base + 0.26 * jitter_target, -0.25, 0.25))
            prev_alt = float(np.clip(getattr(node, 'altruism', 0.5), 0.0, 1.0))
            prev_mem = float(np.clip(getattr(node, 'altruism_memory', 0.0), -1.5, 1.5))
            prev_span = float(np.clip(getattr(node, 'altruism_span', 0.0), 0.0, 4.0))
            solidarity_gain = 0.5 * solidarity + 0.3 * (1.0 - advantage) + 0.2 * lazy_share
            target_alt = float(np.clip(0.6 * altruism_target + 0.4 * solidarity_gain, 0.0, 1.0))
            node.altruism = float(np.clip(0.72 * prev_alt + 0.28 * target_alt, 0.0, 1.0))
            mem_target = float(np.clip(solidarity - stress, -1.5, 1.5))
            node.altruism_memory = float(np.clip(0.6 * prev_mem + 0.4 * mem_target, -1.5, 1.5))
            span_target = float(np.clip(stress + advantage, 0.0, 4.0))
            node.altruism_span = float(np.clip(0.65 * prev_span + 0.35 * span_target, 0.0, 4.0))
        if profile_out is not None:
            profile_out['avg_profile'] = np.asarray(avg_profile, dtype=np.float64)
            profile_out['profile_var'] = np.asarray(var_profile, dtype=np.float64)
            order = comp['order']
            profile_out['final_sensitivity'] = _node_trait_array(genome, order, 'backprop_sensitivity', 1.0)
            profile_out['final_jitter'] = _node_trait_array(genome, order, 'sensitivity_jitter', 0.0, low=-0.25, high=0.25)
            profile_out['final_momentum'] = _node_trait_array(genome, order, 'sensitivity_momentum', 0.0)
            profile_out['final_variance'] = _node_trait_array(genome, order, 'sensitivity_variance', 0.0, low=0.0)
            profile_out['final_altruism'] = _node_trait_array(genome, order, 'altruism', 0.5, low=0.0, high=1.0)
            profile_out['final_altruism_memory'] = _node_trait_array(genome, order, 'altruism_memory', 0.0, low=-1.5, high=1.5)
            profile_out['final_altruism_span'] = _node_trait_array(genome, order, 'altruism_span', 0.0, low=0.0, high=4.0)
            if step_profiles:
                profile_out['step_profiles'] = np.stack(step_profiles, axis=0)
            else:
                profile_out['step_profiles'] = np.zeros((0, len(comp['order'])), dtype=np.float64)
            profile_out['loss_history'] = np.asarray(history, dtype=np.float64)
    try:
        genome.invalidate_caches(weights=True)
    except Exception:
        pass
    return history

def predict_proba(genome: Genome, X):
    comp = compile_genome(genome)
    _, Z = forward_batch(comp, X, comp['w'])
    out = comp['outputs']
    if len(out) == 1:
        z = Z[:, out[0:1]]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.concatenate([1.0 - p, p], axis=1)
    else:
        logits = Z[:, out]
        return _softmax(logits)

def predict(genome: Genome, X):
    P = predict_proba(genome, X)
    return np.argmax(P, axis=1).astype(np.int32)

def complexity_penalty(genome: Genome, alpha_nodes=0.001, alpha_edges=0.0005):
    hidden = sum((1 for n in genome.nodes.values() if n.type == 'hidden'))
    edges = len(genome.enabled_connections())
    return alpha_nodes * hidden + alpha_edges * edges

def fitness_backprop_classifier(
    genome: Genome,
    Xtr,
    ytr,
    Xva,
    yva,
    steps=40,
    lr=0.005,
    l2=0.0001,
    alpha_nodes=0.001,
    alpha_edges=0.0005,
    collective_signal: Optional[Dict[str, float]]=None,
):
    try:
        gg = genome.copy()
        train_with_backprop_numpy(gg, Xtr, ytr, steps=steps, lr=lr, l2=l2, collective_signal=collective_signal)
        pred = predict(gg, Xva)
        acc = (pred == (yva if yva.ndim == 1 else np.argmax(yva, 1))).mean()
        pen = complexity_penalty(gg, alpha_nodes=alpha_nodes, alpha_edges=alpha_edges)
        return float(acc - pen)
    except RuntimeError as e:
        if 'Cycle detected' in str(e):
            return -1.0
        raise

@dataclass
class Scar:
    birth_gen: int
    mode: str = 'split'
    age: int = 0

def layout_by_depth(genome: Genome, x_gap: float=1.5, y_gap: float=1.0) -> Dict[int, Tuple[float, float]]:
    depth = genome.node_depths()
    buckets: Dict[int, List[int]] = {}
    for nid, d in depth.items():
        buckets.setdefault(d, []).append(nid)
    type_rank = {'input': 0, 'bias': 1, 'hidden': 2, 'output': 3}
    for d in buckets:
        buckets[d].sort(key=lambda nid: (type_rank.get(genome.nodes[nid].type, 9), nid))
    pos = {}
    for i, d in enumerate(sorted(buckets.keys())):
        nodes = buckets[d]
        y0 = -(len(nodes) - 1) / 2.0
        for j, nid in enumerate(nodes):
            pos[nid] = (i * x_gap, (y0 + j) * y_gap)
    return pos

def layout_by_depth_union(genomes: List[Genome], x_gap: float=1.5, y_gap: float=1.0) -> Dict[int, Tuple[float, float]]:
    depths_per: Dict[int, List[int]] = {}
    type_by: Dict[int, str] = {}
    for g in genomes:
        d = g.node_depths()
        for nid, dep in d.items():
            depths_per.setdefault(nid, []).append(dep)
            if nid not in type_by:
                type_by[nid] = g.nodes[nid].type
    depth_final = {nid: max(ds) for nid, ds in depths_per.items()}
    buckets: Dict[int, List[int]] = {}
    for nid, dep in depth_final.items():
        buckets.setdefault(dep, []).append(nid)
    type_rank = {'input': 0, 'bias': 1, 'hidden': 2, 'output': 3}
    for d in buckets:
        buckets[d].sort(key=lambda nid: (type_rank.get(type_by.get(nid, 'hidden'), 9), nid))
    pos = {}
    for i, d in enumerate(sorted(buckets.keys())):
        nodes = buckets[d]
        y0 = -(len(nodes) - 1) / 2.0
        for j, nid in enumerate(nodes):
            pos[nid] = (i * x_gap, (y0 + j) * y_gap)
    return pos

def _edge_key(c) -> Tuple[int, int]:
    return (c.in_node, c.out_node)

def edge_sets(prev_genome: Optional[Genome], curr_genome: Genome) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    prev = set((_edge_key(c) for c in (prev_genome.enabled_connections() if prev_genome else [])))
    curr = set((_edge_key(c) for c in (curr_genome.enabled_connections() if curr_genome else [])))
    return (prev & curr, curr - prev, prev - curr)

def _draw_edges(ax, genome: Genome, pos, lw=1.0, alpha=0.8):
    for c in genome.enabled_connections():
        i, o = (c.in_node, c.out_node)
        if i not in pos or o not in pos:
            continue
        p1 = pos[i]
        p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, alpha=alpha)

def _draw_edges_with_diff(ax, prev_genome: Optional[Genome], curr_genome: Genome, pos, lw_base=1.0, lw_added=2.0, lw_removed=1.2, alpha_base=0.7, alpha_added=1.0, alpha_removed=0.6, linestyle_removed=(0, (3, 3))):
    common, added, removed = edge_sets(prev_genome, curr_genome)
    for c in curr_genome.enabled_connections():
        e = _edge_key(c)
        if e in added:
            continue
        i, o = (c.in_node, c.out_node)
        if i not in pos or o not in pos:
            continue
        p1 = pos[i]
        p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw_base, alpha=alpha_base)
    for c in curr_genome.enabled_connections():
        e = _edge_key(c)
        if e not in added:
            continue
        i, o = (c.in_node, c.out_node)
        if i not in pos or o not in pos:
            continue
        p1 = pos[i]
        p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw_added, alpha=alpha_added)
    if prev_genome is not None:
        for c in prev_genome.enabled_connections():
            e = _edge_key(c)
            if e not in removed:
                continue
            i, o = (c.in_node, c.out_node)
            if i not in pos or o not in pos:
                continue
            p1 = pos[i]
            p2 = pos[o]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw_removed, alpha=alpha_removed, linestyle=linestyle_removed)

def _mode_lw_factor(mode: str) -> float:
    if mode == 'head':
        return 1.2
    if mode == 'tail':
        return 1.0
    if mode == 'split':
        return 1.4
    return 1.0

def _draw_nodes(ax, genome: Genome, pos, scars: Optional[Dict[int, 'Scar']]=None, pulse_t: float=0.0, decay_horizon: float=8.0, radius: float=0.1, annotate_type=True, show_mode_mark=True):
    for nid, nd in genome.nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        circ = Circle((x, y), radius=radius, fill=True, alpha=0.9, linewidth=0.0)
        ax.add_patch(circ)
        outline_lw = 1.0
        outline_alpha = 0.9
        mode_char = None
        if scars and nid in scars:
            age = scars[nid].age
            mode_char = scars[nid].mode[:1].upper() if scars[nid].mode else None
            amp = max(0.1, 1.0 - age / float(decay_horizon)) if decay_horizon and decay_horizon > 0 else 1.0
            pulse = 1.0 + 0.25 * math.sin(2 * math.pi * pulse_t) * amp
            outline_lw = 2.0 * pulse * _mode_lw_factor(scars[nid].mode)
            circ2 = Circle((x, y), radius=radius * (1.0 + 0.15 * pulse), fill=False, linewidth=outline_lw, alpha=outline_alpha)
            ax.add_patch(circ2)
            ax.text(x, y + radius * 1.6, f'{age}', ha='center', va='bottom', fontsize=8, alpha=0.9)
        else:
            circ2 = Circle((x, y), radius=radius, fill=False, linewidth=outline_lw, alpha=outline_alpha)
            ax.add_patch(circ2)
        if annotate_type:
            ax.text(x, y - radius * 1.6, f'{nd.type[0]}', ha='center', va='top', fontsize=7, alpha=0.8)
        if show_mode_mark and mode_char is not None:
            ax.text(x + radius * 1.2, y, mode_char, ha='left', va='center', fontsize=8, alpha=0.9)

def diff_scars(prev_genome: Optional[Genome], curr_genome: Genome, prev_scars: Optional[Dict[int, 'Scar']], birth_gen: int, regen_mode_for_new: str='split') -> Dict[int, 'Scar']:
    scars = {} if prev_scars is None else {k: Scar(v.birth_gen, v.mode, v.age + 1) for k, v in prev_scars.items()}
    prev_ids = set(prev_genome.nodes.keys()) if prev_genome is not None else set()
    curr_ids = set(curr_genome.nodes.keys())
    new_nodes = list(curr_ids - prev_ids)
    for nid in new_nodes:
        scars[nid] = Scar(birth_gen=birth_gen, mode=regen_mode_for_new, age=0)
    for nid in list(scars.keys()):
        if nid not in curr_ids:
            scars.pop(nid, None)
    return scars

def draw_genome_png(genome: Genome, scars: Optional[Dict[int, 'Scar']], path: str, title: Optional[str]=None, prev_genome: Optional[Genome]=None, decay_horizon: float=8.0):
    pos = layout_by_depth(genome)
    fig, ax = plt.subplots(figsize=(6, 4))
    if prev_genome is not None:
        _draw_edges_with_diff(ax, prev_genome, genome, pos)
    else:
        _draw_edges(ax, genome, pos, lw=1.0, alpha=0.8)
    _draw_nodes(ax, genome, pos, scars=scars, pulse_t=0.0, decay_horizon=decay_horizon, radius=0.12, annotate_type=True, show_mode_mark=True)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.tight_layout()
    _savefig(fig, path, dpi=200)
    plt.close(fig)

def _normalize_scar_snapshot(entry: Optional[Dict[Any, Any]]) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    """Convert a scar snapshot into numeric node/edge age dictionaries."""
    node_scars: Dict[int, int] = {}
    edge_scars: Dict[Tuple[int, int], int] = {}
    if not entry:
        return (node_scars, edge_scars)

    def _coerce_age(val: Any) -> Optional[int]:
        if hasattr(val, 'age'):
            try:
                return int(getattr(val, 'age'))
            except Exception:
                return None
        if isinstance(val, dict) and 'age' in val:
            try:
                return int(val.get('age'))
            except Exception:
                return None
        if isinstance(val, (int, float)):
            return int(val)
        return None
    if isinstance(entry, dict) and ('nodes' in entry or 'edges' in entry):
        raw_nodes = entry.get('nodes', {}) or {}
        for nid, val in raw_nodes.items():
            age = _coerce_age(val)
            if age is not None:
                try:
                    node_scars[int(nid)] = age
                except Exception:
                    continue
        raw_edges = entry.get('edges', {}) or {}
        for key, val in raw_edges.items():
            age = _coerce_age(val)
            if age is None:
                continue
            if isinstance(key, (tuple, list)) and len(key) == 2:
                try:
                    edge_scars[int(key[0]), int(key[1])] = age
                except Exception:
                    continue
            else:
                continue
        return (node_scars, edge_scars)
    if isinstance(entry, dict):
        for nid, val in entry.items():
            age = _coerce_age(val)
            if age is not None:
                try:
                    node_scars[int(nid)] = age
                except Exception:
                    continue
    return (node_scars, edge_scars)

def _export_morph_gif_with_scars(snapshots_genomes, snapshots_scars, path, *, fps=12, morph_frames=12, decay_horizon=10.0, fixed_layout=True, dpi=130, pulse_period_frames=16):
    """Scar-aware morph GIF helper shared by export_morph_gif; canonical deduped variant."""
    import numpy as _np
    import matplotlib.pyplot as _plt
    if not snapshots_genomes or len(snapshots_genomes) < 2:
        raise ValueError('_export_morph_gif_with_scars: need >= 2 snapshots.')
    if len(snapshots_genomes) != len(snapshots_scars):
        raise ValueError('_export_morph_gif_with_scars: scars_seq length mismatch.')
    pos_union = layout_by_depth_union(snapshots_genomes) if fixed_layout else None
    if pos_union:
        xs = [p[0] for p in pos_union.values()]
        ys = [p[1] for p in pos_union.values()]
        min_x, max_x = (min(xs), max(xs))
        min_y, max_y = (min(ys), max(ys))
        span_x = max(1e-06, max_x - min_x)
        span_y = max(1e-06, max_y - min_y)
        pos_union = {nid: (0.05 + 0.9 * ((x - min_x) / span_x), 0.05 + 0.9 * ((y - min_y) / span_y)) for nid, (x, y) in pos_union.items()}

    def _node_groups_and_depths(_g):
        depth = _g.node_depths()
        groups = {'input': [], 'bias': [], 'hidden': [], 'output': []}
        for nid, n in _g.nodes.items():
            groups.get(n.type, groups['hidden']).append(nid)
        for key in groups:
            groups[key].sort()
        max_d = max(depth.values()) if depth else 1
        return (groups, depth, max_d)

    def _compute_positions(_g):
        groups, depth, max_d = _node_groups_and_depths(_g)
        pos = {}
        bands = [('input', 0.85, 1.0), ('bias', 0.7, 0.82), ('hidden', 0.2, 0.68), ('output', 0.02, 0.18)]
        for gname, y0, y1 in bands:
            arr = groups[gname]
            n = max(1, len(arr))
            ys = _np.linspace(y1, y0, n)
            for idx, nid in enumerate(arr):
                depth_val = depth.get(nid, 0)
                if gname in ('input', 'bias'):
                    x = 0.04
                elif gname == 'output':
                    x = 0.96
                else:
                    x = 0.1 + 0.8 * (depth_val / max(1, max_d))
                pos[nid] = (x, ys[idx])
        return pos

    def _pulse_amp(age: int, frame_idx: int) -> float:
        base = max(0.1, 1.0 - float(age) / max(1e-06, decay_horizon))
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
        edges0 = set(((c.in_node, c.out_node) for c in g0.enabled_connections()))
        edges1 = set(((c.in_node, c.out_node) for c in g1.enabled_connections()))
        for k in range(total_steps):
            t = 0.0 if total_steps <= 1 else k / float(total_steps - 1)
            frame_index = i * total_steps + k
            fig, ax = _plt.subplots(figsize=(6.6, 4.8), dpi=dpi)
            ax.set_axis_off()
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            for u, v in sorted(edges0):
                if u not in pos0 or v not in pos0:
                    continue
                x0, y0 = pos0[u]
                x1, y1 = pos0[v]
                width = 1.6
                if (u, v) in edge_scars0:
                    width += 0.6 * _pulse_amp(edge_scars0[u, v], frame_index)
                ax.plot([x0, x1], [y0, y1], linewidth=width, alpha=max(0.0, 1.0 - t))
            for u, v in sorted(edges1):
                if u not in pos1 or v not in pos1:
                    continue
                x0, y0 = pos1[u]
                x1, y1 = pos1[v]
                width = 1.8
                if (u, v) in edge_scars1:
                    width += 0.6 * _pulse_amp(edge_scars1[u, v], frame_index)
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
                if tname == 'input':
                    size = 35.0
                elif tname == 'bias':
                    size = 28.0
                elif tname == 'output':
                    size = 60.0
                alpha0 = 1.0 if nid in g0.nodes and nid in g1.nodes else 1.0 - t if nid in g0.nodes else t
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

def export_double_exposure(genome: Genome, lineage_edges: List[Tuple[Optional[int], Optional[int], int, int, str]], current_gen: int, out_path: str, title: Optional[str]=None):
    gens = {}
    for m, f, child, gen, kind in lineage_edges:
        if gen > current_gen:
            continue
        gens.setdefault(gen, []).append(child)
    for gen in gens:
        gens[gen] = sorted(gens[gen])
    id_row = {}
    for gen in sorted(gens.keys()):
        for idx, cid in enumerate(gens[gen]):
            id_row[cid] = idx
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, f, child, gen, kind in lineage_edges:
        if gen > current_gen:
            continue
        x2 = gen
        y2 = id_row.get(child, 0)
        if m is not None:
            x1 = gen - 1
            y1 = id_row.get(m, y2)
            ax.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.25)
        if f is not None:
            x1 = gen - 1
            y1 = id_row.get(f, y2)
            ax.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.25)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Lineage Row')
    pos = layout_by_depth(genome)
    for c in genome.enabled_connections():
        i, o = (c.in_node, c.out_node)
        if i not in pos or o not in pos:
            continue
        p1 = pos[i]
        p2 = pos[o]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1.5, alpha=0.9)
    for nid in genome.nodes:
        if nid not in pos:
            continue
        x, y = pos[nid]
        circ = Circle((x, y), radius=0.1, fill=True, alpha=0.9, linewidth=0.0)
        ax.add_patch(circ)
        circ2 = Circle((x, y), radius=0.1, fill=False, linewidth=1.5, alpha=0.9)
        ax.add_patch(circ2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=220)
    plt.close(fig)

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
                if p is None:
                    continue
                target = gen[c] - 1
                if p not in gen or gen[p] > target:
                    gen[p] = target
                    changed = True
    if gen:
        shift = -min(gen.values())
        if shift > 0:
            for k in list(gen.keys()):
                gen[k] += shift
    return gen

def _fallback_lineage_layout(nodes: List[int], gen_map: Dict[int, int]):
    layers: Dict[int, List[int]] = {}
    for n in nodes:
        layers.setdefault(int(gen_map.get(n, 0)), []).append(n)
    for k in layers:
        layers[k] = sorted(layers[k])
    pos = {}
    max_gen = max(layers.keys()) if layers else 0
    for g in range(max_gen + 1):
        row = layers.get(g, [])
        n = len(row) or 1
        xs = np.linspace(0.1, 0.9, n)
        y = 1.0 - g / max(1, max_gen + 0.5)
        for x, nid in zip(xs, row):
            pos[nid] = (x, y)
    for n in nodes:
        if n not in pos:
            pos[n] = (0.5, 0.5)
    return pos

def _summarize_lineage_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ''
    parts: List[str] = []
    changed_nodes = rows[0].get('changed_nodes', 0)
    changed_edges = rows[0].get('changed_edges', 0)
    if changed_nodes:
        parts.append(f'ΔN={changed_nodes}')
    if changed_edges:
        parts.append(f'ΔE={changed_edges}')
    delta_paths = sum(int(row.get('delta_paths', 0) or 0) for row in rows)
    if delta_paths:
        parts.append(f'ΔP={delta_paths:+d}')
    detours = [float(row['detour']) for row in rows if isinstance(row.get('detour'), (int, float))]
    if detours:
        parts.append(f'↺{float(np.mean(detours)):+.2f}')
    if any(int(row.get('heal_flag', 0)) for row in rows):
        parts.append('heal')
    return ' '.join(parts)

def _merge_meta_summary(genome: Genome, summary: str='') -> str:
    latest = None
    meta = getattr(genome, 'meta_reflections', None)
    if meta:
        latest = meta[-1]
    if not latest:
        return summary
    event = latest.get('event', '')
    delta = latest.get('delta_score')
    if event == 'mutate_weights':
        mean_delta = latest.get('mean_abs_delta')
        desc = f'{event}' + (f' μΔw={mean_delta:.3f}' if isinstance(mean_delta, (int, float)) else '')
    else:
        if isinstance(delta, (int, float)):
            desc = f'{event} ΔC={delta:+.2f}'
        else:
            desc = event
    if summary and desc:
        return f'{summary} || {desc}'
    return desc or summary

def render_lineage(neat, path='lineage.png', title='Lineage', max_edges: Optional[int]=10000, highlight: Optional[Iterable[int]]=None, dpi=200):
    edges = getattr(neat, 'lineage_edges', None)
    if not edges:
        raise ValueError('neat.lineage_edges is empty. Run evolve() first.')
    use_edges = edges[-max_edges:] if max_edges and len(edges) > max_edges else edges
    nodes = set()
    for m, f, c, g, tag in use_edges:
        for nid in (m, f, c):
            if nid is not None:
                nodes.add(nid)
    gen_map = _infer_generations(use_edges)
    pos = _fallback_lineage_layout(sorted(nodes), gen_map)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_axis_off()
    ax.set_title(title, loc='left', fontsize=12)
    for m, f, c, g, tag in use_edges:
        kind = tag if tag and tag != 'birth' else 'asexual' if m is None or f is None else 'selfing' if m == f else 'sexual'
        style = 'solid' if kind == 'sexual' else 'dashdot' if kind == 'selfing' else 'dashed' if kind == 'asexual' else 'dotted'
        width = 1.8 if kind in ('sexual', 'asexual') else 1.4
        for p in (m, f):
            if p is None:
                continue
            if p not in pos or c not in pos:
                continue
            x1, y1 = pos[p]
            x2, y2 = pos[c]
            arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, lw=width, linestyle=style, alpha=0.9)
            ax.add_patch(arr)
    reg = getattr(neat, 'node_registry', {})
    xs_f = []
    ys_f = []
    ss_f = []
    xs_m = []
    ys_m = []
    ss_m = []
    xs_h = []
    ys_h = []
    ss_h = []
    xs_u = []
    ys_u = []
    ss_u = []
    for nid, (x, y) in pos.items():
        info = reg.get(nid, {})
        sex = info.get('sex', None)
        regen = bool(info.get('regen', False))
        size = 80 * (1.3 if regen else 1.0)
        if sex == 'female':
            xs_f.append(x)
            ys_f.append(y)
            ss_f.append(size)
        elif sex == 'male':
            xs_m.append(x)
            ys_m.append(y)
            ss_m.append(size)
        elif sex == 'hermaphrodite':
            xs_h.append(x)
            ys_h.append(y)
            ss_h.append(size)
        else:
            xs_u.append(x)
            ys_u.append(y)
            ss_u.append(size)
    if xs_f:
        ax.scatter(xs_f, ys_f, s=ss_f, marker='o', alpha=0.95, c='#FF69B4', label='female')
    if xs_m:
        ax.scatter(xs_m, ys_m, s=ss_m, marker='s', alpha=0.95, c='#4169E1', label='male')
    if xs_h:
        ax.scatter(xs_h, ys_h, s=ss_h, marker='D', alpha=0.95, c='#9370DB', label='hermaphrodite')
    if xs_u:
        ax.scatter(xs_u, ys_u, s=ss_u, marker='^', alpha=0.95, c='#808080')
    if xs_f or xs_m or xs_h:
        ax.legend(loc='best', frameon=False, fontsize=9)
    hi = set(highlight or [])
    if hi:
        hx = []
        hy = []
        for nid in hi:
            if nid in pos:
                x, y = pos[nid]
                hx.append(x)
                hy.append(y)
        if hx:
            ax.scatter(hx, hy, s=300, facecolors='none', edgecolors='black', linewidths=2.4, alpha=0.9)
    if len(nodes) <= 1200:
        for nid, (x, y) in pos.items():
            ax.text(x, y + 0.02, str(nid), fontsize=6, ha='center', va='bottom', alpha=0.9)
    annotations: Dict[int, List[Tuple[int, str]]] = getattr(neat, 'lineage_annotations', {}) or {}
    if annotations:
        for nid, (x, y) in pos.items():
            notes = annotations.get(nid)
            if not notes:
                continue
            snippet = notes[-3:]
            text = '\n'.join(f'g{gen}: {msg}' for gen, msg in snippet if msg)
            if not text:
                continue
            ax.text(x + 0.012, y - 0.05, text, fontsize=5.5, ha='left', va='top', alpha=0.85, family='monospace')
    fig.tight_layout()
    _savefig(fig, path, dpi=dpi)
    plt.close(fig)

def _moving_stats(arr: List[float], window: int):
    arr = np.asarray(arr, dtype=np.float64)
    if window <= 1 or window > len(arr):
        return (arr, np.zeros_like(arr))
    ma = np.convolve(arr, np.ones(window) / window, mode='valid')
    pad = len(arr) - len(ma)
    ma = np.concatenate([np.full(pad, np.nan), ma])
    rs = []
    for i in range(len(arr)):
        j0 = max(0, i - window + 1)
        jj = arr[j0:i + 1]
        rs.append(np.std(jj) if len(jj) > 1 else 0.0)
    return (ma, np.asarray(rs))

def plot_learning_and_complexity(history: List[Tuple[float, float]], hidden_counts_history: List[List[int]], edge_counts_history: List[List[int]], out_path: str, title: str, ma_window: int=7):
    best = [b for b, _ in history]
    avg = [a for _, a in history]
    best_ma, best_std = _moving_stats(best, ma_window)
    avg_ma, avg_std = _moving_stats(avg, ma_window)
    mean_hidden = [float(np.mean(h)) for h in hidden_counts_history]
    mean_edges = [float(np.mean(e)) for e in edge_counts_history]
    gens = np.arange(len(history))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(gens, best, linewidth=1.0, alpha=0.7, linestyle=':')
    ax1.plot(gens, avg, linewidth=1.0, alpha=0.7, linestyle=':')
    ax1.plot(gens, best_ma, linewidth=1.6, alpha=0.95, label='best (MA)')
    ax1.plot(gens, avg_ma, linewidth=1.4, alpha=0.95, label='avg (MA)')
    ax1.plot(gens, avg_ma - avg_std, linewidth=0.9, alpha=0.8, linestyle='--')
    ax1.plot(gens, avg_ma + avg_std, linewidth=0.9, alpha=0.8, linestyle='--')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax2 = ax1.twinx()
    ax2.plot(gens, mean_hidden, linewidth=1.2, alpha=0.75, linestyle='-')
    ax2.plot(gens, mean_edges, linewidth=1.2, alpha=0.75, linestyle='-.')
    ax2.set_ylabel('Complexity')
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=200)
    plt.close(fig)


def export_diversity_summary(div_history: Sequence[Dict[str, Any]], csv_path: str, png_path: str, title: str='Diversity & Environment Trajectory') -> Tuple[Optional[str], Optional[str]]:
    if not div_history:
        return (None, None)
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
    fields = [
        'gen',
        'entropy',
        'scarcity',
        'family_entropy',
        'top_family_share',
        'family_surplus_ratio_max',
        'family_surplus_ratio_mean',
        'family_count',
        'complexity_mean',
        'complexity_std',
        'structural_spread',
        'diversity_bonus',
        'diversity_power',
        'env_noise',
        'env_focus',
        'env_entropy',
        'lazy_share',
        'household_pressure',
        'unique_signatures',
        'complexity_baseline',
        'complexity_span',
        'complexity_span_quantile',
        'complexity_max',
        'complexity_bonus_limit',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in div_history:
            payload = {key: row.get(key, '') for key in fields}
            writer.writerow(payload)
    gens = np.array([int(item.get('gen', idx)) for idx, item in enumerate(div_history)], dtype=np.int32)
    entropy = np.array([float(item.get('entropy', 0.0)) for item in div_history], dtype=np.float64)
    scarcity = np.array([float(item.get('scarcity', 0.0)) for item in div_history], dtype=np.float64)
    family_entropy = np.array([float(item.get('family_entropy', 0.0)) for item in div_history], dtype=np.float64)
    top_family_share = np.array([float(item.get('top_family_share', 0.0)) for item in div_history], dtype=np.float64)
    family_surplus_ratio = np.array([float(item.get('family_surplus_ratio_max', 0.0)) for item in div_history], dtype=np.float64)
    family_surplus_ratio_mean = np.array([float(item.get('family_surplus_ratio_mean', 0.0)) for item in div_history], dtype=np.float64)
    spread = np.array([float(item.get('structural_spread', 0.0)) for item in div_history], dtype=np.float64)
    bonus = np.array([float(item.get('diversity_bonus', 0.0)) for item in div_history], dtype=np.float64)
    env_noise = np.array([float(item.get('env_noise', 0.0)) for item in div_history], dtype=np.float64)
    env_focus = np.array([float(item.get('env_focus', 0.0)) for item in div_history], dtype=np.float64)
    env_entropy = np.array([float(item.get('env_entropy', 0.0)) for item in div_history], dtype=np.float64)
    lazy_share = np.array([float(item.get('lazy_share', 0.0)) for item in div_history], dtype=np.float64)
    household_pressure = np.array([float(item.get('household_pressure', 0.0)) for item in div_history], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.4, 6.0))
    ax_top, ax_bottom = axes
    ax_top.plot(gens, entropy, label='entropy (structural)', color='#1f78b4', linewidth=1.8)
    ax_top.plot(gens, scarcity, label='scarcity', color='#d62728', linewidth=1.6)
    ax_top.plot(gens, family_entropy, label='entropy (family)', color='#6a3d9a', linewidth=1.4, linestyle='-.')
    ax_top.fill_between(gens, 0.0, scarcity, color='#ff9896', alpha=0.25)
    ax_top.set_ylabel('entropy / scarcity')
    ax_top.legend(loc='upper right', frameon=False, fontsize=9)
    ax_top.grid(True, linestyle='--', alpha=0.2)
    ax_mid = ax_bottom.twinx()
    ax_bottom.plot(gens, bonus, label='diversity bonus', color='#2ca02c', linewidth=1.7)
    ax_bottom.plot(gens, spread, label='structural spread', color='#9467bd', linewidth=1.5, linestyle='--')
    ax_bottom.plot(gens, lazy_share, label='lazy share', color='#8c564b', linewidth=1.3, linestyle=':')
    ax_bottom.plot(gens, top_family_share, label='top family share', color='#e377c2', linewidth=1.2, linestyle='-.')
    ax_bottom.plot(gens, family_surplus_ratio, label='family surplus ratio', color='#ffbb78', linewidth=1.1, linestyle='--')
    ax_bottom.set_ylabel('bonus / spread / lazy share')
    ax_bottom.legend(loc='upper left', frameon=False, fontsize=9)
    ax_bottom.grid(True, linestyle='--', alpha=0.2)
    ax_mid.plot(gens, env_noise, label='env noise', color='#17becf', linewidth=1.4)
    ax_mid.plot(gens, env_focus, label='env focus', color='#ff7f0e', linewidth=1.2, linestyle='-.')
    ax_mid.plot(gens, env_entropy, label='env entropy', color='#7f7f7f', linewidth=1.0, linestyle=':')
    ax_mid.plot(gens, household_pressure, label='household', color='#b15928', linewidth=1.1, linestyle='--')
    ax_mid.set_ylabel('environmental metrics')
    ax_mid.legend(loc='upper right', frameon=False, fontsize=8)
    ax_bottom.set_xlabel('generation')
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    _savefig(fig, png_path, dpi=220)
    plt.close(fig)
    return (csv_path, png_path)

def plot_decision_boundary(genome: Genome, X, y, out_path: str, steps: int=50, contour_cmap: str='coolwarm', point_cmap: Optional[str]=None, point_size: float=12.0, point_alpha: float=0.85, add_colorbar: bool=False):
    gg = genome.copy()
    profile: Dict[str, Any] = {}
    history: Sequence[float] = []
    try:
        profile_buffer: Dict[str, Any] = {}
        history = train_with_backprop_numpy(gg, X, y, steps=steps, lr=0.005, l2=0.0001, profile_out=profile_buffer)
        profile = profile_buffer
    except Exception:
        pass
    x_min, x_max = (X[:, 0].min() - 0.2, X[:, 0].max() + 0.2)
    y_min, y_max = (X[:, 1].min() - 0.2, X[:, 1].max() + 0.2)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    P = predict_proba(gg, grid)[:, 1].reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 4))
    cs = ax.contourf(xx, yy, P, levels=20, alpha=0.85, cmap=contour_cmap)
    scatter_kwargs = {'s': point_size, 'alpha': point_alpha, 'linewidths': 0.25, 'edgecolors': 'black'}
    if point_cmap:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=point_cmap, **scatter_kwargs)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='gray', **scatter_kwargs)
    if add_colorbar:
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    _savefig(fig, out_path, dpi=220)
    plt.close(fig)
    history_arr = np.asarray(history, dtype=np.float64) if len(history) else np.zeros(0, dtype=np.float64)
    if profile:
        profile_payload: Dict[str, Any] = {}
        for key, value in profile.items():
            if isinstance(value, np.ndarray):
                profile_payload[key] = np.asarray(value).copy()
            else:
                profile_payload[key] = value
    else:
        profile_payload = {}
    return {'figure': out_path, 'history': history_arr, 'profile': profile_payload}

def export_backprop_variation(genome: Genome, X, y, out_path: str, steps: int=50, lr: float=0.005, l2: float=0.0001):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    gg = genome.copy()
    profile: Dict[str, Any] = {}
    history = train_with_backprop_numpy(gg, X, y, steps=steps, lr=lr, l2=l2, profile_out=profile)
    loss = np.asarray(history, dtype=np.float64)
    step_profiles = np.asarray(
        profile.get('step_profiles', np.zeros((0, len(profile.get('node_order', []))), dtype=np.float64)),
        dtype=np.float64,
    )
    if step_profiles.ndim != 2 or step_profiles.shape[0] == 0:
        step_profiles = None
    node_order = profile.get('node_order', [])
    node_types = profile.get('node_types', [])
    labels = []
    for nid, t in zip(node_order, node_types):
        prefix = (t[0].upper() + ':') if t else 'N:'
        labels.append(f'{prefix}{nid}')
    initial_sens = np.asarray(profile.get('initial_sensitivity', []), dtype=np.float64)
    final_sens = np.asarray(profile.get('final_sensitivity', []), dtype=np.float64)
    final_momentum = np.asarray(profile.get('final_momentum', []), dtype=np.float64)
    final_variance = np.asarray(profile.get('final_variance', []), dtype=np.float64)
    final_jitter = np.asarray(profile.get('final_jitter', []), dtype=np.float64)
    final_altruism = np.asarray(profile.get('final_altruism', []), dtype=np.float64)
    final_altruism_mem = np.asarray(profile.get('final_altruism_memory', []), dtype=np.float64)
    final_altruism_span = np.asarray(profile.get('final_altruism_span', []), dtype=np.float64)
    idx_pool = [i for i, t in enumerate(node_types) if t == 'hidden']
    if not idx_pool:
        idx_pool = list(range(len(labels)))
    delta = final_sens - initial_sens if initial_sens.size and final_sens.size else np.zeros(len(labels), dtype=np.float64)
    order_idx = sorted(idx_pool, key=lambda i: abs(delta[i]) if i < delta.size else 0.0, reverse=True)
    if len(order_idx) > 10:
        order_idx = order_idx[:10]
    scatter_idx = sorted(idx_pool, key=lambda i: (final_variance[i] if i < final_variance.size else 0.0) + 0.25 * abs(final_momentum[i] if i < final_momentum.size else 0.0), reverse=True)
    if len(scatter_idx) > 12:
        scatter_idx = scatter_idx[:12]
    fig = plt.figure(figsize=(10.5, 6.2))
    gs = _gridspec.GridSpec(2, 2, height_ratios=[1.7, 1.0], width_ratios=[1.0, 1.0], hspace=0.32, wspace=0.28)
    ax_top = fig.add_subplot(gs[0, :])
    steps_axis = np.arange(1, loss.size + 1, dtype=np.float64) if loss.size else np.arange(1, steps + 1, dtype=np.float64)
    handles = []
    labels_legend = []
    if step_profiles is not None:
        mean_profile = step_profiles.mean(axis=1)
        std_profile = step_profiles.std(axis=1)
        mp = mean_profile[:steps_axis.size]
        sp = std_profile[:steps_axis.size]
        ax_top.fill_between(steps_axis[:mp.size], mp - sp[:mp.size], mp + sp[:mp.size], color='#8ecae6', alpha=0.35, label='pulse ±σ')
        line_mp, = ax_top.plot(steps_axis[:mp.size], mp, color='#219ebc', lw=2.0, label='pulse mean')
        handles.append(line_mp)
        labels_legend.append('pulse mean')
    if loss.size:
        ax_loss = ax_top.twinx()
        line_loss, = ax_loss.plot(steps_axis[:loss.size], loss, color='#f07167', lw=1.8, label='loss')
        ax_loss.set_ylabel('loss', color='#f07167')
        ax_loss.tick_params(axis='y', colors='#f07167')
        handles.append(line_loss)
        labels_legend.append('loss')
    ax_top.set_xlabel('backprop step')
    ax_top.set_ylabel('pulse magnitude')
    ax_top.set_title('Backprop pulse landscape')
    if handles:
        ax_top.legend(handles, labels_legend, loc='upper right', frameon=False, fontsize=9)
    ax_left = fig.add_subplot(gs[1, 0])
    if order_idx and initial_sens.size and final_sens.size:
        y_pos = np.arange(len(order_idx))
        init_vals = initial_sens[order_idx]
        final_vals = final_sens[order_idx]
        delta_vals = final_vals - init_vals
        ax_left.barh(y_pos - 0.18, init_vals, height=0.3, color='#c1d3fe', alpha=0.75, label='initial')
        ax_left.barh(y_pos + 0.18, final_vals, height=0.3, color='#5e60ce', alpha=0.9, label='post-train')
        ax_left.axvline(1.0, color='#333333', lw=0.8, ls='--', alpha=0.6)
        for k, idx in enumerate(order_idx):
            lbl = labels[idx] if idx < len(labels) else f'n{idx}'
            ax_left.text(final_vals[k] + 0.04, y_pos[k] + 0.2, f'Δ{delta_vals[k]:+.2f}', fontsize=8, color='#333333')
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels([labels[idx] if idx < len(labels) else f'n{idx}' for idx in order_idx])
        ax_left.set_xlabel('sensitivity')
        ax_left.set_title('Hidden node sensitivity drift')
        ax_left.legend(frameon=False, fontsize=8, loc='lower right')
    else:
        ax_left.text(0.5, 0.5, 'No hidden nodes tracked', ha='center', va='center', fontsize=10)
        ax_left.set_axis_off()
    ax_right = fig.add_subplot(gs[1, 1])
    if scatter_idx and final_momentum.size and final_variance.size:
        mom_vals = final_momentum[scatter_idx]
        var_vals = final_variance[scatter_idx]
        color_vals = final_altruism[scatter_idx] if final_altruism.size else np.zeros_like(mom_vals)
        span_vals = final_altruism_span[scatter_idx] if final_altruism_span.size else np.zeros_like(mom_vals)
        sizes = 60.0 + 80.0 * np.clip(span_vals, 0.0, 4.0) / 4.0
        sc = ax_right.scatter(mom_vals, var_vals, c=color_vals, cmap='viridis', s=sizes, edgecolors='k', linewidths=0.35)
        for k, idx in enumerate(scatter_idx):
            lbl = labels[idx] if idx < len(labels) else f'n{idx}'
            ax_right.text(mom_vals[k] + 0.02, var_vals[k] + 0.02, lbl, fontsize=8)
        ax_right.set_xlabel('momentum (smoothed)')
        ax_right.set_ylabel('variance trace')
        ax_right.set_title('Post-train temperament field')
        if len(scatter_idx) >= 3:
            cb = fig.colorbar(sc, ax=ax_right, fraction=0.046, pad=0.04)
            cb.set_label('altruism', fontsize=9)
    else:
        ax_right.text(0.5, 0.5, 'No temperament statistics', ha='center', va='center', fontsize=10)
        ax_right.set_axis_off()
    fig.suptitle(f'Backprop Variation | steps={steps}', fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _savefig(fig, out_path, dpi=220)
    plt.close(fig)
    return {'figure': out_path, 'history': loss, 'profile': profile}

def export_decision_boundaries_all(genome: Genome, out_dir: str, steps: int=50, seed: int=0):
    os.makedirs(out_dir or '.', exist_ok=True)
    Xc, yc = make_circles(512, r=0.6, noise=0.05, seed=seed)
    path_c = os.path.join(out_dir, 'decision_circles.png')
    plot_decision_boundary(genome, Xc, yc, path_c, steps=steps, contour_cmap='cividis', point_cmap='cool', point_size=22.0)
    Xx, yx = make_xor(512, noise=0.05, seed=seed)
    path_x = os.path.join(out_dir, 'decision_xor.png')
    plot_decision_boundary(genome, Xx, yx, path_x, steps=steps, contour_cmap='Spectral', point_cmap='Dark2', point_size=18.0)
    Xs, ys = make_spirals(512, noise=0.05, turns=1.5, seed=seed)
    path_s = os.path.join(out_dir, 'decision_spiral.png')
    plot_decision_boundary(genome, Xs, ys, path_s, steps=steps, contour_cmap='magma', point_cmap='plasma', point_size=14.0)
    return {'circles': path_c, 'xor': path_x, 'spiral': path_s}


def compose_gallery_from_existing(result: Dict[str, str], out_dir: str, task: str='task', idx: int=1) -> Dict[str, str]:
    """Compose a gallery tile (learning curve + decision boundary) without rerunning experiments."""
    os.makedirs(out_dir or '.', exist_ok=True)
    outputs: Dict[str, str] = {}
    lc = result.get('learning_curve')
    db = result.get('decision_boundary')
    if lc and db and os.path.exists(lc) and os.path.exists(db):
        combo = os.path.join(out_dir, f'{idx:02d}_{task}_gallery.png')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, path, title in zip(axes, (lc, db), ('学習曲線', '決定境界')):
            img = _imread_image(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis('off')
        fig.tight_layout()
        _savefig(fig, combo, dpi=220)
        plt.close(fig)
        outputs[f'{idx:02d} {task.upper()} 学習曲線＋決定境界'] = combo
    else:
        if lc:
            outputs[f'{idx:02d} {task.upper()} 学習曲線'] = lc
        if db:
            outputs[f'{idx:02d} {task.upper()} 決定境界'] = db
    topo = result.get('topology')
    if topo and os.path.exists(topo):
        outputs[f'{idx:02d} {task.upper()} トポロジ'] = topo
    lineage = result.get('lineage')
    if lineage and os.path.exists(lineage):
        outputs[f'{idx:02d} {task.upper()} 系統図'] = lineage
    return outputs

def export_task_gallery(tasks: Tuple[str, ...], gens: int, pop: int, steps: int, out_dir: str) -> Dict[str, str]:
    """Run a batch of miniature experiments and collect figure paths."""
    os.makedirs(out_dir or '.', exist_ok=True)
    outputs: Dict[str, str] = {}
    import zlib
    for idx, task in enumerate(tasks, start=1):
        tag = f'{task}_g{gens}_p{pop}_s{steps}'
        seed = zlib.crc32(f'{task}|{gens}|{pop}|{steps}'.encode('utf-8')) & 4294967295
        res = run_backprop_neat_experiment(task, gens=gens, pop=pop, steps=steps, out_prefix=os.path.join(out_dir, tag), make_gifs=False, make_lineage=True, rng_seed=seed)
        lc = res.get('learning_curve')
        db = res.get('decision_boundary')
        if lc and db and os.path.exists(lc) and os.path.exists(db):
            combo = os.path.join(out_dir, f'{idx:02d}_{tag}_gallery.png')
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
            for ax, path, title in zip(axes, (lc, db), ('学習曲線', '決定境界')):
                img = _imread_image(path)
                ax.imshow(img)
                ax.set_title(f'{task.upper()} | {title}')
                ax.axis('off')
            fig.tight_layout()
            _savefig(fig, combo, dpi=220)
            plt.close(fig)
            outputs[f'{idx:02d} {task.upper()} 学習曲線＋決定境界'] = combo
        else:
            if lc:
                outputs[f'{idx:02d} {task.upper()} 学習曲線'] = lc
            if db:
                outputs[f'{idx:02d} {task.upper()} 決定境界'] = db
        topo = res.get('topology')
        if topo:
            outputs[f'{idx:02d} {task.upper()} トポロジ'] = topo
        lineage = res.get('lineage')
        if lineage and os.path.exists(lineage):
            outputs[f'{idx:02d} {task.upper()} 系統図'] = lineage
    return outputs

def make_circles(n=512, r=0.5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (np.sqrt((X ** 2).sum(axis=1)) > r).astype(np.int32)
    X += rng.normal(0, noise, size=X.shape)
    return (X, y)

def make_xor(n=512, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int32)
    X += rng.normal(0, noise, size=X.shape)
    return (X, y)

def make_spirals(n=512, noise=0.1, turns=1.5, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    t = np.linspace(0.0, turns * 2 * np.pi, n2)
    r = np.linspace(0.05, 1.0, n2)
    x1 = r * np.cos(t)
    y1 = r * np.sin(t)
    x2 = r * np.cos(t + np.pi)
    y2 = r * np.sin(t + np.pi)
    X = np.vstack([np.stack([x1, y1], 1), np.stack([x2, y2], 1)])
    X += rng.normal(0, noise, size=X.shape)
    y = np.concatenate([np.zeros(n2, dtype=np.int32), np.ones(n2, dtype=np.int32)])
    return (X, y)

class FitnessBackpropShared:
    """
    Picklable callable that reads datasets from shared memory by label.
    Also provides refine_raw(genome, factor) for adaptive extra-steps.
    """

    def __init__(self, keys=('Xtr', 'ytr', 'Xva', 'yva'), steps=40, lr=0.005, l2=0.0001, alpha_nodes=0.001, alpha_edges=0.0005):
        self.keys = tuple(keys)
        self.steps = int(steps)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.alpha_nodes = float(alpha_nodes)
        self.alpha_edges = float(alpha_edges)
        self.noise_std = 0.0
        self.collective_signal: Optional[Dict[str, float]] = None

    def set_noise_std(self, s: float):
        self.noise_std = float(max(0.0, s))

    def _load(self):
        Xtr = get_shared_dataset(self.keys[0])
        ytr = get_shared_dataset(self.keys[1])
        Xva = get_shared_dataset(self.keys[2])
        yva = get_shared_dataset(self.keys[3])
        return (Xtr, ytr, Xva, yva)

    def _aug(self, X):
        s = float(self.noise_std)
        if s <= 0.0:
            return X
        rng = np.random.default_rng()
        return X + rng.normal(0.0, s, size=X.shape)

    def __call__(self, g: 'Genome') -> float:
        signal = getattr(self, 'collective_signal', None)
        signal = dict(signal) if signal else None
        Xtr, ytr, Xva, yva = self._load()
        return fitness_backprop_classifier(
            g,
            self._aug(Xtr),
            ytr,
            self._aug(Xva),
            yva,
            steps=self.steps,
            lr=self.lr,
            l2=self.l2,
            alpha_nodes=self.alpha_nodes,
            alpha_edges=self.alpha_edges,
            collective_signal=signal,
        )

    def refine_raw(self, g: 'Genome', factor: float=2.0) -> float:
        steps = int(max(1, round(self.steps * float(factor))))
        signal = getattr(self, 'collective_signal', None)
        signal = dict(signal) if signal else None
        Xtr, ytr, Xva, yva = self._load()
        return fitness_backprop_classifier(
            g,
            self._aug(Xtr),
            ytr,
            self._aug(Xva),
            yva,
            steps=steps,
            lr=self.lr,
            l2=self.l2,
            alpha_nodes=self.alpha_nodes,
            alpha_edges=self.alpha_edges,
            collective_signal=signal,
        )

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
        if 'within' not in cfg:
            raise KeyError("Configuration must include 'within' key")
        if not hasattr(cfg['within'], '__len__'):
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
        if stage >= 1:
            if stage >= len(self.cfg['within']):
                return False
            win = self.cfg['within'][stage]
            if win is not None and win >= 0:
                self.finished_samples.add(sample_id)
                return True
        return False

    def is_finished(self, sample_id: int) -> bool:
        """Check if a sample has been marked as finished."""
        return sample_id in self.finished_samples

    def reset(self):
        """Reset all finished samples."""
        self.finished_samples.clear()

def run_backprop_neat_experiment(
    task: str,
    gens=60,
    pop=64,
    steps=80,
    out_prefix='out/exp',
    make_gifs: bool=True,
    make_lineage: bool=True,
    rng_seed: int=0,
    complexity_baseline_quantile: Optional[float]=None,
    complexity_survivor_cap: Optional[float]=None,
    complexity_bonus_limit: Optional[float]=None,
    complexity_bonus_span_quantile: Optional[float]=None,
):
    if task == 'xor':
        Xtr, ytr = make_xor(512, noise=0.05, seed=0)
        Xva, yva = make_xor(256, noise=0.05, seed=1)
    elif task == 'spiral':
        Xtr, ytr = make_spirals(512, noise=0.05, turns=1.5, seed=0)
        Xva, yva = make_spirals(256, noise=0.05, turns=1.5, seed=1)
    else:
        Xtr, ytr = make_circles(512, r=0.6, noise=0.05, seed=0)
        Xva, yva = make_circles(256, r=0.6, noise=0.05, seed=1)
    rng = np.random.default_rng(rng_seed)
    out_dim = 2
    neat_module = sys.modules[__name__]
    neat = neat_module.ReproPlanaNEATPlus(num_inputs=Xtr.shape[1], num_outputs=out_dim, population_size=pop, output_activation='identity', rng=rng)
    _apply_stable_neat_defaults(neat)
    if complexity_baseline_quantile is not None:
        neat.complexity_bonus_baseline_quantile = float(complexity_baseline_quantile)
    if complexity_survivor_cap is not None:
        neat.complexity_survivor_cap = float(complexity_survivor_cap)
    if complexity_bonus_limit is not None:
        neat.complexity_survivor_bonus_limit = float(complexity_bonus_limit)
    if complexity_bonus_span_quantile is not None:
        neat.complexity_bonus_span_quantile = float(complexity_bonus_span_quantile)
    regen_log_path = f'{out_prefix}_regen_log.csv'
    if hasattr(neat, 'lcs_monitor') and neat.lcs_monitor is not None:
        neat.lcs_monitor.csv_path = regen_log_path
        if os.path.exists(regen_log_path):
            os.remove(regen_log_path)
    use_shm = os.environ.get('NEAT_EVAL_BACKEND', 'thread') == 'process'
    if use_shm:
        shm_meta = {}
        try:
            shm_meta['Xtr'] = shm_register_dataset('Xtr', Xtr, readonly=True)
            shm_meta['ytr'] = shm_register_dataset('ytr', ytr, readonly=True)
            shm_meta['Xva'] = shm_register_dataset('Xva', Xva, readonly=True)
            shm_meta['yva'] = shm_register_dataset('yva', yva, readonly=True)
            neat._shm_meta = shm_meta
        except Exception:
            neat._shm_meta = None
    else:
        _SHM_CACHE['Xtr'] = Xtr
        _SHM_CACHE['ytr'] = ytr
        _SHM_CACHE['Xva'] = Xva
        _SHM_CACHE['yva'] = yva
        neat._shm_meta = None
    fit = FitnessBackpropShared(steps=steps, lr=0.005, l2=0.0001, alpha_nodes=0.001, alpha_edges=0.0005)
    try:
        best, hist = neat.evolve(fit, n_generations=gens, verbose=True, env_schedule=_default_difficulty_schedule)
    finally:
        if use_shm:
            try:
                shm_release_all()
            except Exception:
                pass
        else:
            _SHM_CACHE.clear()
    lcs_rows = load_lcs_log(regen_log_path) if os.path.exists(regen_log_path) else []
    lcs_series = _prepare_lcs_series(lcs_rows) if lcs_rows else None
    out_dir = os.path.dirname(out_prefix) or '.'
    os.makedirs(out_dir, exist_ok=True)
    lc_path = f'{out_prefix}_learning_complexity.png'
    plot_learning_and_complexity(hist, neat.hidden_counts_history, neat.edge_counts_history, lc_path, title=f'{task.upper()} | Backprop NEAT', ma_window=7)
    db_path = f'{out_prefix}_decision_boundary.png'
    style_map = {'circles': dict(contour_cmap='cividis', point_cmap='cool', point_size=26.0), 'xor': dict(contour_cmap='Spectral', point_cmap='Dark2', point_size=20.0), 'spiral': dict(contour_cmap='magma', point_cmap='plasma', point_size=16.0)}
    style = style_map.get(task, {})
    plot_decision_boundary(best, Xtr, ytr, db_path, steps=steps, **style)
    topo_path = f'{out_prefix}_topology.png'
    scars = diff_scars(None, best, None, birth_gen=gens, regen_mode_for_new='split')
    draw_genome_png(best, scars, topo_path, title=f'Best Topology (Gen {gens})')
    backprop_variation = None
    bp_path = f'{out_prefix}_backprop_variation.png'
    try:
        backprop_variation = export_backprop_variation(best, Xtr, ytr, bp_path, steps=steps, lr=0.005, l2=0.0001)
    except Exception as bp_err:
        print('[WARN] Backprop variation export failed:', bp_err)
        backprop_variation = None
    regen_gif = None
    morph_gif = None
    if make_gifs and len(neat.snapshots_genomes) >= 2:
        regen_gif = f'{out_prefix}_regen.gif'
        export_regen_gif(neat.snapshots_genomes, neat.snapshots_scars, regen_gif, fps=12, pulse_period_frames=10, decay_horizon=6.0, fixed_layout=True, dpi=150, lcs_series=lcs_series)
        morph_gif = f'{out_prefix}_morph.gif'
        export_morph_gif(neat.snapshots_genomes, neat.snapshots_scars, path=morph_gif, fps=14, morph_frames=14, decay_horizon=8.0)
    lcs_ribbon = None
    lcs_timeline = None
    if lcs_rows:
        ribbon_path = f'{out_prefix}_lcs_ribbon.png'
        try:
            export_lcs_ribbon_png(lcs_rows, ribbon_path, series=lcs_series)
            lcs_ribbon = ribbon_path
        except Exception as ribbon_err:
            print('[WARN] LCS ribbon export failed:', ribbon_err)
        timeline_path = f'{out_prefix}_lcs_timeline.gif'
        try:
            export_lcs_timeline_gif(lcs_rows, timeline_path, series=lcs_series, fps=6)
            lcs_timeline = timeline_path
        except Exception as timeline_err:
            print('[WARN] LCS timeline export failed:', timeline_err)
    scars_spiral = None
    if len(neat.snapshots_genomes) >= 2:
        scars_spiral = f'{out_prefix}_scars_spiral.png'
        export_scars_spiral_map(neat.snapshots_genomes, neat.snapshots_scars, scars_spiral, turns=None, jitter=0.014, marker_size=20, dpi=220)
    lineage_path = None
    if make_lineage:
        lineage_path = f'{out_prefix}_lineage.png'
        render_lineage(neat, path=lineage_path, title=f'{task.upper()} Lineage', max_edges=5000, highlight=neat.best_ids[-10:], dpi=220)
    summary_dir = f'{out_prefix}_decisions'
    summary_paths = export_decision_boundaries_all(best, summary_dir, steps=steps, seed=0)
    top3_paths = []
    if hasattr(neat, 'top3_best_topologies') and len(neat.top3_best_topologies) > 0:
        for idx, (genome, fitness, gen) in enumerate(neat.top3_best_topologies[:3]):
            n_hidden = sum((1 for n in genome.nodes.values() if n.type == 'hidden'))
            n_edges = sum((1 for c in genome.connections.values() if c.enabled))
            top_path = f'{out_prefix}_top{idx + 1}_topology.png'
            top_scars = diff_scars(None, genome, None, birth_gen=gen, regen_mode_for_new='split')
            draw_genome_png(genome, top_scars, top_path, title=f'Top-{idx + 1} Topology (Gen {gen}, fit={fitness:.4f}, {n_hidden}h+{n_edges}e)')
            top3_paths.append(top_path)
            top_db_path = f'{out_prefix}_top{idx + 1}_decision_boundary.png'
            plot_decision_boundary(genome, Xtr, ytr, top_db_path, steps=steps, **style)
            print(f'[INFO] Top-{idx + 1}: {n_hidden} hidden nodes, {n_edges} edges, fitness={fitness:.4f} (gen {gen})')
    genomes_cyto = []
    if hasattr(neat, 'snapshots_genomes') and neat.snapshots_genomes:
        try:
            window = min(len(neat.snapshots_genomes), 48)
            genomes_cyto = [_genome_to_cyto(g) for g in neat.snapshots_genomes[-window:]]
        except Exception:
            genomes_cyto = []
    diversity_csv = None
    diversity_plot = None
    if getattr(neat, 'diversity_history', None):
        div_csv = f'{out_prefix}_diversity.csv'
        div_png = f'{out_prefix}_diversity.png'
        diversity_csv, diversity_plot = export_diversity_summary(
            list(getattr(neat, 'diversity_history', [])),
            div_csv,
            div_png,
            title=f'{task.upper()} | Diversity & Environment',
        )
    resilience_log = None
    failures = list(getattr(neat, '_resilience_failures', [])) if hasattr(neat, '_resilience_failures') else []
    if failures:
        resilience_log = f'{out_prefix}_resilience_log.json'
        try:
            with open(resilience_log, 'w', encoding='utf-8') as fh:
                json.dump({'failures': failures, 'eval_guard': getattr(neat, '_resilience_eval_guard', 0), 'history': getattr(neat, '_resilience_history', [])}, fh, indent=2, ensure_ascii=False)
        except Exception as log_err:
            print('[WARN] Resilience log write failed:', log_err)
            resilience_log = None
    return {
        'learning_curve': lc_path,
        'decision_boundary': db_path,
        'topology': topo_path,
        'top3_topologies': top3_paths,
        'regen_gif': regen_gif,
        'morph_gif': morph_gif,
        'lineage': lineage_path,
        'scars_spiral': scars_spiral,
        'summary_decisions': summary_paths,
        'lcs_log': regen_log_path if os.path.exists(regen_log_path) else None,
        'lcs_ribbon': lcs_ribbon,
        'lcs_timeline': lcs_timeline,
        'history': hist,
        'genomes_cyto': genomes_cyto,
        'resilience_log': resilience_log,
        'backprop_variation': backprop_variation,
        'diversity_csv': diversity_csv,
        'diversity_plot': diversity_plot,
        'complexity_distribution': list(getattr(neat, 'complexity_distribution_history', [])),
    }

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
    if hasattr(fig.canvas, 'tostring_rgb'):
        import numpy as _np
        buf = _np.frombuffer(fig.canvas.tostring_rgb(), dtype=_np.uint8)
        return buf.reshape(h, w, 3)
    if hasattr(fig.canvas, 'tostring_argb'):
        import numpy as _np
        buf = _np.frombuffer(fig.canvas.tostring_argb(), dtype=_np.uint8).reshape(h, w, 4)
        return buf[:, :, 1:4].copy()
    import io
    bio = io.BytesIO()
    _stamp_figure(fig)
    fig.savefig(bio, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.0)
    bio.seek(0)
    img = _imread_image(bio)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    return img

def export_regen_gif(snapshots_genomes, snapshots_scars, path, fps=12, pulse_period_frames=16, decay_horizon=10.0, fixed_layout=True, dpi=130, lcs_series: Optional[Dict[str, Any]]=None):
    """
    Render a regeneration digest GIF from per-generation snapshots.
    This is the retained canonical implementation; duplicates were removed to ensure
    consistent behaviour across exports. Encodes differences without color semantics:
    linestyle/linewidth/alpha only. When lcs_series is provided, overlays
    per-generation LCS summary text.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    if not snapshots_genomes:
        raise ValueError('export_regen_gif: snapshots_genomes is empty.')

    def _node_groups_and_depths(_g):
        depth = _g.node_depths()
        groups = {'input': [], 'bias': [], 'hidden': [], 'output': []}
        for nid, n in _g.nodes.items():
            groups.get(n.type, groups['hidden']).append(nid)
        for k in groups:
            groups[k].sort()
        max_d = max(depth.values()) if depth else 1
        return (groups, depth, max_d)

    def _compute_positions(_g):
        groups, depth, max_d = _node_groups_and_depths(_g)
        pos = {}
        bands = [('input', 0.85, 1.0), ('bias', 0.7, 0.82), ('hidden', 0.2, 0.68), ('output', 0.02, 0.18)]
        for gname, y0, y1 in bands:
            arr = groups[gname]
            n = max(1, len(arr))
            ys = _np.linspace(y1, y0, n)
            for i, nid in enumerate(arr):
                t = depth.get(nid, 0)
                if gname in ('input', 'bias'):
                    x = 0.04
                elif gname == 'output':
                    x = 0.96
                else:
                    x = 0.1 + 0.8 * (t / max(1, max_d))
                pos[nid] = (x, ys[i])
        return pos

    def _pulse_amp(age, frame_idx):
        base = max(0.1, 1.0 - float(age) / max(1e-06, decay_horizon))
        phase = 2.0 * _np.pi * (frame_idx % max(1, pulse_period_frames)) / max(1, pulse_period_frames)
        return float(base * (0.5 + 0.5 * _np.sin(phase)))
    frames = []
    prev_edges = None
    pos_fixed = _compute_positions(snapshots_genomes[0]) if fixed_layout else None
    for t, g in enumerate(snapshots_genomes):
        pos = pos_fixed if fixed_layout else _compute_positions(g)
        cur_edges = set(((c.in_node, c.out_node) for c in g.enabled_connections()))
        added = cur_edges - (prev_edges or set())
        removed = (prev_edges or set()) - cur_edges
        common = cur_edges & (prev_edges or set()) if prev_edges is not None else cur_edges
        node_scars, edge_scars = ({}, {})
        if snapshots_scars and t < len(snapshots_scars):
            node_scars, edge_scars = _normalize_scar_snapshot(snapshots_scars[t])
        fig, ax = _plt.subplots(figsize=(6.6, 4.8), dpi=dpi)
        ax.set_axis_off()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        for u, v in sorted(removed):
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], linestyle='dashed', linewidth=1.0, alpha=0.25)
        for u, v in sorted(common):
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            lw = 1.6
            if (u, v) in edge_scars:
                lw += 0.6 * _pulse_amp(edge_scars[u, v], t)
            ax.plot([x0, x1], [y0, y1], linewidth=lw, alpha=0.9)
        for u, v in sorted(added):
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], linewidth=2.2, alpha=0.95)
        types = {nid: g.nodes[nid].type for nid in g.nodes}
        for nid, (x, y) in pos.items():
            if nid not in types:
                continue
            tname = types[nid]
            sz = 50.0
            if tname == 'input':
                sz = 35.0
            if tname == 'bias':
                sz = 28.0
            if tname == 'output':
                sz = 60.0
            ax.scatter([x], [y], s=sz, alpha=1.0, zorder=3, linewidths=0.8)
            age = node_scars.get(nid, None)
            if age is not None:
                amp = _pulse_amp(age, t)
                circ = _plt.Circle((x, y), 0.018 + 0.012 * amp, fill=False, linewidth=1.0 + 2.0 * amp, alpha=0.6)
                ax.add_patch(circ)
        summary_line = None
        cum_line = None
        gen_label = f'generation {t}'
        if lcs_series is not None:
            _, summary = _latest_gen_summary(lcs_series, t)
            summary_line = _format_lcs_summary(summary)
            heals_cum, breaks_cum = _cumulative_lcs_counts(lcs_series, t)
            cum_line = f'cum heals {heals_cum} | cum breaks {breaks_cum}'
        if summary_line or cum_line:
            fig.subplots_adjust(bottom=0.28)
            fig.text(0.02, 0.18, summary_line or '', fontsize=8, family='monospace')
            if cum_line:
                fig.text(0.02, 0.11, cum_line, fontsize=7, family='monospace')
            fig.text(0.02, 0.06, gen_label, fontsize=7, family='monospace')
        else:
            fig.subplots_adjust(bottom=0.08)
            fig.text(0.02, 0.05, gen_label, fontsize=7, family='monospace')
        img = _fig_to_rgb(fig)
        frames.append(img)
        _plt.close(fig)
        prev_edges = cur_edges
    _mimsave(path, frames, fps=fps)
    return path

def export_morph_gif(snapshots_genomes, snapshots_scars=None, path=None, *, fps=12, morph_frames=8, fixed_layout=True, dpi=130, decay_horizon=8.0):
    """
    Inter-generational morphological transition GIF.
    This canonical export supports legacy calls with scar metadata as well as
    lightweight visualisations that only rely on genomes.
    """
    import os as _os
    import numpy as _np
    import matplotlib.pyplot as _plt
    if path is None and isinstance(snapshots_scars, (str, bytes)):
        path = snapshots_scars
        snapshots_scars = None
    elif path is None and hasattr(snapshots_scars, '__fspath__'):
        path = _os.fspath(snapshots_scars)
        snapshots_scars = None
    if path is None:
        raise ValueError('export_morph_gif: path must be provided')
    if snapshots_scars is not None:
        if len(snapshots_scars) != len(snapshots_genomes):
            raise ValueError('export_morph_gif: scars_seq length must match genomes.')
        return _export_morph_gif_with_scars(snapshots_genomes, snapshots_scars, path, fps=fps, morph_frames=morph_frames, decay_horizon=decay_horizon, fixed_layout=fixed_layout, dpi=dpi)
    if not snapshots_genomes or len(snapshots_genomes) < 2:
        raise ValueError('export_morph_gif: need >= 2 snapshots.')

    def _node_groups_and_depths(_g):
        depth = _g.node_depths()
        groups = {'input': [], 'bias': [], 'hidden': [], 'output': []}
        for nid, n in _g.nodes.items():
            groups.get(n.type, groups['hidden']).append(nid)
        for k in groups:
            groups[k].sort()
        max_d = max(depth.values()) if depth else 1
        return (groups, depth, max_d)

    def _compute_positions(_g):
        groups, depth, max_d = _node_groups_and_depths(_g)
        pos = {}
        bands = [('input', 0.85, 1.0), ('bias', 0.7, 0.82), ('hidden', 0.2, 0.68), ('output', 0.02, 0.18)]
        for gname, y0, y1 in bands:
            arr = groups[gname]
            n = max(1, len(arr))
            ys = _np.linspace(y1, y0, n)
            for i, nid in enumerate(arr):
                t = depth.get(nid, 0)
                if gname in ('input', 'bias'):
                    x = 0.04
                elif gname == 'output':
                    x = 0.96
                else:
                    x = 0.1 + 0.8 * (t / max(1, max_d))
                pos[nid] = (x, ys[i])
        return pos
    pos_fixed = _compute_positions(snapshots_genomes[0]) if fixed_layout else None
    frames = []
    for i in range(len(snapshots_genomes) - 1):
        g0 = snapshots_genomes[i]
        g1 = snapshots_genomes[i + 1]
        pos0 = pos_fixed if fixed_layout else _compute_positions(g0)
        pos1 = pos_fixed if fixed_layout else _compute_positions(g1)
        E0 = set(((c.in_node, c.out_node) for c in g0.enabled_connections()))
        E1 = set(((c.in_node, c.out_node) for c in g1.enabled_connections()))
        kept = E0 & E1
        gone = E0 - E1
        born = E1 - E0
        for k in range(max(1, morph_frames)):
            t = 0.0 if morph_frames <= 1 else k / float(morph_frames - 1)
            fig, ax = _plt.subplots(figsize=(6.6, 4.8), dpi=dpi)
            ax.set_axis_off()
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            for u, v in sorted(gone):
                p = pos0 if u in pos0 and v in pos0 else pos1
                if u not in p or v not in p:
                    continue
                x0, y0 = p[u]
                x1, y1 = p[v]
                ax.plot([x0, x1], [y0, y1], linestyle='dashed', linewidth=1.4, alpha=max(0.0, 1.0 - t))
            for u, v in sorted(kept):
                if u not in pos0 or v not in pos0:
                    continue
                x0, y0 = pos0[u]
                x1, y1 = pos0[v]
                ax.plot([x0, x1], [y0, y1], linewidth=1.8, alpha=0.9)
            for u, v in sorted(born):
                p = pos1 if u in pos1 and v in pos1 else pos0
                if u not in p or v not in p:
                    continue
                x0, y0 = p[u]
                x1, y1 = p[v]
                ax.plot([x0, x1], [y0, y1], linewidth=2.2, alpha=max(0.0, t))
            types = {nid: g1.nodes[nid].type if nid in g1.nodes else g0.nodes.get(nid, None).type for nid in set(list(g0.nodes.keys()) + list(g1.nodes.keys()))}
            p = pos1 if fixed_layout else pos0
            for nid, (x, y) in p.items():
                if nid not in types:
                    continue
                tname = types[nid]
                sz = 50.0
                if tname == 'input':
                    sz = 35.0
                if tname == 'bias':
                    sz = 28.0
                if tname == 'output':
                    sz = 60.0
                ax.scatter([x], [y], s=sz, alpha=1.0, zorder=3, linewidths=0.8)
            img = _fig_to_rgb(fig)
            frames.append(img)
            _plt.close(fig)
    _mimsave(path, frames, fps=fps)
    return path

def export_scars_spiral_map(snapshots_genomes: List[Genome], snapshots_scars: List[Dict[int, 'Scar']], out_path: str, *, turns: Optional[float]=None, jitter: float=0.012, marker_size: float=26.0, dpi: int=220, title: str='Scars Spiral Map'):
    """
    再生痕(=新規ノード誕生)を、世代→角度θ・進行度→半径r のアルキメデス螺旋へ投影して可視化する。

    座標系:
      θ_g = (2π * turns) * g/(G-1)
      r_g = r0 + (r1 - r0) * (θ_g / (2π * turns))   （中心から外周へ等間隔で広がる）

    入力フォーマットは diff_scars の既存形式（Dict[node_id -> Scar]）に対応。
    互換性のため、scarsが無い場合はノード集合差分から「新生ノード」を推定する。
    """
    if not snapshots_genomes:
        raise ValueError('export_scars_spiral_map: snapshots_genomes is empty.')
    import numpy as _np
    import matplotlib.pyplot as _plt
    G = len(snapshots_genomes)
    if G <= 1:
        raise ValueError('export_scars_spiral_map: need >= 2 snapshots.')
    if turns is None:
        turns = float(max(2.0, min(8.0, G / 10.0)))
    theta_max = 2.0 * _np.pi * turns
    r0, r1 = (0.12, 0.96)
    events_by_gen: Dict[int, List[Tuple[int, str]]] = {}
    for g in range(G):
        new_nodes: List[Tuple[int, str]] = []
        scars_g = snapshots_scars[g] if snapshots_scars and g < len(snapshots_scars) else None
        if isinstance(scars_g, dict) and scars_g:
            if 'nodes' in scars_g or 'edges' in scars_g:
                raw_nodes = scars_g.get('nodes', {}) or {}
                for nid, sc in raw_nodes.items():
                    age = getattr(sc, 'age', None)
                    if age is None and isinstance(sc, dict):
                        age = sc.get('age', None)
                    if age is None and isinstance(sc, (int, float)):
                        age = int(sc)
                    birth = getattr(sc, 'birth_gen', g)
                    if isinstance(sc, dict):
                        birth = sc.get('birth_gen', birth)
                    if age == 0 and birth == g:
                        mode = getattr(sc, 'mode', None)
                        if isinstance(sc, dict):
                            mode = sc.get('mode', mode)
                        new_nodes.append((int(nid), mode or 'split'))
            else:
                for nid, sc in scars_g.items():
                    age = getattr(sc, 'age', None)
                    birth = getattr(sc, 'birth_gen', g)
                    mode = getattr(sc, 'mode', None)
                    if isinstance(sc, dict):
                        age = sc.get('age', age)
                        birth = sc.get('birth_gen', birth)
                        mode = sc.get('mode', mode)
                    if age == 0 and birth == g:
                        new_nodes.append((int(nid), mode or 'split'))
        elif g > 0:
            prev_ids = set(snapshots_genomes[g - 1].nodes.keys())
            curr_ids = set(snapshots_genomes[g].nodes.keys())
            born = curr_ids - prev_ids
            mode = getattr(snapshots_genomes[g], 'regen_mode', 'split')
            new_nodes.extend(((int(nid), mode) for nid in born))
        if new_nodes:
            events_by_gen[g] = new_nodes
    xs = {'split': [], 'head': [], 'tail': [], 'other': []}
    ys = {'split': [], 'head': [], 'tail': [], 'other': []}
    heat_x: List[float] = []
    heat_y: List[float] = []

    def _theta_radius(gen_idx: int) -> Tuple[float, float]:
        theta = theta_max * (gen_idx / max(1, G - 1))
        base_r = r0 + (r1 - r0) * (theta / theta_max)
        return (theta, base_r)
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
        offs = _np.linspace(-0.5, 0.5, n) if n > 1 else [0.0]
        for offset, (nid, mode) in zip(offs, items):
            r = base_r + float(offset) * jitter
            x = r * _np.cos(theta)
            y = r * _np.sin(theta)
            key = mode if mode in xs else 'other'
            xs[key].append(x)
            ys[key].append(y)
            heat_x.append(x)
            heat_y.append(y)
    fig, ax = _plt.subplots(figsize=(6, 6), dpi=dpi, subplot_kw={'aspect': 'equal'})
    ts = _np.linspace(0.0, theta_max, 1200)
    rr = r0 + (r1 - r0) * (ts / theta_max)
    ax.plot(rr * _np.cos(ts), rr * _np.sin(ts), linewidth=1.0, alpha=0.32, linestyle='-')
    ring_levels = _np.linspace(r0, r1, 5)
    base_angles = _np.linspace(0.0, 2.0 * _np.pi, 361)
    for level in ring_levels:
        ax.plot(level * _np.cos(base_angles), level * _np.sin(base_angles), linestyle='--', linewidth=0.55, color='#444', alpha=0.18)
    spoke_count = max(6, int(turns * 4.0))
    for t in _np.linspace(0.0, theta_max, spoke_count, endpoint=False):
        ax.plot([r0 * _np.cos(t), r1 * _np.cos(t)], [r0 * _np.sin(t), r1 * _np.sin(t)], linestyle=':', linewidth=0.5, color='#333', alpha=0.18)
    if heat_x and heat_y:
        bins = 160
        heat_grid, xedges, yedges = _np.histogram2d(heat_x, heat_y, bins=bins, range=[[-1.05, 1.05], [-1.05, 1.05]])
        heat_grid = heat_grid.astype(float)
        if heat_grid.max() > 0:
            heat_grid /= heat_grid.max()
            kernel = _np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
            kernel /= kernel.sum()
            padded = _np.pad(heat_grid, 1, mode='reflect')
            smooth = _np.zeros_like(heat_grid)
            for i in range(heat_grid.shape[0]):
                for j in range(heat_grid.shape[1]):
                    smooth[i, j] = _np.sum(padded[i:i + 3, j:j + 3] * kernel)
            ax.imshow(smooth.T, extent=[-1.05, 1.05, -1.05, 1.05], origin='lower', cmap='inferno', alpha=0.45, interpolation='bilinear')
    markers = {'split': 'o', 'head': 's', 'tail': '^', 'other': 'x'}
    labels = {'split': 'split', 'head': 'head', 'tail': 'tail', 'other': 'other'}
    for k in ('split', 'head', 'tail', 'other'):
        if xs[k]:
            ax.scatter(xs[k], ys[k], s=marker_size * 1.1, alpha=0.28, marker=markers[k], linewidths=0.75, label=labels[k], edgecolors='black')
    max_count = max(generation_counts) if generation_counts else 0
    if max_count > 0:
        for idx in range(1, len(generation_centers)):
            prev = generation_centers[idx - 1]
            curr = generation_centers[idx]
            weight = 0.35 + 2.4 * (_np.clip(generation_counts[idx], 0, max_count) / max_count) ** 0.65
            alpha = 0.14 + 0.58 * (_np.clip(generation_counts[idx], 0, max_count) / max_count) ** 0.5
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], linewidth=weight, alpha=alpha, color='black', solid_capstyle='round')
        for (cx, cy), count in zip(generation_centers, generation_counts):
            if count <= 0:
                continue
            radius = 30.0 + 6.0 * count
            ax.scatter([cx], [cy], s=radius, facecolors='none', edgecolors='black', linewidths=0.9, alpha=0.7)
        peak_gen = int(_np.argmax(generation_counts))
        peak_count = generation_counts[peak_gen]
        if peak_count > 0:
            theta_peak, base_peak = _theta_radius(peak_gen)
            px = base_peak * _np.cos(theta_peak)
            py = base_peak * _np.sin(theta_peak)
            radial_boost = 0.12 + 0.18 * (_np.clip(peak_count / max_count, 0.0, 1.0) if max_count > 0 else 0.0)
            norm = float(_np.hypot(px, py))
            if norm < 1e-06:
                tx, ty = (radial_boost, 0.0)
            else:
                scale = norm + radial_boost
                tx = px / norm * scale
                ty = py / norm * scale
            ax.annotate(f'Peak gen {peak_gen}\n{peak_count} births', xy=(px, py), xytext=(tx, ty), textcoords='data', ha='center', va='center', fontsize=8, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', lw=0.6, alpha=0.78), arrowprops=dict(arrowstyle='-', color='black', linewidth=0.65, alpha=0.65, shrinkA=0, shrinkB=4.0, connectionstyle='arc3,rad=0.08'))
    if generation_centers:
        theta_start, r_start = _theta_radius(0)
        sx = r_start * _np.cos(theta_start)
        sy = r_start * _np.sin(theta_start)
        ax.text(sx * 1.04, sy * 1.04, 'Gen 0', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=0.5, alpha=0.85))
        theta_end, r_end = _theta_radius(G - 1)
        ex = r_end * _np.cos(theta_end)
        ey = r_end * _np.sin(theta_end)
        ax.text(ex * 1.04, ey * 1.04, f'Gen {G - 1}', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=0.5, alpha=0.85))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, loc='left')
    if any((xs[k] for k in xs)):
        ax.legend(loc='upper left', frameon=False, fontsize=9, handlelength=1.0)
    fig.tight_layout()
    _savefig(fig, out_path, dpi=dpi)
    _plt.close(fig)
    return out_path

def _softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    denom = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.maximum(denom, 1e-09)

def _import_gym():
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError as exc:
            raise ImportError('Gym/Gymnasium is not installed. RL機能は無効です。pip install gymnasium[toy_text] 等をお試しください。') from exc
    return gym

def output_dim_from_space(space) -> int:
    try:
        from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary, Box
    except Exception:
        from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box
    import numpy as _np
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiBinary):
        return int(_np.prod(space.n if hasattr(space, 'n') else space.shape))
    if isinstance(space, MultiDiscrete):
        return int(_np.sum(space.nvec))
    if isinstance(space, Box):
        return int(_np.prod(space.shape))
    raise ValueError(f'Unsupported action space: {type(space)}')

def obs_dim_from_space(space):
    if hasattr(space, 'shape') and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, 'n'):
        return int(space.n)
    raise ValueError(f'Unsupported observation space: {type(space)}')

def build_action_mapper(space, stochastic=False, temp=1.0):
    try:
        from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary, Box
    except Exception:
        from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Box
    t = max(1e-06, float(temp))
    if isinstance(space, Discrete):

        def f(logits):
            z = np.asarray(logits).reshape(-1)[:space.n]
            probs = _softmax_np(z / t).ravel()
            if stochastic:
                action = int(np.random.choice(len(probs), p=probs))
            else:
                action = int(np.argmax(z))
            return (action, probs)
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
            return (np.asarray(actions, dtype=space.dtype), np.concatenate(probs_all, axis=0))
        return f
    if isinstance(space, MultiBinary):
        n = int(np.prod(space.n if hasattr(space, 'n') else space.shape))

        def f(logits):
            z = np.asarray(logits).reshape(n)
            probs = 1.0 / (1.0 + np.exp(-(z / t)))
            if stochastic:
                actions = (np.random.random(size=n) < probs).astype(space.dtype)
            else:
                actions = (probs >= 0.5).astype(space.dtype)
            return (actions.reshape(space.shape), probs)
        return f
    if isinstance(space, Box):
        low = np.broadcast_to(space.low, space.shape).astype(float)
        high = np.broadcast_to(space.high, space.shape).astype(float)

        def f(logits):
            z = np.asarray(logits).reshape(space.shape)
            tanh = np.tanh(z)
            act = (low + high) / 2.0 + (high - low) / 2.0 * tanh
            return (act.astype(space.dtype), tanh.ravel())
        return f
    raise ValueError(f'Unsupported action space: {type(space)}')

def _cyclic_noise_profile(gen: float, ctx: Optional[Dict[str, Any]]=None) -> Tuple[float, str, Dict[str, Any]]:
    """Compute a cyclical noise profile rotating through spectral archetypes.

    The palette rotates through white noise, black noise, and rhythmic alpha/beta
    waves. Each mode receives a smooth envelope with a deterministic micro-jitter
    to keep the signal lively while remaining reproducible.
    """

    cfg = ctx or {}
    palette = tuple(cfg.get('palette', ('white', 'alpha', 'beta', 'black')))
    if not palette:
        palette = ('white',)
    stage_len = max(1, int(cfg.get('stage_len', 12)))
    jitter_amp = float(cfg.get('jitter_amp', 0.0025))
    base_levels = cfg.get('base_levels')
    if not base_levels:
        base_levels = {
            'white': (0.018, 0.008),
            'alpha': (0.021, 0.009),
            'beta': (0.024, 0.011),
            'black': (0.028, 0.013),
        }
    min_std = float(cfg.get('min_std', 0.006))
    max_std = float(cfg.get('max_std', 0.08))
    cycle_len = stage_len * len(palette)
    phase = float(gen) % max(1, cycle_len)
    stage_idx = int(phase // stage_len) % len(palette)
    local = (phase % stage_len) / stage_len
    kind = palette[stage_idx]
    base, swing = base_levels.get(kind, base_levels.get('white', (0.02, 0.01)))
    envelope = 0.5 - 0.5 * math.cos(math.tau * local)
    spectral_bias = 0.0
    band_label = 'white'
    wave_freq = None
    if kind == 'white':
        spectral_bias = 0.0
        band_label = 'white'
    elif kind == 'black':
        spectral_bias = 2.4
        band_label = 'black'
        envelope = envelope ** 1.6
    elif kind == 'alpha':
        spectral_bias = 0.7
        band_label = 'alpha'
        envelope = 0.5 + 0.5 * math.sin(math.tau * local)
        wave_freq = 10.0 + 2.5 * math.sin(math.tau * (local + 0.25))
    elif kind == 'beta':
        spectral_bias = 0.9
        band_label = 'beta'
        envelope = 0.5 + 0.5 * math.cos(math.tau * (local + 0.25))
        wave_freq = 18.0 + 5.0 * math.sin(math.tau * (local + 0.5))
    drift = swing * envelope
    jitter = jitter_amp * (
        math.sin(0.61 * float(gen)) + 0.5 * math.cos(1.37 * float(gen) + stage_idx)
    )
    if kind == 'black':
        jitter *= 1.3
    std = float(np.clip(base + drift + jitter, min_std, max_std))
    profile: Dict[str, Any] = {
        'cycle_index': float(stage_idx),
        'cycle_phase': float(local),
        'envelope': float(envelope),
        'jitter': float(jitter),
        'spectral_bias': float(spectral_bias),
    }
    if wave_freq is not None:
        profile['wave_freq_hz'] = float(wave_freq)
        profile['wave_phase'] = float((phase % stage_len) / stage_len)
    profile['band_label'] = band_label
    profile['base_level'] = float(base)
    profile['swing'] = float(swing)
    return (std, kind, profile)


@dataclass
class SpectralNoiseWeaver:
    """Blend spectral noise archetypes into resonant mixtures.

    Keeps a tempered distribution over the palette so the environment cycles
    through dominant modes without erasing supporting harmonics. By smoothing
    transitions and preserving cross-mode energy, evaluator populations perceive
    organic fluctuations rather than abrupt jumps.
    """

    palette: Tuple[str, ...] = ('white', 'alpha', 'beta', 'black')
    blend_inertia: float = 0.65
    std_inertia: float = 0.55
    focus_base: float = 0.58
    focus_surge_bonus: float = 0.18
    jitter_std: float = 0.025
    entropy_floor: float = 1e-05
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _weights: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_std: float = field(default=0.05, init=False, repr=False)
    _last_kind: str = field(default='white', init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _palette(self, ctx: Dict[str, Any]) -> Tuple[str, ...]:
        palette = tuple(ctx.get('palette') or self.palette)
        if not palette:
            palette = self.palette or ('white',)
        if not palette:
            palette = ('white',)
        return palette

    def compose(
        self,
        counter: float,
        base_std: float,
        base_kind: str,
        base_profile: Dict[str, Any],
        *,
        ctx: Dict[str, Any],
    ) -> Tuple[float, str, Dict[str, Any]]:
        palette = self._palette(ctx)
        levels = ctx.get('levels') or {}
        min_std = float(ctx.get('min_std', 0.0))
        max_std = float(ctx.get('max_std', 1.0))
        stage_len = max(1, int(ctx.get('stage_len', max(1, len(palette)))))
        surge = bool(ctx.get('surge', False))
        intensity = float(ctx.get('intensity', 0.0))
        focus_target = self.focus_base + self.focus_surge_bonus * (1.0 if surge else 0.0)
        focus_target += 0.12 * math.tanh((intensity - 0.2) / 0.18)
        focus_target = float(np.clip(focus_target, 0.35, 0.96))
        background = max(1e-06, 1.0 - focus_target)
        new_weights: Dict[str, float] = {}
        prev_weights = self._weights or {mode: 1.0 / len(palette) for mode in palette}
        for idx, mode in enumerate(palette):
            prev = float(prev_weights.get(mode, 1.0 / len(palette)))
            if mode == base_kind:
                target = focus_target
            else:
                target = background / max(1, len(palette) - 1)
            blend = self.blend_inertia * prev + (1.0 - self.blend_inertia) * target
            jitter = float(self._rng.normal(0.0, self.jitter_std))
            new_weights[mode] = max(self.entropy_floor, blend + jitter)
        total = float(sum(new_weights.values()))
        if not math.isfinite(total) or total <= 0.0:
            new_weights = {mode: 1.0 / len(palette) for mode in palette}
            total = 1.0
        weights = {mode: float(val / total) for mode, val in new_weights.items()}
        stage_phase = (counter % stage_len) / float(stage_len)
        mix_std = 0.0
        harmonics: Dict[str, float] = {}
        for idx, mode in enumerate(palette):
            base_level, swing = levels.get(mode, levels.get(base_kind, (base_std, 0.0)))
            phase = (stage_phase + idx / max(1, len(palette))) % 1.0
            envelope = 0.5 - 0.5 * math.cos(math.tau * phase)
            if mode == 'alpha':
                envelope = 0.5 + 0.5 * math.sin(math.tau * phase)
            elif mode == 'beta':
                envelope = 0.5 + 0.5 * math.cos(math.tau * (phase + 0.25))
            elif mode == 'black':
                envelope = envelope ** 1.5
            mode_std = float(np.clip(base_level + swing * envelope, min_std, max_std))
            weight = weights.get(mode, 0.0)
            harmonics[mode] = weight
            mix_std += weight * mode_std
        mix_std = self.std_inertia * float(self._last_std) + (1.0 - self.std_inertia) * mix_std
        mix_std = float(np.clip(mix_std, min_std, max_std))
        arr = np.asarray(list(weights.values()), dtype=np.float64)
        if arr.size:
            focus_val = float(np.max(arr))
            entropy = float(-np.sum(arr * np.log(arr + 1e-09)))
        else:
            focus_val = 1.0
            entropy = 0.0
        dominant = max(weights.items(), key=lambda kv: kv[1])[0]
        profile = dict(base_profile)
        profile['harmonics'] = {k: float(v) for k, v in harmonics.items()}
        profile['mix_entropy'] = entropy
        profile['mix_focus'] = focus_val
        profile['headline_kind'] = dominant
        profile['previous_kind'] = self._last_kind
        self._weights = weights
        self._last_std = mix_std
        self._last_kind = dominant
        return (mix_std, dominant, profile)


def _default_difficulty_schedule(gen: int, _ctx: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Enhanced curriculum with chaotic difficulty fluctuations and environmental monotony.

    Creates a challenging environment with:
    - Non-linear difficulty progression with sudden jumps and drops
    - Multiple overlapping oscillations for unpredictability
    - Occasional difficulty spikes and valleys
    - More stable (monotonous) noise levels for consistent environmental conditions
    
    Parameters can be customized via _ctx if needed:
    - chaos_amp1, chaos_freq1: First chaotic component amplitude and frequency
    - chaos_amp2, chaos_freq2: Second chaotic component amplitude and frequency
    - spike_prob: Probability of difficulty spikes
    - spike_amp: Amplitude of difficulty spikes
    """
    ctx = _ctx or {}
    noise_ctx = ctx.get('noise_ctx', {})
    noise_min = float(noise_ctx.get('min_std', 0.006))
    noise_max = float(noise_ctx.get('max_std', 0.06))
    noise_std_raw, noise_kind, noise_profile = _cyclic_noise_profile(
        gen,
        {
            'palette': noise_ctx.get('palette', ('white', 'alpha', 'beta', 'black')),
            'stage_len': noise_ctx.get('stage_len', 16),
            'jitter_amp': noise_ctx.get('jitter_amp', 0.0025),
            'base_levels': noise_ctx.get('base_levels'),
            'min_std': noise_min,
            'max_std': noise_max,
        },
    )

    def _payload(diff_val: float, regen_flag: bool, scale: float) -> Dict[str, Any]:
        scaled_noise = float(np.clip(noise_std_raw * scale, noise_min, noise_max))
        profile = dict(noise_profile)
        profile['stage_scale'] = float(scale)
        profile['cycle_gen'] = int(gen)
        return {
            'difficulty': float(diff_val),
            'noise_std': scaled_noise,
            'noise_kind': noise_kind,
            'noise_profile': profile,
            'enable_regen': regen_flag,
        }

    chaos_amp1 = ctx.get('chaos_amp1', 0.35)
    chaos_freq1_sin = ctx.get('chaos_freq1_sin', 0.28)
    chaos_freq1_cos = ctx.get('chaos_freq1_cos', 0.11)
    chaos_amp2 = ctx.get('chaos_amp2', 0.25)
    chaos_freq2 = ctx.get('chaos_freq2', 0.45)
    chaos_amp3 = ctx.get('chaos_amp3', 0.18)
    chaos_freq3 = ctx.get('chaos_freq3', 0.67)
    spike_prob = ctx.get('spike_prob', 0.12)
    spike_amp = ctx.get('spike_amp', 0.8)
    spike_threshold_phase3 = spike_prob * 100
    spike_threshold_phase4 = spike_prob * 100
    if gen < 15:
        micro_var = 0.05 * np.sin(gen * 0.8)
        return _payload(0.4 + micro_var, False, 0.6)
    if gen < 30:
        base_diff = 0.5 + (gen - 15) * 0.04
        wave1 = 0.15 * np.sin((gen - 15) * 0.35)
        wave2 = 0.1 * np.cos((gen - 15) * 0.52)
        diff = base_diff + wave1 + wave2
        return _payload(max(0.3, diff), False, 0.85)
    if gen < 60:
        base_diff = 1.0 + (gen - 30) * 0.045
        wave1 = 0.25 * np.sin((gen - 30) * 0.22)
        wave2 = 0.2 * np.cos((gen - 30) * 0.38)
        wave3 = 0.15 * np.sin((gen - 30) * 0.61)
        drop_component = -0.4 if (gen - 30) % 17 < 2 else 0.0
        spike_component = spike_amp if gen * 7 % 100 < spike_threshold_phase3 else 0.0
        diff = base_diff + wave1 + wave2 + wave3 + drop_component + spike_component
        return _payload(max(0.5, diff), True, 1.15)
    base_diff = 2.0 + (gen - 60) * 0.03
    chaos1 = chaos_amp1 * np.sin((gen - 60) * chaos_freq1_sin) * np.cos((gen - 60) * chaos_freq1_cos)
    chaos2 = chaos_amp2 * np.cos((gen - 60) * chaos_freq2)
    chaos3 = chaos_amp3 * np.sin((gen - 60) * chaos_freq3)
    jump_component = 0.6 if (gen - 60) % 23 < 3 else 0.0
    drop_component = -0.7 if (gen - 60) % 19 < 2 else 0.0
    spike_component = spike_amp if gen * 13 % 100 < spike_threshold_phase4 else 0.0
    diff = base_diff + chaos1 + chaos2 + chaos3 + jump_component + drop_component + spike_component
    return _payload(max(0.3, diff), True, 1.35)

def _apply_stable_neat_defaults(neat: ReproPlanaNEATPlus):
    """Enhanced defaults for complex topology survival under challenging environments."""
    neat.mode = EvalMode(vanilla=True, enable_regen_reproduction=False, complexity_alpha=neat.mode.complexity_alpha, node_penalty=0.15, edge_penalty=0.08, species_low=neat.mode.species_low, species_high=neat.mode.species_high)
    neat.mutate_add_conn_prob = 0.08
    neat.mutate_add_node_prob = 0.05
    neat.mutate_weight_prob = 0.8

    # Sex & hermaphrodite tuning
    neat.hermaphrodite_init_ratio = getattr(neat, 'hermaphrodite_init_ratio', 0.12)
    neat.mutate_sex_prob = 0.01
    neat.hermaphrodite_inheritance_rate = 0.12
    neat.hermaphrodite_mate_bias = 2.5
    # Regen mix (lower baseline)
    neat.mix_asexual_base = 0.08
    neat.mix_asexual_gain = 0.30
    # Adaptive self-selected mutation
    neat.adaptive_self_mutation = True
    neat.self_mutation_gain = 0.6
    neat.self_mutation_limit = 0.5
    neat.self_mutation_complexity_penalty = 0.35
    neat.regen_mode_mut_rate = 0.08
    neat.mix_asexual_base = 0.1
    neat.complexity_threshold = 8.0
    neat.max_hidden_nodes = 256
    neat.max_edges = 2048

def setup_neat_for_env(env_id: str, population: int=48, output_activation: str='identity'):
    gym = _import_gym()
    env = gym.make(env_id)
    obs_dim = obs_dim_from_space(env.observation_space)
    out_dim = output_dim_from_space(env.action_space)
    neat_module = sys.modules[__name__]
    neat = neat_module.ReproPlanaNEATPlus(num_inputs=obs_dim, num_outputs=out_dim, population_size=population, output_activation=output_activation, rng=np.random.default_rng())
    _apply_stable_neat_defaults(neat)
    return (neat, env)

def _rollout_policy_in_env(genome, env, mapper, max_steps=None, render=False, obs_norm=None):
    """Rollout one episode with a Genome and an action mapper."""
    total, steps, done = (0.0, 0, False)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) >= 1 else reset_out
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
        if max_steps is not None and steps >= int(max_steps):
            break
    return total

def gym_fitness_factory(env_id, stochastic=False, temp=1.0, max_steps=1000, episodes=1, obs_norm=None):
    """Return a fitness function for evolve() that evaluates average episodic reward."""
    gym = _import_gym()
    try:
        env = gym.make(env_id, render_mode='rgb_array')
    except TypeError:
        env = gym.make(env_id)
    mapper = build_action_mapper(env.action_space, stochastic=stochastic, temp=temp)
    n_episodes = max(1, int(episodes))

    def _fitness(genome):
        total = 0.0
        for _ in range(n_episodes):
            total += _rollout_policy_in_env(genome, env, mapper, max_steps=max_steps, render=False, obs_norm=obs_norm)
        return total / float(n_episodes)

    def _close_env():
        try:
            env.close()
        except Exception:
            pass
    _fitness.close_env = _close_env
    _fitness.env = env
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
    return (logits, acts, pre)

def _color_from_value(v: float):
    v = float(np.tanh(v))
    r = 0.2 + 0.6 * max(0.0, v)
    b = 0.2 + 0.6 * max(0.0, -v)
    g = 0.22
    return (r, g, b)

def _draw_nn(ax, genome: 'Genome', acts: Dict[int, float], show_values: bool=False, scars=None, radius: float=0.1):
    pos = layout_by_depth(genome)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    for c in genome.enabled_connections():
        i, o = (c.in_node, c.out_node)
        if i not in pos or o not in pos:
            continue
        p1, p2 = (pos[i], pos[o])
        lw = 0.6 + 2.2 * min(1.0, abs(c.weight))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, alpha=0.75)
    for nid, nd in genome.nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        fc = _color_from_value(acts.get(nid, 0.0))
        ax.add_patch(Circle((x, y), radius=radius, color=fc, alpha=0.95))
        lw = 1.2
        if scars and nid in scars:
            age = scars[nid].age
            lw = 1.2 + 0.8 * math.exp(-0.15 * age)
        ax.add_patch(Circle((x, y), radius=radius, fill=False, linewidth=lw, alpha=0.95))
        ax.text(x, y - radius * 1.7, nd.type[0], ha='center', va='top', fontsize=7, alpha=0.86)
        if show_values:
            ax.text(x, y + radius * 1.6, f'{acts.get(nid, 0.0):+.2f}', ha='center', va='bottom', fontsize=7, alpha=0.9)

def _draw_prob_bars(ax, probs, title='Action probabilities'):
    probs = np.asarray(probs).ravel()
    ax.clear()
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(probs) - 0.5)
    ax.bar(np.arange(len(probs)), probs, alpha=0.9)
    ax.set_xticks(range(len(probs)))
    ax.set_title(title, fontsize=9)
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=7)
    ax.grid(alpha=0.15, linestyle=':')

def _episode_bc_update(genome: 'Genome', obs_list, act_list, ret_list, steps=20, lr=0.01, l2=0.0001, top_frac=0.3):
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
        print('[warn] online update skipped:', e)

def run_policy_in_env(genome: 'Genome', env_id: str, episodes: int=1, max_steps: int=1000, stochastic: bool=True, temp: float=1.0, out_gif: str='out/rl_rollout.gif', fps: int=20, panel_ratio: float=0.58, show_values: bool=True, show_bars: bool=True, rl_update: bool=False, gamma: float=0.99, rl_steps: int=20, rl_lr: float=0.01, rl_l2: float=0.0001, top_frac: float=0.3):
    try:
        import gymnasium as gym
    except Exception:
        import gym
    try:
        env = gym.make(env_id, render_mode='rgb_array')
    except TypeError:
        env = gym.make(env_id)
    mapper = build_action_mapper(env.action_space, stochastic=stochastic, temp=temp)
    frames = []
    for ep in range(int(episodes)):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) >= 1 else reset_out
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        ep_obs, ep_act, ep_rew = ([], [], [])
        for t in range(int(max_steps)):
            logits, acts, _pre = eval_with_node_activations(genome, obs)
            mapped = mapper(logits)
            if isinstance(mapped, tuple):
                action, probs = mapped
            else:
                action, probs = (mapped, None)
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
                ax_env.text(0.5, 0.5, '(render unavailable)', ha='center', va='center')
                ax_env.axis('off')
            _draw_nn(ax_nn, genome, acts, show_values=show_values)
            ax_env.set_title(f'{env_id} | ep {ep + 1} t={t} r={rew:.2f}')
            ax_nn.set_title('Policy network (activations)')
            if show_bars and probs is not None:
                _draw_prob_bars(ax_prob, probs, title='Action probabilities')
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
                    if len(rets_np) > 1 and np.std(rets_np) > 1e-08:
                        rets_np = (rets_np - np.mean(rets_np)) / (np.std(rets_np) + 1e-08)
                    if all((a is not None for a in ep_act)):
                        _episode_bc_update(genome, ep_obs, [int(a) for a in ep_act], rets_np, steps=rl_steps, lr=rl_lr, l2=rl_l2, top_frac=top_frac)
                break
    os.makedirs(os.path.dirname(out_gif) or '.', exist_ok=True)
    _mimsave(out_gif, frames, fps=fps)
    try:
        env.close()
    except Exception:
        pass
    return out_gif

def run_gym_neat_experiment(env_id: str, gens: int=20, pop: int=24, episodes: int=1, max_steps: int=500, stochastic: bool=False, temp: float=1.0, out_prefix: str='out/rl') -> Dict[str, Any]:
    """Convenience wrapper that evolves NEAT agents on a Gym environment."""
    neat, env = setup_neat_for_env(env_id, population=pop, output_activation='identity')
    regen_log_path = f'{out_prefix}_regen_log.csv'
    if hasattr(neat, 'lcs_monitor') and neat.lcs_monitor is not None:
        neat.lcs_monitor.csv_path = regen_log_path
        if os.path.exists(regen_log_path):
            os.remove(regen_log_path)
    try:
        env.close()
    except Exception:
        pass
    fit = gym_fitness_factory(env_id, stochastic=stochastic, temp=temp, max_steps=max_steps, episodes=episodes)
    best, hist = neat.evolve(fit, n_generations=gens, verbose=True, env_schedule=_default_difficulty_schedule)
    lcs_rows = load_lcs_log(regen_log_path) if os.path.exists(regen_log_path) else []
    lcs_series = _prepare_lcs_series(lcs_rows) if lcs_rows else None
    close_env = getattr(fit, 'close_env', None)
    if callable(close_env):
        close_env()
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    rc_png = f'{out_prefix}_reward_curve.png'
    xs = list(range(len(hist)))
    ys_b = [b for b, _a in hist]
    ys_a = [a for _b, a in hist]
    plt.figure()
    plt.plot(xs, ys_b, label='best')
    plt.plot(xs, ys_a, label='avg')
    plt.xlabel('generation')
    plt.ylabel('episode reward')
    plt.title(f'{env_id} | Average Episode Reward')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    _savefig(fig, rc_png, dpi=150)
    plt.close()
    lcs_ribbon = None
    lcs_timeline = None
    if lcs_rows:
        ribbon_path = f'{out_prefix}_lcs_ribbon.png'
        try:
            export_lcs_ribbon_png(lcs_rows, ribbon_path, series=lcs_series)
            lcs_ribbon = ribbon_path
        except Exception as ribbon_err:
            print('[WARN] LCS ribbon export failed:', ribbon_err)
        timeline_path = f'{out_prefix}_lcs_timeline.gif'
        try:
            export_lcs_timeline_gif(lcs_rows, timeline_path, series=lcs_series, fps=6)
            lcs_timeline = timeline_path
        except Exception as timeline_err:
            print('[WARN] LCS timeline export failed:', timeline_err)
    return {'best': best, 'history': hist, 'reward_curve': rc_png, 'lcs_log': regen_log_path if os.path.exists(regen_log_path) else None, 'lcs_ribbon': lcs_ribbon, 'lcs_timeline': lcs_timeline}

def _genome_to_cyto(genome: Genome) -> dict:
    """
    Convert a Genome object to Cytoscape.js-compatible dictionary format.
    Returns dict with 'nodes', 'edges', 'id', and 'meta' keys.
    """
    nodes = []
    for nid, node in genome.nodes.items():
        node_data = {
            'id': str(nid),
            'label': f'{node.type[0].upper()}{nid}',
            'type': node.type,
            'bias': getattr(node, 'bias', 0.0),
            'activation': node.activation,
            'sensitivity': float(getattr(node, 'backprop_sensitivity', 1.0)),
            'jitter': float(getattr(node, 'sensitivity_jitter', 0.0)),
            'momentum': float(getattr(node, 'sensitivity_momentum', 0.0)),
            'variance': float(getattr(node, 'sensitivity_variance', 0.0)),
            'altruism': float(np.clip(getattr(node, 'altruism', 0.5), 0.0, 1.0)),
            'altruism_memory': float(np.clip(getattr(node, 'altruism_memory', 0.0), -1.5, 1.5)),
            'altruism_span': float(np.clip(getattr(node, 'altruism_span', 0.0), 0.0, 4.0)),
        }
        nodes.append({'data': node_data})
    edges = []
    for innov, conn in genome.connections.items():
        edge_weight = float(conn.weight)
        edge_data = {
            'id': f'e{innov}',
            'source': str(conn.in_node),
            'target': str(conn.out_node),
            'weight': edge_weight,
            'abs_weight': abs(edge_weight),
            'enabled': bool(conn.enabled),
        }
        edges.append({'data': edge_data})
    return {'id': genome.id, 'nodes': nodes, 'edges': edges, 'meta': {'fitness': getattr(genome, 'fitness', None), 'birth_gen': genome.birth_gen}}

def export_interactive_html_report(path: str, title: str, history, genomes, *, max_genomes: int=50):
    """
    Write a self-contained interactive HTML report with:
    - Plotly learning curve (pan/zoom/hover)
    - Cytoscape genome viewer with:
        - Right-click context menu on nodes: Fix/Unfix, Color, Note
        - Drag nodes, zoom, click for detail
    No extra Python deps; loads JS via CDN. Per-node edits persist in localStorage.
    """
    import json, os
    xs = list(range(len(history or [])))
    ys_best = [float(b) for b, a in history or []]
    ys_avg = [float(a) for b, a in history or []]
    genomes = genomes or []
    if len(genomes) > max_genomes:
        idxs = [round(i) for i in [k * (len(genomes) - 1) / (max_genomes - 1) for k in range(max_genomes)]]
        genomes = [genomes[i] for i in idxs]
    if genomes and 'nodes' not in genomes[0]:
        genomes = [_genome_to_cyto(g) for g in genomes]
    build_id = _build_stamp_short()
    data = {'title': title, 'lc': {'x': xs, 'best': ys_best, 'avg': ys_avg}, 'genomes': genomes, 'build': build_id}
    html = f"""<!doctype html>\n<html lang="ja">\n<head>\n  <meta charset="utf-8"/>\n  <title>{title}</title>\n  <meta name="viewport" content="width=device-width, initial-scale=1"/>\n  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>\n  <style>\n    :root {{ --grid:#e5e5e5; --fg:#111; --muted:#666; --ok1:#0072B2; --ok2:#D55E00; --ok3:#009E73; --ok4:#CC79A7; --ok5:#F0E442; --ok6:#56B4E9; --ok7:#E69F00; --ok8:#000; }}\n    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--fg); }}\n    .container {{ max-width:1200px; margin:24px auto; padding:0 16px; }}\n    h1 {{ font-size:22px; margin:0 0 12px; }}\n    .card {{ border:1px solid var(--grid); border-radius:8px; padding:12px; background:#fff; margin-bottom:16px; }}\n    #altruismCard {{ margin-top:12px; }}
    #altruismSummary {{ font-size:12px; color:var(--muted); margin-bottom:8px; }}
    #altruismPlot {{ width:100%; height:200px; }}
    #lc {{ height:360px; }}\n    .panel {{ display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:16px; align-items:start; overflow:hidden; }}\n    .panel > * {{ min-width:0; }}\n    #cy {{ width:100%; height:560px; border:1px solid var(--grid); border-radius:6px; background:#fff; box-sizing:border-box; }}\n    .detail {{ border:1px dashed var(--grid); border-radius:6px; padding:8px; font-size:13px; height:560px; overflow:auto; background:#fafafa; box-sizing:border-box; }}\n    .row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }}\n    .row.controls {{ margin-top:4px; font-size:13px; color:var(--muted); }}\n    .row.controls label {{ font-size:13px; color:var(--fg); display:flex; align-items:center; gap:4px; }}\n    select,button,input[type=range] {{ padding:6px 8px; font-size:14px; border:1px solid var(--grid); border-radius:6px; background:#fff; }}\n    input[type=range] {{ padding:0; width:160px; accent-color:#0072B2; }}\n    .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}\n    .legend {{ display:inline-flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:13px; }}\n    .legend-title {{ font-weight:600; color:var(--muted); margin-right:2px; }}\n    .legend-gradient {{ width:120px; height:10px; border-radius:4px; border:1px solid var(--grid); background:linear-gradient(90deg,#2c7bb6,#abd9e9,#ffffbf,#fdae61,#d7191c); }}\n    .legend-text {{ color:var(--muted); font-size:12px; }}\n    .checkbox {{ display:inline-flex; align-items:center; gap:6px; padding:4px 6px; border:1px solid var(--grid); border-radius:6px; background:#fff; color:var(--fg); }}\n    .checkbox input {{ margin:0; }}\n    #genomeStats {{ font-size:13px; color:var(--muted); margin-top:8px; display:grid; gap:4px; line-height:1.4; }}\n    @media (max-width: 900px) {{\n      .panel {{ grid-template-columns:1fr; }}\n      .detail {{ height:auto; min-height:220px; }}\n      #cy {{ height:420px; }}\n    }}\n    /* Tooltip */\n    .tip {{ position:fixed; pointer-events:none; background:rgba(0,0,0,.8); color:#fff; padding:6px 8px; font-size:12px; border-radius:4px; transform:translate(8px,8px); z-index:1000; display:none; max-width:320px; white-space:nowrap; }}\n    /* Context menu */\n    .ctx-menu {{\n      position: fixed; z-index: 2000; display: none; min-width: 220px;\n      background: #fff; color: var(--fg); border: 1px solid var(--grid); border-radius: 8px;\n      box-shadow: 0 8px 20px rgba(0,0,0,.12); padding: 6px;\n    }}\n    .ctx-item {{ font-size: 13px; padding: 8px 10px; cursor: pointer; border-radius:6px; }}\n    .ctx-item:hover {{ background: #f3f3f3; }}\n    .ctx-sep {{ height:1px; background: var(--grid); margin:6px 0; }}\n    .swatches {{ display:flex; flex-wrap:wrap; gap:6px; padding: 4px 2px 2px; }}\n    .swatch {{ width:18px; height:18px; border-radius:50%; cursor:pointer; border:1px solid rgba(0,0,0,.15); }}\n    .ctx-row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}\n  </style>\n</head>\n<body>\n  <div class="container">\n    <h1>{title}</h1>\n    <div class="card">\n      <h2>Learning Curve</h2>\n      <div id="lc"></div>\n    </div>\n    <div class="card">\n      <h2>Genome Viewer</h2>\n      <div class="row">\n        <label for="genomeSelect">Genome:</label>\n        <select id="genomeSelect"></select>\n        <button id="layoutBtn">Re-layout</button>\n        <span class="legend" id="legendBox"></span>\n      </div>\n      <div class="row controls">\n        <label for="nodeColorMode">ノード色:</label>\n        <select id="nodeColorMode">\n          <option value="type">種類</option>\n          <option value="sensitivity">感受性</option>\n          <option value="momentum">モメンタム</option>\n          <option value="variance">分散</option>\n          <option value="altruism">利他性</option>\n          <option value="altruism_memory">利他メモリ</option>\n          <option value="altruism_span">利他スパン</option>\n        </select>\n        <label for="weightFilter">|weight| ≥ <span id="weightFilterValue">0</span></label>\n        <input type="range" id="weightFilter" min="0" max="100" step="1" value="0"/>\n        <label class="checkbox"><input type="checkbox" id="enabledOnly"/> 有効エッジのみ</label>\n      </div>\n      <div class="panel">\n        <div id="cy"></div>\n        <div class="detail" id="detail"><div class="fine">ノードやエッジを選択すると詳細が表示されます。</div></div>\n      </div>\n      <div id="genomeStats"></div>
      <div class="card" id="altruismCard">
        <div class="fine" id="altruismSummary">利他性と集団信号の要約がここに表示されます。</div>
        <div id="altruismPlot"></div>
      </div>\n    </div>\n  </div>\n  <div class="tip" id="tip"></div>\n  <div class="ctx-menu" id="ctx"></div>\n\n  <script>\n    const DATA = {json.dumps(data, ensure_ascii=False)};\n    // Learning curve\n    (function(){{\n      const traces = [];\n      if (DATA.lc && DATA.lc.x.length) {{\n        traces.push({{ x: DATA.lc.x, y: DATA.lc.best, mode:'lines', name:'best', line:{{width:2, color:'#0072B2'}},\n          hovertemplate:'gen=%{{x}}<br>best=%{{y:.4f}}<extra></extra>' }});\n        traces.push({{ x: DATA.lc.x, y: DATA.lc.avg,  mode:'lines', name:'avg',  line:{{width:2, color:'#D55E00'}},\n          hovertemplate:'gen=%{{x}}<br>avg=%{{y:.4f}}<extra></extra>' }});\n      }}\n      Plotly.newPlot('lc', traces, {{\n        margin:{{l:40,r:10,t:10,b:40}},\n        xaxis:{{title:'Generation', gridcolor:'#eee'}},\n        yaxis:{{title:'Fitness', gridcolor:'#eee'}},\n        legend:{{orientation:'h'}}\n      }}, {{displayModeBar:true, responsive:true}});\n    }})();\n\n    // Genome viewer\n    const cyContainer = document.getElementById('cy');\n    const detail = document.getElementById('detail');\n    const tip = document.getElementById('tip');\n    const sel = document.getElementById('genomeSelect');\n    const layoutBtn = document.getElementById('layoutBtn');\n    const ctx = document.getElementById('ctx');\n    const colorModeSelect = document.getElementById('nodeColorMode');\n    const weightSlider = document.getElementById('weightFilter');\n    const weightLabel = document.getElementById('weightFilterValue');\n    const enabledOnly = document.getElementById('enabledOnly');\n    const legendBox = document.getElementById('legendBox');\n    const statsBox = document.getElementById('genomeStats');\n    const altruismSummaryBox = document.getElementById('altruismSummary');\n    const altruismPlotBox = document.getElementById('altruismPlot');\n    let cy = null;\n    let currentGenome = null; // store current genome object\n    let ctxNode = null;\n    const globalStats = computeGlobalStats();\n    let currentEdgeMax = globalStats.maxAbsWeight || 0;\n    let activeColorMode = (colorModeSelect && colorModeSelect.value) || 'type';\n    updateLegend(activeColorMode);\n    applyEdgeFilters();\n\n    function lsKey(gid) {{ return 'NEAT_REPORT_GENOME_STATE::' + String(gid); }}\n    function saveState() {{\n      if (!cy || !currentGenome) return;\n      const st = {{}};\n      cy.nodes().forEach(n => {{\n        st[n.id()] = {{\n          pos: n.position(),\n          locked: n.locked(),\n          color: n.data('color') || null,\n          note: n.data('note') || null\n        }};\n      }});\n      try {{ localStorage.setItem(lsKey(currentGenome.id || 'genome'), JSON.stringify(st)); }} catch(e){{}}\n    }}\n    function applyState() {{\n      if (!cy || !currentGenome) return;\n      let raw = null;\n      try {{ raw = localStorage.getItem(lsKey(currentGenome.id || 'genome')); }} catch(e){{}}\n      if (!raw) return;\n      let st = null;\n      try {{ st = JSON.parse(raw); }} catch(e) {{ return; }}\n      if (!st) return;\n      cy.batch(() => {{\n        cy.nodes().forEach(n => {{\n          const s = st[n.id()]; if (!s) return;\n          if (s.pos && Number.isFinite(s.pos.x) && Number.isFinite(s.pos.y)) n.position(s.pos);\n          if (s.locked) n.lock(); else n.unlock();\n          if (s.color) {{ n.data('color', s.color); n.style('background-color', s.color); }}\n          if (s.note) {{\n            n.data('note', s.note);\n            const orig = n.data('orig_label') || n.data('label') || n.id();\n            n.data('label', orig + '\\n' + s.note);\n          }}\n        }});\n      }});\n    }}\n\n    
function updateRange(range, value) {{
  if (!Number.isFinite(value)) return;
  if (value < range[0]) range[0] = value;
  if (value > range[1]) range[1] = value;
}}
function normalizeRange(range, fallback) {{
  if (!Number.isFinite(range[0]) || !Number.isFinite(range[1]) || range[0] === Infinity || range[1] === -Infinity) {{
    range[0] = fallback;
    range[1] = fallback;
  }}
}}
function computeGlobalStats() {{
  const ranges = {{
    sensitivity: [Infinity, -Infinity],
    momentum: [Infinity, -Infinity],
    variance: [Infinity, -Infinity],
    altruism: [Infinity, -Infinity],
    altruism_memory: [Infinity, -Infinity],
    altruism_span: [Infinity, -Infinity],
  }};
  let maxAbsWeight = 0;
  (DATA.genomes || []).forEach(g => {{
    (g.nodes || []).forEach(n => {{
      const d = n.data || n;
      updateRange(ranges.sensitivity, Number(d.sensitivity));
      updateRange(ranges.momentum, Number(d.momentum));
      updateRange(ranges.variance, Number(d.variance));
      updateRange(ranges.altruism, Number(d.altruism));
      updateRange(ranges.altruism_memory, Number(d.altruism_memory));
      updateRange(ranges.altruism_span, Number(d.altruism_span));
    }});
    (g.edges || []).forEach(e => {{
      const d = e.data || e;
      const absW = Math.abs(Number(d.abs_weight !== undefined ? d.abs_weight : d.weight));
      if (Number.isFinite(absW) && absW > maxAbsWeight) maxAbsWeight = absW;
    }});
  }});
  normalizeRange(ranges.sensitivity, 1);
  normalizeRange(ranges.momentum, 0);
  normalizeRange(ranges.variance, 0);
  normalizeRange(ranges.altruism, 0.5);
  normalizeRange(ranges.altruism_memory, 0);
  normalizeRange(ranges.altruism_span, 0);
  return {{ ranges, maxAbsWeight }};
}}
function formatNumber(value, digits) {{
  if (!Number.isFinite(value)) return '–';
  const places = Number.isFinite(digits) ? digits : 2;
  return Number(value).toFixed(places);
}}
function gradientColor(t) {{
  const stops = [
    [44, 123, 182],
    [171, 217, 233],
    [253, 174, 97],
    [215, 25, 28],
  ];
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const start = stops[idx];
  const end = stops[idx + 1];
  const comp = start.map((s, i) => Math.round(s + (end[i] - s) * frac));
  return '#' + comp.map(c => c.toString(16).padStart(2, '0')).join('');
}}
function calcStats(values) {{
  const arr = values.filter(v => Number.isFinite(v));
  if (!arr.length) return {{ mean: NaN, std: NaN }};
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
  return {{ mean, std: Math.sqrt(variance) }};
}}
function computeEdgeMax(g) {{
  let max = 0;
  (g.edges || []).forEach(e => {{
    const d = e.data || e;
    const absW = Math.abs(Number(d.abs_weight !== undefined ? d.abs_weight : d.weight));
    if (Number.isFinite(absW) && absW > max) max = absW;
  }});
  return max;
}}
function applyNodeColorMode(mode) {{
  activeColorMode = mode || 'type';
  updateLegend(activeColorMode);
  if (!cy) return;
  const keyMap = {{ sensitivity: 'sensitivity', momentum: 'momentum', variance: 'variance', altruism: 'altruism', altruism_memory: 'altruism_memory', altruism_span: 'altruism_span' }};
  const range = globalStats.ranges[activeColorMode] || [0, 1];
  cy.batch(() => {{
    cy.nodes().forEach(n => {{
      const data = n.data();
      const manual = data.color;
      if (manual) {{
        n.data('vizColor', manual);
        n.style('background-color', manual);
        return;
      }}
      let color = '#999999';
      if (activeColorMode === 'type') {{
        if (data.type === 'input') color = '#009E73';
        else if (data.type === 'output') color = '#0072B2';
        else color = '#999999';
      }} else {{
        const key = keyMap[activeColorMode];
        const value = Number(data[key]);
        let t = 0.5;
        if (Number.isFinite(value)) {{
          const [min, max] = range;
          t = (max > min) ? (value - min) / (max - min) : 0.5;
        }}
        color = gradientColor(t);
      }}
      n.data('vizColor', color);
      n.style('background-color', color);
    }});
  }});
}}
function applyEdgeVisuals() {{
  if (!cy) return;
  const denom = currentEdgeMax > 0 ? currentEdgeMax : 1;
  cy.batch(() => {{
    cy.edges().forEach(e => {{
      const data = e.data();
      const w = Number(data.weight) || 0;
      const absW = Math.abs(w);
      const baseColor = w >= 0 ? '#56B4E9' : '#D55E00';
      const width = 0.6 + 3.8 * (absW / denom);
      const clampedWidth = Math.max(0.6, Math.min(width, 6));
      e.data('vizColor', baseColor);
      e.data('vizWidth', clampedWidth);
      e.style('line-color', baseColor);
      e.style('target-arrow-color', baseColor);
      e.style('width', clampedWidth);
    }});
  }});
}}
function applyEdgeFilters() {{
  const sliderVal = weightSlider ? Number(weightSlider.value) : 0;
  const threshold = (currentEdgeMax || 0) * (sliderVal / 100);
  if (weightLabel) weightLabel.textContent = formatNumber(threshold, 3);
  if (!cy) return;
  cy.batch(() => {{
    cy.edges().forEach(edge => {{
      const data = edge.data();
      const absW = Math.abs(Number(data.weight) || 0);
      let hide = false;
      if (threshold > 0 && absW < threshold - 1e-9) hide = true;
      if (enabledOnly && enabledOnly.checked && !data.enabled) hide = true;
      if (hide) edge.addClass('hidden'); else edge.removeClass('hidden');
    }});
  }});
}}
function updateLegend(mode) {{
  if (!legendBox) return;
  if (mode === 'type') {{
    legendBox.innerHTML = '<span class="legend-title">Node:</span>' +
      '<span class="dot" style="background:#009E73"></span> input ' +
      '<span class="dot" style="background:#999999"></span> hidden ' +
      '<span class="dot" style="background:#0072B2"></span> output';
  }} else {{
    const range = globalStats.ranges[mode] || [0, 0];
    legendBox.innerHTML = '<span class="legend-title">Node:</span>' +
      '<span class="legend-gradient"></span>' +
      `<span class="legend-text">low ${{formatNumber(range[0], 2)}}</span>` +
      `<span class="legend-text">high ${{formatNumber(range[1], 2)}}</span>`;
  }}
}}
function updateGenomeStats(g) {{
  if (!statsBox) return;
  const nodes = g.nodes || [];
  const edges = g.edges || [];
  const enabledCount = edges.filter(e => ((e.data||e).enabled)).length;
  const values = key => nodes.map(n => Number((n.data || n)[key])).filter(v => Number.isFinite(v));
  const sens = calcStats(values('sensitivity'));
  const momentum = calcStats(values('momentum'));
  const jitter = calcStats(values('jitter'));
  const variance = calcStats(values('variance'));
  const altruism = calcStats(values('altruism'));
  const altruismMem = calcStats(values('altruism_memory'));
  const altruismSpan = calcStats(values('altruism_span'));
  const parts = [
    `<div>Nodes: ${{nodes.length}} / Edges: ${{edges.length}} (enabled ${{enabledCount}})</div>`,
    `<div>感受性 μ=${{formatNumber(sens.mean, 3)}} σ=${{formatNumber(sens.std, 3)}}</div>`,
    `<div>モメンタム μ=${{formatNumber(momentum.mean, 3)}} σ=${{formatNumber(momentum.std, 3)}} ・ ジッター μ=${{formatNumber(jitter.mean, 3)}} σ=${{formatNumber(jitter.std, 3)}} ・ 分散 μ=${{formatNumber(variance.mean, 3)}}</div>`,
    `<div>利他性 μ=${{formatNumber(altruism.mean, 3)}} σ=${{formatNumber(altruism.std, 3)}} ・ メモリ μ=${{formatNumber(altruismMem.mean, 3)}} σ=${{formatNumber(altruismMem.std, 3)}} ・ スパン μ=${{formatNumber(altruismSpan.mean, 3)}} σ=${{formatNumber(altruismSpan.std, 3)}}</div>`
  ];
  statsBox.innerHTML = parts.join('');
}}
function updateAltruismPanel(g) {{
  if (!altruismSummaryBox) return;
  const nodes = (g.nodes || []).map(n => n.data || n);
  const hidden = nodes.filter(d => d.type === 'hidden');
  if (!hidden.length) {{
    altruismSummaryBox.textContent = '利他性データがありません。';
    if (altruismPlotBox) altruismPlotBox.innerHTML = '<div class="fine">hiddenノードなし</div>';
    return;
  }}
  const idxs = [];
  const altSeries = [];
  const memSeries = [];
  const spanSeries = [];
  hidden.forEach((d, i) => {{
    idxs.push(i + 1);
    const alt = Number(d.altruism);
    const mem = Number(d.altruism_memory);
    const span = Number(d.altruism_span);
    altSeries.push(Number.isFinite(alt) ? alt : 0);
    memSeries.push(Number.isFinite(mem) ? mem : 0);
    spanSeries.push(Number.isFinite(span) ? span : 0);
  }});
  const altStats = calcStats(altSeries);
  const memStats = calcStats(memSeries);
  const spanStats = calcStats(spanSeries);
  altruismSummaryBox.innerHTML = `利他性 μ=${{formatNumber(altStats.mean, 3)}} σ=${{formatNumber(altStats.std, 3)}} / メモリ μ=${{formatNumber(memStats.mean, 3)}} σ=${{formatNumber(memStats.std, 3)}} / スパン μ=${{formatNumber(spanStats.mean, 3)}} σ=${{formatNumber(spanStats.std, 3)}}`;
  if (typeof Plotly !== 'undefined' && altruismPlotBox) {{
    const traces = [
      {{ x: idxs, y: altSeries, name: 'altruism', mode: 'lines+markers', line: {{ color: '#219ebc' }}, marker: {{ size: 5 }} }},
      {{ x: idxs, y: memSeries, name: 'memory', mode: 'lines', line: {{ color: '#8ecae6', dash: 'dot' }} }},
      {{ x: idxs, y: spanSeries, name: 'span', mode: 'lines', line: {{ color: '#ffb703', dash: 'dash' }} }},
    ];
    const layout = {{ margin: {{ l: 36, r: 12, t: 24, b: 28 }}, height: 220, paper_bgcolor: '#fff', plot_bgcolor: '#fff', legend: {{ orientation: 'h', x: 0, y: 1.18 }}, xaxis: {{ title: 'hidden index' }}, yaxis: {{ title: 'value' }} }};
    Plotly.react(altruismPlotBox, traces, layout, {{ displayModeBar: false, responsive: true }});
  }} else if (altruismPlotBox) {{
    altruismPlotBox.textContent = 'Plotly unavailable';
  }}
}}

function genomeToElements(g) {{
      const nodes = (g.nodes||[]).map(n => {{
        const d = Object.assign({{}}, n.data || n);
        if (d.orig_label === undefined) d.orig_label = d.label;
        if (d.vizColor === undefined) d.vizColor = '#999999';
        if (d.vizWidth === undefined) d.vizWidth = 2;
        return {{ data: d }};
      }});
      const edges = (g.edges||[]).map(e => {{
        const d = Object.assign({{}}, e.data || e);
        if (d.vizColor === undefined) {{
          const w = Number(d.weight) || 0;
          d.vizColor = w >= 0 ? '#56B4E9' : '#D55E00';
        }}
        const absW = Math.abs(Number(d.abs_weight !== undefined ? d.abs_weight : d.weight)) || 0;
        if (d.vizWidth === undefined) d.vizWidth = 0.6 + Math.min(3.4, absW);
        return {{ data: d, classes: (d.enabled ? 'enabled' : 'disabled') }};
      }});
      return nodes.concat(edges);
    }}
function populateSelect() {{\n      sel.innerHTML = '';\n      if (!DATA.genomes || DATA.genomes.length===0) {{\n        const opt = document.createElement('option'); opt.text='(no genomes)'; sel.add(opt); sel.disabled=true; return;\n      }}\n      sel.disabled=false;\n      DATA.genomes.forEach((g,i) => {{\n        const meta = g.meta || {{}};\n        const label = (g.id || ('genome_'+i)) + (meta.fitness!==undefined ? (' (fitness='+meta.fitness+')') : '');\n        const opt = document.createElement('option'); opt.value=String(i); opt.text=label; sel.add(opt);\n      }});\n    }}\n    function renderGenome(idx) {{\n      if (!DATA.genomes || !DATA.genomes[idx]) return;\n      const g = DATA.genomes[idx];\n      currentGenome = g;\n      const elements = genomeToElements(g);\n      const styles = [
        {{ selector:'node', style:{{ 'label':'data(label)', 'font-size':11, 'text-valign':'center', 'text-halign':'center',
           'text-wrap':'wrap', 'text-max-width': 90, 'background-color':'data(vizColor)','width':22,'height':22, 'color':'#111','border-color':'#333','border-width':0.5 }} }},
        {{ selector:'edge', style:{{ 'line-color':'data(vizColor)', 'target-arrow-color':'data(vizColor)', 'width':'data(vizWidth)', 'opacity':0.95,
           'curve-style':'bezier','target-arrow-shape':'triangle' }} }},
        {{ selector:'edge.disabled', style:{{ 'line-style':'dotted','opacity':0.35 }} }},
        {{ selector:'edge.hidden', style:{{ 'display':'none' }} }},
        {{ selector:':selected', style:{{ 'border-width':2, 'border-color':'#F0E442' }} }},
      ];\n      if (cy) cy.destroy();
      cy = cytoscape({{ container: cyContainer, elements: elements, style: styles, layout: {{ name:'cose', animate:false }},
        wheelSensitivity:0.2, minZoom:0.2, maxZoom:5 }});
      currentEdgeMax = Math.max(computeEdgeMax(g), globalStats.maxAbsWeight || 0);
      applyEdgeVisuals();
      if (weightSlider) {{
        if (!cy || cy.edges().length === 0) {{
          weightSlider.value = '0';
          weightSlider.disabled = true;
        }} else {{
          weightSlider.disabled = false;
        }}
      }}

      function showDetail(html) {{ detail.innerHTML = html; }}\n      function nodeHtml(d) {{\n        const a=(k)=> (d[k]!==undefined && d[k]!==null ? String(d[k]) : '');\n        return `<div><b>Node</b></div>\n          <div>ID: ${{a('id')}}</div>\n          <div>Label: ${{a('label')}}</div>\n          <div>Type: ${{a('type')}}</div>\n          <div>Bias: ${{a('bias')}}</div>\n          <div>Activation: ${{a('activation')}}</div>\n          <div>Sensitivity: ${{a('sensitivity')}}</div>\n          <div>Jitter: ${{a('jitter')}}</div>\n          <div>Momentum: ${{a('momentum')}}</div>\n          <div>Variance: ${{a('variance')}}</div>\n          <div>Altruism: ${{a('altruism')}}</div>\n          <div>Altruism memory: ${{a('altruism_memory')}}</div>\n          <div>Altruism span: ${{a('altruism_span')}}</div>\n          <div>Note: ${{a('note')}}</div>`;\n      }}\n      function edgeHtml(d) {{
        const a=(k)=> (d[k]!==undefined && d[k]!==null ? String(d[k]) : '');
        return `<div><b>Edge</b></div>
          <div>Source: ${{a('source')}}</div>
          <div>Target: ${{a('target')}}</div>
          <div>Weight: ${{a('weight')}}</div>
          <div>|Weight|: ${{a('abs_weight')}}</div>
          <div>Enabled: ${{a('enabled')}}</div>`;
      }}

      // Click selects → details\n      cy.on('tap','node',(evt)=> showDetail(nodeHtml(evt.target.data())));\n      cy.on('tap','edge',(evt)=> showDetail(edgeHtml(evt.target.data())));\n      cy.on('tap',(evt)=> {{ if (evt.target===cy) showDetail('<div class="fine">ノードやエッジを選択すると詳細が表示されます。</div>'); }});\n\n      // Hover tooltip\n      const moveTip = (e) => {{ tip.style.left=(e.renderedPosition.x + cyContainer.getBoundingClientRect().left)+'px';\n                                tip.style.top=(e.renderedPosition.y + cyContainer.getBoundingClientRect().top)+'px'; }};\n      const metricKeyMap = {{ sensitivity:'sensitivity', momentum:'momentum', variance:'variance' }};\n      const metricLabelMap = {{ sensitivity:'S', momentum:'M', variance:'V' }};\n      cy.on('mouseover','node',(evt)=>{{
        const data = evt.target.data();
        let text = data.label || data.id || '';
        if (activeColorMode !== 'type') {{
          const key = metricKeyMap[activeColorMode];
          const label = metricLabelMap[activeColorMode] || activeColorMode;
          const val = Number(data[key]);
          if (Number.isFinite(val)) text += ' · ' + label + '=' + formatNumber(val, 3);
        }}
        tip.innerHTML = text;
        tip.style.display='block';
        moveTip(evt);
      }});\n      cy.on('mousemove','node',(evt)=> moveTip(evt));\n      cy.on('mouseout','node',()=> {{ tip.style.display='none'; }});\n\n      // Persist position/color/note\n      cy.on('free', 'node', saveState);\n      cy.on('lock unlock', 'node', saveState);\n\n      // Apply saved state for this genome\n      applyState();\n\n      applyNodeColorMode(activeColorMode);\n      applyEdgeFilters();\n      updateGenomeStats(g);
      updateAltruismPanel(g);\n      // Context menu handlers\n      cy.on('cxttapstart', 'node', (evt) => {{\n        ctxNode = evt.target;\n        openCtxAt(evt.renderedPosition);\n      }});\n      cy.on('cxttapstart', (evt) => {{\n        if (evt.target === cy) closeCtx();\n      }});\n      document.addEventListener('click', (e) => {{\n        if (!ctx.contains(e.target)) closeCtx();\n      }});\n      window.addEventListener('resize', closeCtx);\n      document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeCtx(); }});\n    }}\n\n    // UI wiring\n    populateSelect();\n    if (DATA.genomes && DATA.genomes.length>0) renderGenome(0);\n    sel.addEventListener('change', ()=> {{ const idx=parseInt(sel.value,10); if(!Number.isNaN(idx)) renderGenome(idx); }});\n    layoutBtn.addEventListener('click', ()=> {{ if (cy) cy.layout({{name:'cose', animate:true}}).run(); }});\n    if (colorModeSelect) colorModeSelect.addEventListener('change', ()=> {{
      activeColorMode = colorModeSelect.value || 'type';
      applyNodeColorMode(activeColorMode);
    }});\n    if (weightSlider) weightSlider.addEventListener('input', ()=> {{ applyEdgeFilters(); }});\n    if (enabledOnly) enabledOnly.addEventListener('change', ()=> {{ applyEdgeFilters(); }});\n\n    // Context menu building\n    const PALETTE = ['#0072B2','#D55E00','#009E73','#CC79A7','#F0E442','#56B4E9','#E69F00','#000000','#777777','#999999'];\n    const hiddenColorPicker = document.createElement('input'); hiddenColorPicker.type='color'; hiddenColorPicker.style.display='none'; document.body.appendChild(hiddenColorPicker);\n\n    function openCtxAt(renderedPos) {{\n      if (!ctxNode) return;\n      const rect = cyContainer.getBoundingClientRect();\n      const x = rect.left + renderedPos.x;\n      const y = rect.top + renderedPos.y;\n      ctx.innerHTML = '';\n      const menu = document.createElement('div');\n\n      // Title\n      const title = document.createElement('div');\n      title.className='ctx-item';\n      title.style.cursor='default';\n      title.innerHTML = '<b>Node:</b> ' + (ctxNode.data('label') || ctxNode.id());\n      ctx.appendChild(title);\n\n      // Fix/Unfix\n      const fix = document.createElement('div');\n      fix.className='ctx-item';\n      const locked = ctxNode.locked();\n      fix.textContent = locked ? '位置の固定を解除' : '位置を固定';\n      fix.onclick = () => {{\n        if (ctxNode.locked()) ctxNode.unlock(); else ctxNode.lock();\n        saveState(); closeCtx();\n      }};\n      ctx.appendChild(fix);\n\n      // Color row\n      const colorRow = document.createElement('div');\n      colorRow.className='ctx-item';\n      colorRow.innerHTML = '<div class="ctx-row"><span>色を変更</span><span style="font-size:12px;color:var(--muted)">クリックで適用</span></div>';\n      const sw = document.createElement('div'); sw.className='swatches';\n      PALETTE.forEach(c => {{\n        const d = document.createElement('div'); d.className='swatch'; d.style.background=c;\n        d.title = c;\n        d.onclick = () => {{ ctxNode.data('color', c); ctxNode.style('background-color', c); saveState(); closeCtx(); }};\n        sw.appendChild(d);\n      }});\n      // Custom picker\n      const custom = document.createElement('div');\n      custom.className='ctx-item';\n      custom.textContent='カスタムカラー…';\n      custom.onclick = () => {{\n        hiddenColorPicker.value = ctxNode.data('color') || '#999999';\n        hiddenColorPicker.onchange = () => {{\n          const c = hiddenColorPicker.value;\n          ctxNode.data('color', c); ctxNode.style('background-color', c); saveState(); closeCtx();\n        }};\n        hiddenColorPicker.click();\n      }};\n      colorRow.appendChild(sw);\n      ctx.appendChild(colorRow);\n      ctx.appendChild(custom);\n\n      // Note editor\n      const noteBtn = document.createElement('div');\n      noteBtn.className='ctx-item';\n      noteBtn.textContent='注釈を追加/編集…';\n      noteBtn.onclick = () => {{\n        const cur = ctxNode.data('note') || '';\n        const txt = window.prompt('ノードの注釈（空で削除）', cur);\n        if (txt === null) return;\n        const orig = ctxNode.data('orig_label') || ctxNode.data('label') || ctxNode.id();\n        if (txt.trim() === '') {{\n          ctxNode.data('note', null);\n          ctxNode.data('label', orig);\n        }} else {{\n          ctxNode.data('note', txt);\n          ctxNode.data('label', orig + '\\n' + txt);\n        }}\n        saveState(); closeCtx();\n      }};\n      ctx.appendChild(noteBtn);\n\n      // Reset color\n      const resetColor = document.createElement('div');\n      resetColor.className='ctx-item';\n      resetColor.textContent='色をリセット';\n      resetColor.onclick = () => {{\n        ctxNode.data('color', null);\n        // Revert to type-based color by removing inline style\n        ctxNode.removeStyle('background-color');\n        saveState(); closeCtx();\n      }};\n      ctx.appendChild(resetColor);\n\n      // Separator\n      const sep = document.createElement('div'); sep.className='ctx-sep'; ctx.appendChild(sep);\n\n      // Save layout now\n      const saveBtn = document.createElement('div');\n      saveBtn.className='ctx-item';\n      saveBtn.textContent='レイアウトを保存';\n      saveBtn.onclick = () => {{ saveState(); closeCtx(); }};\n      ctx.appendChild(saveBtn);\n\n      // Open\n      ctx.style.left = Math.round(x) + 'px';\n      ctx.style.top  = Math.round(y) + 'px';\n      ctx.style.display = 'block';\n    }}\n\n    function closeCtx() {{ ctx.style.display='none'; ctxNode = null; }}\n\n  </script>\n</body>\n</html>"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    html = html.replace('</style>', ".build-id{margin:0.4rem 0 0;color:#444;font-size:0.9rem;}</style>", 1)
    html = html.replace('<h1>{title}</h1>', f"<h1>{title}</h1><p class='build-id'>Build: {build_id}</p>", 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print('[REPORT]', path)

def _run_default_fractal_demo() -> int:
    """Execute the fractal spinor showcase with polished logging and layout."""
    _ensure_matplotlib_agg(force=True)
    root = os.environ.get('SPINOR_DEFAULT_ROOT', os.path.join('out', 'fractal_default'))
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    out_prefix = os.path.join(root, timestamp, 'spinor_demo')
    out_dir = os.path.dirname(out_prefix) or '.'
    os.makedirs(out_dir, exist_ok=True)
    print('[INFO] No CLI arguments supplied; running default fractal spinor environment demo.')
    print(f'[INFO] Artifacts will be stored under: {out_dir}')
    try:
        artifacts = run_spinor_monolith(out_prefix=out_prefix)
    except Exception as exc:
        print('[ERROR] Fractal spinor environment demo failed.', file=sys.stderr)
        traceback.print_exc()
        return 1
    if not artifacts:
        print('[WARN] Demo completed but did not report any artifacts.')
        return 0
    key_width = max(len(k) for k in artifacts)
    print('[OK] Fractal spinor environment demo completed. Generated artifacts:')
    for key in sorted(artifacts):
        print(f'  - {key.ljust(key_width)} : {artifacts[key]}')
    summary_path = os.path.join(out_dir, 'artifact_manifest.json')
    try:
        with open(summary_path, 'w', encoding='utf-8') as fh:
            json.dump(artifacts, fh, indent=2, ensure_ascii=False)
        print(f'[INFO] Artifact manifest saved to {summary_path}')
    except Exception:
        print('[WARN] Unable to persist artifact manifest to disk.')
    return 0


def main(argv: Optional[Iterable[str]]=None) -> int:
    """Command-line interface entrypoint."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list:
        return _run_default_fractal_demo()
    _ensure_matplotlib_agg(force=True)
    report_override: Optional[bool] = None
    legacy_flags: List[str] = []
    want_help = False
    cleaned_args: List[str] = []
    for arg in argv_list:
        if arg == '--h':
            want_help = True
            continue
        if arg == '--report':
            report_override = True
            legacy_flags.append('--report')
            continue
        if arg == '--no-report':
            report_override = False
            continue
        if arg == '--sync-spinor-seeds':
            legacy_flags.append('--sync-spinor-seeds')
            continue
        cleaned_args.append(arg)
    argv_list = cleaned_args
    ap = argparse.ArgumentParser(
        description='Spiral-NEAT NumPy | built-in CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--task', dest='tasks_single', action='append', nargs='+', choices=['xor', 'circles', 'spiral'], help='one or more supervised tasks to evolve (repeatable)')
    ap.add_argument('--tasks', dest='tasks_single', action='append', nargs='+', choices=['xor', 'circles', 'spiral'], help=argparse.SUPPRESS)
    ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rl-env', type=str)
    ap.add_argument('--rl-gens', type=int, default=20)
    ap.add_argument('--rl-pop', type=int, default=24)
    ap.add_argument('--rl-episodes', type=int, default=1)
    ap.add_argument('--rl-max-steps', type=int, default=500)
    ap.add_argument('--rl-stochastic', action='store_true')
    ap.add_argument('--rl-temp', type=float, default=1.0)
    ap.add_argument('--rl-gameplay-gif', action='store_true')
    ap.add_argument('--out', default='out_monolith_cli')
    ap.add_argument('--version', action='store_true', help='print build information and exit')
    ap.add_argument('--complexity-baseline-quantile', type=float, help='quantile (0-1) for survivor complexity bonus baseline')
    ap.add_argument('--complexity-bonus-span-quantile', type=float, help='upper quantile used to scale survivor complexity distance')
    ap.add_argument('--complexity-survivor-cap', type=float, help='cap applied to normalized survivor complexity bonus ratio')
    ap.add_argument('--complexity-bonus-limit', type=float, help='absolute cap for survivor complexity bonuses')
    ap.add_argument('--no-mandatory', dest='mandatory', action='store_false', help='disable mandatory lazy council steering in the nomology environment')
    ap.add_argument('--mandatory', dest='mandatory', action='store_true', help=argparse.SUPPRESS)
    ap.set_defaults(mandatory=True)
    if want_help:
        ap.print_help()
        return 0
    args = ap.parse_args(argv_list)
    if legacy_flags:
        print(f"[INFO] Ignored legacy flag(s): {', '.join(legacy_flags)} (default behaviour now applies.)")
    requested_tasks: List[str] = []
    if getattr(args, 'tasks_single', None):
        for chunk in args.tasks_single:
            requested_tasks.extend(chunk)
    requested_tasks = list(dict.fromkeys(requested_tasks))
    report_enabled = True if report_override is None else bool(report_override)
    if args.version:
        info = _resolve_build_info()
        print(_build_stamp_text())
        print(f"git hash : {info['hash']}")
        print(f"timestamp: {info['timestamp']}")
        return 0
    print(f"[INFO] {_build_stamp_text()}")
    global _SPINOR_BOUND_SEED
    _SPINOR_BOUND_SEED = args.seed
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'spiral_monolith_neat_numpy.py'
    os.makedirs(args.out, exist_ok=True)
    figs: Dict[str, Optional[str]] = OrderedDict()
    report_meta: Dict[str, Any] = {'supervised': [], 'rl': None}
    supervised_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _add_artifact(label: str, path: Optional[str]):
        if not path:
            return
        try:
            resolved = os.path.abspath(path)
        except Exception:
            resolved = path
        if os.path.exists(resolved):
            figs[label] = resolved

    if requested_tasks:
        np.random.seed(args.seed)
        for task in requested_tasks:
            out_prefix = os.path.join(args.out, task)
            res = run_backprop_neat_experiment(
                task,
                gens=args.gens,
                pop=args.pop,
                steps=args.steps,
                out_prefix=out_prefix,
                make_gifs=True,
                make_lineage=True,
                rng_seed=args.seed,
                complexity_baseline_quantile=args.complexity_baseline_quantile,
                complexity_survivor_cap=args.complexity_survivor_cap,
                complexity_bonus_limit=args.complexity_bonus_limit,
                complexity_bonus_span_quantile=args.complexity_bonus_span_quantile,
            )
            supervised_results[task] = res
            label_base = task.upper()
            _add_artifact(f'{label_base} | Learning Curve + Complexity', res.get('learning_curve'))
            _add_artifact(f'{label_base} | Decision Boundary', res.get('decision_boundary'))
            _add_artifact(f'{label_base} | Best Topology', res.get('topology'))
            for idx, path in enumerate(res.get('top3_topologies') or [], 1):
                _add_artifact(f'{label_base} | Topology Rank {idx}', path)
            regen_gif = res.get('regen_gif')
            _add_artifact(f'{label_base} | Regeneration GIF', regen_gif)
            if regen_gif and os.path.exists(regen_gif) and imageio is not None:
                try:
                    with imageio.get_reader(regen_gif) as reader:
                        idx = max(0, reader.get_length() // 2 - 1)
                        frame = reader.get_data(idx)
                    frame_path = os.path.join(args.out, f'{task}_regen_frame.png')
                    _imwrite_image(frame_path, frame)
                    _add_artifact(f'{label_base} | Regeneration Snapshot', frame_path)
                except Exception:
                    pass
            morph_gif = res.get('morph_gif')
            _add_artifact(f'{label_base} | Morph GIF', morph_gif)
            lineage_path = res.get('lineage')
            _add_artifact(f'{label_base} | Lineage', lineage_path)
            scars_spiral_path = res.get('scars_spiral')
            _add_artifact(f'{label_base} | Spiral Scar Map', scars_spiral_path)
            regen_log = res.get('lcs_log')
            _add_artifact(f'{label_base} | LCS Healing Log', regen_log)
            ribbon = res.get('lcs_ribbon')
            _add_artifact(f'{label_base} | LCS Ribbon', ribbon)
            timeline = res.get('lcs_timeline')
            _add_artifact(f'{label_base} | LCS Timeline', timeline)
            resilience_log = res.get('resilience_log')
            _add_artifact(f'{label_base} | Resilience Tracebacks', resilience_log)
            bp_variant = res.get('backprop_variation')
            if isinstance(bp_variant, dict):
                _add_artifact(f'{label_base} | Backprop Variation', bp_variant.get('figure'))
            _add_artifact(f'{label_base} | Diversity Metrics', res.get('diversity_csv'))
            _add_artifact(f'{label_base} | Diversity Trajectory', res.get('diversity_plot'))
            summary_decisions = res.get('summary_decisions') or {}
            if isinstance(summary_decisions, dict):
                for variant_name, variant_path in summary_decisions.items():
                    _add_artifact(f'{label_base} | Decision {variant_name}', variant_path)
            history = res.get('history') or []
            best_fit = max((b for b, _a in history), default=None)
            final_best = history[-1][0] if history else None
            final_avg = history[-1][1] if history else None
            initial_best = history[0][0] if history else None
            sup_summary = {
                'task': task,
                'gens': args.gens,
                'pop': args.pop,
                'steps': args.steps,
                'best_fit': best_fit,
                'final_best': final_best,
                'final_avg': final_avg,
                'initial_best': initial_best,
                'has_lineage': bool(lineage_path and os.path.exists(lineage_path)),
                'has_regen_log': bool(regen_log and os.path.exists(regen_log)),
                'has_lcs_viz': bool((ribbon and os.path.exists(ribbon)) or (timeline and os.path.exists(timeline))),
                'has_spiral': bool(scars_spiral_path and os.path.exists(scars_spiral_path)),
                'has_resilience': bool(resilience_log and os.path.exists(resilience_log)),
                'has_diversity': bool(res.get('diversity_plot') and os.path.exists(res.get('diversity_plot'))),
            }
            report_meta['supervised'].append(sup_summary)
    rl_history: List[Tuple[float, float]] = []
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
            neat_module = sys.modules[__name__]
            neat = neat_module.ReproPlanaNEATPlus(num_inputs=obs_dim, num_outputs=out_dim, population_size=args.rl_pop, output_activation='identity', rng=np.random.default_rng(args.seed))
            _apply_stable_neat_defaults(neat)
            regen_log_path = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_regen_log.csv")
            if hasattr(neat, 'lcs_monitor') and neat.lcs_monitor is not None:
                neat.lcs_monitor.csv_path = regen_log_path
                if os.path.exists(regen_log_path):
                    os.remove(regen_log_path)
            fit = gym_fitness_factory(args.rl_env, stochastic=args.rl_stochastic, temp=args.rl_temp, max_steps=args.rl_max_steps, episodes=args.rl_episodes)
            best, hist = neat.evolve(fit, n_generations=args.rl_gens, verbose=True, env_schedule=_default_difficulty_schedule)
            rl_history = list(hist)
            rl_resilience_log = None
            rl_failures = list(getattr(neat, '_resilience_failures', [])) if hasattr(neat, '_resilience_failures') else []
            if rl_failures:
                rl_resilience_log = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_resilience_log.json")
                try:
                    with open(rl_resilience_log, 'w', encoding='utf-8') as fh:
                        json.dump({'failures': rl_failures, 'eval_guard': getattr(neat, '_resilience_eval_guard', 0), 'history': getattr(neat, '_resilience_history', [])}, fh, indent=2, ensure_ascii=False)
                except Exception as log_err:
                    print('[WARN] RL resilience log write failed:', log_err)
                    rl_resilience_log = None
            close_env = getattr(fit, 'close_env', None)
            if callable(close_env):
                close_env()
            rc_png = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_reward_curve.png")
            xs = list(range(len(hist)))
            ys_b = [b for b, _a in hist]
            ys_a = [a for _b, a in hist]
            plt.figure()
            plt.plot(xs, ys_b, label='best')
            plt.plot(xs, ys_a, label='avg')
            plt.xlabel('generation')
            plt.ylabel('episode reward')
            plt.title(f'{args.rl_env} | Average Episode Reward')
            plt.legend()
            plt.tight_layout()
            fig = plt.gcf()
            _savefig(fig, rc_png, dpi=150)
            plt.close()
            figs['RL 平均エピソード報酬'] = rc_png
            if rl_resilience_log and os.path.exists(rl_resilience_log):
                figs['RL Resilience Tracebacks'] = rl_resilience_log
            lcs_rows = load_lcs_log(regen_log_path) if os.path.exists(regen_log_path) else []
            lcs_series = _prepare_lcs_series(lcs_rows) if lcs_rows else None
            if os.path.exists(regen_log_path):
                figs[f'LCS Healing Log ({args.rl_env})'] = regen_log_path
            if lcs_rows:
                ribbon_path = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_lcs_ribbon.png")
                try:
                    export_lcs_ribbon_png(lcs_rows, ribbon_path, series=lcs_series)
                    figs[f'LCS Ribbon ({args.rl_env})'] = ribbon_path
                except Exception as ribbon_err:
                    print('[WARN] LCS ribbon export failed:', ribbon_err)
                timeline_path = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_lcs_timeline.gif")
                try:
                    export_lcs_timeline_gif(lcs_rows, timeline_path, series=lcs_series, fps=6)
                    figs[f'LCS Timeline ({args.rl_env})'] = timeline_path
                except Exception as timeline_err:
                    print('[WARN] LCS timeline export failed:', timeline_err)
            gif = None
            if args.rl_gameplay_gif:
                gif = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_gameplay.gif")
                try:
                    out_path = run_policy_in_env(best, args.rl_env, episodes=max(1, args.rl_episodes), max_steps=args.rl_max_steps, stochastic=args.rl_stochastic, temp=args.rl_temp, out_gif=gif)
                    if out_path and os.path.exists(out_path):
                        figs['RL ゲームプレイ'] = out_path
                except Exception as gif_err:
                    print('[WARN] gameplay gif failed:', gif_err)
            rl_best = max((b for b, _a in hist), default=None)
            rl_final_best = hist[-1][0] if hist else None
            rl_final_avg = hist[-1][1] if hist else None
            report_meta['rl'] = {'env': args.rl_env, 'gens': args.rl_gens, 'pop': args.rl_pop, 'episodes': args.rl_episodes, 'best_reward': rl_best, 'final_best': rl_final_best, 'final_avg': rl_final_avg, 'has_lcs_log': bool(os.path.exists(regen_log_path)), 'has_lcs_viz': bool(lcs_rows), 'has_gameplay': bool(gif and os.path.exists(gif)), 'has_resilience': bool(rl_resilience_log and os.path.exists(rl_resilience_log))}
        except Exception as e:
            print('[WARN] RL branch skipped:', e)
    if report_enabled:
        for task, res in supervised_results.items():
            title = f'{task.upper()} | Interactive NEAT Report'
            html_path = os.path.join(args.out, f'{task}_interactive.html')
            try:
                export_interactive_html_report(html_path, title=title, history=res.get('history', []), genomes=res.get('genomes_cyto', []), max_genomes=60)
            except Exception as report_err:
                print(f'[WARN] Interactive report failed for {task}:', report_err)
        if args.rl_env and rl_history:
            title = f'{args.rl_env} | Interactive NEAT Report'
            html_path = os.path.join(args.out, f"{args.rl_env.replace(':', '_')}_interactive.html")
            try:
                export_interactive_html_report(html_path, title=title, history=rl_history, genomes=[], max_genomes=1)
            except Exception as rl_report_err:
                print('[WARN] RL interactive report failed:', rl_report_err)
    else:
        print('[INFO] Interactive report generation disabled (--no-report).')
        if figs:

            def _data_uri(p: str) -> Tuple[str, str]:
                with open(p, 'rb') as f:
                    import base64
                    raw = base64.b64encode(f.read()).decode('ascii')
                mime, _ = mimetypes.guess_type(p)
                if mime is None:
                    if p.lower().endswith('.gif'):
                        mime = 'image/gif'
                    elif p.lower().endswith(('.mp4', '.webm')):
                        mime = 'video/mp4'
                    else:
                        mime = 'application/octet-stream'
                return (f'data:{mime};base64,{raw}', mime)
            html = os.path.join(args.out, 'Sakana_NEAT_Report.html')
            entries = [(k, p) for k, p in figs.items() if p and os.path.exists(p)]
            with open(html, 'w', encoding='utf-8') as f:
                from datetime import datetime
                import html as htmllib
                f.write("<!DOCTYPE html><html lang='ja'><head><meta charset='utf-8'><title>Spiral Monolith NEAT Report | スパイラル・モノリスNEATレポート</title>")
                f.write("<style>body{font-family:'Hiragino Sans','Noto Sans JP',sans-serif;background:#fafafa;color:#222;line-height:1.6;padding:2rem;}header.cover{background:#fff;border:1px solid #ddd;border-radius:12px;padding:1.5rem;margin-bottom:2rem;box-shadow:0 8px 20px rgba(0,0,0,0.05);}header.cover h1{margin-top:0;font-size:1.9rem;}header.cover p.meta{margin:0.35rem 0 0.6rem 0;}header.cover ul{margin:0;padding-left:1.2rem;}section.summary,section.legend,section.narrative,section.examples{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:1.25rem;margin-bottom:2rem;box-shadow:0 10px 24px rgba(0,0,0,0.035);}section.summary h2,section.legend h2,section.narrative h2,section.examples h2{margin-top:0;font-size:1.35rem;}section.summary ul{margin:0;padding-left:1.4rem;}section.legend ol{margin:0;padding-left:1.4rem;}section.narrative p{margin:0 0 0.8rem 0;}section.examples ul{margin:0;padding-left:1.4rem;}section.examples li{margin:0 0 0.65rem 0;}section.examples code{background:#f4f4f4;border-radius:6px;display:block;padding:0.35rem 0.55rem;font-size:0.92rem;}figure{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:1rem;margin:0 0 2rem 0;box-shadow:0 12px 24px rgba(0,0,0,0.04);}figure img,figure video{width:100%;height:auto;border-radius:8px;}figcaption{margin-top:0.75rem;font-weight:600;}</style></head><body>")
                f.write("<header class='cover'>")
                f.write('<h1>Spiral Monolith NEAT Report / スパイラル・モノリスNEATレポート</h1>')
                primary_label = ', '.join(t.upper() for t in requested_tasks) if requested_tasks else 'N/A'
                f.write(f"<p class='meta'>Generated at / 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Supervised Tasks / 教師ありタスク: {primary_label}</p>")
                f.write(f"<p class='meta'>Build / ビルド: {_build_stamp_short()}</p>")
                f.write('<ul>')
                f.write(f'<li>Generations / 世代数: {args.gens}</li>')
                f.write(f'<li>Population / 個体数: {args.pop}</li>')
                f.write(f'<li>Backprop Steps / 学習反復: {args.steps}</li>')
                if args.rl_env:
                    f.write(f'<li>RL Environment / 強化学習環境: {htmllib.escape(args.rl_env)}</li>')
                f.write('</ul>')
                f.write('</header>')
                summary_items = []

                def _fmt_float(val: Optional[float]) -> str:
                    return '–' if val is None else f'{val:.4f}'
                for sup_meta in report_meta.get('supervised') or []:
                    extras = []
                    if sup_meta.get('has_regen_log'):
                        extras.append('LCS log')
                    if sup_meta.get('has_lcs_viz'):
                        extras.append('LCS visuals')
                    if sup_meta.get('has_lineage'):
                        extras.append('lineage')
                    if sup_meta.get('has_spiral'):
                        extras.append('scar map')
                    if sup_meta.get('has_resilience'):
                        extras.append('resilience log')
                    extra_txt = f" [{', '.join(extras)}]" if extras else ''
                    summary_items.append(f"<li><strong>Supervised ({htmllib.escape(sup_meta['task'].upper())})</strong> — best {_fmt_float(sup_meta.get('best_fit'))} | final {_fmt_float(sup_meta.get('final_best'))} / avg {_fmt_float(sup_meta.get('final_avg'))}{extra_txt}</li>")
                rl_meta = report_meta.get('rl')
                if rl_meta and rl_meta.get('env'):
                    extras = []
                    if rl_meta.get('has_lcs_log'):
                        extras.append('LCS log')
                    if rl_meta.get('has_lcs_viz'):
                        extras.append('LCS visuals')
                    if rl_meta.get('has_gameplay'):
                        extras.append('gameplay gif')
                    if rl_meta.get('has_resilience'):
                        extras.append('resilience log')
                    extra_txt = f" [{', '.join(extras)}]" if extras else ''
                    summary_items.append(f"<li><strong>RL ({htmllib.escape(rl_meta['env'])})</strong> — best {_fmt_float(rl_meta.get('best_reward'))} | final {_fmt_float(rl_meta.get('final_best'))} / avg {_fmt_float(rl_meta.get('final_avg'))}{extra_txt}</li>")
                f.write("<section class='summary'><h2>Overview / 概要</h2>")
                f.write(f'<p>Figures included / 図版数: {len(entries)}</p>')
                if summary_items:
                    f.write('<ul>')
                    for item in summary_items:
                        f.write(item)
                    f.write('</ul>')
                else:
                    f.write('<p>No supervised or RL runs were summarised for this report.</p>')
                f.write('</section>')
                f.write("<section class='narrative'><h2>Evolution Digest / 進化ダイジェスト</h2>")
                f.write('<p>Early generations showed smooth structural adaptation and convergence under low-difficulty conditions.</p>')
                f.write('<p>However, as environmental difficulty and noise increased, regeneration-driven mutations began to trigger bursts of morphological diversification, resembling biological punctuated equilibria.</p>')
                any_sup = next((m for m in report_meta.get('supervised') or [] if m.get('has_regen_log')), None)
                if any_sup:
                    f.write('<p>LCS metrics highlighted how severed pathways recovered within the allowed healing window, aligning regenerative bursts with topology repairs.</p>')
                f.write('</section>')
                if entries:
                    f.write("<section class='legend'><h2>Figure Index / 図版リスト</h2><ol>")
                    for label, path in entries:
                        f.write(f'<li><strong>{htmllib.escape(label)}</strong><br><small>{htmllib.escape(os.path.basename(path))}</small></li>')
                    f.write('</ol></section>')
                cli_examples = [
                    f"python {script_name} --task spiral xor --gens {max(args.gens, 60)} --pop {max(args.pop, 64)} --steps {max(args.steps, 80)} --report --out demo_multitask",
                    f"python {script_name} --task circles --gens {max(40, args.gens)} --pop {max(48, args.pop)} --steps {max(60, args.steps)} --report --out circles_run",
                ]
                rl_example_env = args.rl_env or 'CartPole-v1'
                cli_examples.append(f"python {script_name} --rl-env {htmllib.escape(rl_example_env)} --rl-gens {max(args.rl_gens, 30)} --rl-pop {max(args.rl_pop, 32)} --rl-episodes {max(args.rl_episodes, 2)} --rl-max-steps {args.rl_max_steps} --report --out rl_{htmllib.escape(rl_example_env.replace(':', '_'))}")
                f.write("<section class='examples'><h2>CLI Quickstart / CLIクイックスタート</h2>")
                f.write('<p>Use the following commands as templates for supervised batches and Gym integrations.</p>')
                f.write('<ul>')
                for cmd in cli_examples:
                    f.write(f'<li><code>{cmd}</code></li>')
                f.write('</ul></section>')
                for k, p in entries:
                    uri, mime = _data_uri(p)
                    escaped_label = htmllib.escape(k)
                    if mime == 'image/gif':
                        f.write(f"<figure><img src='{uri}' style='max-width:100%'><figcaption><strong>{escaped_label}</strong></figcaption></figure>")
                    elif mime.startswith('image/'):
                        f.write(f"<figure><img src='{uri}' style='max-width:100%'><figcaption><strong>{escaped_label}</strong></figcaption></figure>")
                    elif mime.startswith('video/'):
                        f.write(f"<figure><video autoplay loop muted playsinline style='max-width:100%'><source src='{uri}' type='{mime}'></video><figcaption><strong>{escaped_label}</strong></figcaption></figure>")
                    else:
                        f.write(f"<figure><a href='{uri}'>download {htmllib.escape(os.path.basename(p))}</a><figcaption><strong>{escaped_label}</strong></figcaption></figure>")
                f.write('</body></html>')
            print('[REPORT]', html)
        manifest_path = os.path.join(args.out, 'artifact_manifest.json')
        manifest_payload = {
            'build_id': _build_stamp_short(),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'figures': {k: v for k, v in figs.items()},
            'report_meta': report_meta,
        }
        try:
            with open(manifest_path, 'w', encoding='utf-8') as mf:
                json.dump(manifest_payload, mf, indent=2, ensure_ascii=False)
            print(f'[REPORT] manifest → {manifest_path}')
        except Exception as manifest_err:
            print('[WARN] Failed to write artifact manifest:', manifest_err)
    print('[OK] outputs in:', args.out)
    return 0
try:
    from multiprocessing import shared_memory as _shm
except Exception:
    _shm = None
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
_ensure_matplotlib_agg()
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None
    try:
        from PIL import Image
    except Exception:
        Image = None
    else:
        Image = Image
else:
    Image = None
_SHM_LOCAL = {}
_SHM_META = {}
_SHM_CACHE = {}
_SHM_HANDLES = {}
STRUCTURAL_EPS = 1e-09
__all__ = ['NodeGene', 'ConnectionGene', 'InnovationTracker', 'Genome', 'compatibility_distance', 'HouseholdManager', 'EvalMode', 'ReproPlanaNEATPlus', 'SpinorGroupInteraction', 'SpinorScheduler', 'NomologyEnv', 'SelfReproducingEvaluator', 'SpinorNomologyDatasetController', 'SpinorNomologyFitness', 'compile_genome', 'forward_batch', 'train_with_backprop_numpy', 'predict', 'predict_proba', 'fitness_backprop_classifier', 'make_circles', 'make_xor', 'make_spirals', 'draw_genome_png', 'export_regen_gif', 'export_morph_gif', 'export_double_exposure', 'plot_learning_and_complexity', 'plot_decision_boundary', 'export_backprop_variation', 'export_decision_boundaries_all', 'render_lineage', 'export_scars_spiral_map', 'output_dim_from_space', 'build_action_mapper', 'eval_with_node_activations', 'run_policy_in_env', 'run_gym_neat_experiment', 'LCSMonitor', 'summarize_graph_changes', 'load_lcs_log', 'export_lcs_ribbon_png', 'export_lcs_timeline_gif', 'PerSampleSequenceStopperPro']
INF = 10 ** 12
PATCHED_PATH = __file__
spec = None
neat = sys.modules[__name__]

def rotate_2d(x: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (R @ x.T).T


def _hex_to_rgba(hex_color: Optional[str], alpha: float=0.15) -> Tuple[float, float, float, float]:
    color = str(hex_color or '').strip()
    if color.startswith('#'):
        color = color[1:]
    if len(color) == 3:
        color = ''.join((ch * 2 for ch in color))
    try:
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
    except Exception:
        r = g = b = 0.62
    return (
        float(np.clip(r, 0.0, 1.0)),
        float(np.clip(g, 0.0, 1.0)),
        float(np.clip(b, 0.0, 1.0)),
        float(np.clip(alpha, 0.0, 1.0)),
    )


def augment_with_spinor(
    X: np.ndarray,
    theta: float,
    parity: Optional[int]=None,
    group_embed: Optional[np.ndarray]=None,
) -> Tuple[np.ndarray, int]:
    s = SpinorScheduler.spinor_vec(theta)
    p = SpinorScheduler.parity(theta) if parity is None else int(parity)
    s_tiled = np.tile(s.reshape(1, 2), (X.shape[0], 1))
    p_col = np.full((X.shape[0], 1), float(p), dtype=np.float32)
    parts = [X, s_tiled, p_col]
    if group_embed is not None:
        ge = np.asarray(group_embed, dtype=np.float32).reshape(1, -1)
        ge_tiled = np.tile(ge, (X.shape[0], 1))
        parts.append(ge_tiled)
    return (np.concatenate(parts, axis=1).astype(np.float32), p)

def run_spinor_monolith(
    gens: int=40,
    pop: int=80,
    period_gens: int=36,
    jitter_std: float=0.07,
    nomology_intensity: float=0.25,
    drift_scale: float=0.006,
    seed: int=0,
    out_prefix: str='/mnt/data/spinor_neat',
    spinor_bind_seed: Optional[int]=None,
) -> Dict[str, str]:
    os.environ['NEAT_EVAL_BACKEND'] = 'thread'
    global _SPINOR_BOUND_SEED
    if spinor_bind_seed is None:
        spinor_bind_seed = _SPINOR_BOUND_SEED
    rng = np.random.default_rng(seed)
    group = SpinorGroupInteraction.dihedral(order=6, seed=seed)
    spin = SpinorScheduler(period_gens=period_gens, jitter_std=jitter_std, seed=seed, group=group)
    env = NomologyEnv(
        intensity=nomology_intensity,
        drift_scale=drift_scale,
        seed=seed,
        noise_weaver_seed=spinor_bind_seed if spinor_bind_seed is not None else None,
    )
    tel = Telemetry(f'{out_prefix}_telemetry.csv', f'{out_prefix}_regimes.csv')
    controller = SpinorNomologyDatasetController(
        spin,
        env,
        rng,
        n_tr=512,
        n_va=256,
        evaluator_seed=spinor_bind_seed,
        mandatory_mode=getattr(args, 'mandatory', True),
    )
    controller.telemetry = tel
    controller.update_for_generation(0, shmem=False)
    out_dim = 2
    neat_inst = neat.ReproPlanaNEATPlus(num_inputs=controller.feature_dim, num_outputs=out_dim, population_size=pop, output_activation='identity', rng=rng)
    neat._apply_stable_neat_defaults(neat_inst)
    neat_inst.spinor_controller = controller
    neat_inst.max_hidden_nodes = max(getattr(neat_inst, 'max_hidden_nodes', 128), 192)
    neat_inst.max_edges = max(getattr(neat_inst, 'max_edges', 1024), 2048)
    fit = SpinorNomologyFitness(controller=controller, get_generation=lambda: neat_inst.generation, steps=40, lr=0.005, l2=0.0001, alpha_nodes=0.001, alpha_edges=0.0005)
    hist = neat_inst.evolve(fit, n_generations=gens, target_fitness=None, verbose=True, env_schedule=None)
    resilience_log = None
    failures = list(getattr(neat_inst, '_resilience_failures', [])) if hasattr(neat_inst, '_resilience_failures') else []
    if failures:
        resilience_log = f'{out_prefix}_resilience_log.json'
        try:
            with open(resilience_log, 'w', encoding='utf-8') as fh:
                json.dump({'failures': failures, 'eval_guard': getattr(neat_inst, '_resilience_eval_guard', 0), 'history': getattr(neat_inst, '_resilience_history', [])}, fh, indent=2, ensure_ascii=False)
        except Exception as log_err:
            print('[WARN] Resilience log write failed:', log_err)
            resilience_log = None

    def _load_csv_rows(path: str) -> List[Dict[str, str]]:
        if not os.path.exists(path):
            return []
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader]

    tele_rows = _load_csv_rows(tel.tel_csv)
    reg_rows = _load_csv_rows(tel.reg_csv)

    def _float_col(rows: List[Dict[str, str]], key: str) -> np.ndarray:
        return np.array([float(row.get(key, 'nan')) for row in rows], dtype=np.float64)

    def _int_col(rows: List[Dict[str, str]], key: str) -> np.ndarray:
        out = []
        for row in rows:
            try:
                out.append(int(float(row.get(key, '0'))))
            except ValueError:
                out.append(0)
        return np.array(out, dtype=np.int32)

    def _str_col(rows: List[Dict[str, str]], key: str) -> List[str]:
        return [str(row.get(key, '')) for row in rows]

    g = _int_col(tele_rows, 'gen') if tele_rows else np.array([], dtype=np.int32)
    theta4 = _float_col(tele_rows, 'theta_mod_4pi') if tele_rows else np.array([], dtype=np.float64)
    parity = _int_col(tele_rows, 'parity') if tele_rows else np.array([], dtype=np.int32)
    theta_raw = _float_col(tele_rows, 'theta') if tele_rows else np.array([], dtype=np.float64)
    noise_seq = _float_col(tele_rows, 'noise') if tele_rows else np.array([], dtype=np.float64)
    noise_kind_seq = np.array(_str_col(tele_rows, 'noise_kind')) if tele_rows else np.array([], dtype=object)
    turns_seq = _float_col(tele_rows, 'turns') if tele_rows else np.array([], dtype=np.float64)
    rot_bias_seq = _float_col(tele_rows, 'rot_bias') if tele_rows else np.array([], dtype=np.float64)
    regime_ids = _int_col(tele_rows, 'regime_id') if tele_rows else np.array([], dtype=np.int32)
    group_idx_seq = _int_col(tele_rows, 'group_idx') if tele_rows else np.array([], dtype=np.int32)
    group_energy_seq = _float_col(tele_rows, 'group_energy') if tele_rows else np.array([], dtype=np.float64)
    group_label_seq = np.array(_str_col(tele_rows, 'group_label')) if tele_rows else np.array([], dtype=object)
    evaluator_notes = _str_col(tele_rows, 'evaluator_note') if tele_rows else []

    plt.figure()
    if g.size:
        plt.plot(g, theta4)
    plt.xlabel('generation')
    plt.ylabel('theta mod 4π')
    fig1 = f'{out_prefix}_phase.png'
    plt.tight_layout()
    fig = plt.gcf()
    _savefig(fig, fig1, dpi=160)
    plt.close()
    plt.figure()
    if g.size:
        plt.step(g, parity, where='post')
    plt.xlabel('generation')
    plt.ylabel('parity (2π branch)')
    fig2 = f'{out_prefix}_parity.png'
    plt.tight_layout()
    fig = plt.gcf()
    _savefig(fig, fig2, dpi=160)
    plt.close()
    plt.figure()
    if g.size:
        plt.plot(g, theta4)
    if reg_rows and g.size:
        for row in reg_rows:
            try:
                x = int(float(row.get('gen', 'nan')))
            except (TypeError, ValueError):
                continue
            plt.axvline(x=x, linestyle='--')
    plt.xlabel('generation')
    plt.ylabel('theta mod 4π (regimes)')
    fig3 = f'{out_prefix}_regimes.png'
    plt.tight_layout()
    fig = plt.gcf()
    _savefig(fig, fig3, dpi=160)
    plt.close()

    fig_noise: Optional[str] = None
    if g.size:
        fig_noise = f'{out_prefix}_noise_timeline.png'
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        colors = []
        for idx in range(len(g)):
            kind = noise_kind_seq[idx] if noise_kind_seq.size else ''
            style = _resolve_noise_style(kind, None) if kind else None
            col = style.get('color', '#6c757d') if style else '#6c757d'
            colors.append(col)
        ax.plot(g, noise_seq, color='#495057', lw=1.0, alpha=0.4)
        if colors:
            ax.scatter(g, noise_seq, c=colors, s=28, alpha=0.9, edgecolors='none')
        if noise_kind_seq.size:
            span_start = 0
            current_kind = noise_kind_seq[0]
            for idx in range(1, len(g) + 1):
                if idx == len(g) or noise_kind_seq[idx] != current_kind:
                    start_gen = float(g[span_start])
                    end_gen = float(g[idx - 1]) + 1.0
                    style = _resolve_noise_style(current_kind, None)
                    ax.axvspan(start_gen, end_gen, color=_hex_to_rgba(style.get('color'), 0.12), lw=0)
                    span_start = idx
                    if idx < len(g):
                        current_kind = noise_kind_seq[idx]
        ax.set_xlabel('generation')
        ax.set_ylabel('noise σ')
        ax.set_title('Noise schedule & spectral mode')
        legend_handles: List[Line2D] = []
        seen_kinds: Set[str] = set()
        for kind in noise_kind_seq:
            key = str(kind)
            if not key or key in seen_kinds:
                continue
            style = _resolve_noise_style(key, None)
            handle = Line2D(
                [0],
                [0],
                marker='o',
                linestyle='None',
                markersize=8,
                markerfacecolor=style.get('color', '#6c757d'),
                markeredgecolor='none',
                label=f"{style.get('symbol', '?')} {style.get('label', key)}",
            )
            legend_handles.append(handle)
            seen_kinds.add(key)
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9, title='noise kind')
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.6)
        fig.tight_layout()
        _savefig(fig, fig_noise, dpi=170)
        plt.close(fig)

    spinor_grid_png: Optional[str] = None
    spinor_transition_gif: Optional[str] = None

    if tele_rows and g.size:
        gen_to_idx: Dict[int, int] = {}
        for idx, gen_val in enumerate(g):
            gen_to_idx.setdefault(int(gen_val), idx)

        picks: Set[int] = {0, max(0, len(g) - 1)}
        for row in reg_rows:
            try:
                event_gen = int(float(row.get('gen', 'nan')))
            except (TypeError, ValueError):
                continue
            idx = gen_to_idx.get(event_gen)
            if idx is not None:
                picks.add(idx)
        target = min(6, len(g))
        if target:
            for frac in np.linspace(0.0, 1.0, num=target):
                if len(g) == 1:
                    idx = 0
                else:
                    idx = int(round(frac * (len(g) - 1)))
                picks.add(int(np.clip(idx, 0, len(g) - 1)))
        snapshots = sorted({idx for idx in picks if 0 <= idx < len(g)})

        def _synth_snapshot(idx: int) -> Tuple[np.ndarray, np.ndarray]:
            theta = float(theta_raw[idx])
            noise = float(noise_seq[idx])
            turns = float(turns_seq[idx])
            rot_bias = float(rot_bias_seq[idx])
            gen_val = int(g[idx])
            local_rng = np.random.default_rng((seed + gen_val * 7919) % (1 << 32))
            spiral_seed = int(local_rng.integers(1 << 31))
            X, y = make_spirals(n=240, noise=noise, turns=turns, seed=spiral_seed)
            X_rot = rotate_2d(X, theta + rot_bias)
            return (X_rot, y)

        if snapshots:
            ncols = min(3, len(snapshots))
            nrows = int(math.ceil(len(snapshots) / float(ncols)))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))
            axes_arr = np.atleast_1d(axes).ravel()
            for ax, idx in zip(axes_arr, snapshots):
                X_vis, y_vis = _synth_snapshot(idx)
                ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='coolwarm', s=12, alpha=0.7, edgecolors='none')
                grp_label = group_label_seq[idx] if group_label_seq.size else ''
                note = evaluator_notes[idx] if idx < len(evaluator_notes) else ''
                title = f'gen {int(g[idx])} | regime {int(regime_ids[idx])} | parity {int(parity[idx])}'
                kind = noise_kind_seq[idx] if noise_kind_seq.size else ''
                if kind:
                    style = _resolve_noise_style(kind, None)
                    title += f" | {style.get('symbol', '?')} {style.get('label', kind)}"
                    ax.set_facecolor(_hex_to_rgba(style.get('color'), 0.1))
                else:
                    ax.set_facecolor('#f8f9fa')
                if grp_label:
                    title += f' | {grp_label}'
                if note:
                    title += f' | {note}'
                ax.set_title(title)
                ax.add_patch(FancyArrowPatch((0.0, 0.0), (math.cos(theta_raw[idx] + rot_bias_seq[idx]), math.sin(theta_raw[idx] + rot_bias_seq[idx])), arrowstyle='->', mutation_scale=12, lw=1.5, color='#444444'))
                ax.set_xlim(-1.6, 1.6)
                ax.set_ylim(-1.6, 1.6)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal', 'box')
                ax.grid(False)
            for ax in axes_arr[len(snapshots):]:
                ax.axis('off')
            fig.tight_layout()
            spinor_grid_png = f'{out_prefix}_spinor_transition_grid.png'
            _savefig(fig, spinor_grid_png, dpi=170)
            plt.close(fig)

        step = max(1, len(g) // 60)
        frames: List[np.ndarray] = []
        for idx in range(0, len(g), step):
            X_vis, y_vis = _synth_snapshot(idx)
            fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6))
            ax_data, ax_spin = axs
            ax_data.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='coolwarm', s=14, alpha=0.75, edgecolors='none')
            kind = noise_kind_seq[idx] if noise_kind_seq.size else ''
            style = _resolve_noise_style(kind, None) if kind else None
            if style:
                ax_data.set_facecolor(_hex_to_rgba(style.get('color'), 0.1))
                ax_spin.set_facecolor(_hex_to_rgba(style.get('color'), 0.12))
            else:
                ax_data.set_facecolor('#f8f9fa')
                ax_spin.set_facecolor('#f8f9fa')
            ax_data.set_xlim(-1.6, 1.6)
            ax_data.set_ylim(-1.6, 1.6)
            ax_data.set_aspect('equal', 'box')
            ax_data.set_xticks([])
            ax_data.set_yticks([])
            title_data = f'gen {int(g[idx])} | regime {int(regime_ids[idx])}'
            if style:
                title_data += f" | {style.get('symbol', '?')} {style.get('label', kind)}"
            ax_data.set_title(title_data)
            arrow_dataset = FancyArrowPatch((0.0, 0.0), (math.cos(theta_raw[idx] + rot_bias_seq[idx]), math.sin(theta_raw[idx] + rot_bias_seq[idx])), arrowstyle='->', mutation_scale=12, lw=1.8, color='#2ca02c')
            ax_data.add_patch(arrow_dataset)

            circle = Circle((0.0, 0.0), radius=1.0, fill=False, lw=2.0, alpha=0.6)
            ax_spin.add_patch(circle)
            spin_vec = SpinorScheduler.spinor_vec(theta_raw[idx])
            arrow_spin = FancyArrowPatch((0.0, 0.0), (float(spin_vec[0]), float(spin_vec[1])), arrowstyle='->', mutation_scale=14, lw=2.2, color='#d62728')
            ax_spin.add_patch(arrow_spin)
            ax_spin.set_xlim(-1.3, 1.3)
            ax_spin.set_ylim(-1.3, 1.3)
            ax_spin.set_aspect('equal', 'box')
            ax_spin.axis('off')
            theta_deg = (float(theta_raw[idx]) % (2.0 * math.pi)) * 180.0 / math.pi
            ax_spin.set_title('spinor orientation')
            grp_label = group_label_seq[idx] if group_label_seq.size else ''
            energy = group_energy_seq[idx] if group_energy_seq.size else 0.0
            summary_line = f'θ={theta_deg:5.1f}° | parity {int(parity[idx])}'
            if grp_label:
                summary_line += f' | {grp_label}'
            summary_line += f' | energy {energy:0.2f}'
            if style:
                summary_line += f" | {style.get('symbol', '?')} {style.get('label', kind)}"
            ax_spin.text(0.0, -1.15, summary_line, ha='center', va='top', fontsize=10)
            note = evaluator_notes[idx] if idx < len(evaluator_notes) else ''
            title = 'Spinor-fractal transition overview'
            if note:
                title += f' ← {note}'
            fig.suptitle(title, fontsize=12)
            fig.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frames.append(frame.reshape(h, w, 3))
            plt.close(fig)

        if frames:
            spinor_transition_gif = f'{out_prefix}_spinor_transition.gif'
            try:
                _mimsave(spinor_transition_gif, frames, fps=6)
            except Exception as gif_err:
                print('[WARN] Unable to write spinor transition GIF:', gif_err)
                spinor_transition_gif = None

    artifacts: Dict[str, Optional[str]] = {'telemetry_csv': tel.tel_csv, 'regimes_csv': tel.reg_csv, 'phase_png': fig1, 'parity_png': fig2, 'regimes_png': fig3}
    if fig_noise:
        artifacts['noise_timeline_png'] = fig_noise
    if resilience_log:
        artifacts['resilience_log'] = resilience_log
    if spinor_grid_png:
        artifacts['spinor_transition_grid'] = spinor_grid_png
    if spinor_transition_gif:
        artifacts['spinor_transition_gif'] = spinor_transition_gif
    return artifacts

@dataclass
class SpinorGroupInteraction:
    name: str
    elements: np.ndarray
    cayley: np.ndarray
    generator_weights: Optional[np.ndarray] = None
    jitter: float = 0.0
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _transposes: np.ndarray = field(init=False, repr=False)
    _identity: np.ndarray = field(init=False, repr=False)
    _element_norms: np.ndarray = field(init=False, repr=False)
    _embeddings: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        elems = np.asarray(self.elements, dtype=np.float32)
        if elems.ndim != 3 or elems.shape[1:] != (2, 2):
            raise ValueError('SpinorGroupInteraction.elements must be (k, 2, 2).')
        cayley = np.asarray(self.cayley, dtype=np.int64)
        if cayley.shape != (elems.shape[0], elems.shape[0]):
            raise ValueError('SpinorGroupInteraction.cayley must be square with size equal to number of elements.')
        self.elements = np.ascontiguousarray(elems)
        self._transposes = np.ascontiguousarray(np.transpose(self.elements, (0, 2, 1)))
        self._identity = np.eye(2, dtype=np.float32)
        self.cayley = cayley
        n = elems.shape[0]
        if self.generator_weights is not None:
            weights = np.asarray(self.generator_weights, dtype=np.float64)
            if weights.ndim != 1 or weights.size != n:
                raise ValueError('generator_weights must be 1D with same length as elements.')
            weights = np.maximum(1e-09, weights)
            self.generator_weights = weights / weights.sum()
        else:
            self.generator_weights = np.full(n, 1.0 / n, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)
        flat = self.elements.reshape(n, -1)
        self._element_norms = np.linalg.norm(flat, axis=1).astype(np.float32)
        self._embeddings = flat.astype(np.float32, copy=False)

    @property
    def size(self) -> int:
        return int(self.elements.shape[0])

    @property
    def embed_dim(self) -> int:
        return 4

    def matrix(self, idx: int) -> np.ndarray:
        return self.matrix_safe(idx)

    def matrix_safe(self, idx: Optional[int]) -> np.ndarray:
        if idx is None:
            return self._identity
        try:
            return self.elements[int(idx) % self.size]
        except Exception:
            return self._identity

    def embed(self, idx: int) -> np.ndarray:
        return self.embed_safe(idx)

    def embed_safe(self, idx: Optional[int]) -> np.ndarray:
        if idx is None:
            return self._embeddings[0]
        try:
            return self._embeddings[int(idx) % self.size]
        except Exception:
            return self._embeddings[0]

    def describe(self, idx: Optional[int]) -> str:
        if idx is None:
            return 'identity'
        return f'{self.name}[{int(idx)}]'

    def step(self, state: int) -> int:
        gen_idx = int(self._rng.choice(self.size, p=self.generator_weights))
        next_state = int(self.cayley[state, gen_idx])
        if self.jitter > 0.0:
            noise = float(self._rng.normal(0.0, self.jitter))
            if abs(noise) > 0.5:
                next_state = int(self.cayley[next_state, gen_idx])
        return int(next_state % self.size)

    def energy(self, idx: Optional[int]) -> float:
        if idx is None:
            return 0.0
        try:
            return float(self._element_norms[int(idx) % self.size])
        except Exception:
            return float(self._element_norms[0])

    def apply_to_points(self, idx: Optional[int], points: np.ndarray) -> np.ndarray:
        if idx is None or points.size == 0:
            return points
        try:
            mat_t = self._transposes[int(idx) % self.size]
            return np.asarray(points @ mat_t, dtype=np.float32)
        except Exception:
            return np.asarray(points, dtype=np.float32)

    @classmethod
    def dihedral(cls, order: int=4, twist: float=0.0, seed: Optional[int]=None) -> 'SpinorGroupInteraction':
        if order <= 0:
            raise ValueError('order must be positive for dihedral group.')
        rotations = []
        for k in range(order):
            ang = 2.0 * math.pi * k / order
            c = math.cos(ang)
            s = math.sin(ang)
            rotations.append(np.array([[c, -s], [s, c]], dtype=np.float32))
        reflections = []
        for k in range(order):
            ang = math.pi * k / order + twist
            c = math.cos(ang)
            s = math.sin(ang)
            reflections.append(np.array([[c, s], [s, -c]], dtype=np.float32))
        elems = np.stack(rotations + reflections, axis=0)
        n = elems.shape[0]
        cayley = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                mat = elems[i] @ elems[j]
                diff = np.linalg.norm(elems - mat, axis=(1, 2))
                idx = int(np.argmin(diff))
                cayley[i, j] = idx
        weights = np.ones(n, dtype=np.float64)
        weights[:order] = 2.0
        weights[order:] = 1.0
        return cls(name=f'D_{order}', elements=elems, cayley=cayley, generator_weights=weights, seed=seed)

@dataclass
class SpinorScheduler:
    period_gens: int = 48
    jitter_std: float = 0.06
    seed: Optional[int] = None
    base_phase: float = 0.0
    group: Optional[SpinorGroupInteraction] = None
    max_cached_states: int = 1024
    _rng: np.random.Generator = field(init=False, repr=False)
    _group_states: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _state_order: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.group is not None and not isinstance(self.group, SpinorGroupInteraction):
            raise TypeError('SpinorScheduler.group must be SpinorGroupInteraction or None.')

    def phase(self, gen: int) -> float:
        omega = 4.0 * math.pi / float(max(1, self.period_gens))
        jitter = float(self._rng.normal(0.0, self.jitter_std))
        return self.base_phase + omega * float(gen) + jitter

    def _group_state_for(self, gen: int) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.group is None:
            return (None, None, None)
        if gen in self._group_states:
            idx = self._group_states[gen]
        else:
            if gen == 0:
                idx = 0
            else:
                prev_idx = self._group_state_for(gen - 1)[0] or 0
                idx = self.group.step(prev_idx)
            self._group_states[gen] = idx
            self._state_order.append(gen)
            while len(self._group_states) > max(1, int(self.max_cached_states)):
                old_gen = self._state_order.popleft()
                if old_gen == gen:
                    break
                self._group_states.pop(old_gen, None)
        mat = self.group.matrix_safe(idx)
        embed = self.group.embed_safe(idx)
        return (idx, mat, embed)

    def phase_bundle(self, gen: int) -> Tuple[float, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], float]:
        theta = self.phase(gen)
        parity = SpinorScheduler.parity(theta)
        idx, mat, embed = self._group_state_for(gen)
        energy = self.group.energy(idx) if self.group is not None else 0.0
        return (theta, parity, idx, mat, embed, energy)

    def describe_group(self, idx: Optional[int]) -> str:
        if self.group is None:
            return 'singlet'
        return self.group.describe(idx)

    @staticmethod
    def spinor_vec(theta: float) -> np.ndarray:
        return np.array([math.cos(theta / 2.0), math.sin(theta / 2.0)], dtype=np.float32)

    @staticmethod
    def parity(theta: float) -> int:
        m = theta % (4.0 * math.pi) / (2.0 * math.pi)
        return 0 if m < 1.0 else 1

@dataclass
class NomologyEnv:
    intensity: float = 0.2
    drift_scale: float = 0.004
    seed: Optional[int] = None
    noise_weaver_seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)
    noise: float = 0.06
    turns: float = 1.6
    rot_bias: float = 0.0
    lazy_share: float = 0.0
    lazy_anchor: float = 0.0
    lazy_gap: float = 0.0
    lazy_stasis: float = 0.0
    regime_id: int = 0
    noise_kind: str = 'white'
    noise_profile: Dict[str, Any] = field(
        default_factory=lambda: {
            'cycle_phase': 0.0,
            'envelope': 0.0,
            'jitter': 0.0,
            'spectral_bias': 0.0,
            'band_label': 'white',
        }
    )
    noise_kind_label: str = 'White'
    noise_kind_symbol: str = 'W'
    noise_kind_color: str = '#f6f7fb'
    noise_kind_code: int = 0
    noise_palette: Tuple[str, ...] = ('white', 'alpha', 'beta', 'black')
    noise_stage_len: int = 6
    noise_jitter: float = 0.006
    noise_levels: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'white': (0.045, 0.012),
            'alpha': (0.052, 0.014),
            'beta': (0.058, 0.018),
            'black': (0.068, 0.022),
        }
    )
    noise_min: float = 0.02
    noise_max: float = 0.14
    noise_weaver: Optional[SpectralNoiseWeaver] = None
    noise_focus: float = 0.0
    noise_entropy: float = 0.0
    noise_harmonics: Dict[str, float] = field(default_factory=dict)
    noise_style_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _noise_counter: float = field(default=0.0, init=False, repr=False)
    _last_weaver_error: Optional[str] = field(default=None, init=False, repr=False)
    diversity_signal: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.noise_weaver is None:
            try:
                weaver_seed = self.noise_weaver_seed if self.noise_weaver_seed is not None else self.seed
                self.noise_weaver = SpectralNoiseWeaver(palette=self.noise_palette, seed=weaver_seed)
            except Exception as weaver_err:
                self._last_weaver_error = f'{type(weaver_err).__name__}: {weaver_err}'
                self.noise_weaver = None
        else:
            try:
                self.noise_weaver.palette = tuple(self.noise_palette)
            except Exception:
                pass
            if self.noise_weaver_seed is not None:
                try:
                    self.noise_weaver.seed = self.noise_weaver_seed
                    self.noise_weaver._rng = np.random.default_rng(self.noise_weaver_seed)
                except Exception:
                    pass
        self._noise_counter = 0.0
        self._refresh_noise(surge=True)
        self.last_advantage_score = 0.0
        self.last_altruism_signal = 0.5
        self.last_selfish_drive = 0.0
        self.last_env_shift = 0.0
        self.last_leader_id = None
        self.last_advantage_penalty = False

    def noise_style(self, kind: Optional[str]=None) -> Dict[str, Any]:
        target = kind or self.noise_kind
        return _resolve_noise_style(target, self.noise_style_overrides)

    def _noise_ctx(self, surge: bool=False) -> Dict[str, Any]:
        return {
            'palette': self.noise_palette,
            'stage_len': max(1, int(self.noise_stage_len)),
            'jitter_amp': float(self.noise_jitter * (1.5 if surge else 1.0)),
            'base_levels': dict(self.noise_levels),
            'min_std': float(self.noise_min),
            'max_std': float(self.noise_max),
            'levels': dict(self.noise_levels),
            'surge': bool(surge),
            'intensity': float(self.intensity),
            'regime_id': int(self.regime_id),
            'drift_scale': float(self.drift_scale),
        }

    def register_diversity(self, snapshot: Dict[str, float]) -> None:
        if not isinstance(snapshot, dict):
            return
        payload = {k: float(v) for k, v in snapshot.items() if isinstance(v, (int, float))}
        self.diversity_signal = payload
        scarcity = float(np.clip(payload.get('scarcity', 0.0), 0.0, 1.0))
        spread = float(np.clip(payload.get('structural_spread', 0.0), 0.0, 4.0))
        self.intensity = float(np.clip(self.intensity * (0.92 + 0.28 * scarcity), 0.05, 0.8))
        self.noise_jitter = float(np.clip(self.noise_jitter * (1.0 + 0.2 * spread), 0.001, 0.05))

    def _refresh_noise(self, advance: bool=False, surge: bool=False) -> None:
        if advance:
            self._noise_counter += 1.0
        std, kind, profile = _cyclic_noise_profile(self._noise_counter, self._noise_ctx(surge))
        jitter_extra = float(self._rng.normal(0.0, self.noise_jitter * (0.35 if surge else 0.25)))
        std = float(np.clip(std + jitter_extra, self.noise_min, self.noise_max))
        profile = dict(profile)
        profile.setdefault('spectral_bias', 0.0)
        profile.setdefault('band_label', kind)
        profile.setdefault('cycle_phase', 0.0)
        profile['jitter'] = float(profile.get('jitter', 0.0) + jitter_extra)
        weaver = getattr(self, 'noise_weaver', None)
        if weaver is not None:
            weaver_ctx = self._noise_ctx(surge)
            try:
                std, kind, profile = weaver.compose(
                    self._noise_counter,
                    std,
                    kind,
                    profile,
                    ctx=weaver_ctx,
                )
                self._last_weaver_error = None
            except Exception as weaver_err:
                self._last_weaver_error = f'{type(weaver_err).__name__}: {weaver_err}'
        profile['regime'] = int(self.regime_id)
        profile['counter'] = float(self._noise_counter)
        profile['surge'] = bool(surge)
        profile['noise_jitter_extra'] = jitter_extra
        self.noise = std
        self.noise_kind = kind
        self.noise_profile = profile
        style = self.noise_style(kind)
        self.noise_kind_label = style.get('label', kind)
        self.noise_kind_symbol = style.get('symbol', kind[:1].upper() if kind else '?')
        self.noise_kind_color = style.get('color', '#9e9e9e')
        self.noise_kind_code = int(style.get('index', -1))
        profile['kind_label'] = self.noise_kind_label
        profile['kind_symbol'] = self.noise_kind_symbol
        profile['kind_color'] = self.noise_kind_color
        profile['kind_code'] = self.noise_kind_code
        profile['kind_bias'] = float(style.get('bias', 0.0))
        profile['style'] = {
            'label': self.noise_kind_label,
            'symbol': self.noise_kind_symbol,
            'color': self.noise_kind_color,
            'index': self.noise_kind_code,
            'bias': float(style.get('bias', 0.0)),
        }
        harmonics = profile.get('harmonics') if isinstance(profile, dict) else None
        if isinstance(harmonics, dict) and harmonics:
            clean = {str(k): float(v) for k, v in harmonics.items()}
            total = sum(max(0.0, float(v)) for v in clean.values())
            if total > 0.0 and math.isfinite(total):
                clean = {k: float(max(0.0, float(v)) / total) for k, v in clean.items()}
            arr = np.asarray(list(clean.values()), dtype=np.float64)
            if arr.size:
                focus_val = float(np.max(arr))
                entropy_val = float(-np.sum(arr * np.log(arr + 1e-09)))
            else:
                focus_val = 1.0
                entropy_val = 0.0
            self.noise_harmonics = clean
        else:
            focus_val = 1.0
            entropy_val = 0.0
            self.noise_harmonics = {}
        self.noise_focus = float(profile.get('mix_focus', focus_val))
        self.noise_entropy = float(profile.get('mix_entropy', entropy_val))

    def maybe_switch(self) -> bool:
        triggered = bool(self._rng.random() < self.intensity)
        if triggered:
            self.regime_id += 1
        self._refresh_noise(advance=True, surge=triggered)
        if triggered:
            if self.noise_kind == 'alpha':
                target_turns = 1.85
            elif self.noise_kind == 'beta':
                target_turns = 1.35
            elif self.noise_kind == 'black':
                target_turns = 2.15
            else:
                target_turns = 1.6
            self.turns = float(np.clip(target_turns + self._rng.normal(0.0, 0.35), 0.6, 3.0))
            self.rot_bias = float(self._rng.uniform(-math.pi, math.pi))
        return triggered

    def drift(self) -> None:
        self.rot_bias += float(self._rng.normal(0.0, self.drift_scale))
        self.turns = float(np.clip(self.turns + self._rng.normal(0.0, self.drift_scale), 0.6, 3.2))
        self._refresh_noise(advance=False, surge=False)

@dataclass
class SelfReproducingEvaluator:
    feature_dim: int
    spin: SpinorScheduler
    base_env: NomologyEnv
    population_size: int = 4
    rng: Optional[np.random.Generator] = None
    seed: Optional[int] = None
    mutate_weights_prob: float = 0.65
    mutate_connection_prob: float = 0.35
    mutate_node_prob: float = 0.2
    output_scale: Tuple[float, float, float] = (0.08, 1.0, math.pi / 4.0)
    mandatory_mode: bool = True
    council_top_k: int = 3
    council_random_k: int = 3
    _population: List[Genome] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _innov: InnovationTracker = field(init=False, repr=False)
    _next_gid: int = field(init=False, repr=False)
    last_event: Optional[str] = field(default=None, init=False)
    last_resilience: Optional[str] = field(default=None, init=False)
    _leader_cache_id: Optional[int] = field(default=None, init=False, repr=False)
    _leader_cache_rev: Tuple[int, int] = field(default=(-1, -1), init=False, repr=False)
    _leader_compiled: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _resilience_notes: deque = field(default_factory=lambda: deque(maxlen=16), init=False, repr=False)
    last_leader_council: Tuple[int, ...] = field(default_factory=tuple, init=False, repr=False)
    last_council_dispersion: float = field(default=0.0, init=False, repr=False)
    lazy_feedback_smoothing: float = 0.35
    lazy_feedback_decay: float = 0.25
    _lazy_feedback: Dict[str, Any] = field(
        default_factory=lambda: {'generation': -1, 'share': 0.0, 'anchor': 0.0, 'gap': 0.0, 'stasis': 0.0},
        init=False,
        repr=False,
    )
    _diversity_feedback: Dict[str, Any] = field(
        default_factory=lambda: {'scarcity': 0.0, 'entropy': 0.0, 'structural_spread': 0.0},
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.rng is not None:
            self._rng = self.rng
        else:
            self._rng = np.random.default_rng(self.seed)
        bias = self.feature_dim + 1
        outputs = 3
        self._innov = InnovationTracker(next_node_id=bias + outputs)
        self._next_gid = int(self._rng.integers(1 << 30))
        if not self._population:
            for _ in range(self.population_size):
                self._population.append(self._seed_genome())

    def _seed_genome(self) -> Genome:
        nodes: Dict[int, NodeGene] = {}
        for i in range(self.feature_dim):
            nodes[i] = NodeGene(i, 'input', 'identity')
        bias_id = self.feature_dim
        nodes[bias_id] = NodeGene(bias_id, 'bias', 'identity')
        out_ids = []
        for j in range(3):
            nid = self.feature_dim + 1 + j
            nodes[nid] = NodeGene(nid, 'output', 'tanh')
            out_ids.append(nid)
        conns: Dict[int, ConnectionGene] = {}
        for src in range(self.feature_dim + 1):
            for dst in out_ids:
                inn = self._innov.get_conn_innovation(src, dst)
                weight = float(self._rng.normal(0.0, 0.5))
                conns[inn] = ConnectionGene(src, dst, weight, True, inn)
        gid = self._next_gid
        self._next_gid += 1
        g = Genome(nodes, conns, gid=gid, birth_gen=0, parents=(None, None), cooperative=True)
        g.meta_reflect('init_env_genome', {'role': 'environment'})
        return g

    def _feature_vector(
        self,
        bundle: Tuple[float, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], float],
        env: NomologyEnv,
        generation: int,
    ) -> np.ndarray:
        theta, parity, _, _, embed, energy = bundle
        gen_norm = float((generation % max(1, self.spin.period_gens)) / max(1, self.spin.period_gens))
        noise_norm = float(np.clip(env.noise / 0.2, 0.0, 1.0))
        turns_norm = float(np.clip((env.turns - 0.6) / (3.2 - 0.6 + 1e-09), 0.0, 1.0))
        rot_norm = float((env.rot_bias + math.pi) / (2.0 * math.pi))
        style = env.noise_style() if hasattr(env, 'noise_style') else _resolve_noise_style(getattr(env, 'noise_kind', ''), None)
        palette = getattr(env, 'noise_palette', ('white', 'alpha', 'beta', 'black'))
        palette_span = max(1.0, float(len(palette) - 1))
        code_idx = float(style.get('index', -1))
        if code_idx < 0:
            code_norm = -1.0
        else:
            code_norm = float(np.clip((code_idx / palette_span) * 2.0 - 1.0, -1.0, 1.0))
        focus_norm = float(np.clip(getattr(env, 'noise_focus', 0.0), 0.0, 1.5) / 1.5)
        entropy_norm = float(np.clip(getattr(env, 'noise_entropy', 0.0), 0.0, 4.0) / 4.0)
        bias_norm = float(style.get('bias', 0.0))
        base = [
            math.cos(theta),
            math.sin(theta),
            float(parity),
            gen_norm,
            noise_norm,
            turns_norm,
            rot_norm,
            float(energy / 4.0),
            focus_norm,
            entropy_norm,
            code_norm,
            bias_norm,
        ]
        if embed is not None:
            base.extend(embed.tolist())
        vec = np.asarray(base, dtype=np.float32)
        if vec.size < self.feature_dim:
            vec = np.pad(vec, (0, self.feature_dim - vec.size))
        elif vec.size > self.feature_dim:
            vec = vec[:self.feature_dim]
        return vec.astype(np.float32)

    def _mutate_child(self, parent: Genome, bundle, features: np.ndarray, outputs: np.ndarray, generation: int) -> str:
        child = parent.copy()
        changed = False
        if self._rng.random() < self.mutate_weights_prob:
            child.mutate_weights(self._rng)
            changed = True
        if self._rng.random() < self.mutate_connection_prob:
            changed = child.mutate_add_connection(self._rng, self._innov) or changed
        if self._rng.random() < self.mutate_node_prob:
            changed = child.mutate_add_node(self._rng, self._innov) or changed
        if not changed:
            return 'steady-state'
        child.parents = (parent.id, None)
        child.id = self._next_gid
        self._next_gid += 1
        child.birth_gen = generation + 1
        child.meta_reflect(
            'env_self_reproduce',
            {
                'features': features.tolist(),
                'outputs': outputs.tolist(),
                'generation': generation,
                'parent': parent.id,
            },
        )
        self._population.insert(0, child)
        if len(self._population) > self.population_size:
            self._population.pop()
        return f'spawned {child.id} ← {parent.id}'

    def _compiled_leader(self, leader: Genome):
        rev = (getattr(leader, '_structure_rev', -1), getattr(leader, '_weights_rev', -1))
        if (
            self._leader_compiled is not None
            and self._leader_cache_id == leader.id
            and self._leader_cache_rev == rev
        ):
            return self._leader_compiled
        compiled = compile_genome(leader)
        self._leader_compiled = compiled
        self._leader_cache_id = leader.id
        self._leader_cache_rev = rev
        return compiled

    def update_lazy_feedback(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            return
        payload = dict(feedback)
        prev = getattr(self, '_lazy_feedback', {})
        if not isinstance(prev, dict):
            prev = {}
        alpha = float(np.clip(self.lazy_feedback_smoothing, 0.0, 0.95))
        if prev:
            blended: Dict[str, Any] = {}
            for key, val in payload.items():
                if isinstance(val, (int, float)):
                    old = prev.get(key, val)
                    if isinstance(old, (int, float)):
                        blended[key] = float(old) * alpha + float(val) * (1.0 - alpha)
                    else:
                        blended[key] = float(val)
                else:
                    blended[key] = val
            payload.update(blended)
        payload.setdefault('generation', prev.get('generation', payload.get('generation', -1)))
        self._lazy_feedback = payload

    def update_diversity(self, metrics: Dict[str, Any]) -> None:
        if not isinstance(metrics, dict):
            return
        snapshot = dict(self._diversity_feedback)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                snapshot[key] = float(val)
        self._diversity_feedback = snapshot

    def _record_resilience(self, err: BaseException, generation: int) -> str:
        label = f'{type(err).__name__}@{generation}'
        self.last_resilience = label
        try:
            self._resilience_notes.append(label)
        except Exception:
            pass
        return label

    def resilience_history(self) -> List[str]:
        return list(self._resilience_notes)

    def _step_single_leader(
        self,
        bundle: Tuple[float, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], float],
        env: NomologyEnv,
        generation: int,
    ) -> Dict[str, Any]:
        leader = self._population[0]
        feats = self._feature_vector(bundle, env, generation)
        resilience_flag = ''
        start = time.perf_counter()
        try:
            compiled = self._compiled_leader(leader)
            out = forward_batch(compiled, feats.reshape(1, -1))[0]
            out = np.asarray(out, dtype=np.float32)
        except Exception as err:
            resilience_flag = self._record_resilience(err, generation)
            out = np.zeros(3, dtype=np.float32)
        latency_ms = (time.perf_counter() - start) * 1000.0
        lazy = getattr(self, '_lazy_feedback', {}) or {}
        share = float(np.clip(lazy.get('share', 0.0), 0.0, 1.0))
        stasis = float(np.clip(lazy.get('stasis', 0.0), 0.0, 1.0))
        anchor = float(np.clip(lazy.get('anchor', 0.0), -1.0, 1.0))
        gap = float(np.clip(lazy.get('gap', 0.0), -1.0, 1.0))
        lazy_gen = int(lazy.get('generation', generation)) if isinstance(lazy, dict) else generation
        age = max(0, int(generation) - lazy_gen)
        decay = math.exp(-float(np.clip(self.lazy_feedback_decay, 0.05, 1.0)) * age)
        share *= decay
        stasis *= decay
        anchor *= decay
        gap *= decay
        div_state = getattr(self, '_diversity_feedback', {}) or {}
        scarcity = float(np.clip(div_state.get('scarcity', 0.0), 0.0, 1.0))
        spread = float(np.clip(div_state.get('structural_spread', 0.0), 0.0, 4.0))
        entropy = float(np.clip(div_state.get('entropy', 0.0), 0.0, 1.2))
        share *= float(np.clip(1.0 - 0.25 * scarcity, 0.2, 1.0))
        prev_noise = float(getattr(env, 'noise', 0.05))
        prev_turns = float(getattr(env, 'turns', 1.6))
        prev_rot = float(getattr(env, 'rot_bias', 0.0))
        scale_noise, scale_turns, scale_rot = self.output_scale
        scale_noise *= float(1.0 + 0.35 * scarcity)
        scale_turns *= float(1.0 + 0.2 * spread)
        scale_rot *= float(1.0 + 0.15 * max(0.0, 0.5 - entropy))
        mod_out0 = float(out[0] * (1.0 - 0.35 * share) + anchor * 0.35)
        mod_out1 = float(out[1] * (1.0 - 0.3 * share) + (anchor + gap * 0.5) * 0.3)
        mod_out2 = float(out[2] * (1.0 - 0.3 * share) + gap * 0.6)
        target_noise = float(np.clip(0.05 + scale_noise * mod_out0, 0.0, 0.25))
        target_turns = float(np.clip(1.6 + scale_turns * mod_out1, 0.6, 3.2))
        rot_target = prev_rot + scale_rot * mod_out2
        anchor_noise = float(np.clip(0.05 + scale_noise * anchor, 0.0, 0.25))
        anchor_turns = float(np.clip(1.6 + scale_turns * (anchor + gap * 0.25), 0.6, 3.2))
        rot_anchor = prev_rot + scale_rot * (anchor * 0.4 + gap * 0.6)
        inertia = float(np.clip(0.25 + 0.5 * share + 0.25 * stasis, 0.0, 0.9))
        inertia *= float(np.clip(1.0 - 0.4 * scarcity + 0.15 * spread, 0.2, 1.05))
        slip = max(0.0, 1.0 - inertia)
        anchor_mix = slip * 0.5 * stasis
        leader_mix = slip - anchor_mix
        env.noise = float(np.clip(prev_noise * inertia + target_noise * leader_mix + anchor_noise * anchor_mix, 0.0, 0.25))
        env.turns = float(np.clip(prev_turns * inertia + target_turns * leader_mix + anchor_turns * anchor_mix, 0.6, 3.2))
        rot_blend = prev_rot * inertia + rot_target * leader_mix + rot_anchor * anchor_mix
        env.rot_bias = float(((rot_blend) + math.pi) % (2.0 * math.pi) - math.pi)
        env.lazy_share = float(share)
        env.lazy_anchor = float(anchor)
        env.lazy_gap = float(gap)
        env.lazy_stasis = float(stasis)
        env_shift = (
            abs(env.noise - prev_noise) * 4.0
            + abs(env.turns - prev_turns)
            + 0.5 * abs(rot_blend - prev_rot)
        )
        selfish_drive = float(max(0.0, leader_mix - anchor_mix) * (1.0 - share))
        advantage_score = float(np.clip(selfish_drive * (max(0.0, gap) + 0.35 * scarcity) * (0.5 + env_shift), 0.0, 3.0))
        altruism_signal = float(np.clip(1.0 - min(1.0, advantage_score), 0.0, 1.0))
        self.last_advantage_score = advantage_score
        self.last_leader_id = leader.id
        self.last_leader_council = (leader.id,)
        self.last_council_dispersion = 0.0
        self.last_altruism_signal = altruism_signal
        self.last_selfish_drive = selfish_drive
        self.last_env_shift = env_shift
        try:
            summary = self._mutate_child(leader, bundle, feats, out, generation)
        except Exception as mutate_err:
            resilience_flag = resilience_flag or self._record_resilience(mutate_err, generation)
            summary = f'mutate-failed {type(mutate_err).__name__}'
        noise_kind = getattr(env, 'noise_kind', None)
        if noise_kind:
            label = getattr(env, 'noise_kind_label', noise_kind)
            symbol = getattr(env, 'noise_kind_symbol', noise_kind[:1].upper())
            summary = f"{summary} | noise {symbol}({label})"
        focus = getattr(env, 'noise_focus', None)
        if isinstance(focus, (int, float)) and focus > 0:
            summary = f'{summary} | focus {float(focus):.2f}'
        entropy = getattr(env, 'noise_entropy', None)
        if isinstance(entropy, (int, float)) and entropy > 0:
            summary = f'{summary} | entropy {float(entropy):.2f}'
        if scarcity > 0.0 or spread > 0.0:
            summary = f'{summary} | div {scarcity:.2f}/{spread:.2f}'
        weave_err = getattr(env, '_last_weaver_error', None)
        if weave_err:
            summary = f'{summary} | weave {str(weave_err).split(':', 1)[0]}'
        if resilience_flag:
            summary = f'{summary} | resilience {resilience_flag}'
        if share > 0.0:
            summary = f'{summary} | lazy {share:.2f}'
        if stasis > 0.0:
            summary = f'{summary} | stasis {stasis:.2f}'
        if abs(gap) > 0.01:
            summary = f'{summary} | gap {gap:+.2f}'
        if advantage_score > 0.05:
            summary = f'{summary} | adv {advantage_score:.2f}'
        self.last_event = summary
        return {
            'genome_id': leader.id,
            'note': summary,
            'resilience': resilience_flag,
            'latency_ms': latency_ms,
            'advantage': advantage_score,
            'selfish_drive': selfish_drive,
            'altruism_signal': altruism_signal,
        }

    def _step_council(
        self,
        bundle: Tuple[float, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], float],
        env: NomologyEnv,
        generation: int,
    ) -> Dict[str, Any]:
        feats = self._feature_vector(bundle, env, generation)
        council_seed: List[Genome] = list(self._population[: max(1, min(self.council_top_k, len(self._population)))])
        remaining = self._population[self.council_top_k:]
        if remaining:
            random_take = min(self.council_random_k, len(remaining))
            if random_take > 0:
                for idx in self._rng.permutation(len(remaining))[:random_take]:
                    council_seed.append(remaining[int(idx)])
        council: List[Genome] = []
        seen: Set[int] = set()
        for g in council_seed:
            gid = getattr(g, 'id', None)
            if gid is None or gid in seen:
                continue
            council.append(g)
            seen.add(gid)
        if not council:
            council.append(self._population[0])
        resilience_marks: List[str] = []
        vectors: Dict[int, np.ndarray] = {}
        start = time.perf_counter()
        for genome in council:
            try:
                compiled = compile_genome(genome)
                activations = forward_batch(compiled, feats.reshape(1, -1))[0][0]
                act_arr = np.asarray(activations, dtype=np.float32)
                out_idx = list(compiled.get('outputs', []))
                if out_idx:
                    vec = act_arr[out_idx[:3]]
                else:
                    vec = act_arr[:3]
                vec = np.asarray(vec, dtype=np.float32)
                if vec.size < 3:
                    vec = np.pad(vec, (0, 3 - vec.size))
                else:
                    vec = vec[:3]
                vectors[genome.id] = vec.astype(np.float32, copy=False)
            except Exception as err:
                resilience_marks.append(self._record_resilience(err, generation))
        latency_ms = (time.perf_counter() - start) * 1000.0
        if vectors:
            ordered_ids = list(vectors.keys())
            stacked = np.stack([vectors[i] for i in ordered_ids], axis=0)
            consensus = stacked.mean(axis=0).astype(np.float32)
            dispersion = float(np.mean(np.std(stacked, axis=0)))
        else:
            ordered_ids = [g.id for g in council]
            consensus = np.zeros(3, dtype=np.float32)
            dispersion = 0.0
        all_ids = ordered_ids + [g.id for g in council if g.id not in ordered_ids]
        if not all_ids:
            all_ids = [council[0].id]
        parent = council[int(self._rng.integers(len(council)))] if council else self._population[0]
        parent_outputs = vectors.get(parent.id, consensus)
        lazy = getattr(self, '_lazy_feedback', {}) or {}
        share = float(np.clip(lazy.get('share', 0.0), 0.0, 1.0))
        stasis = float(np.clip(lazy.get('stasis', 0.0), 0.0, 1.0))
        anchor = float(np.clip(lazy.get('anchor', 0.0), -1.0, 1.0))
        gap = float(np.clip(lazy.get('gap', 0.0), -1.0, 1.0))
        lazy_gen = int(lazy.get('generation', generation)) if isinstance(lazy, dict) else generation
        age = max(0, int(generation) - lazy_gen)
        decay = math.exp(-float(np.clip(self.lazy_feedback_decay, 0.05, 1.0)) * age)
        share *= decay
        stasis *= decay
        anchor *= decay
        gap *= decay
        div_state = getattr(self, '_diversity_feedback', {}) or {}
        scarcity = float(np.clip(div_state.get('scarcity', 0.0), 0.0, 1.0))
        spread = float(np.clip(div_state.get('structural_spread', 0.0), 0.0, 4.0))
        entropy = float(np.clip(div_state.get('entropy', 0.0), 0.0, 1.2))
        share *= float(np.clip(1.0 - 0.25 * scarcity, 0.2, 1.0))
        lazy_pressure = float(np.clip(share * (1.0 + 0.5 * stasis + 0.25 * abs(gap)), 0.0, 1.6))
        prev_noise = float(getattr(env, 'noise', 0.05))
        prev_turns = float(getattr(env, 'turns', 1.6))
        prev_rot = float(getattr(env, 'rot_bias', 0.0))
        scale_noise, scale_turns, scale_rot = self.output_scale
        scale_noise *= float(1.0 + 0.35 * scarcity + 0.55 * lazy_pressure)
        scale_turns *= float(1.0 + 0.2 * spread + 0.4 * lazy_pressure)
        scale_rot *= float(1.0 + 0.15 * max(0.0, 0.5 - entropy) + 0.35 * lazy_pressure)
        anchor_pull = float(np.clip(0.35 + 0.25 * lazy_pressure, 0.0, 0.85))
        gap_pull = float(np.clip(0.45 + 0.25 * lazy_pressure, 0.0, 0.9))
        mod_out0 = float(consensus[0] * (1.0 - anchor_pull) + anchor * anchor_pull)
        mod_out1 = float(consensus[1] * (1.0 - anchor_pull) + (anchor + gap * 0.5) * anchor_pull)
        mod_out2 = float(consensus[2] * (1.0 - gap_pull) + gap * gap_pull)
        target_noise = float(np.clip(0.05 + scale_noise * mod_out0, 0.0, 0.25))
        target_turns = float(np.clip(1.6 + scale_turns * mod_out1, 0.6, 3.2))
        rot_target = prev_rot + scale_rot * mod_out2
        anchor_noise = float(np.clip(0.05 + scale_noise * (anchor + 0.2 * lazy_pressure), 0.0, 0.25))
        anchor_turns = float(np.clip(1.6 + scale_turns * (anchor + gap * 0.25 + 0.15 * lazy_pressure), 0.6, 3.2))
        rot_anchor = prev_rot + scale_rot * (anchor * 0.4 + gap * 0.6 + 0.2 * lazy_pressure)
        inertia = float(np.clip(0.25 + 0.6 * share + 0.25 * stasis, 0.0, 0.92))
        inertia *= float(np.clip(1.0 - 0.35 * scarcity + 0.25 * spread + 0.15 * lazy_pressure, 0.2, 1.1))
        slip = max(0.0, 1.0 - inertia)
        anchor_ratio = float(np.clip(0.3 + 0.4 * stasis + 0.3 * lazy_pressure, 0.0, 0.95))
        anchor_mix = min(slip, slip * anchor_ratio)
        leader_mix = slip - anchor_mix
        env.noise = float(np.clip(prev_noise * inertia + target_noise * leader_mix + anchor_noise * anchor_mix, 0.0, 0.25))
        env.turns = float(np.clip(prev_turns * inertia + target_turns * leader_mix + anchor_turns * anchor_mix, 0.6, 3.2))
        rot_blend = prev_rot * inertia + rot_target * leader_mix + rot_anchor * anchor_mix
        env.rot_bias = float(((rot_blend) + math.pi) % (2.0 * math.pi) - math.pi)
        env.lazy_share = float(share)
        env.lazy_anchor = float(anchor)
        env.lazy_gap = float(gap)
        env.lazy_stasis = float(stasis)
        env_shift = (
            abs(env.noise - prev_noise) * 4.0
            + abs(env.turns - prev_turns)
            + 0.5 * abs(rot_blend - prev_rot)
            + dispersion
            + 1.2 * lazy_pressure
        )
        selfish_drive = float(max(0.0, leader_mix - anchor_mix) * (1.0 - 0.6 * share))
        advantage_score = float(
            np.clip(
                selfish_drive * (max(0.0, gap) + 0.35 * scarcity + 0.25 * lazy_pressure) * (0.5 + env_shift),
                0.0,
                3.0,
            )
        )
        altruism_signal = float(np.clip(1.0 - min(1.0, advantage_score), 0.0, 1.0))
        resilience_flag = ''
        if resilience_marks:
            uniq = list(dict.fromkeys(resilience_marks))
            resilience_flag = 'council:' + ','.join(uniq)
        self.last_advantage_score = advantage_score
        self.last_leader_id = parent.id
        self.last_leader_council = tuple(all_ids)
        self.last_council_dispersion = dispersion
        self.last_altruism_signal = altruism_signal
        self.last_selfish_drive = selfish_drive
        self.last_env_shift = env_shift
        try:
            summary = self._mutate_child(parent, bundle, feats, parent_outputs, generation)
        except Exception as mutate_err:
            resilience_flag = resilience_flag or self._record_resilience(mutate_err, generation)
            summary = f'mutate-failed {type(mutate_err).__name__}'
        noise_kind = getattr(env, 'noise_kind', None)
        if noise_kind:
            label = getattr(env, 'noise_kind_label', noise_kind)
            symbol = getattr(env, 'noise_kind_symbol', noise_kind[:1].upper())
            summary = f"{summary} | noise {symbol}({label})"
        focus = getattr(env, 'noise_focus', None)
        if isinstance(focus, (int, float)) and focus > 0:
            summary = f'{summary} | focus {float(focus):.2f}'
        entropy = getattr(env, 'noise_entropy', None)
        if isinstance(entropy, (int, float)) and entropy > 0:
            summary = f'{summary} | entropy {float(entropy):.2f}'
        if scarcity > 0.0 or spread > 0.0:
            summary = f'{summary} | div {scarcity:.2f}/{spread:.2f}'
        weave_err = getattr(env, '_last_weaver_error', None)
        if weave_err:
            summary = f'{summary} | weave {str(weave_err).split(':', 1)[0]}'
        if resilience_flag:
            summary = f'{summary} | resilience {resilience_flag}'
        if share > 0.0:
            summary = f'{summary} | lazy {share:.2f}'
        if stasis > 0.0:
            summary = f'{summary} | stasis {stasis:.2f}'
        if abs(gap) > 0.01:
            summary = f'{summary} | gap {gap:+.2f}'
        if dispersion > 0.0:
            summary = f'{summary} | council σ {dispersion:.2f}'
        summary = f'{summary} | council {len(all_ids)}'
        if lazy_pressure > 0.01:
            summary = f'{summary} | lazyP {lazy_pressure:.2f}'
        if advantage_score > 0.05:
            summary = f'{summary} | adv {advantage_score:.2f}'
        self.last_event = summary
        return {
            'genome_id': parent.id,
            'note': summary,
            'resilience': resilience_flag,
            'latency_ms': latency_ms,
            'advantage': advantage_score,
            'selfish_drive': selfish_drive,
            'altruism_signal': altruism_signal,
            'council_size': len(all_ids),
            'council_dispersion': dispersion,
        }

    def step(
        self,
        bundle: Tuple[float, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], float],
        env: NomologyEnv,
        generation: int,
    ) -> Dict[str, Any]:
        if not self._population:
            self._population.append(self._seed_genome())
        if not self.mandatory_mode:
            return self._step_single_leader(bundle, env, generation)
        return self._step_council(bundle, env, generation)

    @property
    def leader(self) -> Genome:
        if not self._population:
            self._population.append(self._seed_genome())
        return self._population[0]

class Telemetry:

    def __init__(self, tel_csv: str, regime_csv: str) -> None:
        self.tel_csv = tel_csv
        self.reg_csv = regime_csv
        expected_cols = 31
        if os.path.exists(self.tel_csv):
            try:
                with open(self.tel_csv, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
            except Exception:
                header = None
            if not header or len(header) < expected_cols:
                legacy_path = self.tel_csv + '.legacy'
                try:
                    os.replace(self.tel_csv, legacy_path)
                except OSError:
                    pass
        if not os.path.exists(self.tel_csv):
            with open(self.tel_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'gen',
                    'theta',
                    'theta_mod_2pi',
                    'theta_mod_4pi',
                    'parity',
                    'regime_id',
                    'noise',
                    'noise_kind',
                    'noise_kind_label',
                    'noise_kind_symbol',
                    'noise_kind_code',
                    'noise_kind_color',
                    'noise_cycle',
                    'noise_wave_hz',
                    'noise_spectral_bias',
                    'noise_jitter',
                    'noise_focus',
                    'noise_entropy',
                    'noise_harmonics',
                    'turns',
                    'rot_bias',
                    'group_idx',
                    'group_label',
                    'group_energy',
                    'lazy_share',
                    'lazy_anchor',
                    'lazy_gap',
                    'lazy_stasis',
                    'evaluator_id',
                    'evaluator_note',
                    'build_id',
                ])
        if not os.path.exists(self.reg_csv):
            with open(self.reg_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['gen', 'event', 'regime_id', 'build_id'])
        base, _ext = os.path.splitext(self.tel_csv)
        self.div_csv = base + '_diversity.csv'
        if not os.path.exists(self.div_csv):
            with open(self.div_csv, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'gen',
                    'entropy',
                    'scarcity',
                    'complexity_mean',
                    'complexity_std',
                    'structural_spread',
                    'diversity_bonus',
                    'diversity_power',
                    'env_noise',
                    'env_focus',
                    'env_entropy',
                    'lazy_share',
                    'unique_signatures',
                    'build_id',
                ])

    def log_step(
        self,
        gen: int,
        theta: float,
        parity: int,
        env: NomologyEnv,
        group_idx: Optional[int]=None,
        group_label: Optional[str]=None,
        group_energy: float=0.0,
        evaluator_meta: Optional[Dict[str, Any]]=None,
    ) -> None:
        t2 = theta % (2.0 * math.pi)
        t4 = theta % (4.0 * math.pi)
        eval_id = ''
        eval_note = ''
        if isinstance(evaluator_meta, dict):
            eval_id = evaluator_meta.get('genome_id', '')
            eval_note = evaluator_meta.get('note', '')
        profile = getattr(env, 'noise_profile', {}) or {}
        noise_kind = getattr(env, 'noise_kind', '')
        if hasattr(env, 'noise_style'):
            style = env.noise_style(noise_kind)
        else:
            style = _resolve_noise_style(noise_kind, None)
        noise_label = style.get('label', noise_kind)
        noise_symbol = style.get('symbol', noise_kind[:1].upper() if noise_kind else '')
        noise_code = int(style.get('index', -1))
        noise_color = style.get('color', '#9e9e9e')
        noise_cycle = float(profile.get('cycle_phase', 0.0))
        noise_wave = profile.get('wave_freq_hz')
        noise_wave = '' if noise_wave is None else float(noise_wave)
        noise_spectral = float(profile.get('spectral_bias', 0.0))
        noise_jitter = float(profile.get('jitter', 0.0))
        noise_focus = float(getattr(env, 'noise_focus', 0.0))
        noise_entropy = float(getattr(env, 'noise_entropy', 0.0))
        harmonics = getattr(env, 'noise_harmonics', {})
        if isinstance(harmonics, dict) and harmonics:
            harm_payload = _json.dumps({k: float(v) for k, v in harmonics.items()})
        else:
            harm_payload = '{}'
        lazy_share = float(getattr(env, 'lazy_share', 0.0))
        lazy_anchor = float(getattr(env, 'lazy_anchor', 0.0))
        lazy_gap = float(getattr(env, 'lazy_gap', 0.0))
        lazy_stasis = float(getattr(env, 'lazy_stasis', 0.0))
        with open(self.tel_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                gen,
                theta,
                t2,
                t4,
                parity,
                env.regime_id,
                env.noise,
                noise_kind,
                noise_label,
                noise_symbol,
                noise_code,
                noise_color,
                noise_cycle,
                noise_wave,
                noise_spectral,
                noise_jitter,
                noise_focus,
                noise_entropy,
                harm_payload,
                env.turns,
                env.rot_bias,
                '' if group_idx is None else int(group_idx),
                group_label or '',
                float(group_energy),
                lazy_share,
                lazy_anchor,
                lazy_gap,
                lazy_stasis,
                eval_id,
                eval_note,
                _build_stamp_short(),
            ])

    def log_regime(self, gen: int, env: NomologyEnv) -> None:
        with open(self.reg_csv, 'a', newline='') as f:
            csv.writer(f).writerow([gen, 'switch', env.regime_id, _build_stamp_short()])

    def log_diversity(self, gen: int, metrics: Dict[str, float]) -> None:
        if not isinstance(metrics, dict) or not metrics:
            return
        row = [
            int(gen),
            float(metrics.get('entropy', float('nan'))),
            float(metrics.get('scarcity', float('nan'))),
            float(metrics.get('complexity_mean', float('nan'))),
            float(metrics.get('complexity_std', float('nan'))),
            float(metrics.get('structural_spread', float('nan'))),
            float(metrics.get('diversity_bonus', float('nan'))),
            float(metrics.get('diversity_power', float('nan'))),
            float(metrics.get('env_noise', float('nan'))),
            float(metrics.get('env_focus', float('nan'))),
            float(metrics.get('env_entropy', float('nan'))),
            float(metrics.get('lazy_share', float('nan'))),
            int(metrics.get('unique_signatures', -1)),
            _build_stamp_short(),
        ]
        with open(self.div_csv, 'a', newline='') as f:
            csv.writer(f).writerow(row)

class SpinorNomologyDatasetController:
    """Updates shared datasets each generation. Why: make fitness see non-stationary regime."""

    def __init__(
        self,
        spin: SpinorScheduler,
        env: NomologyEnv,
        rng: np.random.Generator,
        n_tr: int=512,
        n_va: int=256,
        evaluator: Optional[SelfReproducingEvaluator]=None,
        evaluator_seed: Optional[int]=None,
        mandatory_mode: bool=True,
    ) -> None:
        self.spin = spin
        self.env = env
        self.rng = rng
        self.n_tr = n_tr
        self.n_va = n_va
        self.last_gen = None
        self.telemetry: Optional[Telemetry] = None
        self.evaluator = evaluator
        self.evaluator_seed = evaluator_seed
        self.mandatory_mode = bool(mandatory_mode)
        embed_dim = self.spin.group.embed_dim if self.spin.group else 0
        self.feature_dim = 12 + embed_dim
        self.evaluator_feature_dim = 12 + embed_dim
        self.last_bundle: Optional[Tuple[float, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], float]] = None
        self.lazy_feedback_smoothing = 0.4
        self._lazy_feedback: Dict[str, Any] = {'generation': -1, 'share': 0.0, 'anchor': 0.0, 'gap': 0.0, 'stasis': 0.0}
        self._diversity_state: Dict[str, Any] = {'generation': -1}
        self.last_evaluator_meta: Optional[Dict[str, Any]] = None
        try:
            self.env.lazy_share = 0.0
            self.env.lazy_anchor = 0.0
            self.env.lazy_gap = 0.0
            self.env.lazy_stasis = 0.0
        except Exception:
            pass
        if self.evaluator is None and self.evaluator_feature_dim > 0:
            if evaluator_seed is None:
                evaluator_seed = int(self.rng.integers(1 << 30))
            eval_rng = np.random.default_rng(int(evaluator_seed))
            self.evaluator = SelfReproducingEvaluator(
                self.evaluator_feature_dim,
                self.spin,
                self.env,
                rng=eval_rng,
                seed=int(evaluator_seed),
                mandatory_mode=self.mandatory_mode,
            )
        elif self.evaluator is not None and evaluator_seed is not None:
            try:
                self.evaluator.seed = int(evaluator_seed)
            except Exception:
                pass
        if self.evaluator is not None:
            try:
                self.evaluator.mandatory_mode = self.mandatory_mode
            except Exception:
                pass

    def _dataset_core(
        self,
        n: int,
        theta: float,
        parity: int,
        group_idx: Optional[int],
        group_embed: Optional[np.ndarray],
        group_matrix: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X, y = neat.make_spirals(n=n, noise=self.env.noise, turns=self.env.turns, seed=int(self.rng.integers(1 << 31)))
        except Exception as err:
            warnings.warn(f'Spinor dataset generation failed: {err}', RuntimeWarning)
            X = np.zeros((n, 2), dtype=np.float32)
            y = np.zeros(n, dtype=np.int64)
        X = rotate_2d(X, theta + self.env.rot_bias)
        if self.spin.group is not None and group_idx is not None:
            X = self.spin.group.apply_to_points(group_idx, X)
        elif group_matrix is not None:
            try:
                X = (group_matrix @ X.T).T
            except Exception:
                pass
        X_aug, _ = augment_with_spinor(X, theta, parity=parity, group_embed=group_embed)
        return (X_aug.astype(np.float32), y.astype(np.int64))

    def set_lazy_feedback(self, generation: int, feedback: Dict[str, Any]) -> None:
        if feedback is None:
            return
        payload = dict(feedback)
        payload['generation'] = int(generation)
        prev = getattr(self, '_lazy_feedback', None)
        if isinstance(prev, dict) and prev:
            mix = {}
            alpha = float(np.clip(self.lazy_feedback_smoothing, 0.0, 0.95))
            for key, val in payload.items():
                if isinstance(val, (int, float)):
                    old = prev.get(key, val)
                    if isinstance(old, (int, float)):
                        mix[key] = float(old) * alpha + float(val) * (1.0 - alpha)
                    else:
                        mix[key] = float(val)
                else:
                    mix[key] = val
            payload.update(mix)
        self._lazy_feedback = payload
        try:
            self.env.lazy_share = float(payload.get('share', 0.0))
            self.env.lazy_anchor = float(payload.get('anchor', 0.0))
            self.env.lazy_gap = float(payload.get('gap', 0.0))
            self.env.lazy_stasis = float(payload.get('stasis', 0.0))
        except Exception:
            pass
        if self.evaluator is not None:
            try:
                self.evaluator.update_lazy_feedback(payload)
            except Exception:
                pass

    def ingest_diversity_metrics(self, generation: int, metrics: Dict[str, Any]) -> None:
        if not isinstance(metrics, dict):
            return
        snapshot = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        snapshot['generation'] = int(generation)
        self._diversity_state = snapshot
        try:
            self.env.register_diversity(snapshot)
        except Exception:
            pass
        if self.evaluator is not None and hasattr(self.evaluator, 'update_diversity'):
            try:
                self.evaluator.update_diversity(snapshot)
            except Exception:
                pass
        if self.telemetry is not None and hasattr(self.telemetry, 'log_diversity'):
            try:
                self.telemetry.log_diversity(generation, snapshot)
            except Exception:
                pass

    def update_for_generation(self, gen: int, shmem=False) -> None:
        if self.last_gen == gen:
            return
        bundle = self.spin.phase_bundle(gen)
        theta, parity, group_idx, group_matrix, group_embed, group_energy = bundle
        self.last_bundle = bundle
        switched = self.env.maybe_switch()
        if switched:
            self.on_regime_switch(gen)
        evaluator_meta = None
        if self.evaluator is not None:
            try:
                self.evaluator.update_lazy_feedback(getattr(self, '_lazy_feedback', {}))
            except Exception:
                pass
            evaluator_meta = self.evaluator.step(bundle, self.env, generation=gen)
        self.env.drift()
        Xtr, ytr = self._dataset_core(self.n_tr, theta, parity, group_idx, group_embed, group_matrix)
        Xva, yva = self._dataset_core(self.n_va, theta, parity, group_idx, group_embed, group_matrix)
        neat._SHM_CACHE['Xtr'] = Xtr
        neat._SHM_CACHE['ytr'] = ytr
        neat._SHM_CACHE['Xva'] = Xva
        neat._SHM_CACHE['yva'] = yva
        self.last_gen = gen
        if self.telemetry is not None:
            self.telemetry.log_step(
                gen,
                theta,
                parity,
                self.env,
                group_idx=group_idx,
                group_label=self.spin.describe_group(group_idx),
                group_energy=group_energy,
                evaluator_meta=evaluator_meta,
            )
        self.last_evaluator_meta = evaluator_meta
    on_regime_switch = lambda self, gen: self.telemetry and self.telemetry.log_regime(gen, self.env)

class SpinorNomologyFitness(neat.FitnessBackpropShared):
    """Wraps FitnessBackpropShared and refreshes datasets on new generation."""

    def __init__(self, controller: SpinorNomologyDatasetController, get_generation, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.get_generation = get_generation
        self._last_checked_gen = None

    def __call__(self, g: 'neat.Genome') -> float:
        gen = int(self.get_generation())
        if gen != self._last_checked_gen:
            self.controller.update_for_generation(gen, shmem=False)
            self._last_checked_gen = gen
        return super().__call__(g)
if __name__ == '__main__':
    raise SystemExit(main())
