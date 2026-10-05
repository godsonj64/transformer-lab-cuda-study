"""
dashboard.py - live pygame dashboard for training the transformer language model.

The same visual language as the maze and chess dashboards (black stage, thin rules,
letter-spaced captions, inferno / magma / viridis / red-blue colour maps). Training runs in
a background thread; the window only reads the trainer's published measurements, and every
request that touches the network (evaluation, a new passage, generation, saving) is queued
to that thread, so the model is never used from two threads at once.

Every number and picture comes from the real model on the real corpus:

    Training    loss curves (train, validation, uniform / unigram / bigram references),
                learning-rate schedule, gradient norm, loss by context position, loss by
                token frequency, per-tensor update ratios
    Data        the corpus and its splits, the tokenizer, Zipf's law on the training
                tokens, the batch the model just trained on, epoch progress
    Architecture  the whole model as a diagram; inside the selected block, every stage
                (norms, Q/K/V, each head's attention, the MLP hidden units, both residual
                additions) as heat maps of the real activations
    Predict     a validation passage coloured by surprisal; top predictions per token;
                calibration and accuracy
    Attention   attention maps of every head on the passage
    Heads       induction, previous-token and duplicate-token scores; in-context copying
    Logit Lens  what every layer's residual stream would predict
    Stream 3-D  the residual stream in 3-D, with the strongest attention links
    Embeddings  principal components of the token embeddings in 3-D; nearest neighbours
    Generate    sampling with temperature / top-k / top-p (training pauses)
    Export      time-lapse capture of measurements and dashboard panels; one-click appendix
                of editorial figures, strips, data sheets and cards (see export.py)
"""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import export as ex
from model import Sampler
from probes import nearest_neighbours, rope_wavelengths
from train import Trainer, lr_at

# --------------------------------------------------------------------------- #
# Palette and colour maps (shared with chess_ai.py / maze_solver.py)
# --------------------------------------------------------------------------- #
CMAPS = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "magma": [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)],
    "inferno": [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
    "cividis": [(0, 34, 78), (64, 77, 107), (124, 123, 120), (188, 175, 111), (255, 234, 70)],
    "rdbu": [(178, 24, 43), (239, 138, 98), (247, 247, 247), (103, 169, 207), (33, 102, 172)],
    "heat": [(40, 60, 190), (40, 140, 240), (70, 210, 230), (250, 225, 90), (250, 120, 40), (225, 30, 35)],
}
_CMAP_ARR = {k: np.array(v, dtype=np.float32) for k, v in CMAPS.items()}
NODATA_C = (40, 43, 45)

INK = (247, 249, 250)
INK_DIM = (200, 206, 209)        # secondary text: 12:1 contrast on black
INK_FAINT = (96, 103, 106)        # panel rules, frames, axes: 3.6:1 (WCAG non-text >= 3)
BG = (0, 0, 0)
PANEL_HI = (22, 25, 27)
CARD = (5, 6, 7)
BORDER = INK_FAINT
TEXT = INK
DIM = INK_DIM
FAINT = (164, 170, 173)          # hints and axis labels: 8.6:1 on black (WCAG AA >= 4.5)
GRID_C = (40, 44, 46)
BAD = (232, 100, 92)
VAL_C = (72, 216, 255)            # validation
LR_C = (252, 214, 64)             # learning rate, highlights
GOOD_C = (88, 222, 128)
WARN_C = (255, 72, 88)
LINK_POS_C = (96, 172, 236)
LINK_NEG_C = (238, 104, 84)
LOG_COLORS = {"eval": VAL_C, "info": INK_DIM, "warn": BAD, "save": GOOD_C, "gen": INK}

VIEWS = (("training", "Training"), ("data", "Data"), ("arch", "Architecture"), ("predict", "Predict"),
         ("attention", "Attention"), ("heads", "Heads"), ("lens", "Logit Lens"), ("network", "Stream 3-D"),
         ("embed", "Embeddings"), ("generate", "Generate"), ("export", "Export"))
VIEW_KEYS = tuple(v[0] for v in VIEWS)
VIEW_NAMES = dict(VIEWS)
BUCKET_NAMES = ("top 10", "11-100", "101-1k", "1k-10k", "> 10k")


def apply_cmap(t: np.ndarray, name: str = "inferno") -> np.ndarray:
    """Values in [0, 1] (NaN -> 0) to uint8 RGB with shape t.shape + (3,)."""
    anchors = _CMAP_ARR[name]
    t = np.nan_to_num(np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0), nan=0.0)
    pos = t * (len(anchors) - 1)
    i = np.clip(pos.astype(int), 0, len(anchors) - 2)
    f = (pos - i)[..., None]
    return (anchors[i] * (1 - f) + anchors[i + 1] * f).astype(np.uint8)


def cmap_color(t: float, name: str = "inferno") -> tuple[int, int, int]:
    return tuple(int(c) for c in apply_cmap(np.array([t]), name)[0])


def heat_rgb(a: np.ndarray, lo: float, hi: float, cmap: str = "inferno", log: bool = False) -> np.ndarray:
    """Heat map on a FIXED scale lo..hi (log uses both bounds); NaN -> NODATA_C."""
    a = np.asarray(a, dtype=np.float64)
    if log:
        lower, upper = math.log1p(max(lo, 0)), math.log1p(max(hi, 0))
        t = (np.log1p(np.maximum(a, 0)) - lower) / (upper - lower) if upper > lower else np.zeros_like(a)
    else:
        t = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    rgb = apply_cmap(t, cmap)
    rgb[~np.isfinite(a)] = NODATA_C
    return rgb


def rgb_surface(pg, rgb: np.ndarray, size: tuple[int, int] | None = None):
    """(H, W, 3) array -> Surface (surfarray indexes [x][y], hence the transpose)."""
    surf = pg.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    return pg.transform.scale(surf, size) if size else surf


def mix_rgb(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


def luminance(rgb) -> float:
    """WCAG relative luminance of an sRGB colour."""
    lin = [(c / 255) / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b) -> float:
    """WCAG contrast ratio between two colours (1 to 21)."""
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def ink_on(rgb) -> tuple[int, int, int]:
    """Black or white text, whichever has the higher contrast on rgb (one of the two always
    reaches at least sqrt(21) = 4.58:1, above WCAG AA)."""
    return (0, 0, 0) if contrast((0, 0, 0), rgb) > contrast((255, 255, 255), rgb) else (255, 255, 255)


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1) x (2r+1) square of the first two axes, zeros outside (running sums);
    extra trailing axes (colour channels) are blurred independently."""
    if r < 1:
        return a
    k = 2 * r + 1
    rest = [(0, 0)] * (a.ndim - 2)
    c = np.cumsum(np.pad(a, [(0, 0), (r + 1, r)] + rest), axis=1)
    a = (c[:, k:] - c[:, :-k]) / k
    c = np.cumsum(np.pad(a, [(r + 1, r), (0, 0)] + rest), axis=0)
    return (c[k:] - c[:-k]) / k


def shutil_disk_free(path: str) -> float:
    import shutil
    return float(shutil.disk_usage(path).free)


def fmt_tokens(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def fmt_duration(s: float) -> str:
    if not math.isfinite(s) or s < 0:
        return "-"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def smooth(a: np.ndarray, k: int) -> np.ndarray:
    """Trailing moving average (same length; the first values average what exists)."""
    a = np.asarray(a, dtype=np.float64)
    if k <= 1 or len(a) < 2:
        return a
    c = np.cumsum(np.insert(a, 0, 0.0))
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - k)
    return (c[idx] - c[lo]) / (idx - lo)


def decimate(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """At most n points: bin means (keeps the curve's shape for long histories)."""
    if len(x) <= n:
        return x, y
    k = int(math.ceil(len(x) / n))
    m = len(x) // k * k
    xs = x[:m].reshape(-1, k).mean(axis=1)
    ys = y[:m].reshape(-1, k).mean(axis=1)
    if m < len(x):
        xs, ys = np.append(xs, x[m:].mean()), np.append(ys, y[m:].mean())
    return xs, ys


class FreetypeFont:
    """pygame.font.Font look-alike for one face of a font collection."""

    def __init__(self, face):
        self.face = face
        face.pad = True
        face.antialiased = True

    def render(self, text: str, antialias: bool, color):
        return self.face.render(text or " ", color)[0]

    def size(self, text: str) -> tuple[int, int]:
        return self.face.get_rect(text or " ").width, self.get_height()

    def get_height(self) -> int:
        return self.face.get_sized_height()


@dataclass
class Camera:
    """Orbit camera: yaw turns around the vertical axis, pitch looks down from above."""
    yaw: float = -0.6
    pitch: float = 0.5
    dist: float = 5.0
    zoom: float = 1.0
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    auto: bool = True

    def orbit(self, dyaw: float, dpitch: float) -> None:
        self.yaw = (self.yaw + dyaw) % (2 * math.pi)
        self.pitch = min(1.45, max(-0.2, self.pitch + dpitch))

    def rotate(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        x, y, z = (p[..., k] - self.target[k] for k in range(3))
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        ce, se = math.cos(self.pitch), math.sin(self.pitch)
        x1, y1 = x * cy - y * sy, x * sy + y * cy
        return x1, y1 * se + z * ce, y1 * ce - z * se

    def project(self, p: np.ndarray, rect) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right, up, away = self.rotate(p)
        depth = self.dist + away
        f = 0.2 * min(rect[2], rect[3]) * self.dist * self.zoom / np.maximum(depth, 0.05)
        return rect[0] + rect[2] / 2 + right * f, rect[1] + rect[3] / 2 - up * f, depth


# --------------------------------------------------------------------------- #
# Background training thread
# --------------------------------------------------------------------------- #
@dataclass
class GenState:
    prompt: str
    prompt_ids: list[int]
    max_new: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    sampler: Sampler | None = None
    tokens: list[int] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    running: bool = True
    seconds: float = 0.0


class Worker:
    """Owns the trainer. All model work happens in its thread (or in service() calls when
    threaded=False, as in the tests); the GUI submits commands and reads results."""

    def __init__(self, trainer: Trainer, save_path: str | None = None, autosave_minutes: float = 10.0,
                 threaded: bool = True, start_paused: bool = False):
        self.tr = trainer
        self.save_path = save_path
        self.autosave_s = autosave_minutes * 60 if autosave_minutes else 0
        self.q: queue.Queue = queue.Queue()
        self.paused = start_paused
        self.hold = False                        # the Generate view holds training
        self.busy = ""
        self.gen: GenState | None = None
        self.neighbors: tuple[int, list] | None = None
        self.saved_step = trainer.step
        self.saved_at: float | None = None
        self.last_save_t = time.time()
        self.error: str | None = None
        self.threaded = threaded
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.initialized = False
        self.wall: deque = deque(maxlen=120)     # (time, tokens) for the wall-clock rate
        self.recorder: ex.FrameRecorder | None = None
        self.frames_dir = ex.default_frames_dir(save_path) if save_path else None
        self.capture_request: int | None = None  # a step whose dashboard panels the GUI should save
        self.capture_done = threading.Event()
        self.panel_capture = None                # set by the GUI: callable(step) that saves panels
        trainer.on_task = self._on_task

    def _on_task(self, name: str) -> None:
        self.busy = name

    def start(self) -> None:
        self.submit("init")
        if self.threaded:
            self.thread = threading.Thread(target=self._loop, name="trainer", daemon=True)
            self.thread.start()

    def submit(self, name: str, **kw) -> None:
        self.q.put((name, kw))

    def stop(self, timeout: float = 60.0) -> None:
        self._stop.set()
        self.q.put(("noop", {}))
        if self.thread is not None:
            self.thread.join(timeout)

    @property
    def can_train(self) -> bool:
        t = self.tr
        return not (self.paused or self.hold or t.done or t.stop_reason or self.error)

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.busy:
            return {"eval": "EVALUATING", "probe": "PROBING", "sample": "SAMPLING", "save": "SAVING",
                    "load": "LOADING", "capture": "CAPTURING"}.get(self.busy, self.busy.upper())
        if self.gen is not None and self.gen.running:
            return "GENERATING"
        if self.hold:
            return "PAUSED FOR GENERATION"
        if self.tr.stop_reason:
            return "STOPPED"
        if self.tr.done:
            return "FINISHED"
        return "PAUSED" if self.paused else "TRAINING"

    def wall_rate(self) -> float:
        if len(self.wall) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self.wall[0], self.wall[-1]
        return (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.service(timeout=0.05)

    def service(self, timeout: float = 0.0) -> None:
        """Run queued commands, then one unit of work (a generated token or a training step)."""
        try:
            handled = False
            while True:
                try:
                    name, kw = self.q.get_nowait()
                except queue.Empty:
                    break
                self._command(name, kw)
                handled = True
            if self._stop.is_set():
                return
            if self.gen is not None and self.gen.running:
                self._gen_step()
            elif self.can_train:
                self._train()
            elif not handled and timeout > 0:
                try:
                    name, kw = self.q.get(timeout=timeout)
                    self._command(name, kw)
                except queue.Empty:
                    pass
        except Exception as e:                         # keep the window alive; report and pause
            self.busy = ""
            self.error = f"{type(e).__name__}: {e}"
            self.tr.add_log("warn", f"training thread error: {self.error}")
            traceback.print_exc()

    def _train(self) -> None:
        if not self.initialized:
            self._init()
        self.tr.train_step()
        self.tr.periodic()
        self.wall.append((time.time(), self.tr.tokens))
        if self.recorder is not None and self.recorder.due(self.tr.step):
            self._capture()
        elif self.recorder is not None and not self.recorder.active and self.recorder.last_message.startswith("recording stopped"):
            self.tr.add_log("info", self.recorder.last_message)
            self.recorder.last_message = "stopped"
        if self.tr.done:
            self.tr.add_log("info", f"reached max_steps {self.tr.cfg.max_steps:,}; "
                                    "restart with a larger --max-steps to continue")
        if self.save_path and self.autosave_s and time.time() - self.last_save_t >= self.autosave_s:
            self._save(self.save_path, "autosaved")

    def _init(self) -> None:
        self.initialized = True
        self.tr.initial_measurements()

    def _capture(self) -> None:
        """One frame: fresh probes at this step, the measurements, and (with a window) every
        dashboard panel, drawn by the GUI thread while training waits."""
        tr, rec = self.tr, self.recorder
        self.busy = "capture"
        try:
            if tr.snapshot is None or tr.snapshot.get("step") != tr.step:
                tr.run_probes()
            rec.capture_data(tr)
            if rec.panels and self.panel_capture is not None:
                if self.threaded:
                    self.capture_done.clear()
                    self.capture_request = tr.step
                    if not self.capture_done.wait(30):
                        tr.add_log("warn", "dashboard frame capture timed out")
                        self.capture_request = None
                else:
                    self.panel_capture(tr.step)
            tr.add_log("info", f"captured frame {len(rec.captured)} at step {tr.step:,}")
        except OSError as e:
            tr.add_log("warn", f"capture failed: {e}")
        finally:
            self.busy = ""

    def _save(self, path: str, verb: str = "saved") -> None:
        self.busy = "save"
        try:
            self.tr.save(path)
            self.saved_step = self.tr.step
            self.saved_at = time.time()
            self.last_save_t = time.time()
            self.tr.add_log("save", f"{verb} {path} at step {self.tr.step:,}")
        except (OSError, RuntimeError) as e:
            self.tr.add_log("warn", f"save failed: {e}")
        finally:
            self.busy = ""

    def _command(self, name: str, kw: dict) -> None:
        tr = self.tr
        if name == "init":
            if not self.initialized:
                self._init()
        elif name == "pause":
            self.paused = kw.get("on", not self.paused)
        elif name == "step":
            if not self.initialized:
                self._init()
            if not (tr.done or tr.stop_reason or self.hold):
                tr.train_step()
                tr.periodic()
        elif name == "hold":
            self.hold = bool(kw.get("on"))
            if not self.hold and self.gen is not None:
                self.gen.running = False
        elif name == "eval":
            tr.run_eval()
        elif name == "snippet":
            tr.new_snippet(kw.get("seed"))
            self.busy = "probe"
            tr.run_probes()
            self.busy = ""
        elif name == "probe":
            self.busy = "probe"
            tr.run_probes()
            self.busy = ""
        elif name == "save":
            self._save(kw.get("path") or self.save_path)
        elif name == "load":
            path = kw.get("path") or self.save_path
            self.busy = "load"
            try:
                from train import load_checkpoint
                tr.load_state(load_checkpoint(path))
                self.saved_step = tr.step
                self.error = None
                tr.add_log("save", f"loaded {path} (step {tr.step:,})")
                tr.run_probes()
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                tr.add_log("warn", f"load failed: {e}")
            finally:
                self.busy = ""
        elif name == "generate":
            ids = tr.data.tokenizer.encode(kw["prompt"]) or [int(tr.data["validation"][0])]
            g = GenState(kw["prompt"], ids, int(kw.get("max_new", 120)), float(kw.get("temperature", 0.8)),
                         int(kw.get("top_k", 0)), float(kw.get("top_p", 0.95)), int(kw.get("seed", 0)))
            g.sampler = Sampler(tr.model, ids, g.temperature, g.top_k, g.top_p, g.seed)
            self.gen = g
        elif name == "gen_stop":
            if self.gen is not None:
                self.gen.running = False
        elif name == "neighbors":
            tok = int(kw["token"])
            self.neighbors = (tok, nearest_neighbours(tr.model, tok, 14))
        elif name == "error_clear":
            self.error = None
        elif name == "record":
            if kw.get("on"):
                directory = kw.get("directory") or self.frames_dir or ex.default_frames_dir(None)
                self.frames_dir = directory
                self.recorder = ex.FrameRecorder(directory, every=int(kw.get("every", 250)),
                                                 from_step=int(kw.get("from_step", tr.step)),
                                                 until_step=kw.get("until_step"), panels=bool(kw.get("panels", True)),
                                                 max_frames=int(kw.get("max_frames", 40)))
                self.recorder.max_frames = len(self.recorder.captured) + int(kw.get("max_frames", 40))
                tr.add_log("info", f"recording every {self.recorder.every} updates into {directory}")
                if kw.get("now", True) and self.recorder.due(tr.step):
                    self._capture()
            elif self.recorder is not None:
                self.recorder.active = False
                tr.add_log("info", f"recording stopped ({len(self.recorder.captured)} frames)")
        elif name == "capture_now":
            if self.recorder is None:
                self.recorder = ex.FrameRecorder(self.frames_dir or ex.default_frames_dir(None), every=10 ** 9,
                                                 from_step=tr.step, panels=bool(kw.get("panels", True)))
                self.recorder.active = False
            self._capture()

    def _gen_step(self) -> None:
        g = self.gen
        t0 = time.perf_counter()
        info = g.sampler.step()
        g.seconds += time.perf_counter() - t0
        g.tokens.append(info["token"])
        g.infos.append(info)
        if len(g.tokens) >= g.max_new:
            g.running = False


# --------------------------------------------------------------------------- #
# The dashboard
# --------------------------------------------------------------------------- #
class LabGUI:
    BASE_W, BASE_H = 1760, 990
    MAIN = (8, 48, 1232, 880)
    SIDE = (1252, 48, 500, 880)
    FOOT_Y = 936

    def __init__(self, worker: Worker, save_path: str = "gpt_wikitext.pt", fps: int = 30,
                 view: str = "training", autosave: bool = True):
        if view not in VIEW_KEYS:
            raise ValueError(f"view must be one of {VIEW_KEYS}")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        self.worker = worker
        self.tr = worker.tr
        self.save_path = save_path
        self.fps = fps
        self.autosave = autosave

        pygame.init()
        pygame.display.set_caption("Transformer Lab - GPT on WikiText-103")
        self.scale = 1.0
        try:
            dw, dh = pygame.display.get_desktop_sizes()[0]
            self.scale = min(1.0, (dw - 40) / self.BASE_W, (dh - 150) / self.BASE_H)
        except Exception:
            pass
        self.scale = max(0.1, self.scale)
        self._windowed_size = (int(self.BASE_W * self.scale), int(self.BASE_H * self.scale))
        self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        self.canvas = pygame.Surface((self.BASE_W, self.BASE_H), 0, 24)
        self.full_window = False
        self._windowed_position = None
        self.capture_select = False
        self._display_failed = False
        self._fit_viewport()
        self.clock = pygame.time.Clock()
        self.f_title = self._display(25)
        self.f_view = self._display(22)
        self.f_big = self._display(30)
        self.f_head = self._mono(15)
        self.f_flow = self._mono(13)
        self.font = self._mono(12)
        self.small = self.tiny = self.f_cap = self._mono(10)
        self._text_cache: dict = {}
        self._width_cache: dict = {}
        self._cache: dict = {}
        self.text_log: list | None = None        # set to [] to record every label's ink box (layout audits)

        self.view = view
        self.prev_view = "training"
        self.mouse = (-1, -1)
        self.tab_rects: dict[str, tuple] = {}
        self.sections: dict[str, tuple] = {}
        self.hits: list[tuple[tuple, str, object]] = []
        self.tip: list[tuple[str, tuple]] | None = None
        self.log_x = False
        self.sel_layer, self.sel_head = 0, 0
        self.avg_heads = False
        self.pinned: int | None = None
        self.lens_offset = 0
        self.gen_hover: int | None = None
        self.embed_sel: int | None = None
        self.prompt = " = = History = = \n The town was founded in"
        self.gen_settings = {"temperature": 0.8, "top_k": 0, "top_p": 0.95, "max_new": 160, "seed": 0}
        self.buttons: list[tuple[tuple, str]] = []
        self.rec_cfg = {"every": 250, "until": None, "max_frames": 40, "panels": True}
        self.export_cfg = {"themes": {"print", "slides"}, "width": "double", "formats": {"png", "pdf", "svg"},
                           "sections": set("ABCDE")}
        self.export_state = {"running": False, "progress": 0.0, "message": "", "out_dir": None, "error": None,
                             "previews": [], "preview_idx": 0}
        self.export_threaded = True
        self._preview_cache: dict = {}
        worker.panel_capture = self.capture_dashboard
        self.cams = {"network": Camera(yaw=-0.5, pitch=0.42, zoom=1.0, target=(0.0, 0.0, 0.0)),
                     "embed": Camera(yaw=-0.7, pitch=0.35, zoom=1.9, target=(0.0, 0.0, 0.0))}
        self._drag: tuple[int, int] | None = None
        self._frame_t = time.perf_counter()

    # -- fonts --------------------------------------------------------------- #
    def _mono(self, size: int):
        pg = self.pg
        try:
            return pg.font.SysFont("geistmono,sfnsmono,sfmono,menlo,dejavusansmono,couriernew", size)
        except Exception:
            return pg.font.Font(None, size + 4)

    def _display(self, size: int):
        pg = self.pg
        try:
            for name in ("geistextralight", "geistthin", "geistlight", "geist"):
                path = pg.font.match_font(name)
                if path:
                    return pg.font.Font(path, size)
            import pygame.freetype as ft
            ft.init()
            return FreetypeFont(ft.Font("/System/Library/Fonts/HelveticaNeue.ttc", size, font_index=12))
        except Exception:
            return self._mono(size)

    # -- text and chrome ----------------------------------------------------- #
    def tw(self, msg: str, font=None) -> int:
        font = font or self.font
        if "\x00" in msg:
            msg = msg.replace("\x00", "<00>")
        key = (msg, id(font))
        w = self._width_cache.get(key)
        if w is None:
            if len(self._width_cache) > 20000:
                self._width_cache.clear()
            w = self._width_cache[key] = font.size(msg)[0]
        return w

    def fit_text(self, msg: str, width: int, font=None) -> str:
        font = font or self.font
        if self.tw(msg, font) <= width:
            return msg
        suffix = "…"
        if self.tw(suffix, font) > width:
            return ""
        lo, hi = 0, len(msg)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.tw(msg[:mid] + suffix, font) <= width:
                lo = mid
            else:
                hi = mid - 1
        return msg[:lo].rstrip() + suffix

    def text(self, msg: str, pos, color=TEXT, font=None, align: str = "left") -> int:
        font = font or self.font
        if "\x00" in msg:                       # SDL_ttf cannot draw NUL (decoded text may contain it)
            msg = msg.replace("\x00", "<00>")
        key = (msg, color, id(font))
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) > 4000:
                self._text_cache.clear()
            img = self._text_cache[key] = font.render(msg, True, color)
        x, y = pos
        if align == "right":
            x -= img.get_width()
        elif align == "center":
            x -= img.get_width() // 2
        self.canvas.blit(img, (x, y))
        if self.text_log is not None and msg.strip():
            self.text_log.append((img.get_bounding_rect().move(x, y), msg))
        return img.get_width()

    def spaced(self, msg: str, pos, color=TEXT, font=None, spacing: float = 2.0, align: str = "left") -> int:
        img = self._spaced_img(msg, color, font or self.f_cap, spacing)
        x, y = pos
        if align == "right":
            x -= img.get_width()
        elif align == "center":
            x -= img.get_width() // 2
        self.canvas.blit(img, (x, y))
        if self.text_log is not None and msg.strip():
            self.text_log.append((img.get_bounding_rect().move(x, y), msg))
        return img.get_width()

    def _spaced_img(self, msg: str, color, font, spacing: float):
        pg = self.pg
        key = ("spaced", msg, color, id(font), spacing)
        img = self._text_cache.get(key)
        if img is None:
            glyphs = [font.render(ch, True, color) for ch in msg]
            width = sum(g.get_width() for g in glyphs) + spacing * max(len(glyphs) - 1, 0)
            img = pg.Surface((max(1, math.ceil(width)), font.get_height()), pg.SRCALPHA)
            x = 0.0
            for g in glyphs:
                img.blit(g, (round(x), 0))
                x += g.get_width() + spacing
            self._text_cache[key] = img
        return img

    def panel(self, rect, title: str | None = None, hint: str | None = None, key: str | None = None) -> None:
        x, y, w, _ = rect
        self.pg.draw.rect(self.canvas, BORDER, rect, 1)
        if key:
            self.sections[key] = (title or key, tuple(rect))
        if title:
            tw = self.spaced(title.upper(), (x + 12, y + 10), INK_DIM)
            if hint:
                self.text(self.fit_text(hint, w - tw - 36, self.small), (x + 12 + tw + 12, y + 10), FAINT, self.small)

    def pill(self, right: int, y: int, label: str, color, filled: bool = True) -> int:
        w = self.spaced(label, (right, y), color, align="right")
        self.pg.draw.rect(self.canvas, color, (right - w - 13, y + 2, 7, 7), 0 if filled else 1)
        return w + 13

    def colorbar(self, x: int, y: int, w: int, cmap: str, lo: str, hi: str, h: int = 8) -> int:
        wl = self.text(lo, (x, y - 2), DIM, self.small)
        grad = apply_cmap(np.linspace(0, 1, max(w, 2))[None, :], cmap)
        self.canvas.blit(rgb_surface(self.pg, np.repeat(grad, h, axis=0)), (x + wl + 4, y))
        wr = self.text(hi, (x + wl + w + 8, y - 2), DIM, self.small)
        return wl + w + 8 + wr

    def heat_tile(self, data: np.ndarray, pos, size: tuple[int, int], lo: float, hi: float,
                  cmap: str = "inferno", frame: bool = True) -> None:
        self.canvas.blit(rgb_surface(self.pg, heat_rgb(data, lo, hi, cmap), size), pos)
        if frame:
            self.pg.draw.rect(self.canvas, INK_FAINT, (pos[0] - 1, pos[1] - 1, size[0] + 2, size[1] + 2), 1)

    def dashed(self, color, p0, p1, dash: int = 5, gap: int = 4) -> None:
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1:
            return
        dx, dy = (x1 - x0) / length, (y1 - y0) / length
        t = 0.0
        while t < length:
            e = min(t + dash, length)
            self.pg.draw.line(self.canvas, color, (x0 + dx * t, y0 + dy * t), (x0 + dx * e, y0 + dy * e))
            t = e + gap

    def hit(self, rect, kind: str, payload=None) -> None:
        self.hits.append((tuple(int(v) for v in rect), kind, payload))

    def hovered(self, kind: str):
        """Payload of the hit box of this kind under the mouse. Boxes drawn later in the
        frame are looked up in the previous frame's boxes."""
        hits = self.hits if any(k == kind for _, k, _ in self.hits) else getattr(self, "hits_prev", [])
        for r, k, p in reversed(hits):
            if k == kind and self._inside(self.mouse, r):
                return p
        return None

    def button(self, rect, label: str, action: str) -> None:
        hot = self._inside(self.mouse, rect)
        self.pg.draw.rect(self.canvas, PANEL_HI if hot else CARD, rect)
        self.pg.draw.rect(self.canvas, INK_DIM if hot else INK_FAINT, rect, 1)
        self.text(label, (rect[0] + rect[2] // 2, rect[1] + (rect[3] - 13) // 2), INK if hot else DIM,
                  self.font, align="center")
        self.buttons.append((tuple(rect), action))

    def key_chip(self, x: int, y: int, key: str, label: str) -> int:
        kw = self.tw(key, self.tiny) + 10
        self.pg.draw.rect(self.canvas, PANEL_HI, (x, y, kw, 14))
        self.pg.draw.rect(self.canvas, INK_FAINT, (x, y, kw, 14), 1)
        self.text(key, (x + kw // 2, y + 1), INK, self.tiny, align="center")
        return kw + 6 + self.text(label, (x + kw + 6, y + 1), DIM, self.small) + 12

    def swatch(self, x: int, y: int, color, label: str, kind: str = "line") -> int:
        pg = self.pg
        if kind == "line":
            pg.draw.line(self.canvas, color, (x, y + 6), (x + 16, y + 6), 2)
        elif kind == "dash":
            self.dashed(color, (x, y + 6), (x + 16, y + 6), 4, 3)
        elif kind == "dot":
            pg.draw.circle(self.canvas, color, (x + 8, y + 6), 3)
        else:
            pg.draw.rect(self.canvas, color, (x + 3, y + 1, 10, 10))
        return 22 + self.text(label, (x + 22, y), DIM, self.small)

    @staticmethod
    def _inside(p, r) -> bool:
        return r[0] <= p[0] < r[0] + r[2] and r[1] <= p[1] < r[1] + r[3]

    # -- token text ----------------------------------------------------------- #
    def tok_text(self, i: int) -> str:
        """A token's text for running prose: real spaces, newline shown as ⏎."""
        key = ("tt", int(i))
        s = self._cache.get(key)
        if s is None:
            b = self.tr.data.tokenizer.token_bytes(int(i))
            s = b.decode("utf-8", errors="replace").replace("\n", "⏎").replace("\t", "  ").replace("\r", "")
            s = "".join(ch if ch.isprintable() or ch == " " else f"<{ord(ch):02x}>" for ch in s)
            if not s:
                s = "?"
            self._cache[key] = s
        return s

    def tok_label(self, i: int) -> str:
        """A token's label in isolation: leading space as ·, newline as ⏎."""
        return self.tr.data.tokenizer.display(int(i))

    def is_newline(self, i: int) -> bool:
        return b"\n" in self.tr.data.tokenizer.token_bytes(int(i))

    def flow(self, ids, rect, colors=None, font=None, line_h: int = 20, text_colors=None,
             kind: str = "tok", start_index: int = 0, outline: set | None = None) -> int:
        """Lay out tokens as wrapped prose; every token gets a background colour (or none)
        and a hit box. Returns the y below the last line."""
        font = font or self.f_flow
        x0, y0, w, h = rect
        x, y = x0, y0
        pg = self.pg
        for j, t in enumerate(ids):
            s = self.tok_text(t)
            wpx = max(4, self.tw(s, font))
            if x + wpx > x0 + w and x > x0:
                x, y = x0, y + line_h
            if y + line_h > y0 + h:
                break
            r = (x, y + 1, wpx, line_h - 3)
            if colors is not None and colors[j] is not None:
                pg.draw.rect(self.canvas, colors[j], r)
            if outline and (start_index + j) in outline:
                pg.draw.rect(self.canvas, INK, (r[0] - 1, r[1] - 1, r[2] + 2, r[3] + 2), 1)
            col = text_colors[j] if text_colors is not None else (
                ink_on(colors[j]) if colors is not None and colors[j] is not None else TEXT)
            self.text(s, (x, y + (line_h - font.get_height()) // 2), col, font)
            self.hit(r, kind, start_index + j)
            x += wpx
            if self.is_newline(t):
                x, y = x0, y + line_h
        return y + line_h

    # -- window ---------------------------------------------------------------- #
    def _fit_viewport(self) -> None:
        surface = self.pg.display.get_surface()
        if surface is not None:
            self.screen = surface
        sw, sh = self.screen.get_size()
        self.scale = min(sw / self.BASE_W, sh / self.BASE_H)
        vw, vh = max(1, round(self.BASE_W * self.scale)), max(1, round(self.BASE_H * self.scale))
        self._viewport = self.pg.Rect((sw - vw) // 2, (sh - vh) // 2, vw, vh)

    def _to_base(self, pos) -> tuple[int, int]:
        self._fit_viewport()
        r = self._viewport
        if not r.collidepoint(pos):
            return (-1, -1)
        return (int((pos[0] - r.x) * self.BASE_W / r.w), int((pos[1] - r.y) * self.BASE_H / r.h))

    def toggle_full_window(self) -> None:
        pg = self.pg
        self._fit_viewport()
        entering = not self.full_window
        get_position = getattr(pg.display, "get_window_position", None)
        set_position = getattr(pg.display, "set_window_position", None)
        old_position = get_position() if get_position else None
        if entering:
            self._windowed_size = self.screen.get_size()
            self._windowed_position = old_position
        old_size, old_flags = self.screen.get_size(), pg.FULLSCREEN if self.full_window else pg.RESIZABLE
        try:
            size = (0, 0) if entering else self._windowed_size
            self.screen = pg.display.set_mode(size, pg.FULLSCREEN if entering else pg.RESIZABLE)
            if not entering:
                pg.event.pump()
                if self.screen.get_size() != size:
                    self.screen = pg.display.set_mode(size, pg.RESIZABLE)
            if not entering and set_position and self._windowed_position is not None:
                set_position(self._windowed_position)
        except (pg.error, OSError, IndexError) as e:
            self.tr.add_log("warn", f"full-window change failed: {e}")
            try:
                self.screen = pg.display.set_mode(old_size, old_flags)
                if not self.full_window and set_position and old_position is not None:
                    set_position(old_position)
            except (pg.error, OSError) as restore_error:
                self._display_failed = True
                self.tr.add_log("warn", f"display unavailable: {restore_error}")
                return
        else:
            self.full_window = entering
            self.tr.add_log("info", "full window enabled (F11 to restore)" if entering else "window restored")
        self._drag = None
        self._fit_viewport()

    def _section_at(self, pos) -> str | None:
        best, area = None, None
        for key, (_, r) in self.sections.items():
            if self._inside(pos, r) and (area is None or r[2] * r[3] < area):
                best, area = key, r[2] * r[3]
        return best

    def _present(self) -> None:
        self._fit_viewport()
        self.screen.fill(BG)
        image = self.pg.transform.smoothscale(self.canvas, self._viewport.size)
        self.screen.blit(image, self._viewport)
        if self.capture_select:
            key = self._section_at(self.mouse)
            if key:
                x, y, w, h = self.sections[key][1]
                r = self._viewport
                outline = (round(r.x + x * r.w / self.BASE_W), round(r.y + y * r.h / self.BASE_H),
                           round(w * r.w / self.BASE_W), round(h * r.h / self.BASE_H))
                self.pg.draw.rect(self.screen, GOOD_C, outline, 2)
            hint = self.font.render("Click a panel to save it; ESC / right-click cancels", True, INK)
            self.screen.blit(hint, (self._viewport.x + 8, self._viewport.y + 4))

    def screenshot(self, section: str | None = None) -> str:
        folder = self.tr.logger.dir if self.tr.logger else "."
        tag = f"_{section}" if section else f"_{self.view}"
        path = os.path.join(folder, f"dashboard{tag}_step{self.tr.step}_{datetime.now().strftime('%H%M%S_%f')}.png")
        try:
            self.draw()
            image = self.canvas
            if section is not None:
                region = self.pg.Rect(self.sections[section][1]).clip(self.canvas.get_rect())
                image = self.canvas.subsurface(region).copy()
            self.pg.image.save(image, path)
            self.tr.add_log("info", f"screenshot saved: {path}")
        except Exception as e:
            self.tr.add_log("warn", f"screenshot failed: {e}")
        return path

    # -- events ------------------------------------------------------------------ #
    def set_view(self, key: str) -> None:
        if key == self.view:
            return
        if key == "generate":
            self.prev_view = self.view
            self.worker.submit("hold", on=True)
        elif self.view == "generate":
            self.worker.submit("hold", on=False)
        self.view = key
        self._drag = None

    def cycle_view(self, step: int) -> None:
        i = VIEW_KEYS.index(self.view)
        self.set_view(VIEW_KEYS[(i + step) % len(VIEW_KEYS)])

    def handle_event(self, event) -> bool:
        pg = self.pg
        if event.type == pg.QUIT:
            return False
        if event.type in (getattr(pg, "VIDEORESIZE", -1), getattr(pg, "WINDOWSIZECHANGED", -2)):
            self._fit_viewport()
            return True
        if event.type == pg.MOUSEMOTION:
            p = self._to_base(event.pos)
            if self._drag is not None and self.view in self.cams:
                cam = self.cams[self.view]
                cam.orbit((p[0] - self._drag[0]) * 0.008, (p[1] - self._drag[1]) * 0.006)
                self._drag = p
            self.mouse = p
            return True
        if event.type == pg.MOUSEBUTTONUP:
            self._drag = None
            return True
        if event.type == pg.MOUSEWHEEL:
            if self.view in self.cams:
                cam = self.cams[self.view]
                cam.zoom = min(4.0, max(0.3, cam.zoom * (1.1 ** event.y)))
            elif self.view == "lens":
                self.lens_offset = max(0, self.lens_offset - int(event.y) * 2)
            return True
        if event.type == pg.MOUSEBUTTONDOWN:
            p = self._to_base(event.pos)
            self.mouse = p
            if self.capture_select:
                if event.button == 1:
                    key = self._section_at(p)
                    if key:
                        self.screenshot(key)
                self.capture_select = False
                return True
            if event.button == 1:
                self.click(p)
            elif event.button == 3:
                self.pinned = None
                self.embed_sel = None if self.view == "embed" else self.embed_sel
            return True
        if event.type == getattr(pg, "TEXTINPUT", -3):
            if self.view == "generate" and event.text and event.text.isprintable():
                self.prompt = (self.prompt + event.text)[-2000:]
            return True
        if event.type == pg.KEYDOWN:
            return self.key(event)
        return True

    def click(self, p) -> None:
        for key, r in self.tab_rects.items():
            if self._inside(p, r):
                self.set_view(key)
                return
        for r, action in self.buttons:
            if self._inside(p, r):
                self.do_action(action)
                return
        for r, kind, payload in reversed(self.hits):
            if not self._inside(p, r):
                continue
            if kind == "head":
                self.sel_layer, self.sel_head = payload
                self.avg_heads = False
                return
            if kind == "block":
                self.sel_layer = int(payload)
                return
            if kind == "tok" and self.view == "predict":
                self.pinned = None if self.pinned == payload else payload
                return
            if kind == "gen":
                self.gen_hover = payload
                return
            if kind == "point":
                self.select_embed(payload)
                return
            if kind == "neighbor":
                self.select_embed(payload)
                return
        if self.view in self.cams and self._inside(p, self.MAIN):
            self._drag = p

    def select_embed(self, token: int) -> None:
        self.embed_sel = int(token)
        self.worker.submit("neighbors", token=int(token))

    def do_action(self, action: str) -> None:
        if action == "generate":
            self.start_generation()
        elif action == "stop":
            self.worker.submit("gen_stop")
        elif action == "clear":
            self.prompt = ""
        elif action.startswith("set:"):
            _, name, d = action.split(":")
            self.adjust(name, int(d))
        elif action == "rec_toggle":
            rec = self.worker.recorder
            if rec is not None and rec.active:
                self.worker.submit("record", on=False)
            else:
                c = self.rec_cfg
                self.worker.submit("record", on=True, every=c["every"], until_step=c["until"],
                                   max_frames=c["max_frames"], panels=c["panels"], from_step=self.tr.step)
        elif action == "capture_now":
            self.worker.submit("capture_now", panels=self.rec_cfg["panels"])
        elif action.startswith("rec:"):
            self.adjust_rec(action.split(":", 1)[1])
        elif action.startswith("exp:"):
            self.adjust_export(action.split(":", 1)[1])
        elif action == "export":
            self.start_export()
        elif action == "open_export":
            out = self.export_state.get("out_dir")
            if out and os.path.isdir(out) and sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", out])
        elif action.startswith("preview:"):
            n = len(self.export_state["previews"])
            if n:
                self.export_state["preview_idx"] = (self.export_state["preview_idx"] + int(action.split(":")[1])) % n

    def adjust(self, name: str, d: int) -> None:
        g = self.gen_settings
        if name == "temperature":
            g["temperature"] = round(min(2.0, max(0.0, g["temperature"] + 0.1 * d)), 2)
        elif name == "top_k":
            k = g["top_k"]
            g["top_k"] = (0 if k <= 1 else k // 2) if d < 0 else (1 if k == 0 else min(4096, k * 2))
        elif name == "top_p":
            g["top_p"] = round(min(1.0, max(0.05, g["top_p"] + 0.05 * d)), 2)
        elif name == "max_new":
            g["max_new"] = int(min(1000, max(10, g["max_new"] + 20 * d)))
        elif name == "seed":
            g["seed"] = max(0, g["seed"] + d)

    EVERY_CHOICES = (10, 25, 50, 100, 250, 500, 1000, 2500)

    def adjust_rec(self, what: str) -> None:
        c = self.rec_cfg
        name, _, d = what.partition(":")
        d = int(d or 0)
        if name == "every":
            i = min(range(len(self.EVERY_CHOICES)), key=lambda k: abs(self.EVERY_CHOICES[k] - c["every"]))
            c["every"] = self.EVERY_CHOICES[max(0, min(len(self.EVERY_CHOICES) - 1, i + d))]
        elif name == "until":
            span = max(c["every"] * 4, 1000)
            if d > 0:
                c["until"] = (c["until"] or self.tr.step) + span
            elif c["until"] is not None:
                c["until"] = None if c["until"] - span <= self.tr.step else c["until"] - span
        elif name == "max":
            c["max_frames"] = int(max(2, min(500, c["max_frames"] + 10 * d if c["max_frames"] >= 10 else c["max_frames"] + d)))
        elif name == "panels":
            c["panels"] = not c["panels"]

    def adjust_export(self, what: str) -> None:
        c = self.export_cfg
        kind, _, val = what.partition(":")
        if kind == "theme":
            if val in c["themes"] and len(c["themes"]) > 1:
                c["themes"].discard(val)
            else:
                c["themes"].add(val)
        elif kind == "width":
            c["width"] = val
        elif kind == "format":
            if val in c["formats"] and len(c["formats"]) > 1:
                c["formats"].discard(val)
            else:
                c["formats"].add(val)
        elif kind == "section":
            if val in c["sections"] and len(c["sections"]) > 1:
                c["sections"].discard(val)
            else:
                c["sections"].add(val)

    def capture_dashboard(self, step: int) -> str:
        """Draw every tab offscreen and save each panel (and each full tab) as PNG into the
        frame of `step`. The run-statistics column is saved once (from the Training tab)."""
        rec = self.worker.recorder
        directory = os.path.join(rec.step_dir(step) if rec else os.path.join(self.worker.frames_dir or ".",
                                                                            f"step_{step:07d}"), "panels")
        os.makedirs(directory, exist_ok=True)
        view, mouse, tip_log = self.view, self.mouse, self.text_log
        self.mouse, self.text_log = (-1, -1), None
        try:
            for key in VIEW_KEYS:
                if key in ("generate", "export"):
                    continue
                self.view = key
                self.draw()
                self.pg.image.save(self.canvas.subsurface(self.pg.Rect(self.MAIN)).copy(),
                                   os.path.join(directory, f"{key}__full.png"))
                for name, (_, rect) in self.sections.items():
                    if name == "view" or (name in ("stats", "graphs", "log") and key != "training"):
                        continue
                    r = self.pg.Rect(rect).clip(self.canvas.get_rect())
                    if r.w > 4 and r.h > 4:
                        self.pg.image.save(self.canvas.subsurface(r).copy(), os.path.join(directory, f"{key}__{name}.png"))
        finally:
            self.view, self.mouse, self.text_log = view, mouse, tip_log
        return directory

    def service_capture(self) -> None:
        """Answer the training thread's request to save dashboard frames (threaded mode)."""
        step = self.worker.capture_request
        if step is not None:
            try:
                self.capture_dashboard(step)
            except Exception as e:                       # never leave the trainer waiting
                self.tr.add_log("warn", f"dashboard frame capture failed: {e}")
            finally:
                self.worker.capture_request = None
                self.worker.capture_done.set()

    def start_export(self) -> None:
        st = self.export_state
        if st["running"]:
            return
        c = self.export_cfg
        opts = ex.ExportOptions(themes=tuple(t for t in ("print", "slides") if t in c["themes"]), width=c["width"],
                                formats=tuple(f for f in ("png", "pdf", "svg") if f in c["formats"]),
                                sections=tuple(k for k in "ABCDE" if k in c["sections"]))
        try:
            ctx = ex.context_from_trainer(self.tr, self.worker.frames_dir, self.save_path)
        except Exception as e:
            st.update(error=str(e), message=f"export failed: {e}")
            return
        out = ex.default_export_dir(self.save_path, self.tr.step)
        st.update(running=True, progress=0.0, message="starting", out_dir=out, error=None)

        def progress(msg: str, frac: float) -> None:
            st["message"], st["progress"] = msg, frac

        def job() -> None:
            try:
                readme = ex.build_appendix(ctx, out, opts, progress)
                pngs = []
                for root, _, files in os.walk(out):
                    pngs += [os.path.join(root, f) for f in sorted(files) if f.endswith(".png")]

                def rank(p: str) -> tuple:
                    b = os.path.basename(p)
                    order = (b.startswith("B0_"), "timelapse" in b, b.startswith("A1_"), b.startswith("C"),
                             b.startswith("B"), b.startswith("D"), b.startswith("E_"))
                    first = next((i for i, v in enumerate(order) if v), len(order))
                    return (first, "_slides" in b, b)

                st["previews"] = sorted(pngs, key=rank)
                st["preview_idx"] = 0
                n = sum(len(fs) for _, _, fs in os.walk(out))
                st["message"] = f"done: {n} files, index {os.path.relpath(readme, os.path.dirname(out))}"
                self.tr.add_log("save", f"appendix exported to {out}")
            except Exception as e:
                st["error"] = f"{type(e).__name__}: {e}"
                st["message"] = f"export failed: {st['error']}"
                self.tr.add_log("warn", st["message"])
                traceback.print_exc()
            finally:
                st["running"] = False
                st["progress"] = 1.0

        if self.export_threaded:
            threading.Thread(target=job, name="export", daemon=True).start()
        else:
            job()

    def start_generation(self) -> None:
        g = self.gen_settings
        self.gen_hover = None
        self.worker.submit("generate", prompt=self.prompt or " ", **g)

    def key(self, event) -> bool:
        pg = self.pg
        k, mod = event.key, getattr(event, "mod", 0)
        shift = bool(mod & pg.KMOD_SHIFT)
        ctrl = bool(mod & (pg.KMOD_CTRL | pg.KMOD_META))
        if self.capture_select:
            if k == pg.K_ESCAPE:
                self.capture_select = False
            return True
        if k == pg.K_F11:
            self.toggle_full_window()
            return True
        if k == pg.K_TAB:
            self.cycle_view(-1 if shift else 1)
            return True
        if pg.K_F1 <= k <= pg.K_F1 + len(VIEW_KEYS) - 1:
            self.set_view(VIEW_KEYS[k - pg.K_F1])
            return True
        if self.view == "generate":
            return self.generate_key(k, shift, ctrl)
        if k in (pg.K_ESCAPE, pg.K_q):
            return False
        if k == pg.K_SPACE:
            self.worker.submit("pause", on=not self.worker.paused)
            self.worker.paused = not self.worker.paused
        elif k == pg.K_n:
            self.worker.submit("step")
        elif k == pg.K_e:
            self.worker.submit("eval")
        elif k == pg.K_s:
            self.worker.submit("save", path=self.save_path)
        elif k == pg.K_l:
            self.worker.submit("load", path=self.save_path)
        elif k == pg.K_r:
            self.worker.submit("snippet")
            self.pinned = None
            self.lens_offset = 0
        elif k == pg.K_c:
            if shift:
                self.capture_select = True
            else:
                self.screenshot()
        elif k == pg.K_v:
            self.cycle_view(1)
        elif pg.K_1 <= k <= pg.K_9:
            self.set_view(VIEW_KEYS[k - pg.K_1])
        elif k == pg.K_0 and len(VIEW_KEYS) > 9:
            self.set_view(VIEW_KEYS[9])
        elif k == pg.K_MINUS and len(VIEW_KEYS) > 10:
            self.set_view(VIEW_KEYS[10])
        elif k == pg.K_o and self.view in self.cams:
            self.cams[self.view].auto = not self.cams[self.view].auto
        elif k == pg.K_x:
            self.log_x = not self.log_x
        elif k == pg.K_a:
            self.avg_heads = not self.avg_heads
        elif k in (pg.K_RETURN, pg.K_KP_ENTER) and self.view == "export":
            self.start_export()
        elif k == pg.K_k and self.view == "export":
            self.do_action("capture_now")
        elif k in (pg.K_LEFT, pg.K_RIGHT) and self.view == "export":
            self.do_action(f"preview:{1 if k == pg.K_RIGHT else -1}")
        elif k in (pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN):
            self.arrow(k)
        return True

    def arrow(self, k) -> None:
        pg = self.pg
        cfg = self.tr.model_cfg
        if self.view in ("attention", "heads", "network", "arch"):
            if k == pg.K_LEFT:
                self.sel_head = (self.sel_head - 1) % cfg.n_head
            elif k == pg.K_RIGHT:
                self.sel_head = (self.sel_head + 1) % cfg.n_head
            elif k == pg.K_UP:
                self.sel_layer = (self.sel_layer - 1) % cfg.n_layer
            else:
                self.sel_layer = (self.sel_layer + 1) % cfg.n_layer
            self.avg_heads = False
        elif self.view == "lens":
            snap = self.tr.snapshot
            n = len(snap["nll"]) if snap is not None else 0
            if k == pg.K_LEFT:
                self.lens_offset = max(0, self.lens_offset - 4)
            elif k == pg.K_RIGHT:
                self.lens_offset = max(0, min(n - 4, self.lens_offset + 4))
        elif self.view == "predict":
            snap = self.tr.snapshot
            if snap is not None:
                n = len(snap["nll"])
                cur = self.pinned if self.pinned is not None else -1
                self.pinned = max(0, min(n - 1, cur + (1 if k in (pg.K_RIGHT, pg.K_DOWN) else -1)))

    def generate_key(self, k, shift: bool, ctrl: bool) -> bool:
        pg = self.pg
        if k == pg.K_ESCAPE:
            self.set_view(self.prev_view if self.prev_view != "generate" else "training")
        elif k in (pg.K_RETURN, pg.K_KP_ENTER):
            if shift:
                self.prompt += "\n"
            else:
                self.start_generation()
        elif k == pg.K_BACKSPACE:
            self.prompt = "" if ctrl else self.prompt[:-1]
        elif k == pg.K_UP:
            self.adjust("top_k" if shift else "temperature", 1)
        elif k == pg.K_DOWN:
            self.adjust("top_k" if shift else "temperature", -1)
        elif k == pg.K_RIGHT:
            self.adjust("top_p", 1)
        elif k == pg.K_LEFT:
            self.adjust("top_p", -1)
        elif k == pg.K_PAGEUP:
            self.adjust("max_new", 1)
        elif k == pg.K_PAGEDOWN:
            self.adjust("max_new", -1)
        elif ctrl and k == pg.K_s:
            self.worker.submit("save", path=self.save_path)
        elif ctrl and k == pg.K_c:
            self.screenshot()
        elif ctrl and k == pg.K_q:
            return False
        return True

    # -- main loop ----------------------------------------------------------------- #
    def run(self, max_frames: int | None = None) -> None:
        pg = self.pg
        frames = 0
        self.worker.start()
        try:
            running = True
            while running and (max_frames is None or frames < max_frames):
                for event in pg.event.get():
                    if not self.handle_event(event):
                        running = False
                if not running:
                    break
                if not self.worker.threaded:
                    self.worker.service()
                self.service_capture()
                self.draw()
                try:
                    self._present()
                    pg.display.flip()
                except pg.error as e:
                    self.tr.add_log("warn", f"display stopped: {e}")
                    break
                self.clock.tick(self.fps)
                frames += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.worker.stop()
            self.save_on_exit()
            pg.quit()

    def save_on_exit(self) -> str | None:
        """Keep this session's training: save if the model changed since the last save or
        load (window close, ESC, Q and Ctrl+C all end here)."""
        w, tr = self.worker, self.tr
        if not self.autosave or tr.step == w.saved_step:
            return None
        try:
            tr.save(self.save_path)
        except (OSError, RuntimeError) as e:
            print(f"error: could not save the model to {self.save_path}: {e}", file=sys.stderr)
            return None
        w.saved_step = tr.step
        print(f"model saved to {self.save_path} (step {tr.step:,}); continue with "
              f"--resume {self.save_path}", flush=True)
        return self.save_path

    # =========================================================================== #
    # Frame
    # =========================================================================== #
    def draw(self) -> None:
        now = time.perf_counter()
        dt, self._frame_t = min(0.1, now - self._frame_t), now
        self.canvas.fill(BG)
        self.sections = {}
        self.hits_prev = self.hits
        self.hits = []
        self.buttons = []
        self.tip = None
        self.draw_header()
        view = {"training": self.draw_training, "data": self.draw_data, "arch": self.draw_arch,
                "predict": self.draw_predict,
                "attention": self.draw_attention, "heads": self.draw_heads, "lens": self.draw_lens,
                "network": lambda r: self.draw_network(r, dt), "embed": lambda r: self.draw_embed(r, dt),
                "generate": self.draw_generate, "export": self.draw_export}[self.view]
        self.sections["view"] = (VIEW_NAMES[self.view], self.MAIN)
        view(self.MAIN)
        self.draw_side()
        self.draw_footer()
        self.draw_tooltip()

    def draw_header(self) -> None:
        pg = self.pg
        w = self.spaced("TRANSFORMER LAB", (14, 7), INK, self.f_title, spacing=5)
        x = 14 + w + 16
        pg.draw.line(self.canvas, INK_DIM, (x, 21), (x + 28, 21))
        d = self.tr.data
        name = d.name.upper() if d.name in ("wikitext-103", "wikitext-2") else "TEXT CORPUS"
        sub = self.fit_text(f"GPT · {name} · BPE {d.vocab_size // 1024}K", 260, self.f_cap)
        self.spaced(sub, (x + 40, 15), INK_DIM)
        pad, gap, h, y = 11, 5, 26, 8
        labels = [(key, label.upper()) for key, label in VIEWS]
        widths = [self._spaced_img(label, INK_DIM, self.f_cap, 2.0).get_width() + 2 * pad for _, label in labels]
        x = self.BASE_W - 10 - sum(widths) - gap * (len(widths) - 1)
        self.tab_rects = {}
        for (key, label), tw in zip(labels, widths):
            r = (x, y, tw, h)
            active, hover = key == self.view, self._inside(self.mouse, r)
            if active:
                pg.draw.rect(self.canvas, PANEL_HI, r)
            pg.draw.rect(self.canvas, INK if active else (INK_DIM if hover else INK_FAINT), r, 1)
            self.spaced(label, (x + tw // 2, y + 8), INK if active or hover else INK_DIM, align="center")
            self.tab_rects[key] = r
            x += tw + gap

    def draw_footer(self) -> None:
        y = self.FOOT_Y
        x = 12
        if self.view == "generate":
            rows = ((("TYPE", "prompt"), ("ENTER", "generate"), ("SHIFT+ENTER", "newline"), ("⌫", "delete"),
                     ("CTRL+⌫", "clear"), ("↑↓", "temperature"), ("SHIFT+↑↓", "top-k"), ("←→", "top-p"),
                     ("PGUP/PGDN", "length")),
                    (("TAB", "next view"), ("ESC", "back"), ("CTRL+S", "save"), ("CTRL+C", "shot"),
                     ("F11", "full window"), ("CTRL+Q", "quit")))
        else:
            extra = {"training": (("X", "log/linear tokens"),),
                     "predict": (("CLICK", "pin token"), ("←→", "move pin"), ("RIGHT-CLICK", "unpin")),
                     "attention": (("CLICK", "head"), ("←→", "head"), ("↑↓", "layer"), ("A", "mean of heads")),
                     "heads": (("CLICK", "head"), ("←→↑↓", "select head")),
                     "lens": (("←→ / WHEEL", "scroll positions"),),
                     "network": (("DRAG", "orbit"), ("WHEEL", "zoom"), ("O", "auto-orbit"), ("↑↓", "layer")),
                     "embed": (("CLICK", "token"), ("DRAG", "orbit"), ("WHEEL", "zoom"), ("O", "auto-orbit")),
                     "arch": (("CLICK", "block"), ("↑↓", "layer"), ("←→", "head"), ("R", "new passage")),
                     "export": (("CLICK", "buttons"), ("ENTER", "export appendix"), ("←→", "preview"),
                                ("K", "capture a frame now")),
                     "data": ()}.get(self.view, ())
            rows = ((("SPACE", "pause"), ("N", "step"), ("E", "evaluate"), ("R", "new passage"), ("S", "save"),
                     ("L", "load"), ("1-0 / TAB", "views"), ("C", "shot"), ("SHIFT+C", "panel shot"),
                     ("F11", "full window"), ("ESC", "quit")),
                    extra)
        for i, row in enumerate(rows):
            xx = x
            for key, label in row:
                xx += self.key_chip(xx, y + i * 18, key, label)

    def draw_tooltip(self) -> None:
        if not self.tip:
            return
        pg = self.pg
        lines = self.tip
        wbox = 22 + max(self.tw(t, self.small) for t, _ in lines)
        hbox = 10 + 15 * len(lines)
        mx, my = self.mouse
        bx = mx + 18 if mx + 18 + wbox < self.BASE_W else mx - wbox - 18
        by = min(my + 18, self.BASE_H - hbox - 4)
        pg.draw.rect(self.canvas, BG, (bx, by, wbox, hbox))
        pg.draw.rect(self.canvas, INK_DIM, (bx, by, wbox, hbox), 1)
        for i, (t, col) in enumerate(lines):
            self.text(t, (bx + 11, by + 6 + 15 * i), col, self.small)

    def waiting(self, rect, msg: str = "waiting for the first measurement…") -> None:
        x, y, w, h = rect
        self.text(msg, (x + w // 2, y + h // 2 - 6), FAINT, self.small, align="center")

    # =========================================================================== #
    # Right column: status, live graphs, log
    # =========================================================================== #
    def draw_side(self) -> None:
        x, y, w, h = self.SIDE
        self.draw_stats((x, y, w, 300))
        self.draw_graphs((x, y + 308, w, 330))
        self.draw_log((x, y + 646, w, h - 646))

    def draw_stats(self, rect) -> None:
        pg, tr, wk = self.pg, self.tr, self.worker
        self.panel(rect, "Run", key="stats")
        x, y, w, h = rect
        status = wk.status
        col = {"TRAINING": INK, "EVALUATING": VAL_C, "ERROR": BAD, "FINISHED": GOOD_C, "STOPPED": BAD,
               "GENERATING": LR_C}.get(status, INK_DIM)
        self.pill(x + w - 12, y + 10, status, col, filled=status in ("TRAINING", "EVALUATING", "GENERATING"))
        cfg, mc = tr.cfg, tr.model_cfg
        hist = tr.hist
        loss_s = f"{np.mean(hist['loss'][-50:]):.4f}" if hist["loss"] else "-"
        e = tr.last_eval
        val = f"{e['loss']:.4f} @ {e['step']:,}" if e else "-"
        ppl = f"{e['ppl']:,.1f} · {e['bpb']:.3f} bpb" if e else "-"
        wppl = f"{e['word_ppl']:,.1f}" if e and math.isfinite(e.get("word_ppl", float('nan'))) else "-"
        best = min((r["loss"] for r in hist["eval"]), default=float("nan"))
        step_rate = float(np.median(hist["tok_s"][-30:])) if hist["tok_s"] else 0.0
        wall = wk.wall_rate()
        flops = step_rate * tr.flops_per_token
        sec_per_step = tr.tokens_per_step / step_rate if step_rate else float("nan")
        eta = (cfg.max_steps - tr.step) * sec_per_step
        if wk.saved_at:
            saved = f"step {wk.saved_step:,}, {fmt_duration(time.time() - wk.saved_at)} ago"
        else:
            saved = "unsaved" if tr.step != wk.saved_step else f"in sync (step {wk.saved_step:,})"
        left = [
            ("Step", f"{tr.step:,} / {cfg.max_steps:,}"),
            ("Tokens", f"{fmt_tokens(tr.tokens)} · {tr.epoch:.3f} ep"),
            ("Train loss", loss_s),
            ("Val loss", val),
            ("Val ppl", ppl),
            ("Word ppl", wppl),
            ("Best val", f"{best:.4f}" if math.isfinite(best) else "-"),
            ("LR", f"{tr.lr():.2e} {cfg.schedule}"),
            ("Grad norm", f"{hist['grad_norm'][-1]:.3f}" if hist["grad_norm"] else "-"),
            ("Batch", f"{cfg.batch_size}×{cfg.grad_accum}×{mc.block_size}"),
            ("Weight decay", f"{cfg.weight_decay:g}, clip {cfg.grad_clip:g}"),
        ]
        right = [
            ("Speed", f"{step_rate:,.0f} tok/s" if step_rate else "-"),
            ("Wall", f"{wall:,.0f} tok/s" if wall else "-"),
            ("Compute", f"{flops / 1e12:.2f} TFLOP/s" if flops else "-"),
            ("Trained", fmt_duration(tr.train_seconds)),
            ("To finish", fmt_duration(eta) if math.isfinite(eta) else "-"),
            ("Params", f"{tr.model.num_params() / 1e6:.2f}M"),
            ("Non-embed", f"{tr.model.num_params(True) / 1e6:.2f}M"),
            ("Shape", f"{mc.n_layer}L {mc.n_head}H {mc.d_model}d"),
            ("Blocks", f"{mc.pos}, {mc.norm}, {mc.mlp}"),
            ("Device", f"{tr.device.type} · {'bf16' if tr.amp else 'fp32'}"),
            ("Saved", saved),
        ]
        mid = x + w // 2
        pg.draw.line(self.canvas, INK_FAINT, (mid - 4, y + 32), (mid - 4, y + h - 10))
        for cx, cr, rows in ((x + 12, mid - 14, left), (mid + 6, x + w - 12, right)):
            for i, (k, v) in enumerate(rows):
                yy = y + 32 + i * 23
                lw = self.text(k.upper(), (cx, yy + 2), DIM, self.small)
                self.text(self.fit_text(v, max(0, cr - cx - lw - 8), self.font), (cr, yy), TEXT, self.font,
                          align="right")

    def mini(self, rect, data, title: str, color, smooth_k: int = 1, ref=None, fmt: str = "{:.3g}", fixed=None,
             xs=None) -> None:
        pg = self.pg
        x0, y0, w, h = rect
        pg.draw.rect(self.canvas, CARD, rect)
        pg.draw.rect(self.canvas, INK_FAINT, rect, 1)
        arr = np.asarray([v for v in data if v is not None and math.isfinite(v)], dtype=np.float64)
        if len(arr) < 2:
            self.text(self.fit_text(title.upper(), w - 12, self.small), (x0 + 6, y0 + 4), DIM, self.small)
            self.text("waiting…", (x0 + 6, y0 + h // 2), FAINT, self.small)
            return
        vw = self.text(fmt.format(arr[-1]), (x0 + w - 6, y0 + 4), color, self.small, align="right")
        self.text(self.fit_text(title.upper(), w - 20 - vw, self.small), (x0 + 6, y0 + 4), DIM, self.small)
        sm = smooth(arr, smooth_k)
        lo, hi = fixed if fixed else (float(min(arr.min(), ref if ref is not None else arr.min())),
                                      float(max(arr.max(), ref if ref is not None else arr.max())))
        if hi - lo < 1e-12:
            lo, hi = lo - 1, hi + 1
        hi_s, lo_s = fmt.format(hi), fmt.format(lo)
        gutter = 11 + max(self.tw(hi_s, self.small), self.tw(lo_s, self.small))
        px0, pw = x0 + gutter, max(8, w - gutter - 6)
        top, bot = y0 + 22, y0 + h - 6
        xi = np.arange(len(arr), dtype=np.float64)
        xd, yd = decimate(xi, arr, pw)
        _, sd = decimate(xi, sm, pw)
        pg.draw.line(self.canvas, INK_FAINT, (px0 - 3, top), (px0 - 3, bot))

        def ys(v):
            return np.clip(bot - (v - lo) / (hi - lo) * (bot - top), top, bot)

        if ref is not None and lo <= ref <= hi:
            self.dashed(INK_FAINT, (px0, int(ys(ref))), (px0 + pw, int(ys(ref))))
        sx = px0 + (xd - xd[0]) / max(xd[-1] - xd[0], 1e-9) * (pw - 1)
        raw = list(zip(sx.astype(int), ys(yd).astype(int)))
        line = list(zip(sx.astype(int), ys(sd).astype(int)))
        if len(line) > 1:
            pg.draw.polygon(self.canvas, mix_rgb(CARD, color, 0.1), [(line[0][0], bot)] + line + [(line[-1][0], bot)])
            if smooth_k > 1:
                pg.draw.lines(self.canvas, mix_rgb(CARD, color, 0.45), False, raw, 1)
            pg.draw.aalines(self.canvas, color, False, line)
        pg.draw.rect(self.canvas, color, (line[-1][0] - 1, line[-1][1] - 1, 3, 3))
        self.text(hi_s, (x0 + 5, top - 3), FAINT, self.small)
        self.text(lo_s, (x0 + 5, bot - 11), FAINT, self.small)

    def draw_graphs(self, rect) -> None:
        tr = self.tr
        self.panel(rect, "Live graphs", key="graphs")
        x, y, w, h = rect
        gw = (w - 20 - 8) // 2
        gh = (h - 34 - 2 * 6) // 3
        ev, pr = tr.hist["eval"], tr.hist["probe"]
        bl = tr.data.meta.get("baselines", {})
        specs = [
            ([r["loss"] for r in ev], "val loss", VAL_C, 1, bl.get("bigram_val"), "{:.3f}"),
            ([r["bpb"] for r in ev], "val bits / byte", VAL_C, 1, None, "{:.3f}"),
            (tr.hist["grad_norm"], "grad norm", INK, 25, tr.cfg.grad_clip or None, "{:.3g}"),
            (tr.hist["tok_s"], "tokens / s", INK, 20, None, "{:,.0f}"),
            ([r["induction_max"] for r in pr], "max induction score", LR_C, 1, None, "{:.2f}"),
            ([r["loss_repeat"] for r in pr], "loss on repeated tokens", LR_C, 1, None, "{:.2f}"),
        ]
        for i, (data, title, col, sk, ref, fmt) in enumerate(specs):
            gx = x + 10 + (i % 2) * (gw + 8)
            gy = y + 32 + (i // 2) * (gh + 6)
            self.mini((gx, gy, gw, gh), list(data), title, col, sk, ref, fmt)

    def draw_log(self, rect) -> None:
        self.panel(rect, "Log", key="log")
        x, y, w, h = rect
        line_h = 15
        for i, (kind, msg) in enumerate(reversed(list(self.tr.log))):
            yy = y + 29 + i * line_h
            if yy > y + h - line_h:
                break
            col = LOG_COLORS.get(kind, TEXT)
            self.pg.draw.rect(self.canvas, col, (x + 12, yy + 5, 3, 3))
            self.text(self.fit_text(msg, w - 36, self.small), (x + 22, yy), col, self.small)

    # =========================================================================== #
    # Charts
    # =========================================================================== #
    def axes(self, rect, xr, yr, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}", nx: int = 5, ny: int = 5,
             log_x: bool = False, log_y: bool = False, right_fmt=None, xlabel: str = "", ylabel: str = ""):
        """Grid, ticks and labels; returns world -> screen mappers."""
        pg = self.pg
        x, y, w, h = rect
        x0, x1 = xr
        y0, y1 = yr
        tx = (lambda v: np.log10(np.maximum(v, 1e-300))) if log_x else (lambda v: np.asarray(v, dtype=np.float64))
        ty = (lambda v: np.log10(np.maximum(v, 1e-300))) if log_y else (lambda v: np.asarray(v, dtype=np.float64))
        a0, a1 = float(tx(x0)), float(tx(x1))
        b0, b1 = float(ty(y0)), float(ty(y1))
        if a1 - a0 < 1e-12:
            a1 = a0 + 1
        if b1 - b0 < 1e-12:
            b1 = b0 + 1

        def X(v):
            return x + (tx(v) - a0) / (a1 - a0) * w

        def Y(v):
            return y + h - (ty(v) - b0) / (b1 - b0) * h

        pg.draw.rect(self.canvas, CARD, rect)
        for i in range(ny + 1):
            bv = b0 + (b1 - b0) * i / ny
            yy = int(y + h - (bv - b0) / (b1 - b0) * h)
            pg.draw.line(self.canvas, GRID_C, (x, yy), (x + w, yy))
            v = 10 ** bv if log_y else bv
            self.text(yfmt(v), (x - 6, yy - 6), FAINT, self.small, align="right")
            if right_fmt is not None:
                self.text(right_fmt(v), (x + w + 6, yy - 6), FAINT, self.small)
        for i in range(nx + 1):
            av = a0 + (a1 - a0) * i / nx
            xx = int(x + (av - a0) / (a1 - a0) * w)
            pg.draw.line(self.canvas, GRID_C, (xx, y), (xx, y + h))
            v = 10 ** av if log_x else av
            self.text(xfmt(v), (xx, y + h + 5), FAINT, self.small, align="center" if 0 < i < nx else
                      ("left" if i == 0 else "right"))
        pg.draw.rect(self.canvas, INK_FAINT, (x, y, w + 1, h + 1), 1)
        if xlabel:
            self.text(xlabel, (x + w, y + h + 18), DIM, self.small, align="right")
        if ylabel:
            self.text(ylabel, (x, y - 16), DIM, self.small)
        return X, Y

    def polyline(self, X, Y, xs, ys, color, width: int = 1, clip=None) -> list:
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 2:
            return []
        px, py = X(xs[ok]), Y(ys[ok])
        if clip is not None:
            py = np.clip(py, clip[1], clip[1] + clip[3])
        pts = list(zip(px.tolist(), py.tolist()))
        if width <= 1:
            self.pg.draw.aalines(self.canvas, color, False, pts)
        else:
            self.pg.draw.lines(self.canvas, color, False, pts, width)
        return pts

    # =========================================================================== #
    # View: Training
    # =========================================================================== #
    def draw_training(self, rect) -> None:
        x, y, w, h = rect
        self.draw_loss((x, y, 800, 430))
        self.draw_lr((x + 808, y, w - 808, 211))
        self.draw_gradnorm((x + 808, y + 219, w - 808, 211))
        y2 = y + 438
        h2 = h - 438
        self.draw_pos_loss((x, y2, 404, h2))
        self.draw_freq_loss((x + 412, y2, 404, h2))
        self.draw_health((x + 824, y2, w - 824, h2))

    def draw_loss(self, rect) -> None:
        tr, pg = self.tr, self.pg
        self.panel(rect, "Loss", "nats per token · right axis = perplexity · X toggles log tokens", key="loss")
        x, y, w, h = rect
        plot = (x + 58, y + 64, w - 58 - 64, h - 64 - 40)
        steps = np.asarray(tr.hist["tokens"], dtype=np.float64)
        loss = np.asarray(tr.hist["loss"], dtype=np.float64)
        ev = tr.hist["eval"]
        bl = tr.data.meta.get("baselines", {})
        refs = [(bl.get("uniform"), "uniform"), (bl.get("unigram_val"), "unigram"), (bl.get("bigram_val"), "bigram")]
        refs = [(v, n) for v, n in refs if v is not None]
        if len(steps) < 2 and not ev:
            self.axes(plot, (0, 1), (0, 10))
            self.waiting(plot)
            return
        ex = np.asarray([r["tokens"] for r in ev], dtype=np.float64)
        ey = np.asarray([r["loss"] for r in ev], dtype=np.float64)
        sm = smooth(loss, 50)
        xmax = max(float(steps[-1]) if len(steps) else 1.0, float(ex.max()) if len(ex) else 1.0, 1.0)
        vals = np.concatenate([sm[len(sm) // 20:] if len(sm) else sm, ey])
        lo = float(np.nanmin(vals)) if len(vals) else 0.0
        hi_ref = max(v for v, _ in refs) if refs else float(np.nanmax(vals))
        hi = max(hi_ref, float(np.nanmax(ey)) if len(ey) else hi_ref) + 0.2
        lo = max(0.0, min(lo, min(v for v, _ in refs) if refs else lo) - 0.3)
        log_x = self.log_x and xmax > 10 * tr.tokens_per_step
        xr = (tr.tokens_per_step if log_x else 0.0, xmax * (1.0 if log_x else 1.02))
        X, Y = self.axes(plot, xr, (lo, hi), xfmt=fmt_tokens, yfmt=lambda v: f"{v:.1f}",
                         right_fmt=lambda v: fmt_tokens(math.exp(v)) if v < 30 else "", log_x=log_x,
                         xlabel="tokens seen", ny=6)
        clip = plot
        self.canvas.set_clip(pg.Rect(plot[0], plot[1], plot[2] + 1, plot[3] + 1))
        try:
            ep_tokens = tr.sampler.n_windows * tr.model_cfg.block_size
            k = 1
            while k * ep_tokens < xr[1]:
                xx = float(X(k * ep_tokens))
                self.dashed(FAINT, (xx, plot[1]), (xx, plot[1] + plot[3]), 3, 5)
                self.text(f"epoch {k}", (xx + 4, plot[1] + 4), FAINT, self.small)
                k += 1
            for v, _ in refs:
                yy = float(Y(v))
                self.dashed(INK_DIM, (plot[0], yy), (plot[0] + plot[2], yy))
            if len(steps) > 1:
                xd, yd = decimate(steps, loss, plot[2])
                self.polyline(X, Y, xd, yd, mix_rgb(CARD, INK, 0.45), 1, clip)
                xd, yd = decimate(steps, sm, plot[2])
                self.polyline(X, Y, xd, yd, INK, 2, clip)
            if len(ex):
                if log_x:
                    keep = ex > 0
                    exx, eyy = ex[keep], ey[keep]
                else:
                    exx, eyy = ex, ey
                self.polyline(X, Y, exx, eyy, VAL_C, 2, clip)
                for a, b in zip(X(exx), Y(eyy)):
                    pg.draw.circle(self.canvas, VAL_C, (int(a), int(b)), 3)
        finally:
            self.canvas.set_clip(None)
        for v, name in refs:
            yy = float(Y(v))
            if plot[1] <= yy <= plot[1] + plot[3]:
                self.text(f"{name} {v:.2f}", (plot[0] + plot[2] - 6, int(yy) - 14), INK_DIM, self.small, align="right")
        lx = x + 14
        ly = y + 34
        lx += self.swatch(lx, ly, INK, "train (50-step mean)") + 10
        lx += self.swatch(lx, ly, mix_rgb(CARD, INK, 0.55), "train (per step)") + 10
        lx += self.swatch(lx, ly, VAL_C, "validation (all tokens)") + 10
        lx += self.swatch(lx, ly, INK_DIM, "reference models", "dash") + 10
        if self._inside(self.mouse, plot) and len(steps) > 1:
            mx = self.mouse[0]
            xs = X(steps)
            i = int(np.argmin(np.abs(xs - mx)))
            pg.draw.line(self.canvas, INK_FAINT, (int(xs[i]), plot[1]), (int(xs[i]), plot[1] + plot[3]))
            lines = [(f"step {int(tr.hist['step'][i]):,} · {fmt_tokens(steps[i])} tokens", TEXT),
                     (f"train loss {loss[i]:.4f} (mean {sm[i]:.4f}, ppl {math.exp(min(sm[i], 30)):,.1f})", DIM),
                     (f"learning rate {tr.hist['lr'][i]:.2e}", DIM)]
            if len(ex):
                j = int(np.argmin(np.abs(ex - steps[i])))
                r = ev[j]
                lines.append((f"val @ step {r['step']:,}: {r['loss']:.4f}, ppl {r['ppl']:,.1f}, {r['bpb']:.3f} bpb",
                              VAL_C))
            self.tip = lines

    def draw_lr(self, rect) -> None:
        tr, pg = self.tr, self.pg
        cfg = tr.cfg
        self.panel(rect, "Learning rate", f"warm-up {cfg.warmup_steps:,} · {cfg.schedule} to "
                                          f"{cfg.min_lr_ratio:g}× · {cfg.max_steps:,} steps", key="lr")
        x, y, w, h = rect
        plot = (x + 62, y + 36, w - 62 - 16, h - 36 - 26)
        n = cfg.max_steps
        s = np.unique(np.linspace(0, n - 1, 300).astype(int))
        lr = np.array([lr_at(cfg, int(v)) for v in s])
        X, Y = self.axes(plot, (0, n), (0, cfg.lr * 1.08), xfmt=lambda v: fmt_tokens(v), yfmt=lambda v: f"{v:.0e}",
                         ny=3, nx=4)
        self.polyline(X, Y, s, lr, mix_rgb(CARD, LR_C, 0.5), 1)
        done = s <= tr.step
        if done.sum() > 1:
            self.polyline(X, Y, s[done], lr[done], LR_C, 2)
        cx, cy = float(X(tr.step)), float(Y(tr.lr()))
        pg.draw.circle(self.canvas, LR_C, (int(cx), int(cy)), 4)
        self.text(f"{tr.lr():.2e}", (int(cx) + 8 if cx < plot[0] + plot[2] - 70 else int(cx) - 70, int(cy) - 16),
                  LR_C, self.small)

    def draw_gradnorm(self, rect) -> None:
        tr, pg = self.tr, self.pg
        self.panel(rect, "Gradient norm", f"global L2 before clipping at {tr.cfg.grad_clip:g}", key="grad")
        x, y, w, h = rect
        plot = (x + 62, y + 36, w - 62 - 16, h - 36 - 26)
        g = np.asarray(tr.hist["grad_norm"], dtype=np.float64)
        if len(g) < 2:
            self.axes(plot, (0, 1), (0, 1), ny=3, nx=4)
            self.waiting(plot)
            return
        st = np.asarray(tr.hist["step"], dtype=np.float64)
        hi = max(float(np.percentile(g, 99)) * 1.1, (tr.cfg.grad_clip or 0) * 1.2, 1e-3)
        X, Y = self.axes(plot, (0, float(st[-1])), (0, hi), xfmt=fmt_tokens, yfmt=lambda v: f"{v:.2g}", ny=3, nx=4)
        self.canvas.set_clip(pg.Rect(plot[0], plot[1], plot[2] + 1, plot[3] + 1))
        try:
            xd, yd = decimate(st, g, plot[2])
            self.polyline(X, Y, xd, yd, mix_rgb(CARD, INK, 0.45), 1)
            xd, yd = decimate(st, smooth(g, 25), plot[2])
            self.polyline(X, Y, xd, yd, INK, 2)
            if tr.cfg.grad_clip:
                yy = float(Y(tr.cfg.grad_clip))
                self.dashed(BAD, (plot[0], yy), (plot[0] + plot[2], yy))
        finally:
            self.canvas.set_clip(None)
        clipped = float((g[-200:] > tr.cfg.grad_clip).mean()) if tr.cfg.grad_clip else 0.0
        self.text(f"clipped in {100 * clipped:.0f}% of the last {min(200, len(g))} steps", (plot[0] + 6, plot[1] + 4),
                  DIM, self.small)

    def draw_pos_loss(self, rect) -> None:
        tr, pg = self.tr, self.pg
        self.panel(rect, "Loss by position", "validation · tokens of context", key="pos")
        x, y, w, h = rect
        plot = (x + 48, y + 64, w - 48 - 16, h - 64 - 64)
        ev = [r for r in tr.hist["eval"] if r.get("pos_loss")]
        if not ev:
            self.axes(plot, (0, 1), (0, 1))
            self.waiting(plot)
            return
        T = len(ev[-1]["pos_loss"])
        curves = [np.asarray(r["pos_loss"], dtype=np.float64) for r in ev]
        first = max(0, len(curves) // 2) if len(curves) > 6 else min(1, len(curves) - 1)
        curves = curves[first:]
        last = curves[-1]
        lo = float(np.nanmin(last)) - 0.15
        hi = float(np.nanmax(smooth(np.nan_to_num(curves[0], nan=lo), 5))) + 0.15
        X, Y = self.axes(plot, (1, T), (lo, hi), xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.2f}", log_x=True,
                         nx=3, ny=4, xlabel="position (log)")
        self.canvas.set_clip(pg.Rect(plot[0], plot[1], plot[2] + 1, plot[3] + 1))
        try:
            pick = np.unique(np.linspace(0, len(curves) - 1, min(6, len(curves))).astype(int))
            for j, i in enumerate(pick):
                c = curves[i]
                col = VAL_C if i == len(curves) - 1 else cmap_color(0.25 + 0.6 * j / max(1, len(pick) - 1), "viridis")
                pos = np.arange(1, len(c) + 1)
                xd, yd = decimate(pos, smooth(c, 5), 200)
                self.polyline(X, Y, xd, yd, col, 2 if i == len(curves) - 1 else 1)
            for p in (50, 500):
                if p <= T:
                    xx = float(X(p))
                    self.dashed(FAINT, (xx, plot[1]), (xx, plot[1] + plot[3]), 3, 4)
        finally:
            self.canvas.set_clip(None)
        icl = ev[-1].get("icl", float("nan"))
        yy = y + h - 44
        self.text(f"in-context score (loss@500 − loss@50): {icl:+.3f}" if math.isfinite(icl) else
                  "in-context score: needs 512-token context", (x + 12, yy), VAL_C, self.small)
        self.text(self.fit_text(f"curves: {len(pick)} evaluations, latest in cyan", w - 24, self.small),
                  (x + 12, yy + 16), FAINT, self.small)
        lx = x + 14
        self.swatch(lx, y + 34, VAL_C, f"step {ev[-1]['step']:,}")
        if len(ev) > 1:
            self.swatch(lx + 110, y + 34, cmap_color(0.25, "viridis"), "earlier")

    def draw_freq_loss(self, rect) -> None:
        tr = self.tr
        self.panel(rect, "Loss by token frequency", "validation · bucket = training frequency rank", key="freq")
        x, y, w, h = rect
        plot = (x + 48, y + 64, w - 48 - 16, h - 64 - 64)
        ev = [r for r in tr.hist["eval"] if r.get("freq_loss")]
        if len(ev) < 1:
            self.axes(plot, (0, 1), (0, 1))
            self.waiting(plot)
            return
        xs = np.asarray([r["tokens"] for r in ev], dtype=np.float64)
        fl = np.asarray([r["freq_loss"] for r in ev], dtype=np.float64)
        lo, hi = float(np.nanmin(fl)) - 0.2, float(np.nanmax(fl[-min(len(fl), 8):])) + 0.4
        hi = max(hi, float(np.nanmax(fl[-1])) + 0.4)
        X, Y = self.axes(plot, (0, max(float(xs[-1]), 1.0)), (max(0.0, lo), hi), xfmt=fmt_tokens,
                         yfmt=lambda v: f"{v:.1f}", nx=3, ny=4, xlabel="tokens seen")
        self.canvas.set_clip(self.pg.Rect(plot[0], plot[1], plot[2] + 1, plot[3] + 1))
        try:
            for k in range(fl.shape[1]):
                col = cmap_color(0.15 + 0.8 * k / (fl.shape[1] - 1), "viridis")
                self.polyline(X, Y, xs, fl[:, k], col, 2)
        finally:
            self.canvas.set_clip(None)
        lx, ly = x + 14, y + 34
        for k in range(fl.shape[1]):
            col = cmap_color(0.15 + 0.8 * k / (fl.shape[1] - 1), "viridis")
            lx += self.swatch(lx, ly, col, BUCKET_NAMES[k]) + 4
        last = fl[-1]
        self.text(self.fit_text("now: " + " · ".join(f"{v:.2f}" for v in last), w - 24, self.small),
                  (x + 12, y + h - 44), DIM, self.small)
        self.text(self.fit_text("frequent tokens are learned first", w - 24, self.small), (x + 12, y + h - 28),
                  FAINT, self.small)

    def draw_health(self, rect) -> None:
        tr, pg = self.tr, self.pg
        self.panel(rect, "Layer health", "log10 |ΔW| / |W| per update", key="health")
        x, y, w, h = rect
        hist = tr.hist["health"][-90:]
        if not hist:
            self.waiting((x, y, w, h))
            return
        names = list(hist[-1][1].keys())
        ratio = np.array([[np.log10(max(rec[1].get(n, (0, 0, 1e-12))[2], 1e-12)) for rec in hist] for n in names])
        top, left = y + 40, x + 64
        cols_w = 128
        hm_w = w - (left - x) - cols_w - 12
        row_h = max(6, min(16, (h - 40 - 52) // len(names)))
        img = heat_rgb(ratio, -5.0, -1.0, "inferno")
        self.canvas.blit(rgb_surface(pg, img, (hm_w, row_h * len(names))), (left, top))
        pg.draw.rect(self.canvas, INK_FAINT, (left - 1, top - 1, hm_w + 2, row_h * len(names) + 2), 1)
        cx = left + hm_w + 10
        self.text("|W| rms", (cx, top - 14), FAINT, self.small)
        self.text("log ratio", (cx + 62, top - 14), FAINT, self.small)
        cur = hist[-1][1]
        for i, n in enumerate(names):
            yy = top + i * row_h
            if row_h >= 10 or i % 2 == 0:
                lbl = n.replace("attn_out", "o").replace("mlp_in", "up").replace("mlp_out", "down")
                self.text(lbl, (left - 6, yy + (row_h - 11) // 2), DIM if "." in n else INK, self.small, align="right")
                wv, gv, rv = cur[n]
                self.text(f"{wv:.3f}", (cx, yy + (row_h - 11) // 2), DIM, self.small)
                lr_ = math.log10(max(rv, 1e-12))
                col = GOOD_C if -3.6 <= lr_ <= -2.4 else (LR_C if -4.5 <= lr_ <= -1.5 else BAD)
                self.text(f"{lr_:+.2f}", (cx + 62, yy + (row_h - 11) // 2), col, self.small)
        yb = top + row_h * len(names) + 10
        self.colorbar(left, yb + 2, 90, "inferno", "1e-5", "1e-1")
        self.text(self.fit_text(f"last {len(hist)} measurements · ~1e-3 is the usual healthy range", w - 24, self.small),
                  (x + 12, yb + 18), FAINT, self.small)
        if self._inside(self.mouse, (left, top, hm_w, row_h * len(names))):
            i = (self.mouse[1] - top) // row_h
            j = int((self.mouse[0] - left) / hm_w * len(hist))
            if 0 <= i < len(names) and 0 <= j < len(hist):
                step, rec = hist[j]
                wv, gv, rv = rec[names[i]]
                self.tip = [(f"{names[i]} at step {step:,}", TEXT), (f"weight rms {wv:.4f}", DIM),
                            (f"gradient rms {gv:.3g}", DIM), (f"update / weight {rv:.2e}", DIM)]

    # =========================================================================== #
    # View: Data
    # =========================================================================== #
    def draw_data(self, rect) -> None:
        x, y, w, h = rect
        self.draw_corpus((x, y, 600, 318))
        self.draw_zipf((x + 608, y, w - 608, 318))
        self.draw_batch((x, y + 326, 760, h - 326))
        self.draw_tokenizer((x + 768, y + 326, w - 768, 316))
        self.draw_epoch((x + 768, y + 650, w - 768, h - 650))

    def draw_corpus(self, rect) -> None:
        tr, pg = self.tr, self.pg
        d = tr.data
        meta = d.meta
        src = meta.get("source", {})
        if "repo" in src:
            hint = f"{src['repo']} · {src['config']} · sha256 verified"
        elif "file" in src:
            hint = os.path.basename(src["file"])
        else:
            hint = "in-memory corpus"
        self.panel(rect, "Corpus", hint, key="corpus")
        x, y, w, h = rect
        cols = (x + 14, x + 190, x + 300, x + 370, x + 480, x + 586)      # right edges after the first
        heads = ("SPLIT", "ROWS", "WORDS", "MB", "TOKENS", "BYTES/TOK")
        yy = y + 36
        for i, (cx, hd) in enumerate(zip(cols, heads)):
            self.text(hd, (cx, yy), DIM, self.small, align="left" if i == 0 else "right")
        for s in ("train", "validation", "test"):
            st = meta.get("splits", {}).get(s)
            if not st:
                continue
            yy += 18
            vals = (s, f"{st['rows']:,}", f"{st['words']:,}", f"{st['bytes'] / 1e6:.1f}", f"{st['tokens']:,}",
                    f"{st['bytes'] / max(st['tokens'], 1):.2f}")
            for i, (cx, v) in enumerate(zip(cols, vals)):
                self.text(v, (cx, yy), TEXT if i else INK, self.font, align="left" if i == 0 else "right")
        yy += 30
        pg.draw.line(self.canvas, INK_FAINT, (x + 12, yy - 8), (x + w - 12, yy - 8))
        self.text("VALIDATION CROSS-ENTROPY", (x + 14, yy), DIM, self.small)
        self.text("NATS/TOK", (x + 330, yy), DIM, self.small, align="right")
        self.text("PPL", (x + 420, yy), DIM, self.small, align="right")
        self.text("BITS/BYTE", (x + 530, yy), DIM, self.small, align="right")
        bl = meta.get("baselines", {})
        vs = meta.get("splits", {}).get("validation", {})
        bpt = vs.get("tokens", 1) / max(vs.get("bytes", 1), 1) / math.log(2)
        rows = [("uniform over the vocabulary", bl.get("uniform"), DIM),
                ("unigram (token frequencies)", bl.get("unigram_val"), DIM),
                (f"interpolated bigram (λ={bl.get('bigram_lambda', 0):.2f})", bl.get("bigram_val"), DIM)]
        e = tr.last_eval
        best = min((r["loss"] for r in tr.hist["eval"]), default=None)
        if best is not None:
            rows.append((f"this transformer, best (step {min(tr.hist['eval'], key=lambda r: r['loss'])['step']:,})",
                         best, VAL_C))
        if e is not None:
            rows.append((f"this transformer, latest (step {e['step']:,})", e["loss"], VAL_C))
        for label, v, col in rows:
            if v is None:
                continue
            yy += 18
            self.text(self.fit_text(label, 250, self.font), (x + 14, yy), col if col != DIM else TEXT, self.font)
            self.text(f"{v:.3f}", (x + 330, yy), col if col != DIM else TEXT, self.font, align="right")
            self.text(f"{math.exp(min(v, 30)):,.1f}", (x + 420, yy), col if col != DIM else TEXT, self.font,
                      align="right")
            self.text(f"{v * bpt:.3f}", (x + 530, yy), col if col != DIM else TEXT, self.font, align="right")
        tok = d.tokenizer
        self.text(self.fit_text(f"tokenizer: byte-level BPE, {tok.vocab_size:,} tokens, GPT-4 split pattern, "
                                f"trained on the training split only · {tok.fingerprint()[:12]}", w - 28, self.small),
                  (x + 14, y + h - 22), FAINT, self.small)

    def draw_zipf(self, rect) -> None:
        tr, pg = self.tr, self.pg
        self.panel(rect, "Zipf's law", "training-token frequency against rank (log-log)", key="zipf")
        x, y, w, h = rect
        counts = tr.data.unigram
        plot = (x + 60, y + 40, w - 60 - 20, h - 40 - 46)
        if counts is None or counts.sum() == 0:
            self.waiting(plot, "no frequency table")
            return
        key = ("zipf", id(counts))
        if key not in self._cache:
            c = np.sort(counts[counts > 0])[::-1].astype(np.float64)
            r = np.arange(1, len(c) + 1, dtype=np.float64)
            hi_r = min(len(c), 5000)
            sel = slice(9, hi_r)
            if hi_r > 20:
                slope, icpt = np.polyfit(np.log10(r[sel]), np.log10(c[sel]), 1)
            else:
                slope, icpt = float("nan"), 0.0
            order = np.argsort(-counts, kind="stable")
            self._cache[key] = (r, c, slope, icpt, order)
        r, c, slope, icpt, order = self._cache[key]
        X, Y = self.axes(plot, (1, r[-1]), (max(c[-1], 1), c[0] * 1.5), log_x=True, log_y=True,
                         xfmt=lambda v: fmt_tokens(v), yfmt=lambda v: fmt_tokens(v), nx=4, ny=4, xlabel="rank")
        idx = np.unique(np.geomspace(1, len(r), 400).astype(int)) - 1
        self.polyline(X, Y, r[idx], c[idx], VAL_C, 2)
        if math.isfinite(slope):
            fx = np.array([10.0, min(len(r), 5000)])
            self.dashed(LR_C, (float(X(fx[0])), float(Y(10 ** (icpt + slope * np.log10(fx[0]))))),
                        (float(X(fx[1])), float(Y(10 ** (icpt + slope * np.log10(fx[1]))))))
        for k in (1, 10, 100, 1000, 10000):
            if k > len(c):
                break
            t = int(order[k - 1])
            px, py = float(X(k)), float(Y(c[k - 1]))
            pg.draw.circle(self.canvas, INK, (int(px), int(py)), 3)
            lbl = f"#{k:,} {self.tok_label(t)}"
            if px + 4 + self.tw(lbl, self.small) > plot[0] + plot[2] - 4:
                self.text(lbl, (int(px) - 6, int(py) - 18), INK, self.small, align="right")
            else:
                self.text(lbl, (int(px) + 4, int(py) + 6), INK, self.small)
        total = counts.sum()
        cover = np.cumsum(c) / total
        n50 = int(np.searchsorted(cover, 0.5) + 1)
        n90 = int(np.searchsorted(cover, 0.9) + 1)
        self.text(self.fit_text(f"fitted exponent {-slope:.2f} on ranks 10–5,000 · {n50:,} tokens cover 50% of the "
                                f"text, {n90:,} cover 90%", w - 24, self.small), (x + 14, y + h - 22), DIM, self.small)

    def draw_batch(self, rect) -> None:
        tr = self.tr
        b = tr.last_batch
        hint = (f"step {b['step']:,} · sequence 1 of {tr.cfg.batch_size} · coloured by its training loss"
                if b else "the batch the model just trained on")
        self.panel(rect, "Current training batch", hint, key="batch")
        x, y, w, h = rect
        if b is None:
            self.waiting(rect)
            return
        nll = b["nll"]
        cols = [tuple(int(v) for v in c) for c in heat_rgb(nll, 0.0, 12.0, "inferno")]
        ids = list(b["x"][1:]) + [b["y"][-1]]          # token t+1 coloured by the loss of predicting it
        self.flow(ids, (x + 14, y + 36, w - 28, h - 36 - 40), cols, kind="batchtok")
        self.colorbar(x + 14, y + h - 24, 140, "inferno", "0 nats", "12")
        self.text(f"mean {float(nll.mean()):.3f} nats over {len(nll)} tokens", (x + w - 14, y + h - 26), DIM,
                  self.small, align="right")
        i = self.hovered("batchtok")
        if i is not None and 0 <= i < len(nll):
            self.tip = [(f"{self.tok_label(ids[i])!s}", TEXT), (f"loss {nll[i]:.3f} nats · p = {math.exp(-nll[i]):.3%}", DIM)]

    def draw_tokenizer(self, rect) -> None:
        tr, pg = self.tr, self.pg
        tok = tr.data.tokenizer
        self.panel(rect, "Tokenizer", "learned merges and token lengths", key="tokenizer")
        x, y, w, h = rect
        self.text("FIRST MERGES (ID ORDER = MERGE ORDER)", (x + 14, y + 34), DIM, self.small)
        xx, yy = x + 14, y + 52
        for i in range(256, min(tok.vocab_size, 256 + 60)):
            lbl = tok.display(i)
            cw = self.tw(lbl, self.small) + 10
            if xx + cw > x + w - 14:
                xx, yy = x + 14, yy + 20
                if yy > y + 52 + 20 * 2:
                    break
            pg.draw.rect(self.canvas, PANEL_HI, (xx, yy, cw, 16))
            pg.draw.rect(self.canvas, INK_FAINT, (xx, yy, cw, 16), 1)
            self.text(lbl, (xx + 5, yy + 2), INK, self.small)
            xx += cw + 4
        key = ("longest", tok.fingerprint())
        if key not in self._cache:
            lens = np.array(tok.token_lengths())
            longest = [int(i) for i in np.argsort(-lens, kind="stable")[:40]
                       if tok.display(int(i)).isprintable()][:8]
            hist = np.bincount(np.minimum(lens, 16), minlength=17)[1:]
            counts = tr.data.unigram
            used = np.bincount(np.minimum(lens, 16), weights=counts, minlength=17)[1:] if counts is not None else hist
            self._cache[key] = (longest, hist, used)
        longest, hist, used = self._cache[key]
        yy = y + 52 + 3 * 20 + 6
        self.text("LONGEST TOKENS", (x + 14, yy), DIM, self.small)
        self.text(self.fit_text("  ".join(tok.display(i) for i in longest), w - 28, self.small), (x + 14, yy + 16),
                  INK, self.small)
        yy += 42
        self.text("BYTES PER TOKEN", (x + 14, yy), DIM, self.small)
        lx = x + w - 14
        lx -= self.swatch(lx - 110, yy - 1, VAL_C, "in the text", "box")
        self.swatch(lx - 140, yy - 1, INK_DIM, "in the vocab", "box")
        bx, by, bw, bh = x + 14, yy + 18, w - 28, y + h - (yy + 18) - 24
        n = len(hist)
        cw = bw / n
        hv, uv = hist / max(hist.max(), 1), used / max(used.max(), 1)
        for i in range(n):
            x0 = bx + i * cw
            a = int(bh * hv[i])
            b = int(bh * uv[i])
            pg.draw.rect(self.canvas, INK_DIM, (int(x0 + 2), by + bh - a, max(1, int(cw / 2 - 2)), a))
            pg.draw.rect(self.canvas, VAL_C, (int(x0 + cw / 2), by + bh - b, max(1, int(cw / 2 - 2)), b))
            if i % 3 == 0 or i == n - 1:
                self.text(f"{i + 1}{'+' if i == n - 1 else ''}", (int(x0 + cw / 2), by + bh + 3), FAINT, self.small,
                          align="center")

    def draw_epoch(self, rect) -> None:
        tr, pg = self.tr, self.pg
        self.panel(rect, "Data order", "epochs of shuffled, non-overlapping windows", key="epoch")
        x, y, w, h = rect
        ep = tr.epoch
        frac = ep - math.floor(ep)
        bx, by, bw = x + 14, y + 44, w - 28
        pg.draw.rect(self.canvas, INK_FAINT, (bx, by, bw, 10), 1)
        pg.draw.rect(self.canvas, VAL_C, (bx + 1, by + 1, int((bw - 2) * frac), 8))
        self.text(f"epoch {math.floor(ep) + 1}: {100 * frac:.1f}% of {tr.sampler.n_windows:,} windows",
                  (bx, by + 16), TEXT, self.font)
        n = len(tr.data["train"])
        rows = [
            ("training tokens", f"{n:,}"),
            ("window", f"{tr.model_cfg.block_size} tokens + 1 target"),
            ("tokens per update", f"{tr.tokens_per_step:,}"),
            ("updates per epoch", f"{tr.sampler.n_windows / (tr.cfg.batch_size * tr.cfg.grad_accum):,.0f}"),
            ("order", f"seeded shuffle per epoch (seed {tr.cfg.seed})"),
            ("resume", "batch = f(seed, step): exact"),
        ]
        for i, (k, v) in enumerate(rows):
            yy = by + 44 + i * 19
            self.text(k.upper(), (bx, yy + 2), DIM, self.small)
            self.text(self.fit_text(v, bw - 150, self.font), (x + w - 14, yy), TEXT, self.font, align="right")

    # =========================================================================== #
    # View: Architecture
    # =========================================================================== #
    ARCH_N = 24                                    # tokens shown inside the block

    @staticmethod
    def pool_cols(a: np.ndarray, width: int) -> np.ndarray:
        """(rows, cols) -> (rows, <= width): each output column is the entry of largest
        magnitude in its group of input columns, so signs and extremes survive."""
        a = np.asarray(a, dtype=np.float32)
        r, c = a.shape
        g = max(1, math.ceil(c / width))
        pad = (-c) % g
        if pad:
            a = np.concatenate([a, np.zeros((r, pad), np.float32)], axis=1)
        b = a.reshape(r, -1, g)
        idx = np.abs(b).argmax(axis=2)
        return np.take_along_axis(b, idx[..., None], axis=2)[..., 0]

    def act_tile(self, data: np.ndarray, rect, name: str, shape: str, cmap: str = "rdbu", tip: str = "",
                 lo: float | None = None, hi: float | None = None, highlight: bool = False,
                 compact: bool = False, label_w: int | None = None) -> None:
        """A heat map of real activations (rows = tokens) with its name above and its shape
        and RMS below; diverging colours on a symmetric scale of the tile's own maximum."""
        pg = self.pg
        x, y, w, h = rect
        a = np.asarray(data, dtype=np.float32)
        if cmap == "rdbu":
            m = float(np.abs(a).max()) or 1.0
            img = heat_rgb(self.pool_cols(a, w) if a.shape[1] > w else a, -m, m, "rdbu")
            scale_s = f"±{m:.2g}"
        else:
            lo_, hi_ = (0.0 if lo is None else lo), (float(a.max()) if hi is None else hi)
            img = heat_rgb(np.sqrt(np.clip(a, 0, None)), math.sqrt(max(lo_, 0)), math.sqrt(max(hi_, 1e-9)), cmap)
            scale_s = f"0..{hi_:.2g}"
        self.canvas.blit(rgb_surface(pg, img, (w, h)), (x, y))
        pg.draw.rect(self.canvas, INK if highlight else INK_FAINT, (x - 1, y - 1, w + 2, h + 2), 1)
        self.spaced(self.fit_text(name.upper(), w + 30, self.f_cap), (x, y - 15), INK)
        rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
        lw = w + 20 if label_w is None else label_w
        if compact:
            self.text(self.fit_text(f"{shape} · rms {rms:.3g}", lw, self.small), (x, y + h + 4), DIM, self.small)
        else:
            self.text(self.fit_text(shape, lw, self.small), (x, y + h + 4), DIM, self.small)
            rms_s = f"rms {rms:.3g} · {scale_s}" if self.tw(f"rms {rms:.3g} · {scale_s}", self.small) <= lw else f"{rms:.3g}"
            self.text(self.fit_text(rms_s, lw, self.small), (x, y + h + 17), FAINT, self.small)
        self.hit(rect, "tile", (name, a, tip))

    def draw_arrow(self, p0, p1, color=INK_DIM, label: str = "") -> None:
        pg = self.pg
        pg.draw.aaline(self.canvas, color, p0, p1)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for s in (-1, 1):
            q = (p1[0] - 7 * math.cos(ang + s * 0.45), p1[1] - 7 * math.sin(ang + s * 0.45))
            pg.draw.aaline(self.canvas, color, p1, q)
        if label:
            self.text(label, ((p0[0] + p1[0]) // 2, min(p0[1], p1[1]) - 14), FAINT, self.small, align="center")

    def op(self, center, sym: str) -> None:
        pg = self.pg
        pg.draw.circle(self.canvas, BG, center, 11)
        pg.draw.circle(self.canvas, INK, center, 11, 1)
        self.text(sym, (center[0], center[1] - 8), INK, self.f_head, align="center")

    def draw_arch(self, rect) -> None:
        x, y, w, h = rect
        self.draw_model_strip((x, y, 262, h))
        self.draw_block_inside((x + 270, y, w - 270, h))

    def draw_model_strip(self, rect) -> None:
        pg, tr = self.pg, self.tr
        mc = tr.model_cfg
        m = tr.model
        snap = tr.snapshot
        self.panel(rect, "The model", f"{m.num_params() / 1e6:.2f}M parameters", key="model_strip")
        x, y, w, h = rect
        bx, bw = x + 14, w - 28
        boxes = []
        emb_p = mc.vocab_size * mc.d_model + (mc.block_size * mc.d_model if mc.pos == "learned" else 0)
        per_block = sum(p.numel() for p in m.blocks[0].parameters())
        toks = snap["tokens"][:6] if snap is not None else []
        boxes.append(("tokens", " ".join(self.tok_label(int(t)) for t in toks) or "token ids", "", None))
        boxes.append(("token embedding", f"{mc.vocab_size:,} × {mc.d_model}" +
                      (" + positions" if mc.pos == "learned" else " · RoPE in attention"), f"{emb_p / 1e6:.2f}M", None))
        for l in range(mc.n_layer):
            writes = ""
            if snap is not None:
                writes = (f"attn {float(snap['attn_out_rms'][l].mean()):.2f} · "
                          f"mlp {float(snap['mlp_out_rms'][l].mean()):.2f}")
            boxes.append((f"block {l + 1}", writes or f"{mc.n_head} heads · MLP {mc.hidden:,}",
                          f"{per_block / 1e6:.2f}M", l))
        boxes.append((f"final {'RMSNorm' if mc.norm == 'rms' else 'LayerNorm'}", f"{mc.d_model}",
                      f"{sum(p.numel() for p in m.ln_f.parameters()):,}", None))
        boxes.append(("unembedding", f"{mc.d_model} × {mc.vocab_size:,}" + (" (tied)" if mc.tie_embeddings else ""),
                      "shared" if mc.tie_embeddings else f"{mc.d_model * mc.vocab_size / 1e6:.2f}M", None))
        nxt = ""
        if snap is not None:
            i = min(len(snap["nll"]), self.ARCH_N) - 1
            nxt = f"after '{self.tok_label(int(snap['tokens'][i]))}': {self.tok_label(int(snap['top_id'][i, 0]))} " \
                  f"{float(snap['top_p'][i, 0]):.0%}"
        boxes.append(("softmax", nxt or "p(next token)", "", None))
        top = y + 38
        gap = 14
        bh = int((h - 38 - 40 - gap * (len(boxes) - 1)) / len(boxes))
        for i, (name, sub_, params, layer) in enumerate(boxes):
            by = top + i * (bh + gap)
            r = (bx, by, bw, bh)
            sel = layer is not None and layer == self.sel_layer
            hot = layer is not None and self._inside(self.mouse, r)
            pg.draw.rect(self.canvas, PANEL_HI if (sel or hot) else CARD, r)
            pg.draw.rect(self.canvas, INK if sel else (INK_DIM if hot or layer is not None else INK_FAINT), r, 1)
            self.spaced(name.upper(), (bx + 10, by + 6), INK if sel or layer is None else DIM)
            if params:
                self.text(params, (bx + bw - 8, by + 5), FAINT, self.small, align="right")
            self.text(self.fit_text(sub_, bw - 20, self.small), (bx + 10, by + bh - 17), DIM, self.small)
            if layer is not None:
                self.hit(r, "block", layer)
            if i:
                self.draw_arrow((bx + bw // 2, by - gap + 1), (bx + bw // 2, by - 2))
            if sel:
                pg.draw.line(self.canvas, INK, (bx + bw, by + bh // 2), (x + w, by + bh // 2))
        self.text("click a block to open it", (x + 14, y + h - 24), FAINT, self.small)

    def draw_block_inside(self, rect) -> None:
        pg, tr = self.pg, self.tr
        mc = tr.model_cfg
        snap = tr.snapshot
        l = self.sel_layer
        norm = "RMSNorm" if mc.norm == "rms" else "LayerNorm"
        self.panel(rect, f"Inside block {l + 1} of {mc.n_layer}",
                   f"pre-norm residual block · real activations, first {self.ARCH_N} tokens of the passage (rows)",
                   key="block_inside")
        x, y, w, h = rect
        if snap is None or "block" not in snap:
            self.waiting(rect)
            return
        b = snap["block"]
        n = min(self.ARCH_N, b["ln1"].shape[1])
        H, hd, d, hid = mc.n_head, mc.head_dim, mc.d_model, mc.hidden
        hsel = self.sel_head
        th, dw, qw, hw, aw = 150, 88, 46, 150, 128  # tile height (n token rows); widths: d, head, hidden, attention
        r_in = snap["resid"][l][:n].astype(np.float32)
        r_mid = r_in + b["attn_out"][l][:n].astype(np.float32)
        r_out = snap["resid"][l + 1][:n].astype(np.float32)
        xs = x + 22
        p_in = xs + dw // 2

        # lane A: attention sub-layer ------------------------------------------------
        self.spaced("ATTENTION SUB-LAYER", (x + 14, y + 36), INK_DIM)
        self.text(f"x ← x + W_o · concat_h softmax(Q_h K_h^T / √{hd} + causal mask) V_h,   Q, K, V = W_qkv · "
                  f"{norm}(x)" + ("  (Q, K rotated by RoPE)" if mc.pos == "rope" else ""), (x + 14, y + 52), FAINT,
                  self.small)
        ya = y + 122
        bus_a = ya - 32
        mid = ya + th // 2
        self.act_tile(r_in, (xs, ya, dw, th), "x (residual)", f"{n} × {d}", tip="residual stream entering the block")
        x1 = xs + dw + 30
        self.draw_arrow((xs + dw + 4, mid), (x1 - 4, mid))
        self.act_tile(b["ln1"][l][:n], (x1, ya, dw, th), norm, f"{n} × {d}", tip=f"{norm}(x): unit RMS per token × gain")
        x2 = x1 + dw + 40
        for j, key in enumerate(("q", "k", "v")):
            self.act_tile(b[key][l, hsel][:n], (x2 + j * (qw + 8), ya, qw, th), key.upper(), f"{n}×{hd}", label_w=qw + 6,
                          tip=f"{key.upper()} of head {hsel + 1}" + (" after RoPE" if key != "v" and mc.pos == "rope" else ""))
        self.draw_arrow((x1 + dw + 4, mid), (x2 - 4, mid), label="W_qkv")
        self.text(f"head {hsel + 1} of {H}  (←/→)", (x2, ya + th + 34), LR_C, self.small)
        x3 = x2 + 3 * (qw + 8) + 24
        att = snap["attn"][l][:, :n, :n].astype(np.float32)
        self.act_tile(att[hsel], (x3, ya + th - aw, aw, aw), f"softmax(QK^T/√{hd})", f"{n} × {n} · head {hsel + 1}",
                      cmap="magma", lo=0.0, hi=1.0, tip=f"attention probabilities of head {hsel + 1} (rows sum to 1)")
        self.draw_arrow((x3 - 22, mid), (x3 - 4, mid))
        mini = max(10, (th - 5 * (H - 1)) // H)
        for hh in range(H):
            r = (x3 + aw + 10, ya + hh * (mini + 5), mini, mini)
            self.canvas.blit(rgb_surface(pg, heat_rgb(np.sqrt(att[hh]), 0, 1, "magma"), (mini, mini)), r[:2])
            pg.draw.rect(self.canvas, INK if hh == hsel else INK_FAINT, (r[0] - 1, r[1] - 1, mini + 2, mini + 2), 1)
            self.hit(r, "head", (l, hh))
        x4 = x3 + aw + 10 + mini + 32
        self.act_tile(b["heads"][l][:n], (x4, ya, dw, th), "heads · V", f"{n} × {H}·{hd}",
                      tip="every head's attention-weighted values, concatenated")
        self.draw_arrow((x4 - 30, mid), (x4 - 4, mid), label="·V")
        x5 = x4 + dw + 38
        self.act_tile(b["attn_out"][l][:n], (x5, ya, dw, th), "attention out", f"{n} × {d}",
                      tip="W_o · heads: what the attention sub-layer adds to the residual")
        self.draw_arrow((x4 + dw + 4, mid), (x5 - 4, mid), label="W_o")
        px = x5 + dw + 30
        self.op((px, mid), "+")
        self.draw_arrow((x5 + dw + 4, mid), (px - 12, mid))
        pg.draw.line(self.canvas, INK_DIM, (p_in, ya - 18), (p_in, bus_a))
        pg.draw.line(self.canvas, INK_DIM, (p_in, bus_a), (px, bus_a))
        self.draw_arrow((px, bus_a), (px, mid - 12))
        self.text("residual connection: x passes through unchanged", (px - 8, bus_a - 14), FAINT, self.small,
                  align="right")
        # x after attention feeds the MLP lane through a gutter below lane A
        gutter = ya + th + 58
        yb = gutter + 84
        pg.draw.line(self.canvas, INK_DIM, (px, mid + 12), (px, gutter))
        pg.draw.line(self.canvas, INK_DIM, (px, gutter), (x + 10, gutter))
        pg.draw.line(self.canvas, INK_DIM, (x + 10, gutter), (x + 10, yb + th // 2))
        self.draw_arrow((x + 10, yb + th // 2), (xs - 4, yb + th // 2))

        # lane B: MLP sub-layer -------------------------------------------------------
        self.spaced("MLP SUB-LAYER", (x + 24, gutter + 12), INK_DIM)
        formula = (f"x ← x + W_down (SiLU(W_gate h) ⊙ W_up h),   h = {norm}(x)" if mc.mlp == "swiglu"
                   else f"x ← x + W_2 GELU(W_1 h),   h = {norm}(x)")
        self.text(formula, (x + 24, gutter + 28), FAINT, self.small)
        bus_b = yb - 32
        mb = yb + th // 2
        self.act_tile(r_mid, (xs, yb, dw, th), "x (residual)", f"{n} × {d}", tip="residual after the attention sub-layer")
        y1 = x1
        self.draw_arrow((xs + dw + 4, mb), (y1 - 4, mb))
        self.act_tile(b["ln2"][l][:n], (y1, yb, dw, th), norm, f"{n} × {d}", tip=f"{norm} of the residual")
        y2 = y1 + dw + 50
        if mc.mlp == "swiglu":
            hh_ = (th - 36) // 2
            up_y = yb + hh_ + 36
            self.act_tile(b["mlp_pre"][l][:n], (y2, yb, hw, hh_), "gate", f"{n}×{hid:,}", tip="W_gate h", compact=True)
            self.act_tile(b["mlp_up"][l][:n], (y2, up_y, hw, hh_), "up", f"{n}×{hid:,}", tip="W_up h", compact=True)
            self.draw_arrow((y1 + dw + 4, mb), (y2 - 4, yb + hh_ // 2), label="W_gate")
            self.draw_arrow((y1 + dw + 4, mb), (y2 - 4, up_y + hh_ // 2))
            self.text("W_up", (y1 + dw + 14, up_y + hh_ // 2 - 4), FAINT, self.small)
            ox = y2 + hw + 42
            self.op((ox, mb), "×")
            self.draw_arrow((y2 + hw + 4, yb + hh_ // 2), (ox - 9, mb - 7))
            self.draw_arrow((y2 + hw + 4, up_y + hh_ // 2), (ox - 9, mb + 7))
            self.text("SiLU(gate)", (ox, mb - 44), FAINT, self.small, align="center")
            self.text("× up", (ox, mb + 18), FAINT, self.small, align="center")
            y3 = ox + 42
            self.draw_arrow((ox + 12, mb), (y3 - 4, mb))
        else:
            self.act_tile(b["mlp_pre"][l][:n], (y2, yb, hw, th), "W_1 h", f"{n} × {hid:,}", tip="pre-activation")
            self.draw_arrow((y1 + dw + 4, mb), (y2 - 4, mb), label="W_1")
            y3 = y2 + hw + 46
            self.draw_arrow((y2 + hw + 4, mb), (y3 - 4, mb), label="GELU")
        act = b["mlp_act"][l][:n].astype(np.float32)
        busy = 100 * float((np.abs(act) > 0.1 * max(float(np.abs(act).max()), 1e-12)).mean())
        self.act_tile(act, (y3, yb, hw, th), "hidden units", f"{n} × {hid:,}",
                      tip=f"{hid:,} hidden units; {busy:.0f}% above 10% of the largest")
        y4 = y3 + hw + 46
        self.act_tile(b["mlp_out"][l][:n], (y4, yb, dw, th), "MLP out", f"{n} × {d}",
                      tip="what the MLP adds to the residual")
        self.draw_arrow((y3 + hw + 4, mb), (y4 - 4, mb), label="W_down" if mc.mlp == "swiglu" else "W_2")
        qx = y4 + dw + 34
        self.op((qx, mb), "+")
        self.draw_arrow((y4 + dw + 4, mb), (qx - 12, mb))
        pg.draw.line(self.canvas, INK_DIM, (p_in, yb - 18), (p_in, bus_b))
        pg.draw.line(self.canvas, INK_DIM, (p_in, bus_b), (qx, bus_b))
        self.draw_arrow((qx, bus_b), (qx, mb - 12))
        zx = qx + 30
        if zx + dw <= x + w - 8:
            self.act_tile(r_out, (zx, yb, dw, th), f"to block {l + 2}" if l + 1 < mc.n_layer else "to final norm",
                          f"{n} × {d}", tip="residual stream leaving the block", label_w=x + w - 8 - zx)
            self.draw_arrow((qx + 12, mb), (zx - 4, mb))

        # rows, weights of this block, legend -------------------------------------------
        toks = snap["tokens"][:n]
        ry = yb + th + 44
        self.text("ROWS = TOKENS", (x + 14, ry), DIM, self.small)
        self.text(self.fit_text("  ".join(self.tok_label(int(t)) for t in toks), w - 130, self.small), (x + 116, ry),
                  INK, self.small)
        self.draw_block_weights((x + 14, ry + 24, w - 28, y + h - 52 - (ry + 24)), l)
        ly = y + h - 40
        wcb = self.colorbar(x + 14, ly + 2, 120, "rdbu", "−", "+ activation (tile's own scale)")
        self.colorbar(x + 14 + wcb + 30, ly + 2, 90, "magma", "0", "1 attention (sqrt)")
        self.text("wide tiles: each column shows the largest-magnitude channel of its group", (x + 14, ly + 18),
                  FAINT, self.small)
        self.text(f"weights after {tr.step:,} updates · R = new passage", (x + w - 14, ly + 18), FAINT, self.small,
                  align="right")
        tv = self.hovered("tile")
        if tv is not None:
            name, a, tip = tv
            lines = [(name, TEXT)]
            if tip:
                lines.append((tip, DIM))
            lines.append((f"min {float(a.min()):+.3f} · max {float(a.max()):+.3f}", DIM))
            for r_, kind_, payload in self.hits:
                if kind_ == "tile" and payload is tv and self._inside(self.mouse, r_):
                    row = int((self.mouse[1] - r_[1]) / r_[3] * a.shape[0])
                    if 0 <= row < len(toks):
                        v = a[row]
                        lines.append((f"row {row}: {self.tok_label(int(toks[row]))} · rms "
                                      f"{float(np.sqrt((v ** 2).mean())):.3f}", VAL_C))
            self.tip = lines

    def draw_block_weights(self, rect, l: int) -> None:
        """The block's weight matrices: shape, parameters and the live RMS / update size."""
        tr = self.tr
        mc = tr.model_cfg
        x, y, w, h = rect
        if h < 60:
            return
        blk = tr.model.blocks[l]
        health = tr.hist["health"][-1][1] if tr.hist["health"] else {}
        rows = [("W_qkv", f"{mc.d_model} → 3 × {mc.d_model}", blk.attn.qkv.weight.numel(), f"{l + 1}.qkv",
                 "queries, keys and values of all heads"),
                ("W_o", f"{mc.d_model} → {mc.d_model}", blk.attn.proj.weight.numel(), f"{l + 1}.attn_out",
                 "mixes the heads back into the residual"),
                ("W_gate | W_up" if mc.mlp == "swiglu" else "W_1",
                 f"{mc.d_model} → {'2 × ' if mc.mlp == 'swiglu' else ''}{mc.hidden:,}", blk.mlp.up.weight.numel(),
                 f"{l + 1}.mlp_in", "expands to the hidden units"),
                ("W_down" if mc.mlp == "swiglu" else "W_2", f"{mc.hidden:,} → {mc.d_model}",
                 blk.mlp.down.weight.numel(), f"{l + 1}.mlp_out", "projects back to the residual"),
                ("2 × norm gain", f"{mc.d_model} each", sum(p.numel() for p in blk.ln1.parameters()) +
                 sum(p.numel() for p in blk.ln2.parameters()), None, "per-channel scale (no weight decay)")]
        self.spaced(f"WEIGHTS OF BLOCK {l + 1}", (x, y), INK_DIM)
        cols = (x, x + 150, x + 330, x + 430, x + 540, x + 660)
        for cx, hd_ in zip(cols, ("MATRIX", "SHAPE", "PARAMS", "RMS", "UPDATE/W", "ROLE")):
            self.text(hd_, (cx, y + 18), DIM, self.small)
        total = 0
        for i, (name, shape, n_p, key, role) in enumerate(rows):
            yy = y + 36 + i * 18
            if yy > y + h - 14:
                break
            total += n_p
            self.text(name, (cols[0], yy), INK, self.font)
            self.text(shape, (cols[1], yy), TEXT, self.font)
            self.text(f"{n_p:,}", (cols[2], yy), TEXT, self.font)
            if key and key in health:
                wv, gv, rv = health[key]
                self.text(f"{wv:.4f}", (cols[3], yy), TEXT, self.font)
                lr_ = math.log10(max(rv, 1e-12))
                col = GOOD_C if -3.6 <= lr_ <= -2.4 else (LR_C if -4.5 <= lr_ <= -1.5 else BAD)
                self.text(f"{rv:.1e}", (cols[4], yy), col, self.font)
            else:
                self.text("-", (cols[3], yy), FAINT, self.font)
            self.text(self.fit_text(role, x + w - cols[5], self.small), (cols[5], yy + 1), DIM, self.small)
        yy = y + 36 + len(rows) * 18
        if yy <= y + h - 14:
            self.text(f"{total:,} parameters in this block · {total * mc.n_layer:,} in all {mc.n_layer} blocks",
                      (cols[0], yy + 4), DIM, self.small)

    # =========================================================================== #
    # View: Predict
    # =========================================================================== #
    def draw_predict(self, rect) -> None:
        x, y, w, h = rect
        snap = self.tr.snapshot
        self.draw_passage((x, y, 820, 560), snap)
        self.draw_prediction((x + 828, y, w - 828, 560), snap)
        y2, h2 = y + 568, h - 568
        self.draw_calibration((x, y2, 404, h2))
        self.draw_accuracy((x + 412, y2, 404, h2))
        self.draw_rank_hist((x + 824, y2, w - 824, h2), snap)

    def focus_token(self, snap) -> int | None:
        i = self.hovered("tok")
        if i is None:
            i = self.pinned
        if i is None or snap is None or i >= len(snap["nll"]):
            return None
        return i

    def draw_passage(self, rect, snap) -> None:
        hint = (f"validation offset {self.tr.snippet_start:,} · step {snap['step']:,} · colour = surprisal −log p"
                if snap else "validation passage")
        self.panel(rect, "Next-token predictions", hint, key="passage")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        nll = snap["nll"]
        ids = list(snap["tokens"][1:])
        cols = [tuple(int(v) for v in c) for c in heat_rgb(nll, 0.0, 10.0, "inferno")]
        outline = {self.pinned} if self.pinned is not None else set()
        self.text("context →", (x + 14, y + 36), FAINT, self.small)
        self.text(self.fit_text(self.tok_text(int(snap["tokens"][0])), 200, self.f_flow), (x + 90, y + 33), DIM,
                  self.f_flow)
        self.flow(ids, (x + 14, y + 56, w - 28, h - 56 - 46), cols, line_h=22, outline=outline)
        yb = y + h - 30
        self.colorbar(x + 14, yb + 2, 160, "inferno", "0 nats (certain)", "10+ (surprised)")
        m = float(nll.mean())
        self.text(f"passage mean {m:.3f} nats · ppl {math.exp(m):,.1f}", (x + w - 14, yb), DIM, self.small,
                  align="right")

    def draw_prediction(self, rect, snap) -> None:
        pg, tr = self.pg, self.tr
        i = self.focus_token(snap)
        self.panel(rect, "Prediction", "hover or click a token" if i is None else
                   f"position {i + 1} of {len(snap['nll'])}", key="prediction")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        if i is None:
            i = int(np.argmax(snap["nll"]))
            self.text("showing the most surprising token", (x + 14, y + 34), FAINT, self.small)
        toks = snap["tokens"]
        ctx = tr.data.tokenizer.decode(toks[max(0, i - 24):i + 1]).replace("\n", "⏎ ")
        self.text("CONTEXT", (x + 14, y + 52), DIM, self.small)
        lines = self.wrap("…" + ctx, w - 28, self.small, 3)
        for k, ln in enumerate(lines):
            self.text(ln, (x + 14, y + 68 + 14 * k), TEXT, self.small)
        yy = y + 68 + 14 * len(lines) + 12
        true = int(toks[i + 1])
        p_true = math.exp(-float(snap["nll"][i]))
        self.text("ACTUAL NEXT", (x + 14, yy), DIM, self.small)
        self.text(self.tok_label(true), (x + 14, yy + 14), INK, self.f_head)
        self.text(f"p = {p_true:.2%} · rank {int(snap['rank'][i]) + 1:,} · {float(snap['nll'][i]):.2f} nats",
                  (x + w - 14, yy + 18), VAL_C, self.small, align="right")
        yy += 44
        self.text(f"MODEL'S TOP {snap['top_id'].shape[1]} · entropy {float(snap['entropy'][i]):.2f} nats", (x + 14, yy),
                  DIM, self.small)
        yy += 18
        top_p, top_id = snap["top_p"][i], snap["top_id"][i]
        bar_x, bar_w = x + 140, w - 140 - 70
        for k, (p, t) in enumerate(zip(top_p, top_id)):
            ry = yy + k * 24
            hit = int(t) == true
            col = GOOD_C if hit else INK
            self.text(self.fit_text(self.tok_label(int(t)), 120, self.font), (x + 14, ry + 3), col, self.font)
            pg.draw.rect(self.canvas, INK_FAINT, (bar_x, ry + 6, bar_w, 6), 1)
            pg.draw.rect(self.canvas, col if hit else VAL_C, (bar_x, ry + 6, max(1, int(bar_w * float(p))), 6))
            self.text(f"{float(p):.1%}", (x + w - 14, ry + 3), col, self.font, align="right")
        yy += len(top_p) * 24 + 10
        self.text(self.fit_text("green = the actual next token", w - 28, self.small), (x + 14, min(yy, y + h - 22)),
                  FAINT, self.small)

    def wrap(self, s: str, width: int, font, max_lines: int) -> list[str]:
        words = s.split(" ")
        lines, cur = [], ""
        for wd in words:
            t = (cur + " " + wd) if cur else wd
            if self.tw(t, font) <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = wd
            if len(lines) >= max_lines:
                break
        if cur and len(lines) < max_lines:
            lines.append(cur)
        lines = lines[-max_lines:]
        return [self.fit_text(ln, width, font) for ln in lines]

    def draw_calibration(self, rect) -> None:
        pg = self.pg
        e = self.tr.last_eval
        self.panel(rect, "Calibration", "validation · top-1 confidence vs accuracy", key="calibration")
        x, y, w, h = rect
        plot = (x + 48, y + 38, w - 48 - 20, h - 38 - 50)
        X, Y = self.axes(plot, (0, 1), (0, 1), xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: f"{v:.1f}", nx=5, ny=5,
                         xlabel="confidence")
        if e is None or "calib_acc" not in e or not isinstance(e["calib_acc"], np.ndarray):
            if e is not None and "ece" in e:
                self.text(f"ECE {e['ece']:.4f} (bins after the next evaluation)", (plot[0] + 6, plot[1] + 6), DIM,
                          self.small)
            else:
                self.waiting(plot)
            return
        self.dashed(INK_DIM, (float(X(0)), float(Y(0))), (float(X(1)), float(Y(1))))
        acc, conf, n = e["calib_acc"], e["calib_conf"], e["calib_n"]
        nb = len(acc)
        tot = max(n.sum(), 1)
        for b in range(nb):
            if not n[b]:
                continue
            x0, x1 = float(X(b / nb)), float(X((b + 1) / nb))
            ya = float(Y(acc[b]))
            pg.draw.rect(self.canvas, mix_rgb(CARD, VAL_C, 0.55), (int(x0) + 1, int(ya), max(1, int(x1 - x0) - 2),
                                                                     int(float(Y(0)) - ya)))
            yc = float(Y(conf[b]))
            pg.draw.line(self.canvas, LR_C, (int(x0) + 1, int(yc)), (int(x1) - 1, int(yc)), 2)
            share = n[b] / tot
            hh = int(18 * min(1.0, share * 4))
            pg.draw.rect(self.canvas, INK_DIM, (int(x0) + 1, int(float(Y(0))) - hh, max(1, int(x1 - x0) - 2), hh), 1)
        self.text(f"ECE {e['ece']:.4f} · top-1 {e['top1']:.1%}", (plot[0] + 6, plot[1] + 6), INK, self.small)
        lx = x + 14
        lx += self.swatch(lx, y + h - 22, mix_rgb(CARD, VAL_C, 0.55), "accuracy", "box") + 6
        lx += self.swatch(lx, y + h - 22, LR_C, "mean confidence") + 6

    def draw_accuracy(self, rect) -> None:
        tr = self.tr
        self.panel(rect, "Accuracy", "validation · next token in the model's top 1 / top 5", key="accuracy")
        x, y, w, h = rect
        plot = (x + 48, y + 38, w - 48 - 20, h - 38 - 50)
        ev = tr.hist["eval"]
        if len(ev) < 1:
            self.axes(plot, (0, 1), (0, 1))
            self.waiting(plot)
            return
        xs = np.asarray([r["tokens"] for r in ev], dtype=np.float64)
        t1 = np.asarray([r["top1"] for r in ev])
        t5 = np.asarray([r["top5"] for r in ev])
        X, Y = self.axes(plot, (0, max(float(xs[-1]), 1.0)), (0, max(0.1, float(t5.max()) * 1.15)), xfmt=fmt_tokens,
                         yfmt=lambda v: f"{v:.0%}", nx=3, ny=4, xlabel="tokens seen")
        self.polyline(X, Y, xs, t5, LR_C, 2)
        self.polyline(X, Y, xs, t1, VAL_C, 2)
        lx = x + 14
        lx += self.swatch(lx, y + h - 22, VAL_C, f"top-1 {t1[-1]:.1%}") + 6
        self.swatch(lx, y + h - 22, LR_C, f"top-5 {t5[-1]:.1%}")

    def draw_rank_hist(self, rect, snap) -> None:
        pg = self.pg
        self.panel(rect, "Rank of the actual token", "this passage", key="ranks")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        r = np.asarray(snap["rank"]) + 1
        edges = [1, 2, 6, 11, 51, 101, 1001, 10 ** 9]
        names = ["1", "2-5", "6-10", "11-50", "51-100", "101-1k", "> 1k"]
        cnt = np.array([((r >= a) & (r < b)).sum() for a, b in zip(edges[:-1], edges[1:])], dtype=np.float64)
        frac = cnt / max(cnt.sum(), 1)
        bx, by, bw, bh = x + 20, y + 44, w - 40, h - 44 - 56
        cw = bw / len(names)
        for i, (f, nm) in enumerate(zip(frac, names)):
            hh = int(bh * f)
            col = cmap_color(0.9 - 0.75 * i / (len(names) - 1), "viridis")
            pg.draw.rect(self.canvas, col, (int(bx + i * cw + 3), by + bh - hh, int(cw - 6), hh))
            self.text(f"{f:.0%}", (int(bx + i * cw + cw / 2), by + bh - hh - 14), TEXT, self.small, align="center")
            self.text(nm, (int(bx + i * cw + cw / 2), by + bh + 4), DIM, self.small, align="center")
        pg.draw.line(self.canvas, INK_FAINT, (bx, by + bh), (bx + bw, by + bh))
        self.text(self.fit_text(f"median rank {int(np.median(r))} · mean entropy {float(snap['entropy'].mean()):.2f} nats",
                                w - 28, self.small), (x + 14, y + h - 24), DIM, self.small)

    # =========================================================================== #
    # View: Attention
    # =========================================================================== #
    def draw_attention(self, rect) -> None:
        x, y, w, h = rect
        snap = self.tr.snapshot
        self.draw_head_grid((x, y, 560, 560), snap)
        self.draw_head_map((x + 568, y, w - 568, 560), snap)
        self.draw_attention_text((x, y + 568, w, h - 568), snap)

    def att_n(self, snap) -> int:
        return min(40, snap["attn"].shape[-1])

    def head_matrix(self, snap) -> np.ndarray:
        a = snap["attn"].astype(np.float32)
        if self.avg_heads:
            return a[self.sel_layer].mean(axis=0)
        return a[self.sel_layer, self.sel_head]

    def draw_head_grid(self, rect, snap) -> None:
        pg = self.pg
        self.panel(rect, "Every head", "first 40 tokens · rows = queries, columns = keys", key="heads_grid")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        L, H = snap["attn"].shape[:2]
        n = self.att_n(snap)
        gap = 8
        cell = int(min((w - 60 - gap * (H - 1)) / H, (h - 92 - gap * (L - 1)) / L))
        ox, oy = x + 44, y + 54
        key = ("grid", snap["step"], id(snap), cell)
        if self._cache.get("grid_key") != key:
            tiles = {}
            for l in range(L):
                for hh in range(H):
                    m = snap["attn"][l, hh, :n, :n].astype(np.float32)
                    tiles[(l, hh)] = rgb_surface(pg, heat_rgb(np.sqrt(m), 0.0, 1.0, "magma"), (cell, cell))
            self._cache["grid_tiles"] = tiles
            self._cache["grid_key"] = key
        tiles = self._cache["grid_tiles"]
        for hh in range(H):
            self.text(f"H{hh + 1}", (ox + hh * (cell + gap) + cell // 2, oy - 16), DIM, self.small, align="center")
        ent = snap["head_entropy"]
        for l in range(L):
            self.text(f"L{l + 1}", (ox - 8, oy + l * (cell + gap) + cell // 2 - 6), DIM, self.small, align="right")
            for hh in range(H):
                px, py = ox + hh * (cell + gap), oy + l * (cell + gap)
                self.canvas.blit(tiles[(l, hh)], (px, py))
                sel = (l == self.sel_layer and (hh == self.sel_head or self.avg_heads))
                pg.draw.rect(self.canvas, INK if sel else INK_FAINT, (px - 1, py - 1, cell + 2, cell + 2), 2 if sel else 1)
                self.hit((px, py, cell, cell), "head", (l, hh))
        hv = self.hovered("head")
        if hv is not None:
            l, hh = hv
            self.tip = [(f"layer {l + 1}, head {hh + 1}", TEXT),
                        (f"mean attention entropy {float(ent[l, hh]):.2f} nats", DIM), ("click to inspect", FAINT)]
        self.colorbar(x + 14, y + h - 20, 120, "magma", "0", "1 (sqrt scale)")

    def draw_head_map(self, rect, snap) -> None:
        pg = self.pg
        title = (f"Layer {self.sel_layer + 1} · " + ("mean of heads" if self.avg_heads else f"head {self.sel_head + 1}"))
        self.panel(rect, title, "attention probabilities softmax(QK^T/√d)", key="head_map")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        n = min(28, snap["attn"].shape[-1])
        m = self.head_matrix(snap)[:n, :n]
        toks = snap["tokens"][:n]
        lab_w = 110
        cell = int(min((w - lab_w - 30) / n, (h - 150) / n))
        ox, oy = x + lab_w + 10, y + 110
        img = heat_rgb(np.sqrt(m), 0.0, 1.0, "magma")
        self.canvas.blit(rgb_surface(pg, img, (cell * n, cell * n)), (ox, oy))
        pg.draw.rect(self.canvas, INK_FAINT, (ox - 1, oy - 1, cell * n + 2, cell * n + 2), 1)
        for i, t in enumerate(toks):
            lbl = self.fit_text(self.tok_label(int(t)), lab_w - 8, self.small)
            self.text(lbl, (ox - 6, oy + i * cell + (cell - 11) // 2), DIM, self.small, align="right")
            key = ("rot", lbl)
            img_t = self._cache.get(key)
            if img_t is None:
                img_t = self._cache[key] = pg.transform.rotate(self.small.render(lbl, True, DIM), 90)
            self.canvas.blit(img_t, (ox + i * cell + (cell - img_t.get_width()) // 2, oy - 6 - img_t.get_height()))
        if self._inside(self.mouse, (ox, oy, cell * n, cell * n)):
            qi = (self.mouse[1] - oy) // cell
            ki = (self.mouse[0] - ox) // cell
            if 0 <= qi < n and 0 <= ki < n:
                pg.draw.rect(self.canvas, VAL_C, (ox + ki * cell, oy + qi * cell, cell, cell), 1)
                self.tip = [(f"query {qi}: {self.tok_label(int(toks[qi]))}", TEXT),
                            (f"key {ki}: {self.tok_label(int(toks[ki]))}", TEXT),
                            (f"attention {float(m[qi, ki]):.3f}" + (" (future: masked)" if ki > qi else ""), VAL_C)]
        a = self.head_matrix(snap)
        T = a.shape[0]
        diag = float(np.mean([a[i, i] for i in range(T)]))
        prev = float(np.mean([a[i, i - 1] for i in range(1, T)]))
        first = float(a[1:, 0].mean())
        self.text(self.fit_text(f"on the whole passage ({T} tokens): self {diag:.2f} · previous token {prev:.2f} · "
                                f"first token {first:.2f}", w - 28, self.small), (x + 14, y + h - 22), DIM, self.small)

    def draw_attention_text(self, rect, snap) -> None:
        self.panel(rect, "Attention from a token", "hover a token: background = attention it pays to earlier tokens",
                   key="att_text")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        toks = list(snap["tokens"][:-1])
        q = self.hovered("atok")
        a = self.head_matrix(snap)
        cols = None
        if q is not None and q < a.shape[0]:
            row = a[q]
            mx = max(float(row.max()), 1e-6)
            cols = [tuple(int(v) for v in cmap_color(float(row[j]) / mx, "magma")) if j <= q else None
                    for j in range(len(toks))]
            cols[q] = (40, 90, 120)
            top = np.argsort(-row[:q + 1])[:3]
            self.tip = [(f"query {self.tok_label(int(toks[q]))} attends to:", TEXT)] + [
                (f"{self.tok_label(int(toks[k]))}  {float(row[k]):.3f}", DIM) for k in top]
        self.flow(toks, (x + 14, y + 36, w - 28, h - 46), cols, kind="atok", line_h=21)

    # =========================================================================== #
    # View: Heads
    # =========================================================================== #
    def draw_heads(self, rect) -> None:
        x, y, w, h = rect
        pr = self.tr.last_probe
        snap = self.tr.snapshot
        cw = (w - 3 * 8) // 4
        specs = [("induction", "Induction score", "attention to the token after the previous copy", "inferno"),
                 ("previous", "Previous-token score", "attention to position i − 1", "viridis"),
                 ("duplicate", "Duplicate-token score", "attention to the previous copy itself", "cividis"),
                 ("entropy", "Attention entropy", "nats, on the validation passage", "magma")]
        for k, (key, title, hint, cmap) in enumerate(specs):
            data = None
            if key == "entropy":
                data = snap["head_entropy"] if snap is not None else None
            elif pr is not None:
                data = pr[key]
            self.score_grid((x + k * (cw + 8), y, cw, 300), data, title, hint, cmap, key)
        y2 = y + 308
        h2 = h - 308
        self.draw_induction_history((x, y2, 616, h2 // 2 - 4))
        self.draw_copy_loss((x, y2 + h2 // 2 + 4, 616, h2 - h2 // 2 - 4))
        self.draw_induction_example((x + 624, y2, w - 624, h2), pr)

    def score_grid(self, rect, data, title: str, hint: str, cmap: str, key: str) -> None:
        pg = self.pg
        self.panel(rect, title, None, key=f"score_{key}")
        x, y, w, h = rect
        self.text(self.fit_text(hint, w - 24, self.small), (x + 12, y + 26), FAINT, self.small)
        if data is None:
            self.waiting(rect)
            return
        L, H = data.shape
        lo, hi = (0.0, max(1e-6, float(data.max()))) if key == "entropy" else (0.0, max(0.05, float(data.max())))
        if key != "entropy":
            hi = max(hi, 0.3) if key == "induction" else hi
        cell = int(min((w - 50) / H, (h - 100) / L))
        ox, oy = x + 34, y + 58
        for hh in range(H):
            self.text(f"{hh + 1}", (ox + hh * cell + cell // 2, oy - 14), DIM, self.small, align="center")
        for l in range(L):
            self.text(f"L{l + 1}", (ox - 6, oy + l * cell + cell // 2 - 6), DIM, self.small, align="right")
            for hh in range(H):
                v = float(data[l, hh])
                t = (v - lo) / (hi - lo) if hi > lo else 0
                col = cmap_color(t, cmap)
                r = (ox + hh * cell, oy + l * cell, cell - 2, cell - 2)
                pg.draw.rect(self.canvas, col, r)
                if l == self.sel_layer and hh == self.sel_head:
                    pg.draw.rect(self.canvas, INK, (r[0] - 1, r[1] - 1, r[2] + 2, r[3] + 2), 1)
                if cell >= 34:
                    self.text(f"{v:.2f}", (r[0] + r[2] // 2, r[1] + r[3] // 2 - 6), ink_on(col), self.small,
                              align="center")
                self.hit(r, "head", (l, hh))
                if self._inside(self.mouse, r):
                    self.tip = [(f"{title}: layer {l + 1}, head {hh + 1}", TEXT), (f"{v:.4f}", DIM)]
        self.colorbar(x + 12, y + h - 20, 80, cmap, f"{lo:.2g}", f"{hi:.2g}")

    def draw_induction_history(self, rect) -> None:
        tr = self.tr
        self.panel(rect, "Induction heads over training", "max score per layer", key="ind_hist")
        x, y, w, h = rect
        plot = (x + 48, y + 36, w - 48 - 130, h - 36 - 30)
        pr = tr.hist["probe"]
        if len(pr) < 2:
            self.axes(plot, (0, 1), (0, 1), ny=4)
            self.waiting(plot)
            return
        xs = np.asarray([r["tokens"] for r in pr], dtype=np.float64)
        ind = np.asarray([np.max(np.asarray(r["induction"]), axis=1) for r in pr])
        X, Y = self.axes(plot, (0, max(float(xs[-1]), 1.0)), (0, max(0.3, float(ind.max()) * 1.1)), xfmt=fmt_tokens,
                         yfmt=lambda v: f"{v:.2f}", nx=4, ny=4, xlabel="tokens seen")
        L = ind.shape[1]
        for l in range(L):
            col = cmap_color(0.2 + 0.75 * l / max(1, L - 1), "inferno")
            self.polyline(X, Y, xs, ind[:, l], col, 2)
            self.swatch(x + w - 120, y + 40 + 16 * l, col, f"layer {l + 1}  {ind[-1, l]:.2f}")

    def draw_copy_loss(self, rect) -> None:
        tr = self.tr
        self.panel(rect, "In-context copying", "loss on random tokens: first pass vs repeat", key="copy")
        x, y, w, h = rect
        plot = (x + 48, y + 36, w - 48 - 130, h - 36 - 30)
        pr = tr.hist["probe"]
        if len(pr) < 2:
            self.axes(plot, (0, 1), (0, 1), ny=4)
            self.waiting(plot)
            return
        xs = np.asarray([r["tokens"] for r in pr], dtype=np.float64)
        a = np.asarray([r["loss_first"] for r in pr])
        b = np.asarray([r["loss_repeat"] for r in pr])
        hi = float(max(a.max(), b.max())) + 0.5
        X, Y = self.axes(plot, (0, max(float(xs[-1]), 1.0)), (0, hi), xfmt=fmt_tokens, yfmt=lambda v: f"{v:.0f}",
                         nx=4, ny=4, xlabel="tokens seen")
        self.polyline(X, Y, xs, a, INK_DIM, 2)
        self.polyline(X, Y, xs, b, LR_C, 2)
        self.swatch(x + w - 120, y + 40, INK_DIM, f"first {a[-1]:.2f}")
        self.swatch(x + w - 120, y + 56, LR_C, f"repeat {b[-1]:.2f}")
        self.text("nats/token", (x + w - 120, y + 76), FAINT, self.small)

    def draw_induction_example(self, rect, pr) -> None:
        pg = self.pg
        self.panel(rect, "Strongest heads on a repeated sequence", None, key="ind_example")
        x, y, w, h = rect
        if pr is None:
            self.waiting(rect)
            return
        S = pr["seq_len"]
        att = pr["example_attn"].astype(np.float32)
        il, ih = np.unravel_index(int(np.argmax(pr["induction"])), pr["induction"].shape)
        pl, ph = np.unravel_index(int(np.argmax(pr["previous"])), pr["previous"].shape)
        sel = att[self.sel_layer, self.sel_head]
        size = min((w - 3 * 14) // 2, (h - 130) // 2 + 40)
        items = [(att[il, ih], f"induction: L{il + 1} H{ih + 1} ({pr['induction'][il, ih]:.2f})"),
                 (att[pl, ph], f"previous token: L{pl + 1} H{ph + 1} ({pr['previous'][pl, ph]:.2f})")]
        for k, (m, cap) in enumerate(items):
            px, py = x + 14 + k * (size + 14), y + 52
            self.text(self.fit_text(cap, size, self.small), (px, py - 16), INK, self.small)
            self.canvas.blit(rgb_surface(pg, heat_rgb(np.sqrt(m), 0, 1, "magma"), (size, size)), (px, py))
            pg.draw.rect(self.canvas, INK_FAINT, (px - 1, py - 1, size + 2, size + 2), 1)
            mid = px + size * S // (2 * S)
            self.dashed(INK_DIM, (mid, py), (mid, py + size), 3, 3)
            self.dashed(INK_DIM, (px, py + size // 2), (px + size, py + size // 2), 3, 3)
        py = y + 52 + size + 14
        sm = min(size, h - (py - y) - 60)
        if sm > 60:
            self.text(f"selected: L{self.sel_layer + 1} H{self.sel_head + 1}", (x + 14, py), DIM, self.small)
            self.canvas.blit(rgb_surface(pg, heat_rgb(np.sqrt(sel), 0, 1, "magma"), (sm, sm)), (x + 14, py + 16))
            pg.draw.rect(self.canvas, INK_FAINT, (x + 13, py + 15, sm + 2, sm + 2), 1)
            tx = x + 14 + sm + 16
            lines = [f"{S} random tokens, then the same {S} again.",
                     "An induction head attends from each token",
                     "in the repeat to the token that followed",
                     "its earlier copy: a stripe one step right",
                     "of the lower-left diagonal. A previous-",
                     "token head is the line just below the",
                     "main diagonal. Dashed lines: the repeat",
                     "begins. Scores = mean attention on these",
                     f"positions over 8 sequences (step {pr['step']:,})."]
            for i, ln in enumerate(lines):
                self.text(self.fit_text(ln, x + w - 14 - tx, self.small), (tx, py + 16 + 15 * i), DIM, self.small)

    # =========================================================================== #
    # View: Logit lens
    # =========================================================================== #
    def draw_lens(self, rect) -> None:
        x, y, w, h = rect
        snap = self.tr.snapshot
        self.draw_lens_grid((x, y, 876, h), snap)
        sx, sw = x + 884, w - 884
        self.draw_lens_loss((sx, y, sw, 290), snap)
        self.draw_resid_norm((sx, y + 298, sw, 290), snap)
        self.draw_block_writes((sx, y + 596, sw, h - 596), snap)

    def draw_lens_grid(self, rect, snap) -> None:
        pg = self.pg
        self.panel(rect, "Logit lens", "each layer's residual stream decoded by the final norm + unembedding",
                   key="lens_grid")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        T = len(snap["nll"])
        ncol = 13
        self.lens_offset = max(0, min(self.lens_offset, T - ncol))
        o = self.lens_offset
        rows = snap["lens_top"].shape[0]
        lab_w = 92
        cw = (w - lab_w - 28) // ncol
        top = y + 74
        rh = min(100, (h - 74 - 90) // rows)
        toks = snap["tokens"]
        ox = x + lab_w + 14
        self.text(f"positions {o + 1}-{o + ncol} of {T}  (←/→ scroll)", (x + w - 14, y + 34), DIM, self.small,
                  align="right")
        self.text("INPUT", (ox - 10, top - 26), DIM, self.small, align="right")
        for c in range(ncol):
            i = o + c
            if i >= T:
                break
            cx = ox + c * cw
            self.text(self.fit_text(self.tok_label(int(toks[i])), cw - 6, self.font), (cx + cw // 2, top - 26), INK,
                      self.font, align="center")
        for r in range(rows):
            ry = top + (rows - 1 - r) * rh                   # embedding at the bottom, output on top
            name = "embed" if r == 0 else (f"block {r}" + (" = out" if r == rows - 1 else ""))
            self.text(name, (ox - 10, ry + rh // 2 - 7), INK if r == rows - 1 else DIM, self.small, align="right")
            for c in range(ncol):
                i = o + c
                if i >= T:
                    break
                p_true = float(snap["lens_p_true"][r, i])
                t = int(snap["lens_top"][r, i])
                pt = float(snap["lens_p_top"][r, i])
                col = cmap_color(math.sqrt(p_true), "viridis")
                cell = (ox + c * cw + 1, ry + 1, cw - 2, rh - 2)
                pg.draw.rect(self.canvas, col, cell)
                correct = t == int(toks[i + 1])
                if correct:
                    pg.draw.rect(self.canvas, INK, cell, 2)
                tc = ink_on(col)
                self.text(self.fit_text(self.tok_label(t), cw - 8, self.font), (cell[0] + cell[2] // 2, cell[1] + rh // 2 - 14),
                          tc, self.font, align="center")
                self.text(f"{pt:.0%}", (cell[0] + cell[2] // 2, cell[1] + rh // 2 + 4), tc, self.small, align="center")
                self.hit(cell, "lens", (r, i))
        yb = top + rows * rh + 8
        self.text("NEXT", (ox - 10, yb + 2), DIM, self.small, align="right")
        for c in range(ncol):
            i = o + c
            if i >= T:
                break
            self.text(self.fit_text(self.tok_label(int(toks[i + 1])), cw - 6, self.font), (ox + c * cw + cw // 2, yb),
                      GOOD_C, self.font, align="center")
        self.colorbar(x + 14, y + h - 24, 140, "viridis", "p(actual next) 0", "1 (sqrt)")
        self.text("cell text = the layer's top token and its probability · white frame = top token is correct",
                  (x + w - 14, y + h - 26), FAINT, self.small, align="right")
        hv = self.hovered("lens")
        if hv is not None:
            r, i = hv
            self.tip = [(f"{'embedding' if r == 0 else f'after block {r}'} · position {i + 1}", TEXT),
                        (f"top: {self.tok_label(int(snap['lens_top'][r, i]))} ({float(snap['lens_p_top'][r, i]):.1%})", DIM),
                        (f"actual {self.tok_label(int(toks[i + 1]))}: p {float(snap['lens_p_true'][r, i]):.2%}, "
                         f"{float(snap['lens_nll'][r, i]):.2f} nats", VAL_C)]

    def draw_lens_loss(self, rect, snap) -> None:
        self.panel(rect, "Lens loss by depth", "mean over the passage", key="lens_loss")
        x, y, w, h = rect
        plot = (x + 48, y + 38, w - 48 - 18, h - 38 - 40)
        if snap is None:
            self.waiting(rect)
            return
        v = snap["lens_nll"].mean(axis=1)
        n = len(v)
        X, Y = self.axes(plot, (0, n - 1), (0, float(v.max()) * 1.1), xfmt=lambda t: "emb" if t < 0.5 else f"{t:.0f}",
                         yfmt=lambda t: f"{t:.0f}", nx=n - 1, ny=4, xlabel="block")
        self.polyline(X, Y, np.arange(n), v, VAL_C, 2)
        for i, val in enumerate(v):
            self.pg.draw.circle(self.canvas, VAL_C, (int(X(i)), int(Y(val))), 3)
        self.text(f"output {v[-1]:.2f} nats", (plot[0] + plot[2] - 4, plot[1] + 4), VAL_C, self.small, align="right")

    def draw_resid_norm(self, rect, snap) -> None:
        self.panel(rect, "Residual stream size", "RMS per position, mean and range", key="resid_norm")
        x, y, w, h = rect
        plot = (x + 48, y + 38, w - 48 - 18, h - 38 - 40)
        if snap is None:
            self.waiting(rect)
            return
        r = snap["resid_rms"]
        m, lo, hi = r.mean(axis=1), np.percentile(r, 10, axis=1), np.percentile(r, 90, axis=1)
        n = len(m)
        X, Y = self.axes(plot, (0, n - 1), (0, float(hi.max()) * 1.1), xfmt=lambda t: "emb" if t < 0.5 else f"{t:.0f}",
                         yfmt=lambda t: f"{t:.2g}", nx=n - 1, ny=4, xlabel="block")
        pts = [(float(X(i)), float(Y(hi[i]))) for i in range(n)] + [(float(X(i)), float(Y(lo[i]))) for i in reversed(range(n))]
        self.pg.draw.polygon(self.canvas, mix_rgb(CARD, INK, 0.12), pts)
        self.polyline(X, Y, np.arange(n), m, INK, 2)

    def draw_block_writes(self, rect, snap) -> None:
        pg = self.pg
        self.panel(rect, "What each block writes", "RMS of attention and MLP outputs", key="writes")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        a = snap["attn_out_rms"].mean(axis=1)
        m = snap["mlp_out_rms"].mean(axis=1)
        L = len(a)
        mx = max(float(a.max()), float(m.max()), 1e-9)
        bx, by, bw, bh = x + 40, y + 40, w - 60, h - 40 - 56
        gw = bw / L
        for l in range(L):
            ha, hm = int(bh * a[l] / mx), int(bh * m[l] / mx)
            x0 = int(bx + l * gw)
            pg.draw.rect(self.canvas, VAL_C, (x0 + 4, by + bh - ha, int(gw / 2 - 5), ha))
            pg.draw.rect(self.canvas, LR_C, (x0 + int(gw / 2), by + bh - hm, int(gw / 2 - 5), hm))
            self.text(f"{l + 1}", (x0 + int(gw / 2), by + bh + 4), DIM, self.small, align="center")
        pg.draw.line(self.canvas, INK_FAINT, (bx, by + bh), (bx + bw, by + bh))
        lx = x + 14
        lx += self.swatch(lx, y + h - 22, VAL_C, "attention", "box") + 8
        self.swatch(lx, y + h - 22, LR_C, "MLP", "box")

    # =========================================================================== #
    # View: Network (3-D residual stream)
    # =========================================================================== #
    NET_T, NET_C = 28, 64

    def _splat(self, x, y, w, h, rgb: np.ndarray) -> None:
        """Filled rectangles centred on (x, y), written straight into the pixel array."""
        arr = self.pg.surfarray.pixels3d(self.canvas)
        try:
            W, H = arr.shape[:2]
            w = np.clip(np.rint(w), 1, 24).astype(np.int64)
            h = np.clip(np.rint(h), 1, 24).astype(np.int64)
            x0, y0 = np.rint(x - w / 2).astype(np.int64), np.rint(y - h / 2).astype(np.int64)
            for dy in range(int(h.max())):
                for dx in range(int(w.max())):
                    m = (dx < w) & (dy < h)
                    px, py = x0[m] + dx, y0[m] + dy
                    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                    arr[px[ok], py[ok]] = rgb[m][ok]
        finally:
            del arr

    def _net_geometry(self, snap):
        key = ("netgeo", snap["step"], id(snap))
        if self._cache.get("netgeo_key") == key:
            return self._cache["netgeo"]
        resid = snap["resid"].astype(np.float32)                 # (L+1, T, d)
        Lp1, T, d = resid.shape
        nt = min(self.NET_T, T)
        nc = min(self.NET_C, d)
        sub = resid[:, :nt]
        chan = np.argsort(-sub.std(axis=(0, 1)))[:nc]
        chan.sort()
        vals = sub[:, :, chan]                                    # (L+1, nt, nc)
        rms = np.sqrt((resid[:, :nt] ** 2).mean(axis=(1, 2)))[:, None, None]
        t = 0.5 + 0.5 * np.clip(vals / (2.5 * np.maximum(rms, 1e-6)), -1, 1)
        colors = apply_cmap(t, "rdbu").reshape(Lp1, -1, 3)
        zs = np.linspace(-1.6, 1.6, Lp1)
        xs = np.linspace(-1.55, 1.55, nt)
        ys = np.linspace(-0.5, 0.5, nc)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        planes = [np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], -1) for z in zs]
        att = snap["attn"].astype(np.float32).mean(axis=1)[:, :nt, :nt]   # (L, nt, nt), mean over heads
        links = []
        for l in range(att.shape[0]):
            for q in range(1, nt):
                row = att[l, q, :q + 1]
                for k in np.argsort(-row)[:2]:
                    if row[k] > 0.12 and k != q:
                        links.append((l, int(k), q, float(row[k])))
        geo = {"colors": colors, "planes": planes, "zs": zs, "xs": xs, "nt": nt, "nc": nc, "links": links,
               "rms": rms.ravel(), "chan": chan}
        self._cache["netgeo"], self._cache["netgeo_key"] = geo, key
        return geo

    def draw_network(self, rect, dt: float) -> None:
        pg = self.pg
        snap = self.tr.snapshot
        self.sections["view"] = ("Stream 3-D", rect)
        x, y, w, h = rect
        pg.draw.rect(self.canvas, BORDER, rect, 1)
        if snap is None:
            self.waiting(rect)
            return
        cam = self.cams["network"]
        if cam.auto and self._drag is None:
            cam.orbit(0.10 * dt, 0.0)
        geo = self._net_geometry(snap)
        stage = (x, y + 40, w, h - 40)
        self.canvas.set_clip(pg.Rect(rect))
        try:
            order = np.argsort([-float(cam.project(np.array([0, 0, z]), stage)[2]) for z in geo["zs"]])
            hx, hy = 1.62, 0.58
            for li in order:
                z = geo["zs"][li]
                corners = np.array([(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)])
                cx_, cy_, _ = cam.project(corners, stage)
                pts = list(zip(cx_.tolist(), cy_.tolist()))
                pg.draw.polygon(self.canvas, (8, 9, 10), pts)
                pg.draw.polygon(self.canvas, INK_FAINT if li != self.sel_layer + 1 else INK_DIM, pts, 1)
                P = geo["planes"][li]
                sx, sy, dep = cam.project(P, stage)
                c0 = cam.project(np.array([0.0, 0.0, z]), stage)
                ax = cam.project(np.array([geo["xs"][1] - geo["xs"][0], 0.0, z]), stage)
                ay = cam.project(np.array([0.0, 1.24 / max(geo["nc"] - 1, 1), z]), stage)
                du = abs(ax[0] - c0[0]) + abs(ay[0] - c0[0])
                dv = abs(ax[1] - c0[1]) + abs(ay[1] - c0[1])
                ordr = np.argsort(-dep)
                self._splat(sx[ordr], sy[ordr], np.full(len(ordr), max(1.0, du * 0.9)),
                            np.full(len(ordr), max(1.0, dv * 0.9)), geo["colors"][li][ordr])
            # attention links along the back edge, drawn over the stack (weakest first)
            for (l, k, q, wgt) in sorted(geo["links"], key=lambda t: t[3]):
                za, zb = geo["zs"][l], geo["zs"][l + 1]
                a = cam.project(np.array([geo["xs"][k], hy, za]), stage)
                b = cam.project(np.array([geo["xs"][q], hy, zb]), stage)
                strength = min(1.0, wgt)
                col = mix_rgb(BG, LINK_POS_C, 0.3 + 0.7 * strength)
                pg.draw.line(self.canvas, col, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                             2 if strength > 0.5 else 1)
        finally:
            self.canvas.set_clip(None)
        self.net_labels(rect, stage, cam, geo, snap)

    def net_labels(self, rect, stage, cam: Camera, geo: dict, snap) -> None:
        pg = self.pg
        x, y, w, h = rect
        cx = x + w // 2
        self.spaced("RESIDUAL STREAM", (cx, y + 10), INK, self.f_view, spacing=4, align="center")
        pg.draw.line(self.canvas, INK_DIM, (cx - 28, y + 43), (cx + 28, y + 43))
        self.spaced(f"{geo['nt']} TOKENS × {geo['nc']} OF {self.tr.model_cfg.d_model} CHANNELS × "
                    f"{len(geo['zs'])} STAGES · STEP {snap['step']:,}", (cx, y + 51), INK_DIM, align="center")
        nt = geo["nt"]
        toks = snap["tokens"]
        z0, z1 = geo["zs"][0], geo["zs"][-1]
        for i in range(0, nt, 2):
            px, py, _ = cam.project(np.array([geo["xs"][i], -0.66, z0]), stage)
            self.text(self.fit_text(self.tok_label(int(toks[i])), 60, self.small), (int(px), int(py)), DIM, self.small,
                      align="center")
            px, py, _ = cam.project(np.array([geo["xs"][i], -0.66, z1 + 0.18]), stage)
            self.text(self.fit_text(self.tok_label(int(snap["top_id"][i, 0])), 60, self.small), (int(px), int(py) - 12),
                      VAL_C, self.small, align="center")
        for li, z in enumerate(geo["zs"]):
            px, py, _ = cam.project(np.array([1.75, 0.7, z]), stage)
            name = "embedding" if li == 0 else f"after block {li}"
            self.text(name, (int(px) + 6, int(py) - 6), INK if li == self.sel_layer + 1 else DIM, self.small)
            self.text(f"rms {geo['rms'][li]:.2f}", (int(px) + 6, int(py) + 6), FAINT, self.small)
        x0, y0 = x + 22, y + h - 196
        self.legend_card(x0 - 12, y0 - 12, 276, 184)
        self.spaced("UNITS", (x0, y0), INK)
        self.text("one cell = one channel of one token", (x0, y0 + 16), DIM, self.small)
        grad = apply_cmap(np.linspace(0, 1, 140)[None, :], "rdbu")
        self.canvas.blit(rgb_surface(pg, np.repeat(grad, 8, axis=0)), (x0, y0 + 36))
        self.text("−2.5 rms     0     +2.5 rms", (x0, y0 + 48), DIM, self.small)
        self.text("bottom labels: input tokens", (x0, y0 + 70), DIM, self.small)
        self.text("top labels: the model's next-token guess", (x0, y0 + 86), VAL_C, self.small)
        self.spaced("LINKS", (x0, y0 + 110), INK)
        self.swatch(x0, y0 + 126, LINK_POS_C, "attention ≥ 0.12 (mean of heads),")
        self.text("key token (lower) → query token (upper)", (x0, y0 + 142), DIM, self.small)
        yaw = (math.degrees(cam.yaw) + 180.0) % 360.0 - 180.0
        self.spaced(f"YAW {yaw:+06.1f}°   TILT {math.degrees(cam.pitch):04.1f}°", (x + w - 16, y + h - 40), INK_DIM,
                    align="right")
        self.text(f"drag orbit · wheel zoom · O auto-orbit {'on' if cam.auto else 'off'}", (x + w - 16, y + h - 22),
                  FAINT, self.small, align="right")

    def legend_card(self, x: int, y: int, w: int, h: int) -> None:
        key = ("card", w, h)
        if key not in self._cache:
            pg = self.pg
            card = pg.Surface((w, h), pg.SRCALPHA)
            card.fill((0, 0, 0, 215))
            pg.draw.rect(card, INK_FAINT + (255,), (0, 0, w, h), 1)
            self._cache[key] = card
        self.canvas.blit(self._cache[key], (x, y))

    # =========================================================================== #
    # View: Embeddings
    # =========================================================================== #
    def draw_embed(self, rect, dt: float) -> None:
        x, y, w, h = rect
        self.draw_cloud((x, y, 800, h), dt)
        sx, sw = x + 808, w - 808
        self.draw_neighbors((sx, y, sw, 300))
        self.draw_variance((sx, y + 308, sw, 250))
        self.draw_positional((sx, y + 566, sw, h - 566))

    def particles(self, rect, sx, sy, lum, glow: float = 0.55, key: str = "cloud", ink=INK,
                  colors: np.ndarray | None = None) -> None:
        """Additive glowing particles (the splash-screen look): each point's light is split
        bilinearly over the four pixels around its exact position, a soft halo comes from two
        box blurs, and a 1 - exp(-x) curve maps accumulated light to colour like exposure on
        film, so crowded regions burn toward white. `colors` (N x 3, 0-255) tints each point;
        otherwise every point has the ink colour. The exposure follows the brightest 0.5% of
        pixels, smoothed between frames."""
        x0, y0, W, H = (int(v) for v in rect)
        res = 1.0 if W * H < 250_000 else 0.5                  # large panels: accumulate at half resolution
        w, h = max(2, int(W * res)), max(2, int(H * res))
        fx = (np.asarray(sx, dtype=np.float64) - x0) * (w / W)
        fy = (np.asarray(sy, dtype=np.float64) - y0) * (h / H)
        lum = np.asarray(lum, dtype=np.float64)
        col = (np.tile(np.asarray(ink, dtype=np.float64), (len(lum), 1)) if colors is None
               else np.asarray(colors, dtype=np.float64)) / 255.0
        ok = (fx >= 0) & (fx < w - 1) & (fy >= 0) & (fy < h - 1) & np.isfinite(lum)
        fx, fy, lum, col = fx[ok], fy[ok], lum[ok], col[ok]
        if not len(fx):
            return
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        ax, ay = fx - ix, fy - iy
        base = iy * w + ix
        idx = np.concatenate([base, base + 1, base + w, base + w + 1])          # the four bilinear corners
        wts = np.concatenate([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay]) * np.tile(lum, 4)
        acc = np.empty((h * w, 3), dtype=np.float32)
        for ch in range(3):
            acc[:, ch] = np.bincount(idx, weights=wts * np.tile(col[:, ch], 4), minlength=w * h)
        acc = acc.reshape(h, w, 3)
        r1, r2 = (1, 2) if res < 1 else (1, 3)
        if glow > 0:                                                       # halo: two box blurs
            acc += (glow * (1.5 if res < 1 else 6.0)) * box_blur(box_blur(acc, r1), r2).astype(np.float32)
        peak = acc[::2, ::2].max(axis=2)
        lit = peak[peak > 0]
        ref = float(np.percentile(lit, 99.5)) if len(lit) else 1.0
        gain = self._cache.get(("gain", key))
        target = 1.6 / max(ref, 1e-12)
        gain = target if gain is None else gain + (target - gain) * 0.15
        self._cache[("gain", key)] = gain
        exposure = np.float32(gain) * acc
        rgb = 1.0 - np.exp(-exposure)
        burn = 1.0 - np.exp(-0.25 * np.maximum(exposure.sum(axis=2, keepdims=True) - 2.4, 0.0))
        rgb += (1.0 - rgb) * burn                                          # dense cores burn toward white
        surf = rgb_surface(self.pg, (255.0 * rgb).astype(np.uint8))
        if (w, h) != (W, H):
            surf = self.pg.transform.smoothscale(surf, (W, H))
        self.canvas.blit(surf, (x0, y0), special_flags=self.pg.BLEND_RGB_ADD)

    def draw_cloud(self, rect, dt: float) -> None:
        pg = self.pg
        emb = self.tr.embed
        self.panel(rect, None, None, key="cloud")
        self.sections["cloud"] = ("Token embeddings", tuple(rect))
        x, y, w, h = rect
        if emb is None:
            self.waiting(rect)
            return
        cam = self.cams["embed"]
        if cam.auto and self._drag is None:
            cam.orbit(0.06 * dt, 0.0)
        c = emb["coords"].astype(np.float64)
        ids = emb["ids"]
        n = len(c)
        centre = np.median(c, axis=0)                                         # frame the whole vocabulary
        scale = np.percentile(np.abs(c - centre), 98, axis=0)
        P = (c - centre) / np.maximum(scale, 1e-9) * 0.8
        stage = (x + 1, y + 1, w - 2, h - 2)
        sx, sy, away = cam.project(P, stage)
        near = -cam.rotate(P)[2]
        cnt = np.asarray(emb.get("counts", np.zeros(n)), dtype=np.float64)
        b = np.log1p(cnt) / max(float(np.log1p(cnt.max())), 1e-9)
        t = np.clip((near + 0.7) / 1.3, 0, 1)
        depth = 0.18 + 0.82 * t * t * (3 - 2 * t)                             # the splash's depth fall-off
        lum = (0.22 + 0.78 * b) * depth
        rank_t = 1.0 - np.arange(n) / max(n - 1, 1)                           # ids are in frequency order
        heat = apply_cmap(rank_t, "heat")                                     # blue = rare ... red = frequent
        self.canvas.set_clip(pg.Rect(stage))
        try:
            for a in range(3):                                                 # principal axes, faint
                e = np.zeros((2, 3))
                e[0, a], e[1, a] = -1.05, 1.05
                ex, ey, _ = cam.project(e, stage)
                self.dashed(INK_FAINT, (ex[0], ey[0]), (ex[1], ey[1]), 2, 4)
                self.text(f"PC{a + 1}", (int(ex[1]) + 4, int(ey[1]) - 6), FAINT, self.small)
            self.particles(stage, sx, sy, lum, colors=heat)
            sel_idx = None
            if self.embed_sel is not None:
                hits = np.nonzero(ids == self.embed_sel)[0]
                sel_idx = int(hits[0]) if len(hits) else None
            nb = self.worker.neighbors
            nb_ids = {t_ for t_, _ in nb[1]} if nb and nb[0] == self.embed_sel else set()
            pos_of = {int(t_): i for i, t_ in enumerate(ids[:20000])}
            labels = list(range(min(28, n)))
            labels += [pos_of[t_] for t_ in nb_ids if t_ in pos_of]
            inner = (x + 10, y + 34, w - 80, h - 70)
            for i in sorted(set(labels), key=lambda j: -away[j]):
                if not self._inside((sx[i], sy[i]), inner):
                    continue
                col = INK if int(ids[i]) in nb_ids else mix_rgb(BG, INK, 0.62 + 0.38 * float(depth[i]))
                if int(ids[i]) in nb_ids:
                    pg.draw.circle(self.canvas, INK, (int(sx[i]), int(sy[i])), 4, 1)
                self.text(self.tok_label(int(ids[i])), (int(sx[i]) + 6, int(sy[i]) - 7), col, self.small)
            if sel_idx is not None and self._inside((sx[sel_idx], sy[sel_idx]), inner):
                px_, py_ = int(sx[sel_idx]), int(sy[sel_idx])
                pg.draw.circle(self.canvas, INK, (px_, py_), 8, 1)
                for d0, d1 in ((10, 22),):
                    pg.draw.line(self.canvas, INK, (px_ - d1, py_), (px_ - d0, py_))
                    pg.draw.line(self.canvas, INK, (px_ + d0, py_), (px_ + d1, py_))
                    pg.draw.line(self.canvas, INK, (px_, py_ - d1), (px_, py_ - d0))
                    pg.draw.line(self.canvas, INK, (px_, py_ + d0), (px_, py_ + d1))
                self.text(self.tok_label(int(ids[sel_idx])), (px_ + 12, py_ + 8), INK, self.font)
        finally:
            self.canvas.set_clip(None)
        if self._inside(self.mouse, stage):
            d2 = (sx - self.mouse[0]) ** 2 + (sy - self.mouse[1]) ** 2
            d2 = np.where(lum > 0.02, d2, np.inf)
            j = int(np.argmin(d2))
            if d2[j] < 49:
                t_ = int(ids[j])
                self.tip = [(self.tok_label(t_), TEXT), (f"frequency rank {j + 1:,} · {int(cnt[j]):,} occurrences", DIM),
                            (f"|e| = {float(emb['norms'][j]):.3f} · click for neighbours", DIM)]
                self.hit((int(sx[j]) - 7, int(sy[j]) - 7, 14, 14), "point", t_)
        # readouts in the corners (splash-screen layout)
        ev = emb["explained"]
        self.spaced("TOKEN EMBEDDINGS", (x + 16, y + 14), INK_DIM)
        self.spaced(f"PCA  {100 * ev[0]:04.1f} · {100 * ev[1]:04.1f} · {100 * ev[2]:04.1f} %", (x + 16, y + 30), FAINT)
        yaw = (math.degrees(cam.yaw) + 180.0) % 360.0 - 180.0
        self.spaced(f"YAW {'−' if yaw < 0 else '+'}{abs(yaw):05.1f}°   TILT {math.degrees(cam.pitch):04.1f}°",
                    (x + w - 16, y + 14), INK_DIM, align="right")
        self.spaced(f"{n:,} TOKENS · STEP {emb['step']:,}", (x + w - 16, y + 30), FAINT, align="right")
        cx, yy, bw = x + w // 2, y + h - 40, 160
        self.spaced("RARE", (cx - bw // 2 - 12, yy), INK_DIM, align="right")
        grad = apply_cmap(np.linspace(0, 1, bw)[None, :], "heat")
        self.canvas.blit(rgb_surface(pg, np.repeat(grad, 3, axis=0)), (cx - bw // 2, yy + 4))
        self.spaced("FREQUENT", (cx + bw // 2 + 12, yy), INK_DIM)
        self.text("colour = training-frequency rank · brightness = log frequency × depth", (cx, yy + 16), FAINT,
                  self.small, align="center")
        self.text("drag orbit · wheel zoom · O auto-orbit · click a token", (x + w - 16, y + h - 62), FAINT,
                  self.small, align="right")
        self.text("axes: frequency-weighted principal components", (x + 16, y + h - 62), FAINT, self.small)

    def draw_neighbors(self, rect) -> None:
        pg = self.pg
        sel = self.embed_sel
        self.panel(rect, "Nearest neighbours", "cosine similarity of token embeddings", key="neighbors")
        x, y, w, h = rect
        nb = self.worker.neighbors
        if sel is None:
            self.text("click a token in the cloud", (x + 14, y + 40), FAINT, self.small)
            return
        self.text(self.tok_label(sel), (x + 14, y + 34), LR_C, self.f_head)
        if nb is None or nb[0] != sel:
            self.text("computing…", (x + 14, y + 60), FAINT, self.small)
            return
        bar_x, bar_w = x + 150, w - 150 - 60
        for k, (t, s) in enumerate(nb[1][:11]):
            ry = y + 62 + k * 20
            self.text(self.fit_text(self.tok_label(t), 128, self.font), (x + 14, ry), INK, self.font)
            pg.draw.rect(self.canvas, INK_FAINT, (bar_x, ry + 5, bar_w, 5), 1)
            pg.draw.rect(self.canvas, VAL_C, (bar_x, ry + 5, max(1, int(bar_w * max(0.0, s))), 5))
            self.text(f"{s:.3f}", (x + w - 14, ry), DIM, self.font, align="right")
            self.hit((x + 14, ry, w - 28, 18), "neighbor", t)

    def draw_variance(self, rect) -> None:
        pg = self.pg
        emb = self.tr.embed
        self.panel(rect, "Embedding geometry", "explained variance · norm vs frequency, all tokens", key="variance")
        x, y, w, h = rect
        if emb is None:
            self.waiting(rect)
            return
        ev = emb["explained"]
        for k, v in enumerate(ev):
            ry = y + 38 + k * 18
            self.text(f"PC{k + 1}", (x + 14, ry), DIM, self.small)
            pg.draw.rect(self.canvas, INK_FAINT, (x + 50, ry + 4, 120, 6), 1)
            pg.draw.rect(self.canvas, VAL_C, (x + 50, ry + 4, max(1, int(120 * float(v) / max(float(ev[0]), 1e-9))), 6))
            self.text(f"{float(v):.1%}", (x + 178, ry), TEXT, self.small)
        plot = (x + 250, y + 38, w - 250 - 16, h - 38 - 40)
        norms = emb["norms"]
        cnt = np.maximum(np.asarray(emb["counts"], dtype=np.float64), 1)
        X, Y = self.axes(plot, (float(cnt.min()), float(cnt.max())), (0, float(norms.max()) * 1.1), log_x=True,
                         xfmt=fmt_tokens, yfmt=lambda v: f"{v:.1f}", nx=2, ny=3, xlabel="count")
        px, py = X(cnt), Y(norms)
        rq = 1.0 - np.arange(len(px)) / max(len(px) - 1, 1)
        self.particles(plot, px, py, np.full(len(px), 1.0), glow=0.3, key="norms", colors=apply_cmap(rq, "heat"))
        self.text("|e|", (plot[0] + 4, plot[1] + 4), DIM, self.small)

    def draw_positional(self, rect) -> None:
        tr = self.tr
        snap = tr.snapshot
        mc = tr.model_cfg
        self.panel(rect, "Position", "attention by distance per layer" + (" · RoPE" if mc.pos == "rope" else ""),
                   key="positional")
        x, y, w, h = rect
        if snap is None:
            self.waiting(rect)
            return
        d = snap["attn_by_distance"]
        L, D = d.shape
        plot = (x + 48, y + 38, w - 48 - 16, h - 38 - 64)
        X, Y = self.axes(plot, (1, D), (max(1e-4, float(d[:, :D].min()) * 0.8), 1.0), log_x=True, log_y=True,
                         xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0e}", nx=3, ny=3,
                         xlabel="distance + 1 (0 = itself)")
        for l in range(L):
            col = cmap_color(0.2 + 0.75 * l / max(1, L - 1), "inferno")
            self.polyline(X, Y, np.arange(1, D + 1), np.maximum(d[l], 1e-6), col, 2 if l == self.sel_layer else 1)
        if mc.pos == "rope":
            wl = rope_wavelengths(mc.head_dim, mc.rope_base)
            txt = f"rotary wavelengths {wl.min():.1f} … {wl.max():,.0f} tokens over {len(wl)} frequency pairs"
        else:
            txt = "learned absolute position embeddings"
        self.text(self.fit_text(txt, w - 28, self.small), (x + 14, y + h - 22), DIM, self.small)
        lx = x + 14
        for l in range(L):
            col = cmap_color(0.2 + 0.75 * l / max(1, L - 1), "inferno")
            lx += self.swatch(lx, y + h - 40, col, f"L{l + 1}")

    # =========================================================================== #
    # View: Export
    # =========================================================================== #
    def toggle(self, rect, label: str, on: bool, action: str) -> int:
        """A pill-shaped toggle button; returns its width."""
        pg = self.pg
        x, y = rect[:2]
        w = max(rect[2], self.tw(label, self.font) + 24)
        r = (x, y, w, rect[3])
        hot = self._inside(self.mouse, r)
        pg.draw.rect(self.canvas, PANEL_HI if on or hot else CARD, r)
        pg.draw.rect(self.canvas, INK if on else (INK_DIM if hot else INK_FAINT), r, 1)
        pg.draw.rect(self.canvas, GOOD_C if on else INK_FAINT, (x + 7, y + rect[3] // 2 - 3, 6, 6), 0 if on else 1)
        self.text(label, (x + 18, y + (rect[3] - 13) // 2), INK if on else DIM, self.font)
        self.buttons.append((r, action))
        return w

    def draw_export(self, rect) -> None:
        x, y, w, h = rect
        self.draw_capture((x, y, 600, 360))
        self.draw_frames((x, y + 368, 600, h - 368))
        self.draw_export_panel((x + 608, y, w - 608, 360))
        self.draw_preview((x + 608, y + 368, w - 608, h - 368))

    def draw_capture(self, rect) -> None:
        pg, wk = self.pg, self.worker
        rec = wk.recorder
        c = self.rec_cfg
        recording = rec is not None and rec.active
        self.panel(rect, "Time-lapse capture", "measurements + every dashboard panel at chosen updates",
                   key="capture")
        x, y, w, h = rect
        dot = BAD if recording else INK_FAINT
        pg.draw.circle(self.canvas, dot, (x + 22, y + 46), 6, 0 if recording else 1)
        self.spaced("RECORDING" if recording else "IDLE", (x + 36, y + 41), BAD if recording else DIM, self.f_cap)
        if recording:
            nxt = rec.from_step + ((self.tr.step - rec.from_step) // rec.every + 1) * rec.every
            self.text(f"next frame at update {nxt:,}" + (f" · until {rec.until_step:,}" if rec.until_step else ""),
                      (x + 150, y + 40), TEXT, self.font)
        self.button((x + 14, y + 66, 180, 26), "■ stop recording" if recording else "● start recording", "rec_toggle")
        self.button((x + 204, y + 66, 150, 26), "capture now (K)", "capture_now")
        rows = [("every", f"every {c['every']:,} updates", "rec:every"),
                ("until", "until stopped" if c["until"] is None else f"until update {c['until']:,}", "rec:until"),
                ("max", f"at most {c['max_frames']} frames", "rec:max")]
        yy = y + 106
        for _, label, act in rows:
            self.button((x + 14, yy, 24, 22), "−", f"{act}:-1")
            self.text(label, (x + 48, yy + 4), TEXT, self.font)
            self.button((x + 250, yy, 24, 22), "+", f"{act}:1")
            yy += 30
        self.toggle((x + 14, yy + 2, 200, 24), "dashboard panels (PNG)", c["panels"], "rec:panels")
        yy += 40
        frames_dir = wk.frames_dir or "-"
        n = len(rec.captured) if rec else len(ex.FrameRecorder(frames_dir).existing_steps()) if wk.frames_dir and os.path.isdir(frames_dir) else 0
        key = ("disk", n, frames_dir)
        if key not in self._preview_cache:
            used = rec.disk_bytes() if rec else 0
            try:
                free = shutil_disk_free(os.path.dirname(os.path.abspath(frames_dir)) if frames_dir != "-" else ".")
            except OSError:
                free = float("nan")
            self._preview_cache = {k: v for k, v in self._preview_cache.items() if k[0] != "disk"}
            self._preview_cache[key] = (used, free)
        used, free = self._preview_cache[key]
        info = [("frames", f"{n} captured"), ("on disk", f"{used / 1e6:.1f} MB · {free / 1e9:.1f} GB free"),
                ("folder", frames_dir)]
        for i, (k, v) in enumerate(info):
            self.text(k.upper(), (x + 14, yy + i * 18 + 2), DIM, self.small)
            self.text(self.fit_text(v, w - 110, self.font), (x + 90, yy + i * 18), BAD if k == "on disk" and free < 1e9
                      else TEXT, self.font)
        self.text(self.fit_text("a frame = attention, logit lens, head scores, embeddings, block internals and the "
                                "latest evaluation at that update", w - 28, self.small), (x + 14, y + h - 22), FAINT,
                  self.small)

    def draw_frames(self, rect) -> None:
        pg, tr, wk = self.pg, self.tr, self.worker
        self.panel(rect, "Captured frames", "ticks on the training timeline · validation loss behind", key="frames")
        x, y, w, h = rect
        rec = wk.recorder
        steps = list(rec.captured) if rec else (ex.FrameRecorder(wk.frames_dir).existing_steps()
                                                 if wk.frames_dir and os.path.isdir(wk.frames_dir) else [])
        steps.sort()
        plot = (x + 48, y + 40, w - 70, 150)
        ev = tr.hist["eval"]
        xmax = max([tr.step, 1] + steps)
        lo = min((r["loss"] for r in ev), default=0.0) - 0.2
        hi = max((r["loss"] for r in ev), default=10.0) + 0.2
        X, Y = self.axes(plot, (0, xmax), (lo, hi), xfmt=fmt_tokens, yfmt=lambda v: f"{v:.1f}", nx=4, ny=3,
                         xlabel="update")
        if len(ev) > 1:
            self.polyline(X, Y, [r["step"] for r in ev], [r["loss"] for r in ev], mix_rgb(CARD, VAL_C, 0.6), 1)
        for s_ in steps:
            xx = int(X(s_))
            pg.draw.line(self.canvas, LR_C, (xx, plot[1]), (xx, plot[1] + plot[3]), 2)
        cx = int(X(tr.step))
        pg.draw.line(self.canvas, INK, (cx, plot[1] - 4), (cx, plot[1] + plot[3] + 4))
        yy = plot[1] + plot[3] + 36
        for k_, hd in zip((x + 14, x + 120, x + 230, x + 340), ("UPDATE", "TOKENS", "VAL LOSS", "PANELS")):
            self.text(hd, (k_, yy), DIM, self.small)
        tpu = tr.tokens_per_step
        for i, s_ in enumerate(reversed(steps[-12:])):
            ry = yy + 18 + i * 18
            if ry > y + h - 24:
                break
            prior = [r for r in ev if r["step"] <= s_]
            panels_dir = os.path.join(wk.frames_dir or "", f"step_{s_:07d}", "panels")
            n_png = len(os.listdir(panels_dir)) if os.path.isdir(panels_dir) else 0
            self.text(f"{s_:,}", (x + 14, ry), TEXT, self.font)
            self.text(fmt_tokens(s_ * tpu), (x + 120, ry), TEXT, self.font)
            self.text(f"{prior[-1]['loss']:.4f}" if prior else "-", (x + 230, ry), TEXT, self.font)
            self.text(f"{n_png} PNG" if n_png else "-", (x + 340, ry), DIM, self.font)
        if not steps:
            self.text("no frames yet: start recording or capture now", (x + 14, yy + 20), FAINT, self.small)

    def draw_export_panel(self, rect) -> None:
        pg = self.pg
        c, st = self.export_cfg, self.export_state
        self.panel(rect, "Appendix export", "editorial figures, strips, data sheets and cards", key="export_panel")
        x, y, w, h = rect
        yy = y + 38
        groups = [("THEME", [("print (manuscript)", "print", "theme"), ("slides 16:9", "slides", "theme")]),
                  ("WIDTH", [("89 mm", "single", "width"), ("120 mm", "onehalf", "width"), ("183 mm", "double", "width")]),
                  ("FORMATS", [("PNG 300 dpi", "png", "format"), ("PDF", "pdf", "format"), ("SVG", "svg", "format")]),
                  ("SECTIONS", [("A cards", "A", "section"), ("B curves", "B", "section"), ("C maps", "C", "section"),
                                ("D sheets", "D", "section"), ("E frames", "E", "section")])]
        for title, opts in groups:
            self.text(title, (x + 14, yy + 6), DIM, self.small)
            xx = x + 96
            for label, val, kind in opts:
                on = (val in c["themes"]) if kind == "theme" else (c["width"] == val) if kind == "width" else \
                     (val in c["formats"]) if kind == "format" else (val in c["sections"])
                bw = self.toggle((xx, yy, 60, 24), label, on, f"exp:{kind}:{val}")
                xx += bw + 6
            yy += 34
        self.button((x + 14, yy + 6, 200, 30), "exporting…" if st["running"] else "⇩ export appendix (ENTER)", "export")
        if st["out_dir"] and os.path.isdir(st["out_dir"]):
            self.button((x + 224, yy + 6, 140, 30), "open folder", "open_export")
        yy += 48
        pg.draw.rect(self.canvas, INK_FAINT, (x + 14, yy, w - 28, 6), 1)
        pg.draw.rect(self.canvas, GOOD_C if not st["error"] else BAD, (x + 15, yy + 1, int((w - 30) * st["progress"]), 4))
        self.text(self.fit_text(st["message"] or "not exported yet", w - 28, self.font), (x + 14, yy + 14),
                  BAD if st["error"] else TEXT, self.font)
        if st["out_dir"]:
            self.text(self.fit_text(st["out_dir"], w - 28, self.small), (x + 14, yy + 34), DIM, self.small)
        self.text(self.fit_text("print: white, journal column widths, 300 dpi + vector · slides: dashboard black, "
                                "1920×1080", w - 28, self.small), (x + 14, y + h - 22), FAINT, self.small)

    def draw_preview(self, rect) -> None:
        st = self.export_state
        prev = st["previews"]
        name = os.path.basename(prev[st["preview_idx"]]) if prev else ""
        self.panel(rect, "Preview", f"{st['preview_idx'] + 1} / {len(prev)} · {name} · ←/→" if prev else
                   "exported figures appear here", key="preview")
        x, y, w, h = rect
        if not prev:
            self.waiting(rect, "export the appendix to preview its figures")
            return
        path = prev[st["preview_idx"]]
        box = (x + 12, y + 34, w - 24, h - 46)
        key = (path, box[2], box[3])
        img = self._preview_cache.get(key)
        if img is None:
            try:
                raw = self.pg.image.load(path)
                iw, ih = raw.get_size()
                s_ = min(box[2] / iw, box[3] / ih)
                img = self.pg.transform.smoothscale(raw.convert() if self.pg.display.get_surface() else raw,
                                                    (max(1, int(iw * s_)), max(1, int(ih * s_))))
            except Exception:
                img = False
            self._preview_cache = {k: v for k, v in self._preview_cache.items() if k[0] == "disk"}
            self._preview_cache[key] = img
        if img:
            self.canvas.blit(img, (box[0] + (box[2] - img.get_width()) // 2, box[1] + (box[3] - img.get_height()) // 2))
            self.pg.draw.rect(self.canvas, INK_FAINT, (box[0] + (box[2] - img.get_width()) // 2 - 1,
                                                       box[1] + (box[3] - img.get_height()) // 2 - 1,
                                                       img.get_width() + 2, img.get_height() + 2), 1)
            for dx, label, d in ((box[0], "◀", -1), (box[0] + box[2] - 28, "▶", 1)):
                self.button((dx, box[1] + box[3] // 2 - 14, 28, 28), label, f"preview:{d}")

    # =========================================================================== #
    # View: Generate
    # =========================================================================== #
    def draw_generate(self, rect) -> None:
        x, y, w, h = rect
        self.draw_prompt((x, y, w, 132))
        self.draw_output((x, y + 140, 820, h - 140))
        sx, sw = x + 828, w - 828
        self.draw_candidates((sx, y + 140, sw, 400))
        self.draw_gen_charts((sx, y + 548, sw, h - 548))

    def draw_prompt(self, rect) -> None:
        pg = self.pg
        g = self.worker.gen
        st = self.gen_settings
        self.panel(rect, "Prompt", "type · ENTER generates · training is paused while this view is open",
                   key="prompt")
        x, y, w, h = rect
        box = (x + 14, y + 32, w - 28 - 300, 52)
        pg.draw.rect(self.canvas, CARD, box)
        pg.draw.rect(self.canvas, INK_DIM, box, 1)
        shown = self.prompt.replace("\n", "⏎")
        line = self.fit_text(shown[::-1], box[2] - 24, self.f_flow)[::-1] if self.tw(shown, self.f_flow) > box[2] - 24 else shown
        tw = self.text(line, (box[0] + 10, box[1] + 18), INK, self.f_flow)
        if int(time.time() * 2) % 2 == 0:
            pg.draw.line(self.canvas, INK, (box[0] + 12 + tw, box[1] + 14), (box[0] + 12 + tw, box[1] + 36), 1)
        n_tok = len(self.tr.data.tokenizer.encode(self.prompt)) if self.prompt else 0
        self.text(f"{n_tok} tokens", (box[0] + box[2] - 8, box[1] + 4), FAINT, self.small, align="right")
        bx = box[0] + box[2] + 10
        running = g is not None and g.running
        self.button((bx, box[1], 130, 24), "■ stop" if running else "▶ generate", "stop" if running else "generate")
        self.button((bx, box[1] + 28, 130, 24), "clear prompt", "clear")
        if g is not None:
            rate = len(g.tokens) / g.seconds if g.seconds else 0
            self.text(f"{len(g.tokens)}/{g.max_new} tokens", (bx + 140, box[1] + 4), TEXT, self.small)
            self.text(f"{rate:,.0f} tok/s · seed {g.seed}", (bx + 140, box[1] + 20), DIM, self.small)
            self.text(f"step {self.tr.step:,} weights", (bx + 140, box[1] + 36), DIM, self.small)
        sx, sy = x + 14, y + h - 34
        specs = [("temperature", f"temperature {st['temperature']:.2f}"),
                 ("top_k", f"top-k {st['top_k'] if st['top_k'] else 'off'}"),
                 ("top_p", f"top-p {st['top_p']:.2f}"), ("max_new", f"length {st['max_new']}"),
                 ("seed", f"seed {st['seed']}")]
        for name, label in specs:
            self.button((sx, sy, 22, 22), "−", f"set:{name}:-1")
            lw = max(110, self.tw(label, self.font) + 16)
            self.text(label, (sx + 28 + lw // 2 - 8, sy + 4), TEXT, self.font, align="center")
            self.button((sx + 28 + lw - 4, sy, 22, 22), "+", f"set:{name}:1")
            sx += 28 + lw + 34

    def draw_output(self, rect) -> None:
        g = self.worker.gen
        self.panel(rect, "Generated text", "colour = probability the model gave each sampled token", key="output")
        x, y, w, h = rect
        if g is None:
            self.text("press ENTER to sample a continuation of the prompt", (x + 14, y + 40), FAINT, self.small)
            return
        ids = list(g.prompt_ids[-60:]) + list(g.tokens)
        n_prompt = len(g.prompt_ids[-60:])
        cols = [None] * n_prompt + [tuple(int(v) for v in cmap_color(math.sqrt(i["p_model"]), "viridis")) for i in g.infos]
        tcols = [DIM] * n_prompt + [ink_on(c) for c in cols[n_prompt:]]
        self.flow(ids, (x + 14, y + 36, w - 28, h - 36 - 40), cols, text_colors=tcols, kind="gen",
                  start_index=-n_prompt, line_h=22,
                  outline={self.gen_hover} if self.gen_hover is not None else set())
        self.colorbar(x + 14, y + h - 24, 140, "viridis", "p 0", "1 (sqrt)")
        if g.infos:
            mean_lp = float(np.mean([math.log(max(i["p_model"], 1e-12)) for i in g.infos]))
            self.text(f"sample perplexity under the model {math.exp(-mean_lp):,.1f} · grey = prompt",
                      (x + w - 14, y + h - 26), DIM, self.small, align="right")

    def draw_candidates(self, rect) -> None:
        pg = self.pg
        g = self.worker.gen
        idx = self.hovered("gen")
        if idx is None or idx < 0:
            idx = self.gen_hover if (self.gen_hover is not None and self.gen_hover >= 0) else None
        if g is not None and g.infos and (idx is None or idx >= len(g.infos)):
            idx = len(g.infos) - 1
        self.panel(rect, "Candidates", None if idx is None else f"generated token {idx + 1}", key="candidates")
        x, y, w, h = rect
        if g is None or not g.infos or idx is None:
            self.waiting(rect, "no generation yet")
            return
        info = g.infos[idx]
        self.text(f"sampled {self.tok_label(info['token'])}", (x + 14, y + 32), LR_C, self.f_head)
        self.text(f"model p {info['p_model']:.2%} · after filtering {info['p_sample']:.2%} · entropy "
                  f"{info['entropy']:.2f} nats", (x + 14, y + 54), DIM, self.small)
        bar_x, bar_w = x + 140, w - 140 - 64
        for k, (t, p, ps) in enumerate(info["top"][:12]):
            ry = y + 78 + k * 24
            chosen = t == info["token"]
            col = LR_C if chosen else INK
            self.text(self.fit_text(self.tok_label(t), 120, self.font), (x + 14, ry), col, self.font)
            pg.draw.rect(self.canvas, INK_FAINT, (bar_x, ry + 5, bar_w, 6), 1)
            pg.draw.rect(self.canvas, VAL_C, (bar_x, ry + 5, max(1, int(bar_w * p)), 6))
            if ps > 0:
                xx = bar_x + int(bar_w * min(1.0, ps))
                pg.draw.line(self.canvas, LR_C, (xx, ry + 2), (xx, ry + 13), 2)
            self.text(f"{p:.1%}", (x + w - 14, ry), col, self.font, align="right")
        lx = x + 14
        lx += self.swatch(lx, y + h - 22, VAL_C, "model probability", "box") + 6
        self.swatch(lx, y + h - 22, LR_C, "sampling probability")

    def draw_gen_charts(self, rect) -> None:
        g = self.worker.gen
        self.panel(rect, "Per token", "entropy of the model's distribution · p(sampled)", key="gen_charts")
        x, y, w, h = rect
        if g is None or len(g.infos) < 2:
            self.waiting(rect, "no generation yet")
            return
        ent = [i["entropy"] for i in g.infos]
        pm = [i["p_model"] for i in g.infos]
        gw = w - 28
        gh = (h - 40 - 8) // 2
        self.mini((x + 14, y + 32, gw, gh), ent, "entropy (nats)", INK, 1, None, "{:.2f}")
        self.mini((x + 14, y + 32 + gh + 8, gw, gh), pm, "p(sampled token)", VAL_C, 1, None, "{:.2f}", fixed=(0, 1))
