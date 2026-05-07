"""Profiler de regiões com torch.cuda.synchronize() obrigatório.

PAPER_CONTEXT Seção 6 e Seção 10 item 1: sem sync antes/depois mede tempo
de enfileirar (~us), não de executar (~ms). Esta classe sincroniza por
padrão. Custo: 2 syncs por região; em ~1700 entradas/run em 100s, overhead
fica em ~0.5%.

Duas APIs:
- `with profiler.region("name"):`  — context manager; ideal para halo e
  blocos pequenos.
- `profiler.start("name")` / `profiler.end("name")` — pares explícitos
  para evitar reindentação em funções longas (forward()).

NVTX é opcional (use_nvtx=True) para visualização em nsys/Nsight Systems.

Uso típico:
    from profiling import profiler
    profiler.enable(sync=True, use_nvtx=False)
    ...
    profiler.dump_csv(path, extra_cols={"rank": rank, "strategy": "async_b"})
"""

import csv
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch

try:
    import torch.cuda.nvtx as _nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False


class Profiler:
    def __init__(self):
        self.enabled = False
        self.sync = True
        self.use_nvtx = False
        self._stats = OrderedDict()  # name -> {"count": int, "time": float}
        self._open = {}  # name -> start_t (para start/end)
        # Pilha de estágios atualmente abertos (push_stage/pop_stage).
        # Permite atribuir tempo de halo ao estágio onde ocorreu:
        # halo_total dentro de "multigrid" também acumula em halo_in_multigrid.
        self._stage_stack = []

    def enable(self, sync=True, use_nvtx=False):
        self.enabled = True
        self.sync = sync
        self.use_nvtx = use_nvtx and _HAS_NVTX

    def disable(self):
        self.enabled = False

    def reset(self):
        self._stats.clear()
        self._open.clear()
        self._stage_stack.clear()

    def _sync(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record(self, name, dt):
        entry = self._stats.get(name)
        if entry is None:
            entry = {"count": 0, "time": 0.0}
            self._stats[name] = entry
        entry["count"] += 1
        entry["time"] += dt

    @contextmanager
    def region(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        if self.use_nvtx:
            _nvtx.range_push(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            t1 = time.perf_counter()
            if self.use_nvtx:
                _nvtx.range_pop()
            dt = t1 - t0
            self._record(name, dt)
            # Atribui o tempo desta região ao estágio em que ela ocorreu.
            # Útil para halo_total dentro de multigrid/predictor/etc:
            # gera a contrapartida halo_in_<estagio>.
            if name == "halo_total" and self._stage_stack:
                self._record(f"halo_in_{self._stage_stack[-1]}", dt)

    def start(self, name):
        if not self.enabled:
            return
        self._sync()
        if self.use_nvtx:
            _nvtx.range_push(name)
        self._open[name] = time.perf_counter()

    def end(self, name):
        if not self.enabled:
            return
        if name not in self._open:
            return
        self._sync()
        t1 = time.perf_counter()
        if self.use_nvtx:
            _nvtx.range_pop()
        dt = t1 - self._open.pop(name)
        self._record(name, dt)

    def push_stage(self, name):
        """Como start(), mas também empilha o estágio para atribuição de halos.

        Halos chamados dentro do bloco também acumulam em halo_in_<name>.
        Use este par para os 5 estágios do forward (predictor, corrector,
        pre_pressure, multigrid, post_pressure).
        """
        if not self.enabled:
            return
        self.start(name)
        self._stage_stack.append(name)

    def pop_stage(self, name):
        if not self.enabled:
            return
        self.end(name)
        if self._stage_stack and self._stage_stack[-1] == name:
            self._stage_stack.pop()

    def report(self):
        out = []
        for name, e in self._stats.items():
            count = e["count"]
            total = e["time"]
            mean = total / count if count else 0.0
            out.append((name, count, total, mean))
        return out

    def dump_csv(self, path, extra_cols=None):
        extra = dict(extra_cols or {})
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(extra.keys()) + ["region", "count", "time_s", "time_per_call_us"])
            for name, count, total_s, mean_s in self.report():
                w.writerow(
                    list(extra.values())
                    + [name, count, f"{total_s:.6f}", f"{mean_s * 1e6:.3f}"]
                )


profiler = Profiler()
