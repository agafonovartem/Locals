"""Sliding-window KV caches that can be rewound (Flint's 35 sliding-window layers).

mlx-lm keeps a sliding-window layer in a ring of exactly the window (RotatingKVCache; BatchRotatingKVCache for a
batch). Once the ring is full a token cannot be taken back, so the batch engine's prompt-lookup drafts, verified in
one pass with the rejected part rewound, were off for any model with such a layer.

Here a layer keeps ``SLACK`` tokens more than its window; the attention mask still sees only the window (the model
passes ``window_size`` to it), and the last ``SLACK`` tokens can be dropped. One request's cache is mlx-lm's ring
with that size (``WindowKVCache``); a batch's is a ``BatchKVCache``, which trims, whose oldest slots are dropped once
``DROP`` more than ``window + SLACK`` have gathered (``WindowBatchKVCache``), so its memory stays bounded as the
ring's: a 513-token window takes at most 513 + 64 + 64 slots. ``offset`` stays each row's absolute position (RoPE's),
the buffer's slots are ``_idx``.
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, RotatingKVCache

SLACK = 64  # tokens a sliding-window layer can take back: a draft of up to 16, the token after it, room to spare
DROP = 64  # slots a batch's window cache gathers past window + SLACK before it drops its oldest (one copy each time)


class WindowKVCache(RotatingKVCache):
    """One request's sliding-window layer: mlx-lm's ring of ``window + SLACK`` slots, merged into a batch as a
    WindowBatchKVCache. (Its trimming stays mlx-lm's, possible until the ring is full: the prompt cache trims every
    layer by the same count, and a ring that wrapped can give back only part of it. Drafts are rewound in the
    batch's cache, which trims.)"""

    def __init__(self, window: int, keep: int = 0, max_size: int = None):
        super().__init__(max_size if max_size is not None else window + SLACK, keep)

    @property
    def window(self) -> int:  # (from max_size: a cache loaded from disk is built without __init__)
        return self.max_size - SLACK

    @classmethod
    def merge(cls, caches):
        return WindowBatchKVCache.merge(caches)


class WindowBatchKVCache(BatchKVCache):
    """A batch's sliding-window layer: linear (it trims), its oldest slots dropped in chunks."""

    def __init__(self, left_padding: List[int], max_size: int):
        super().__init__(left_padding)
        self.max_size = max_size  # window + SLACK: the slots always kept

    @property
    def window(self) -> int:
        return self.max_size - SLACK

    def _drop_oldest(self):
        d = self._idx - self.max_size
        if d <= 0:
            return
        self.keys = self.keys[..., d : self._idx, :]
        self.values = self.values[..., d : self._idx, :]
        self._idx -= d
        self.left_padding = mx.maximum(self.left_padding - d, 0)

    def update_and_fetch(self, keys, values):
        # (not while rows are right-padded in a prompt read together: a shorter row's tokens are its first slots)
        if self.keys is not None and self._idx >= self.max_size + DROP and getattr(self, "_right_padding", None) is None:
            self._drop_oldest()
        return super().update_and_fetch(keys, values)

    def extract(self, idx):
        padding = max(int(self.left_padding.tolist()[idx]), self._idx - self.max_size)
        c = WindowKVCache(self.window)
        c.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        c.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        c.offset = int(self.offset.tolist()[idx])
        c._idx = c.keys.shape[2]  # in temporal order: the ring's next write rotates from there
        return c

    @classmethod
    def merge(cls, caches):
        """Rings (WindowKVCache) into one batch: each row's tokens in temporal order, left-padded to the longest;
        ``offset`` each row's position."""
        rows = []
        for c in caches:
            if c.keys is None:
                rows.append(None)
                continue
            n = c.size()
            rows.append((c._temporal_order(c.keys)[..., -n:, :], c._temporal_order(c.values)[..., -n:, :]))
        lengths = [0 if r is None else r[0].shape[2] for r in rows]
        L = max(lengths)
        padding = [L - n for n in lengths]
        out = cls(padding, caches[0].max_size)
        out.offset = mx.array([c.offset for c in caches])
        if L == 0:
            return out
        k0 = next(r[0] for r in rows if r is not None)
        v0 = next(r[1] for r in rows if r is not None)
        B, H = len(caches), k0.shape[1]
        keys = mx.zeros((B, H, L, k0.shape[3]), dtype=k0.dtype)
        values = mx.zeros((B, H, L, v0.shape[3]), dtype=v0.dtype)
        for i, (p, r) in enumerate(zip(padding, rows)):
            if r is not None:
                keys[i : i + 1, :, p:] = r[0]
                values[i : i + 1, :, p:] = r[1]
        out.keys, out.values, out._idx = keys, values, L
        return out


def window_caches(model) -> None:
    """A Flint model's sliding-window layers in WindowKVCache (its make_cache wrapped: whatever it gives the full
    layers stays, 8-bit ones included)."""
    register()
    base = model.make_cache

    def make_cache():
        return [WindowKVCache(c.max_size) if type(c) is RotatingKVCache else c for c in base()]  # (max_size: the window)

    model.make_cache = make_cache


def register() -> None:
    """Make WindowKVCache loadable by mlx-lm's load_prompt_cache, which finds a cache's class by name in its module
    (the server's prompt snapshots on disk)."""
    import mlx_lm.models.cache as mc

    if getattr(mc, "WindowKVCache", None) is not WindowKVCache:
        mc.WindowKVCache = WindowKVCache
        mc.WindowBatchKVCache = WindowBatchKVCache
