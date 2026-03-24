from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from datasets import DatasetDict
from torch.utils.data import DataLoader

from .data import build_dataloaders


@dataclass(frozen=True)
class MemoryStats:
    mem_total: int
    mem_available: int
    swap_total: int
    swap_used: int


def read_memory_stats() -> MemoryStats | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None

    values: dict[str, int] = {}
    with meminfo_path.open("r", encoding="utf-8") as f:
        for line in f:
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                values[key] = int(parts[0]) * 1024
            except ValueError:
                continue

    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    if any(key not in values for key in required):
        return None

    swap_total = values["SwapTotal"]
    swap_free = values["SwapFree"]
    return MemoryStats(
        mem_total=values["MemTotal"],
        mem_available=values["MemAvailable"],
        swap_total=swap_total,
        swap_used=max(0, swap_total - swap_free),
    )


class DynamicBatchController:
    def __init__(
        self,
        runtime_data_cfg: dict,
        ds: DatasetDict,
        logger: Logger,
        device: str,
        shuffle: bool,
    ) -> None:
        self.dataset = ds
        self.logger = logger
        self.device = device
        self.shuffle = shuffle
        self.num_workers = int(runtime_data_cfg.num_workers)
        self.batch_size = max(1, int(runtime_data_cfg.batch_size))

        dynamic_cfg = runtime_data_cfg.get("dynamic_batch", {})
        self.enabled = bool(dynamic_cfg.get("enabled", False))
        self.min_batch_size = min(
            self.batch_size,
            max(1, int(dynamic_cfg.get("min_batch_size", 1))),
        )
        self.reduce_factor = float(dynamic_cfg.get("reduce_factor", 0.5))
        if self.reduce_factor <= 0.0 or self.reduce_factor >= 1.0:
            self.reduce_factor = 0.5
        self.min_available_ratio = float(dynamic_cfg.get("min_available_ratio", 0.10))
        self.max_swap_growth_bytes = int(
            float(dynamic_cfg.get("max_swap_growth_mb", 128)) * 1024 * 1024
        )
        self._supported = True

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        return build_dataloaders(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            device=self.device,
        )

    def checkpoint_meta(self) -> dict[str, int]:
        return {"batch_size": self.batch_size}

    def load_checkpoint_meta(self, meta: dict) -> tuple[DataLoader, DataLoader] | None:
        loaded_batch_size = meta.get("batch_size", None)
        if loaded_batch_size is None:
            return None

        new_batch_size = max(1, int(loaded_batch_size))
        if new_batch_size == self.batch_size:
            return None

        self.batch_size = new_batch_size
        return self.build_dataloaders()

    def sample_memory(self) -> MemoryStats | None:
        if not self.enabled or not self._supported:
            return None

        stats = read_memory_stats()
        if stats is None:
            self._supported = False
            self.logger.warning(
                "Dynamic batch sizing is enabled but /proc/meminfo is unavailable. "
                "Keeping a fixed batch size of %d.",
                self.batch_size,
            )
            return None
        return stats

    def maybe_reduce_after_epoch(
        self,
        start_stats: MemoryStats | None,
        end_stats: MemoryStats | None,
    ) -> tuple[DataLoader, DataLoader] | None:
        if not self.enabled or start_stats is None or end_stats is None:
            return None

        reasons: list[str] = []
        available_ratio = end_stats.mem_available / max(end_stats.mem_total, 1)
        if available_ratio < self.min_available_ratio:
            reasons.append(
                f"available memory {available_ratio:.1%} below {self.min_available_ratio:.1%}"
            )

        swap_growth = max(0, end_stats.swap_used - start_stats.swap_used)
        if swap_growth >= self.max_swap_growth_bytes:
            reasons.append(f"swap grew by {swap_growth / (1024 * 1024):.1f} MB")

        if not reasons:
            return None

        if self.batch_size <= self.min_batch_size:
            self.logger.warning(
                "Memory pressure detected (%s), but batch size is already at the minimum of %d.",
                "; ".join(reasons),
                self.batch_size,
            )
            return None

        new_batch_size = max(
            self.min_batch_size,
            int(self.batch_size * self.reduce_factor),
        )
        if new_batch_size >= self.batch_size:
            new_batch_size = self.batch_size - 1
        new_batch_size = max(self.min_batch_size, new_batch_size)

        if new_batch_size == self.batch_size:
            return None

        old_batch_size = self.batch_size
        self.batch_size = new_batch_size
        self.logger.warning(
            "Reducing batch size from %d to %d for the next epoch (%s).",
            old_batch_size,
            new_batch_size,
            "; ".join(reasons),
        )
        return self.build_dataloaders()
