"""Global configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("data")
    asset_subdir: str = "input"
    filename_pattern: str = r"^(?P<asset>[^_]+)_(?P<start>\d{8})_(?P<end>\d{8})\.parquet$"
    timestamp_col: str = "timestamp"
    asset_col: str = "asset_id"
    class_col: str = "class"
    output_dir: Path = Path("outputs")
    random_state: int = 42
    extras: dict = field(default_factory=dict)

    def asset_dir(self, asset_id: str) -> Path:
        return self.data_root / asset_id / self.asset_subdir
