from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..version import __version__ as PKG_VERSION


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit_short() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (res.stdout or "").strip()
        return out or None
    except Exception:
        return None


@dataclass
class RunManifest:
    env_id: str = "SpecKitAI/ColumnPopper-v1"
    seed: Optional[int] = None
    reward_preset: str = "default"
    created_at: str = field(default_factory=_now_iso)

    # Environment metadata
    pkg_version: str = field(default=PKG_VERSION)
    git_commit: Optional[str] = field(default_factory=_git_commit_short)
    python_version: str = field(default_factory=lambda: platform.python_version())
    os: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.version())

    # Free-form extras (e.g., CLI args)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunManifest":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "RunManifest":
        return cls.from_dict(json.loads(s))


def write_json(path: str, manifest: RunManifest) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
        f.write("\n")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")


__all__ = ["RunManifest", "write_json", "append_jsonl"]

