"""Compatibility entry point for the byte-preserving split WB-PIML source."""
from pathlib import Path as _SplitPath
import hashlib as _split_hashlib
import json as _split_json

_split_root = _SplitPath(__file__).resolve().parent
_split_manifest = _split_json.loads(
    (_split_root / "wb_piml_parts" / "source_manifest.json").read_text(encoding="utf-8")
)
for _split_part in _split_manifest["parts"]:
    _split_path = _split_root / "wb_piml_parts" / _split_part["file"]
    _split_bytes = _split_path.read_bytes()
    _split_actual = _split_hashlib.sha256(_split_bytes).hexdigest()
    if _split_actual != _split_part["sha256"]:
        raise RuntimeError(f"Split-source integrity mismatch: {_split_part['file']}")
    exec(compile(_split_bytes, str(_split_path), "exec"), globals(), globals())
