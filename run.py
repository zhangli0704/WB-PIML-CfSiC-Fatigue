"""Run the WB-PIML revision workflow."""
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import os

ROOT = Path(__file__).resolve().parent


def canonical_source():
    """Load the ordered, hash-verified source parts."""
    manifest = json.loads((ROOT / 'wb_piml_parts/source_manifest.json').read_text(encoding='utf-8'))
    if len(manifest['parts']) != 12:
        raise RuntimeError('This release requires twelve source parts.')
    fragments = []
    for part in manifest['parts']:
        payload = (ROOT / 'wb_piml_parts' / part['file']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != part['sha256']:
            raise RuntimeError('Source integrity mismatch: ' + part['file'])
        fragments.append(payload)
    source = b''.join(fragments)
    digest = hashlib.sha256(source).hexdigest()
    if digest != manifest['canonical_source_sha256'] or len(source) != manifest['canonical_source_byte_count']:
        raise RuntimeError('The fragments do not reconstruct the validated analysis source.')
    directory = ROOT / '.wb_piml_runtime' / digest
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / manifest['canonical_source_filename']
    if not target.exists() or target.read_bytes() != source:
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(source)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def load_analysis():
    """Import the complete analysis without starting training."""
    source = canonical_source()
    location = str(source.parent)
    if location not in sys.path:
        sys.path.insert(0, location)
    return importlib.import_module(source.stem)


def _has_option(arguments, option):
    return any(item == option or item.startswith(option + '=') for item in arguments)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    source = canonical_source()
    # Relocate only default paths, preserving every numerical setting.
    external = _has_option(arguments, '--predict') or _has_option(arguments, '--evaluate')
    if not external and not _has_option(arguments, '--data'):
        arguments += ['--data', str(ROOT / 'data.xlsx')]
    if not _has_option(arguments, '--out'):
        mode = 'predict' if _has_option(arguments, '--predict') else 'evaluate' if _has_option(arguments, '--evaluate') else 'full'
        arguments += ['--out', str(ROOT / 'runs' / (mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))]
    old_argv = sys.argv
    try:
        sys.argv = [str(source), *arguments]
        runpy.run_path(str(source), run_name='__main__')
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
