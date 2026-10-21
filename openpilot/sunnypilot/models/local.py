"""Opt-in, hash-pinned local Chestnut builds. Registration never downloads code.

Manifests in <model_root>/local/*.json are trusted local configuration, not a
remote catalog. Like any model pickle, register only builds you trust.
"""
import hashlib
import json
from pathlib import Path
import re

from openpilot.common.file_chunker import get_chunk_name
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.cereal import custom

LOCAL_URI = "local://"
_CACHE: dict[str, tuple[tuple, dict]] = {}


def _sha(value):
  if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
    raise ValueError("Expected lowercase SHA256")
  return value


def _file(root, name):
  if not isinstance(name, str) or not re.fullmatch(r"local-[A-Za-z0-9_.-]+", name) or ".." in name:
    raise ValueError("Local artifacts must have safe local- filenames")
  path = root / name
  if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
    raise ValueError("Local artifact missing or not a regular in-root file")
  return path


def _signature(paths):
  return tuple((str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
               for p in paths for s in [p.stat()])


def _identity(bundle):
  data = bundle.to_dict()
  data.pop("status", None)
  data.pop("index", None)
  for model in data.get("models", []):
    model["artifact"].pop("downloadProgress", None)
  return data


def _read_manifest(path, root):
  from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION

  if path.is_symlink() or path.stat().st_size > 128 * 1024:
    raise ValueError("Invalid local manifest")
  manifest = json.loads(path.read_text())
  if manifest["version"] != 1 or manifest["source"] != "chestnut" or not re.fullmatch(r"gfx[0-9a-f]+", manifest["target_arch"]):
    raise ValueError("Unsupported local manifest target")
  _sha(manifest["source_onnx_sha256"])
  if not re.fullmatch(r"[0-9a-f]{40}", manifest["compiler_commit"]):
    raise ValueError("Missing compiler provenance")
  bundle = custom.ModelManagerSP.ModelBundle(**manifest["bundle"])
  if bundle.runner != custom.ModelManagerSP.Runner.tinygrad or bundle.minimumSelectorVersion != REQUIRED_JSON_VERSION:
    raise ValueError("Incompatible local model runner or selector")
  if not bundle.internalName.startswith("local-") or len(bundle.models) != 1 or bundle.models[0].type != custom.ModelManagerSP.Model.Type.supercombo:
    raise ValueError("Expected one local supercombo model")
  artifact = bundle.models[0].artifact
  full_hash = _sha(artifact.downloadUri.sha256)
  if bundle.ref != f"local:{full_hash}" or artifact.downloadUri.uri != LOCAL_URI + full_hash:
    raise ValueError("Local identity must be pinned to the artifact SHA256")
  # Resolve and validate every filename before parsing or touching a model.
  name = artifact.fileName
  if not re.fullmatch(r"local-[A-Za-z0-9_-]+\.pkl", name):
    raise ValueError("Invalid local pickle filename")
  expected = []
  paths = [path]
  if artifact.chunks:
    if len(artifact.chunks) > 100:
      raise ValueError("Too many chunks")
    if (root / name).exists():
      raise ValueError("Ambiguous flat and chunked artifact")
    chunk_manifest = _file(root, name + ".chunkmanifest")
    if chunk_manifest.read_text().strip() != str(len(artifact.chunks)):
      raise ValueError("Chunk manifest does not match registration")
    paths.append(chunk_manifest)
    for i, chunk in enumerate(artifact.chunks):
      chunk_name = get_chunk_name(name, i, len(artifact.chunks))
      if chunk.fileName != chunk_name:
        raise ValueError("Noncanonical local chunk name")
      expected.append((_file(root, chunk_name), _sha(chunk.sha256)))
  else:
    if (root / (name + ".chunkmanifest")).exists():
      raise ValueError("Ambiguous flat and chunked artifact")
    expected.append((_file(root, name), full_hash))
  paths.extend(p for p, _ in expected)
  signature = _signature(paths)
  cached = _CACHE.get(str(path))
  if cached is None or cached[0] != signature:
    combined = hashlib.sha256()
    for file, digest in expected:
      individual = hashlib.sha256()
      with file.open("rb") as stream:
        while block := stream.read(1024 * 1024):
          individual.update(block)
          combined.update(block)
      if individual.hexdigest() != digest:
        raise ValueError("Local chunk hash mismatch")
    if combined.hexdigest() != full_hash or _signature(paths) != signature:
      raise ValueError("Local artifact hash mismatch or changed during verification")
    _CACHE[str(path)] = (signature, manifest)
  return bundle, manifest


def get_local_bundles(source):
  if source != "chestnut":
    return []
  root = Path(Paths.model_root())
  directory = root / "local"
  if directory.is_symlink():
    return []
  bundles = []
  refs = set()
  for path in sorted(directory.glob("*.json")):
    try:
      bundle, _ = _read_manifest(path, root)
      if bundle.ref in refs:
        raise ValueError("Duplicate local model identity")
      refs.add(bundle.ref)
      bundles.append(bundle)
    except Exception as error:
      _CACHE.pop(str(path), None)
      cloudlog.warning(f"Ignoring local model registration {path.name}: {error}")
  return bundles


def is_registered_local_bundle(bundle):
  return any(_identity(bundle) == _identity(registered) for registered in get_local_bundles("chestnut"))


def check_local_model_target(bundle, arch):
  """Check before unpickling, with the architecture detected by the AMD runtime."""
  if not is_registered_local_bundle(bundle):
    raise ValueError("Local model is missing, changed or unregistered")
  root = Path(Paths.model_root())
  for path in sorted((root / "local").glob("*.json")):
    try:
      registered, manifest = _read_manifest(path, root)
    except Exception:
      continue
    if _identity(bundle) == _identity(registered):
      if manifest["target_arch"] != arch:
        raise ValueError(f"Local model targets {manifest['target_arch']}, GPU is {arch}")
      return
  raise ValueError("Local model registration not found")
