import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from openpilot.cereal import custom
from openpilot.sunnypilot.models import local, helpers
from openpilot.sunnypilot.models.fetcher import ModelFetcher, get_cached_bundles
from openpilot.sunnypilot.models.manager import ModelManagerSP


class TestLocalModels(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.root.joinpath("local").mkdir()
    self.path_patch = patch("openpilot.common.hardware.hw.Paths.model_root", return_value=str(self.root))
    self.path_patch.start()
    self.addCleanup(self.path_patch.stop)
    local._CACHE.clear()
    helpers._LAST_VALIDATED_RAW.clear()
    self.data = b"trusted locally compiled test artifact"
    self.sha = hashlib.sha256(self.data).hexdigest()
    self.name = "local-test.pkl"
    self.root.joinpath(self.name).write_bytes(self.data)
    self.manifest_path = self.root / "local" / "test.json"
    self.manifest = {
      "version": 1, "source": "chestnut", "target_arch": "gfx1102",
      "source_onnx_sha256": "a" * 64, "compiler_commit": "b" * 40,
      "bundle": {
        "index": 0, "internalName": "local-test", "displayName": "Local test (gfx1102)",
        "generation": 12, "runner": "tinygrad", "minimumSelectorVersion": 19,
        "ref": "local:" + self.sha, "is20hz": True,
        "models": [{"type": "supercombo", "artifact": {
          "fileName": self.name, "downloadUri": {"uri": "local://" + self.sha, "sha256": self.sha}}}],
      },
    }
    self.write_manifest()

  def write_manifest(self):
    self.manifest_path.write_text(json.dumps(self.manifest))

  def bundle(self):
    return custom.ModelManagerSP.ModelBundle(**self.manifest["bundle"])

  def test_registered_in_chestnut_only(self):
    self.assertEqual(local.get_local_bundles("qcom"), [])
    self.assertEqual(len(local.get_local_bundles("chestnut")), 1)
    self.assertTrue(local.is_registered_local_bundle(self.bundle()))

  def test_catalog_and_offline_cache_include_local(self):
    fetcher = ModelFetcher.__new__(ModelFetcher)
    with patch.object(fetcher, "_get_remote_bundles_for_source", return_value=[]):
      self.assertEqual(len(fetcher.get_bundles_for_source("chestnut")), 1)
    params = Mock()
    params.get.return_value = None
    self.assertEqual(len(get_cached_bundles(params, "chestnut")), 1)

  def test_selection_survives_catalog_refresh(self):
    bundle = self.bundle()
    self.assertFalse(helpers._bundle_needs_reset(bundle, local.get_local_bundles("chestnut")))
    params = Mock()
    params.get.return_value = bundle.to_dict()
    helpers._validate_active_bundle(params, "chestnut", local.get_local_bundles("chestnut"))
    params.remove.assert_not_called()
    params.put.assert_not_called()

  def test_corruption_invalidates_cache_and_selection(self):
    self.assertEqual(len(local.get_local_bundles("chestnut")), 1)
    params = Mock()
    params.get.return_value = self.bundle().to_dict()
    helpers._validate_active_bundle(params, "chestnut", local.get_local_bundles("chestnut"))
    self.root.joinpath(self.name).write_bytes(b"corrupt")
    self.assertEqual(local.get_local_bundles("chestnut"), [])
    helpers._validate_active_bundle(params, "chestnut", [])
    params.put.assert_called_once_with(helpers.ACTIVE_BUNDLE_KEYS["chestnut"], {}, block=True)

  def test_wrong_architecture_rejected_before_load(self):
    local.check_local_model_target(self.bundle(), "gfx1102")
    with self.assertRaises(ValueError):
      local.check_local_model_target(self.bundle(), "gfx1200")

  def test_tampered_selection_rejected(self):
    bundle = self.bundle()
    bundle.generation = 1
    self.assertFalse(local.is_registered_local_bundle(bundle))

  def test_missing_or_malformed_registration(self):
    for content in ["{", "{}", "null", "[]"]:
      self.manifest_path.write_text(content)
      self.assertEqual(local.get_local_bundles("chestnut"), [])
    self.manifest_path.unlink()
    self.assertEqual(local.get_local_bundles("chestnut"), [])

  def test_bad_paths_and_provenance_rejected(self):
    original = copy.deepcopy(self.manifest)
    for field, value in [("source", "qcom"), ("target_arch", "../gfx1102"), ("source_onnx_sha256", "bad"),
                         ("compiler_commit", "bad"), ("version", 2)]:
      self.manifest = copy.deepcopy(original)
      self.manifest[field] = value
      self.write_manifest()
      self.assertEqual(local.get_local_bundles("chestnut"), [])
    for name in ["../local-test.pkl", "/tmp/local-test.pkl", "local-../test.pkl", "official.pkl"]:
      self.manifest = copy.deepcopy(original)
      self.manifest["bundle"]["models"][0]["artifact"]["fileName"] = name
      self.write_manifest()
      self.assertEqual(local.get_local_bundles("chestnut"), [])

  def test_symlink_rejected(self):
    self.root.joinpath(self.name).unlink()
    target = self.root / "other.pkl"
    target.write_bytes(self.data)
    self.root.joinpath(self.name).symlink_to(target)
    self.assertEqual(local.get_local_bundles("chestnut"), [])

  def test_chunked_model_and_manifest_validation(self):
    self.root.joinpath(self.name).unlink()
    chunks = [self.data[:10], self.data[10:]]
    artifact = self.manifest["bundle"]["models"][0]["artifact"]
    artifact["chunks"] = []
    for i, data in enumerate(chunks):
      name = f"{self.name}.chunk{i+1:02d}of02"
      self.root.joinpath(name).write_bytes(data)
      artifact["chunks"].append({"fileName": name, "sha256": hashlib.sha256(data).hexdigest()})
    self.root.joinpath(self.name + ".chunkmanifest").write_text("2")
    self.write_manifest()
    self.assertEqual(len(local.get_local_bundles("chestnut")), 1)
    self.root.joinpath(self.name + ".chunkmanifest").write_text("3")
    self.assertEqual(local.get_local_bundles("chestnut"), [])

  def test_artifact_download_never_fetches_or_deletes_local(self):
    manager = ModelManagerSP.__new__(ModelManagerSP)
    manager.selected_bundle = None
    manager._download_start_times = {}
    manager._download_interrupted = Mock(return_value=False)
    manager._sync_artifact_progress = Mock()
    manager._report_status = Mock()
    artifact = self.bundle().models[0].artifact
    with patch("requests.get", side_effect=AssertionError("No network allowed")) as request:
      asyncio.run(manager._process_artifact(artifact, str(self.root)))
      self.root.joinpath(self.name).write_bytes(b"corrupt but preserved")
      with self.assertRaises(ValueError):
        asyncio.run(manager._process_artifact(artifact, str(self.root)))
      request.assert_not_called()
    self.assertEqual(self.root.joinpath(self.name).read_bytes(), b"corrupt but preserved")

  def test_cache_clear_preserves_unselected_local_files(self):
    manager = ModelManagerSP.__new__(ModelManagerSP)
    manager.params = Mock()
    manager.params.get.return_value = None
    self.root.joinpath("remote-obsolete.pkl").write_bytes(b"old download")
    manager.clear_model_cache()
    self.assertTrue(self.root.joinpath(self.name).exists())
    self.assertTrue(self.manifest_path.exists())
    self.assertFalse(self.root.joinpath("remote-obsolete.pkl").exists())
