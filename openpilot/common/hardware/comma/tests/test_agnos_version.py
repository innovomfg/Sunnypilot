import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[5]

# Only substitute hardware identity reads. All policy and updater shell code
# runs unchanged, with a NUL-terminated device-tree model as on the device.
MOCK_IDENTITY = r'''
cat() {
  case "$1" in
    /VERSION)
      [ "$MOCK_OS_VERSION" != missing ] || return 1
      printf '%s\n' "$MOCK_OS_VERSION" ;;
    /sys/firmware/devicetree/base/model)
      [ "$MOCK_DEVICE_MODEL" != missing ] || return 1
      printf '%s\0' "$MOCK_DEVICE_MODEL" ;;
    *) command cat "$@" ;;
  esac
}
'''


class TestAgnosVersion(unittest.TestCase):
  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(self.tempdir.cleanup)
    identity = Path(self.tempdir.name) / "identity.sh"
    identity.write_text(MOCK_IDENTITY)
    self.env = {
      "BASH_ENV": str(identity),
      "MOCK_OS_VERSION": "19.8.1",
      "MOCK_DEVICE_MODEL": "comma mici",
    }

  def target_version(self, version, model, override=None, staged=False):
    env = {**os.environ, **self.env, "MOCK_OS_VERSION": version, "MOCK_DEVICE_MODEL": model}
    env.pop("AGNOS_VERSION", None)
    if override is not None:
      env["AGNOS_VERSION"] = override
    script = ('unset AGNOS_VERSION; ' if staged else '') + 'source launch_env.sh; printf "%s" "$AGNOS_VERSION"'
    return subprocess.check_output(["bash", "-euc", script], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, text=True)

  def test_narrow_hardware_and_version_policy(self):
    cases = [
      ("19.8.1", "comma mici", "19.8.1"),
      ("19.7", "comma mici", "19.7"),
      ("19.8.2", "comma mici", "19.7"),
      ("19.9", "comma mici", "19.7"),
      ("19.8.1", "comma tici", "19.7"),
      ("19.8.1", "comma tizi", "19.7"),
      ("19.8.1", "missing", "19.7"),
      ("missing", "comma mici", "19.7"),
    ]
    for version, model, target in cases:
      with self.subTest(version=version, model=model):
        self.assertEqual(self.target_version(version, model), target)

  def test_existing_explicit_override_is_preserved(self):
    self.assertEqual(self.target_version("19.8.1", "comma mici", override="19.9"), "19.9")
    self.assertEqual(self.target_version("19.8.1", "comma mici", override=""), "19.8.1")

  def test_staged_activation_recomputes_policy(self):
    self.assertEqual(self.target_version("19.8.1", "comma mici", override="19.7", staged=True), "19.8.1")
    self.assertEqual(self.target_version("19.8.2", "comma mici", override="19.8.1", staged=True), "19.7")

  def test_background_updater_uses_staged_policy_without_flashing(self):
    from openpilot.system.updated import updated

    def run(cmd, cwd=None):
      # Do not inherit an SSH socket as stdin. Bash would then source .bashrc
      # instead of BASH_ENV, bypassing the mocked identity reads.
      return subprocess.check_output(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT, encoding="utf8")

    # Run the actual updater entry point and its actual shell subprocess.
    # The staged tree has the same configuration as this checkout. All flash,
    # slot selection, and staging-marker writes are mocked, never performed.
    for version, model, should_flash in [
      ("19.8.1", "comma mici", False),
      ("19.7", "comma mici", False),
      ("19.8.2", "comma mici", True),
      ("19.8.1", "comma tizi", True),
    ]:
      with self.subTest(version=version, model=model), \
           patch.dict(os.environ, {**self.env, "MOCK_OS_VERSION": version, "MOCK_DEVICE_MODEL": model,
                                   "AGNOS_VERSION": "inherited-value-must-be-ignored"}), \
           patch.object(updated, "OVERLAY_MERGED", str(ROOT)), \
           patch.object(updated, "run", side_effect=run), \
           patch.object(updated.HARDWARE, "get_os_version", return_value=version), \
           patch.object(updated, "set_consistent_flag") as consistent, \
           patch("openpilot.common.hardware.comma.agnos.get_target_slot_number", return_value=1) as target_slot, \
           patch("openpilot.common.hardware.comma.agnos.flash_agnos_update") as flash:
        updated.handle_agnos_update()
        if should_flash:
          consistent.assert_called_once_with(False)
          target_slot.assert_called_once_with()
          flash.assert_called_once_with(str(ROOT / "openpilot/system/hardware/comma/agnos.json"), 1, updated.cloudlog)
        else:
          consistent.assert_not_called()
          target_slot.assert_not_called()
          flash.assert_not_called()


if __name__ == "__main__":
  unittest.main()
