import ast
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from openpilot.sunnypilot.modeld_v2.startup import load_model, wait_for_gpu_boot, gpu_link_ready


class TestStartup(unittest.TestCase):
  def test_drop_before_constructor_and_warmup(self):
    order = []
    model = Mock()
    model.warmup.side_effect = lambda: order.append("warmup")
    factory = Mock(side_effect=lambda **kw: order.append("construct") or model)
    with patch("openpilot.sunnypilot.modeld_v2.startup.drop_realtime", side_effect=lambda: order.append("normal")):
      self.assertIs(load_model(factory, chestnut=True), model)
    self.assertEqual(order, ["normal", "construct", "warmup"])
    factory.assert_called_once_with(chestnut=True)

  def test_background_worker_drops_inherited_policy(self):
    with patch("openpilot.sunnypilot.modeld_v2.startup.drop_realtime") as drop:
      thread = threading.Thread(target=load_model, args=(Mock(),))
      thread.start()
      thread.join(5)
      self.assertFalse(thread.is_alive())
      drop.assert_called_once_with()

  def test_failure_does_not_enter_realtime(self):
    with patch("openpilot.sunnypilot.modeld_v2.startup.drop_realtime") as drop:
      with self.assertRaises(RuntimeError):
        load_model(Mock(side_effect=RuntimeError("bad model")))
      drop.assert_called_once_with()

  def test_main_scheduling_order_without_running_hardware(self):
    tree = ast.parse((Path(__file__).parents[1] / "modeld.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    self.assertEqual(main.body[0].value.func.id, "drop_realtime")
    realtime = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "config_realtime_process"]
    self.assertEqual(len(realtime), 1)
    self.assertIsInstance(main.body[-1], ast.While)
    self.assertEqual(main.body[-2].value, realtime[0])
    self.assertEqual([n.value for n in realtime[0].args], [7, 54])
    constructors = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ModelState"]
    self.assertEqual(constructors, [], "All startup models must use the normal-priority warmup helper")

  def test_import_does_not_open_any_gpu(self):
    # A Device.DEFAULT function default previously opened AMD just by importing
    # compile_modeld, before the modeld startup grace/readiness gate could run.
    package_path = str(Path(__file__).parents[1])
    code = f'''
from unittest.mock import patch
import openpilot.sunnypilot.modeld_v2 as package
package.__path__.insert(0, {package_path!r})
from tinygrad.device import Device
with patch.object(type(Device), '__getitem__', side_effect=AssertionError('premature device initialization')) as opened:
  import openpilot.sunnypilot.modeld_v2.modeld
  opened.assert_not_called()
'''
    subprocess.run([sys.executable, '-c', code], check=True, stdin=subprocess.DEVNULL, timeout=30)


class FakeClock:
  def __init__(self):
    self.now = 0.

  def __call__(self):
    return self.now

  def sleep(self, seconds):
    self.now += seconds


class TestGpuBootWait(unittest.TestCase):
  def setUp(self):
    self.clock = FakeClock()

  def wait(self, ready, **kwargs):
    return wait_for_gpu_boot(ready, clock=self.clock, sleep=self.clock.sleep, **kwargs)

  def test_ready_gpu_still_gets_full_grace_and_stable_period(self):
    probes = []
    self.assertTrue(self.wait(lambda: probes.append(self.clock()) or True))
    self.assertGreaterEqual(min(probes), 20.)
    self.assertEqual(self.clock(), 22.)

  def test_late_gpu_enumeration(self):
    self.assertTrue(self.wait(lambda: self.clock() >= 35.))
    self.assertEqual(self.clock(), 37.)

  def test_absent_or_unpowered_gpu_times_out(self):
    self.assertFalse(self.wait(lambda: False))
    self.assertEqual(self.clock(), 60.)

  def test_unstable_link_must_settle_again(self):
    self.assertTrue(self.wait(lambda: not 21. <= self.clock() < 24.))
    self.assertEqual(self.clock(), 26.)

  def test_small_model_preload_counts_toward_grace(self):
    self.clock.now = 15.
    self.assertTrue(self.wait(lambda: True, started_at=0.))
    self.assertEqual(self.clock(), 22.)

  def test_expired_budget_does_not_touch_gpu(self):
    self.clock.now = 61.
    ready = Mock()
    self.assertFalse(self.wait(ready, started_at=0.))
    ready.assert_not_called()

  def test_probe_crossing_deadline_cannot_succeed(self):
    def slow_probe():
      self.clock.now = 61.
      return True
    self.assertFalse(self.wait(slow_probe))

  def test_invalid_timing_rejected(self):
    for kwargs in [{'delay': -1}, {'timeout': 10}, {'stable': -1}, {'delay': float('nan')}, {'timeout': float('inf')}]:
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        self.wait(lambda: True, **kwargs)

  def test_adapter_without_ready_pcie_is_not_ready(self):
    device = {'vendorId': 0x3801, 'productId': 1, 'product': 'custom ed4e39b7-CLEAN', 'speedMbps': 5000}
    with patch('openpilot.common.hardware.usb.get_usb_state', return_value=[device]), \
         patch('openpilot.system.hardware.chestnut.flash.link_up', return_value=False) as link:
      self.assertFalse(gpu_link_ready())
      link.assert_called_once_with()

  def test_wrong_firmware_or_slow_usb_never_probes_link(self):
    device = {'vendorId': 0x3801, 'productId': 1, 'product': 'custom ed4e39b7-CLEAN', 'speedMbps': 5000}
    for devices in [[], [device, device], [{**device, 'speedMbps': 480}], [{**device, 'product': 'old firmware'}]]:
      with self.subTest(devices=devices), patch('openpilot.common.hardware.usb.get_usb_state', return_value=devices), \
           patch('openpilot.system.hardware.chestnut.flash.link_up') as link:
        self.assertFalse(gpu_link_ready())
        link.assert_not_called()

  def test_healthy_link_is_ready(self):
    device = {'vendorId': 0x3801, 'productId': 1, 'product': 'custom ed4e39b7-CLEAN', 'speedMbps': 5000}
    with patch('openpilot.common.hardware.usb.get_usb_state', return_value=[device]), \
         patch('openpilot.system.hardware.chestnut.flash.link_up', return_value=True):
      self.assertTrue(gpu_link_ready())
