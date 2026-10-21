import ast
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from openpilot.sunnypilot.modeld_v2.startup import load_model


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
