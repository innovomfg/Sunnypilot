"""Keep expensive startup work out of the real-time inference thread."""
import math
import time

from openpilot.common.realtime import drop_realtime

GPU_BOOT_DELAY = 20.0
GPU_BOOT_TIMEOUT = 60.0
GPU_READY_STABLE = 2.0


def gpu_link_ready():
  """Probe the adapter/PCIe link without opening tinygrad's AMD device.

  USB enumeration alone is insufficient: the adapter also runs on USB power
  while the external GPU is off. link_up only checks LTSSM; it does not flash.
  """
  from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, get_usb_state, is_chestnut_usb_id
  from openpilot.system.hardware.chestnut.flash import link_up

  devices = [d for d in get_usb_state() if is_chestnut_usb_id(d['vendorId'], d['productId'], include_bootloader=True)]
  return (len(devices) == 1 and devices[0]['product'] == CHESTNUT_USB_PRODUCT
          and devices[0]['speedMbps'] >= 5000 and link_up())


def wait_for_gpu_boot(ready=gpu_link_ready, *, started_at=None, delay=GPU_BOOT_DELAY, timeout=GPU_BOOT_TIMEOUT,
                      stable=GPU_READY_STABLE, clock=time.monotonic, sleep=time.sleep):
  """Allow external power to settle, then require a continuously ready link.

  Camera setup and small-model preload count toward the grace period. The boot
  budget is separate from the existing GPU model load/warmup timeout. Polls may
  overrun the deadline by one bounded USB probe; never retry a failed GPU load.
  """
  if not all(math.isfinite(v) and v >= 0 for v in (delay, timeout, stable)) or timeout < delay + stable:
    raise ValueError('Invalid GPU boot timing')
  start = clock() if started_at is None else started_at
  deadline = start + timeout
  ready_since = None
  while (now := clock()) < deadline:
    if now < start + delay:
      sleep(min(start + delay - now, deadline - now, 0.25))
      continue
    healthy = ready()
    now = clock()
    if now >= deadline:
      return False
    if healthy:
      if ready_since is None:
        ready_since = now
      if now - ready_since >= stable:
        return True
    else:
      ready_since = None
    sleep(min(0.25, deadline - now))
  return False


def load_model(factory, **kwargs):
  # Linux threads inherit scheduling policy. This must also run in background
  # loaders, even if their caller has already entered the inference loop.
  drop_realtime()
  model = factory(**kwargs)
  model.warmup()
  return model
