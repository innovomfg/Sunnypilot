"""Keep expensive startup work out of the real-time inference thread."""
from openpilot.common.realtime import drop_realtime


def load_model(factory, **kwargs):
  # Linux threads inherit scheduling policy. This must also run in background
  # loaders, even if their caller has already entered the inference loop.
  drop_realtime()
  model = factory(**kwargs)
  model.warmup()
  return model
