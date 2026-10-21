#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

if [ -z "${AGNOS_VERSION:-}" ]; then
  export AGNOS_VERSION="19.7"

  # Preserve an already-installed 19.8.1 on comma four (mici). This is not
  # an installation target: the bundled image manifest still describes 19.7.
  # Keep this policy here: both startup and the staged background updater
  # source this file, including after clearing an inherited AGNOS_VERSION.
  if [ "$(cat /VERSION 2>/dev/null)" = "19.8.1" ] && \
     [ "$(cat /sys/firmware/devicetree/base/model 2>/dev/null | tr -d '\000')" = "comma mici" ]; then
    export AGNOS_VERSION="19.8.1"
  fi
fi

export STAGING_ROOT="/data/safe_staging"
