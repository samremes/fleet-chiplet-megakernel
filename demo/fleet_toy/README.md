# Fleet three-task tutorial

This example runs three small task bodies through Fleet's real graph generator,
event scheduler, worker queues, XCD discovery, and gang dispatch:

1. RMSNorm: four ordinary tasks, one per row.
2. Linear: eight gang descriptors, one per XCD. Fleet broadcasts each
   descriptor to the workers resident on that XCD; `tile_idx` selects a row.
3. Residual add: four ordinary tasks, one per row.

There are no per-operation HIP launches. The current GPT-OSS branch uses its
production split runtime: one persistent worker kernel and one persistent
scheduler kernel, preceded by a small prepare kernel. Therefore "megakernel"
means that all three compute operations execute inside the persistent worker
kernel; it does not mean the complete host call contains exactly one physical
HIP launch.

The prepare kernel is shared Fleet runtime machinery, not toy computation. It
resets worker/scheduler queues, global and per-XCD event counters, XCD
discovery/election state, and—when precomputed dispatch is enabled—iteration,
termination, rank, and gang-barrier state. It then seeds the end-of-graph event
that starts scheduler processing. GPT-OSS and Qwen use it too because
`launch_persistent_kernel` invokes it before every persistent launch.

## Branch

The branch `tutorial-three-task-fusion` is based on
`amd_mi355_gpt_oss120b` commit `8631fa7`. The example is isolated under
`demo/fleet_toy/`; it does not construct a GPT-OSS or Qwen model.

## Build

```bash
git submodule sync --recursive
git submodule update --init --recursive

export ROCM_PATH=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel
export PATH="$HOME/.cargo/bin:$PATH"
python3 -m pip install \
  z3-solver cython graphviz safetensors \
  transformers==4.57.1 accelerate==1.8.0
python3 -m pip install -e . -v --no-build-isolation --no-deps
```

`setup.py` now forwards `ROCM_PATH` to CMake. This is required in Python-wheel
ROCm environments where `/opt/rocm` does not exist.

## Run

```bash
demo/fleet_toy/run.sh
```

The run detects the attached GPU architecture and enables Fleet gang dispatch.
On gfx950 the verified result is:

```text
fleet_toy: PASS max_abs_err=0.000000 compute_tasks=4_rms+8_gang_descriptors+4_add runtime=Fleet_split_worker_scheduler
```

The generated graph contains 18 total tasks: one begin task, 16 compute
descriptors, and one end task. Each run writes the inspectable generator output
to `artifacts/task_graph_rank0.json` and `artifacts/test_rank0.cu`; the directory
is ignored because the generated C++ embeds process-local tensor addresses.

## Fleet integration points

Adding these task types requires changes at every layer of Fleet's closed task
registry:

- `include/mirage/persistent_kernel/tasks/mi300/fleet_toy.cuh`: device bodies.
- `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh`: body include.
- `include/mirage/persistent_kernel/runtime_header.h`: task enum values.
- `include/mirage/kernel/task_register.h` and
  `src/kernel/task_register.cc`: registrar declarations and generated calls.
- `src/kernel/graph.cc`: public task names, arity, type, and gang tile count.
- `include/mirage/persistent_kernel/persistent_kernel.cuh`: gang predicate.
- `src/kernel/runtime.cc`: metadata, name map, ordinary exclusion, and gang
  inclusion.
- `python/mirage/mpk/persistent_kernel.py`: three graph-building methods.

This duplication is why the example lives on a Fleet fork. Fleet does not
currently expose an external task-plugin ABI.

## Runtime settings

The example uses:

- `USE_GANG=1`, which emits `_execute_gang_task`.
- `PRECOMPUTED_DISPATCH=0`, which uses Fleet's general scheduler path rather
  than the GPT-OSS-specific precomputed queues.
- `MPK_W13_BIAS_COUNTED_WAIT=0`, because that unrelated GPT-OSS optimization
  requires another model-specific compile option.
- `mode="online_notoken"`, which runs a bounded graph without model-serving
  token logic.

The branch also makes decode-progress storage available outside precomputed
dispatch and clears an initialization-time stale HIP error before checking
launch status. Both are needed for the general scheduler path on this branch.
