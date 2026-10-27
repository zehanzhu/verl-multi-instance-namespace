# Multi-instance GPU validation

This branch adds instance-scoped names for Ray actors, placement groups, vLLM
servers, teacher replicas, and weight-transfer IPC endpoints. Use this guide on
a Linux GPU host after fetching the `multi-instance-namespace` branch.

## 1. Fetch the exact commit

```bash
git clone -b multi-instance-namespace <your-personal-verl-repository> verl
cd verl
git checkout 164d1acc
git status --short                 # should be clean
```

Run the checks that do not require a GPU first:

```bash
python -m compileall -q verl
python -m pytest \
  tests/single_controller/ray/test_instance_naming.py \
  tests/single_controller/ray/test_instance_components_on_cpu.py \
  tests/single_controller/ray/test_instance_pd_on_cpu.py \
  tests/single_controller/ray/test_instance_runtime_on_cpu.py -q
```

## 2. Prepare the GPU environment

Use Python 3.10, 3.11, or 3.12. Python 3.13 is not supported by this
repository. Install the dependency set matching the selected backend and the
VERL commit; for the vLLM path, install the vLLM version pinned by
`pyproject.toml`/`uv.lock` rather than reusing an unrelated system vLLM.

Confirm the runtime before starting Ray:

```bash
python --version
python - <<'PY'
import torch, ray, vllm
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("ray", ray.__version__)
print("vllm", getattr(vllm, "__version__", "unknown"))
PY
nvidia-smi
```

Use one Ray cluster for the validation run. Stop stale jobs from previous
attempts so that old actors and IPC socket files cannot obscure the result:

```bash
ray stop --force
ray start --head --num-cpus <cpu-count> --num-gpus <gpu-count>
```

## 3. Validate instance isolation without training

Run the focused CPU/Ray tests on the GPU host as well. The important assertions
are:

* `instance.id=policy_a` and `instance.id=policy_b` produce different actor
  and placement-group names;
* both instances can start in the same Ray namespace;
* killing one instance's actor does not affect the other;
* no `ActorAlreadyExists`, placement-group name collision, or `EADDRINUSE`
  error appears.

## 4. v1 `separate_async` acceptance test

Start with a very small model and dataset available on the GPU host. A typical
single-update run should use:

```bash
export MODEL_PATH=/path/to/a-small-causal-lm
export RUN_DIR=/tmp/verl-instance-validation
mkdir -p "$RUN_DIR"
```

Use the repository's v1 separate-async script as the base command when it
matches the installed backend:

```bash
bash tests/special_e2e/run_v1_separate_async.sh \
  trainer.total_training_steps=1 \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.separate_async.num_warmup_batches=1 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=<rollout-gpus> \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=<training-gpus> \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  instance.id=policy_a \
  trainer.default_local_dir="$RUN_DIR/policy_a"
```

For a second independent trainer, use a separate process and change at least:

```text
instance.id=policy_b
trainer.default_local_dir=$RUN_DIR/policy_b
```

The two jobs must use disjoint GPU allocations. They may share the same Ray
cluster, but should not share the same training or rollout GPUs.

## 5. Required evidence

The first successful run must show all of the following in its logs:

1. rollout and training actors start successfully;
2. the first checkpoint-engine-to-vLLM weight synchronization completes;
3. at least one rollout batch is generated and receives a reward;
4. one PPO actor update completes;
5. a post-update weight synchronization completes;
6. both `policy_a` and `policy_b` logs contain distinct instance-prefixed Ray
   names and no actor lookup or socket collision errors.

Save the command line, environment versions, Ray logs, and the two trainer log
directories under `$RUN_DIR`. If the run fails, preserve the first traceback,
the actor names, and the computed IPC endpoint before retrying.

## 6. Common blockers

* `vllm version None not supported`: the installed vLLM is missing or does not
  match this VERL revision; install the pinned version.
* `EADDRINUSE` on an IPC endpoint: stop stale Ray workers, remove stale
  `/tmp/rl-colocate-zmq-*.sock` files, and verify both sender and receiver use
  the same Ray job and instance ID.
* `ActorNotFound` or `ActorAlreadyExists`: verify both jobs use this commit and
  that their `instance.id` values are non-empty and different.
* CUDA out-of-memory: reduce model size, sequence lengths, rollout batch size,
  and GPU memory utilization before changing the namespace settings.

The CPU tests prove naming and routing behavior only. GPU execution is required
to claim successful vLLM weight synchronization and a completed training update.
