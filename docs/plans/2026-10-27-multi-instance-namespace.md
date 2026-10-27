# VERL Multi-Instance Namespace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native instance-scoped names to VERL so multiple trainers can share one Ray cluster without monkey patches.

**Architecture:** Introduce one immutable `InstanceNameBuilder` derived from `config.instance.id`. Pass it through resource-pool, reward-loop, rollout, vLLM and weight-transfer naming paths. Preserve existing names when no instance ID is configured.

**Tech Stack:** Python, dataclasses, Ray, OmegaConf-compatible VERL configs, pytest.

---

### Task 1: Add instance naming primitives and tests

**Files:**
- Create: `verl/single_controller/ray/instance_naming.py`
- Test: `tests/single_controller/ray/test_instance_naming.py`

Add validation, stable names, resource-specific length limits, and backward-compatible defaults.

### Task 2: Thread naming through resource pools

**Files:**
- Modify: `verl/single_controller/ray/base.py`
- Test: `tests/single_controller/ray/test_instance_resource_names.py`

Pass the builder through `ResourcePoolManager`, `RayResourcePool`, and `SubRayResourcePool`; use it for placement-group names.

### Task 3: Isolate reward and rollout worker actor names

**Files:**
- Modify: `verl/experimental/reward_loop/reward_loop.py`
- Modify: `verl/workers/rollout/replica.py`
- Tests: existing reward/rollout tests plus focused fake-actor tests

Use the same builder for reward workers, checkpoint workers and rollout worker groups.

### Task 4: Isolate vLLM server names

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- Modify: `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- Create or modify: shared vLLM naming helper and tests

Make server creation and lookup call one shared naming function.

### Task 5: Isolate weight-transfer IPC endpoints

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/utils.py`
- Modify: vLLM server/adapter call sites
- Tests: socket-name unit tests

Pass instance ID explicitly to sender and spawned receiver; avoid recipe-side environment mutation.

### Task 6: Add end-to-end Ray CPU coverage and validate

**Files:**
- Modify/create: `tests/...` appropriate Ray CPU integration test

Verify two instance IDs can create independent names and do not collide; run formatting, focused tests, and the relevant existing suite.

