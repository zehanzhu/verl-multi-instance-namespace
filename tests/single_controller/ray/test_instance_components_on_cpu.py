"""Exercise resource and teacher components without loading a model."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager, merge_resource_pool
from verl.single_controller.ray.instance_naming import InstanceNameBuilder, build_weight_transfer_socket


def test_socket_byte_limit_and_identity():
    endpoints = {
        build_weight_transfer_socket("job", "policy_" + "x" * 200 + suffix, replica, rank)
        for suffix in ("a", "b") for replica in (0, 1) for rank in (0, 1)
    }
    assert len(endpoints) == 8
    assert all(len(endpoint.removeprefix("ipc://").encode()) <= 103 for endpoint in endpoints)
    assert build_weight_transfer_socket("job", None, 0, 0) == (
        "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"
    )


def test_cross_instance_merge_rejected_before_allocating(monkeypatch):
    pools = [RayResourcePool([1], use_gpu=False, instance_name_builder=InstanceNameBuilder(i)) for i in ("a", "b")]
    for pool in pools:
        monkeypatch.setattr(pool, "get_placement_groups", lambda **kw: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="instance"):
        merge_resource_pool(*pools)


def test_manager_creates_pools_with_instance_identity(monkeypatch):
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)
    manager = ResourcePoolManager({"global_pool": [1]}, {}, instance_id="a")
    manager.create_resource_pool()
    assert manager.resource_pool_dict["global_pool"].instance_name_builder.instance_id == "a"


def test_teacher_inherits_pool_identity_without_mutating_shared_config(monkeypatch):
    from verl.experimental.teacher_loop import teacher_model as module
    from verl.workers.config import RolloutConfig

    inference = RolloutConfig(name="vllm", tensor_model_parallel_size=1)
    captured = []

    def replica(**kwargs):
        captured.append(kwargs["config"])
        return SimpleNamespace(init_colocated=AsyncMock(), _server_handle=None, _server_address="localhost")

    monkeypatch.setattr(module, "get_rollout_replica_class", lambda name: replica)
    monkeypatch.setattr(module, "split_resource_pool", lambda pool, split_size: [pool])
    monkeypatch.setattr(module.RLInsightLogger, "enabled", lambda: False)
    for instance_id in ("a", "b"):
        manager = object.__new__(module.TeacherModelManager)
        manager.teacher_model_config = SimpleNamespace(
            inference=inference, per_replica_world_size=1, num_replicas=1, key="teacher", model_path="unused"
        )
        manager.distillation_config = SimpleNamespace(n_gpus_per_node=1)
        manager.resource_pool = RayResourcePool([1], use_gpu=False, instance_name_builder=InstanceNameBuilder(instance_id))
        manager._validate_replica_node_alignment = lambda *args: None
        manager._initialize_llm_servers()
    assert [config.instance_id for config in captured] == ["a", "b"]
    assert inference.instance_id is None
