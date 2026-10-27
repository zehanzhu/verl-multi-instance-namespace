import pytest

from verl.single_controller.ray.instance_naming import (
    InstanceNameBuilder,
    build_vllm_server_prefix,
    build_weight_transfer_socket,
    get_instance_id,
)
from verl.single_controller.ray.base import ResourcePoolManager


def test_default_names_are_backward_compatible():
    names = InstanceNameBuilder()
    assert names.actor("reward_loop_worker_0") == "reward_loop_worker_0"
    assert names.placement_group("global_poolverl_group_0") == "global_poolverl_group_0"


def test_instance_names_are_isolated_and_stable():
    policy_a = InstanceNameBuilder("policy_1")
    policy_b = InstanceNameBuilder("policy_2")

    assert policy_a.actor("reward_loop_worker_0") != policy_b.actor("reward_loop_worker_0")
    assert policy_a.socket("replica_0_rank_0") == policy_a.socket("replica_0_rank_0")


def test_invalid_instance_id_is_rejected():
    with pytest.raises(ValueError, match="instance_id"):
        InstanceNameBuilder("bad/id")


def test_long_names_use_stable_hash_suffix():
    names = InstanceNameBuilder("p" * 80, max_name_length=64)
    value = names.actor("actor")
    assert len(value) <= 64
    assert value == names.actor("actor")
    assert value.endswith(names.digest("p" * 80 + "__actor"))


def test_vllm_prefix_and_weight_socket_share_instance_id():
    assert build_vllm_server_prefix("policy_1") == "policy_1__vllm_"
    assert build_weight_transfer_socket("job", "policy_1", 0, 1) == (
        "ipc:///tmp/rl-colocate-zmq-job-policy_1-replica-0-rank-1.sock"
    )


def test_instance_id_can_be_read_from_nested_or_rollout_config():
    assert get_instance_id({"instance": {"id": "policy_1"}}) == "policy_1"
    assert get_instance_id({"actor_rollout_ref": {"rollout": {"instance_id": "policy_2"}}}) == "policy_2"


def test_resource_pool_manager_propagates_instance_builder():
    manager = ResourcePoolManager(resource_pool_spec={}, mapping={}, instance_id="policy_1")
    assert manager.instance_name_builder.actor("global_pool") == "policy_1__global_pool"
