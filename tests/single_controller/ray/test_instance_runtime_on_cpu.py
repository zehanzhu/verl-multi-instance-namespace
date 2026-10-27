"""Two instances share real Ray names and placement groups in one namespace."""

import uuid

import pytest
import ray
from ray.util.placement_group import remove_placement_group

from verl.single_controller.ray.base import RayResourcePool
from verl.single_controller.ray.instance_naming import InstanceNameBuilder


@ray.remote(num_cpus=0)
class IdentityWorker:
    def __init__(self, identity):
        self.identity = identity

    def ping(self):
        return self.identity


def test_two_instances_share_runtime_and_release_independently():
    owns_runtime = not ray.is_initialized()
    if owns_runtime:
        ray.init(num_cpus=2, include_dashboard=False)
    actors, groups = [], []
    namespace = "instance_test_" + uuid.uuid4().hex
    try:
        for identity in ("policy_a", "policy_b"):
            names = InstanceNameBuilder(identity)
            pool = RayResourcePool([1], use_gpu=False, name_prefix=namespace,
                                   max_colocate_count=1, instance_name_builder=names)
            groups.extend(pool.get_placement_groups())
            actor = IdentityWorker.options(name=names.actor("worker"), namespace=namespace).remote(identity)
            actors.append(actor)
        assert groups[0].id != groups[1].id
        assert ray.get([actor.ping.remote() for actor in actors], timeout=30) == ["policy_a", "policy_b"]
        for identity in ("policy_a", "policy_b"):
            handle = ray.get_actor(InstanceNameBuilder(identity).actor("worker"), namespace=namespace)
            assert ray.get(handle.ping.remote(), timeout=30) == identity
        ray.kill(actors.pop(0))
        remove_placement_group(groups.pop(0))
        assert ray.get(actors[0].ping.remote(), timeout=30) == "policy_b"
    finally:
        for actor in actors:
            ray.kill(actor)
        for group in groups:
            remove_placement_group(group)
        if owns_runtime:
            ray.shutdown()
