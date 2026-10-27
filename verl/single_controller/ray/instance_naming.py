"""Stable names for resources owned by one VERL runtime instance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_VALID_INSTANCE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class InstanceNameBuilder:
    """Build names that are isolated by an optional instance identifier.

    An unset identifier deliberately preserves VERL's existing names.  The
    builder is immutable so it can safely be shared by resource managers and
    copied into Ray actor construction arguments.
    """

    instance_id: str | None = None
    max_name_length: int = 255

    def __post_init__(self) -> None:
        if self.max_name_length < 16:
            raise ValueError("max_name_length must be at least 16")
        if self.instance_id is not None:
            if not isinstance(self.instance_id, str) or not self.instance_id.strip():
                raise ValueError("instance_id must be a non-empty string")
            if not _VALID_INSTANCE_ID.fullmatch(self.instance_id):
                raise ValueError(
                    "instance_id must contain only ASCII letters, digits, '_' or '-'")

    def actor(self, logical_name: str) -> str:
        return self._build(logical_name)

    def placement_group(self, logical_name: str) -> str:
        return self._build(logical_name)

    def socket(self, logical_name: str) -> str:
        return self._build(logical_name)

    def digest(self, value: str) -> str:
        """Return the stable suffix used when a generated name is too long."""
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]

    def _build(self, logical_name: str) -> str:
        if not isinstance(logical_name, str) or not logical_name:
            raise ValueError("logical_name must be a non-empty string")
        value = (
            logical_name
            if self.instance_id is None
            else f"{self.instance_id}__{logical_name}"
        )
        if len(value) <= self.max_name_length:
            return value
        digest = self.digest(value)
        return f"{value[: self.max_name_length - len(digest) - 1]}_{digest}"


def get_instance_id(config: object) -> str | None:
    """Read the public instance identifier from mapping or config objects."""
    if config is None:
        return None
    instance = config.get("instance") if hasattr(config, "get") else getattr(config, "instance", None)
    if instance is not None:
        value = instance.get("id") if hasattr(instance, "get") else getattr(instance, "id", None)
        if value is not None:
            return str(value)
    value = config.get("instance_id") if hasattr(config, "get") else getattr(config, "instance_id", None)
    if value is not None:
        return str(value)
    rollout = config.get("actor_rollout_ref", {}).get("rollout") if hasattr(config, "get") else None
    if rollout is not None:
        value = rollout.get("instance_id") if hasattr(rollout, "get") else getattr(rollout, "instance_id", None)
        return None if value is None else str(value)
    return None


def build_vllm_server_prefix(instance_id: str | None, base: str = "vllm_") -> str:
    """Return the server prefix shared by vLLM creators and clients."""
    if instance_id is None:
        return base
    return f"{instance_id}__{base}"


def build_weight_transfer_socket(
    ray_job_id: str,
    instance_id: str | None,
    replica_rank: str | int,
    trainer_rank: str | int,
) -> str:
    """Build the sender/receiver IPC endpoint for weight synchronization."""
    instance_suffix = f"-{instance_id}" if instance_id else ""
    path = (
        f"/tmp/rl-colocate-zmq-{ray_job_id}{instance_suffix}"
        f"-replica-{replica_rank}-rank-{trainer_rank}.sock"
    )
    # sockaddr_un has a platform-dependent fixed capacity. Keep the path at
    # most 103 bytes (excluding the NUL), also safe on systems with sun_path[104].
    # Hash the entire identity, including ranks, so shortening never drops them.
    if len(path.encode("utf-8")) > 103:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        path = f"/tmp/rl-colocate-zmq-{digest}.sock"
    return f"ipc://{path}"

