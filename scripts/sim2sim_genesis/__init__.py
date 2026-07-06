"""Modular Genesis sim2sim evaluator package.

The package exports the common symbols used by the Genesis evaluator, but keeps
imports lazy so lightweight modules such as metrics/ONNX parsing can be reused
from non-Genesis environments without importing torch or Genesis.
"""

__all__ = [
    "EvalConfig",
    "DomainRandomizer",
    "GenesisSceneAdapter",
    "ObservationBuilder",
    "OnnxMotionPolicy",
    "OutputTargets",
    "PdController",
    "PolicyMeta",
    "Sim2SimRunner",
    "TrackingMetricsEvaluator",
    "TrajectoryRecorder",
]

_EXPORT_MODULES = {
    "EvalConfig": "sim2sim_genesis.config",
    "OutputTargets": "sim2sim_genesis.config",
    "DomainRandomizer": "sim2sim_genesis.domain_randomization",
    "GenesisSceneAdapter": "sim2sim_genesis.scene",
    "ObservationBuilder": "sim2sim_genesis.observations",
    "OnnxMotionPolicy": "sim2sim_genesis.onnx_policy",
    "PolicyMeta": "sim2sim_genesis.onnx_policy",
    "PdController": "sim2sim_genesis.control",
    "Sim2SimRunner": "sim2sim_genesis.runner",
    "TrackingMetricsEvaluator": "sim2sim_genesis.metrics",
    "TrajectoryRecorder": "sim2sim_genesis.trajectory_io",
}


def __getattr__(name: str):
    """Lazily resolve package-level exports."""

    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
