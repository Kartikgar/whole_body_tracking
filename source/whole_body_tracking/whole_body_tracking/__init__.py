"""
Python module serving as a project/extension template.
"""

# Register Gym environments when Isaac Lab is available. Lightweight script
# environments, such as MuJoCo sim2sim evaluation, may install this package
# without Isaac Lab and should still be able to import package metadata.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_tasks":
        raise
