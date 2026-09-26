"""
Python module serving as a project/extension template.
"""

# Register Gym environments when Isaac Lab is available. Lightweight script
# environments, such as MuJoCo sim2sim evaluation, may install this package
# without Isaac Lab and should still be able to import package metadata.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    # Utility modules such as SONIC motion/policy adapters do not require a
    # running Omniverse application. Isaac packages may be installed while
    # their ``omni`` modules are unavailable until AppLauncher starts.
    if exc.name != "isaaclab_tasks" and not exc.name.startswith(("omni.", "pxr")):
        raise
