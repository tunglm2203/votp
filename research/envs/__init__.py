# Register environment classes here
from .base import Empty

# Import the D4RL environments
import d4rl

# If we want to register environments in gym.
# These will be loaded when we import the research package.
from gym.envs import register


from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

# Add the meta world test environments.
# For each one, register the different tasks.

for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_name}"
    register(id=ID, entry_point="research.envs.metaworld:MetaWorldSawyerEnv", kwargs={"env_name": env_name})

del register
