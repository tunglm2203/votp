from functools import partial

import gym
import torch
import numpy as np

import research

"""
There are two special network functions used by research lightning
1. output_space - this is used to give the observation_space to different networks in a container group
2. compile - this is used when torch.compile is called.
"""


def reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class ModuleContainer(torch.nn.Module):
    CONTAINERS = []

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs) -> None:
        super().__init__()
        # save the classes and containers
        base_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_class") and not k.endswith("_kwargs")}

        output_space = observation_space
        for container in self.CONTAINERS:
            module_class = kwargs.get(container + "_class", torch.nn.Identity)
            module_class = vars(research.networks)[module_class] if isinstance(module_class, str) else module_class
            if module_class is torch.nn.Identity:
                module_kwargs = dict()
            else:
                module_kwargs = base_kwargs.copy()
                module_kwargs.update(kwargs.get(container + "_kwargs", dict()))
            # Create the module, and attach it to self
            module = module_class(output_space, action_space, **module_kwargs)
            setattr(self, container, module)

            # Set a reset function
            setattr(self, "reset_" + container, partial(self._reset, container))

            if hasattr(getattr(self, container), "output_space"):
                # update the output space
                output_space = getattr(self, container).output_space

        # Done creating all sub-modules.

    @classmethod
    def create_subset(cls, containers):
        assert all([container in cls.CONTAINERS for container in containers])
        name = "".join([container.capitalize() for container in containers]) + "Subset"
        return type(name, (ModuleContainer,), {"CONTAINERS": containers})

    def _reset(self, container: str) -> None:
        module = getattr(self, container)
        with torch.no_grad():
            module.apply(reset)

    def compile(self, **kwargs):
        for container in self.CONTAINERS:
            attr = getattr(self, container)
            if type(attr).forward == torch.nn.Module.forward:
                assert hasattr(attr, "compile"), (
                    "container " + container + " is nn.Module without forward() but didn't define `compile`."
                )
                attr.compile(**kwargs)
            else:
                setattr(self, container, torch.compile(attr, **kwargs))


class ActorCriticPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "critic"]


class ActorCriticValuePolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "critic", "value"]


class ActorValuePolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "value"]


class ActorPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor"]


class ActorCriticRewardPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "critic", "reward"]


class ActorCriticValueRewardPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "critic", "value", "reward"]


class ModuleContainer_RewardSpecial(torch.nn.Module):
    CONTAINERS = []

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs) -> None:
        super().__init__()
        # save the classes and containers
        base_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_class") and not k.endswith("_kwargs")}
        img_size = None
        if 'image_size' in base_kwargs:
            img_size = base_kwargs.pop('image_size')

        output_space = observation_space
        for container in self.CONTAINERS:
            module_class = kwargs.get(container + "_class", torch.nn.Identity)
            module_class = vars(research.networks)[module_class] if isinstance(module_class, str) else module_class
            if module_class is torch.nn.Identity:
                module_kwargs = dict()
            else:
                if 'DrQv2Reward' in kwargs[f"{container}_class"]:
                    module_kwargs = dict()
                else:
                    module_kwargs = base_kwargs.copy()
                module_kwargs.update(kwargs.get(container + "_kwargs", dict()))
            # Create the module, and attach it to self
            if module_class is not torch.nn.Identity and 'DrQv2Reward' in kwargs[f"{container}_class"]:
                visual_output_space = gym.spaces.Box(low=0, high=255, shape=(3, img_size, img_size), dtype=np.uint8)
                module = module_class(visual_output_space, action_space, **module_kwargs)
            else:
                module = module_class(output_space, action_space, **module_kwargs)
            setattr(self, container, module)

            # Set a reset function
            setattr(self, "reset_" + container, partial(self._reset, container))

            if hasattr(getattr(self, container), "output_space"):
                # update the output space
                output_space = getattr(self, container).output_space

        # Done creating all sub-modules.

    @classmethod
    def create_subset(cls, containers):
        assert all([container in cls.CONTAINERS for container in containers])
        name = "".join([container.capitalize() for container in containers]) + "Subset"
        return type(name, (ModuleContainer,), {"CONTAINERS": containers})

    def _reset(self, container: str) -> None:
        module = getattr(self, container)
        with torch.no_grad():
            module.apply(reset)

    def compile(self, **kwargs):
        for container in self.CONTAINERS:
            attr = getattr(self, container)
            if type(attr).forward == torch.nn.Module.forward:
                assert hasattr(attr, "compile"), (
                    "container " + container + " is nn.Module without forward() but didn't define `compile`."
                )
                attr.compile(**kwargs)
            else:
                setattr(self, container, torch.compile(attr, **kwargs))


class ActorCriticValueRewardSpecialPolicy(ModuleContainer_RewardSpecial):
    CONTAINERS = [
        "reward",
        "encoder",
        "actor",
        "critic",
        "value",
    ]