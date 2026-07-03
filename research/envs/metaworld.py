"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import copy

import gym
import numpy as np
import mujoco_py
from scipy.spatial.transform import Rotation



def trim_mw_obs(obs):
    # Remove the double robot observation from the environment.
    # Only stack two object observations
    # this helps keep everything more markovian
    return np.concatenate((obs[:18], obs[22:]), dtype=np.float32)


class MetaWorldSawyerEnv(gym.Env):
    def __init__(self, env_name, seed=False, randomize_hand=True, sparse: bool = False, horizon: int = 250):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self.env_name = env_name
        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._partially_observable = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.
        self.randomize_hand = randomize_hand
        self.sparse = sparse
        assert self._env.observation_space.shape[0] == 39
        low, high = self._env.observation_space.low, self._env.observation_space.high
        self.observation_space = gym.spaces.Box(low=trim_mw_obs(low), high=trim_mw_obs(high), dtype=np.float32)
        self.action_space = self._env.action_space
        self._max_episode_steps = min(horizon, self._env.max_path_length)

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self._env.seed(0)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, done, info = self._env.step(action)
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Remove history from the observations. It makes it too hard to reset.
        if self.sparse:
            reward = float(info["success"])  # Reward is now if we succeed or fail.
        else:
            reward = reward / 10

        self.obs_origin = copy.deepcopy(obs.astype(np.float32))
        return trim_mw_obs(obs.astype(np.float32)), reward, done, info

    def _get_obs(self):
        self.obs_origin = copy.deepcopy(self._env._get_obs())
        return trim_mw_obs(self.obs_origin)

    def get_state(self):
        joint_state, mocap_state = self._env.get_env_state()
        qpos, qvel = joint_state.qpos, joint_state.qvel
        mocap_pos, mocap_quat = mocap_state
        self._split_shapes = np.cumsum(
            np.array([qpos.shape[0], qvel.shape[0], mocap_pos.shape[1], mocap_quat.shape[1]])
        )
        return np.concatenate([qpos, qvel, mocap_pos[0], mocap_quat[0], self._env._last_rand_vec], axis=0)

    def set_state(self, state):
        joint_state = self._env.sim.get_state()
        if not hasattr(self, "_split_shapes"):
            self.get_state()  # Load the split
        qpos, qvel, mocap_pos, mocap_quat, rand_vec = np.split(state, self._split_shapes, axis=0)
        if not np.all(self._env._last_rand_vec == rand_vec):
            # We need to set the rand vec and then reset
            self._env._freeze_rand_vec = True
            self._env._last_rand_vec = rand_vec
            self._env.reset()
        joint_state.qpos[:] = qpos
        joint_state.qvel[:] = qvel
        self._env.set_env_state((joint_state, (np.expand_dims(mocap_pos, axis=0), np.expand_dims(mocap_quat, axis=0))))
        self._env.sim.forward()

    def reset(self, **kwargs):
        self._episode_steps = 0
        self._env.reset(**kwargs).astype(np.float32)
        if self.randomize_hand:
            # Hand init pos is usually set to self.init_hand_pos
            # We will add some uniform noise to it.
            high = np.array([0.25, 0.15, 0.2], dtype=np.float32)
            hand_init_pos = self.hand_init_pos + np.random.uniform(low=-high, high=high)
            hand_init_pos = np.clip(hand_init_pos, a_min=self._env.mocap_low, a_max=self._env.mocap_high)
            hand_init_pos = np.expand_dims(hand_init_pos, axis=0)
            for _ in range(50):
                self._env.data.set_mocap_pos("mocap", hand_init_pos)
                self._env.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
                self._env.do_simulation([-1, 1], self._env.frame_skip)

        # Get the obs once to reset history.
        self._get_obs()
        return self._get_obs().astype(np.float32)

    def render(self, mode="rgb_array", camera_name="corner2", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        for ctx in self._env.sim.render_contexts:
            ctx.opengl_context.make_context_current()
        return self._env.render(offscreen=True, camera_name=camera_name, resolution=(width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)


class MetaWorldSawyerImageEnv(gym.Wrapper):
    # Focus on object
    camera_config = {
        'door-open-v2': {
            'distance': 1.35,
            'azimuth': -40,
            'elevation': -145,
            'lookat': [0.0, 0.65, 0.0]
        },
        'drawer-open-v2': {
            'distance': 1.1,
            'azimuth': -40,
            'elevation': -145,
            'lookat': [0.0, 0.65, 0.0]
        },
        'plate-slide-v2': {
            'distance': 1.1,
            'azimuth': -40,
            'elevation': -145,
            'lookat': [0.0, 0.65, 0.0]
        },
        'sweep-into-v2': {
            'distance': 1.2,
            'azimuth': 80,
            'elevation': -135,
            'lookat': [0.0, 0.65, 0.0]
        },
    }

    # Focus on object in closer distance
    camera_config_v1 = {
        'door-open-v2': {
            'distance': 1.1,
            'azimuth': -45,
            'elevation': -145,
            'lookat': [0.1, 0.65, 0.1]
        },
        'drawer-open-v2': {
            'distance': 0.8,
            'azimuth': -35,
            'elevation': -145,
            'lookat': [0, 0.65, 0]
        },
        'plate-slide-v2': {
            'distance': 0.8,
            'azimuth': -40,
            'elevation': -152,
            'lookat': [0.0, 0.65, 0.0]
        },
        'sweep-into-v2': {
            'distance': 0.85,
            'azimuth': 180,
            'elevation': -135,
            'lookat': [0.0, 0.75, 0.0]
        },
    }

    def __init__(self, env, width=64, height=64, camera="corner2_v2", show_goal=False):
        assert isinstance(
            env.unwrapped, MetaWorldSawyerEnv
        ), "MetaWorld Wrapper must be used with a MetaWorldSawyerEnv class"
        super().__init__(env)
        self._width = width
        self._height = height

        self.BASIC_CAMERA = ["corner2", "corner2_v1", "corner2_v2"]   # (1) default, (2) modification from CPL paper, and same with (2) but slightly modified
        self.CUSTOM_CAMERA = ["env_specific_v1", "env_specific_v2"] # modification from RL-VLM-F and ours (slightly different)
        self.CUSTOM_CAMERA_v2 = ["env_specific_preference", "env_specific_preference_v1"] # our new modification, more intuitively
        assert camera in self.BASIC_CAMERA + self.CUSTOM_CAMERA + self.CUSTOM_CAMERA_v2

        self._camera = camera
        self._show_goal = show_goal
        shape = (3, self._height, self._width)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        if self._camera in self.CUSTOM_CAMERA_v2:
            self.my_viewer = mujoco_py.MjRenderContextOffscreen(self.unwrapped.sim, device_id=-1)
            if self._camera == "env_specific_preference":
                self.my_viewer.cam.distance = self.camera_config[self.spec.kwargs['env_name']]['distance']
                self.my_viewer.cam.azimuth = self.camera_config[self.spec.kwargs['env_name']]['azimuth']
                self.my_viewer.cam.elevation = self.camera_config[self.spec.kwargs['env_name']]['elevation']
                self.my_viewer.cam.lookat[0] = self.camera_config[self.spec.kwargs['env_name']]['lookat'][0]
                self.my_viewer.cam.lookat[1] = self.camera_config[self.spec.kwargs['env_name']]['lookat'][1]
                self.my_viewer.cam.lookat[2] = self.camera_config[self.spec.kwargs['env_name']]['lookat'][2]
            elif self._camera == "env_specific_preference_v1":
                self.my_viewer.cam.distance = self.camera_config_v1[self.spec.kwargs['env_name']]['distance']
                self.my_viewer.cam.azimuth = self.camera_config_v1[self.spec.kwargs['env_name']]['azimuth']
                self.my_viewer.cam.elevation = self.camera_config_v1[self.spec.kwargs['env_name']]['elevation']
                self.my_viewer.cam.lookat[0] = self.camera_config_v1[self.spec.kwargs['env_name']]['lookat'][0]
                self.my_viewer.cam.lookat[1] = self.camera_config_v1[self.spec.kwargs['env_name']]['lookat'][1]
                self.my_viewer.cam.lookat[2] = self.camera_config_v1[self.spec.kwargs['env_name']]['lookat'][2]

    def _get_image(self):
        if self._camera in self.CUSTOM_CAMERA_v2:
            if not self._show_goal:
                try:
                    self.env.unwrapped._set_pos_site("goal", np.inf * self.env.unwrapped._target_pos)
                except ValueError:
                    pass  # If we don't have the goal site, just continue.

            self.my_viewer.render(self._width, self._height)
            img = self.my_viewer.read_pixels(self._width, self._height, depth=False)

        else:
            if not self._show_goal:
                try:
                    self.env.unwrapped._set_pos_site("goal", np.inf * self.env.unwrapped._target_pos)
                except ValueError:
                    pass  # If we don't have the goal site, just continue.
            img = self.env.render(mode="rgb_array", camera_name="corner2", width=self._width, height=self._height)

            if self._camera in ["env_specific_v1"]:
                if self.env_name == "drawer-open-v2":
                    img = img[::-1, :, :]

        return img.transpose(2, 0, 1)

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        # Throw away the state-based observation.
        info["state"] = state_obs
        return self._get_image().copy(), reward, done, info

    def reset(self):
        camera_name = "corner2"
        index = self.model.camera_name2id(camera_name)

        # Initiate viewer
        if self._camera in ["corner2"]:
            # Use default config in xyz_base.xml
            pass
        elif self._camera in ["corner2_v1"]:
            # Original XYZ is 1.3 -0.2 1.1
            # Zoom in camera corner2 to make it better for control
            # I found this view to work well across a lot of the tasks.
            self.model.cam_fovy[index] = 20.0  # FOV
            self.model.cam_pos[index][0] = 1.5  # X
            self.model.cam_pos[index][1] = -0.35  # Y
            self.model.cam_pos[index][2] = 1.1  # Z

        elif self._camera in ["corner2_v2"]:
            # Original XYZ is 1.3 -0.2 1.1
            # Zoom in camera corner2 to make it better for control
            # I found this view to work well across a lot of the tasks.
            if self.spec.kwargs['env_name'] in ["door-open-v2"]:
                self.model.cam_fovy[index] = 23.0  # FOV
                self.model.cam_pos[index][0] = 1.5  # X
                self.model.cam_pos[index][1] = -0.2  # Y
                self.model.cam_pos[index][2] = 1.1  # Z
            else:
                self.model.cam_fovy[index] = 20.0  # FOV
                self.model.cam_pos[index][0] = 1.5  # X
                self.model.cam_pos[index][1] = -0.35  # Y
                self.model.cam_pos[index][2] = 1.1  # Z

        elif self._camera in self.CUSTOM_CAMERA_v2:
            # No need to initialize anything
            pass

        elif self._camera in self.CUSTOM_CAMERA:
            if self.env.spec.kwargs['env_name'] == "drawer-open-v2":
                if self._camera == "env_specific_v1":
                    # NOTE: old config  (from RL-VLM-F, suitable for invisible arm)
                    self.model.cam_fovy[index] = 60.0  # FOV
                    self.model.cam_pos[index][0] = 0.0  # X
                    self.model.cam_pos[index][1] = 0.5  # Y
                    self.model.cam_pos[index][2] = 0.5  # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('xyz', [0.5, 0.0, 0.0], degrees=False)
                    rot_quat = rot_euler.as_quat()      # x, y, z, w
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])

                elif self._camera == "env_specific_v2":
                    # NOTE: New config
                    self.model.cam_fovy[index] = 50.0  # FOV
                    self.model.cam_pos[index][0] = 0  # X
                    self.model.cam_pos[index][1] = 0.4  # Y
                    self.model.cam_pos[index][2] = 0.65  # Z
                    rot_euler = Rotation.from_euler('zxy', [180, 30.0, 0.0], degrees=True)
                    rot_quat = rot_euler.as_quat()      # x, y, z, w
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])

            elif self.env.spec.kwargs['env_name'] == "sweep-into-v2":
                if self._camera == "env_specific_v1":
                    # NOTE: old config  (from RL-VLM-F, suitable for invisible arm)
                    self.model.cam_fovy[index] = 45.0  # FOV
                    self.model.cam_pos[index][0] = 0.0  # X
                    self.model.cam_pos[index][1] = 1.1  # Y
                    self.model.cam_pos[index][2] = 0.75  # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('xyz', [-0.5, 0.0, 0.0], degrees=False)
                    rot_quat = rot_euler.as_quat()
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])

                elif self._camera == "env_specific_v2":
                    # NOTE: old config  (from RL-VLM-F)
                    self.model.cam_fovy[index] = 45.0  # FOV
                    self.model.cam_pos[index][0] = 0.0  # X
                    self.model.cam_pos[index][1] = 1.2 # Y
                    self.model.cam_pos[index][2] = 0.7 # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('xyz', [-35.0, 0.0, 0.0], degrees=True)
                    rot_quat = rot_euler.as_quat()
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])

            elif self.env.spec.kwargs['env_name'] == "plate-slide-v2":
                if self._camera == "env_specific_v1":
                    # NOTE: old config  (from RL-VLM-F, suitable for invisible arm)
                    self.model.cam_fovy[index] = 60.0  # FOV
                    self.model.cam_pos[index][0] = 0.25  # X
                    self.model.cam_pos[index][1] = 0.4  # Y
                    self.model.cam_pos[index][2] = 0.6  # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('xyz', [-0.69642844, -0.03207371, -2.45142656], degrees=False)
                    rot_quat = rot_euler.as_quat()
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])

                elif self._camera == "env_specific_v2":
                    # NOTE: new config (TODO: Need to tune more)
                    self.model.cam_fovy[index] = 60.0  # FOV
                    self.model.cam_pos[index][0] = 0.2  # X
                    self.model.cam_pos[index][1] = 0.4  # Y
                    self.model.cam_pos[index][2] = 0.5 # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('zxy', [-135.0, 35.0, 25.0], degrees=True)
                    # rot_euler = Rotation.from_euler('zxy', [-180.0, 0.0, 0.0], degrees=True)
                    rot_quat = rot_euler.as_quat()
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])


            elif self.env.spec.kwargs['env_name'] == "door-open-v2":
                if self._camera == "env_specific_v1":
                    # TODO: this is wrong, please use config in Metaworld/metaworld/envs/assets_v2/objects/assets/xyz_base_transparent.xml
                    # NOTE: old config  (from RL-VLM-F, suitable for invisible arm)
                    self.model.cam_fovy[index] = 60.0  # FOV
                    self.model.cam_pos[index][0] = 0.7  # X
                    self.model.cam_pos[index][1] = 0.35  # Y
                    self.model.cam_pos[index][2] = 0.6  # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('xyz', [3.9, 2.3, 0.6], degrees=False)
                    rot_quat = rot_euler.as_quat()
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])
                elif self._camera == "env_specific_v2":
                    # NOTE: new config
                    self.model.cam_fovy[index] = 50.0  # FOV
                    self.model.cam_pos[index][0] = 0.0  # X
                    self.model.cam_pos[index][1] = 0.1  # Y
                    self.model.cam_pos[index][2] = 0.95  # Z
                    # self.model.cam_mode[index] = 0  # 'fixed',  [fixed, track, trackcom, targetbody, targetbodycom]
                    rot_euler = Rotation.from_euler('zxy', [180.0, 30.0, 0.0], degrees=True)
                    rot_quat = rot_euler.as_quat()
                    self.model.cam_quat[index] = np.array([rot_quat[-1], rot_quat[0], rot_quat[1], rot_quat[2]])

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


        self.env.reset()
        return self._get_image().copy()  # Return the image observation

