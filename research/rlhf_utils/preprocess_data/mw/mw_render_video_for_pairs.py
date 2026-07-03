import os
from tqdm import tqdm

import gym
import numpy as np

from research.datasets import storage
from research.envs.metaworld import MetaWorldSawyerImageEnv
from research.utils.utils import make_video_from_images


def main():
    all_envs = ["mw_door-open-v2", "mw_drawer-open-v2", "mw_plate-slide-v2", "mw_sweep-into-v2"]
    resolution = 256
    camera = "corner2_v2"
    for env_name in all_envs:
        data_path = f"datasets/metaworld/segments/data/{env_name}_nseg20000_size64.npz"
        video_save_dir = "datasets/metaworld/segments/video"

        data = storage.load_data(data_path, exclude_keys=["mask"])
        assert "state" in data

        save_path = os.path.join(video_save_dir, env_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        env = gym.make(env_name)
        env = MetaWorldSawyerImageEnv(env, width=resolution, height=resolution, camera=camera)
        env.reset()  # Moves the camera

        num_segments, segment_length = data["obs"].shape[:2]
        assert num_segments == 20000 and segment_length == 64, "Wrong data (we used data from CPL paper), please check again."

        for segment_idx in tqdm(range(0, num_segments), desc=f"Rendering segments - {env_name}"):
            clip_segment = []
            for t in range(segment_length):
                env.set_state(data["state"][segment_idx, t])    # Set the environment into a state
                img = env._get_image()    # CxHxW
                clip_segment.append(img)

            clip_segment = np.array(clip_segment)  # SxCxHxW
            clip_segment = clip_segment.transpose(0, 2, 3, 1)   # SxCxHxW -> # SxHxWxC

            make_video_from_images(f"{save_path}/{camera}_res{resolution}_seg{segment_idx}.mp4", frames=clip_segment)


if __name__ == "__main__":
    main()

