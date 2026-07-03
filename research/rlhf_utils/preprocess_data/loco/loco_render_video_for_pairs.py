import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm, trange

import gym

from research.utils.utils import print_yellow, make_video_from_images


video_size = {"medium": (500, 500), "large": (600, 450)}



"""
Code from PT
"""

def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)


def qlearning_mujoco_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    xy_ = []
    done_bef_ = []

    qpos_ = []
    qvel_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    original_indices = []
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)

        qpos = dataset["infos/qpos"][i]
        qvel = dataset["infos/qvel"][i]

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        original_indices.append(i)
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        xy_.append(xy)
        done_bef_.append(done_bef)

        qpos_.append(qpos)
        qvel_.append(qvel)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
        "qposes": np.array(qpos_),
        "qvels": np.array(qvel_),
        "original_indices": np.array(original_indices)
    }


def new_get_trj_idx(env, terminate_on_end=False, **kwargs):
    if not hasattr(env, 'get_dataset'):
        dataset = kwargs['dataset']
    else:
        dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in range(N - 1):
        if env.spec and 'maze' in env.spec.id:
            done_bool = sum(dataset['infos/goal'][i + 1] - dataset['infos/goal'][i]) > 0
        else:
            done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx - 1])
            start_idx = data_idx
            continue
        if done_bool or final_timestep:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1

        episode_step += 1
        data_idx += 1

    trj_idx_list.append([start_idx, data_idx])

    return trj_idx_list


def load_queries_with_indices(env, dataset, num_query, len_query, label_type, saved_indices, saved_labels,
                              balance=False, scripted_teacher=False):
    trj_idx_list = new_get_trj_idx(env, dataset=dataset)  # get_nonmdp_trj_idx(env)

    # to-do: parallel implementation
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1

    assert max(trj_len_list) > len_query

    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))

    observation_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]
    qpos_dim = dataset["qposes"].shape[-1]
    qvel_dim = dataset["qvels"].shape[-1]

    total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
    total_next_obs_seq_1, total_next_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
    total_act_seq_1, total_act_seq_2 = np.zeros((num_query, len_query, action_dim)), np.zeros((num_query, len_query, action_dim))
    total_timestep_1, total_timestep_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros((num_query, len_query), dtype=np.int32)
    total_original_indices_1, total_original_indices_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros((num_query, len_query), dtype=np.int32)

    # tung: add for debug
    total_qpos_seq_1, total_qpos_seq_2 = np.zeros((num_query, len_query, qpos_dim)), np.zeros((num_query, len_query, qpos_dim))
    total_qvel_seq_1, total_qvel_seq_2 = np.zeros((num_query, len_query, qvel_dim)), np.zeros((num_query, len_query, qvel_dim))

    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))

    for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):
        temp_count = 0
        while (temp_count < 2):
            start_idx = saved_indices[temp_count][i]
            end_idx = start_idx + len_query

            reward_seq = dataset['rewards'][start_idx:end_idx]
            obs_seq = dataset['observations'][start_idx:end_idx]
            next_obs_seq = dataset['next_observations'][start_idx:end_idx]
            act_seq = dataset['actions'][start_idx:end_idx]
            timestep_seq = np.arange(1, len_query + 1)

            original_indices_seq = dataset['original_indices'][start_idx:end_idx]

            qpos_seq = dataset['qposes'][start_idx:end_idx]
            qvel_seq = dataset['qvels'][start_idx:end_idx]

            if temp_count == 0:
                total_reward_seq_1[query_count] = reward_seq
                total_obs_seq_1[query_count] = obs_seq
                total_next_obs_seq_1[query_count] = next_obs_seq
                total_act_seq_1[query_count] = act_seq
                total_timestep_1[query_count] = timestep_seq
                total_qpos_seq_1[query_count] = qpos_seq
                total_qvel_seq_1[query_count] = qvel_seq
                total_original_indices_1[query_count] = original_indices_seq
            else:
                total_reward_seq_2[query_count] = reward_seq
                total_obs_seq_2[query_count] = obs_seq
                total_next_obs_seq_2[query_count] = next_obs_seq
                total_act_seq_2[query_count] = act_seq
                total_timestep_2[query_count] = timestep_seq
                total_qpos_seq_2[query_count] = qpos_seq
                total_qvel_seq_2[query_count] = qvel_seq
                total_original_indices_2[query_count] = original_indices_seq

            temp_count += 1

    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()

    seg_obs_1 = total_obs_seq_1.copy()
    seg_obs_2 = total_obs_seq_2.copy()

    seg_next_obs_1 = total_next_obs_seq_1.copy()
    seg_next_obs_2 = total_next_obs_seq_2.copy()

    seq_act_1 = total_act_seq_1.copy()
    seq_act_2 = total_act_seq_2.copy()

    seq_timestep_1 = total_timestep_1.copy()
    seq_timestep_2 = total_timestep_2.copy()

    seg_qpos_1 = total_qpos_seq_1.copy()
    seg_qpos_2 = total_qpos_seq_2.copy()

    seg_qvel_1 = total_qvel_seq_1.copy()
    seg_qvel_2 = total_qvel_seq_2.copy()

    seq_original_index_1 = total_original_indices_1.copy()
    seq_original_index_2 = total_original_indices_2.copy()

    if label_type == 0:  # perfectly rational
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
    elif label_type == 1:
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
        rational_labels[margin_index] = 0.5

    batch = {}
    if scripted_teacher:
        # counter part of human label for comparing with human label.
        batch['labels'] = rational_labels
    else:
        human_labels = np.zeros((len(saved_labels), 2))
        human_labels[np.array(saved_labels) == 0, 0] = 1.
        human_labels[np.array(saved_labels) == 1, 1] = 1.
        human_labels[np.array(saved_labels) == -1] = 0.5
        human_labels = human_labels[query_range]
        batch['labels'] = human_labels
    batch['script_labels'] = rational_labels

    batch['observations'] = seg_obs_1  # for compatibility, remove "_1"
    batch['next_observations'] = seg_next_obs_1
    batch['actions'] = seq_act_1
    batch['observations_2'] = seg_obs_2
    batch['next_observations_2'] = seg_next_obs_2
    batch['actions_2'] = seq_act_2
    batch['timestep_1'] = seq_timestep_1
    batch['timestep_2'] = seq_timestep_2
    batch['start_indices'] = saved_indices[0]
    batch['start_indices_2'] = saved_indices[1]
    batch['reward'] = seg_reward_1
    batch['reward_2'] = seg_reward_2
    batch['qpos'] = seg_qpos_1
    batch['qpos_2'] = seg_qpos_2
    batch['qvel'] = seg_qvel_1
    batch['qvel_2'] = seg_qvel_2
    batch['original_indices_1'] = seq_original_index_1
    batch['original_indices_2'] = seq_original_index_2


    if balance:
        nonzero_condition = np.any(batch["labels"] != [0.5, 0.5], axis=1)
        nonzero_idx, = np.where(nonzero_condition)
        zero_idx, = np.where(np.logical_not(nonzero_condition))
        selected_zero_idx = np.random.choice(zero_idx, len(nonzero_idx))
        for key, val in batch.items():
            batch[key] = val[np.concatenate([selected_zero_idx, nonzero_idx])]
        print(f"size of batch after balancing: {len(batch['labels'])}")

    return batch


def visualize_query(
    gym_env, dataset, batch, query_len, num_query, width=500, height=500, save_dir="./video", seed=1, verbose=False
):
    for seg_idx in trange(num_query):
        start_1, start_2 = (
            batch["start_indices"][seg_idx],
            batch["start_indices_2"][seg_idx],
        )
        frames = []
        frames_2 = []

        start_indices = range(start_1, start_1 + query_len)
        start_indices_2 = range(start_2, start_2 + query_len)

        gym_env.reset()

        if verbose:
            print(f"start pos of first one: {dataset['qposes'][start_indices[0]][:2]}")
            print("=" * 50)
            print(f"start pos of second one: {dataset['qposes'][start_indices_2[0]][:2]}")

        camera_name = "track"

        for t in trange(query_len, leave=False):
            gym_env.set_state(dataset["qposes"][start_indices[t]], dataset["qvels"][start_indices[t]])
            curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
            frames.append(np.flipud(curr_frame))

        gym_env.reset()
        for t in trange(query_len, leave=False):
            gym_env.set_state(dataset["qposes"][start_indices_2[t]], dataset["qvels"][start_indices_2[t]])
            curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
            frames_2.append(np.flipud(curr_frame))

        make_video_from_images(os.path.join(save_dir, f"res{width}_seg{seg_idx}_1.mp4"), np.array(frames))
        make_video_from_images(os.path.join(save_dir, f"res{width}_seg{seg_idx}_2.mp4"), np.array(frames_2))
        # video = np.concatenate((np.array(frames), np.array(frames_2)), axis=2)
        # make_video_from_images(os.path.join(save_dir, f"./idx{seg_idx}.mp4"), video)


    print("save query indices.")
    np.savez(
        os.path.join(save_dir, f"human_indices_numq{num_query}_len{query_len}_s{seed}.npz"),
        start_indices_1=batch["start_indices"],
        start_indices_2=batch["start_indices_2"],
        reward_1=batch["reward"],
        reward_2=batch["reward_2"],
        script_labels=batch["script_labels"],
        obs_1=batch["observations"],
        obs_2=batch["observations_2"],
        action_1=batch["actions"],
        action_2=batch["actions_2"],
        qpos_1=batch["qpos"],
        qpos_2=batch["qpos_2"],
        qvel_1=batch["qvel"],
        qvel_2=batch["qvel_2"],
        original_indices_1=batch["original_indices_1"],
        original_indices_2=batch["original_indices_2"]
    )


if __name__ == "__main__":
    num_query = 10000
    query_len = 100
    width = height = 256
    label_type = 1

    # for env_name in env_names:
    for env_name in ["hopper-medium-expert-v2", "hopper-medium-replay-v2", "walker2d-medium-expert-v2", "walker2d-medium-replay-v2"]:
        gym_env = gym.make(env_name)
        set_seed(gym_env, seed=1)

        print(f"Env name: {env_name}")
        save_dir = f"datasets/locomotion/segments/video"
        save_dir = os.path.join(save_dir, gym_env.spec.id)
        os.makedirs(save_dir, exist_ok=True)

        ds = qlearning_mujoco_dataset(gym_env)

        if num_query == 10000:
            query_path = "datasets/locomotion/segments/data/human_label_10000"
        else:
            raise ValueError
        base_path = os.path.join(query_path, env_name)
        human_indices_2_file, human_indices_1_file, _ = sorted(os.listdir(base_path))

        with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:  # Unpickling
            human_indices = pickle.load(fp)
        with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:  # Unpickling
            human_indices_2 = pickle.load(fp)
        human_labels = None

        batch = load_queries_with_indices(
            gym_env,
            ds,
            saved_indices=[human_indices, human_indices_2],
            saved_labels=human_labels,
            num_query=num_query,
            len_query=query_len,
            label_type=1,
            scripted_teacher=True
        )

        # Dump PT dataset to IPL format
        # pref_label_type = "script"
        # # pref_label_type = "human"
        #
        # path = "datasets/locomotion/segments/data"
        # if pref_label_type == "script":
        #     filename = f"{path}/{env_name}/num10000_script_train_extra_info.npz"
        #     pref_label = np.argmax(batch["labels"], axis=1).astype(np.float64)
        # else:
        #     filename = f"{path}/{env_name}/num10000_human_train_extra_info.npz"
        #     num_q = 500 if "medium-replay" in env_name else 100
        #     pref_label = np.load(os.path.join(path, env_name, f"num{num_q}_human_train.npz"))
        #     pref_label = pref_label['label']
        #
        # np.savez(
        #     filename,
        #     obs_1=batch["observations"],
        #     obs_2=batch["observations_2"],
        #     action_1=batch["actions"],
        #     action_2=batch["actions_2"],
        #     reward_1=batch["reward"],
        #     reward_2=batch["reward_2"],
        #     label=pref_label,
        #     script_labels=batch["script_labels"],
        #     start_indices_1=batch["start_indices"],
        #     start_indices_2=batch["start_indices_2"],
        #     original_indices_1=batch["original_indices_1"],
        #     original_indices_2=batch["original_indices_2"],
        # )

        # TODO: render segments
        visualize_query(gym_env, ds, batch, query_len, num_query, width=width, height=height, save_dir=save_dir)