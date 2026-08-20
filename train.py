import argparse
import os
import wandb

from research.utils.config import Config
from research.utils.utils import str2bool


def try_wandb_setup(path, config, mode="disabled"):
    # 'debug' runs are always disabled regardless of the requested mode.
    if 'debug' in path:
        mode = 'disabled'

    project_dir = os.path.dirname(__file__)
    exp_dir = os.path.join(project_dir, path)
    project = os.environ.get("WANDB_PROJECT", "VOTP")

    tags = [config.config['eval_env']]
    group = os.path.basename(os.path.dirname(exp_dir))
    name = os.path.basename(exp_dir)
    wandb.init(
        name=name,
        group=group,
        tags=tags,
        dir=exp_dir,
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        config=config.flatten(),
        mode=mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="debug")

    parser.add_argument("--pbrl", type=str2bool, default=True)
    parser.add_argument("--reward_loss_type", type=str, default="ce")
    parser.add_argument("--equal_pref_threshold_teacher", type=float, default=0.0)

    # VOTP parameters
    parser.add_argument("--use_pseudo_label", type=str2bool, default=True)
    parser.add_argument("--preference_threshold", type=float, default=0.0)
    parser.add_argument("--n_labels", type=int, default=10)
    parser.add_argument("--max_pairs", type=int, default=-1)

    parser.add_argument("--d4rl_human_label", type=str2bool, default=False)
    parser.add_argument("--wandb", type=str, default="disabled", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    config = Config.load(args.config)
    assert config.config['env'] is None, f"This code support training offline only, then 'env' must be None"

    config.config['seed'] = args.seed
    config.config['eval_env'] = args.env_name

    sub_exp_name = args.exp_name + f"_s{config.config['seed']}"
    logging_dir = os.path.join(args.logdir, args.env_name, args.exp_name, sub_exp_name)
    os.makedirs(logging_dir, exist_ok=True)

    metaworld_domain = ['mw_door-open-v2', 'mw_drawer-open-v2', 'mw_plate-slide-v2', 'mw_sweep-into-v2']
    locomotion_domain = ['hopper-medium-replay-v2', 'hopper-medium-expert-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2']

    # Hard-coded value for specific domain (please strictly follow dataset layout here)
    if args.env_name in metaworld_domain:
        offline_data_path = f"datasets/metaworld/offline_dataset/{args.env_name}_ep2500_n0.3"
        segment_data_path = f"datasets/metaworld/segments/data/{args.env_name}_nseg20000_size64.npz"
        if args.pbrl:
            config.config['dataset_kwargs']['replay_kwargs']['path'] = offline_data_path
            config.config['dataset_kwargs']['feedback_kwargs']['path'] = segment_data_path
        else:
            config.config['dataset_kwargs']['path'] = offline_data_path

    elif args.env_name in locomotion_domain:
        teacher = "human" if args.d4rl_human_label else "script"
        segment_data_path = f"datasets/locomotion/segments/data/{args.env_name}/num10000_{teacher}.npz"
        if args.pbrl:
            config.config['dataset_kwargs']['replay_kwargs']['name'] = args.env_name    # for loading D4RL dataset
            config.config['dataset_kwargs']['feedback_kwargs']['path'] = segment_data_path
        else:
            config.config['dataset_kwargs']['name'] = args.env_name # for loading D4RL dataset

        config.config['dataset_kwargs']['feedback_kwargs']['human_label'] = args.d4rl_human_label
    else:
        raise ValueError

    # Set common PbRL flags
    if args.pbrl:
        config.config['dataset_kwargs']['feedback_kwargs']['equal_pref_threshold_teacher'] = args.equal_pref_threshold_teacher
        config.config['dataset_kwargs']['feedback_kwargs']['use_pseudo_label'] = args.use_pseudo_label
        config.config['dataset_kwargs']['feedback_kwargs']['preference_threshold'] = args.preference_threshold
        config.config['dataset_kwargs']['feedback_kwargs']['env_name'] = args.env_name
        config.config['dataset_kwargs']['feedback_kwargs']['n_labels'] = args.n_labels
        config.config['dataset_kwargs']['feedback_kwargs']['max_pairs'] = args.max_pairs if args.max_pairs > args.n_labels else (50000 if "mw_" in args.env_name else 10000)

        if 'reward_loss_type' in config.config['alg_kwargs']:
            config.config['alg_kwargs']['reward_loss_type'] = args.reward_loss_type

    # print(config)
    try_wandb_setup(logging_dir, config, mode=args.wandb)
    config.save(logging_dir)  # Save the config

    # Parse the config file to resolve names
    config = config.parse()

    # Get the model
    model = config.get_model()

    # Get the trainer
    trainer = config.get_trainer()

    # Plug the model into trainer
    trainer.set_model(model)

    # Train the model
    trainer.train(logging_dir)
