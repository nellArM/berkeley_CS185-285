import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


def build_agent(config: dict, args: argparse.Namespace):
    """Create the environment, the offline dataset and a fresh agent."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env, dataset = config["make_env_and_dataset"]()

    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch['observations'].shape[1:],
        example_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )
    return env, dataset, agent


def agent_optimizers(agent) -> dict:
    return {k: v for k, v in vars(agent).items() if isinstance(v, torch.optim.Optimizer)}


def save_checkpoint(agent, train_logger, eval_logger, path: str) -> str:
    """Save networks, optimizer states and the log rows so far (used to hand over to the online phase)."""
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in agent_optimizers(agent).items()},
            "train_rows": train_logger.rows,
            "eval_rows": eval_logger.rows,
        },
        path,
    )
    return path


def load_checkpoint(agent, path: str) -> dict:
    ckpt = torch.load(path, map_location=ptu.device, weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    for k, opt in agent_optimizers(agent).items():
        if k in ckpt["optimizers"]:
            opt.load_state_dict(ckpt["optimizers"][k])
    return ckpt


def update_agent(agent, batch: dict, step: int) -> dict:
    batch = {k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
    return agent.update(
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["next_observations"],
        batch["dones"],
        step,
    )


def evaluate(env, agent, args: argparse.Namespace, eval_logger, step: int) -> float:
    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    trajectories = utils.sample_n_trajectories(env, agent, args.num_eval_trajectories, ep_len)
    success = float(np.mean([t["episode_statistics"]["s"] for t in trajectories]))
    eval_logger.log({"eval/success_rate": success}, step=step)
    return success


def run_offline_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, start_step: int = 0):
    """
    Run offline training loop.

    Returns the path of the offline checkpoint and a dict with the live objects
    (env, dataset, agent) so the online phase can continue with the same agent and optimizers.
    """
    env, dataset, agent = build_agent(config, args)
    checkpoint_path = os.path.join(args.save_dir, "offline_checkpoint.pt")

    if args.offline_checkpoint is not None:
        # Reuse an offline phase that was already trained with identical settings:
        # restore networks + optimizers and replay its log rows into this run's logs.
        ckpt = load_checkpoint(agent, args.offline_checkpoint)
        for row in ckpt["train_rows"]:
            row = dict(row)
            train_logger.log(row, step=row.pop("step"))
        for row in ckpt["eval_rows"]:
            row = dict(row)
            eval_logger.log(row, step=row.pop("step"))
        return args.offline_checkpoint, {"env": env, "dataset": dataset, "agent": agent}

    for step in tqdm.trange(start_step, start_step + args.offline_training_steps + 1, dynamic_ncols=True):
        batch = dataset.sample(config["batch_size"])
        metrics = update_agent(agent, batch, step)

        if step % args.log_interval == 0:
            metrics["phase"] = 0  # 0 = offline, 1 = online
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            evaluate(env, agent, args, eval_logger, step)

    save_checkpoint(agent, train_logger, eval_logger, checkpoint_path)
    return checkpoint_path, {"env": env, "dataset": dataset, "agent": agent}


def prefill_with_offline_data(replay_buffer: ReplayBuffer, dataset: ReplayBuffer, num_transitions: int) -> None:
    """Copy `num_transitions` randomly chosen offline transitions into the (empty) replay buffer."""
    n = min(num_transitions, dataset.size, replay_buffer.max_size)
    idx = np.random.choice(dataset.size, size=n, replace=False)
    for name in ["observations", "actions", "rewards", "next_observations", "dones"]:
        src = getattr(dataset, name)
        dst = np.empty((replay_buffer.max_size, *src.shape[1:]), dtype=src.dtype)
        dst[:n] = src[idx]
        setattr(replay_buffer, name, dst)
    replay_buffer.size = n


def sample_online_batch(replay_buffer: ReplayBuffer, dataset: ReplayBuffer, batch_size: int, offline_ratio: float) -> dict:
    """Sample a batch for an online update; optionally mix in a fixed fraction of offline data."""
    n_offline = int(round(batch_size * offline_ratio))
    n_online = batch_size - n_offline
    if n_offline == 0:
        return replay_buffer.sample(batch_size)
    offline = dataset.sample(n_offline)
    if n_online == 0:
        return offline
    online = replay_buffer.sample(n_online)
    return {k: np.concatenate([offline[k], online[k]], axis=0) for k in offline}


def run_online_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, agent_path: str, start_step: int = 0, state: dict = None):
    """
    Run online training loop.

    `state` holds the live env / dataset / agent from the offline phase. If it is None, they are
    rebuilt and the agent is restored from `agent_path`.
    """
    if state is None:
        env, dataset, agent = build_agent(config, args)
        if agent_path is not None:
            load_checkpoint(agent, agent_path)
    else:
        env, dataset, agent = state["env"], state["dataset"], state["agent"]

    # Agents may switch hyperparameters for the online phase (used by the custom method).
    if hasattr(agent, "set_online"):
        agent.set_online(True)

    batch_size = config["batch_size"]
    capacity = max(args.replay_buffer_capacity, args.offline_data + args.online_training_steps)
    replay_buffer = ReplayBuffer(capacity=capacity)

    # Section 4.1: retain some offline data by pre-filling the replay buffer with it.
    if args.offline_data > 0:
        prefill_with_offline_data(replay_buffer, dataset, args.offline_data)

    online_eval_interval = args.online_eval_interval or args.eval_interval

    observation, _ = env.reset()
    metrics = None
    for i in tqdm.trange(1, args.online_training_steps + 1, dynamic_ncols=True):
        step = start_step + i

        # Collect one transition with the current policy.
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        replay_buffer.insert(
            observation=np.asarray(observation, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=np.asarray(reward, dtype=np.float32),
            next_observation=np.asarray(next_observation, dtype=np.float32),
            done=np.asarray(float(terminated), dtype=np.float32),  # time-limit truncation is not a terminal
        )
        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation

        # Section 4.2 (WSRL): only collect data with the frozen offline policy for the first
        # `wsrl_steps` steps, then start updating.
        warmed_up = i > args.wsrl_steps
        have_data = args.offline_ratio >= 1.0 or replay_buffer.size >= batch_size
        if warmed_up and have_data:
            for _ in range(args.utd):
                batch = sample_online_batch(replay_buffer, dataset, batch_size, args.offline_ratio)
                metrics = update_agent(agent, batch, step)

        if i % args.log_interval == 0 and metrics is not None:
            metrics["phase"] = 1
            train_logger.log(metrics, step=step)

        if i % online_eval_interval == 0 or i == args.online_training_steps:
            evaluate(env, agent, args, eval_logger, step)
            # Evaluation shares the environment with data collection, so start a fresh episode.
            observation, _ = env.reset()

    return agent


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    
    # Online retention of offline data
    # Number of offline transitions copied into the replay buffer before the online phase starts
    parser.add_argument("--offline_data", type=int, default=0)
    # Fraction of every online batch drawn directly from the offline dataset (RLPD-style symmetric sampling)
    parser.add_argument("--offline_ratio", type=float, default=0.0)

    # WSRL
    # Number of initial online steps that only collect data with the frozen offline policy
    parser.add_argument("--wsrl_steps", type=int, default=0)

    # Extra options
    parser.add_argument("--utd", type=int, default=1)  # gradient updates per online environment step
    parser.add_argument("--online_eval_interval", type=int, default=None)  # defaults to --eval_interval
    parser.add_argument("--offline_checkpoint", type=str, default=None)  # reuse a finished offline phase
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--agent_arg", action="append", default=[])  # extra agent kwargs as key=value
    parser.add_argument("--config_arg", action="append", default=[])  # extra config kwargs (e.g. n_ensembles=4)


    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    def parse_value(value: str):
        return float(value) if "." in value or "e" in value else int(value)

    config_kwargs = {k: parse_value(v) for k, v in (item.split("=", 1) for item in args.config_arg)}
    config = configs.configs[args.base_config](args.env_name, **config_kwargs)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    
    config['offline_data'] = args.offline_data
    config['offline_ratio'] = args.offline_ratio
    config['wsrl_steps'] = args.wsrl_steps
    config['utd'] = args.utd
    config['offline_checkpoint'] = args.offline_checkpoint

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.discount is not None:
        config['agent_kwargs']['discount'] = args.discount
        exp_name = f"{exp_name}_g{args.discount}"
    for key, value in config_kwargs.items():
        exp_name = f"{exp_name}_{key}{value}"
    for item in args.agent_arg:
        key, value = item.split("=", 1)
        config['agent_kwargs'][key] = float(value) if "." in value or "e" in value else int(value)
        exp_name = f"{exp_name}_{key}{value}"
    if args.offline_data > 0:
        exp_name = f"{exp_name}_od{args.offline_data}"
    if args.offline_ratio > 0:
        exp_name = f"{exp_name}_or{args.offline_ratio}"
    if args.wsrl_steps > 0:
        exp_name = f"{exp_name}_ws{args.wsrl_steps}"
    if args.utd > 1:
        exp_name = f"{exp_name}_utd{args.utd}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    agent_path, state = None, None
    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        # TODO(student): Implement offline training loop
        # Hint: You might consider passing the agent's path to the online training loop
        agent_path, state = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        start_step = args.offline_training_steps
        
    
    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        # TODO(student): Implement online training loop
        agent = run_online_training_loop(config, train_logger, eval_logger, args, agent_path, start_step=start_step, state=state)
    else:
        agent = state["agent"]

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
