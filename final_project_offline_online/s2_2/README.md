# Offline to Online RL Final Project

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Examples

Here are some example commands. Run them in the `final_project_offline_online` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```


* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.


* To run 4 jobs on a single GPU in parallel on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000"
  ```

## Notes on this solution

All commands of this solution were run locally (no Modal) with `WANDB_MODE=offline MUJOCO_GL=egl`.

* Training script: `src/scripts/train_offline_online.py` (500K offline + 100K online steps by default in the report). Added options:
  * `--offline_data N`: pre-fill the online replay buffer with `N` offline transitions (Part II).
  * `--wsrl_steps N`: WSRL warm-up with the frozen pre-trained policy for `N` steps (Part II).
  * `--offline_ratio R`: symmetric sampling, a fraction `R` of every online minibatch comes from the offline dataset.
  * `--utd K`: `K` gradient updates per online environment step.
  * `--offline_checkpoint PATH`: skip the offline phase and start from the `offline_checkpoint.pt` of an earlier run (the offline logs of that run are copied, so the new logs cover all steps).
  * `--config_arg key=value`, `--agent_arg key=value`: extra keyword arguments for the config / agent.
* Agents: `sacbc`, `fql` (Part I), `ifql`, `dsrl`, `qsm` (Part II) and `fqlx` (Part III, `src/agents/fqlx_agent.py`).
  The QSM config has an extra option `mean_target` (default 0 = clipped double-Q target as in the handout); `--config_arg mean_target=1` uses the mean of the two target critics, which was needed on antsoccer-arena (see the report).
* Final method (FQL-X):
  ```bash
  uv run src/scripts/train_offline_online.py --run_group=s3_yours --base_config=fqlx --env_name=<ENV> --seed=<SEED> \
      --offline_training_steps 500000 --online_training_steps 100000 --alpha=<300 for cube tasks, 10 for antsoccer> \
      --config_arg num_samples=1 --offline_data 1000000 --utd 4
  ```
* `exp/` holds the runs selected for submission as `exp/s<X>_<method>/<task>/` (`s3_yours/<task>-seed1` is the second seed). `exp_all/` (not part of the zips) holds every raw run; the figures of the report are made from it with `uv run src/scripts/plot.py --runs exp_all --out report`.
* Size limit: to keep `s2_1/submit.zip` and `s2_2/submit.zip` below 100 MB, the `agent.pt` of the cube-single (sanity-check) runs is left out of these two archives; their `eval.csv`, `train.csv`, `flags.json` and `log.pkl` are included.
