# PrismWAM

**Contributor:** PrismWAM contributors | **Paper:** [arXiv:2610.26564](https://arxiv.org/abs/2610.26564) | **Original code:** [THUSI-Lab/PrismWAM](https://github.com/THUSI-Lab/PrismWAM) | **Weights:** [🤗 THUSI-Lab/PrismWAM](https://huggingface.co/THUSI-Lab/PrismWAM)

PrismWAM RGB/depth policies for the RoboDojo and RoboTwin joint-action recipes. LIBERO
(delta end-effector) is not exposed here. Argument conventions, checkpoint naming,
split-machine deployment and `EVAL_ENV_TYPE` follow the
[XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab#readme). Results:
[RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install this directory as `XPolicyLab/policy/PrismWAM`. Set up PrismWAM following its
[installation](https://github.com/THUSI-Lab/PrismWAM/blob/main/docs/installation.md), then,
in that environment, add the XPolicyLab transport. The simulator client keeps its own
environment.

```bash
export PRISMWAM_ROOT=/path/to/PrismWAM
cd /path/to/XPolicyLab/policy/PrismWAM
bash install.sh
```

`POLICY_ENV` in `eval.sh` takes a venv path, a conda env name, or `current`.

Robot metadata comes from the benchmark installation: `<workspace>/XPolicyLab/` must sit
beside `<workspace>/env_cfg/<robot>.yml` and `<workspace>/env_cfg/robot/_robot_info.json`.

## Data Processing

Training reads the native RGB/depth HDF5 recordings (not the LeRobot 2.1/3.0 exports) plus
the recipe's statistics and text cache.

```bash
bash process_data.sh RoboDojo my_run arx_x5 joint robodojo-index \
  --root /path/to/data/robodojo --output /path/to/data/robodojo/index.json
```

Arguments after the first four go to `PrismWAM/scripts/prepare_data.py`. This only builds
the index; see [data preparation](https://github.com/THUSI-Lab/PrismWAM/blob/main/docs/data.md)
for the statistics and text cache. The robot must be registered the same way in the
benchmark's runtime registry and in XPolicyLab's training registry.

## Training

`train.sh` runs one recipe stage; hyperparameters live in `PrismWAM/configs/<recipe>/<stage>.yaml`.
Activate the policy environment first.

```bash
bash train.sh BENCH CHECKPOINT ROBOT joint SEED GPU_IDS RECIPE stage1_depth|stage2_joint [KEY=VALUE...]

# RoboDojo, Stage 2, on node 0 of 4 (the global batch of 1024 is kept on any GPU count).
export NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1
DRY_RUN=1 bash train.sh RoboDojo my_run arx_x5 joint 42 0,1,2,3,4,5,6,7 \
  robodojo stage2_joint paths.data_root=/path/to/data paths.weights_root=/path/to/weights
```

Run `stage1_depth` first, on every node with its own `NODE_RANK`. Outputs go to
`outputs/<official-run-name>/<recipe>/<stage>/`, and Stage 2 loads the Stage-1 checkpoint
from there. `SEED` replaces the recipe seed (42); `KEY=VALUE` overrides the recipe; set
`PYTHON` if the interpreter is not `python`. LIBERO, non-joint and the mixed 7D/14D real-world recipes use
`PrismWAM/scripts/<recipe>/` instead.

## Evaluation

Evaluation needs a model folder, released or exported from your Stage-2 checkpoint; a raw `.pt`
file is not enough.

```bash
cd "$PRISMWAM_ROOT"
# Released policy:
hf download THUSI-Lab/PrismWAM --include "robodojo/*" --local-dir weights
export MODEL="$PRISMWAM_ROOT/weights/robodojo"
# Or your own training run:
RUN=/path/to/XPolicyLab/policy/PrismWAM/outputs/my_run/robodojo/stage2_joint
export MODEL=/path/to/XPolicyLab/policy/PrismWAM/checkpoints/my_run
python scripts/export_model.py \
  --config "$RUN/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_040000.pt" \
  --output "$MODEL"

cd /path/to/XPolicyLab/policy/PrismWAM
export PRISMWAM_RECIPE=robodojo
export PRISMWAM_DEPTH_KEY=approximate_depth PRISMWAM_DEPTH_SCALE=1
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$MODEL" arx_x5 joint \
  0 0 0 /path/to/envs/prismwam robodojo
```

All launchers accept `DRY_RUN=1`. For split-machine evaluation, use the standard
server/client scripts as in the upstream guide.

`ckpt_name` is an absolute model path, a path relative to this directory, or a name under
`checkpoints/`; `model_path` in `deploy.yml` or `PRISMWAM_MODEL` takes precedence.
For the mixed real-world checkpoint, export one model folder per task with `--task-name <task>`
and include that task's training text cache with `--text-cache`. Single-arm tasks
normalize their 7 real values before padding, and the output is unpadded before
denormalization.

## Configuration

| Key | Meaning |
| --- | --- |
| `recipe` | Recipe ID (required); or `PRISMWAM_RECIPE`. |
| `model_path` | Model folder, released or exported. Not downloaded automatically. |
| `camera_map` | Maps external/left/right to observation cameras; default `cam_head`, `cam_left_wrist`, `cam_right_wrist`. |
| `depth_scale` | Multiplier from raw depth to millimetres: `1` for mm, `1000` for metres; or `PRISMWAM_DEPTH_SCALE`. |
| `depth_key` | Depth field in each camera; default `depth`, or `PRISMWAM_DEPTH_KEY`. No fallback when missing. |
| `device`, `dtype` | Default `cuda`, `bfloat16`; or `PRISMWAM_DEVICE`, `PRISMWAM_DTYPE`. |
| `seed` | Scene/layout seed from the evaluation launcher. |
| `inference_seed` | Model-noise seed. Default `null` reuses `seed`, as in the evaluated runs; or `PRISMWAM_INFERENCE_SEED`. |
| `eval_batch` | Default false; state stays per environment, inference runs sequentially on shared weights. |

Denoising steps, replanning interval and normalization always come from the model folder.
Images must arrive as decoded RGB; the adapter does not decode or swap channels. One server
serves one evaluation client, which may batch several environments; run a separate server
and port for each concurrent client.

## Notes

- The model must match the robot and action representation; dimensions come from
  XPolicyLab's robot helpers.
- Joint recipes order actions as left-arm joints and gripper, then right-arm joints and
  gripper.
- Released weights: [THUSI-Lab/PrismWAM](https://huggingface.co/THUSI-Lab/PrismWAM)
  (`robotwin/`, `robodojo/`) and, for RoboTwin, PrismWAM\* at
  [THUSI-Lab/PrismWAM-Star](https://huggingface.co/THUSI-Lab/PrismWAM-Star) (`robotwin/`).
