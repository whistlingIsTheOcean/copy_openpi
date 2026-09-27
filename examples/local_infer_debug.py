"""Local (no-server) JAX inference debug for pi0 / pi0-LoRA.

Runs the FULL inference pipeline exactly like the policy server does, but on a REAL
observation pulled from your LeRobot dataset, so you can set breakpoints and inspect
every stage:

  observation in "inference key" format
    -> Policy.infer()                                   src/openpi/policies/policy.py
       1. input transforms (see create_trained_policy)  src/openpi/policies/policy_config.py
            InjectDefaultPrompt -> LiberoInputs -> [DeltaActions]
            -> Normalize -> ResizeImages -> TokenizePrompt -> PadStatesAndActions
       2. batching + Observation.from_dict              src/openpi/models/model.py
       3. model.sample_actions (flow-matching denoise)  src/openpi/models/pi0.py
       4. output transforms: Unnormalize -> AbsoluteActions -> LiberoOutputs
    -> ABSOLUTE joint action chunk, shape (action_horizon, 7) = 6 joints + gripper

Usage:
  export OPENPI_DATA_HOME=/data/yuxuan/openpi_cache
  export HF_LEROBOT_HOME=/data/yuxuan/lerobot_home
  export CUDA_VISIBLE_DEVICES=1

  uv run --no-sync examples/local_infer_debug.py \
      --config-name pi0_realdata_lora \
      --checkpoint checkpoints/pi0_realdata_lora/my_exp/19999 \
      --episode 0 --frame 10

Debugging flags:
  --no-jit       run sample_actions eagerly (so breakpoints INSIDE the denoising loop are hit)
  --num-steps N  number of denoising steps (default 10; training default is 10)
  --no-gt        skip loading ground-truth actions (use when the dataset is unavailable)
"""

import dataclasses
import time

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Must match the config used for training (model architecture / transforms / norm stats).
    config_name: str = "pi0_realdata_lora"
    # Checkpoint dir, e.g. checkpoints/pi0_realdata_lora/my_exp/19999 (contains params/ + assets/).
    checkpoint: str = "checkpoints/pi0_realdata_lora/my_exp/19999"
    # LeRobot repo id, resolved under HF_LEROBOT_HOME.
    repo_id: str = "real_data0710/libero"
    # Which dataset frame to use as the observation.
    episode: int = 0
    frame: int = 0
    # Prompt. If None, the dataset's own task string for that frame is used.
    prompt: str | None = "Pick up the colored cup and move it into the metal cup"
    # Denoising steps.
    num_steps: int = 10
    # Disable jit on sample_actions so breakpoints inside the denoising loop hit.
    no_jit: bool = False
    # Skip ground-truth comparison (when the dataset is not available).
    no_gt: bool = False


def _to_numpy(x):
    """Torch tensor / jax array / numpy -> numpy (CPU)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def load_dataset_frame(repo_id: str, episode: int, frame: int, action_horizon: int) -> tuple[dict, np.ndarray]:
    """Pull one frame from the LeRobot dataset and build the inference-format observation.

    The dataset stores ABSOLUTE joint targets, same space that Policy.infer returns,
    so the returned `actions` can be compared with the prediction directly.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    # delta_timestamps mirrors what openpi does during training: a chunk of `action_horizon` future actions.
    ds = LeRobotDataset(repo_id, delta_timestamps={"actions": [t / meta.fps for t in range(action_horizon)]})

    # Global index of (episode, frame).
    start = sum(meta.episodes[i]["length"] for i in range(episode))
    idx = start + frame
    sample = ds[idx]

    obs = {
        "observation/image": _to_numpy(sample["image"]),
        "observation/wrist_image": _to_numpy(sample["wrist_image"]),
        "observation/state": _to_numpy(sample["state"]),
        "prompt": str(meta.tasks[int(sample["task_index"])]),
    }
    gt_actions = _to_numpy(sample["actions"])  # (action_horizon, 7) absolute
    return obs, gt_actions


def main(args: Args) -> None:
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    cfg = _config.get_config(args.config_name)
    print(f"config={cfg.name}  action_horizon={cfg.model.action_horizon}  action_dim={cfg.model.action_dim}")

    policy = _policy_config.create_trained_policy(
        cfg,
        args.checkpoint,
        default_prompt=args.prompt,
        sample_kwargs={"num_steps": args.num_steps},
    )
    print(f"policy loaded from {args.checkpoint}")

    if args.no_jit:
        # Policy.__init__ wraps sample_actions in nnx_utils.module_jit. Bypass it so you can
        # step into src/openpi/models/pi0.py::sample_actions and watch each denoising step.
        policy._sample_actions = policy._model.sample_actions
        print("[no-jit] sample_actions will run eagerly")

    gt_actions = None
    if args.no_gt:
        # Fake observation in the correct inference format.
        obs = {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(7, dtype=np.float32),
            "prompt": args.prompt,
        }
        print("[no-gt] using a zero observation")
    else:
        obs, gt_actions = load_dataset_frame(
            args.repo_id, args.episode, args.frame, cfg.model.action_horizon
        )
        print(f"observation from {args.repo_id} episode={args.episode} frame={args.frame}")

    print("obs:", {k: (np.asarray(v).shape, np.asarray(v).dtype) for k, v in obs.items()})
    print("state:", obs["observation/state"])
    print("prompt:", obs["prompt"])

    # ---------------------------------------------------------------- BREAKPOINT HERE
    t0 = time.monotonic()
    out = policy.infer(obs)
    elapsed_ms = (time.monotonic() - t0) * 1000

    pred = np.asarray(out["actions"])
    print(f"\npredicted actions: shape={pred.shape} dtype={pred.dtype}  ({elapsed_ms:.1f} ms)")
    print("pred[0]:", pred[0])
    print("policy_timing:", out.get("policy_timing"))

    if gt_actions is not None:
        print(f"\nground-truth actions: shape={gt_actions.shape}")
        print("gt[0]  :", gt_actions[0])
        # Both are absolute joint targets (6 joints + gripper width), so a direct diff is meaningful.
        diff = np.abs(pred - gt_actions[: len(pred)])
        print(f"\nmean |pred - gt| per dim: {diff.mean(axis=0)}")
        print(f"mean |pred - gt| overall : {diff.mean():.4f}")
        print("\n(first 6 dims are joint angles; 7th is gripper width)")
    print("\nINFERENCE DEBUG DONE")


if __name__ == "__main__":
    main(tyro.cli(Args))
