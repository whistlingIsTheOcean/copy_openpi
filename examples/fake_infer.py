"""Offline "fake" inference for pi0: breakpoint-debug the model output without a real robot.

Modes:
  dummy          : random tiny model, no download, no checkpoint       [default]
  real_structure : real pi0 architecture (2B+300M), random init, no download
  pi0_base       : real pi0_base checkpoint (auto-downloads ~3GB on first run)

Usage:
  uv run --no-sync examples/fake_infer.py --mode dummy
  uv run --no-sync examples/fake_infer.py --mode pi0_base

Or set a breakpoint on the `actions =` line and step into `sample_actions` to inspect the
denoising / flow-matching internals.
"""
import argparse

import jax
import numpy as np
from flax import nnx

from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import weight_loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dummy", "real_structure", "pi0_base"],
        default="pi0_base",
        help="Which model to run: fake tiny model / real random-init / real pi0_base ckpt.",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Number of denoising steps.")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    # 1) Pick the model config.
    #    "dummy" variant = random tiny weights, no download. Everything else uses the real
    #    pi0 architecture (paligemma 2B + action expert 300M).
    if args.mode == "dummy":
        config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    else:
        config = pi0_config.Pi0Config()

    # 2) Initialize the model (random weights).
    model = config.create(jax.random.key(0))

    # 3) Optionally load the real pi0_base checkpoint (auto-downloads on first run).
    if args.mode == "pi0_base":
        print("Loading pi0_base weights (gs://openpi-assets/checkpoints/pi0_base/params)...")
        loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params")
        graphdef, state = nnx.split(model)
        partial = loader.load(state.to_pure_dict())
        state.replace_by_pure_dict(partial)
        model = nnx.merge(graphdef, state)
        print("pi0_base weights loaded.")

    # 4) Fake observation: official random-input constructor (images + state + prompt tokens).
    obs = config.fake_obs(batch_size=args.batch_size)
    print("obs keys:", sorted(obs.to_dict().keys()))

    # 5) Sample actions. SET BREAKPOINT HERE / step into `sample_actions`.
    actions = model.sample_actions(jax.random.key(1), obs, num_steps=args.num_steps)
    actions = np.asarray(actions)
    print("actions shape:", actions.shape)  # (batch, action_horizon=50, action_dim=32)
    print("actions[0,0,:10]:", actions[0, 0, :10])

    # NOTE: with `dummy` the values are random noise (weights are random). With `pi0_base`
    # they are real (normalized) actions from the pre-trained model, but still need
    # unnormalization + your robot-specific postprocessing before they mean anything.


if __name__ == "__main__":
    main()
