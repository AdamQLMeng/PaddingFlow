import os
from typing import List, Optional
import argparse
from collections import namedtuple

import torch
import numpy as np
import tqdm
from jrl.robots import Robot, get_robot

from model.ikflow_solver import IKFlowSolver
from model.lt_data import IkfLitDataset
from model.config import DATASET_TAG_NON_SELF_COLLIDING

from utils.utils import set_seed, boolean_string
from utils.evaluation_utils import evaluate_solutions

set_seed()

_DEFAULT_LATENT_DISTRIBUTION = "gaussian"
_DEFAULT_LATENT_SCALE = 1.
_DEFAULT_SAMPLES_PER_POSE = 100
ErrorStats = namedtuple(
    "ErrorStats", "mean_l2_error_mm mean_angular_error_deg pct_joint_limits_exceeded pct_self_colliding"
)


def calculate_error_stats(
    ik_solver: IKFlowSolver,
    robot: Robot,
    testset: np.ndarray,
    latent_distribution: str,
    latent_scale: float,
    samples_per_pose: int,
) -> ErrorStats:
    """Evaluate the given `ik_solver` on the provided `testset`.

    NOTE: Returns positional error in millimeters and rotational error in degrees
    """
    ik_solver.nn_model.eval()

    l2_errs: List[List[float]] = []
    ang_errs: List[List[float]] = []
    jlims_exceeded_count = 0
    self_collisions_count = 0
    B = testset.shape[0]
    with torch.inference_mode():
        with tqdm.tqdm(desc="Evaluate model", total=B) as pbar:
            for i in range(B):
                pbar.update(1)
                ee_pose_target = testset[i]
                samples = ik_solver.solve(
                    ee_pose_target,
                    samples_per_pose,
                    latent_distribution=latent_distribution,
                    latent_scale=latent_scale,
                    clamp_to_joint_limits=False,
                    refine_solutions=False,
                )
                l2_errors, ang_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
                    robot, ee_pose_target, samples
                )
                l2_errs.append(l2_errors)
                ang_errs.append(ang_errors)
                jlims_exceeded_count += joint_limits_exceeded.sum().item()
                self_collisions_count += self_collisions.sum().item()
    n_total = testset.shape[0] * samples_per_pose
    return ErrorStats(
        float(1000 * np.mean(l2_errs)),
        float(np.rad2deg(np.mean(ang_errors))),
        100 * (jlims_exceeded_count / n_total),
        100 * (self_collisions_count / n_total),
    )


def pp_results(args: argparse.Namespace, error_stats: ErrorStats):
    print(f"\n----------------------------------------")
    print(f"> Results for {args.model_file}")
    print(f"\n  Average positional error:              {round(error_stats.mean_l2_error_mm, 4)} mm")
    print(f"  Average rotational error:         {round(error_stats.mean_angular_error_deg, 4)} deg")
    print(f"  Percent joint limits exceeded: {round(error_stats.pct_joint_limits_exceeded, 4)} %")
    print(f"  Percent self-colliding:        {round(error_stats.pct_self_colliding, 4)} %")


def evaluate_model(args):
    assert os.path.exists(args.model_file), f"checkpoint is not found! ({args.model_file})"

    print("Resume from: ", args.model_file)
    checkpoint = torch.load(args.model_file, map_location='cpu')
    print(f"Epochs: {checkpoint['epoch']}")
    print(f"Global steps: {checkpoint['global_step']}")
    base_hparams = checkpoint["hyper_parameters"]["base_hparams"]
    print("base hyperparameters:")
    print(base_hparams)
    if base_hparams.noise_type == "soft":
        base_hparams.softflow_enabled = True
    else:
        base_hparams.softflow_enabled = False
    robot = get_robot(args.robot_name)
    ik_solver = IKFlowSolver(base_hparams, robot)
    print("dim_cond", ik_solver.dim_cond)
    print("_network_width", ik_solver._network_width)
    state_dict = dict()
    for k, v in checkpoint['state_dict'].items():
        if k[:8] == "nn_model":
            k = k[9:]  # remove the head (nn_model.)
        state_dict[k] = v
    ik_solver.nn_model.load_state_dict(state_dict)

    data_module = IkfLitDataset(robot.name, 1, args.testset_size, args.dataset_tags)
    testset = data_module._endpoints_te[:args.testset_size]
    print(f"evaluate on {len(testset)} ik problems!")
    error_stats = calculate_error_stats(
        ik_solver,
        robot,
        testset,
        _DEFAULT_LATENT_DISTRIBUTION,
        _DEFAULT_LATENT_SCALE,
        _DEFAULT_SAMPLES_PER_POSE
    )
    pp_results(args, error_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--non_self_colliding_dataset", type=str, default="true")
    parser.add_argument("--testset_size", default=150000, type=int)
    parser.add_argument("--robot_name", type=str, default="panda")
    parser.add_argument("--model_file", type=str)
    args = parser.parse_args()
    args.non_self_colliding_dataset = boolean_string(args.non_self_colliding_dataset)
    args.dataset_tags = []
    if args.non_self_colliding_dataset:
        args.dataset_tags.append(DATASET_TAG_NON_SELF_COLLIDING)

    # Build IKFlowSolver and set weights
    print("\n-------------")
    print(f"Evaluating model '{args.model_file}'")
    evaluate_model(args)
