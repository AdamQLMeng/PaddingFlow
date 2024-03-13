import argparse
import os

from jrl.robots import get_robot

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
from pytorch_lightning import Trainer, seed_everything
import torch

from model.config import DATASET_TAG_NON_SELF_COLLIDING

from model.ikflow_solver import IKFlowSolver
from model.lt_model import IkfLitModel, IkflowModelParameters
from model.lt_data import IkfLitDataset
from utils.utils import boolean_string


DEFAULT_MAX_EPOCHS = 2000
SEED = 0
seed_everything(SEED, workers=True)

# Model parameters
DEFAULT_COUPLING_LAYER = "glow"
DEFAULT_RNVP_CLAMP = 2.5
DEFAULT_SOFTFLOW_NOISE_SCALE = 0.001
DEFAULT_SOFTFLOW_ENABLED = True
DEFAULT_N_NODES = 12
DEFAULT_DIM_LATENT_SPACE = 8
DEFAULT_COEFF_FN_INTERNAL_SIZE = 1024
DEFAULT_COEFF_FN_CONFIG = 3
DEFAULT_Y_NOISE_SCALE = 1e-7
DEFAULT_ZEROS_NOISE_SCALE = 1e-3
DEFAULT_SIGMOID_ON_OUTPUT = False
DEFAULT_NOISE_TYPE = "soft"

# Training parameters
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_LR = 0.00005
DEFAULT_BATCH_SIZE = 512
DEFAULT_SAVE_EVERY_N_EPOCHS = 1
DEFAULT_GAMMA = 0.9794578299341784
DEFAULT_STEP_LR_EVERY = int(int(2.5 * 1e6) / 64)
DEFAULT_GRADIENT_CLIP_VAL = 1


# Logging stats
DEFAULT_EVAL_EVERY = int(DEFAULT_MAX_EPOCHS/1000)
DEFAULT_VAL_SET_SIZE = 10
DEFAULT_LOG_EVERY = 100
DEFAULT_CHECKPOINT_EVERY = int(DEFAULT_MAX_EPOCHS/10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")

    # Note: WandB saves artifacts by the run ID (i.e. '34c2gimi') not the run name ('dashing-forest-33'). This is
    # slightly annoying because you need to click on a run to get its ID.
    parser.add_argument("--wandb_run_id_to_load_checkpoint", type=str, help="Example: '34c2gimi'")
    parser.add_argument("--robot_name", type=str)

    # Model parameters
    parser.add_argument("--coupling_layer", type=str, default=DEFAULT_COUPLING_LAYER)
    parser.add_argument("--rnvp_clamp", type=float, default=DEFAULT_RNVP_CLAMP)
    parser.add_argument("--softflow_noise_scale", type=float, default=DEFAULT_SOFTFLOW_NOISE_SCALE)
    # NOTE: NEVER use 'bool' type with argparse. It will cause you pain and suffering.
    parser.add_argument("--softflow_enabled", type=str, default=DEFAULT_SOFTFLOW_ENABLED)
    parser.add_argument("--nb_nodes", type=int, default=DEFAULT_N_NODES)
    parser.add_argument("--dim_latent_space", type=int, default=DEFAULT_DIM_LATENT_SPACE)
    parser.add_argument("--coeff_fn_config", type=int, default=DEFAULT_COEFF_FN_CONFIG)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=DEFAULT_COEFF_FN_INTERNAL_SIZE)
    parser.add_argument("--y_noise_scale", type=float, default=DEFAULT_Y_NOISE_SCALE)
    parser.add_argument("--zeros_noise_scale", type=float, default=DEFAULT_ZEROS_NOISE_SCALE)
    # See note above about pain and suffering.
    parser.add_argument("--sigmoid_on_output", type=str, default=DEFAULT_SIGMOID_ON_OUTPUT)

    # Training parameters
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--save_every_n_epochs", type=int, default=DEFAULT_SAVE_EVERY_N_EPOCHS)
    parser.add_argument("--step_lr_every", type=int, default=DEFAULT_STEP_LR_EVERY)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--lambd", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1.8e-05)

    # Logging options
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--val_set_size", type=int, default=DEFAULT_VAL_SET_SIZE)
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--dataset_tags", nargs="+", type=str, default=[DATASET_TAG_NON_SELF_COLLIDING])
    parser.add_argument("--run_description", type=str)
    parser.add_argument("--disable_progress_bar", action="store_true")

    parser.add_argument('--noise_type', type=str, default=DEFAULT_NOISE_TYPE,
                        choices=["none", "soft", "padding"])
    parser.add_argument("--padding_scale", type=int, default=2)
    args = parser.parse_args()

    if args.dataset_tags is None:
        args.dataset_tags = []

    assert (
        DATASET_TAG_NON_SELF_COLLIDING in args.dataset_tags
    ), "The 'non-self-colliding' dataset should be specified (for now)"
    assert args.optimizer in ["ranger", "adadelta", "adamw"]
    assert 0 <= args.lambd and args.lambd <= 1

    # Load model
    robot = get_robot(args.robot_name)
    base_hparams = IkflowModelParameters()
    base_hparams.run_description = args.run_description
    base_hparams.coupling_layer = args.coupling_layer
    base_hparams.nb_nodes = args.nb_nodes
    base_hparams.dim_latent_space = args.dim_latent_space
    base_hparams.coeff_fn_config = args.coeff_fn_config
    base_hparams.coeff_fn_internal_size = args.coeff_fn_internal_size
    base_hparams.rnvp_clamp = args.rnvp_clamp
    base_hparams.softflow_noise_scale = args.softflow_noise_scale
    base_hparams.y_noise_scale = args.y_noise_scale
    base_hparams.zeros_noise_scale = args.zeros_noise_scale
    base_hparams.sigmoid_on_output = boolean_string(args.sigmoid_on_output)
    base_hparams.noise_type = args.noise_type
    base_hparams.padding_scale = args.padding_scale
    base_hparams.softflow_noise_scale = args.softflow_noise_scale
    print()
    print(base_hparams)

    torch.autograd.set_detect_anomaly(False)
    data_module = IkfLitDataset(robot.name, args.batch_size, args.val_set_size, args.dataset_tags)

    if args.noise_type == "soft":
        if base_hparams.softflow_noise_scale <= 0:
            base_hparams.softflow_noise_scale = 0.001
        base_hparams.softflow_enabled = True
        dir = f"{base_hparams.noise_type}_{base_hparams.dim_latent_space - robot.n_dofs}_{base_hparams.softflow_noise_scale}"
    elif args.noise_type == "none":
        base_hparams.softflow_enabled = False
        dir = f"{base_hparams.noise_type}"
    else:
        base_hparams.softflow_enabled = False
        dir = f"{base_hparams.noise_type}_{base_hparams.dim_latent_space - robot.n_dofs}_{base_hparams.padding_scale}_{base_hparams.softflow_noise_scale}"
    filedir = f"./experiments/ik/{dir}/"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    model_latest = filedir + "model_latest.ckpt"
    print(f"latest model: {model_latest} ({os.path.exists(model_latest)})")

    print("\nArgparse arguments:")
    for k, v in vars(args).items():
        print(f"  {k}={v}")

    ik_solver = IKFlowSolver(base_hparams, robot)
    print("dim_cond", ik_solver.dim_cond)
    print("_network_width", ik_solver._network_width)
    model = IkfLitModel(
        ik_solver=ik_solver,
        base_hparams=base_hparams,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        gradient_clip=args.gradient_clip_val,
        lambd=args.lambd,
        gamma=args.gamma,
        step_lr_every=args.step_lr_every,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        sigmoid_on_output=boolean_string(args.sigmoid_on_output),
        resume=model_latest
    )

    # Train
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=None,
        callbacks=None,
        # check_val_every_n_epoch=args.eval_every,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        log_every_n_steps=args.log_every,
        max_epochs=args.epochs,
        enable_progress_bar=False if (os.getenv("IS_SLURM") is not None) or args.disable_progress_bar else True,
    )
    trainer.fit(model, data_module)
