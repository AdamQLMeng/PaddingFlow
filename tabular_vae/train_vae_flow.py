# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import math
import random

import os

import datetime

import tqdm
from PIL import Image
from einops import repeat

import lib.utils as utils
import lib.layers.odefunc as odefunc

import vae_lib.models.VAE as VAE
import vae_lib.models.CNFVAE as CNFVAE
from metrics.evaluation_metrics import compute_all_metrics
from train_misc import count_parameters
from util import set_seed
from vae_lib.optimization.training import train, evaluate
from vae_lib.utils.load_data import load_dataset
from vae_lib.utils.plotting import plot_training_curve
from vae_lib.utils.visual_evaluation import plot_reconstructions, multinomial_class, plot_images

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser(description='PyTorch Sylvester Normalizing flows')

parser.add_argument(
    '-d', '--dataset', type=str, default='mnist', choices=['mnist', 'freyfaces', 'omniglot', 'caltech'],
    metavar='DATASET', help='Dataset choice.'
)

parser.add_argument(
    '-freys', '--freyseed', type=int, default=123, metavar='FREYSEED',
    help="""Seed for shuffling frey face dataset for test split. Ignored for other datasets.
                    Results in paper are produced with seeds 123, 321, 231"""
)

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, default=0, help='manual seed, if not given resorts to random seed.')

parser.add_argument(
    '-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
    help='how many batches to wait before logging training status'
)

parser.add_argument(
    '-od', '--save', type=str, default='experiments/vae', metavar='OUT_DIR',
    help='output directory for model snapshots etc.'
)

# optimization settings
parser.add_argument(
    '-e', '--epochs', type=int, default=2000, metavar='EPOCHS', help='number of epochs to train (default: 2000)'
)
parser.add_argument(
    '-es', '--early_stopping_epochs', type=int, default=35, metavar='EARLY_STOPPING',
    help='number of early stopping epochs'
)

parser.add_argument(
    '-bs', '--batch_size', type=int, default=100, metavar='BATCH_SIZE', help='input batch size for training'
)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, metavar='LEARNING_RATE', help='learning rate')

parser.add_argument(
    '-w', '--warmup', type=int, default=100, metavar='N',
    help='number of epochs for warm-up. Set to 0 to turn warmup off.'
)
parser.add_argument('--max_beta', type=float, default=1., metavar='MB', help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB', help='min beta for warm-up')
parser.add_argument(
    '-f', '--flow', type=str, default='no_flow', choices=[
        'planar', 'iaf', 'householder', 'orthogonal', 'triangular', 'cnf', 'cnf_bias', 'cnf_hyper', 'cnf_rank',
        'cnf_lyper', 'no_flow'
    ], help="""Type of flows to use, no flows can also be selected"""
)
parser.add_argument('-r', '--rank', type=int, default=1)
parser.add_argument(
    '-nf', '--num_flows', type=int, default=4, metavar='NUM_FLOWS',
    help='Number of flow layers, ignored in absence of flows'
)
parser.add_argument(
    '-nv', '--num_ortho_vecs', type=int, default=8, metavar='NUM_ORTHO_VECS',
    help=""" For orthogonal flow: How orthogonal vectors per flow do you need.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-nh', '--num_householder', type=int, default=8, metavar='NUM_HOUSEHOLDERS',
    help=""" For Householder Sylvester flow: Number of Householder matrices per flow.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-mhs', '--made_h_size', type=int, default=320, metavar='MADEHSIZE',
    help='Width of mades for iaf. Ignored for all other flows.'
)
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE', help='how many stochastic hidden units')
# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

# CNF settings
parser.add_argument(
    "--layer_type", type=str, default="concat",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='512-512')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=False)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)
# evaluation
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--retrain_encoder', type=eval, default=False, choices=[True, False])

parser.add_argument('--noise_type', type=str, default="none",
                    choices=["none", "padding"])
parser.add_argument('--padding_dim', type=int, default=1)
parser.add_argument('--padding_noise_scale', type=int, default=2)
parser.add_argument('--fixed_noise_scale', type=float, default=0.0)
parser.add_argument('--val_metrics_freq', type=int, default=1)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

seed = args.manual_seed if args.manual_seed is not None else int(torch.rand(1) * 1000)
set_seed(seed)
if args.noise_type == "none":
    save_dir = os.path.join(args.save, args.dataset, args.noise_type)
else:
    save_dir = os.path.join(args.save, args.dataset,
                            f"{args.noise_type}_{args.padding_dim}_{args.padding_noise_scale}_{args.fixed_noise_scale}")

# logger
args.snap_dir = save_dir + "/"
utils.makedirs(save_dir)
logger = utils.get_logger(logpath=os.path.join(save_dir, 'logs'), filepath=os.path.abspath(__file__))


def denoise(x):
    return x[:, :args.z_size]


# for PaddingFlow
def add_paddingflow_noise(x):
    B, C = x.shape
    noise = args.fixed_noise_scale * torch.randn_like(x)
    x_pad = args.padding_noise_scale * torch.randn([B, args.padding_dim]).to(x)
    x_noisy = torch.cat([x + noise, x_pad], dim=1)
    return x_noisy


args.add_noise = {
        "none": torch.nn.Identity(),
        "padding": add_paddingflow_noise,
    }[args.noise_type]
args.denoise = denoise

if args.cuda:
    # gpu device number
    torch.cuda.set_device(args.gpu_num)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

def trans(x, im_size):
    t = T.Compose([T.Resize(im_size), T.ToTensor()])
    res = []
    for xx in x:
        xx = Image.fromarray(xx)
        res.append(t(xx).unsqueeze(0))
    return torch.cat(res, dim=0)


def evaluate_on_metrics(data, model, subset):
    num_classes = 256
    model.eval()
    samples = []
    refs = []
    logger.info(
        f"evaluate model on the subset {subset} (size: {data.shape[0]}, shape: {model.input_size}, "
        f"input type: {model.input_type})")

    batch_size = 100
    num_splits = int(max(np.ceil(data.shape[0] / batch_size), 1))
    with tqdm.tqdm(desc="[Sampling]", total=num_splits) as pbar:
        for i in range(num_splits):
            pbar.update(1)
            with torch.no_grad():
                x = data[i * batch_size: (i + 1) * batch_size]
                if type(x) == np.ndarray:
                    x = trans(x, args.input_size[-1])
                x = x.view(-1, *args.input_size).cuda()
                x_mean, z_mu, z_var, ldj, z0, zk = model(x)
                if i == 0:
                    plot_reconstructions(x, x_mean, 0, subset, 1, args, dir=subset + "/")
                if args.input_type == 'multinomial':
                    x_mean = multinomial_class(x_mean, args.input_size)
                samples.append(x_mean.cpu().detach().numpy())
                refs.append(x.cpu().detach().numpy())

    refs = torch.from_numpy(np.concatenate(refs, axis=0)).cuda().float()
    B, C, H, W = refs.shape
    samples = torch.from_numpy(np.concatenate(samples, axis=0)).to(refs)
    logger.info(f"calculate metrics between samples (shape: {samples.shape}) and refs (shape: {refs.shape})!")
    refs = refs.reshape(B, C, -1).permute(0, 2, 1)
    samples = samples.reshape(B, C, -1).permute(0, 2, 1)

    res = compute_all_metrics(samples, refs, data_type="img")
    return res


def run(args, kwargs):
    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    # SAVING
    config_file = os.path.join(save_dir, f'{args.noise_type}.config')
    torch.save(args, config_file)

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
    args.input_size_noisy = (args.input_size[0] + args.padding_dim, *args.input_size[1:])
    # ==============================================================================================================
    # SELECT MODEL
    # ==============================================================================================================
    # flow parameters and architecture choice are passed on to model through args
    if args.noise_type == "none":
        args.padding_dim = 0
    if args.flow == 'no_flow':
        model = VAE.VAE(args)
    elif args.flow == 'planar':
        model = VAE.PlanarVAE(args)
    elif args.flow == 'iaf':
        model = VAE.IAFVAE(args)
    elif args.flow == 'orthogonal':
        model = VAE.OrthogonalSylvesterVAE(args)
    elif args.flow == 'householder':
        model = VAE.HouseholderSylvesterVAE(args)
    elif args.flow == 'triangular':
        model = VAE.TriangularSylvesterVAE(args)
    elif args.flow == 'cnf':
        model = CNFVAE.CNFVAE(args)
    elif args.flow == 'cnf_bias':
        model = CNFVAE.AmortizedBiasCNFVAE(args)
    elif args.flow == 'cnf_hyper':
        model = CNFVAE.HypernetCNFVAE(args)
    elif args.flow == 'cnf_lyper':
        model = CNFVAE.LypernetCNFVAE(args)
    elif args.flow == 'cnf_rank':
        model = CNFVAE.AmortizedLowRankCNFVAE(args)
    else:
        raise ValueError('Invalid flow choice')

    if args.retrain_encoder:
        logger.info(f"Initializing decoder from {args.model_path}")
        dec_model = torch.load(args.model_path)
        dec_sd = {}
        for k, v in dec_model.state_dict().items():
            if 'p_x' in k:
                dec_sd[k] = v
        model.load_state_dict(dec_sd, strict=False)

    logger.info("Model on GPU")
    model.cuda()

    if args.retrain_encoder:
        parameters = []
        logger.info('Optimizing over:')
        for name, param in model.named_parameters():
            if 'p_x' not in name:
                logger.info(name)
                parameters.append(param)
    else:
        parameters = model.parameters()

    optimizer = optim.Adamax(parameters, lr=args.learning_rate, eps=1.e-7)

    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    train_loss = []
    val_loss = []

    # for early stopping
    best_loss = np.inf
    best_bpd = np.inf
    e = 0
    start_epoch = 1

    train_times = []

    if not args.evaluate:
        # Auto Resume
        ckpt_latest = os.path.join(save_dir, 'checkpt_latest.pth')
        logger.info(f"latest model: {ckpt_latest} ({os.path.exists(ckpt_latest)})")
        if os.path.exists(ckpt_latest):
            checkpt = torch.load(ckpt_latest)
            model = checkpt["model"]
            optimizer.load_state_dict(checkpt['optimizer'])
            best_loss = checkpt['best_loss']
            if "e" in checkpt:
                e = checkpt["e"]
            if "epoch" in checkpt:
                start_epoch = checkpt["epoch"] + 1
            logger.info(
                f"Resume from: {ckpt_latest}; Best Loss: {best_loss}; N epochs without improvement: {e}; Epoch: {start_epoch}")
        else:
            logger.info("----------------Arguments----------------")
            for k, v in vars(args).items():
                logger.info(k + f" = {v}")
            logger.info("----------------Structure of model----------------")
            logger.info(model)
            num_params_trn, num_params = count_parameters(model)
            logger.info("----------------Number of parameters----------------")
            logger.info(f"Number of parameters: {round(num_params / 1024 ** 2, 2)}G; \n"
                        f"Number of trainable parameters: {round(num_params_trn / 1024 ** 2, 2)}G")

        for epoch in range(start_epoch, args.epochs + 1):
            t_start = time.time()
            tr_loss = train(epoch, train_loader, model, optimizer, args, logger)
            train_loss.append(tr_loss)
            train_times.append(time.time() - t_start)
            logger.info('One training epoch took %.2f seconds' % (time.time() - t_start))

            v_loss, v_bpd = evaluate(val_loader, model, args, logger, epoch=epoch)
            val_loss.append(v_loss)

            # early-stopping
            if v_loss < best_loss:
                e = 0
                best_loss = v_loss
                # if args.input_type != 'binary':
                best_bpd = v_bpd
                torch.save(model,  os.path.join(save_dir, 'best.model'))
                logger.info('--> Best model has been saved!')

            elif (args.early_stopping_epochs > 0) and (epoch >= args.warmup):
                e += 1
                if e > args.early_stopping_epochs:
                    break

            logger.info(
                '--> Early stopping: {}/{} ([Best] loss: {:.4f}, bpd: {:.4f} [Current] loss: {:.4f}, bpd: {:.4f})'.
                format(e, args.early_stopping_epochs, best_loss, best_bpd, v_loss, v_bpd)
            )

            torch.save({
                'args': args,
                'model': model,
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "e": e,
                "epoch": epoch,
            }, os.path.join(save_dir, 'checkpt_latest.pth'))
            logger.info('--> Latest model has been saved!\n')

            if math.isnan(v_loss):
                raise ValueError('NaN encountered!')

        train_loss = np.hstack(train_loss)
        val_loss = np.array(val_loss)

        plot_training_curve(train_loss, val_loss, fname=save_dir + '/training_curve_%s.pdf' % args.flow)

        # training time per epoch
        train_times = np.array(train_times)
        mean_train_time = np.mean(train_times)
        std_train_time = np.std(train_times, ddof=1)
        logger.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

        # ==================================================================================================================
        # EVALUATION
        # ==================================================================================================================

        logger.info(args)
        logger.info('Stopped after %d epochs' % epoch)
        logger.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

        best_model = torch.load(os.path.join(save_dir, 'best.model'))
        logger.info('Evaluate best model on valid set!')
        validation_loss, validation_bpd = evaluate(val_loader, best_model, args, logger)
        logger.info(f'[Evaluate] result on validation set, -ELBO: {validation_loss:4f}, -ELBO (BPD): {validation_bpd:4f}')

    else:
        best_model_path = os.path.join(save_dir, 'best.model')
        logger.info(f"Loading model from {best_model_path} ({os.path.exists(best_model_path)})")
        assert os.path.exists(best_model_path), f"No model found at {best_model_path}, it might be not trained yet!"
        best_model = torch.load(best_model_path)
        best_model.add_noise = args.add_noise
        best_model.denoise = args.denoise

    logger.info("---------------Calculate ELBO--------------")
    nll, nll_bpd, nelbo, nelbo_bpd, rec = evaluate(test_loader, best_model, args, logger, testing=True)
    logger.info(f"[Test] NLL: {nll}, NLL (BPD): {nll_bpd}, -ELBO: {nelbo}, -ELBO (BPD): {nelbo_bpd}), REC: {rec}")

    logger.info("---------------Calculate metrics--------------")
    x, _ = test_loader.dataset.tensors
    set_seed(0)
    idxs = torch.randperm(len(x))
    x = x[idxs]
    res = evaluate_on_metrics(x, best_model, f"test_best")
    logger.info("--------------------Results-------------------")
    for k, v in res.items():
        logger.info(k + f": {v}")


if __name__ == "__main__":
    run(args, kwargs)
