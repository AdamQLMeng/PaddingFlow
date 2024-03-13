import argparse
import os
import random
import sys
import time
import tqdm

import numpy as np
import torch

import lib.utils as utils
import lib.layers.odefunc as odefunc
from lib.custom_optimizers import Adam

import datasets
from metrics.evaluation_metrics import _pairwise_EMD_CD_

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn
from util import set_seed

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='miniboone'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--hdim_factor', type=int, default=10)
parser.add_argument('--nhidden', type=int, default=1)
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-8)
parser.add_argument('--rtol', type=float, default=1e-6)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/tabular')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--evaluate_on_all', action='store_true')
parser.add_argument('--num_splits', type=int, default=5)
parser.add_argument('--testset_split_len', type=int, default=20000)
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--val_metrics_freq', type=int, default=2000)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=200)
parser.add_argument('--noise_type', type=str, default="none",
                    choices=["none", "padding"])
parser.add_argument('--padding_dim', type=int, default=1)
parser.add_argument('--padding_noise_scale', type=int, default=2)
parser.add_argument('--fixed_noise_scale', type=float, default=0.0)
parser.add_argument('--times_sample_val', type=int, default=100)
parser.add_argument('--times_sample_test', type=int, default=100)
args = parser.parse_args()

if args.noise_type == "none":
    save_dir = os.path.join(args.save, args.data, args.noise_type)
elif args.noise_type == "padding":
    save_dir = os.path.join(args.save, args.data, args.noise_type+f"_{args.padding_dim}_{args.padding_noise_scale}_{args.fixed_noise_scale}")

# logger
utils.makedirs(save_dir)
logger = utils.get_logger(logpath=os.path.join(save_dir, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0
    args.train_T = False

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size


# for PaddingFlow
def add_paddingflow_noise(x):
    noise = args.fixed_noise_scale * torch.randn_like(x).to(x)
    x_pad = args.padding_noise_scale * torch.randn([x.shape[0], args.padding_dim]).to(x)
    x = torch.cat([x + noise, x_pad], dim=1)
    return x


add_noise = {
        "none": torch.nn.Identity(),
        "padding": add_paddingflow_noise
    }[args.noise_type]


def batch_iter(X, batch_size=args.batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


ndecs = 0


def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping // 3 * 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 100
        ndecs = 2
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10**ndecs


def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')


def nll_loss(z, delta_logp):
    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


def compute_loss(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)
    x_noisy = add_noise(x)

    z, delta_logp = model(x_noisy, zero)  # run model forward

    loss = nll_loss(z, delta_logp)
    return loss


def sample(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)
    x_noisy = add_noise(x)
    z = torch.randn_like(x_noisy)

    x_samples, delta_logp = model(z, zero, reverse=True)  # run model backward

    x_samples_denoise = x_samples[:, :x.shape[1]]

    return x_samples_denoise


def evaluate_on_metrics(x, model, times_sample):
    logger.info(f"evaluate model on the subset (size: {x.shape[0]}, shape: {x.shape})")
    cd_list = []
    emd_list = []
    with tqdm.tqdm(desc="[Calculating Metrics]: ", total=times_sample) as pbar:
        for _ in range(times_sample):
            pbar.update(1)
            x_samples = sample(x, model).unsqueeze(dim=0)
            cd, emd = _pairwise_EMD_CD_(x_samples, x.unsqueeze(dim=0))
            cd_list.append(cd.cpu().item())
            emd_list.append(emd.cpu().item())
        t = pbar.last_print_t - pbar.start_t
    logger.info(f"metrics were calculated on {times_sample} times of sampling! "
                f"(metrics' shapes: {len(cd_list), len(emd_list)}; "
                f"consumed time: {int(t//3600)}:{int(t%3600//60)}:{round(t%60, 2)})")
    results = {}
    results.update({"CD": np.mean(cd_list)})
    results.update({"EMD": np.mean(emd_list)})
    results.update({"MMD-CD": np.min(cd_list)})
    results.update({"MMD-EMD": np.min(emd_list)})
    return results


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model.to(device), checkpt["best_loss"]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    logger.info(f'Using {device}.')

    data = load_data(args.data)
    data.trn.x = torch.from_numpy(data.trn.x)
    data.val.x = torch.from_numpy(data.val.x)
    data.tst.x = torch.from_numpy(data.tst.x)

    args.dims = '-'.join([str(args.hdim_factor * data.n_dims)] * args.nhidden)

    if args.noise_type in ["padding"]:
        model = build_model_tabular(args, data.n_dims+args.padding_dim, ()).to(device)
    else:
        model = build_model_tabular(args, data.n_dims, ()).to(device)

    set_cnf_options(args, model)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    itr = 0
    n_vals_without_improvement = 0
    best_loss = float('inf')

    ckpt_latest = os.path.join(save_dir, 'checkpt_latest.pth')
    logger.info(f"latest model: {ckpt_latest} ({os.path.exists(ckpt_latest)})")
    if args.resume is None and os.path.exists(ckpt_latest):
        args.resume = ckpt_latest
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        filtered_state_dict = {}
        for k, v in checkpt['state_dict'].items():
            if 'diffeq.diffeq' not in k:
                filtered_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(filtered_state_dict)
        optimizer.load_state_dict(checkpt['optimizer'])
        best_loss = checkpt['best_loss']
        if "n_vals_without_improvement" in checkpt.keys():
            n_vals_without_improvement = checkpt['n_vals_without_improvement']
        logger.info(f"Resume from: {args.resume}; Best Loss: {best_loss}; n_vals_without_improvement: {n_vals_without_improvement}")
        del checkpt, filtered_state_dict
    elif not args.evaluate:
        logger.info("----------------Arguments----------------")
        for k, v in vars(args).items():
            logger.info(k + f" = {v}")
        logger.info("----------------Structure of model----------------")
        logger.info(model)
        num_params_trn, num_params = count_parameters(model)
        logger.info("----------------Number of parameters----------------")
        logger.info(f"Number of parameters: {round(num_params / 1024, 1)}M; \n"
                    f"Number of trainable parameters: {round(num_params_trn / 1024, 1)}M")

    if not args.evaluate:

        time_meter = utils.RunningAverageMeter(0.98)
        loss_meter = utils.RunningAverageMeter(0.98)
        nfef_meter = utils.RunningAverageMeter(0.98)
        nfeb_meter = utils.RunningAverageMeter(0.98)
        tt_meter = utils.RunningAverageMeter(0.98)

        end = time.time()
        model.train()
        while True:
            if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                break

            for x in batch_iter(data.trn.x, shuffle=True):
                if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                    break

                optimizer.zero_grad()

                x = cvt(x)
                loss = compute_loss(x, model)
                loss_meter.update(loss.item())

                total_time = count_total_time(model)
                nfe_forward = count_nfe(model)

                loss.backward()
                optimizer.step()

                nfe_total = count_nfe(model)
                nfe_backward = nfe_total - nfe_forward
                nfef_meter.update(nfe_forward)
                nfeb_meter.update(nfe_backward)

                time_meter.update(time.time() - end)
                tt_meter.update(total_time)

                if itr % args.log_freq == 0:
                    log_message = (
                        'Iter {:06d} | Epoch {:.2f} | LR {:.6f} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | '
                        'NFE Forward {:.0f}({:.1f}) | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                            itr,
                            float(itr) / (data.trn.x.shape[0] / float(args.batch_size)), optimizer.param_groups[0]["lr"], time_meter.val, time_meter.avg,
                            loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg, nfeb_meter.val,
                            nfeb_meter.avg, tt_meter.val, tt_meter.avg
                        )
                    )
                    logger.info(log_message)
                itr += 1
                end = time.time()

                # Validation loop.
                if itr % args.val_freq == 0 or itr == 2:  # "or itr == 2" for sanity check
                    model.eval()
                    utils.makedirs(save_dir)
                    start_time = time.time()
                    with torch.no_grad():
                        val_loss = utils.AverageMeter()
                        val_nfe = utils.AverageMeter()
                        for x in batch_iter(data.val.x, batch_size=test_batch_size):
                            x = cvt(x)
                            val_loss.update(compute_loss(x, model).item(), x.shape[0])
                            val_nfe.update(count_nfe(model))

                        if val_loss.avg < best_loss:
                            best_loss = val_loss.avg
                            utils.makedirs(args.save)
                            torch.save({
                                'args': args,
                                'state_dict': model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "best_loss": best_loss,
                                "n_vals_without_improvement": n_vals_without_improvement,
                            }, os.path.join(save_dir, 'checkpt_best.pth'))
                            n_vals_without_improvement = 0
                            logger.info('--> Best model has been saved!')
                        else:
                            n_vals_without_improvement += 1
                        update_lr(optimizer, n_vals_without_improvement)

                        log_message = (
                            '[VAL] Iter {:06d} | Val Loss {:.6f} | Best Loss {:.6f} | '
                            'NoImproveEpochs {:02d}/{:02d}'.format(
                                itr, val_loss.avg, best_loss, n_vals_without_improvement, args.early_stopping
                            )
                        )
                        logger.info(log_message)
                    model.train()

                if itr % args.save_freq == 0:
                    utils.makedirs(save_dir)
                    torch.save({
                            'args': args,
                            'state_dict': model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_loss": best_loss,
                            "n_vals_without_improvement": n_vals_without_improvement,
                        }, os.path.join(save_dir, 'checkpt_latest.pth'))
                    logger.info('--> Latest model has been saved!\n')

        logger.info(f'Training has finished! (Best loss: {best_loss})')
        model, model_loss = restore_model(model, os.path.join(save_dir, 'checkpt_best.pth'))
        set_cnf_options(args, model)
    else:
        model, model_loss = restore_model(model, os.path.join(save_dir, 'checkpt_best.pth'))
        set_cnf_options(args, model)

    logger.info(f"Evaluate best model! "
                f"(loss: {model_loss}; "
                f"Path: {os.path.join(save_dir, 'checkpt_best.pth')}; "
                f"Is best model: {best_loss==model_loss})")
    model.eval()
    set_seed(0)
    with torch.no_grad():
        logger.info("---------------Calculate metrics--------------")
        x = cvt(data.tst.x)
        idxs = torch.randperm(x.shape[0])
        num = args.testset_split_len
        num_splits = min(x.shape[0]//num, args.num_splits) if args.evaluate_on_all else 1
        results = []
        for i in range(num_splits):
            logger.info(f"-------------------Split {i+1}/{num_splits}------------------")
            start_idx = num*i
            idxs_split = idxs[start_idx: num + start_idx]
            logger.info(f"Test set contains {x.shape[0]} data, "
                        f"sample {idxs_split.shape[0]} data for calculating metrics (start index: {idxs_split[:2]}), "
                        f"start at No.{start_idx}!"
                        )
            x_split = x[idxs_split]
            res = evaluate_on_metrics(x_split, model, args.times_sample_test)
            logger.info("--------------------Results-------------------")
            for k, v in res.items():
                logger.info(k + f": {v}")
            results.append(res)

        if num_splits > 1:
            res_m_std = dict()
            for k, _ in results[0].items():
                v = []
                for r in results:
                    v.append(r[k])
                res_m_std[k+"_mean"] = np.mean(v)
                res_m_std[k + "_std"] = np.std(v)
                res_m_std[k + "_rate(%)"] = np.std(v) * 200 / np.mean(v)
            logger.info("--------------------Mean and std of results-------------------")
            for k, v in res_m_std.items():
                logger.info(k + f": {v}")

        if args.noise_type == "padding":
            sys.exit()
        logger.info("------------------------Calculate NLL---------------------")
        # override_divergence_fn(model, "brute_force")
        test_loss = utils.AverageMeter()
        test_nfe = utils.AverageMeter()
        with tqdm.tqdm(desc="[Testing]", total=round(data.tst.x.shape[0]/test_batch_size)) as pbar:
            for itr, x in enumerate(batch_iter(data.tst.x, batch_size=test_batch_size)):
                pbar.update(1)
                x = cvt(x)
                test_loss.update(compute_loss(x, model).item(), x.shape[0])
                test_nfe.update(count_nfe(model))
        log_message = '[TEST] Iter {:06d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss.avg, test_nfe.avg)
        logger.info(log_message)


