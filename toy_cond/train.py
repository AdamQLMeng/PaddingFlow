import matplotlib
import tqdm
from einops import repeat

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time

import torch
import torch.optim as optim

import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import standard_normal_logprob
from lib.utils import count_nfe, count_total_time
from lib.utils import build_model_tabular

from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc


SOLVERS = ["dopri5"]
parser = argparse.ArgumentParser('PaddingFlow')
parser.add_argument(
    '--data',
    choices=['circles', 'circles_conditional', 'sines', 'sines_conditional'],
    type=str,
    default='2circles'
)
parser.add_argument("--layer_type", type=str, default="concatsquash", choices=["concatsquash"])
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# for noise
parser.add_argument('--std_min', type=float, default=0.0)
parser.add_argument('--std_max', type=float, default=0.1)
parser.add_argument('--std_weight', type=float, default=2)
parser.add_argument('--gauss_variance', type=float, default=0.01)

# for training
parser.add_argument('--niters', type=int, default=36000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--viz_freq', type=int, default=10)
parser.add_argument('--val_freq', type=int, default=400)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resume_from', type=str,
                    # default='results/2sines/none/checkpt.pth'
                    )
parser.add_argument('--noise_type', type=str, default="none",
                    choices=["none", "padding", "soft"])
parser.add_argument('--padding_noise_scale', type=float, default=2)
parser.add_argument('--fixed_noise_scale', type=float, default=0.01)
parser.add_argument('--evaluate', type=bool, default=False)
args = parser.parse_args()

# logger
save_path = os.path.join('experiments/toy/', args.data, args.noise_type)
utils.makedirs(save_path)
logger = utils.get_logger(logpath=os.path.join(save_path, 'logs'), filepath=os.path.abspath(__file__))

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")


def get_transforms(model, cond_dims):

    def sample_fn(z, c, logpz=None):
        zeors_std = torch.zeros(z.shape[0], cond_dims).to(z)
        if c is not None:
            zeors_std[:, :1] = repeat(c[0], "1 -> n", n=z.shape[0]).unsqueeze(dim=1)
        if logpz is not None:
            return model(z, zeors_std, logpz, reverse=True)
        else:
            return model(z, zeors_std, reverse=True)

    def density_fn(x, c, logpx=None):
        zeors_std = torch.zeros(x.shape[0], cond_dims).to(x)
        if c is not None:
            zeors_std[:, :1] = repeat(c[0], "1 -> n", n=x.shape[0]).unsqueeze(dim=1)
        if logpx is not None:
            return model(x, zeors_std, logpx, reverse=False)
        else:
            return model(x, zeors_std, reverse=False)

    return sample_fn, density_fn


# for SoftFlow
def add_softflow_noise(x):
    x_pad = torch.zeros([x.shape[0], args.dist_dims - 2]).to(x)  # padding
    xx = torch.cat([x, x_pad], dim=1)
    std = torch.rand_like(x[:,0]).view(-1,1)
    eps = torch.randn_like(x) * std * 0.1
    eps = torch.cat([eps, x_pad], dim=1)
    std_in = std * 2
    return xx + eps, std_in


def add_paddingflow_noise(x):
    noise = args.fixed_noise_scale * torch.randn_like(x).to(x)
    x_pad = args.padding_noise_scale * torch.randn([x.shape[0], 1]).to(x)
    xx = torch.cat([x+noise, x_pad], dim=1)
    return xx, None


def add_none(x):
    return x, None


add_noise = {
        "none": add_none,
        "padding": add_paddingflow_noise,
        "soft": add_softflow_noise,
    }[args.noise_type]


def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    x, cond = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    x, std_in = add_noise(x)
    if std_in is not None and cond is not None:
        cond = torch.concat([cond, std_in], dim=1)
    elif std_in is not None:
        cond = std_in
    # print(args.dist_dims, args.cond_dims, args.data, "conditional" in args.data)
    # print(x.shape, cond.shape)
    z, delta_logp = model(x, cond.to(x), zero)

    # compute nll loss
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)
    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


def visualize(model, itr):
    x, cond = toy_data.inf_train_gen(args.data, batch_size=2000)
    if "soft" not in args.noise_type:
        print("add noise", args.noise_type, "soft" not in args.noise_type)
        x, _ = add_noise(torch.tensor(x))
    sample_fn, density_fn = get_transforms(model, args.cond_dims)

    plt.figure(figsize=(9, 3))
    visualize_transform(
        x, cond, torch.randn, standard_normal_logprob, transform=sample_fn,
        samples=True, npts=200, device=device, args=args
    )
    fig_format = "png"
    fig_filename = os.path.join(save_path, 'figs', f'{itr}.{fig_format}')
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename, format=fig_format, dpi=1200)
    plt.close()
    print(f"figs have been saved in the picture. (file name: {fig_filename})")


if __name__ == '__main__':
    if "conditional" in args.data:
        args.cond_dims = 1
    else:
        args.cond_dims = 0

    if args.noise_type == "soft":
        args.dist_dims = 2
        args.cond_dims += 1
    elif args.noise_type == "padding":
        args.dist_dims = 3
    else:
        args.dist_dims = 2

    model = build_model_tabular(args, args.dist_dims, args.cond_dims).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    start_itr = 1
    ckpt_latest = os.path.join(save_path, 'checkpt.pth')
    if args.resume_from is None and os.path.exists(ckpt_latest):
        args.resume_from = ckpt_latest
    if args.resume_from:
        ext = os.path.splitext(args.resume_from)[-1]
        if ext == '.pkl':
            model.load_state_dict(args.resume_from)
        elif ext == '.pth':
            checkpoint = torch.load(args.resume_from, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_itr = checkpoint['itr'] + 1
        print('Resumed from: ' + args.resume_from, 'iterations: ', start_itr)

    end = time.time()
    best_loss = float('inf')
    model.train()
    with tqdm.tqdm(total=args.niters-start_itr+1) as pbar:
        pbar.update(1)  # align
        for itr in range(start_itr, args.niters + 1):
            if args.evaluate:
                break
            pbar.update(1)
            optimizer.zero_grad()

            loss = compute_loss(args, model)
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

            log_message = (
                'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
                ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                    itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                    nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
                )
            )
            if itr % args.log_freq == 0:
                logger.info(log_message)

            if itr % args.val_freq == 0 or itr == args.niters:
                with torch.no_grad():
                    model.eval()
                    test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                    test_nfe = count_nfe(model)
                    log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss, test_nfe)
                    logger.info(log_message)

                    if test_loss.item() < best_loss:
                        best_loss = test_loss.item()
                        utils.makedirs(save_path)
                        torch.save({
                            'args': args,
                            'itr': itr,
                            "optimizer": optimizer.state_dict(),
                            'state_dict': model.state_dict(),
                        }, os.path.join(save_path, 'checkpt.pth'))
                    model.train()

            if itr % args.viz_freq == 0:
                with torch.no_grad():
                    model.eval()
                    visualize(model, itr)
                    model.train()

            end = time.time()

    logger.info('Training has finished.')
    model.eval()
    if "conditional" in args.data:
        for i in range(5):
            visualize(model, f"final_{i}")
    else:
        visualize(model, "final")

