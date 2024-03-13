import torch
import numpy as np
import warnings

import tqdm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
from .StructuralLosses.match_cost import match_cost
from .StructuralLosses.nn_distance import nn_distance
from torch.nn.parallel import DistributedDataParallel as DDP

# # Import CUDA version of CD, borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
# try:
#     from . chamfer_distance_ext.dist_chamfer import chamferDist
#     CD = chamferDist()
#     def distChamferCUDA(x,y):
#         return CD(x,y,gpu)
# except:


def distChamferCUDA(x, y):
    dr, dl = nn_distance(x, y)
    return dr.mean(1) + dl.mean(1)


def torch_bmm_split(x, y, split_len=40000):
    B, N, D = x.shape
    assert B == y.shape[0]
    assert N == y.shape[2]
    assert D == y.shape[1]
    num_splits = int(np.ceil(N / split_len))
    res = []
    for i in range(num_splits):
        x_split = x[:, split_len * i:split_len * (i+1), :]
        dots = []
        for j in range(num_splits):
            y_split = y[:, :, split_len * j:split_len * (j+1)]
            dot = torch.bmm(x_split, y_split)
            dots.append(dot.cpu())
        dots = torch.cat(dots, dim=2)
        res.append(dots)
    res = torch.cat(res, dim=1)
    return res


def distChamfer(a, b):
    x, y = a, b
    B, N, D = x.shape
    if N > 20000:
        torch_bmm = torch_bmm_split
    else:
        torch_bmm = torch.bmm
    xx = torch.mul(x, x).sum(-1, keepdim=True).expand(1, N, N)
    yy = torch.mul(y, y).sum(-1, keepdim=True).expand(1, N, N).transpose(2, 1) + xx
    del xx
    zz = torch_bmm(x, y.transpose(2, 1))
    P = (yy - 2 * zz)
    min1, min2 = P.min(1)[0], P.min(2)[0]
    return min1.mean(dim=1) + min2.mean(dim=1)


def distChamferImg(a, b):
    x, y = a, b
    xy = x - y
    dist = torch.mul(xy, xy).sum(-1).sum(-1, keepdim=True)
    return dist


def emd_approx(x, y):
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False).to(x)  # (bs, npts, mpts)
    del x, y

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(dist)
    return emd_torch


try:
    from .StructuralLosses.match_cost import match_cost
    def emd_approx_cuda(sample, ref):
        B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
        assert N == N_ref, "Not sure what would EMD do in this case"
        emd = match_cost(sample, ref)  # (B,)
        emd_norm = emd / float(N)  # (B,)
        return emd_norm
except:
    print("emd_approx_cuda not available. Fall back to slower version.")
    def emd_approx_cuda(sample, ref):
        return emd_approx(sample, ref)


def EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=False, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            cd = distChamferCUDA(sample_batch, ref_batch)
        else:
            cd = distChamfer(sample_batch, ref_batch)
        cd_lst.append(cd)

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, decs="disable", accelerated_emd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    if decs != "disable":
        pbar = tqdm.tqdm(desc=decs, total=N_sample * N_ref)
    for sample_batch in sample_pcs:
        sample_batch = sample_batch.contiguous().unsqueeze(dim=0)

        cd_lst = []
        emd_lst = []
        for ref_batch in ref_pcs:
            if decs != "disable":
                pbar.update(1)
            ref_batch = ref_batch.contiguous().unsqueeze(dim=0)

            # CD
            cd = distChamfer(sample_batch, ref_batch)
            cd_lst.append(cd)

            # EMD
            sample_batch, ref_batch = sample_batch[:, :20000, :], ref_batch[:, :20000, :]
            if accelerated_emd:
                emd_batch = emd_approx_cuda(sample_batch, ref_batch)
            else:
                emd_batch = emd_approx(sample_batch.cpu(), ref_batch.cpu())
            emd_lst.append(emd_batch.view(1, -1))
        cd_lst = torch.cat(cd_lst, dim=0).view(1, -1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    if decs != "disable":
        pbar.close()
    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def _pairwise_dist_(sample, ref, dist_fn, decs="disable"):
    N_sample = sample.shape[0]
    N_ref = ref.shape[0]
    all_dist = []
    if decs != "disable":
        pbar = tqdm.tqdm(desc=decs, total=N_sample * N_ref)
    for sample_batch in sample:
        sample_batch = sample_batch.contiguous().unsqueeze(dim=0)

        dist_list = []
        for ref_batch in ref:
            if decs != "disable":
                pbar.update(1)
            ref_batch = ref_batch.contiguous().unsqueeze(dim=0)

            dist = dist_fn(sample_batch, ref_batch)
            dist_list.append(dist)

        dist_list = torch.cat(dist_list, dim=0).view(1, -1)
        all_dist.append(dist_list)

    if decs != "disable":
        pbar.close()
    all_dist = torch.cat(all_dist, dim=0)  # N_sample, N_ref

    return all_dist


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample = all_dist.size(0)
    _, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / N_sample
    cov = torch.tensor(cov).to(all_dist)
    return {
        'MMD': mmd,
        'COV': cov,
    }


def compute_all_metrics_for_one_dist(sample, ref, dist_fn, dist_name):
    results = {}

    rs_dist = _pairwise_dist_(ref, sample, dist_fn, decs=f"[Calculate RS {dist_name}]")

    # MMD and COV results
    res = lgan_mmd_cov(rs_dist.t())
    results.update({
        f"{k}-{dist_name}": v for k, v in res.items()
    })
    return results


def compute_all_metrics(sample, ref, data_type="point_cloud"):
    results = {}
    print("Computing metrics between {} and {}".format(sample.shape, ref.shape))

    if data_type == "point_cloud":
        res_cd = compute_all_metrics_for_one_dist(sample, ref, distChamfer, "CD")
        res_emd = compute_all_metrics_for_one_dist(sample, ref, emd_approx_cuda, "EMD")
        results.update(res_cd)
        results.update(res_emd)
    elif data_type == "img":
        res_l2 = compute_all_metrics_for_one_dist(sample, ref, distChamferImg, "L2")
        results.update(res_l2)
    else:
        raise ValueError("Data type not supported, it should be 'point_cloud' or 'img', instead {}".format(data_type))

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))
