import numpy as np
import sklearn
import sklearn.datasets
import torch
from sklearn.utils import shuffle as util_shuffle
from einops import repeat


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.0)[0]
        data = data.astype("float32")
        data *= 3
        return data, None

    elif data == "sines":
        x = (rng.rand(batch_size) -0.5) * 2 * np.pi
        u = (rng.binomial(1,0.5,batch_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5 * 0.6
        return np.stack((x, y), 1), None

    elif data == "circles_conditional":
        cond = np.random.randint(1, 3, [1, 1])*0.25
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=float(cond), noise=0.0)[0]
        data = data.astype("float32")
        data *= 3
        return data, repeat(torch.tensor(cond), '1 1 -> n 1', n=batch_size)

    elif data == "sines_conditional":
        cond = np.random.randint(1, 3) * 0.3
        x = (rng.rand(batch_size) -0.5) * 2 * np.pi
        u = (rng.binomial(1,0.5,batch_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5 * cond
        return np.stack((x, y), 1), repeat(torch.tensor(cond).reshape([1, 1]), '1 1 -> n 1', n=batch_size)

    else:
        return inf_train_gen("2circles", rng, batch_size)



