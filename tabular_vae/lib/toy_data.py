import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.0)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "sines":
        x = (rng.rand(batch_size) -0.5) * 2 * np.pi
        u = (rng.binomial(1,0.5,batch_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5 * 0.6
        return np.stack((x, y), 1)

    else:
        return inf_train_gen("circles", rng, batch_size)
