import numpy as np
import torch

from ffjord.metrics.evaluation_metrics import torch_bmm_split, distChamfer

if __name__ == "__main__":
    # B, N, D = 1, 4, 2
    # x = torch.arange(B*N*D).reshape(B, N, D)
    # y = torch.arange(B*N*D).reshape(B, D, N)
    # print(x)
    # print(y)
    # dot1 = distChamfer(x, x)
    # print(dot1)
    # dot2 = torch_bmm_split(x, y, 3)
    # print(dot2, (dot1 == dot2).all())
    d = [0.15383227974176406, 0.15374910101294517, 0.15138730823993682, 0.1468006680905819, 0.15200234889984132]
    m = np.mean(d)
    std = np.std(d)
    print(std, f"{round(std * 200 / m, 2)}%")
