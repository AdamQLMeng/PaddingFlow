from __future__ import print_function
import os
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def multinomial_class(x, input_size):
    # data is already between 0 and 1
    num_classes = 256
    # Find largest class logit
    tmp = x.view(-1, num_classes, *input_size).max(dim=1)[1]
    x = tmp.float() / (num_classes - 1.)
    return x


def plot_reconstructions(data, recon_mean, loss, loss_type, epoch, args, dir = 'reconstruction/'):

    if args.input_type == 'multinomial':
        recon_mean = multinomial_class(recon_mean, args.input_size)

    if epoch == 1:
        if not os.path.exists(args.snap_dir + dir):
            os.makedirs(args.snap_dir + dir)
        # VISUALIZATION: plot real images
        plot_images(args, data.data.cpu().numpy(), args.snap_dir + dir, 'real', size_x=4, size_y=4)#.take([0,1,2,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],0),
    # VISUALIZATION: plot reconstructions
    if loss_type == 'bpd':
        fname = str(epoch) + '_bpd_%5.3f' % loss
    elif loss_type == 'elbo':
        fname = str(epoch) + '_elbo_%6.4f' % loss
    else:
        fname = loss_type
    plot_images(args, recon_mean.data.cpu().numpy(), args.snap_dir + dir, fname, size_x=4, size_y=4)


# 定义图像拼接函数
def plot_images(args, imgs, dir, file_name, size_x=3, size_y=3):
    img_size = (args.input_size[0], args.input_size[2], args.input_size[1])
    to_image = Image.new('RGB', (size_x * img_size[1], size_y * img_size[2]))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    total_num = 0
    for y in range(size_y):
        for x in range(size_x):
            img = (imgs[size_x * y + x]*255).transpose(1,2,0).astype(np.uint8).squeeze()
            from_image = Image.fromarray(img)
            box = (x * img_size[1], y * img_size[2], (x + 1) * img_size[1], (y + 1) * img_size[2])
            to_image.paste(from_image, box=box)
            total_num += 1
            if total_num == len(imgs):
                break
    to_image.save(dir + file_name + '.png')  # 保存新图


# def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):
#
#     fig = plt.figure(figsize=(size_x, size_y))
#     # fig = plt.figure(1)
#     gs = gridspec.GridSpec(size_x, size_y)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(x_sample):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
#         sample = sample.swapaxes(0, 2)
#         sample = sample.swapaxes(0, 1)
#         if (args.input_type == 'binary') or (args.input_type in ['multinomial'] and args.input_size[0] == 1):
#             sample = sample[:, :, 0]
#             plt.imshow(sample, cmap='gray', vmin=0, vmax=1)
#         else:
#             plt.imshow(sample)
#
#     plt.savefig(dir + file_name + '.png', bbox_inches='tight')
#     plt.close(fig)
