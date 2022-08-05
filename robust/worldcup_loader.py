'''
train and test set on WorldCup for Nie et al. (A robust and efficient framework for sports-field registration)
'''
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import utils
import glob
import random
import skimage.segmentation as ss
from typing import Optional


class PublicWorldCupDataset(data.Dataset):

    def __init__(self, root, data_type, mode, noise_trans: Optional[float] = None, noise_rotate: Optional[float] = None):

        self.frame_h = 720
        self.frame_w = 1280
        self.root = root
        self.data_type = data_type
        self.mode = mode
        self.noise_trans = noise_trans
        self.noise_rotate = noise_rotate

        frame_list = [osp.basename(name) for name in glob.glob(
            osp.join(self.root, self.data_type, '*.jpg'))]
        self.frames = [img for img in sorted(
            frame_list, key=lambda x: int(x[:-4]))]

        homographies_list = [osp.basename(name) for name in glob.glob(
            osp.join(self.root, self.data_type, '*.homographyMatrix'))]
        self.homographies = [mat for mat in sorted(
            homographies_list, key=lambda x: int(x[:-17]))]

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # ImageNet
        ])

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, index):

        image = np.array(Image.open(
            osp.join(self.root, self.data_type, self.frames[index])))

        gt_h = np.loadtxt(
            osp.join(self.root, self.data_type, self.homographies[index]))

        template_grid = utils.gen_template_grid()  # template grid shape (91, 3)

        # Warp grid shape (N, 3), N is based on number of points in image view
        warp_image, warp_grid, homo_mat = utils.gen_im_partial_grid(
            self.mode, image, gt_h, template_grid, self.noise_trans, self.noise_rotate, index)
        num_pts = warp_grid.shape[0]
        pil_image = Image.fromarray(warp_image)

        # TODO apply random horizontal flip to the image and grid points
        if self.mode == 'train' and random.random() < 0.5:
            pil_image, warp_grid = utils.put_lrflip_augmentation(
                pil_image, warp_grid)

        # Default all keypoints belong to background
        heatmaps = np.zeros(
            (self.frame_h // 4, self.frame_w // 4), dtype=np.float32)

        # Dilate pixels
        for keypts_label in range(num_pts):
            px = np.rint(warp_grid[keypts_label, 0] / 4).astype(np.int32)
            py = np.rint(warp_grid[keypts_label, 1] / 4).astype(np.int32)
            if 0 <= px < (self.frame_w // 4) and 0 <= py < (self.frame_h // 4):
                heatmaps[py, px] = warp_grid[keypts_label, 2]

        dilated_heatmaps = ss.expand_labels(heatmaps, distance=5)
        image_tensor = self.preprocess(pil_image)

        return image_tensor, utils.to_torch(dilated_heatmaps), utils.to_torch(heatmaps), homo_mat


if __name__ == "__main__":

    worldcup_loader = PublicWorldCupDataset(
        root='./dataset/soccer_worldcup_2014/soccer_data', data_type='train_val', mode='train', noise_trans=5.0, noise_rotate=0.0084)
    # worldcup_loader = PublicWorldCupDataset(
    #     root='./dataset/soccer_worldcup_2014/soccer_data', data_type='test', mode='test')

    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cnt = 0
    for image, gt_heatmap, target, gt_homo in worldcup_loader:
        pass
        # print(gt_homo.dtype)
        # plt.imsave('./target.png', target, vmin=0, vmax=91)
        # print(image.shape, gt_heatmap.shape)
        # print(image.dtype, gt_heatmap.dtype)
        # gt_heatmap = gt_heatmap.squeeze(0)
        # image = np.transpose(image.cpu().numpy(), (1, 2, 0))
        # plt.imsave(osp.join('./0723', '%03d_out_warp_rgb.png' % cnt), image)
        # plt.imsave('./out_heatmaps.png', heatmaps, vmin=0, vmax=91, cmap='tab20b')
        # plt.imsave(osp.join('./0723', '%03d_out_heatmaps.png' %
        #    cnt), gt_heatmap, vmin=0, vmax=91)
        # print(gt_heatmap.shape, gt_heatmap.dtype)
        # plt.imsave('./next_dilated_circle.png', heatmaps, vmin=0, vmax=91, cmap='tab20b')
        cnt += 1
        # assert False
        # utils.visualize_heatmaps(image, gt_heatmap)
