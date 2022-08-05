'''
train and test set on TS-WorldCup for Nie et al. (A robust and efficient framework for sports-field registration)
'''
import random
import glob
import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import skimage.segmentation as ss
from typing import Optional
import utils


class MainTestSVDataset(data.Dataset):

    def __init__(self, root, data_type, mode, noise_trans: Optional[float] = None, noise_rotate: Optional[float] = None):

        self.frame_h = 720
        self.frame_w = 1280
        self.root = root
        self.data_type = data_type
        self.mode = mode
        self.noise_trans = noise_trans
        self.noise_rotate = noise_rotate

        sequence_interval = '80_95'
        self.image_path = osp.join(
            self.root, 'Dataset', sequence_interval)
        self.anno_path = osp.join(
            self.root, 'Annotations', sequence_interval)
        imgset_path = osp.join(self.root, self.data_type)

        self.videos = []
        self.num_frames = {}
        self.num_homographies = {}
        self.frames = []
        self.homographies = []

        with open(imgset_path + '.txt', 'r') as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(
                    glob.glob(osp.join(self.image_path, _video, '*.jpg')))
                self.num_homographies[_video] = len(
                    glob.glob(osp.join(self.anno_path, _video, '*_homography.npy')))

                frames = sorted(os.listdir(
                    osp.join(self.image_path, _video)))
                for img in frames:
                    self.frames.append(
                        osp.join(self.image_path, _video, img))

                homographies = sorted(os.listdir(
                    osp.join(self.anno_path, _video)))
                for mat in homographies:
                    self.homographies.append(
                        osp.join(self.anno_path, _video, mat))

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # ImageNet
        ])

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, index):

        image = np.array(Image.open(self.frames[index]))

        gt_h = np.load(self.homographies[index])

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

    # worldcup_loader = MainTestSVDataset(
    #     root='dataset/WorldCup_2014_2018', data_type='train', mode='train', noise_trans=5.0, noise_rotate=0.0084)
    worldcup_loader = MainTestSVDataset(
        root='dataset/WorldCup_2014_2018', data_type='test', mode='test')

    import shutil
    cnt = 1
    visual_dir = osp.join('visual', 'main_test_sv')
    if osp.exists(visual_dir):
        print(f'Remove directory: {visual_dir}')
        shutil.rmtree(visual_dir)
    print(f'Create directory: {visual_dir}')
    os.makedirs(visual_dir, exist_ok=True)

    for image, gt_heatmap, target, gt_homo in worldcup_loader:
        plt.imsave(osp.join(visual_dir, 'Rgb%03d.jpg' % (
            cnt)), utils.im_to_numpy(image))
        plt.imsave(osp.join(visual_dir, 'Seg%03d.jpg' % (
            cnt)), utils.to_numpy(gt_heatmap), vmin=0, vmax=91)
        cnt += 1
        if cnt == 10:
            break
        pass
