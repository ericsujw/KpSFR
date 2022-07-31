'''
random pick 4 from all keypoints
'''

import random
import glob
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import skimage.segmentation as ss
from typing import Optional
import utils


class StaticTransformDataset(data.Dataset):

    def __init__(self, root, data_type, mode, num_objects, noise_trans: Optional[float] = None, noise_rotate: Optional[float] = None):

        self.frame_h = 720
        self.frame_w = 1280
        self.root = root
        self.data_type = data_type
        self.mode = mode
        self.num_objects = num_objects
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
                                 [0.229, 0.224, 0.225]),  # ImageNet
        ])

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, index):

        image = np.array(Image.open(
            osp.join(self.root, self.data_type, self.frames[index])))

        gt_h = np.loadtxt(
            osp.join(self.root, self.data_type, self.homographies[index]))

        template_grid = utils.gen_template_grid()  # template grid shape (91, 3)

        image_list = []
        homo_mat_list = []
        pairwise_seed = random.randint(0, 2147483647)
        f1_seed = random.randint(0, 2147483647)
        f2_seed = random.randint(0, 2147483647)
        f3_seed = random.randint(0, 2147483647)
        choice1_cls_seed = random.randint(0, 2147483647)
        choice2_cls_seed = random.randint(0, 2147483647)
        choice3_cls_seed = random.randint(0, 2147483647)
        obj_seed = random.randint(0, 2147483647)
        dilated_hm_list = []

        # TODO: augmentation to get warp_image, warp_grid, heatmap, pert_homo of each training sample
        for f in range(3):
            if f == 0:
                random.seed(f1_seed)
            elif f == 1:
                random.seed(f2_seed)
            elif f == 2:
                random.seed(f3_seed)

            # warp grid shape (91, 3)
            warp_image, warp_grid, homo_mat = utils.gen_im_whole_grid(
                self.mode, image, f, gt_h, template_grid, self.noise_trans, self.noise_rotate, index)

            # Each keypoints is considered as an object
            num_pts = warp_grid.shape[0]
            pil_image = Image.fromarray(warp_image)

            # TODO: apply random horizontal flip to all the image and grid points
            random.seed(pairwise_seed)
            if self.mode == 'train' and random.random() < 0.5:
                pil_image, warp_grid = utils.put_lrflip_augmentation(
                    pil_image, warp_grid)

            image_tensor = self.preprocess(pil_image)
            image_list.append(image_tensor)
            homo_mat_list.append(homo_mat)

            # By default, all keypoints belong to background
            # C*H*W, C:91, exclude background class
            heatmaps = np.zeros(
                (num_pts, self.frame_h // 4, self.frame_w // 4), dtype=np.float32)
            dilated_heatmaps = np.zeros_like(heatmaps)

            for keypts_label in range(num_pts):
                if np.isnan(warp_grid[keypts_label, 0]) and np.isnan(warp_grid[keypts_label, 1]):
                    continue
                px = np.rint(warp_grid[keypts_label, 0] / 4).astype(np.int32)
                py = np.rint(warp_grid[keypts_label, 1] / 4).astype(np.int32)
                cls = int(warp_grid[keypts_label, 2]) - 1
                if 0 <= px < (self.frame_w // 4) and 0 <= py < (self.frame_h // 4):
                    heatmaps[cls][py, px] = warp_grid[keypts_label, 2]
                    dilated_heatmaps[cls] = ss.expand_labels(
                        heatmaps[cls], distance=5)
            dilated_hm_list.append(dilated_heatmaps)

        # Those keypoints appears on the first frame
        labels = np.unique(dilated_hm_list[0])
        labels = labels[labels != 0]  # Remove background class

        dilated_hm_list = np.stack(dilated_hm_list, axis=0)  # 3*91*H*W
        T, _, H, W = dilated_hm_list.shape

        # TODO: keypoints appear/disappear augmentation
        target_dilated_hm_list = torch.zeros((self.num_objects, T, H, W))
        lookup_list = []
        for f in range(3):
            labels = np.unique(dilated_hm_list[0])
            labels = labels[labels != 0]  # remove background class
            lookup = np.ones(self.num_objects, dtype=np.float32) * -1

            # hard level
            if f == 0:
                random.seed(choice1_cls_seed)
            elif f == 1:
                random.seed(choice2_cls_seed)
            elif f == 2:
                random.seed(choice3_cls_seed)

            if len(labels) < 4:
                print('b', labels.tolist())
                for idx, obj in enumerate(labels):
                    lookup[idx] = obj
            else:
                for idx in range(self.num_objects):
                    if len(labels) > 0:
                        target_object = random.choice(labels)
                        labels = labels[labels != target_object]
                        lookup[idx] = target_object
                    else:
                        print('Less than four classes')

            lookup_list.append(lookup)

        lookup_list = np.stack(lookup_list, axis=0)  # T*CK:4

        # Label reorder
        new_lookup_list = torch.ones((3, self.num_objects)) * -1
        new_selector_list = torch.ones_like(new_lookup_list)

        inter01 = np.intersect1d(lookup_list[0], lookup_list[1])
        non_inter01 = np.setdiff1d(lookup_list[0], lookup_list[1])
        non_inter10 = np.setdiff1d(lookup_list[1], lookup_list[0])
        new0 = np.concatenate((inter01, non_inter01), axis=0)
        new1 = np.concatenate((inter01, non_inter10), axis=0)
        inter12, inter1_ind, _ = np.intersect1d(
            new1, lookup_list[2], return_indices=True)
        non_inter21 = np.setdiff1d(lookup_list[2], new1)
        new_lookup_list[0, :] = utils.to_torch(new0)
        new_lookup_list[1, :] = utils.to_torch(new1)
        new_lookup_list[2, inter1_ind] = utils.to_torch(inter12)
        remain_ind = torch.where(new_lookup_list[2] == -1)[0]
        new_lookup_list[2, remain_ind] = utils.to_torch(non_inter21)

        new_selector_list[new_lookup_list == -1] = 0

        dilated_hm_list = utils.to_torch(dilated_hm_list)
        for f in range(3):
            for idx, obj in enumerate(new_lookup_list[f]):
                if obj != -1:
                    target_dilated_hm = dilated_hm_list[f, int(
                        obj)-1].clone()  # H*W
                    target_dilated_hm[target_dilated_hm == obj] = 1
                    target_dilated_hm_list[idx, f] = target_dilated_hm

        # TODO: union of ground truth segmentation of all objects
        cls_gt = torch.zeros((3, H, W))
        for f in range(3):
            for idx in range(self.num_objects):
                cls_gt[f][target_dilated_hm_list[idx, f] == 1] = idx + 1

        image_list = torch.stack(image_list, dim=0)  # (3, 3, 720, 1280)
        homo_mat_list = np.stack(homo_mat_list, axis=0)

        data = {}
        data['rgb'] = image_list
        data['target_dilated_hm'] = target_dilated_hm_list
        data['cls_gt'] = cls_gt
        data['gt_homo'] = homo_mat_list
        data['selector'] = new_selector_list
        data['lookup'] = new_lookup_list

        return data


if __name__ == "__main__":

    static_loader = StaticTransformDataset(
        root='./dataset/soccer_worldcup_2014/soccer_data', data_type='train_val', mode='train', num_objects=4, noise_trans=5.0, noise_rotate=0.0084)

    import shutil
    cnt = 1
    visual_dir = osp.join('visual', 'static')
    if osp.exists(visual_dir):
        print(f'Remove directory: {visual_dir}')
        shutil.rmtree(visual_dir)
    print(f'Create directory: {visual_dir}')
    os.makedirs(visual_dir, exist_ok=True)

    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for idx, data in enumerate(static_loader):
        image = data['rgb']
        mask = data['target_dilated_hm']
        cls_gt = data['cls_gt']
        lookup = data['lookup']
        # === debug ===
        print(f'number of frames: {cls_gt.shape[0]}')
        print(image.shape, mask.shape, cls_gt.shape)
        print('lookup:\n', lookup)
        for j in range(cls_gt.shape[0]):
            print(torch.unique(cls_gt[j]))
            plt.imsave(osp.join(visual_dir, 'seg%03d.jpg' %
                       (j + 1)), utils.to_numpy(cls_gt[j]), vmin=0, vmax=4)
            plt.imsave(osp.join(visual_dir, 'rgb%03d.jpg' %
                       (j + 1)), utils.im_to_numpy(denorm(image[j])))
            for i in range(4):
                plt.imsave(osp.join(visual_dir, '%d_dilated_mask_obj%d.jpg' % (
                    j + 1, i + 1)), utils.to_numpy(mask[i, j]))
        cnt += 1
        assert False
        pass
