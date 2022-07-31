from numpy.lib import tile
from options import CustomOptions
from models.eval_network import EvalKpSFR
from models.inference_core import InferenceCore
from worldcup_test_loader import WorldcupTestDataset
from ts_worldcup_test_loader import MainTestDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import os.path as osp
import time
from PIL import Image
from tqdm import tqdm
import shutil

import cv2
import matplotlib.pyplot as plt
import utils
import metrics
import skimage.segmentation as ss


# Get input arguments
opt = CustomOptions(train=False)
opt = opt.parse()

# Log on tensorboard
# writer = SummaryWriter('runs/' + opt.name)

# Setup GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
print('CUDA Visible Devices: %s' % opt.gpu_ids)
device = torch.device('cuda:0')
print('device: %s' % device)


def calc_euclidean_distance(a, b, _norm=np.linalg.norm, axis=None):
    return _norm(a - b, axis=axis)


def my_mseloss(gt, pred):
    return torch.mean(torch.square(pred - gt))


def postprocessing(scores, pred, target, num_classes, nms_thres):

    # TODO: decode the heatmaps into keypoint sets using non-maximum suppression
    pred_cls_dict = {k: [] for k in range(1, num_classes)}
    for cls in range(1, num_classes):
        pred_inds = pred == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(pred_inds):
            continue

        values = scores[pred_inds]
        max_score = values.max()
        max_index = values.argmax()

        val_inds = np.where(values == max_score)[0]

        indices = np.where(pred_inds)
        coords = list(zip(indices[0], indices[1]))

        l = []
        for idx in range(val_inds.shape[0]):
            l.append(coords[val_inds[idx]])
        l = np.array(l).mean(axis=0).astype(np.int64)

        # the only keypoint with max confidence is greater than threshold or not
        if max_score >= nms_thres:
            pred_cls_dict[cls].append(max_score)
            pred_cls_dict[cls].append(l)

    gt_cls_dict = {k: [] for k in range(1, num_classes)}
    for cls in range(1, num_classes):
        gt_inds = target == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(gt_inds):
            continue
        coords = np.argwhere(gt_inds)[0]

        # coordinate order is (y, x)
        gt_cls_dict[cls].append((coords[0], coords[1]))

    return gt_cls_dict, pred_cls_dict


def calc_keypts_metrics(gt_cls_dict, pred_cls_dict, pr_thres):

    num_gt_pos = 0
    num_pred_pos = 0
    num_both_keypts_appear = 0
    tp = 0
    mse_loss = 0.0

    for (gk, gv), (pk, pv) in zip(gt_cls_dict.items(), pred_cls_dict.items()):
        if gv:
            num_gt_pos += 1

        if pv:
            num_pred_pos += 1

        if gv and pv:
            num_both_keypts_appear += 1
            mse_loss += my_mseloss(torch.FloatTensor(gv[0]),
                                   torch.FloatTensor(pv[1]))

            if calc_euclidean_distance(np.array(gv[0]), np.array(pv[1])) <= pr_thres:
                tp += 1

    if num_both_keypts_appear == 0:
        return 0.0, 0.0, 0.0
    return tp / num_pred_pos, tp / num_gt_pos, mse_loss / num_both_keypts_appear


def class_mapping(rgb):

    # TODO: class mapping
    template = utils.gen_template_grid()  # grid shape (91, 3), (x, y, label)
    src_pts = rgb.copy()
    cls_map_pts = []

    for ind, elem in enumerate(src_pts):
        coords = np.where(elem[2] == template[:, 2])[0]  # find correspondence
        cls_map_pts.append(template[coords[0]])
    dst_pts = np.array(cls_map_pts, dtype=np.float32)

    return src_pts[:, :2], dst_pts[:, :2]


def test():

    num_classes = 92
    # num_objects = opt.num_objects
    num_objects = 91
    non_local = bool(opt.use_non_local)
    model_archi = opt.model_archi

    # Initialize models
    eval_model = EvalKpSFR(model_archi=model_archi,
                           num_objects=num_objects, non_local=non_local).to(device)

    if opt.train_stage == 0:
        # Load testing data
        print('Loading public worldcup testing data...')
        test_dataset = WorldcupTestDataset(
            root=opt.public_worldcup_root,
            data_type=opt.testset,
            mode='test',
            num_objects=num_objects
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
    elif opt.train_stage == 1:
        # Load testing data
        print('Loading time sequence worldcup testing data...')
        test_dataset = MainTestDataset(
            root=opt.custom_worldcup_root,
            data_type=opt.testset,
            mode='test',
            num_objects=num_objects,
            sfp_finetuned=opt.sfp_finetuned
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

    total_epoch = opt.train_epochs

    # Set data path
    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    test_visual_dir = osp.join(exp_name_path, 'imgs', 'test_visual')
    if osp.exists(test_visual_dir):
        print(f'Remove directory: {test_visual_dir}')
        shutil.rmtree(test_visual_dir)
    print(f'Create directory: {test_visual_dir}')
    os.makedirs(test_visual_dir, exist_ok=True)

    # iou_visual_dir = osp.join(test_visual_dir, 'iou')
    # os.makedirs(iou_visual_dir, exist_ok=True)

    homo_visual_dir = osp.join(exp_name_path, 'homography')
    os.makedirs(homo_visual_dir, exist_ok=True)

    vid1_path = osp.join(
        homo_visual_dir, 'left/2014_Match_Highlights1_clip_00007-1')
    os.makedirs(vid1_path, exist_ok=True)

    vid2_path = osp.join(
        homo_visual_dir, 'left/2014_Match_Highlights2_clip_00006-1')
    os.makedirs(vid2_path, exist_ok=True)

    vid3_path = osp.join(
        homo_visual_dir, 'left/2014_Match_Highlights3_clip_00004-2')
    os.makedirs(vid3_path, exist_ok=True)

    vid4_path = osp.join(
        homo_visual_dir, 'left/2014_Match_Highlights3_clip_00009-1')
    os.makedirs(vid4_path, exist_ok=True)

    vid5_path = osp.join(
        homo_visual_dir, 'left/2014_Match_Highlights6_clip_00013-1')
    os.makedirs(vid5_path, exist_ok=True)

    vid6_path = osp.join(
        homo_visual_dir, 'right/2014_Match_Highlights3_clip_00013-1')
    os.makedirs(vid6_path, exist_ok=True)

    vid7_path = osp.join(
        homo_visual_dir, 'right/2014_Match_Highlights5_clip_00010-2')
    os.makedirs(vid7_path, exist_ok=True)

    vid8_path = osp.join(
        homo_visual_dir, 'right/2018_Match_Highlights5_clip_00016-1')
    os.makedirs(vid8_path, exist_ok=True)

    vid9_path = osp.join(
        homo_visual_dir, 'right/2018_Match_Highlights6_clip_00015-2')
    os.makedirs(vid9_path, exist_ok=True)

    vid10_path = osp.join(
        homo_visual_dir, 'right/2018_Match_Highlights6_clip_00023-3')
    os.makedirs(vid10_path, exist_ok=True)

    field_model = Image.open(
        osp.join(opt.template_path, 'worldcup_field_model.png'))

    best_epoch = 0
    best_mean_iou_part = float('-inf')
    best_median_iou_part = float('-inf')
    best_mean_iou_whole = float('-inf')
    best_median_iou_whole = float('-inf')
    best_mean_proj = float('inf')
    best_median_proj = float('inf')
    best_mean_reproj = float('inf')
    best_median_reproj = float('inf')

    # TODO: Load pretrained model or resume training
    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print('Loading weights: ', load_weights_path)
        assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
        checkpoint = torch.load(load_weights_path, map_location=device)
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print('Checkpoint Epoch:', epoch)

    # epoch = 300
    print("Testing...")
    eval_model.eval()
    avg_batch_l2loss = 0.0
    avg_precision_list = []
    avg_recall_list = []
    avg_iou_part_list = []
    avg_iou_whole_list = []
    avg_proj_error_list = []
    avg_reproj_error_list = []
    test_progress_bar = tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False)
    test_progress_bar.set_description(
        f'Epoch: {epoch}/{total_epoch}')

    total_process_time = 0
    total_frames = 0

    with torch.no_grad():
        for step, data in test_progress_bar:
            image = data['rgb'].to(device)  # b*t*c*h*w
            target_dilated_hm = data['target_dilated_hm'][0].to(
                device)  # k*t*1*h*w
            cls_gt = data['cls_gt'][0]  # t*h*w
            gt_homo = data['gt_homo'][0]
            selector = data['selector'][0].to(device)  # k:91 or t*k
            lookup = data['lookup'][0].to(device)  # k:91 or t*k
            info = data['info']
            k = info['num_objects'][0]  # 91
            sfp_path = info['single_frame_path'][0]

            torch.cuda.synchronize()
            process_begin = time.time()

            processor = InferenceCore(eval_model, image, device, k, lookup)
            # selector does not use
            processor.interact(0, image.shape[1], selector)

            size = target_dilated_hm.shape[-2:]
            out_masks = torch.zeros((processor.t, 1, *size), device=device)
            out_scores = torch.zeros_like(out_masks)

            for ti in range(processor.t):
                prob = processor.prob[:, ti]
                out_scores[ti], out_masks[ti] = torch.max(
                    prob, dim=0)  # 1*h*w
            out_masks = out_masks.detach().cpu().numpy()[:, 0]  # t*h*w
            out_scores = out_scores.detach().cpu().numpy()[:, 0]  # t*h*w

            image = np.transpose(denorm(image[0]).detach(
            ).cpu().numpy(), (0, 2, 3, 1))  # t*h*w*c
            cls_gt = cls_gt.cpu().numpy()  # t*h*w
            gt_homo = gt_homo.cpu().numpy()

            torch.cuda.synchronize()
            total_process_time += time.time() - process_begin
            total_frames += out_masks.shape[0]

            print(f'Video {step + 1} start processing...')
            for ti in range(processor.t):
                print(f'Current frame is {ti}')
                # print('scores: ',
                #       out_scores[ti].min(), out_scores[ti].max())
                gt_cls_dict, pred_cls_dict = postprocessing(
                    out_scores[ti], out_masks[ti], cls_gt[ti], num_classes, opt.nms_thres)
                # No any point after postprocessing
                if not any(pred_cls_dict.values()):
                    print(f'not keypts at {ti}')
                    plt.imsave(osp.join(exp_name_path, 'imgs', 'test_%05d_%05d_pred_not_keypts.png' % (
                        epoch, step)), out_masks[ti], vmin=0, vmax=processor.k)
                    continue

                p, r, loss2 = calc_keypts_metrics(
                    gt_cls_dict, pred_cls_dict, opt.pr_thres)
                if p == 0 and r == 0 and loss2 == 0:  # No common point appeared
                    print(f'diff location at {ti}')
                    plt.imsave(osp.join(exp_name_path, 'imgs', 'test_%05d_%05d_pred_diff_location.png' % (
                        epoch, step)), out_masks[ti], vmin=0, vmax=processor.k)
                    continue

                avg_precision_list.append(p)
                avg_recall_list.append(r)

                avg_batch_l2loss += loss2.detach()

                # TODO: show keypoints visual result after postprocessing
                pred_keypoints = np.zeros_like(out_masks[0])
                pred_rgb = []
                for ind, (pk, pv) in enumerate(pred_cls_dict.items()):
                    if pv:
                        pred_keypoints[pv[1][0],
                                       pv[1][1]] = pk  # (H, W)
                        # camera view point sets (x, y, label) in rgb domain not heatmap domain
                        pred_rgb.append(
                            [pv[1][1] * 4, pv[1][0] * 4, pk])
                pred_rgb = np.asarray(
                    pred_rgb, dtype=np.float32)  # (?, 3)

                pred_homo = None
                if pred_rgb.shape[0] >= 4:  # at least four points
                    src_pts, dst_pts = class_mapping(pred_rgb)

                    pred_homo, _ = cv2.findHomography(
                        src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10)

                    if pred_homo is not None:
                        iou_part, gt_part_mask, pred_part_mask, part_merge_result = metrics.calc_iou_part(
                            pred_homo, gt_homo[ti], image[ti], field_model)
                        avg_iou_part_list.append(iou_part)

                        # Bugs still existing
                        iou_whole, whole_line_merge_result, whole_fill_merge_result = metrics.calc_iou_whole_with_poly(
                            pred_homo, gt_homo[ti], image[ti], field_model)
                        avg_iou_whole_list.append(iou_whole)

                        proj_error = metrics.calc_proj_error(
                            pred_homo, gt_homo[ti], image[ti], field_model)
                        avg_proj_error_list.append(proj_error)

                        reproj_error = metrics.calc_reproj_error(
                            pred_homo, gt_homo[ti], image[ti], field_model)
                        avg_reproj_error_list.append(reproj_error)
                    else:
                        print(f'pred homo is None at {ti}')
                        avg_iou_part_list.append(float('nan'))
                        avg_iou_whole_list.append(float('nan'))
                        avg_proj_error_list.append(float('nan'))
                        avg_reproj_error_list.append(float('nan'))
                else:
                    print(f'less than four points at {ti}')
                    avg_iou_part_list.append(float('nan'))
                    avg_iou_whole_list.append(float('nan'))
                    avg_proj_error_list.append(float('nan'))
                    avg_reproj_error_list.append(float('nan'))

                # TODO: save undilated heatmap for each testing video
                if step == 0:
                    if opt.train_stage == 1:
                        vid1_path_m = osp.join(
                            exp_name_path, model_archi, '80_95/left/2014_Match_Highlights1_clip_00007-1')
                    elif opt.train_stage == 0:
                        vid1_path_m = osp.join(
                            exp_name_path, model_archi, 'worldcup_2014')  # for evaluate worldcup test set
                    os.makedirs(vid1_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid1_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid1_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 1:
                    if opt.train_stage == 1:
                        vid2_path_m = osp.join(
                            exp_name_path, model_archi, '80_95/left/2014_Match_Highlights2_clip_00006-1')
                    elif opt.train_stage == 0:
                        vid2_path_m = osp.join(
                            exp_name_path, model_archi, 'worldcup_2014')  # for evaluate worldcup test set
                    os.makedirs(vid2_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid2_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid2_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 2:
                    if opt.train_stage == 1:
                        vid3_path_m = osp.join(
                            exp_name_path, model_archi, '80_95/left/2014_Match_Highlights3_clip_00004-2')
                    elif opt.train_stage == 0:
                        vid3_path_m = osp.join(
                            exp_name_path, model_archi, 'worldcup_2014')  # for evaluate worldcup test set
                    os.makedirs(vid3_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid3_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid3_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 3:
                    vid4_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/left/2014_Match_Highlights3_clip_00009-1')
                    os.makedirs(vid4_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid4_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid4_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 4:
                    vid5_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/left/2014_Match_Highlights6_clip_00013-1')
                    os.makedirs(vid5_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid5_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid5_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 5:
                    vid6_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/right/2014_Match_Highlights3_clip_00013-1')
                    os.makedirs(vid6_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid6_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid6_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 6:
                    vid7_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/right/2014_Match_Highlights5_clip_00010-2')
                    os.makedirs(vid7_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid7_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid7_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 7:
                    vid8_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/right/2018_Match_Highlights5_clip_00016-1')
                    os.makedirs(vid8_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid8_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid8_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 8:
                    vid9_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/right/2018_Match_Highlights6_clip_00015-2')
                    os.makedirs(vid9_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid9_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid9_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))
                elif step == 9:
                    vid10_path_m = osp.join(
                        exp_name_path, model_archi, '80_95/right/2018_Match_Highlights6_clip_00023-3')
                    os.makedirs(vid10_path_m, exist_ok=True)
                    cv2.imwrite(osp.join(vid10_path_m, '%05d.png' %
                                ti), np.uint8(pred_keypoints))
                    cv2.imwrite(osp.join(vid10_path_m, '%05d_gt.png' %
                                ti), np.uint8(cls_gt[ti]))

                # TODO: save heatmap for visual result
                if True:
                    # if False:
                    gt_keypoints = ss.expand_labels(
                        cls_gt[ti], distance=5)
                    plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_gt_seg%02d.png' % (
                        epoch, step, ti)), gt_keypoints, vmin=0, vmax=processor.k)
                    plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred_seg%02d.png' % (
                        epoch, step, ti)), out_masks[ti], vmin=0, vmax=processor.k)
                    pred_keypoints = ss.expand_labels(
                        pred_keypoints, distance=5)
                    plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred_keypts%02d.png' % (
                        epoch, step, ti)), pred_keypoints, vmin=0, vmax=processor.k)

                # TODO: save homography
                if True:
                    # if False:
                    if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                        # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_gt_iou_part%02d.png' % (
                        #     epoch, step, ti)), gt_part_mask)
                        # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_pred_iou_part%02d.png' % (
                        #     epoch, step, ti)), pred_part_mask)
                        # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_merge_iou_part%02d.png' % (
                        #     epoch, step, ti)), part_merge_result)
                        # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_line_iou_whole%02d.png' % (
                        #     epoch, step, ti)), whole_line_merge_result)
                        # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_fill_iou_whole%02d.png' % (
                        #     epoch, step, ti)), whole_fill_merge_result)

                        if step == 0:
                            np.save(osp.join(vid1_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid1_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 1:
                            np.save(osp.join(vid2_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid2_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 2:
                            np.save(osp.join(vid3_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid3_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 3:
                            np.save(osp.join(vid4_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid4_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 4:
                            np.save(osp.join(vid5_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid5_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 5:
                            np.save(osp.join(vid6_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid6_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 6:
                            np.save(osp.join(vid7_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid7_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 7:
                            np.save(osp.join(vid8_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid8_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 8:
                            np.save(osp.join(vid9_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid9_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)
                        elif step == 9:
                            np.save(osp.join(vid10_path, 'test_%05d_%05d_gt_homography%02d.npy' % (
                                epoch, step, ti)), gt_homo[ti])
                            np.save(osp.join(vid10_path, 'test_%05d_%05d_pred_homography%02d.npy' % (
                                epoch, step, ti)), pred_homo)

            print(f'Video {step + 1} is done...')

            del image
            del target_dilated_hm
            del selector
            del lookup
            del processor

        avg_batch_l2loss /= len(avg_precision_list)

        # TODO: log loss
        print(f'Testing MSE Loss: {avg_batch_l2loss:.4f}')
        # writer.add_scalar('Loss/MSE', avg_batch_l2loss, epoch)

        average_precision = np.array(avg_precision_list).mean()
        average_recall = np.array(avg_recall_list).mean()
        # average_precision = 0
        # average_recall = 0
        print(
            f'Average Precision: {average_precision:.2f}, Recall: {average_recall:.2f}')
        # writer.add_scalar(
        #     'Metrics/average keypoints precision', average_precision, epoch)
        # writer.add_scalar(
        #     'Metrics/average keypoints recall', average_recall, epoch)

        iou_part_list = np.array(avg_iou_part_list)
        iou_whole_list = np.array(avg_iou_whole_list)
        # print('IoU part length:', len(iou_part_list),
        #       'exclude frame 0:', len(iou_part_list)-10)
        mean_iou_part = np.nanmean(iou_part_list)
        mean_iou_whole = np.nanmean(iou_whole_list)
        # mean_iou_whole = 0
        print(
            f'Mean IOU part: {mean_iou_part * 100.:.1f}, IOU whole: {mean_iou_whole * 100.:.1f}')
        # writer.add_scalar('Metrics/mean IOU part',
        #                   mean_iou_part * 100., epoch)
        # writer.add_scalar('Metrics/mean IOU whole',
        #                   mean_iou_whole * 100., epoch)

        median_iou_part = np.nanmedian(iou_part_list)
        median_iou_whole = np.nanmedian(iou_whole_list)
        # median_iou_whole = 0
        print(
            f'Median IOU part: {median_iou_part * 100.:.1f}, IOU whole: {median_iou_whole * 100.:.1f}')
        # writer.add_scalar('Metrics/median IOU part',
        #                   median_iou_part * 100., epoch)
        # writer.add_scalar('Metrics/median IOU whole',
        #                   median_iou_whole * 100., epoch)

        proj_error_list = np.array(avg_proj_error_list)
        reproj_error_list = np.array(avg_reproj_error_list)
        mean_proj_error = np.nanmean(proj_error_list)
        mean_reproj_error = np.nanmean(reproj_error_list)
        print(
            f'Mean Projection Error: {mean_proj_error:.2f}, Reprojection Error: {mean_reproj_error:.3f}')
        # writer.add_scalar('Metrics/mean Projection Error',
        #                   mean_proj_error, epoch)
        # writer.add_scalar(
        #     'Metrics/mean Reprojection Error', mean_reproj_error, epoch)

        median_proj_error = np.nanmedian(proj_error_list)
        median_reproj_error = np.nanmedian(reproj_error_list)
        print(
            f'Median Projection Error: {median_proj_error:.2f}, Reprojection Error: {median_reproj_error:.3f}')
        # writer.add_scalar(
        #     'Metrics/median Projection Error', median_proj_error, epoch)
        # writer.add_scalar(
        #     'Metrics/median Reprojection Error', median_reproj_error, epoch)

        with open(osp.join(exp_name_path, 'metrics_%05d.txt' % epoch), 'w') as out_file:
            out_file.write(
                f'Loading weights: {load_weights_path}')
            out_file.write('\n')
            out_file.write(
                f'Path of single frame prediction: {sfp_path}')
            out_file.write('\n')
            out_file.write(f'Model architecture: {model_archi}')
            out_file.write('\n')
            out_file.write(
                f'Average Precision: {average_precision:.2f}, Recall: {average_recall:.2f}')
            out_file.write('\n')
            out_file.write(
                f'Mean IOU part: {mean_iou_part * 100.:.1f}, IOU whole: {mean_iou_whole * 100.:.1f}')
            out_file.write('\n')
            out_file.write(
                f'Median IOU part: {median_iou_part * 100.:.1f}, IOU whole: {median_iou_whole * 100.:.1f}')
            out_file.write('\n')
            out_file.write(
                f'Mean Projection Error: {mean_proj_error:.2f}, Reprojection Error: {mean_reproj_error:.3f}')
            out_file.write('\n')
            out_file.write(
                f'Median Projection Error: {median_proj_error:.2f}, Reprojection Error: {median_reproj_error:.3f}')
            out_file.write('\n')

        cnt_metrics = 0
        if mean_iou_part >= best_mean_iou_part:
            cnt_metrics += 1
        if median_iou_part >= best_median_iou_part:
            cnt_metrics += 1
        if mean_iou_whole >= best_mean_iou_whole:
            cnt_metrics += 1
        if median_iou_whole >= best_median_iou_whole:
            cnt_metrics += 1
        if mean_proj_error <= best_mean_proj:
            cnt_metrics += 1
        if median_proj_error <= best_median_proj:
            cnt_metrics += 1
        if mean_reproj_error <= best_mean_reproj:
            cnt_metrics += 1
        if median_reproj_error <= best_median_reproj:
            cnt_metrics += 1
        if cnt_metrics >= 8:
            best_epoch = epoch
            best_mean_iou_part = mean_iou_part
            best_median_iou_part = median_iou_part
            best_mean_iou_whole = mean_iou_whole
            best_median_iou_whole = median_iou_whole
            best_mean_proj = mean_proj_error
            best_median_proj = median_proj_error
            best_mean_reproj = mean_reproj_error
            best_median_reproj = median_reproj_error
        print('-'*10)
        print(f'The Best Epoch: {best_epoch}')
        print(
            f'The Best Mean Iou Part: {best_mean_iou_part * 100.:.1f}')
        print(
            f'The Best Median Iou Part: {best_median_iou_part * 100.:.1f}')
        print(
            f'The Best Mean Iou Whole: {best_mean_iou_whole * 100.:.1f}')
        print(
            f'The Best Median Iou Whole: {best_median_iou_whole * 100.:.1f}')
        print(
            f'The Best Mean Projection Error: {best_mean_proj:.2f}')
        print(
            f'The Best Median Projection Error: {best_median_proj:.2f}')
        print(
            f'The Best Mean Reprojection Error: {best_mean_reproj:.3f}')
        print(
            f'The Best Median Reprojection Error: {best_median_reproj:.3f}')
        print('-'*10)
        print('Total processing time: ', total_process_time)
        print('Total processed frames: ', total_frames)
        print(f'FPS: {(total_frames / total_process_time):.3f}')


def main():

    test()
    # writer.flush()
    # writer.close()


if __name__ == '__main__':

    start_time = time.time()
    main()
    print(f'Done...Take {(time.time() - start_time):.4f} (sec)')
