'''
testing file for Nie et al. (A robust and efficient framework for sports-field registration)
'''
import sys
sys.path.append('..')
from options import CustomOptions
from models.model import EncDec
from worldcup_loader import PublicWorldCupDataset
from ts_worldcup_loader import MainTestSVDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import os.path as osp
import time
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import utils
import metrics
import skimage.segmentation as ss


# Get input arguments
opt = CustomOptions(train=False)
opt = opt.parse()

# Setup GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
print('CUDA_VISIBLE_DEVICES: %s' % opt.gpu_ids)
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

        indices = np.where(pred_inds)
        coords = list(zip(indices[0], indices[1]))

        # the only keypoint with max confidence is greater than threshold or not
        if max_score >= nms_thres:
            pred_cls_dict[cls].append(max_score)
            pred_cls_dict[cls].append(coords[max_index])

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
    non_local = bool(opt.use_non_local)
    layers = 18

    # Initialize models
    model = EncDec(layers, num_classes, non_local).to(device)

    # Setup dataset
    if opt.train_stage == 0:
        # Load testing data
        print('Loading data from public world cup dataset...')
        test_dataset = PublicWorldCupDataset(
            root=opt.public_worldcup_root,
            data_type=opt.testset,
            mode='test'
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
    elif opt.train_stage == 1:
        # Load testing data
        print('Loading data from time sequence world cup dataset...')
        test_dataset = MainTestSVDataset(
            root=opt.custom_worldcup_root,
            data_type=opt.testset,
            mode='test'
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

    # Loss function
    class_weights = torch.ones(num_classes) * 100
    class_weights[0] = 1
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights)  # TODO: put class weight

    # Set data path
    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    test_visual_dir = osp.join(exp_name_path, 'imgs', 'test_visual')
    os.makedirs(test_visual_dir, exist_ok=True)

    iou_visual_dir = osp.join(test_visual_dir, 'iou')
    os.makedirs(iou_visual_dir, exist_ok=True)

    homo_visual_dir = osp.join(test_visual_dir, 'homography')
    os.makedirs(homo_visual_dir, exist_ok=True)

    field_model = Image.open(
        osp.join(opt.template_path, 'worldcup_field_model.png'))

    # TODO:: Load pretrained model or resume training
    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print('Loading weights: ', load_weights_path)
        assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
        checkpoint = torch.load(load_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print('Checkpoint Epoch: ', epoch)

    if opt.sfp_finetuned and opt.train_stage == 1:
        sfp_out_path = 'SingleFramePredict_finetuned_with_normalized'

    elif not opt.sfp_finetuned and opt.train_stage == 1:
        sfp_out_path = 'SingleFramePredict_with_normalized'

    elif not opt.sfp_finetuned and opt.train_stage == 0:
        # for test on worldcup dataset
        sfp_out_path = 'robust_worldcup_testset_dilated'

    print("Testing...")
    model.eval()
    batch_celoss = 0.0
    batch_l2loss = 0.0
    precision_list = []
    recall_list = []
    iou_part_list = []
    iou_whole_list = []
    proj_error_list = []
    reproj_error_list = []
    test_progress_bar = tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False)
    test_progress_bar.set_description(
        f'Epoch: {epoch}/{opt.train_epochs}')

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    cnt8 = 0
    cnt9 = 0
    cnt10 = 0

    with torch.no_grad():
        for step, (image, gt_heatmap, target, gt_homo) in test_progress_bar:
            image = image.to(device)
            gt_heatmap = gt_heatmap.to(device).long()

            pred_heatmap = model(image)

            # (B, 92, 180, 320), (B, 180, 320)
            loss = criterion(pred_heatmap, gt_heatmap)

            pred_heatmap = torch.softmax(pred_heatmap, dim=1)
            scores, pred_heatmap = torch.max(pred_heatmap, dim=1)
            scores = scores[0].detach().cpu().numpy()
            pred_heatmap = pred_heatmap[0].detach().cpu().numpy()
            gt_heatmap = gt_heatmap[0].cpu().numpy()
            target = target[0].cpu().numpy()
            gt_homo = gt_homo[0].cpu().numpy()

            gt_cls_dict, pred_cls_dict = postprocessing(
                scores, pred_heatmap, target, num_classes, opt.nms_thres)

            p, r, loss2 = calc_keypts_metrics(
                gt_cls_dict, pred_cls_dict, opt.pr_thres)

            precision_list.append(p)
            recall_list.append(r)

            batch_celoss += loss.detach()
            batch_l2loss += loss2.detach()

            # TODO: log loss
            if step % 10 == 9:
                batch_celoss /= 10
                print('Step: {}/{}\tTesting CE Loss: {:.4f}'.format(step,
                      len(test_loader), batch_celoss))
                batch_celoss = 0.0

                batch_l2loss /= 10
                print('Step: {}/{}\tTesting MSE Loss: {:.4f}'.format(step,
                      len(test_loader), batch_l2loss))
                batch_l2loss = 0.0

            image = utils.im_to_numpy(denorm(image[0]))

            # TODO: show keypoints visual result after postprocessing
            pred_keypoints = np.zeros_like(
                pred_heatmap, dtype=np.uint8)
            pred_rgb = []
            for ind, (pk, pv) in enumerate(pred_cls_dict.items()):
                if pv:
                    pred_keypoints[pv[1][0], pv[1][1]] = pk  # (H, W)
                    # camera view point sets (x, y, label) in rgb domain not heatmap domain
                    pred_rgb.append([pv[1][1] * 4, pv[1][0] * 4, pk])
            pred_rgb = np.asarray(pred_rgb, dtype=np.float32)  # (?, 3)
            pred_homo = None
            if pred_rgb.shape[0] >= 4:  # at least four points
                src_pts, dst_pts = class_mapping(pred_rgb)
                pred_homo, _ = cv2.findHomography(
                    src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10)
                if pred_homo is not None:
                    iou_part, gt_part_mask, pred_part_mask, part_merge_result = metrics.calc_iou_part(
                        pred_homo, gt_homo, image, field_model)
                    iou_part_list.append(iou_part)

                    iou_whole, whole_line_merge_result, whole_fill_merge_result = metrics.calc_iou_whole_with_poly(
                        pred_homo, gt_homo, image, field_model)
                    iou_whole_list.append(iou_whole)

                    proj_error = metrics.calc_proj_error(
                        pred_homo, gt_homo, image, field_model)
                    proj_error_list.append(proj_error)

                    reproj_error = metrics.calc_reproj_error(
                        pred_homo, gt_homo, image, field_model)
                    reproj_error_list.append(reproj_error)
                else:
                    print(f'pred homo is None at {step + 1}')
                    iou_part_list.append(float('nan'))
                    iou_whole_list.append(float('nan'))
                    proj_error_list.append(float('nan'))
                    reproj_error_list.append(float('nan'))
            else:
                print(f'less than four points at {step + 1}')
                iou_part_list.append(float('nan'))
                iou_whole_list.append(float('nan'))
                proj_error_list.append(float('nan'))
                reproj_error_list.append(float('nan'))

            # # TODO: save pic
            if True:
                # if False:
                pred_keypoints = ss.expand_labels(
                    pred_keypoints, distance=5)

                # TODO: save undilated heatmap for each testing video
                if step < 89:
                    if opt.train_stage == 1:
                        vid1_path = osp.join(
                            exp_name_path, sfp_out_path, '80_95/left/2014_Match_Highlights1_clip_00007-1')
                    elif opt.train_stage == 0:  # for test on worldcup dataset
                        vid1_path = osp.join(
                            exp_name_path, sfp_out_path, 'worldcup_2014')
                    os.makedirs(vid1_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid1_path, '%05d.png' %
                                (cnt1)), pred_keypoints)
                    cnt1 += 1

                elif step >= 89 and step < 172:
                    if opt.train_stage == 1:
                        vid2_path = osp.join(
                            exp_name_path, sfp_out_path, '80_95/left/2014_Match_Highlights2_clip_00006-1')
                    elif opt.train_stage == 0:  # for test on worldcup dataset
                        vid2_path = osp.join(
                            exp_name_path, sfp_out_path, 'worldcup_2014')
                    os.makedirs(vid2_path, exist_ok=True)
                    if opt.train_stage == 0:
                        cv2.imwrite(osp.join(vid2_path, '%05d.png' %
                                    (cnt1)), pred_keypoints)
                        cnt1 += 1
                    elif opt.train_stage == 1:
                        cv2.imwrite(osp.join(vid2_path, '%05d.png' %
                                    (cnt2)), pred_keypoints)
                        cnt2 += 1

                elif step >= 172 and step < 253:
                    if opt.train_stage == 1:
                        vid3_path = osp.join(
                            exp_name_path, sfp_out_path, '80_95/left/2014_Match_Highlights3_clip_00004-2')
                    elif opt.train_stage == 0:
                        vid3_path = osp.join(
                            exp_name_path, sfp_out_path, 'worldcup_2014')
                    os.makedirs(vid3_path, exist_ok=True)
                    if opt.train_stage == 0:
                        cv2.imwrite(osp.join(vid3_path, '%05d.png' %
                                    (cnt1)), pred_keypoints)
                        cnt1 += 1
                    elif opt.train_stage == 1:
                        cv2.imwrite(osp.join(vid3_path, '%05d.png' %
                                    (cnt3)), pred_keypoints)
                        cnt3 += 1

                elif step >= 253 and step < 340:
                    vid4_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/left/2014_Match_Highlights3_clip_00009-1')
                    os.makedirs(vid4_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid4_path, '%05d.png' %
                                (cnt4)), pred_keypoints)
                    cnt4 += 1

                elif step >= 340 and step < 426:
                    vid5_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/left/2014_Match_Highlights6_clip_00013-1')
                    os.makedirs(vid5_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid5_path, '%05d.png' %
                                (cnt5)), pred_keypoints)
                    cnt5 += 1

                elif step >= 426 and step < 514:
                    vid6_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/right/2014_Match_Highlights3_clip_00013-1')
                    os.makedirs(vid6_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid6_path, '%05d.png' %
                                (cnt6)), pred_keypoints)
                    cnt6 += 1

                elif step >= 514 and step < 609:
                    vid7_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/right/2014_Match_Highlights5_clip_00010-2')
                    os.makedirs(vid7_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid7_path, '%05d.png' %
                                (cnt7)), pred_keypoints)
                    cnt7 += 1

                elif step >= 609 and step < 702:
                    vid8_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/right/2018_Match_Highlights5_clip_00016-1')
                    os.makedirs(vid8_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid8_path, '%05d.png' %
                                (cnt8)), pred_keypoints)
                    cnt8 += 1

                elif step >= 702 and step < 793:
                    vid9_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/right/2018_Match_Highlights6_clip_00015-2')
                    os.makedirs(vid9_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid9_path, '%05d.png' %
                                (cnt9)), pred_keypoints)
                    cnt9 += 1

                elif step >= 793 and step < 887:
                    vid10_path = osp.join(
                        exp_name_path, sfp_out_path, '80_95/right/2018_Match_Highlights6_clip_00023-3')
                    os.makedirs(vid10_path, exist_ok=True)
                    cv2.imwrite(osp.join(vid10_path, '%05d.png' %
                                (cnt10)), pred_keypoints)
                    cnt10 += 1

                # plt.imsave(osp.join(test_visual_dir,
                #            'test_%05d_%05d_rgb.jpg' % (epoch, step)), image)
                plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred.png' % (
                    epoch, step)), pred_heatmap, vmin=0, vmax=91)
                plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_gt.png' % (
                    epoch, step)), gt_heatmap, vmin=0, vmax=91)
                # pred_keypoints = ss.expand_labels(
                #     pred_keypoints, distance=5)
                plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred_keypts.png' % (
                    epoch, step)), pred_keypoints, vmin=0, vmax=91)

            if True:
            # if False:
                if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                    # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_gt_iou_part.png' % (
                    #     epoch, step)), gt_part_mask)
                    # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_pred_iou_part.png' % (
                    #     epoch, step)), pred_part_mask)
                    # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_merge_iou_part.png' % (
                    #     epoch, step)), part_merge_result)
                    # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_line_iou_whole.png' % (
                    #     epoch, step)), whole_line_merge_result)
                    # plt.imsave(osp.join(iou_visual_dir, 'test_%05d_%05d_fill_iou_whole.png' % (
                    #     epoch, step)), whole_fill_merge_result)
                    np.save(osp.join(homo_visual_dir, 'test_%05d_%05d_gt_homography.npy' % (
                        epoch, step)), gt_homo)
                    np.save(osp.join(homo_visual_dir, 'test_%05d_%05d_pred_homography.npy' % (
                        epoch, step)), pred_homo)

        average_precision = np.array(precision_list).mean()
        average_recall = np.array(recall_list).mean()
        print(
            f'Average Precision: {average_precision:.2f}, Recall: {average_recall:.2f}')

        iou_part_list = np.array(iou_part_list)
        iou_whole_list = np.array(iou_whole_list)
        mean_iou_part = np.nanmean(iou_part_list)
        mean_iou_whole = np.nanmean(iou_whole_list)
        print(
            f'Mean IOU part: {mean_iou_part * 100.:.1f}, IOU whole: {mean_iou_whole * 100.:.1f}')

        median_iou_part = np.nanmedian(iou_part_list)
        median_iou_whole = np.nanmedian(iou_whole_list)
        print(
            f'Median IOU part: {median_iou_part * 100.:.1f}, IOU whole: {median_iou_whole * 100.:.1f}')

        proj_error_list = np.array(proj_error_list)
        reproj_error_list = np.array(reproj_error_list)
        mean_proj_error = np.nanmean(proj_error_list)
        mean_reproj_error = np.nanmean(reproj_error_list)
        print(
            f'Mean Projection Error: {mean_proj_error:.2f}, Reprojection Error: {mean_reproj_error:.3f}')

        median_proj_error = np.nanmedian(proj_error_list)
        median_reproj_error = np.nanmedian(reproj_error_list)
        print(
            f'Median Projection Error: {median_proj_error:.2f}, Reprojection Error: {median_reproj_error:.3f}')

        with open(osp.join(exp_name_path, 'metrics_%03d.txt' % epoch), 'w') as out_file:
            out_file.write(
                f'Loading weights: {load_weights_path}')
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


def main():

    test()


if __name__ == '__main__':

    start_time = time.time()
    main()
    print(f'Done...Take {(time.time() - start_time):.4f} (sec)')
