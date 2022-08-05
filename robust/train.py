'''
training and testing file for Nie et al. (A robust and efficient framework for sports-field registration)
'''
import sys
sys.path.append('..')
import skimage.segmentation as ss
import metrics
import utils
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import time
import os.path as osp
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
import torch
from ts_worldcup_loader import MainTestSVDataset
from worldcup_loader import PublicWorldCupDataset
from models.model import EncDec
from options import CustomOptions
from numpy.lib.npyio import load


# Get input arguments
opt = CustomOptions(train=True)
opt = opt.parse()

# Log on tensorboard
writer = SummaryWriter('runs/' + opt.name)

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


def train():

    num_classes = 92
    non_local = bool(opt.use_non_local)
    layers = 18

    # Reproducibility
    utils.reseed(seed=14159265)

    # Initialize models
    model = EncDec(layers, num_classes, non_local).to(device)

    # Setup dataset
    if opt.train_stage == 0:
        # Load training data
        print('Loading data from public world cup dataset...')
        train_dataset = PublicWorldCupDataset(
            root=opt.public_worldcup_root,
            data_type=opt.trainset,
            mode='train',
            noise_trans=opt.noise_trans,
            noise_rotate=opt.noise_rotate
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Load testing data
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
        # Load training data
        print('Loading data from time sequence world cup dataset...')
        train_dataset = MainTestSVDataset(
            root=opt.custom_worldcup_root,
            data_type=opt.trainset,
            mode='train',
            noise_trans=opt.noise_trans,
            noise_rotate=opt.noise_rotate
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Load testing data
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
    class_weights = torch.ones(num_classes, device=device) * 100
    class_weights[0] = 1
    criterion = nn.CrossEntropyLoss(
        weight=class_weights)  # TODO: put class weight

    optimizer = optim.Adam(model.parameters(),
                           lr=opt.train_lr,
                           betas=(0.9, 0.999),
                           weight_decay=opt.weight_decay
                           )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.step_size, gamma=0.1)

    # Set data path
    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    train_visual_dir = osp.join(exp_name_path, 'imgs', 'train_visual')
    os.makedirs(train_visual_dir, exist_ok=True)

    test_visual_dir = osp.join(exp_name_path, 'imgs', 'test_visual')
    os.makedirs(test_visual_dir, exist_ok=True)

    weight_save_dir = osp.join(exp_name_path, 'weights')
    os.makedirs(weight_save_dir, exist_ok=True)

    field_model = Image.open(
        osp.join(opt.template_path, 'worldcup_field_model.png'))

    best_epoch = 0
    best_mean_iou_part = float('-inf')
    best_median_iou_part = float('-inf')
    best_mean_proj = float('inf')
    best_median_proj = float('inf')
    best_mean_reproj = float('inf')
    best_median_reproj = float('inf')

    # TODO: Load pretrained model or resume training
    start_epoch = 0
    load_weights_path = ''
    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print('Loading weights: ', load_weights_path)
        assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
        checkpoint = torch.load(load_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if opt.resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print('Start Epoch: ', start_epoch)

    # Training loop
    for epoch in range(start_epoch, opt.train_epochs):
        print("Training...")
        model.train()
        batch_loss = 0.0
        epoch_loss = 0.0
        train_progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False)

        for step, (image, gt_heatmap, target, gt_homo) in train_progress_bar:
            image = image.to(device, non_blocking=True)
            gt_heatmap = gt_heatmap.to(device, non_blocking=True).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # return heatmap of uniform grid, N+1 channels
            pred_heatmap = model(image)

            # (4, 92, 180, 320), (4, 180, 320)
            loss = criterion(pred_heatmap, gt_heatmap)

            # accumulates the parameters gradients
            loss.backward()
            optimizer.step()

            train_progress_bar.set_description(
                f'Epoch: {epoch + 1}/{opt.train_epochs}')
            train_progress_bar.set_postfix(loss=loss.detach())

            batch_loss += loss.detach()
            epoch_loss += loss.detach()

            # TODO: log loss
            if step % 10 == 9:  # every 10 mini-batches
                batch_loss /= 10
                print('Epoch: {}/{}, Step: {}/{}\tTraining Loss: {:.4f}'.format(epoch +
                      1, opt.train_epochs, step, len(train_loader), batch_loss))
                # writer.add_scalar('Loss/train', batch_loss,
                #                   epoch * len(train_loader) + step)
                batch_loss = 0.0

            # TODO: save pic and eval metrics
            # if epoch % 100 == 99 and step % 10 == 9:  # WorldCup
            if epoch % 100 == 99 and step % 100 == 99:  # TS-WorldCup
                image = utils.im_to_numpy(denorm(image[0]))
                pred_heatmap = torch.softmax(pred_heatmap, dim=1)
                scores, pred_heatmap = torch.max(pred_heatmap, dim=1)
                pred_heatmap = pred_heatmap[0].detach().cpu().numpy()
                gt_heatmap = gt_heatmap[0].cpu().numpy()

                plt.imsave(osp.join(
                    train_visual_dir, 'train_%05d_%05d_rgb.jpg' % (epoch + 1, step)), image)
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_pred.png' % (
                    epoch + 1, step)), pred_heatmap, vmin=0, vmax=91)
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_gt.png' % (
                    epoch + 1, step)), gt_heatmap, vmin=0, vmax=91)

        epoch_loss /= len(train_loader)
        print('Epoch: {}/{}\t\t\tTraining Loss: {:.4f}'.format(epoch +
                                                               1, opt.train_epochs, epoch_loss))
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)

        scheduler.step()

        # TODO: test the model after 10 epochs
        if (epoch + 1) >= 10:
            print("Testing...")
            model.eval()
            batch_celoss = 0.0
            batch_l2loss = 0.0
            epoch_celoss = 0.0
            epoch_l2loss = 0.0
            precision_list = []
            recall_list = []
            iou_part_list = []
            iou_whole_list = []
            proj_error_list = []
            reproj_error_list = []
            test_progress_bar = tqdm(
                enumerate(test_loader), total=len(test_loader), leave=False)
            test_progress_bar.set_description(
                f'Epoch: {epoch + 1}/{opt.train_epochs}')

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
                    if not any(pred_cls_dict.values()):  # No any point after postprocessing
                        print(f'not keypts at {step + 1}')
                        plt.imsave(osp.join(exp_name_path, 'imgs', 'test_%05d_%05d_pred_not_keypts.png' % (
                            epoch + 1, step)), pred_heatmap, vmin=0, vmax=91)
                        continue

                    p, r, loss2 = calc_keypts_metrics(
                        gt_cls_dict, pred_cls_dict, opt.pr_thres)
                    if p == 0 and r == 0 and loss2 == 0:  # No common point appeared
                        print(f'diff location at {step + 1}')
                        plt.imsave(osp.join(exp_name_path, 'imgs', 'test_%05d_%05d_pred_diff_location.png' % (
                            epoch + 1, step)), pred_heatmap, vmin=0, vmax=91)
                        continue

                    precision_list.append(p)
                    recall_list.append(r)

                    batch_celoss += loss.detach()
                    epoch_celoss += loss.detach()
                    batch_l2loss += loss2.detach()
                    epoch_l2loss += loss2.detach()

                    # TODO: log loss
                    if step % 10 == 9:
                        batch_celoss /= 10
                        print('Epoch: {}/{}, Step: {}/{}\tTesting CE Loss: {:.4f}'.format(epoch +
                                                                                          1, opt.train_epochs, step, len(test_loader), batch_celoss))
                        batch_celoss = 0.0

                        batch_l2loss /= 10
                        print('Epoch: {}/{}, Step: {}/{}\tTesting MSE Loss: {:.4f}'.format(epoch +
                                                                                           1, opt.train_epochs, step, len(test_loader), batch_l2loss))
                        batch_l2loss = 0.0

                    image = utils.im_to_numpy(denorm(image[0]))

                    # TODO: show keypoints visual result after postprocessing
                    pred_keypoints = np.zeros_like(pred_heatmap)
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

                    # TODO: save pic
                    # if epoch % 100 == 99 and step % 10 == 9:  # WorldCup
                    if epoch % 100 == 99 and step % 100 == 99:  # TS-WorldCup
                        plt.imsave(
                            osp.join(test_visual_dir, 'test_%05d_%05d_rgb.jpg' % (epoch + 1, step)), image)
                        plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred.png' % (
                            epoch + 1, step)), pred_heatmap, vmin=0, vmax=91)
                        plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_gt.png' % (
                            epoch + 1, step)), gt_heatmap, vmin=0, vmax=91)

                        pred_keypoints = ss.expand_labels(
                            pred_keypoints, distance=5)
                        plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred_keypts.png' % (
                            epoch + 1, step)), pred_keypoints, vmin=0, vmax=91)

                    # if epoch % 100 == 99 and step % 10 == 9:  # WorldCup
                    if epoch % 100 == 99 and step % 100 == 99:  # TS-WorldCup
                        if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                            plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_gt_iou_part.png' % (
                                epoch + 1, step)), gt_part_mask)
                            plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_pred_iou_part.png' % (
                                epoch + 1, step)), pred_part_mask)
                            plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_merge_iou_part.png' % (
                                epoch + 1, step)), part_merge_result)
                            # plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_line_iou_whole.png' % (
                            #     epoch + 1, step)), whole_line_merge_result)
                            # plt.imsave(osp.join(test_visual_dir, 'test_%05d_%05d_fill_iou_whole.png' % (
                            #     epoch + 1, step)), whole_fill_merge_result)
                            # np.save(osp.join(test_visual_dir, 'test_%05d_%05d_gt_homography.npy' % (
                            #     epoch + 1, step)), gt_homo)
                            # np.save(osp.join(test_visual_dir, 'test_%05d_%05d_pred_homography.npy' % (
                            #     epoch + 1, step)), pred_homo)

                epoch_celoss /= len(test_loader)
                epoch_l2loss /= len(test_loader)
                print('Epoch: {}/{}\t\t\tTesting CE Loss: {:.4f}'.format(epoch +
                                                                         1, opt.train_epochs, epoch_celoss))
                writer.add_scalar('Loss/test', epoch_celoss, epoch + 1)
                print('Epoch: {}/{}\t\t\tTesting MSE Loss: {:.4f}'.format(epoch +
                                                                          1, opt.train_epochs, epoch_l2loss))
                writer.add_scalar('Loss/MSE in test', epoch_l2loss, epoch + 1)

                average_precision = np.array(precision_list).mean()
                average_recall = np.array(recall_list).mean()
                print(
                    f'Average Precision: {average_precision:.2f}, Recall: {average_recall:.2f}')
                writer.add_scalar(
                    'Metrics/average keypoints precision', average_precision, epoch + 1)
                writer.add_scalar(
                    'Metrics/average keypoints recall', average_recall, epoch + 1)

                iou_part_list = np.array(iou_part_list)
                iou_whole_list = np.array(iou_whole_list)
                mean_iou_part = np.nanmean(iou_part_list)
                mean_iou_whole = np.nanmean(iou_whole_list)
                print(
                    f'Mean IOU part: {mean_iou_part * 100.:.1f}, IOU whole: {mean_iou_whole * 100.:.1f}')
                writer.add_scalar('Metrics/mean IOU part',
                                  mean_iou_part * 100., epoch + 1)
                writer.add_scalar('Metrics/mean IOU whole',
                                  mean_iou_whole * 100., epoch + 1)

                median_iou_part = np.nanmedian(iou_part_list)
                median_iou_whole = np.nanmedian(iou_whole_list)
                print(
                    f'Median IOU part: {median_iou_part * 100.:.1f}, IOU whole: {median_iou_whole * 100.:.1f}')
                writer.add_scalar('Metrics/median IOU part',
                                  median_iou_part * 100., epoch + 1)
                writer.add_scalar('Metrics/median IOU whole',
                                  median_iou_whole * 100., epoch + 1)

                proj_error_list = np.array(proj_error_list)
                reproj_error_list = np.array(reproj_error_list)
                mean_proj_error = np.nanmean(proj_error_list)
                mean_reproj_error = np.nanmean(reproj_error_list)
                print(
                    f'Mean Projection Error: {mean_proj_error:.2f}, Reprojection Error: {mean_reproj_error:.3f}')
                writer.add_scalar('Metrics/mean Projection Error',
                                  mean_proj_error, epoch + 1)
                writer.add_scalar(
                    'Metrics/mean Reprojection Error', mean_reproj_error, epoch + 1)

                median_proj_error = np.nanmedian(proj_error_list)
                median_reproj_error = np.nanmedian(reproj_error_list)
                print(
                    f'Median Projection Error: {median_proj_error:.2f}, Reprojection Error: {median_reproj_error:.3f}')
                writer.add_scalar(
                    'Metrics/median Projection Error', median_proj_error, epoch + 1)
                writer.add_scalar(
                    'Metrics/median Reprojection Error', median_reproj_error, epoch + 1)

                with open(osp.join(exp_name_path, 'metrics_%03d.txt' % (epoch + 1)), 'w') as out_file:
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

                # TODO: save training model weight
                cnt_metrics = 0
                if mean_iou_part >= best_mean_iou_part:
                    cnt_metrics += 1
                if median_iou_part >= best_median_iou_part:
                    cnt_metrics += 1
                if mean_proj_error <= best_mean_proj:
                    cnt_metrics += 1
                if median_proj_error <= best_median_proj:
                    cnt_metrics += 1
                if mean_reproj_error <= best_mean_reproj:
                    cnt_metrics += 1
                if median_reproj_error <= best_median_reproj:
                    cnt_metrics += 1
                if cnt_metrics >= 4:
                    best_epoch = epoch + 1
                    best_mean_iou_part = mean_iou_part
                    best_median_iou_part = median_iou_part
                    best_mean_proj = mean_proj_error
                    best_median_proj = median_proj_error
                    best_mean_reproj = mean_reproj_error
                    best_median_reproj = median_reproj_error
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, osp.join(weight_save_dir, 'train_%05d_weights.pth' % (epoch + 1)))

                print('-'*10)
                print(f'The Best Epoch: {best_epoch}')
                print(
                    f'The Best Mean Iou Part: {best_mean_iou_part * 100.:.1f}')
                print(
                    f'The Best Median Iou Part: {best_median_iou_part * 100.:.1f}')
                print(
                    f'The Best Mean Projection Error: {best_mean_proj:.2f}')
                print(
                    f'The Best Median Projection Error: {best_median_proj:.2f}')
                print(
                    f'The Best Mean Reprojection Error: {best_mean_reproj:.3f}')
                print(
                    f'The Best Median Reprojection Error: {best_median_reproj:.3f}')
                print('-'*10)


def main():

    train()
    writer.flush()
    writer.close()


if __name__ == '__main__':

    start_time = time.time()
    main()
    print(f'Done...Take {(time.time() - start_time):.4f} (sec)')
