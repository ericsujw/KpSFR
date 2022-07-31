from loss import BinaryDiceLoss
import utils
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from PIL import Image
import math
import time
import os.path as osp
from options import CustomOptions
from models.network import KpSFR
from worldcup_train_loader import StaticTransformDataset
from ts_worldcup_train_loader import CustomWorldCupDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import skimage.segmentation as ss


# Get input arguments
opt = CustomOptions(train=True)
opt = opt.parse()

# Log on tensorboard
writer = SummaryWriter('runs/' + opt.name)

# Setup GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
print('CUDA Visible Devices: %s' % opt.gpu_ids)
device = torch.device('cuda:0')
print('device: %s' % device)


def train():

    num_objects = opt.num_objects
    non_local = bool(opt.use_non_local)
    model_archi = opt.model_archi
    loss_mode = opt.loss_mode

    # Reproducibility
    utils.reseed(seed=14159265)

    # Initialize models
    model = KpSFR(model_archi=model_archi, num_objects=num_objects,
                  non_local=non_local).to(device)

    # Setup dataset
    if opt.train_stage == 0:
        # Load training data
        print('Loading public worldcup data in pre-training...')
        train_dataset = StaticTransformDataset(
            root=opt.public_worldcup_root,
            data_type=opt.trainset,
            mode='train',
            num_objects=num_objects,
            noise_trans=opt.noise_trans,
            noise_rotate=opt.noise_rotate
        )
        # train_dataset = Subset(train_dataset, list(range(0, 4)))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    elif opt.train_stage == 1:
        # Load training data
        print('Loading time sequence worldcup data in main training...')
        train_dataset = CustomWorldCupDataset(
            root=opt.custom_worldcup_root,
            data_type=opt.trainset,
            mode='train',
            num_objects=num_objects,
            noise_trans=opt.noise_trans,
            noise_rotate=opt.noise_rotate
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    total_epoch = opt.train_epochs

    # Loss function
    dice_criterion = BinaryDiceLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    class_weights = torch.ones(num_objects + 1, device=device) * 100
    class_weights[0] = 1
    wce_criterion = nn.CrossEntropyLoss(
        weight=class_weights)  # TODO: put class weight

    optimizer = optim.Adam(model.parameters(),
                           lr=opt.train_lr,
                           betas=(0.9, 0.999),
                           weight_decay=opt.weight_decay,
                           eps=1e-4
                           )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.step_size, gamma=0.1)

    scaler = GradScaler()

    # Set data path
    denorm = utils.UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    train_visual_dir = osp.join(exp_name_path, 'imgs', 'train_visual')
    if opt.resume == False:
        if osp.exists(train_visual_dir):
            print(f'Remove directory: {train_visual_dir}')
            shutil.rmtree(train_visual_dir)
    print(f'Create directory: {train_visual_dir}')
    os.makedirs(train_visual_dir, exist_ok=True)

    weight_save_dir = osp.join(exp_name_path, 'weights')
    os.makedirs(weight_save_dir, exist_ok=True)

    # TODO: Load pretrained model or resume training
    start_epoch = 0
    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print('Loading weights: ', load_weights_path)
        assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
        checkpoint = torch.load(load_weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if opt.resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            print('Start Epoch: ', start_epoch)

    # Training loop
    # with torch.autograd.detect_anomaly(): # check loss nan but not work if use amp
    for epoch in range(start_epoch, total_epoch):
        print("Training...")
        model.train()
        # model.eval()
        batch_loss = 0.0
        epoch_loss = 0.0
        train_progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False)

        for step, data in train_progress_bar:
            image = data['rgb'].to(device, non_blocking=True)  # b*t*c*h*w
            target_dilated_hm = data['target_dilated_hm'].to(
                device, non_blocking=True)  # b*objs*t*h*w
            cls_gt = data['cls_gt'].to(
                device, non_blocking=True).long()  # b*t*h*w
            # gt_homo = data['gt_homo']
            selector = data['selector'].to(
                device, non_blocking=True)  # b*t*k(objs)
            lookup = data['lookup'].to(device, non_blocking=True)  # b*t*k

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            # Runs the forward pass (model + loss) with autocasting
            with autocast():
                # Key features never change, only compute once
                assert torch.isnan(image).sum() == 0, print('image: ', image)
                kf32, kf16, kf8, kf4 = model(
                    'encode_key', image)
                assert torch.isnan(kf32).sum() == 0, print('kf32: ', kf32)
                assert torch.isnan(kf16).sum() == 0, print('kf16: ', kf16)
                assert torch.isnan(kf8).sum() == 0, print('kf8: ', kf8)
                assert torch.isnan(kf4).sum() == 0, print('kf4: ', kf4)

                assert num_objects == target_dilated_hm.shape[1], 'Number of objects are inconsistent'

                # TODO: random pick 4
                ref_hm = target_dilated_hm.clone()
                ref_v = []
                for idx in range(num_objects):
                    chunks = torch.split(
                        ref_hm, [1, num_objects - 1], dim=1)
                    mask = chunks[0]  # b*1*t*h*w
                    other_masks = chunks[1]  # b*(objs-1)*t*h*w

                    fg_mask = torch.zeros_like(mask)

                    # TODO: Check label in the previous heatmap appears in the current heatmap or not
                    for b in range(lookup.shape[0]):
                        if lookup[b, 0, idx] not in lookup[b, 1].tolist():  # non-overlap
                            pass
                            # print('set to zero map')
                        else:
                            fg_mask[b, 0, 0] = mask[b, 0, 0]

                    if torch.isnan(image[:, 0]).sum():
                        for f in range(image.shape[0]):
                            plt.imsave(osp.join(exp_name_path, 'train_%05d_%05d_ref_rgb_nan_%d.png' % (
                                epoch + 1, step, f)), utils.im_to_numpy(denorm(image[f, 0])))
                        assert False
                    if torch.isnan(fg_mask[:, :, 0]).sum():
                        for f in range(fg_mask.shape[0]):
                            plt.imsave(osp.join(exp_name_path, 'train_%05d_%05d_ref_mask_nan_%d.png' % (
                                epoch + 1, step, f)), utils.to_numpy(fg_mask[f, 0, 0]))
                        assert False
                    out_v = model(
                        'encode_value', image[:, 0], kf32[:, 0], fg_mask[:, :, 0], isFirst=True)
                    ref_v.append(out_v)
                    ref_hm = torch.cat([other_masks, mask], dim=1)

                ref_v = torch.stack(ref_v, dim=1)  # b*k*c*t*h*w
                assert torch.isnan(ref_v).sum() == 0, print('ref_v: ', ref_v)

                # Segment qframe 1(k32[:, :, 1]) with mframe 0(k32[:, :, 0:1])
                prev_x, prev_logits, prev_heatmap = model(
                    'segment', kf32[:, 1], kf16[:, 1], kf8[:, 1], kf4[:, 1], num_objects, lookup[:, 1], selector[:, 1])

                assert torch.isnan(prev_x).sum(
                ) == 0, print('prev_x: ', prev_x)
                assert torch.isnan(prev_logits).sum() == 0, print(
                    'prev_logits: ', prev_logits)
                assert torch.isnan(prev_heatmap).sum() == 0, print(
                    'prev_heatmap: ', prev_heatmap)

                # TODO: random pick 4
                prev_hm = prev_heatmap.clone().detach()
                prev_v = []
                for idx in range(num_objects):
                    chunks = torch.split(
                        prev_hm, [1, num_objects - 1], dim=1)
                    mask = chunks[0]  # b*1*h*w
                    other_masks = chunks[1]  # b*(objs-1)*h*w

                    fg_mask = torch.zeros_like(mask)

                    # TODO: Check label in the previous heatmap appears in the current heatmap or not
                    for b in range(lookup.shape[0]):
                        if lookup[b, 1, idx] not in lookup[b, 2].tolist():  # non-overlap
                            pass
                            # print('set to zero map')
                        else:
                            fg_mask[b, 0] = mask[b, 0]
                    if torch.isnan(image[:, 1]).sum():
                        for f in range(image.shape[0]):
                            plt.imsave(osp.join(exp_name_path, 'train_%05d_%05d_prev_rgb_nan_%d.png' % (
                                epoch + 1, step, f)), utils.im_to_numpy(denorm(image[f, 1])))
                        assert False
                    if torch.isnan(fg_mask).sum():
                        for f in range(fg_mask.shape[0]):
                            plt.imsave(osp.join(exp_name_path, 'train_%05d_%05d_prev_mask_nan_%d.png' % (
                                epoch + 1, step, f)), utils.to_numpy(fg_mask[f, 0]))
                        assert False
                    out_v = model(
                        'encode_value', image[:, 1], kf32[:, 1], fg_mask, isFirst=False)
                    prev_v.append(out_v)
                    prev_hm = torch.cat([other_masks, mask], dim=1)

                prev_v = torch.stack(prev_v, dim=1)  # b*k*c*t*h*w
                assert torch.isnan(prev_v).sum(
                ) == 0, print('prev_v: ', prev_v)

                del ref_v

                # Segment qframe 2(k32[:, :, 2]) with mframe 0 and 1(k32[:, :, 0:2])
                this_x, this_logits, this_heatmap = model(
                    'segment', kf32[:, 2], kf16[:, 2], kf8[:, 2], kf4[:, 2], num_objects, lookup[:, 2], selector[:, 2])

                assert torch.isnan(this_x).sum(
                ) == 0, print('this_x: ', this_x)
                assert torch.isnan(this_logits).sum() == 0, print(
                    'this_logits: ', this_logits)
                assert torch.isnan(this_heatmap).sum() == 0, print(
                    'this_heatmap: ', this_heatmap)

                total_loss = 0.0
                b = target_dilated_hm.shape[0]
                size = target_dilated_hm.shape[-2:]

                prev_x = F.interpolate(
                    prev_x, size, mode='bilinear', align_corners=False)
                prev_logits = F.interpolate(
                    prev_logits, size, mode='bilinear', align_corners=False)
                prev_heatmap = F.interpolate(
                    prev_heatmap, size, mode='bilinear', align_corners=False)
                assert torch.isnan(prev_x).sum(
                ) == 0, print('prev_x: ', prev_x)
                assert torch.isnan(prev_logits).sum() == 0, print(
                    'prev_logits: ', prev_logits)
                assert torch.isnan(prev_heatmap).sum() == 0, print(
                    'prev_heatmap: ', prev_heatmap)

                this_x = F.interpolate(
                    this_x, size, mode='bilinear', align_corners=False)
                this_logits = F.interpolate(
                    this_logits, size, mode='bilinear', align_corners=False)
                this_heatmap = F.interpolate(
                    this_heatmap, size, mode='bilinear', align_corners=False)
                assert torch.isnan(this_x).sum(
                ) == 0, print('this_x: ', this_x)
                assert torch.isnan(this_logits).sum() == 0, print(
                    'this_logits: ', this_logits)
                assert torch.isnan(this_heatmap).sum() == 0, print(
                    'this_heatmap: ', this_heatmap)

                assert prev_x.shape[-2:] == target_dilated_hm.shape[-2:], 'shape inconsistent'
                assert prev_logits.shape[-2:] == cls_gt.shape[-2:], 'shape inconsistent'
                assert prev_heatmap.shape[-2:] == target_dilated_hm.shape[-2:], 'shape inconsistent'
                assert this_x.shape[-2:] == target_dilated_hm.shape[-2:], 'shape inconsistent'
                assert this_logits.shape[-2:] == cls_gt.shape[-2:], 'shape inconsistent'
                assert this_heatmap.shape[-2:] == target_dilated_hm.shape[-2:], 'shape inconsistent'

                for i in range(1, num_objects + 1):
                    for j in range(b):
                        loss_1 = 0.0
                        loss_2 = 0.0
                        if selector[j, 1, i-1] != 0:
                            if loss_mode == 'all':
                                loss_1 = dice_criterion(prev_heatmap[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 1]) + \
                                    bce_criterion(prev_x[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 1]) + \
                                    wce_criterion(
                                        prev_logits[j:j+1], cls_gt[j:j+1, 1])

                            elif loss_mode == 'dice_bce':
                                # TODO: Ablation: dice and bce loss
                                loss_1 = dice_criterion(prev_heatmap[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 1]) + \
                                    bce_criterion(
                                        prev_x[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 1])

                            elif loss_mode == 'dice_wce':
                                # TODO: Ablation: dice and wce loss
                                loss_1 = dice_criterion(prev_heatmap[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 1]) + \
                                    wce_criterion(
                                        prev_logits[j:j+1], cls_gt[j:j+1, 1])

                        if selector[j, 2, i-1] != 0:
                            if loss_mode == 'all':
                                loss_2 = dice_criterion(this_heatmap[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 2]) + \
                                    bce_criterion(this_x[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 2]) + \
                                    wce_criterion(
                                        this_logits[j:j+1], cls_gt[j:j+1, 2])

                            elif loss_mode == 'dice_bce':
                                # TODO: Ablation: dice and bce loss
                                loss_2 = dice_criterion(this_heatmap[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 2]) + \
                                    bce_criterion(
                                        this_x[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 2])

                            elif loss_mode == 'dice_wce':
                                # TODO: Ablation: dice and wce loss
                                loss_2 = dice_criterion(this_heatmap[j:j+1, i-1:i], target_dilated_hm[j:j+1, i-1:i, 2]) + \
                                    wce_criterion(
                                        this_logits[j:j+1], cls_gt[j:j+1, 2])

                        total_loss += loss_1 + loss_2
                total_loss = total_loss / (num_objects * 2.) / b / 4

            assert torch.isnan(total_loss).sum(
            ) == 0, print('Loss before backward call: ', total_loss)

            # if use amp
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            total_loss.register_hook(lambda grad: print(
                'Gradient after backward call:', grad))
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            # if not use amp
            # # Accumulates the parameters gradients
            # total_loss.backward()
            # optimizer.step()

            train_progress_bar.set_description(
                f'Epoch: {epoch + 1}/{total_epoch}')
            train_progress_bar.set_postfix(loss=total_loss.detach())

            batch_loss += total_loss.detach()
            epoch_loss += total_loss.detach()

            # TODO: log loss
            if step % 5 == 4:  # every 5 mini-batches
                batch_loss /= 5
                print('Epoch: {}/{}, Step: {}/{}\tTraining Loss: {:.4f}'.format(epoch +
                                                                                1, total_epoch, step, len(train_loader), batch_loss))
                # writer.add_scalar('Loss/train', batch_loss,
                #                   epoch * len(train_loader) + step)
                batch_loss = 0.0

            # TODO: save pic
            if epoch % 100 == 99 and step % 10 == 9:  # pretrain
                # if epoch % 100 == 99 and step % 3 == 2:  # maintrain
                # if epoch % 10 == 9 and step % 10 == 9:  # maintrain for worldcup settings
                frame0 = utils.im_to_numpy(denorm(image[0, 0]))
                frame1 = utils.im_to_numpy(denorm(image[0, 1]))
                frame2 = utils.im_to_numpy(denorm(image[0, 2]))
                plt.imsave(osp.join(
                    train_visual_dir, 'train_%05d_%05d_frame0.jpg' % (epoch + 1, step)), frame0)
                plt.imsave(osp.join(
                    train_visual_dir, 'train_%05d_%05d_frame1.jpg' % (epoch + 1, step)), frame1)
                plt.imsave(osp.join(
                    train_visual_dir, 'train_%05d_%05d_frame2.jpg' % (epoch + 1, step)), frame2)

                gt_ref_heatmap = cls_gt[0, 0].cpu().numpy()
                gt_prev_heatmap = cls_gt[0, 1].cpu().numpy()
                gt_this_heatmap = cls_gt[0, 2].cpu().numpy()
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_gt_seg0.png' % (
                    epoch + 1, step)), gt_ref_heatmap, vmin=0, vmax=num_objects)
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_gt_seg1.png' % (
                    epoch + 1, step)), gt_prev_heatmap, vmin=0, vmax=num_objects)
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_gt_seg2.png' % (
                    epoch + 1, step)), gt_this_heatmap, vmin=0, vmax=num_objects)

                pred_prev_heatmap = torch.argmax(prev_logits, dim=1)[
                    0].detach().cpu().numpy()
                pred_this_heatmap = torch.argmax(this_logits, dim=1)[
                    0].detach().cpu().numpy()
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_pred_seg1.png' % (
                    epoch + 1, step)), pred_prev_heatmap, vmin=0, vmax=num_objects)
                plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_pred_seg2.png' % (
                    epoch + 1, step)), pred_this_heatmap, vmin=0, vmax=num_objects)

                # Visualize for each channel of class
                for idx in range(num_objects):
                    pred_prev_heatmap = prev_heatmap[0, idx].detach(
                    ).cpu().numpy()
                    pred_this_heatmap = this_heatmap[0, idx].detach(
                    ).cpu().numpy()
                    plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_pred_seg1_%d.png' % (
                        epoch + 1, step, idx + 1)), pred_prev_heatmap)
                    plt.imsave(osp.join(train_visual_dir, 'train_%05d_%05d_pred_seg2_%d.png' % (
                        epoch + 1, step, idx + 1)), pred_this_heatmap)

            del image
            del target_dilated_hm
            del cls_gt
            del selector
            del lookup

        epoch_loss /= len(train_loader)
        print('Epoch: {}/{}\t\t\tTraining Loss: {:.4f}'.format(epoch +
                                                               1, total_epoch, epoch_loss))

        if len(opt.ckpt_path) > 0:
            if opt.resume:
                if opt.train_stage == 0:
                    # stage 0
                    writer.add_scalar('Loss/pre-train', epoch_loss, epoch + 1)
                elif opt.train_stage == 1:
                    # stage 1
                    writer.add_scalar('Loss/main-train', epoch_loss, epoch + 1)
            else:  # Fine-tuning
                writer.add_scalar('Loss/stage01', epoch_loss, epoch + 1)
        else:
            if opt.train_stage == 0:
                # stage 0
                writer.add_scalar('Loss/pre-train', epoch_loss, epoch + 1)
            elif opt.train_stage == 1:
                # stage 1
                writer.add_scalar('Loss/main-train', epoch_loss, epoch + 1)

        scheduler.step()

        if (epoch + 1) >= 1450 or epoch % 100 == 99:  # pretrain
            # if (epoch + 1) >= 2450 or epoch % 100 == 99:  # maintrain
            # maintrain for fine-tuned
            # if (epoch + 1) >= 85 or ((epoch + 1) >= 30 and (epoch + 1) <= 50) or epoch % 10 == 9:
            # if ((epoch + 1) >= 100 and (epoch + 1) < 300 and epoch % 5 == 4) or (epoch + 1) >= 1450 or epoch % 100 == 99:  # pretrain for loss nan
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, osp.join(weight_save_dir, 'train_%05d_weights.pth' % (epoch + 1)))

            print('Checkpoint saved to %s.' % weight_save_dir)


def main():

    train()
    writer.flush()
    writer.close()


if __name__ == '__main__':

    start_time = time.time()
    main()
    print(f'Done...Take {(time.time() - start_time):.4f} (sec)')
