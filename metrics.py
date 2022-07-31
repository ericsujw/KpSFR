import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from shapely.geometry import Point, Polygon, MultiPoint
import utils
import torch
import os.path as osp


def calc_euclidean_distance(a, b, _norm=np.linalg.norm, axis=None):
    return _norm(a - b, axis=axis)


def calc_iou_part(pred_h, gt_h, frame, template, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    # TODO: calculate iou part
    # === render ===
    render_w, render_h = template.size  # (1050, 680)
    dst = np.array(template)

    # Create three channels (680, 1050, 3)
    dst = np.stack((dst, ) * 3, axis=-1)

    scaling_mat = np.eye(3)
    scaling_mat[0, 0] = render_w / template_w
    scaling_mat[1, 1] = render_h / template_h

    frame = np.uint8(frame * 255)  # 0-1 map to 0-255
    gt_mask_render = cv2.warpPerspective(
        frame, scaling_mat @ gt_h, (render_w, render_h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    pred_mask_render = cv2.warpPerspective(
        frame, scaling_mat @ pred_h, (render_w, render_h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    # === blending ===
    dstf = dst.astype(float) / 255
    gt_mask_renderf = gt_mask_render.astype(float) / 255
    gt_resultf = cv2.addWeighted(dstf, 0.3, gt_mask_renderf, 0.7, 0.0)
    gt_result = np.uint8(gt_resultf * 255)
    pred_mask_renderf = pred_mask_render.astype(float) / 255
    pred_resultf = cv2.addWeighted(dstf, 0.3, pred_mask_renderf, 0.7, 0.0)
    pred_result = np.uint8(pred_resultf * 255)

    # field template binary mask
    field_mask = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    gt_mask = cv2.warpPerspective(field_mask, gt_h, (template_w, template_h),
                                  cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    pred_mask = cv2.warpPerspective(field_mask, pred_h, (template_w, template_h),
                                    cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    gt_mask[gt_mask > 0] = 255
    pred_mask[pred_mask > 0] = 255

    intersection = ((gt_mask > 0) * (pred_mask > 0)).sum()
    union = (gt_mask > 0).sum() + (pred_mask > 0).sum() - intersection

    if union <= 0:
        print('part union', union)
        # iou = float('nan')
        iou = 0.
    else:
        iou = float(intersection) / float(union)

    # === blending ===
    gt_white_area = (gt_mask[:, :, 0] == 255) & (
        gt_mask[:, :, 1] == 255) & (gt_mask[:, :, 2] == 255)
    gt_fill = gt_mask.copy()
    gt_fill[gt_white_area, 0] = 255
    gt_fill[gt_white_area, 1] = 0
    gt_fill[gt_white_area, 2] = 0
    pred_white_area = (pred_mask[:, :, 0] == 255) & (
        pred_mask[:, :, 1] == 255) & (pred_mask[:, :, 2] == 255)
    pred_fill = pred_mask.copy()
    pred_fill[pred_white_area, 0] = 0
    pred_fill[pred_white_area, 1] = 255
    pred_fill[pred_white_area, 2] = 0
    gt_maskf = gt_fill.astype(float) / 255
    pred_maskf = pred_fill.astype(float) / 255
    fill_resultf = cv2.addWeighted(gt_maskf, 0.5,
                                   pred_maskf, 0.5, 0.0)
    fill_result = np.uint8(fill_resultf * 255)

    return iou, gt_result, pred_result, fill_result


def calc_iou_whole_with_poly(pred_h, gt_h, frame, template, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    corners = np.array([[0, 0],
                        [frame_w - 1, 0],
                        [frame_w - 1, frame_h - 1],
                        [0, frame_h - 1]], dtype=np.float64)

    mapping_mat = np.linalg.inv(gt_h)
    mapping_mat /= mapping_mat[2, 2]

    gt_corners = cv2.perspectiveTransform(
        corners[:, None, :], gt_h)  # inv_gt_mat * (gt_mat * [x, y, 1])
    gt_corners = cv2.perspectiveTransform(
        gt_corners, np.linalg.inv(gt_h))
    gt_corners = gt_corners[:, 0, :]

    pred_corners = cv2.perspectiveTransform(
        corners[:, None, :], gt_h)  # inv_pred_mat * (gt_mat * [x, y, 1])
    pred_corners = cv2.perspectiveTransform(
        pred_corners, np.linalg.inv(pred_h))
    pred_corners = pred_corners[:, 0, :]

    gt_poly = Polygon(gt_corners.tolist())
    pred_poly = Polygon(pred_corners.tolist())

    # f, axarr = plt.subplots(1, 2, figsize=(16, 12))
    # axarr[0].plot(*gt_poly.exterior.coords.xy)
    # axarr[1].plot(*pred_poly.exterior.coords.xy)
    # plt.show()

    if pred_poly.is_valid is False:
        return 0., None, None

    if not gt_poly.intersects(pred_poly):
        print('not intersects')
        iou = 0.
    else:
        intersection = gt_poly.intersection(pred_poly).area
        union = gt_poly.area + pred_poly.area - intersection
        if union <= 0.:
            print('whole union', union)
            iou = 0.
        else:
            iou = intersection / union

    return iou, None, None


def calc_proj_error(pred_h, gt_h, frame, template, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    # TODO get visible field area of the camera image
    dst = np.array(template)

    # Create three channels (680, 1050, 3)
    dst = np.stack((dst, ) * 3, axis=-1)

    field_mask = np.ones((template_h, template_w, 3), dtype=np.uint8) * 255
    gt_mask = cv2.warpPerspective(field_mask, np.linalg.inv(
        gt_h), (frame_w, frame_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    gt_gray = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    _, contours, hierarchy = cv2.findContours(
        gt_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.squeeze(contours[0])
    poly = Polygon(contour)
    sample_pts = []
    num_pts = 2500
    while len(sample_pts) <= num_pts:
        x = random.sample(range(0, frame_w), 1)
        y = random.sample(range(0, frame_h), 1)
        p = Point(x[0], y[0])
        if p.within(poly):
            sample_pts.append([x[0], y[0]])
    sample_pts = np.array(sample_pts, dtype=np.float32)

    field_dim_x, field_dim_y = 100, 60
    x_scale = field_dim_x / template_w
    y_scale = field_dim_y / template_h
    scaling_mat = np.eye(3)
    scaling_mat[0, 0] = x_scale
    scaling_mat[1, 1] = y_scale
    gt_temp_grid = cv2.perspectiveTransform(
        sample_pts.reshape(-1, 1, 2), scaling_mat @ gt_h)
    gt_temp_grid = gt_temp_grid.reshape(-1, 2)
    pred_temp_grid = cv2.perspectiveTransform(
        sample_pts.reshape(-1, 1, 2), scaling_mat @ pred_h)
    pred_temp_grid = pred_temp_grid.reshape(-1, 2)

    # TODO compute distance in top view
    gt_grid_list = []
    pred_grid_list = []
    for gt_pts, pred_pts in zip(gt_temp_grid, pred_temp_grid):
        if 0 <= gt_pts[0] < field_dim_x and 0 <= gt_pts[1] < field_dim_y and \
                0 <= pred_pts[0] < field_dim_x and 0 <= pred_pts[1] < field_dim_y:
            gt_grid_list.append(gt_pts)
            pred_grid_list.append(pred_pts)
    gt_grid_list = np.array(gt_grid_list)
    pred_grid_list = np.array(pred_grid_list)

    if gt_grid_list.shape != pred_grid_list.shape:
        print('proj error:', gt_grid_list.shape, pred_grid_list.shape)
    assert gt_grid_list.shape == pred_grid_list.shape, 'shape mismatch'

    if gt_grid_list.size != 0 and pred_grid_list.size != 0:
        distance_list = calc_euclidean_distance(
            gt_grid_list, pred_grid_list, axis=1)
        return distance_list.mean()  # average all keypoints
    else:
        print(gt_grid_list)
        print(pred_grid_list)
        return float('nan')


def calc_reproj_error(pred_h, gt_h, frame, template, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    uniform_grid = utils.gen_template_grid()  # grid shape (91, 3), (x, y, label)
    template_grid = uniform_grid[:, :2].copy()
    template_grid = template_grid.reshape(-1, 1, 2)

    gt_warp_grid = cv2.perspectiveTransform(template_grid, np.linalg.inv(gt_h))
    gt_warp_grid = gt_warp_grid.reshape(-1, 2)
    pred_warp_grid = cv2.perspectiveTransform(
        template_grid, np.linalg.inv(pred_h))
    pred_warp_grid = pred_warp_grid.reshape(-1, 2)

    # TODO compute distance in camera view
    gt_grid_list = []
    pred_grid_list = []
    for gt_pts, pred_pts in zip(gt_warp_grid, pred_warp_grid):
        if 0 <= gt_pts[0] < frame_w and 0 <= gt_pts[1] < frame_h and \
                0 <= pred_pts[0] < frame_w and 0 <= pred_pts[1] < frame_h:
            gt_grid_list.append(gt_pts)
            pred_grid_list.append(pred_pts)
    gt_grid_list = np.array(gt_grid_list)
    pred_grid_list = np.array(pred_grid_list)

    if gt_grid_list.shape != pred_grid_list.shape:
        print('reproj error:', gt_grid_list.shape, pred_grid_list.shape)
    assert gt_grid_list.shape == pred_grid_list.shape, 'shape mismatch'

    if gt_grid_list.size != 0 and pred_grid_list.size != 0:
        distance_list = calc_euclidean_distance(
            gt_grid_list, pred_grid_list, axis=1)
        distance_list /= frame_h  # normalize by image height
        return distance_list.mean()  # average all keypoints
    else:
        print(gt_grid_list)
        print(pred_grid_list)
        return float('nan')


if __name__ == "__main__":

    # image = np.array(Image.open(
    #     osp.join('./dataset/soccer_worldcup_2014/soccer_data/test', '1.jpg')))
    image = np.array(Image.open(
        osp.join('./assets', 'IMG_001.jpg')))
    # print(image.shape)
    # gt_homo = np.loadtxt(
    #     osp.join('./dataset/soccer_worldcup_2014/soccer_data/test', '1.homographyMatrix'))
    gt_homo = np.load(
        osp.join('./assets', 'IMG_001_homography.npy'))
    pred_homo = np.load(
        osp.join('./assets', 'IMG_002_homography.npy'))
    # gt_homo = np.load(
    #     osp.join('./nms/debug/test_visual', 'test_00010_gt_homography.npy'))
    # pred_homo = np.load(
    #     osp.join('./nms/debug/test_visual', 'test_00010_pred_homography.npy'))
    field_model = Image.open(
        osp.join('./assets', 'worldcup_field_model.png'))
    m = np.array([[0.02, 0, 0],
                  [0, 0.02, 0],
                  [0, 0, 0]], dtype=np.float32)
    print(gt_homo)
    # pred_homo = gt_homo
    # pred_homo[0, 0] *= -1
    # pred_homo = gt_homo - m
    # pred_homo = np.eye(3, dtype=np.float32)
    print(pred_homo)

    iou_part, gt_part_mask, pred_part_mask, part_merge_result = calc_iou_part(
        pred_homo, gt_homo, image, field_model)
    print(f'{iou_part * 100.:.1f}')
    print(iou_part)

    # iou_whole, whole_line_merge_result, whole_fill_merge_result = calc_iou_whole(
    #     pred_homo, gt_homo, image, field_model)
    # print(f'{iou_whole * 100.:.1f}')
    # print(iou_whole)
