# from datasets import build_dataset
import os
import numpy as np
from tqdm import tqdm
from mmdet3d.datasets import SUNRGBDDataset
from mmdet3d.datasets.pipelines import data_augment_utils
import matplotlib.pyplot as plt

class Args():
    def __init__(self):
        self.dataset_name = 'sunrgbd'
        self.dataset_root_dir = './data/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1'
        self.use_color = False
        self.model_name = '3detr'
        self.enc_dim = 256
        self.enc_type = 'masked'

data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
    'night_stand', 'bookshelf', 'bathtub')

n_points = 100000

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                   (1333, 576), (1333, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# class_names = ('bed')

type2class = {
    "bed": 0,
    "table": 1,
    "sofa": 2,
    "chair": 3,
    "toilet": 4,
    "desk": 5,
    "dresser": 6,
    "night_stand": 7,
    "bookshelf": 8,
    "bathtub": 9,
}

class2type = {
    0: "bed",
    1: "table",
    2: "sofa",
    3: "chair",
    4: "toilet",
    5: "desk",
    6: "dresser",
    7: "night_stand",
    8: "bookshelf",
    9: "bathtub"
}

obj_folder = './data/SunRGBD/objects/'

import numpy as np
def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points
    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])
    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.
    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.
    """

    idx = np.where(
          (points[:, 0] > min_x) & 
          (points[:, 0] < max_x) & 
          (points[:, 1] > min_y) & 
          (points[:, 1] < max_y) &
          (points[:, 2] > min_z) & 
          (points[:, 2] < max_z) 
        )
    res = points[idx[0], :]
    return res

if __name__ == "__main__":
    dataset_args = Args()
    # datasets, dataset_config = build_dataset(dataset_args)
    train_dataset = SUNRGBDDataset(
        data_root,
        data_root + 'sunrgbd_infos_train.pkl',
        pipeline=train_pipeline,
        filter_empty_gt=False,
        classes=class_names,
        box_type_3d='Depth'
    )

    for cls in class2type:

        cur_cls = class2type[cls]
        print('Class', cur_cls)
        all_len = []
        for idx, inp in tqdm(enumerate(train_dataset)):
            # print(inp.keys())
            # print(inp['img_metas'].__dict__['_data']['gt_bboxes_3d_corners'])
            # print(inp['img_metas'][0]['gt_bboxes_3d_corners'])
            box_corners = inp['img_metas'].__dict__['_data']['gt_bboxes_3d_corners'].numpy()
            labels = inp['gt_labels_3d'].__dict__['_data'].numpy()
            points = inp['points'].__dict__['_data'].numpy()
            # print(points)
            for obj_idx, obj_id in enumerate(labels):
                obj_type = class2type[obj_id]
                if obj_type == cur_cls:
                    cur_bbox = box_corners[obj_idx]
                    min_x = np.min(cur_bbox[:,0])
                    max_x = np.max(cur_bbox[:,0])
                    min_y = np.min(cur_bbox[:,1])
                    max_y = np.max(cur_bbox[:,1])
                    min_z = np.min(cur_bbox[:,2])
                    max_z = np.max(cur_bbox[:,2])
                    new_pts = bounding_box(points, min_x, max_x, min_y, max_y, min_z, max_z)
                    # print(new_pts.shape)
                    if new_pts.shape[0] > 0:
                        all_len.append(new_pts.shape[0])
                # coll_mat = data_augment_utils.box_collision_test(np.array([cur_bbox]), np.array([cur_bbox]))
                # print(coll_mat)
                # break

        # frequency, bins = np.histogram(new_pts, bins=10, range=[0, 100])

        # # Pretty Print
        # for b, f in zip(bins[1:], frequency):
        #     print(round(b, 1), ' '.join(np.repeat('*', f)))

        plt.hist(all_len, bins=50)
        plt.gca().set(title='Class: '+cur_cls, ylabel='Frequency')
        print('30 perc: ', np.percentile(all_len, 30))
        print('60 perc: ', np.percentile(all_len, 60))
        plt.savefig('freq_'+cur_cls+'.png')
        plt.cla()
        plt.clf()
        plt.close()