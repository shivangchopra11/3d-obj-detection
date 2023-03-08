from datasets import build_dataset
import os
import numpy as np
from tqdm import tqdm

class Args():
    def __init__(self):
        self.dataset_name = 'sunrgbd'
        self.dataset_root_dir = '../data/SunRGBD/sunrgbd_pc_bbox_votes_50k_v1'
        self.use_color = False
        self.model_name = '3detr'
        self.enc_dim = 256
        self.enc_type = 'masked'

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

obj_folder = '/home/shivangchopra11/MSR/data/SunRGBD/objects/'

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
    datasets, dataset_config = build_dataset(dataset_args)

    for k in range(10):
        os.mkdir(obj_folder + class2type[k])

    

    train_dataset = datasets['train']
    # ctr = 0
    my_box = []
    counters = np.zeros(10)

    print('Saving Object Point Clouds from Trainihng Dataset...')

    for idx, inp in tqdm(enumerate(train_dataset)):
        for obj_idx in range(len(inp['gt_box_sem_cls_label'])):
            obj_id = inp['gt_box_sem_cls_label'][obj_idx]
            if obj_idx == 0:
                my_box = inp['gt_box_corners'][obj_idx]
            obj_type = class2type[obj_id]
            box_size = inp['gt_box_sizes'][obj_idx]
            obj_box = inp['gt_box_corners'][obj_idx]
            obj_box_center = inp['gt_box_centers'][obj_idx]
            points = inp['point_clouds']
            min_x = obj_box_center[0] - box_size[0]
            max_x = obj_box_center[0] + box_size[0]
            min_y = obj_box_center[1] - box_size[1]
            max_y = obj_box_center[1] + box_size[1]
            min_z = obj_box_center[2] - box_size[2]
            max_z = obj_box_center[2] + box_size[2]
            new_pts = bounding_box(points, min_x, max_x, min_y, max_y, min_z, max_z)
            print(new_pts.shape)
            if box_size[0] > 0.0 and box_size[1] > 0.0 and box_size[2] > 0.0 and new_pts.shape[0]>0:
                if counters[obj_id] <= 200:
                    save_path = obj_folder + obj_type + '/' + str(int(counters[obj_id])) + '.npy'
                    counters[obj_id]+=1
                    np.save(save_path, new_pts)
