import pickle
from mmdet3d.core.bbox import DepthInstance3DBoxes, get_box_type
import numpy as np
from tqdm import tqdm

full_val_dataset_path = 'data/sunrgbd/sunrgbd_infos_val.pkl'
low_occlusion_val_dataset_path = 'data/sunrgbd/sunrgbd_infos_val_low.pkl'
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
occlusion_levels = {
    'bed': {
        'low': 14241,
        'high': 33732
    },
    'table': {
        'low': 2807,
        'high': 9895
    },
    'sofa': {
        'low': 5856,
        'high': 17830
    },
    'chair': {
        'low': 840,
        'high': 3339
    },
    'toilet': {
        'low': 3413,
        'high': 9431
    },
    'desk': {
        'low': 3093,
        'high': 11800
    },
    'dresser': {
        'low': 1629,
        'high': 6491
    },
    'night_stand': {
        'low': 467,
        'high': 2266
    },
    'bookshelf': {
        'low': 2793,
        'high': 12445
    },
    'bathtub': {
        'low': 3946,
        'high': 11572
    }
}

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


if __name__ == '__main__':
    with open(full_val_dataset_path, 'rb') as f:
        val_data = pickle.load(f)
    # print(len(val_data))
    # print(val_data[0])
    # print(val_data[0]['annos'])
    cur_cls = 'bed'
    cur_occlusion = 'low'
    new_scenes = []
    for scene in tqdm(val_data):
        # print(scene)
        pts_path = scene['pts_path']
        base_path = '/workdir/tr3d/data/sunrgbd/'
        pts_filename = base_path + pts_path
        points = np.fromfile(pts_filename, dtype=np.float32)
        points = points.reshape(-1, 6)
        points = points[:,:3]
        # print(points.shape)
        # exit()
        annos = scene['annos']
        if annos['gt_num'] == 0:
            continue
        # print(annos)
        bboxes = annos['gt_boxes_upright_depth'].astype(
                np.float32) 
        _, box_type = get_box_type('Depth')
        gt_bboxes_3d = DepthInstance3DBoxes(
                bboxes, origin=(0.5, 0.5, 0.5)).convert_to(box_type)
        bboxes_corners = gt_bboxes_3d.corners
        new_names = []
        new_bbox = []
        new_calib = []
        new_rt = []
        new_location = []
        new_dimension = []
        new_rotation = []
        new_depth = []
        new_class = []
        ctr = 0
        for obj_idx, obj_type in enumerate(scene['annos']['name']):
            if obj_type == cur_cls:
                cur_bbox = bboxes_corners[obj_idx].numpy()
                # print(cur_bbox.shape)
                min_x = np.min(cur_bbox[:,0])
                max_x = np.max(cur_bbox[:,0])
                min_y = np.min(cur_bbox[:,1])
                max_y = np.max(cur_bbox[:,1])
                min_z = np.min(cur_bbox[:,2])
                max_z = np.max(cur_bbox[:,2])
                new_pts = bounding_box(points, min_x, max_x, min_y, max_y, min_z, max_z)
                if cur_occlusion == 'low' and new_pts.shape[0] <= occlusion_levels[cur_cls]['low']:
                    ctr += 1
                    # new_calib.append(scene['calib']['K'][obj_idx])
                    # new_rt.append(scene['calib']['Rt'][obj_idx])
                    new_names.append(scene['annos']['name'][obj_idx])
                    new_bbox.append(scene['annos']['bbox'][obj_idx])
                    new_location.append(scene['annos']['location'][obj_idx])
                    new_dimension.append(scene['annos']['dimensions'][obj_idx])
                    new_rotation.append(scene['annos']['rotation_y'][obj_idx])
                    new_class.append(scene['annos']['class'][obj_idx])
                    new_depth.append(scene['annos']['gt_boxes_upright_depth'][obj_idx])
        new_index = np.arange(ctr)
        # scene['calib']['K'] = new_calib
        # scene['calib']['Rt'] = new_rt
        scene['annos']['name'] = new_names
        scene['annos']['bbox'] = new_bbox
        scene['annos']['location'] = new_location
        scene['annos']['dimensions'] = new_dimension
        scene['annos']['rotation_y'] = new_rotation
        scene['annos']['class'] = new_class
        scene['annos']['gt_boxes_upright_depth'] = new_depth
        scene['annos']['index'] = new_index
        new_scenes.append(scene)
        # break
    with open(low_occlusion_val_dataset_path, 'wb') as f:
        pickle.dump(new_scenes, f)
        