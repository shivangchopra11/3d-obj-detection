from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import time
import pickle
import os
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
from datasets import build_dataset
from models import build_model
from datasets import build_dataset
from tqdm import tqdm

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Args():
    def __init__(self):
        self.dataset_name = 'sunrgbd'
        self.dataset_root_dir = '../data/SunRGBD/sunrgbd_pc_bbox_votes_50k_v1'
        self.use_color = False
        self.model_name = '3detr'
        self.enc_dim = 256
        self.preenc_npoints = 512
        self.enc_type = 'masked'
        self.enc_nhead = 4
        self.enc_ffn_dim = 128
        self.enc_dropout = 0.1
        self.enc_activation = 'relu'
        self.enc_nlayers = 3
        self.dec_dim = 256
        self.dec_nhead = 4
        self.dec_ffn_dim = 256
        self.dec_dropout = 0.1 
        self.dec_nlayers = 8
        self.mlp_dropout = 0.3
        self.nqueries = 128
        self.test_ckpt = './pretrained/sunrgbd_masked_ep1080.pth'

dataset_args = Args()

paral_num = 10
nimg_per_cat = 100
occ_level='ZERO'
occ_type=''
vc_num = 512
categories = {
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
imgs_par_cat =np.zeros(len(categories))

feat_folder = './features/'
obj_folder = '/home/shivangchopra11/MSR/data/SunRGBD/objects/'

bool_pytorch = True
datasets, dataset_config = build_dataset(dataset_args)
model, _ = build_model(dataset_args, dataset_config)
sd = torch.load(dataset_args.test_ckpt, map_location=torch.device("cpu"))
model.load_state_dict(sd['model'], strict=False)
model = model.cuda()

for category in categories:
    print('Category: ', category)
    # os.mkdir(feat_folder + category)
    cat_idx = categories[category]
    point_clouds = []
    N = len(os.listdir(obj_folder + category))
    class_feats = []
    class_points = []
    class_queries = []
    for ii,fi in enumerate(os.listdir(obj_folder + category)):
        try:
            pc = np.load(obj_folder + category + '/' + fi)
            point_clouds.append(pc)
        except:
            pass
    point_clouds = np.asarray(point_clouds)
    data_loader = DataLoader(
        point_clouds,
        sampler = torch.utils.data.SequentialSampler(point_clouds),
        batch_size=1,
        num_workers=1,
        worker_init_fn=my_worker_init_fn,
    )
    for idx, inp in tqdm(enumerate(data_loader)):
        inp_min, _ = inp.min(axis=1)
        inp_max, _ = inp.max(axis=1)
        inputs = {
            "point_clouds": inp.cuda(),
            "point_cloud_dims_min": inp_min.cuda(),
            "point_cloud_dims_max": inp_max.cuda(),
        }
        queries, features, points = model(inputs, final_feats=True)
        class_feats.append(features.detach().cpu().numpy())
        class_points.extend(points.detach().cpu().numpy())
        class_queries.extend(queries.detach().cpu().numpy())
        # print(features.shape)
        # exit()
    class_feats = np.array(class_feats)
    class_points = np.array(class_points)
    class_queries = np.array(class_queries)
    print(class_feats.shape, class_points.shape, class_queries.shape)

    np.save(feat_folder + category + '/features.npy', class_feats)
    np.save(feat_folder + category + '/points.npy', class_points)
    np.save(feat_folder + category + '/queries.npy', class_queries)
    