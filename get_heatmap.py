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

act_folder = './activations/'
aug_folder = '/home/shivangchopra11/MSR/data/SunRGBD/aug_train/'

bool_pytorch = True
datasets, dataset_config = build_dataset(dataset_args)
model, _ = build_model(dataset_args, dataset_config)
sd = torch.load(dataset_args.test_ckpt, map_location=torch.device("cpu"))
model.load_state_dict(sd['model'], strict=False)
model = model.cuda()

point_clouds = []
for fi in tqdm(os.listdir(aug_folder)[:10]):
    # print(fi)
    pc = np.load(aug_folder + fi)
    point_clouds.append(pc['arr_0'])
    # print(pc['arr_0'].shape)
    # for k in pc.files:
    #     print(k)

point_clouds = np.asarray(point_clouds, dtype=np.float32)
data_loader = DataLoader(
    point_clouds,
    sampler = torch.utils.data.SequentialSampler(point_clouds),
    batch_size=1,
    num_workers=1,
    worker_init_fn=my_worker_init_fn,
)

class_probs = []
class_points = []
class_queries = []
for idx, inp in tqdm(enumerate(data_loader)):
    inp_min, _ = inp.min(axis=1)
    inp_max, _ = inp.max(axis=1)
    inputs = {
        "point_clouds": inp.cuda(),
        "point_cloud_dims_min": inp_min.cuda(),
        "point_cloud_dims_max": inp_max.cuda(),
    }
    queries, features, points = model(inputs, final_feats=True)
    queries = queries[0,:,:]
    points = points[0,:,:]
    probs = features[-1,:,:]
    # print(points.shape, probs.shape)
    class_probs.append(probs.detach().cpu().numpy())
    class_points.append(points.detach().cpu().numpy())
    class_queries.append(queries.detach().cpu().numpy())
#     # print(features.shape)
class_probs = np.array(class_probs)
class_points = np.array(class_points)
class_queries = np.array(class_queries)
print(class_probs.shape, class_points.shape, class_queries.shape)

np.save(act_folder + '/activations.npy', class_probs)
np.save(act_folder + '/points.npy', class_points)
np.save(act_folder + '/queries.npy', class_queries)
    