import numpy as np
import scipy
if tuple(map(int, scipy.__version__.split('.'))) < (1, 0, 0):
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp
import time
from tqdm import tqdm
import pickle


def normalize_features(features):
	'''features: n by d matrix'''
	assert(len(features.shape)==2)
	norma=np.sqrt(np.sum(features ** 2, axis=1).reshape(-1, 1))+1e-6
	return features/norma

class vMFMM:
	def __init__(self, cls_num, init_method = 'random'):
		self.cls_num = cls_num
		self.init_method = init_method


	def fit(self, features, kappa, max_it=300, tol = 5e-5, normalized=False, verbose=True):
		self.features = features
		if not normalized:
			self.features = normalize_features(features)
		print('Normalised')
		self.n, self.d = self.features.shape
		self.kappa = kappa

		self.pi = np.random.random(self.cls_num)
		self.pi /= np.sum(self.pi)
		if self.init_method =='random':
			self.mu = np.random.random((self.cls_num, self.d))
			self.mu = normalize_features(self.mu)
			print('Mu initialised')
		elif self.init_method =='k++':
			centers = []
			centers_i = []

			if self.n > 15000:
				rdn_index = np.random.choice(self.n, size=(15000,), replace=False)
			else:
				rdn_index = np.array(range(self.n), dtype=int)
			print(self.features[rdn_index].shape)
			cos_dis = 1-np.dot(self.features[rdn_index], self.features[rdn_index].T)
			# cos_dis = 1
			print('Cos Dist Calculated')
			centers_i.append(np.random.choice(rdn_index))
			centers.append(self.features[centers_i[0]])
			for i in range(self.cls_num-1):

				cdisidx = [np.where(rdn_index==cci)[0][0] for cci in centers_i]
				prob = np.min(cos_dis[:,cdisidx], axis=1)**2
				prob /= np.sum(prob)
				centers_i.append(np.random.choice(rdn_index, p=prob))
				centers.append(self.features[centers_i[-1]])

			self.mu = np.array(centers)
			del(cos_dis)

		self.mllk_rec = []
		for itt in tqdm(range(max_it)):
			_st = time.time()
			self.e_step()
			# print('E done')
			self.m_step()
			# print('M done')
			_et = time.time()

			self.mllk_rec.append(self.mllk)
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				#print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break


	def fit_soft(self, features, p, mu, pi, kappa, max_it=300, tol = 1e-6, normalized=False, verbose=True):
		self.features = features
		if not normalized:
			self.features = normalize_features(features)

		self.p = p
		self.mu = mu
		self.pi = pi
		self.kappa = kappa

		self.n, self.d = self.features.shape

		for itt in range(max_it):
			self.e_step()
			self.m_step()

			self.mllk_rec.append(self.mllk)
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				#print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break


	def e_step(self):
		# update p
		logP = np.dot(self.features, self.mu.T)*self.kappa + np.log(self.pi).reshape(1,-1)  # n by k
		logP_norm = logP - logsumexp(logP, axis=1).reshape(-1,1)
		self.p = np.exp(logP_norm)
		self.mllk = np.mean(logsumexp(logP, axis=1))


	def m_step(self):
		# update pi and mu
		self.pi = np.sum(self.p, axis=0)/self.n

		# fast version, requires more memory
		self.mu = np.dot(self.p.T, self.features)/np.sum(self.p, axis=0).reshape(-1,1)

		self.mu = normalize_features(self.mu)



vc_num = 10
vMF_kappa = 30
feat_folder = './features/'
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

if __name__ == '__main__':
	all_centers = []
	for cat in categories:
		print(cat)
		for head in range(8):
			model = vMFMM(vc_num, 'k++')
			feats = np.load(feat_folder + cat + '/features.npy')[:,head,:,:]
			print(feats.shape)
			features_flat = feats.reshape((-1, 256))
			if feats.shape[0] > 0:
				# print(features_flat.shape)
				model.fit(features_flat, vMF_kappa, max_it=150)
				# print(model.mu.shape)
				all_centers.extend(model.mu)
				with open(feat_folder + cat + '/center_dict_'+str(head)+'.pickle', 'wb') as fh:
					pickle.dump(model.mu, fh)
    # print(np.shape(np.array(all_centers)))
    # with open(feat_folder + 'center_dict.pickle', 'wb') as fh:
    #     pickle.dump(np.array(all_centers), fh)
    # np.save(feat_folder + 'center_dict.npy', np.array(all_centers))