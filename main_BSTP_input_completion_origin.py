import os

node_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = node_id
import numpy as np
import torch
import numpy
import matplotlib
import copy

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

device = torch.device('cuda')
from datetime import date

m = int(2e4)  # Input size
n = int(1.5e4)  # Output size
fq = 0.005   # Plateau probability
fq_half = fq / 2
fp = 0.005
p_w = 0.6
data_name = 'feedback_M_masked_' + date.today().strftime("%d-%m") + '_pw_' + str(p_w) + 'fq' + str(
	fq) + '_orin_v' + node_id + '.mat'


def fix_random_seeds(seed_value=0):
	np.random.seed(seed_value)  # Fix random seed for numpy
	torch.random.seed()


fix_random_seeds(1111)


def np2torch(x):
	return torch.from_numpy(x).cuda()


def torch2np(x):
	return x.cpu().detach().numpy()

precison = torch.float32

num_img_list = np.arange(1e3, 2e4, 2e3)
num_mask_list = np.arange(0.0, 0.9, 0.02)

# fasr
num_img_list = [1e4,1.5e4,2e4]
num_mask_list = [0.3, 0.5, 0.7]
records = []
for num_images in num_img_list:
	# Initialize weight matrices
	print('\n\n num_images', num_images)
	num_images = int(num_images)
	X = torch.Tensor(np.random.binomial(n=1, p=fp, size=(m, num_images))).cuda().to(precison).T
	plateaus = (torch.rand(num_images, n).cuda() <= fq_half).to(precison)
	sum_W = X.T @ plateaus
	W_feed_init = 0.
	W_back_init = 0.
	W_feed_init_control = 0.

	sum_W = X.T @ plateaus
	W_mask1 = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
	W_mask2 = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
	image_intensity = X.sum(1).mean()
	thr_ca1 = int(m * fp * p_w * 0.6)
	thr_ca2 = int(m * fp * p_w * 0.6)
	## select one threshold for learning
	# Method1:  fast simulation (recommended)
	W_feed_init = X.T @ plateaus
	W_feed_init = (W_feed_init * W_mask1) % 2
	y_sum = X @ W_feed_init
	spikes1 = (y_sum > thr_ca1).to(precison)
	W_back_init += spikes1.T @ X

	W_feed = W_feed_init * W_mask1
	W_back = W_back_init * W_mask2.T
	W_back = (W_back >= 1).to(precison)
	
	W_feed = W_feed_init * W_mask1
	W_back = W_back_init * W_mask2.T
	del W_mask1, W_mask2, W_back_init, W_feed_init
	torch.cuda.empty_cache()
	
	for masked_ratio in num_mask_list:
		# Mask the top half of the patterns and project the results using Wf
		X_masked = X.clone()
		X_masked[:, : int(m * masked_ratio)] = 0
		reconstruct_results = []
		zab = 2 * m * (1 - X.mean()) * X.mean()
		zab = zab.item()
		err1 = (X - X_masked).abs().mean()
		"""
		BTSP
		"""
		winner_k = int(n*fq_half*0.75)
		input_sum_ca1 = X_masked @ W_feed
		# Get the topk indices along the dimension dim=1
		_, topk_indices = torch.topk(input_sum_ca1, k=winner_k, dim=1)



		# Create a binary mask with the same shape as input_sum_ca1, initialized to zeros
		mask = torch.zeros_like(input_sum_ca1, dtype=torch.bool)

		# Scatter 1s to the positions of the top k elements
		mask.scatter_(dim=1, index=topk_indices, value=True)
		thr_ca1 = 1
		y_ = mask.to(precison)
		X_projected = y_ @ W_back
		if X_projected.max() > 200:
			max_range = 200
		else:
			max_range = max(X_projected.max(), 2)
		max_range = int(max_range)
		steps = int(max_range / 100) + 1
		for thr_ca3 in range(0, max_range, steps):
			tmp = (X_projected >= thr_ca3).float()
			err0 = (tmp - X).abs().mean()
			items = [thr_ca1, thr_ca3, err0.item(), err1.item()]
			reconstruct_results.append(items)
		reconstruct_array = np.array(reconstruct_results)

		# print(reconstruct_array,max_range,steps )
		idx_min_err = reconstruct_array[:, 2].argmin()
		opt_thr_ca1, opt_thr_ca3, opt_err = reconstruct_array[idx_min_err][:3]
		
		"""
		control_model
		"""
		reconstruct_control = []
		input_sum_ca1_control = X_masked @ W_feed
		for thr_ca1_ratio in np.arange(0.05, 0.9, 0.05):
			thr_ca1 = int(image_intensity * thr_ca1_ratio)
			y_control = (input_sum_ca1_control >= thr_ca1).float()
			X_projected_control = y_control @ W_back
			if X_projected_control.max() > 200:
				max_range = 200
			else:
				max_range = max(X_projected_control.max(), 2)
			steps = int(max_range / 100) + 1
			for thr_ca3 in range(0, int(max_range), steps):
				tmp = (X_projected_control >= thr_ca3).float()
				err0 = (tmp - X).abs().mean()
				items = [thr_ca1, thr_ca3, err0.item(), err1.item()]
				reconstruct_control.append(items)
		
		reconstruct_control_array = np.array(reconstruct_control)
		idx_min_err = reconstruct_control_array[:, 2].argmin()
		opt_thr_ca1_control, opt_thr_ca3_ca1_control, opt_err_ca1_control = reconstruct_control_array[idx_min_err][:3]
		
		record_items = [num_images, masked_ratio, opt_thr_ca1, opt_thr_ca3, opt_err, err1.item(), zab,
		                opt_thr_ca1_control,
		                opt_thr_ca3_ca1_control, opt_err_ca1_control]
		records.append(record_items)
		raw_err = err1
		print('masking fraction {:.4f}, raw err {:.4f}, reconstruction  by btsp{:.4f} ratio {:.4f} '
		      ' reconstruction   by control{:.4f} ratio {:.4f}'.format(masked_ratio, raw_err, opt_err,
		                                                               opt_err / (raw_err + 1e-4),
		                                                               opt_err_ca1_control,
		                                                               opt_err_ca1_control / (raw_err + 1e-4),
		                                                               ))
	
	import scipy.io as sio
	
	data_path = './results/'
	record_array = np.array(records)
	filename = data_path + data_name
	sio.savemat(filename, {'num_img_list': num_img_list, 'num_mask_list': num_mask_list, 'm': m, 'n': n,
	                       'records': record_array})
