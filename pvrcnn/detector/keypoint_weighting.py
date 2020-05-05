import torch
import math
from torch import nn
import torch.nn.functional as F

from pvrcnn.ops import sigmoid_focal_loss, batched_nms_rotated
from pvrcnn.core.geometry import points_in_boxes
from .layers import MLP


class KeypointFeatureWeighting(nn.Module):
	"""
	MLP for deciding if point is in the foreground or background
	"""

	def __init__(self, cfg):
		super(KeypointFeatureWeighting, self).__init__()
		self.cfg = cfg
		self.mlp = self.build_mlp(cfg)

	def build_mlp(self, cfg):
		"""
		TODO: Check if should use bias.
		"""
		channels = cfg.KEYPOINT_WEIGHT.MLPS
		mlp = MLP(channels, bias=True, bn=False, relu=[True, False])
		return mlp

	def generate_ground_truth(self, item):
		"""
		This will generate the ground truth mask as defined by the GT bounding boxes
		- Check if each keypoint is in GT bounding box - box is [x, y, z, w, l, h, theta], point is [x,y,z]
		- Output is a numpy array of zeros and ones.
		"""
		mask = torch.zeros((item['keypoints'].size()[0], item['keypoints'].size()[1]), dtype=torch.int32)
		for i in range(item['keypoints'].size()[0]):
			boxes = item['boxes'][i]           # Tensor of shape [b, 7]
			keypoints = item['keypoints'][i]   # Tensor of shape [2048, 3]
			mask[i] = torch.from_numpy(points_in_boxes(keypoints, boxes))
		item['G_keypoint_seg'] = mask
		return item

	def subset_keypoints(self, item, point_features):
		"""
		This will subset item['keypoints'] and keep only the ones that pass, and subset point features in same way
		- Based on predictions and a threshold
		TODO Write cleaner code to deal with masking
		"""
		threshold = self.cfg.KEYPOINT_WEIGHT.THRESHOLD
		point_features = point_features.permute(0, 2, 1)
		mask = (item['P_keypoint_seg'] > threshold)
		point_features = point_features[mask, :]
		if len(point_features.shape) < 3:
			point_features = point_features.view(-1, *point_features.shape) # Add lost dimension
		point_features = point_features.permute(0, 2, 1)  
		item['keypoints'] = item['keypoints'][mask]              
		return item, point_features

	def forward(self, item, point_features):
		"""
		in:
		point_features expects (B, 512, 2048)
		out:
		point_features is (B, 512, masked(2048))
		"""
		item = self.generate_ground_truth(item)
		P_keypoint_seg = torch.sigmoid(self.mlp(point_features.permute(0, 2, 1))) # [1, 2048, 1]
		item['P_keypoint_seg'] = torch.squeeze(P_keypoint_seg, dim=2)             # [1, 2048]
		point_features, item = self.subset_keypoints(item, point_features)
		return point_features, item

	def inference(self, item, point_features):
		P_keypoint_seg = torch.sigmoid(self.mlp(point_features.permute(0, 2, 1))) # [1, 2048, 1]
		item['P_keypoint_seg'] = torch.squeeze(P_keypoint_seg, dim=2)             # [1, 2048]
		point_features, item = self.subset_keypoints(item, point_features, point_probabilties)
		return point_features, item


class KeypointWeightingLoss(nn.Module):
	"""
	This is basically going to be the focal loss of whether or not a point appears in the 
	foreground or the background.
	"""

	def __init__(self, cfg):
		super(KeypointWeightingLoss, self).__init__()
		self.cfg = cfg
