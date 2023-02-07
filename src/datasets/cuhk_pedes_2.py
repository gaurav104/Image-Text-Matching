import os
import glob

import numpy as np
# import scipy.io as sio
# from PIL import Image
# import kornia.augmentation as F


import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import subprocess
import random
import pickle
import json
# import kornia as K


from PIL import Image, ImageOps, ImageChops
# from scipy.misc import imread, imresize

from random import random, randint, uniform, choice
# from kornia.augmentation import AugmentationBase






class CUHK_pedes(data.Dataset):
	pklname_list = ['train_sort.pkl', 'val_sort.pkl', 'test_sort.pkl']
	
	def __init__(self, split, image_root, anno_root, max_length, transforms=None):
		
		self.split = split
		self.image_root = image_root
		self.anno_root = anno_root
		self.max_length = max_length
		self.transforms = transforms
		# self.modality = modality
		# self.augmentation = augmentation
		# self.mapping = {
		#     0: 0,
		#     255: 1              
		# }

		if self.split == 'train':
			self.pklname = self.pklname_list[0]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				self.captions = data['caption_id']
				self.images = data['images_path']

		elif self.split == 'val':
			self.pklname = self.pklname_list[1]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				self.captions = data['caption_id']
				self.images = data['images_path']

		elif self.split == 'test':
			self.pklname = self.pklname_list[2]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				self.captions = data['caption_id']
				self.images = data['images_path']


		#initilializing imgs with binary labels for classificatio

		

	def __getitem__(self, index):

		img_path, caption, label = self.images[index], self.captions[index], self.labels[index]
		
		img_path = os.path.join(self.image_root, img_path)
		img = Image.open(img_path).convert('RGB')
		img = img.resize((128,256))
		# if len(img.shape) == 2:
		#     img = np.dstack((img,img,img))
		# img = Image.fromarray(img)

		if self.transforms is not None:
			img = self.transforms(img)


		caption = np.array(caption)
		caption, mask = self.fix_length(caption)

		return img, caption, label, mask

	def fix_length(self, caption):
		caption_len = caption.shape[0]
		if caption_len < self.max_length:
			pad = np.zeros((self.max_length - caption_len, 1), dtype=np.int64)
			caption = np.append(caption, pad)
		else:
			caption = caption[:self.max_length]

		caption_len = min(caption_len, self.max_length)
		return caption, caption_len
		
	

	def __len__(self):
		return len(self.labels)





class CUHKLoader:
	def __init__(self, config):
		self.config = config

		# if self.config.run_on_cluster:
		# 	output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
		# 	output_string = output_bytes.decode('utf-8').strip()
		# 	root_classification = os.path.join(output_string, 'brats_classification/')
		# 	root_infer = os.path.join(output_string, 'brats_data/')
		# 	self.config.data_root = root_classification
		# 	self.config.data_root_infer = root_infer

		if self.config.run_on_cluster:
			output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
			output_string = output_bytes.decode('utf-8').strip()
			image_dir = os.path.join(output_string, 'CUHK-PEDES/imgs')
			anno_dir = os.path.join(output_string, 'CUHK-PEDES/processed_data_2')
			# root_infer = os.path.join(output_string, 'brats_data/')
			self.config.image_dir = image_dir
			self.config.anno_dir = anno_dir

		assert self.config.mode in ['train', 'test']
   
		self.input_transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
														transforms.RandomResizedCrop((256,128), scale=(0.8,0.8), ratio=(0.5,0.5), interpolation=2),
														transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        												transforms.ToTensor(),
        												transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		self.input_transform = transforms.Compose([transforms.ToTensor(),
												transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

		# image_root, anno_root, max_length, transforms=None

		train_set = CUHK_pedes('train', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							self.input_transform_train
						   )
		validation_set = CUHK_pedes('val', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							transforms=self.input_transform)
        
		test_set = CUHK_pedes('test', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							transforms=self.input_transform)
        
# 			if len(self.config.modality)==1:


		self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
		self.validation_iterations  = (len(validation_set) + self.config.batch_size) // self.config.batch_size
		self.test_iterations  = (len(test_set) + self.config.batch_size) // self.config.batch_size



		self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
		self.val_loader = DataLoader(validation_set, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
		self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)


	def finalize(self):
		pass

# def classification_augmentations(x):
# 	transforms = 

# 	return transforms(x)





	
if __name__ == '__main__':

	transformations = transforms.Compose([transforms.RandomCrop((224,112)),transforms.RandomHorizontalFlip(),
        												transforms.ToTensor(),
        												transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_set = CUHK_pedes('train', "/home/gaurav/Desktop/person-id/CUHK-PEDES/CUHK-PEDES/imgs","/home/gaurav/Desktop/person-id/processed_data",100,transformations)

	# # # valid_set = CellSeg('val', "../../Dataset/",
	# # #                            transforms.Compose([transforms.ToTensor()]))
	# # # valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)

	train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

	for image, caption, label, mask in train_loader:
		print(image.shape)
	#     .s