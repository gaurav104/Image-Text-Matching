import os
import glob

import numpy as np
# import scipy.io as sio
# from PIL import Image
# import kornia.augmentation as F


import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, BatchSampler
from torchvision import transforms
import subprocess
import random
import pickle
import json
# import kornia as K

from PIL import Image, ImageOps, ImageChops
# from scipy.misc import imread, imresize

from random import random, randint, uniform, choice, shuffle
# from kornia.augmentation import AugmentationBase


class Flickr30k(data.Dataset):
	pklname_list = ['train_sort.pkl', 'val_sort.pkl', 'test_sort.pkl']
	
	def __init__(self, split, image_root, anno_root, max_length, transforms=None):
		
		self.split = split
		self.image_root = image_root
		self.anno_root = anno_root
		self.max_length = max_length
		self.transforms = transforms

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
		self.captions, self.masks = list(zip(*[self.fix_length(np.array(caption)) for caption in self.captions]))


		#initilializing imgs with binary labels for classificatio

		
	def __getitem__(self, index):

		img_path, caption, label, mask = self.images[index], self.captions[index], self.labels[index], self.masks[index]
		
		img_path = os.path.join(self.image_root, img_path)
		img = Image.open(img_path).convert('RGB')
		img = img.resize((224,224))
		# if len(img.shape) == 2:
		#     img = np.dstack((img,img,img))
		# img = Image.fromarray(img)

		if self.transforms is not None:
			img = self.transforms(img)


		# caption = np.array(caption)
		# caption, mask = self.fix_length(caption)

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

class Flickr30k_captions(data.Dataset):
	'''
	Unique caption loader with labels  for inference

	'''
	pklname_list = ['train_sort.pkl', 'val_sort.pkl', 'test_sort.pkl']
	
	def __init__(self, split, image_root, anno_root, max_length, transforms=None):
		
		self.split = split
		# self.image_root = image_root
		self.anno_root = anno_root
		self.max_length = max_length


		if self.split == 'train':
			self.pklname = self.pklname_list[0]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				self.captions = data['caption_id']
				# self.images = data['images_path']

		elif self.split == 'val':
			self.pklname = self.pklname_list[1]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				self.captions = data['caption_id']
				# self.images = data['images_path']

		elif self.split == 'test':
			self.pklname = self.pklname_list[2]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				self.captions = data['caption_id']
				# self.images = data['images_path']

		self.captions, self.masks = list(zip(*[self.fix_length(np.array(caption)) for caption in self.captions]))

		#initilializing imgs with binary labels for classificatio

	def __getitem__(self, index):

		caption, label, mask =  self.captions[index], self.labels[index], self.masks[index]
		
		# img_path = os.path.join(self.image_root, img_path)
		# img = Image.open(img_path).convert('RGB')
		# img = img.resize((128,256))
		# if len(img.shape) == 2:
		#     img = np.dstack((img,img,img))
		# img = Image.fromarray(img)

		# if self.transforms is not None:
		# 	img = self.transforms(img)


		# caption = np.array(caption)
		# caption, mask = self.fix_length(caption)

		#self.remove_duplicate

		return caption, label, mask

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


class Flickr30k_images(data.Dataset):
	'''
	Unique image loader with labels  for inference

	'''
	pklname_list = ['train_sort.pkl', 'val_sort.pkl', 'test_sort.pkl']
	
	def __init__(self, split, image_root, anno_root, max_length, transforms=None):
		
		self.split = split
		self.image_root = image_root
		self.anno_root = anno_root
		# self.max_length = max_length
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
				# self.captions = data['caption_id']
				self.images = data['images_path']

		elif self.split == 'val':
			self.pklname = self.pklname_list[1]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				# self.captions = data['caption_id']
				self.images = data['images_path']

		elif self.split == 'test':
			self.pklname = self.pklname_list[2]
			with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
				data = pickle.load(f_pkl)
				self.labels = data['labels']
				# self.captions = data['caption_id']
				self.images = data['images_path']

		self.unq_images, self.unq_labels = self.remove_duplicate(self.images, self.labels)

		#initilializing imgs with binary labels for classificatio

	def __getitem__(self, index):

		img_path, label = self.unq_images[index],  self.unq_labels[index]
		
		img_path = os.path.join(self.image_root, img_path)
		img = Image.open(img_path).convert('RGB')
		img = img.resize((128,256))
		# if len(img.shape) == 2:
		#     img = np.dstack((img,img,img))
		# img = Image.fromarray(img)

		if self.transforms is not None:
			img = self.transforms(img)

		# caption = np.array(caption)
		# caption, mask = self.fix_length(caption)

		return img, label

	def remove_duplicate(self, feature_seq, label_seq):

		feature_label_pairs = list(zip(feature_seq, label_seq))
		unq_feature_label_pairs = list(set(feature_label_pairs))

		# images_labels_unzipped = list(zip(*images_labels_unique)) 

		unq_feature_label_pairs_unzipped = list(zip(*unq_feature_label_pairs))

		unq_feature_seq, unq_label_seq = list(map(list,unq_feature_label_pairs_unzipped))

		return unq_feature_seq, unq_label_seq

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
		return len(self.unq_labels)



class Flickr30kLoader:
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
			image_dir = os.path.join(output_string, 'Flickr30k/images')
			anno_dir = os.path.join(output_string, 'Flickr30k/processed_data')
			# root_infer = os.path.join(output_string, 'brats_data/')
			self.config.image_dir = image_dir
			self.config.anno_dir = anno_dir

		assert self.config.mode in ['train', 'test']
   
		self.input_transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
														transforms.RandomResizedCrop((224,224), scale=(0.8,0.8), ratio=(1.0,1.0), interpolation=2),
														transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        												transforms.ToTensor(),
        												transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		self.input_transform = transforms.Compose([transforms.ToTensor(),
												transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

		# image_root, anno_root, max_length, transforms=None

		train_set = Flickr30k('train', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							self.input_transform_train
						   )
		validation_set = Flickr30k('val', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							transforms=self.input_transform)

		test_set = Flickr30k('test', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							transforms=self.input_transform)
        
		test_set_images = Flickr30k_images('test', 
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							transforms=self.input_transform)

		test_set_captions = Flickr30k_captions('test',
							self.config.image_dir,
							self.config.anno_dir,
							self.config.max_length,
							transforms=self.input_transform)
        
# 			if len(self.config.modality)==1:


		self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
		self.validation_iterations  = (len(validation_set) + self.config.batch_size) // self.config.batch_size
		# self.test_iterations  = (len(test_set) + self.config.batch_size) // self.config.batch_size


		# with open(os.path.join(self.config.anno_dir, 'train_sort.pkl'), 'rb') as f_pkl:
		# 	data = pickle.load(f_pkl)
		# 	labels = torch.tensor(data['labels'])

		# self.train_loader_balanced = DataLoader(train_set, batch_sampler = BalancedBatchSampler(labels, self.config.classes_pb , self.config.samples_pc))

		
		self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)



		self.val_loader = DataLoader(validation_set, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
		self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)

		self.test_loader_image = DataLoader(test_set_images, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
		self.test_loader_caption = DataLoader(test_set_captions, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)


	def finalize(self):
		pass

# def classification_augmentations(x):
# 	transforms = 

# 	return transforms(x)
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.n_distinct_classes = len(self.labels_set) 

    def __iter__(self):
        self.count = 0
        shuffle(self.labels_set)
        self.initial_index = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)#self.labels_set[self.initial_index:min(self.initial_index+self.n_classes, self.n_distinct_classes)] 
            # np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            shuffle(indices)
            # print(self.count)
            yield indices
            self.count += self.n_classes * self.n_samples
            self.initial_index += self.n_classes
            if self.initial_index>self.n_distinct_classes:
            	self.initial_index=0

    def __len__(self):
        return self.n_dataset // self.batch_size




	
if __name__ == '__main__':

	transformations = transforms.Compose([transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop((224,224), scale=(0.8,0.8), ratio=(1.0,1.0), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	
	# with open(os.path.join("/home/gaurav/Desktop/person-id/processed_data", 'train_sort.pkl'), 'rb') as f_pkl:
	# 	data = pickle.load(f_pkl)
	# 	labels = torch.tensor(data['labels'])

	# print(labels[-1])

	train_set = Flickr30k('train', "/home/gaurav/Desktop/person-id/Datasets/Flickr30k/images","/home/gaurav/Desktop/person-id/Datasets/Flickr30k/processed_data",30,transformations)

	# # # valid_set = CellSeg('val', "../../Dataset/",
	# # #                            transforms.Compose([transforms.ToTensor()]))
	# # # valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)



	# train_loader = DataLoader(train_set, batch_sampler = BalancedBatchSampler(labels, 16 , 4))
	train_loader = DataLoader(train_set, batch_size = 8,shuffle=True)


	# label1 = 0
	# label2 = 0

	# for ep in range(4):
	for step, (image, caption, label, mask) in enumerate(train_loader):
		print(mask.dtype)

	# 	if step == 4:
	# 		break

	# for step, (image, caption, label, mask) in enumerate(train_loader):
	# 	print(label)

	# 	if step == 4:
	# 		break


	# print(step)


	# for image, caption, label, mask in train_loader:
	# 	label2 = label
	# 	break

	# print(label1)
	# print(label2)
	# #     .s


