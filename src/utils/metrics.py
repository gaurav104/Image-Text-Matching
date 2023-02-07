"""
This file will contain the metrics of the framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from hyfi.hypernn import MobiusLinear, MobiusMLR, ToPoincare


logger = logging.getLogger()                                                                                                                                                                            
logger.setLevel(logging.INFO)

class IOUMetric:
	"""
	Class to calculate mean-iou using fast_hist method
	"""

	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.hist = np.zeros((num_classes, num_classes))

	def _fast_hist(self, label_pred, label_true):
		mask = (label_true >= 0) & (label_true < self.num_classes)
		hist = np.bincount(
			self.num_classes * label_true[mask].astype(int) +
			label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
		return hist

	def add_batch(self, predictions, gts):
		for lp, lt in zip(predictions, gts):
			self.hist += self._fast_hist(lp.flatten(), lt.flatten())

	def evaluate(self):
		acc = np.diag(self.hist).sum() / self.hist.sum()
		acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
		acc_cls = np.nanmean(acc_cls)
		iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
		mean_iu = np.nanmean(iu)
		freq = self.hist.sum(axis=1) / self.hist.sum()
		fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
		return acc, acc_cls, iu, mean_iu, fwavacc


class AverageMeter:
	"""
	Class to be an average meter for any average metric like loss, accuracy, etc..
	"""

	def __init__(self):
		self.value = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.reset()

	def reset(self):
		self.value = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.value = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	@property
	def val(self):
		return self.avg


class AverageMeterList:
	"""
	Class to be an average meter for any average metric List structure like mean_iou_per_class
	"""

	def __init__(self, num_cls):
		self.cls = num_cls
		self.value = [0] * self.cls
		self.avg = [0] * self.cls
		self.sum = [0] * self.cls
		self.count = [0] * self.cls
		self.reset()

	def reset(self):
		self.value = [0] * self.cls
		self.avg = [0] * self.cls
		self.sum = [0] * self.cls
		self.count = [0] * self.cls

	def update(self, val, n=1):
		for i in range(self.cls):
			self.value[i] = val[i]
			self.sum[i] += val[i] * n
			self.count[i] += n
			self.avg[i] = self.sum[i] / self.count[i]

	@property
	def val(self):
		return self.avg


def cls_accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k / batch_size)
	return res

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
from torch.autograd import Variable

logger = logging.getLogger()                                                                                                                                                                            
logger.setLevel(logging.INFO)



class EMA():
	def __init__(self, decay=0.999):
		self.decay = decay
		self.shadow = {}

	def register(self, name, val):
		self.shadow[name] = val.cpu().detach()

	def get(self, name):
		return self.shadow[name]

	def update(self, name, x):
		assert name in self.shadow
		new_average = (1.0 - self.decay) * x.cpu().detach() + self.decay * self.shadow[name]
		self.shadow[name] = new_average.clone()


def pairwise_distance(A, B):
	"""
	Compute distance between points in A and points in B
	:param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
	:param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
	:return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
	"""
	A_square = torch.sum(A * A, dim=1, keepdim=True)
	B_square = torch.sum(B * B, dim=1, keepdim=True)

	distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

	return distance


def one_hot_coding(index, k):
	if type(index) is torch.Tensor:
		length = len(index)
	else:
		length = 1
	out = torch.zeros((length, k), dtype=torch.int64).cuda()
	index = index.reshape((len(index), 1))
	out.scatter_(1, index, 1)
	return out


# deprecated due to the large memory usage
def constraints_old(features, labels):
	distance = pairwise_distance(features, features)
	labels_reshape = torch.reshape(labels, (features.shape[0], 1))
	labels_dist = labels_reshape - labels_reshape.t()
	labels_mask = (labels_dist == 0).float()

	# Average loss with each matching pair
	num = torch.sum(labels_mask) - features.shape[0]
	if num == 0:
		con_loss = 0.0
	else:
		con_loss = torch.sum(distance * labels_mask) / num

	return con_loss


def constraints(features, labels):
	labels = torch.reshape(labels, (labels.shape[0],1))
	con_loss = AverageMeter()
	index_dict = {k.item() for k in labels}
	for index in index_dict:
		labels_mask = (labels == index)
		feas = torch.masked_select(features, labels_mask)
		feas = feas.view(-1, features.shape[1])
		distance = pairwise_distance(feas, feas)
		#torch.sqrt_(distance)
		num = feas.shape[0] * (feas.shape[0] - 1)
		loss = torch.sum(distance) / num
		con_loss.update(loss, n = num / 2)
	return con_loss.avg


def constraints_loss(data_loader, network, config):
	network.eval()
	max_size = config.batch_size * len(data_loader)
	images_bank = torch.zeros((max_size, config.feature_size)).cuda()
	text_bank = torch.zeros((max_size,config.feature_size)).cuda()
	labels_bank = torch.zeros(max_size).cuda()
	index = 0
	con_images = 0.0
	con_text = 0.0
	with torch.no_grad():
		for images, captions, labels, captions_length in data_loader:
			images = images.cuda()
			captions = captions.cuda(),
			captions_length = captions_length.cuda()
			interval = images.shape[0]
			image_embeddings, text_embeddings = network(images, captions, captions_length)
			images_bank[index: index + interval] = image_embeddings
			text_bank[index: index + interval] = text_embeddings
			labels_bank[index: index + interval] = labels
			index = index + interval
		images_bank = images_bank[:index]
		text_bank = text_bank[:index]
		labels_bank = labels_bank[:index]
	
	if config.constraints_text:
		con_text = constraints(text_bank, labels_bank)
	if config.constraints_images:
		con_images = constraints(images_bank, labels_bank)

	return con_images, con_text
   
class LossSingleModality(nn.Module):
	def __init__(self, config):
		super(LossSingleModality, self).__init__()
		self.CMPM = config.CMPM
		self.CMPC = config.CMPC
		self.epsilon = config.epsilon
		self.num_classes = config.num_classes
		self.e  = config.e


		# if config.resume:
		#     checkpoint = torch.load(config.model_path)
		#     self.W = Parameter(checkpoint['W'])
		#     print('=========> Loading in parameter W from pretrained models')
		# else:
		self.W = Parameter(torch.randn(config.feature_size, config.num_classes))
		self.init_weight()

	def init_weight(self):
		nn.init.xavier_uniform_(self.W.data, gain=1)
		

	def compute_cmpc_loss(self, embeddings, labels):
		"""
		Cross-Modal Projection Classfication loss(CMPC)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
		"""
		criterion = nn.CrossEntropyLoss(reduction='mean')
		self.W_norm = self.W / self.W.norm(dim=0)
		#labels_onehot = one_hot_coding(labels, self.num_classes).float()
#         image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
#         text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

#         image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
#         text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

#         image_logits = torch.matmul(image_proj_text, self.W_norm)
#         text_logits = torch.matmul(text_proj_image, self.W_norm)

		logits = torch.matmul(embeddings, self.W_norm)
		# text_logits = torch.matmul(text_embeddings, self.W_norm)
		
		#labels_one_hot = one_hot_coding(labels, num_classes)
		'''
		ipt_loss = criterion(input=image_logits, target=labels)
		tpi_loss = criterion(input=text_logits, target=labels)
		cmpc_loss = ipt_loss + tpi_loss
		'''
		cmpc_loss = criterion(logits, labels)# + criterion(text_logits, labels)
		#cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
		#cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
		# classification accuracy for observation
		pred = torch.argmax(logits, dim=1)
		# text_pred = torch.argmax(text_logits, dim=1)

		precision = torch.mean((pred == labels).float())
		# text_precision = torch.mean((text_pred == labels).float())

		return cmpc_loss, precision


	def compute_cmpm_loss(self, embeddings, labels):
		"""
		Cross-Modal Projection Matching Loss(CMPM)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
			i2t_loss: cmpm loss for image projected to text
			t2i_loss: cmpm loss for text projected to image
			pos_avg_sim: average cosine-similarity for positive pairs
			neg_avg_sim: averate cosine-similarity for negative pairs
		"""

		batch_size = embeddings.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_mask = (labels_dist == 0)
		
		norm_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
		# text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
		cosine_similarity = torch.matmul(norm_embeddings, norm_embeddings.t())
		# text_proj_image = torch.matmul(text_embeddings, image_norm.t())

		# normalize the true matching distribution
		labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)
		# print(labels_mask_norm[0])
		 
		cosine_similarity_dist = F.softmax(cosine_similarity, dim=1)
		#i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		cosine_contrastive_loss = cosine_similarity_dist * (F.log_softmax(cosine_similarity, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))
		cosine_kl_loss = labels_mask_norm*(torch.log(labels_mask_norm*self.e + self.epsilon) - F.log_softmax(cosine_similarity, dim=1))
		
		# t2i_pred = F.softmax(text_proj_image, dim=1)
		# #t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		# t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))

		cmpm_loss = torch.mean(torch.sum(cosine_kl_loss, dim=1))

		# sim_cos = torch.matmul(norm_embeddings, norm_embeddings.t())

		pos_avg_sim = torch.mean(torch.masked_select(cosine_similarity, labels_mask))
		neg_avg_sim = torch.mean(torch.masked_select(cosine_similarity, labels_mask == 0))
		
		return cmpm_loss, pos_avg_sim, neg_avg_sim

	def forward(self, embeddings, labels):
		cmpm_loss = 0.0
		cmpc_loss = 0.0
		precision = 0.0
		# text_precision = 0.0
		neg_avg_sim = 0.0
		pos_avg_sim =0.0
		if self.CMPM:
			cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(embeddings, labels)
		if self.CMPC:
			cmpc_loss, precision= self.compute_cmpc_loss(embeddings, labels)
		
		loss = cmpm_loss + cmpc_loss #
		
		return cmpm_loss, cmpc_loss, loss, precision, pos_avg_sim, neg_avg_sim

		


class Loss(nn.Module):
	def __init__(self, config):
		super(Loss, self).__init__()
		self.CMPM = config.CMPM
		self.CMPC = config.CMPC
		self.epsilon = config.epsilon
		self.num_classes = config.num_classes
		self.e  =config.e


		# if config.resume:
		#     checkpoint = torch.load(config.model_path)
		#     self.W = Parameter(checkpoint['W'])
		#     print('=========> Loading in parameter W from pretrained models')
		# else:
		self.W = Parameter(torch.randn(config.feature_size, config.num_classes))
		self.init_weight()

	def init_weight(self):
		nn.init.xavier_uniform_(self.W.data, gain=1)
		

	def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
		"""
		Cross-Modal Projection Classfication loss(CMPC)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
		"""
		criterion = nn.CrossEntropyLoss(reduction='mean')
		self.W_norm = self.W / self.W.norm(dim=0)
		#labels_onehot = one_hot_coding(labels, self.num_classes).float()
#         image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
#         text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

#         image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
#         text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

#         image_logits = torch.matmul(image_proj_text, self.W_norm)
#         text_logits = torch.matmul(text_proj_image, self.W_norm)

		image_logits = torch.matmul(image_embeddings, self.W_norm)
		text_logits = torch.matmul(text_embeddings, self.W_norm)
		
		#labels_one_hot = one_hot_coding(labels, num_classes)
		'''
		ipt_loss = criterion(input=image_logits, target=labels)
		tpi_loss = criterion(input=text_logits, target=labels)
		cmpc_loss = ipt_loss + tpi_loss
		'''
		cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
		#cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
		#cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
		# classification accuracy for observation
		image_pred = torch.argmax(image_logits, dim=1)
		text_pred = torch.argmax(text_logits, dim=1)

		image_precision = torch.mean((image_pred == labels).float())
		text_precision = torch.mean((text_pred == labels).float())

		return cmpc_loss, image_precision, text_precision


	def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
		"""
		Cross-Modal Projection Matching Loss(CMPM)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
			i2t_loss: cmpm loss for image projected to text
			t2i_loss: cmpm loss for text projected to image
			pos_avg_sim: average cosine-similarity for positive pairs
			neg_avg_sim: averate cosine-similarity for negative pairs
		"""

		batch_size = image_embeddings.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_mask = (labels_dist == 0)
		
		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
		image_proj_text = torch.matmul(image_embeddings, text_norm.t())
		text_proj_image = torch.matmul(text_embeddings, image_norm.t())

		# normalize the true matching distribution
		labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)
		# print(labels_mask_norm[0])
		 
		i2t_pred = F.softmax(image_proj_text, dim=1)
		#i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))
		
		t2i_pred = F.softmax(text_proj_image, dim=1)
		#t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))

		cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

		sim_cos = torch.matmul(image_norm, text_norm.t())

		pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
		neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
		
		return cmpm_loss, pos_avg_sim, neg_avg_sim

	def forward(self, image_embeddings, text_embeddings, labels):
		cmpm_loss = 0.0
		cmpc_loss = 0.0
		image_precision = 0.0
		text_precision = 0.0
		neg_avg_sim = 0.0
		pos_avg_sim =0.0
		if self.CMPM:
			cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
		if self.CMPC:
			cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings, labels)
		
		loss = cmpm_loss + cmpc_loss #
		
		return cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim


class LossHyperbolic(nn.Module):
	"""docstring for LossHyperbolic"""
	def __init__(self, config):
		super(LossHyperbolic, self).__init__()
		self.ENTL = config.ENTL
		self.HYPNL = config.HYPNL
		self.CLS = config.CLS
		self.alpha = config.alpha
		self.K = config.K
		self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))
		self.epsilon = 1e-5
		self.e = config.e
		# self.use_CNN = use_CNN
		# self.dataloader = None

		# self.classifier_layer_hyp = MobiusMLR(config.feature_size, config.num_classes)
		self.classifier_layer_hyp = MobiusLinear(config.feature_size, config.num_classes, hyperbolic_bias=False)
		self.classifier_layer_hyp.float()

	def compute_classification(self, image_embeddings, text_embeddings, labels):

		criterion = nn.CrossEntropyLoss(reduction='mean')

		image_logits = self.classifier_layer_hyp(image_embeddings)
		text_logits = self.classifier_layer_hyp(text_embeddings)

		loss =  criterion(text_logits, labels) + criterion(image_logits, labels)#resnet101_bilstm_300emb_pretrained_64_Adam2e-6_w0_hyp

		return loss



	def compute_E_operator_pairwise(self, x, y):
		
		original_shape = x.shape
		x = x.view(-1, original_shape[-1])
		y = y.view(-1, original_shape[-1])

		x_norm = torch.norm(x, p=2, dim=1)
		y_norm = torch.norm(y, p=2, dim=1)
		x_y_dist = torch.norm(x - y, p=2, dim=1)

		x_dot_y = torch.sum(x*y, dim=1)

		acos_arg = (x_dot_y*(1+x_norm**2)-(x_norm**2)*(1+y_norm**2))/(x_norm*x_y_dist*torch.sqrt(1+(x_norm*y_norm)**2-2*x_dot_y))
		
		# in angle space (radians)
		theta_between_x_y = torch.acos(torch.clamp(acos_arg, min=-1+1e-5, max=1-1e-5))
		psi_x = torch.asin(torch.clamp(self.K*(1-x_norm**2)/x_norm, min=-1+1e-5, max=1-1e-5))

		# in cos space
		# theta_between_x_y = acos_arg
		# psi_x = -torch.sqrt(1 - (self.K*(1-x_norm**2)/x_norm)**2)

		return torch.clamp(theta_between_x_y - psi_x, min=0.0).view(original_shape[:-1])

	def compute_E_operator_dist(self,x, y):
		dtype, device = x.dtype, y.device
		rows, cols = self.meshgrid_from_sizes(x, y, dim=0)
		output = torch.zeros(rows.size(), dtype=dtype, device=device)
		rows, cols = rows.flatten(), cols.flatten()
		distances = self.compute_E_operator_pairwise(x[rows], y[cols])
		output[rows, cols] = distances

		return output


	# 




	def compute_entailment(self, x, y, labels):

		batch_size = labels.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_postive = (labels_dist == 0)
		labels_negative = (labels_dist != 0)

		E_values = self.compute_E_operator_dist(x,y)

		E_positive_idx_row, E_positive_idx_col = torch.where(labels_postive)
		E_negative_idx_row, E_negative_idx_col = torch.where(labels_negative)

		e_for_u_v_positive = E_values[E_positive_idx_row, E_positive_idx_col]
		e_for_u_v_negative = E_values[E_negative_idx_row, E_negative_idx_col]

		S = self.get_image_label_loss(e_for_u_v_positive, e_for_u_v_negative)

		return S

	def compute_entailment_allclass(self,class_embeddings_model, y, labels):

		batch_size = labels.shape[0]
		num_classes = class_embeddings_model.embeddings.weight.data.shape[0]
		class_embedding_weight = class_embeddings_model.embeddings.weight

		E_values = self.compute_E_operator_dist(class_embedding_weight, y)

		row_index = labels
		col_index = torch.arange(batch_size)

		label_mask = torch.zeros((num_classes, batch_size))
		label_mask[row_index, col_index]=1
		
		E_positive_idx_row, E_positive_idx_col = torch.where(label_mask==1) 
		E_negative_idx_row, E_negative_idx_col = torch.where(label_mask==0)

		e_for_u_v_positive = E_values[E_positive_idx_row, E_positive_idx_col]
		e_for_u_v_negative = E_values[E_negative_idx_row, E_negative_idx_col]

		S = self.get_image_label_loss(e_for_u_v_positive, e_for_u_v_negative)
		return S

	def compute_hypernym_allclass(self, class_embeddings_model, y, labels):

		criterion = nn.CrossEntropyLoss(reduction='mean')

		batch_size = labels.shape[0]
		num_classes = class_embeddings_model.embeddings.weight.data.shape[0]
		class_embedding_weight = class_embeddings_model.embeddings.weight

		u_v_dist = dist_matrix(class_embedding_weight, y)
		u_v_dist = torch.transpose(u_v_dist, 0, 1)

		loss = criterion(-u_v_dist, labels)

		return loss

	def compute_hypernym(self,x, y, labels):

		# criterion = nn.CrossEntropyLoss(reduction='mean')
		batch_size = labels.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_mask = (labels_dist == 0)

		labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)

		u_v_dist = dist_matrix(x, y)
		u_v_dist_softmax = F.softmax(-u_v_dist, dim=1)

		loss = u_v_dist_softmax * (F.log_softmax(-u_v_dist, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))

		loss = torch.mean(torch.sum(loss, dim=1))

		# i2t_pred = F.softmax(image_proj_text, dim=1)

		# return loss



		
		# num_classes = class_embeddings_model.embeddings.weight.data.shape[0]
		# class_embedding_weight = class_embeddings_model.embeddings.weight


		
		# u_v_dist = torch.transpose(u_v_dist, 0, 1)

		# loss = criterion(-u_v_dist, labels)

		return loss


	

	def get_image_label_loss(self, e_for_u_v_positive, e_for_u_v_negative):
		S = torch.sum(e_for_u_v_positive) + torch.sum(torch.clamp(self.alpha - e_for_u_v_negative, min=0.0))
		return S


	def meshgrid_from_sizes(self, x, y, dim=0):
		a = torch.arange(x.size(dim), device=x.device)
		b = torch.arange(y.size(dim), device=y.device)
		return torch.meshgrid(a, b)



	def forward(self, image_embeddings, text_embeddings, class_embeddings_model, labels):

		class_embeddings = class_embeddings_model(labels)

		# batch_size = self.labels

		entl_loss = 0.0
		hypernym_loss = 0.0
		class_loss = 0.0

		if self.ENTL:
			# image_entailment_loss = self.compute_entailment_allclass(class_embeddings_model, image_embeddings, labels)
			# text_entailment_loss = self.compute_entailment_allclass(class_embeddings_model, text_embeddings, labels)

			identity = torch.arange(labels.size(0), device=labels.device)

			image_entailment_loss = self.compute_entailment(class_embeddings, image_embeddings, labels)
			text_entailment_loss = self.compute_entailment(class_embeddings, text_embeddings, labels)
			# image_text_entailment_loss = self.compute_entailment(image_embeddings, text_embeddings, identity)
			# text_image_entailment_loss = self.compute_entailment(text_embeddings, image_embeddings, labels)
			entl_loss =  text_entailment_loss + image_entailment_loss

		if self.CLS:

			class_loss = self.compute_classification(image_embeddings, text_embeddings, labels)



		if self.HYPNL:
			# image_hypernym_loss = self.compute_hypernym_allclass(class_embeddings_model, image_embeddings, labels)
			# text_hypernym_loss = self.compute_hypernym_allclass(class_embeddings_model, text_embeddings, labels) 
			image_hypernym_loss = self.compute_hypernym(class_embeddings, image_embeddings, labels)
			text_hypernym_loss = self.compute_hypernym(class_embeddings, text_embeddings, labels)
			hypernym_loss = image_hypernym_loss + text_hypernym_loss

		loss = entl_loss + hypernym_loss + class_loss

		return entl_loss, hypernym_loss, loss





class LossFeatureTransfer(nn.Module):
	def __init__(self, config):
		super(LossFeatureTransfer, self).__init__()
		self.CMPM = config.CMPM
		self.CMPC = config.CMPC
		self.epsilon = config.epsilon
		self.num_classes = config.num_classes
		self.e  =config.e


		# if config.resume:
		#     checkpoint = torch.load(config.model_path)
		#     self.W = Parameter(checkpoint['W'])
		#     print('=========> Loading in parameter W from pretrained models')
		# else:
		self.W = Parameter(torch.randn(config.feature_size, config.num_classes))
		self.init_weight()

	def init_weight(self):
		nn.init.xavier_uniform_(self.W.data, gain=1)
		

	def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
		"""
		Cross-Modal Projection Classfication loss(CMPC)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
		"""
		criterion = nn.CrossEntropyLoss(reduction='mean')
		self.W_norm = self.W / self.W.norm(dim=0)
		#labels_onehot = one_hot_coding(labels, self.num_classes).float()
#         image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
#         text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

#         image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
#         text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

#         image_logits = torch.matmul(image_proj_text, self.W_norm)
#         text_logits = torch.matmul(text_proj_image, self.W_norm)

		image_logits = torch.matmul(image_embeddings, self.W_norm)
		text_logits = torch.matmul(text_embeddings, self.W_norm)
		
		#labels_one_hot = one_hot_coding(labels, num_classes)
		'''
		ipt_loss = criterion(input=image_logits, target=labels)
		tpi_loss = criterion(input=text_logits, target=labels)
		cmpc_loss = ipt_loss + tpi_loss
		'''
		cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
		#cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
		#cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
		# classification accuracy for observation
		image_pred = torch.argmax(image_logits, dim=1)
		text_pred = torch.argmax(text_logits, dim=1)

		image_precision = torch.mean((image_pred == labels).float())
		text_precision = torch.mean((text_pred == labels).float())

		return cmpc_loss, image_precision, text_precision


	def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
		"""
		Cross-Modal Projection Matching Loss(CMPM)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
			i2t_loss: cmpm loss for image projected to text
			t2i_loss: cmpm loss for text projected to image
			pos_avg_sim: average cosine-similarity for positive pairs
			neg_avg_sim: averate cosine-similarity for negative pairs
		"""

		batch_size = image_embeddings.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_mask = (labels_dist == 0)
		
		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
		image_proj_text = torch.matmul(image_embeddings, text_norm.t())
		text_proj_image = torch.matmul(text_embeddings, image_norm.t())

		# normalize the true matching distribution
		labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)
		# print(labels_mask_norm[0])
		 
		i2t_pred = F.softmax(image_proj_text, dim=1)
		#i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))
		
		t2i_pred = F.softmax(text_proj_image, dim=1)
		#t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))

		cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

		sim_cos = torch.matmul(image_norm, text_norm.t())

		pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
		neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
		
		return cmpm_loss, pos_avg_sim, neg_avg_sim

	def cosine_similarity_loss(self, target_embeddings, image_embeddings, text_embeddings, eps=1e-12):
	# Normalize each vector by its norm
		# output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
		# output_net = output_net / (output_net_norm + eps)
		# output_net[output_net != output_net] = 0

		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

		target_norm = target_embeddings / target_embeddings.norm(dim=1, keepdim=True)

		# target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
		# target_net = target_net / (target_net_norm + eps)
		# target_net[target_net != target_net] = 0

		# Calculate the cosine similarity
		model_similarity = torch.matmul(text_norm, image_norm.t())
		target_similarity = torch.matmul(target_norm, target_norm.t())

		# Scale cosine similarity to 0..1
		# model_similarity = (model_similarity + 1.0) / 2.0
		# target_similarity = (target_similarity + 1.0) / 2.0

		# # Transform them into probabilities
		# model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
		# target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

		model_similarity_prob = F.softmax(model_similarity/10,dim=1)
		target_similarity_prob = F.softmax(target_similarity/10,dim=1)

		# Calculate the KL-divergence
		loss = torch.mean(100*target_similarity_prob * torch.log((target_similarity_prob + eps) / (model_similarity_prob + eps)))

		return loss

	def forward(self,image_embeddings, text_embeddings, labels, target_embeddings_image=None, target_embeddings_text=None):
		cmpm_loss = 0.0
		cmpc_loss = 0.0
		image_precision = 0.0
		text_precision = 0.0
		neg_avg_sim = 0.0
		pos_avg_sim =0.0
		pkt_loss_text = 0
		pkt_loss_image = 0
		if self.CMPM:
			cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
		if self.CMPC:
			cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings, labels)

		pkt_loss_text = self.cosine_similarity_loss(target_embeddings_image, image_embeddings, text_embeddings)

		loss = cmpm_loss + cmpc_loss + pkt_loss_text + pkt_loss_image
		
		return cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim

	


class Loss_Cross_Project(nn.Module):
	def __init__(self, config):
		super(Loss_Cross_Project, self).__init__()
		self.CMPM = config.CMPM
		self.CMPC = config.CMPC
		self.epsilon = config.epsilon
		self.num_classes = config.num_classes
		self.e  =config.e


		# if config.resume:
		#     checkpoint = torch.load(config.model_path)
		#     self.W = Parameter(checkpoint['W'])
		#     print('=========> Loading in parameter W from pretrained models')
		# else:
		self.W = Parameter(torch.randn(config.feature_size, config.num_classes))
		self.init_weight()

	def init_weight(self):
		nn.init.xavier_uniform_(self.W.data, gain=1)
		

	def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
		"""
		Cross-Modal Projection Classfication loss(CMPC)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
		"""
		criterion = nn.CrossEntropyLoss(reduction='mean')
		self.W_norm = self.W / self.W.norm(dim=0)
		#labels_onehot = one_hot_coding(labels, self.num_classes).float()
		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

		image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
		text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

		image_logits = torch.matmul(image_proj_text, self.W_norm)
		text_logits = torch.matmul(text_proj_image, self.W_norm)

		# image_logits = torch.matmul(image_embeddings, self.W_norm)
		# text_logits = torch.matmul(text_embeddings, self.W_norm)
		
		#labels_one_hot = one_hot_coding(labels, num_classes)
		'''
		ipt_loss = criterion(input=image_logits, target=labels)
		tpi_loss = criterion(input=text_logits, target=labels)
		cmpc_loss = ipt_loss + tpi_loss
		'''
		cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
		#cmpc_loss = - (F.log_softmax(image_logits, dim=1) + F.log_softmax(text_logits, dim=1)) * labels_onehot
		#cmpc_loss = torch.mean(torch.sum(cmpc_loss, dim=1))
		# classification accuracy for observation
		image_pred = torch.argmax(image_logits, dim=1)
		text_pred = torch.argmax(text_logits, dim=1)

		image_precision = torch.mean((image_pred == labels).float())
		text_precision = torch.mean((text_pred == labels).float())

		return cmpc_loss, image_precision, text_precision


	def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
		"""
		Cross-Modal Projection Matching Loss(CMPM)
		:param image_embeddings: Tensor with dtype torch.float32
		:param text_embeddings: Tensor with dtype torch.float32
		:param labels: Tensor with dtype torch.int32
		:return:
			i2t_loss: cmpm loss for image projected to text
			t2i_loss: cmpm loss for text projected to image
			pos_avg_sim: average cosine-similarity for positive pairs
			neg_avg_sim: averate cosine-similarity for negative pairs
		"""

		batch_size = image_embeddings.shape[0]
		labels_reshape = torch.reshape(labels, (batch_size, 1))
		labels_dist = labels_reshape - labels_reshape.t()
		labels_mask = (labels_dist == 0)
		
		image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
		text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
		image_proj_text = torch.matmul(image_embeddings, text_norm.t().detach())
		text_proj_image = torch.matmul(text_embeddings, image_norm.t().detach())

		# normalize the true matching distribution
		labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)
		# print(labels_mask_norm[0])
		 
		i2t_pred = F.softmax(image_proj_text, dim=1)
		#i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))
		
		t2i_pred = F.softmax(text_proj_image, dim=1)
		#t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
		t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1)- torch.log(labels_mask_norm*self.e + self.epsilon))

		cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

		sim_cos = torch.matmul(image_norm, text_norm.t())

		pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
		neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
		
		return cmpm_loss, pos_avg_sim, neg_avg_sim


	def forward(self, image_embeddings, text_embeddings, labels):
		cmpm_loss = 0.0
		cmpc_loss = 0.0
		image_precision = 0.0
		text_precision = 0.0
		neg_avg_sim = 0.0
		pos_avg_sim =0.0
		if self.CMPM:
			cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
		if self.CMPC:
			cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings, labels)
		
		loss = cmpm_loss + cmpc_loss #
		
		return cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim


# class AverageMeter(object):
#     """
#     Computes and stores the averate and current value
#     Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
#     """
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += n * val
#         self.count += n
#         self.avg = self.sum / self.count


def compute_topk(query, gallery, target_query, target_gallery, k=[1,5, 10], reverse=False):
	result = []
	query = query / query.norm(dim=1,keepdim=True)
	gallery = gallery / gallery.norm(dim=1,keepdim=True)
	sim_cosine = torch.matmul(query, gallery.t())
	result.extend(topk(sim_cosine, target_gallery, target_query, k=k))
	if reverse:
		result.extend(topk(sim_cosine , target_query, target_gallery, k=k, dim=0))
	return result


def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
	result = []
	maxk = max(k)
	size_total = len(target_query)
	_, pred_index = sim.topk(maxk, dim, True, True)
	pred_labels = target_gallery[pred_index]
	if dim == 1:
		pred_labels = pred_labels.t()
	correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

	for topk in k:
		#correct_k = torch.sum(correct[:topk]).float()
		correct_k = torch.sum(correct[:topk], dim=0)
		correct_k = torch.sum(correct_k > 0).float()
		result.append(correct_k * 100 / size_total)
	return result

def compute_topk_same(query, gallery, target_query, target_gallery, k=[1,5, 10], reverse=False):
	result = []
	query = query / query.norm(dim=1,keepdim=True)
	gallery = gallery / gallery.norm(dim=1,keepdim=True)
	sim_cosine = torch.matmul(query, gallery.t())
	result.extend(topk_same(sim_cosine, target_gallery, target_query, k=k))
	if reverse:
		result.extend(topk_same(sim_cosine , target_query, target_gallery, k=k, dim=0))
	return result


def topk_same(sim, target_gallery, target_query, k=[1,10], dim=1):
	result = []
	maxk = max(k)
	size_total = len(target_query)
	_, pred_index = sim.topk(maxk, dim, True, True)
	pred_labels = target_gallery[pred_index]
	if dim == 1:
		pred_labels = pred_labels.t()
	correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

	for topk in k:
		#correct_k = torch.sum(correct[:topk]).float()
		correct_k = torch.sum(correct[1:topk+1], dim=0)
		correct_k = torch.sum(correct_k > 0).float()
		result.append(correct_k * 100 / size_total)
	return result


def compute_topk_2(text_query, image_gallery, text_pid, image_pid, topk=[1, 5, 10]):
	image_gallery = F.normalize(image_gallery, p=2, dim=1)
	text_query = F.normalize(text_query, p=2, dim=1)
	similarity = torch.matmul(text_query, image_gallery.t())

	return rank(similarity, text_pid, image_pid, max(topk))




def rank(similarity, q_pids, g_pids, max_rank=10):
	num_q, num_g = similarity.size()
	indices = torch.argsort(similarity, dim=1, descending=True)
	matches = g_pids[indices].eq(q_pids.view(-1, 1))

	# compute cmc curve for each query
	all_cmc = [] # number of valid query
	for q_idx in range(num_q):
		# compute cmc curve
		# binary vector, positions with value 1 are correct matches
		orig_cmc = matches[q_idx]
		cmc = orig_cmc.cumsum(0)
		cmc[cmc > 1] = 1
		all_cmc.append(cmc[:max_rank])

	all_cmc = torch.stack(all_cmc).float()
	all_cmc = all_cmc.sum(0) / num_q
	return all_cmc


def _tensor_dot(x, y):
	res = torch.einsum("ij,kj->ik", (x, y))
	return res

def _mobius_addition_batch(x, y, c):
	xy = _tensor_dot(x, y)  # B x C
	x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
	y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
	num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
	num = num.unsqueeze(2) * x.unsqueeze(1)
	num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
	denom_part1 = 1 + 2 * c * xy  # B x C
	denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
	denom = denom_part1 + denom_part2
	res = num / (denom.unsqueeze(2) + 1e-5)
	return res

def _dist_matrix(x, y, c):
	sqrt_c = c ** 0.5
	return (
		2
		/ sqrt_c
		* artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
	)

def dist_matrix(x, y, c=1.0):
	c = torch.as_tensor(c).type_as(x)
	return _dist_matrix(x, y, c)



class Artanh(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		x = x.clamp(-1 + 1e-5, 1 - 1e-5)
		ctx.save_for_backward(x)
		res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
		return res

	@staticmethod
	def backward(ctx, grad_output):
		(input,) = ctx.saved_tensors
		return grad_output / (1 - input ** 2)

def artanh(x):
	return Artanh.apply(x)