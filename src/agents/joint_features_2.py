import os
import numpy as np
from tqdm import tqdm
import shutil
import random
import json

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from graphs.models.model import Model
from datasets.cuhk_pedes import CUHKLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, Loss, constraints_loss, compute_topk
from utils.train_utils import MultipleOptimizers, MultipleSchedulers
# from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class JointFeature2(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # define models
        self.model = Model(self.config)

        # define data_loader
        self.data_loader = CUHKLoader(self.config)

        # define loss
        self.loss = Loss(self.config)

        # define optimizer
        param_image = self.model.image_model.parameters()
        param_text = self.model.bilstm.parameters()
        param_id_classifier = self.loss.parameters()
#         param_image_projection = self.model.conv_images.parameters()
#         param_text_projection = self.model.conv_text.parameters()
    
#         cnn_params = list(map(id, network.module.image_model.parameters()))
#         other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
#         other_params = list(other_params)
        
#         param_image = list(map(id, self.model.image_model.parameters()))
#         param_other = filter(lambda p: id(p) not in param_image, self.model.parameters())
#         param_other = list(param_other)

        # parameters = [{'params': param_image, 'weight_decay':self.config.weight_decay}, {'params': param_text}, {'params': param_id_classifier}]
#                      {'params': param_image_projection}, {'params': param_text_projection}]

        # 
        # self.optimizer = optim.Adam(parameters, lr=self.config.learning_rate)

        parameters = [{'params': param_image, 'weight_decay':self.config.weight_decay}, {'params': param_text}, {'params': param_id_classifier}]
#                      {'params': param_image_projection}, {'params': param_text_projection}]

        
        self.optimizer = optim.Adam(parameters, lr=self.config.learning_rate)
#         optim.SGD(parameters, lr=self.config.learning_rate, momentum=0.9)
        self.scheduler_image = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_image, 'max', factor=0.1, patience=10,min_lr=2e-6)
        self.scheduler_text = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_text, 'max', factor=0.1, patience=10,min_lr=2e-6)
        self.scheduler_id_classifier = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_id_classifier, 'max', factor=0.1, patience=10,min_lr=2e-6)

        
#         

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.R1_text_to_image = 0
        self.R5_text_to_image = 0
        self.R10_text_to_image = 0
        self.AP50_text_to_image = 0
        self.best_loss = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
#             print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        self.multiple_optimizers = MultipleOptimizers(self.optimizer_image, self.optimizer_text, self.optimizer_id_classifier)
        self.multiple_schedulers = MultipleSchedulers(self.scheduler_image, self.scheduler_text, self.scheduler_id_classifier)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.loss.load_state_dict(checkpoint['loss_state_dict'])

            self.optimizer_image.load_state_dict(checkpoint['optimizer_image'])
            self.optimizer_text.load_state_dict(checkpoint['optimizer_text'])
            self.optimizer_id_classifier.load_state_dict(checkpoint['optimizer_id_classifier'])

            self.scheduler_image.load_state_dict(checkpoint['scheduler_image'])
            self.scheduler_text.load_state_dict(checkpoint['scheduler_text'])
            self.scheduler_id_classifier.load_state_dict(checkpoint['scheduler_id_classifier'])

            self.R1_text_to_image = checkpoint['best_R1_t2i']
            self.R5_text_to_image = checkpoint['best_R5_t2i']
            self.R10_text_to_image = checkpoint['best_R10_t2i']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")


    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
            'optimizer_image': self.optimizer_image.state_dict(),
            'optimizer_text': self.optimizer_text.state_dict(),
            'optimizer_id_classifier': self.optimizer_id_classifier.state_dict(),
            'scheduler_image': self.scheduler_image.state_dict(),
            'scheduler_text': self.scheduler_text.state_dict(),
            'scheduler_id_classifier': self.scheduler_id_classifier.state_dict(),
            'best_R1_t2i': self.R1_text_to_image,
            'best_R5_t2i': self.R5_text_to_image,
            'best_R10_t2i': self.R10_text_to_image,
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.max_epoch + 1):
            self.train_one_epoch()
            ac_top1_i2t, _, _, R1_t2i, R5_t2i , R10_t2i = self.validate()

            is_best = R1_t2i > self.R1_text_to_image
            

            if(is_best):
#                 self.best_loss = R1_t2i
                self.R1_text_to_image = R1_t2i
            
            if R5_t2i > self.R5_text_to_image:
                self.R5_text_to_image = R5_t2i
            if R10_t2i>self.R10_text_to_image:
                self.R10_text_to_image = R10_t2i
                
                
            data = {'R1_T2I': self.R1_text_to_image.item(), 'R5_T2I': self.R5_text_to_image.item(),'R10_T2I': self.R10_text_to_image.item()}    
            
            with open(os.path.join(self.config.project_directory,'experiments',self.config.exp_name,'data.json' ), 'w') as fp:
                json.dump(data, fp)

                
            self.save_checkpoint(is_best=is_best)
            self.multiple_schedulers.step(R1_t2i)

            self.current_epoch += 1
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        batch_time = AverageMeter()
        train_loss = AverageMeter()
        image_pre = AverageMeter()
        text_pre = AverageMeter()



        self.model.train()
        for step, (images, captions, labels, captions_length) in enumerate(self.data_loader.train_loader):

            N,C,H,W = images.size()


            images, captions, labels, captions_length = images.to(self.device), captions.to(self.device), labels.to(self.device), captions_length.to(self.device)
            self.multiple_optimizers.zero_grad()
            
            image_embeddings, text_embeddings = self.model(images, captions, captions_length)
            
            cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_arg_sim = self.loss(image_embeddings, text_embeddings, labels)

            # loss = F.nll_loss(output, target)
            # if (self.config.constraints_images or self.config.constraints_text) and step == self.data_loader.train_iterations - 1:
            #     con_images, con_text = constraints_loss(self.data_loader.train_loader, self.model, self.config)
            #     loss += (con_images + con_text)
            #     print('epoch:{}, step:{}, con_images:{:.3f}, con_text:{:.3f}'.format(epoch, step, con_images, con_text))
            
            loss.backward()
            self.multiple_optimizers.step()
            if step % 200 == 0:
                print('epoch:{}, step:{}, cmpm_loss:{:.3f}, cmpc_loss:{:.3f}'.format(self.current_epoch, step, cmpm_loss, cmpc_loss))
    #             ac_top1_i2t, _, _, R1_t2i, R5_t2i , R10_t2i = self.validate()

    #             is_best = R1_t2i > self.R1_text_to_image
                

    #             if(is_best):
    # #                 self.best_loss = R1_t2i
    #                 self.R1_text_to_image = R1_t2i
                
    #             if R5_t2i > self.R5_text_to_image:
    #                 self.R5_text_to_image = R5_t2i
    #             if R10_t2i>self.R10_text_to_image:
    #                 self.R10_text_to_image = R10_t2i
                    
                    
    #             data = {'R1_T2I': self.R1_text_to_image.item(), 'R5_T2I': self.R5_text_to_image.item(),'R10_T2I': self.R10_text_to_image.item()}    
                
    #             with open(os.path.join(self.config.project_directory,'experiments',self.config.exp_name,'data.json' ), 'w') as fp:
    #                 json.dump(data, fp)

    #             multiple_schedulers.step(R1_t2i)

                    
    #             self.save_checkpoint(is_best=is_best)


            # if batch_idx % self.config.log_interval == 0:
            #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
            #                100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1
            train_loss.update(loss, N)
            image_pre.update(image_precision, N)
            text_pre.update(text_precision, N)
            
            

        self.logger.info('Epoch:  [{}|{}], train_loss: {:.3f}'.format(self.current_epoch,self.config.max_epoch, train_loss.val))
        self.logger.info('image_precision: {:.3f}, text_precision: {:.3f}'.format(image_pre.val, text_pre.val))

    def validate(self):

        ac_i2t_top1_best = 0.0
        ac_i2t_top10_best = 0.0
        ac_t2i_top1_best = 0.0
        ac_t2i_top10_best = 0.0

        
        self.model.eval()
        max_size = self.config.batch_size * len(self.data_loader.val_loader)
        images_bank = torch.zeros((max_size, self.config.feature_size)).cuda()
        text_bank = torch.zeros((max_size, self.config.feature_size)).cuda()
        labels_bank = torch.zeros(max_size).cuda()

        index = 0

        with torch.no_grad():
            for step, (images, captions, labels, captions_length) in enumerate(self.data_loader.test_loader):
                images = images.cuda()
                captions = captions.cuda()

                captions_length = captions_length.cuda()

                interval = images.shape[0]
                image_embeddings, text_embeddings = self.model(images, captions, captions_length)
                images_bank[index: index + interval] = image_embeddings
                text_bank[index: index + interval] = text_embeddings
                labels_bank[index: index + interval] = labels

                index = index + interval

            images_bank = images_bank[:index]
            text_bank = text_bank[:index]
            labels_bank = labels_bank[:index]

            ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,10], True)
            
            print("ac_top1_i2t: {:.3f}".format(ac_top1_i2t))
            print("ac_top1_t2i: {:.3f}".format(ac_top1_t2i))
            print("ac_top5_i2t: {:.3f}".format(ac_top5_i2t))
            print("ac_top5_t2i: {:.3f}".format(ac_top5_t2i))
            print("ac_top10_i2t: {:.3f}".format(ac_top10_i2t))
            print("ac_top10_t2i: {:.3f}".format(ac_top10_t2i))
            
            return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i,ac_top10_t2i






        """
        One cycle of model validation
        :return:
        """
        # self.model.eval()
        # test_loss = 0
        # correct = 0
        # with torch.no_grad():
        #     for data, target in self.data_loader.test_loader:
        #         data, target = data.to(self.device), target.to(self.device)
        #         output = self.model(data)
        #         test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        #         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #         correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(self.data_loader.test_loader.dataset)
        # self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(self.data_loader.test_loader.dataset),
        #     100. * correct / len(self.data_loader.test_loader.dataset)))




        pass
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
# 		for i in range(self.num_models):
        self.save_checkpoint()

# 		self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir,"all_scalars.json"))
# 		# export_jsondump(self.summary_writer)
# 		self.summary_writer.flush()
# 		self.summary_writer.close()
