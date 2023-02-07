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
from graphs.models.discriminator import Discriminator
from datasets.cuhk_pedes import CUHKLoader

from random import random

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, Loss, Loss_Cross_Project,constraints_loss, compute_topk, compute_topk_2
# from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class JointFeatureAdv(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # define models
        self.model = Model(self.config)
        self.discriminator = Discriminator(self.config)

        # define data_loader
        self.data_loader = CUHKLoader(self.config)

        # define loss
        self.loss = Loss(self.config)
        self.loss_bce = nn.BCEWithLogitsLoss()

        # define optimizer
        param_image = self.model.image_model.parameters()
        param_text = self.model.bilstm.parameters()
        param_id_classifier = self.loss.parameters()
        param_discriminator = self.discriminator.parameters()
#         param_image_projection = self.model.conv_images.parameters()
#         param_text_projection = self.model.conv_text.parameters()
    
#         cnn_params = list(map(id, network.module.image_model.parameters()))
#         other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
#         other_params = list(other_params)
        
#         param_image = list(map(id, self.model.image_model.parameters()))
#         param_other = filter(lambda p: id(p) not in param_image, self.model.parameters())
#         param_other = list(param_other)

        parameters = [{'params': param_image, 'weight_decay':self.config.weight_decay}, {'params': param_text}, {'params': param_id_classifier}]
#                      {'params': param_image_projection}, {'params': param_text_projection}]

        
        self.optimizer = optim.Adam(parameters, lr=self.config.learning_rate)

        self.optimizer_disc = optim.Adam(param_discriminator, lr=self.config.learning_rate)
#         optim.SGD(parameters, lr=self.config.learning_rate, momentum=0.9)
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
            self.loss_bce = self.loss_bce.to(self.device)
            self.discriminator = self.discriminator.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
#             print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        if self.config.pretrained and self.config.mode == 'train' and self.current_epoch == 0:
            self.load_pretrained('model_best.pth.tar')
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.exp_name)

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
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.R1_text_to_image = checkpoint['best_R1_t2i']
            self.R5_text_to_image = checkpoint['best_R5_t2i']
            self.R10_text_to_image = checkpoint['best_R10_t2i']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def load_pretrained(self, file_name):
        project_path = "/home/gp104/projects/def-josedolz/gp104/text-person/experiments/"

        filename = os.path.join(project_path, self.config.pretrained_exp_name,"checkpoints",file_name)
        # filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading pretrained model '{}'".format(filename))
            checkpoint = torch.load(filename)

            # self.current_epoch = checkpoint['epoch']
            # self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.loss.load_state_dict(checkpoint['loss_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.R1_text_to_image = checkpoint['best_R1_t2i']
            self.R5_text_to_image = checkpoint['best_R5_t2i']
            self.R10_text_to_image = checkpoint['best_R10_t2i']

            self.logger.info("Pretrained model loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No Pretrained model exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
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
            'discriminator_state_dict': self.discriminator.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
            'optimizer': self.optimizer.state_dict(),
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


    def ones_target(self, size):
#     Instead of having 1 as the target, one-sided label smoothing replaces the target witth 0.9 
#     data = Variable(torch.ones(size, 1).type(dtype))
        data = Variable(torch.FloatTensor(size, 1).uniform_(0.8, 1.2))
        return data

    def zeros_target(self, size):
        data = Variable(torch.FloatTensor(size, 1).uniform_(0.0, 0.3))
        return data

    def train_discriminator(self, optimizer, image_embedding, text_embedding):
        N = image_embedding.size(0)
        optimizer.zero_grad()


        image_label = self.ones_target(N)
        text_label = self.zeros_target(N)

        if random() >= 0.8:
            image_label = self.zeros_target(N)
            text_label = self.ones_target(N)

        
    #     Train the discriminator on the real data
        prediction_image = self.discriminator(image_embedding)
        error_image = self.loss_bce(prediction_image, image_label.cuda())
        error_image.backward()
        
    #     Now train it on the generated data
        prediction_text = self.discriminator(text_embedding)
        error_text = self.loss_bce(prediction_text, text_label.cuda())
        error_text.backward()

        optimizer.step()

        return (error_image + error_text)/2, prediction_image, prediction_text

    def generator_loss(self, image_embedding, text_embedding):
        N = image_embedding.size(0)
        # optimizer.zero_grad()
        
    #     Run the generated data through the discriminator
        prediction_image = self.discriminator(image_embedding)

    #     Train the generator with the flipped targets, i.e. the target is 0.9
        error_image = self.loss_bce(prediction_image, torch.zeros(N,1).cuda())

        prediction_text = self.discriminator(text_embedding)
        
        error_text = self.loss_bce(prediction_text,torch.ones(N,1).cuda())     
        # error.backward()

        

        return error_image, error_text


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

        if self.config.balanced_loader:
            data_loader = self.data_loader.train_loader_balanced
        else:
            data_loader = self.data_loader.train_loader


        self.model.train()
        for step, (images, captions, labels, captions_length) in enumerate(data_loader):

            N,C,H,W = images.size()


            images, captions, labels, captions_length = images.to(self.device), captions.to(self.device), labels.to(self.device), captions_length.to(self.device)

            
                
            
            
            image_embeddings, text_embeddings = self.model(images, captions, captions_length)

            disc_loss, _, _ = self.train_discriminator(self.optimizer_disc,image_embeddings.detach(), text_embeddings.detach())
            
            self.optimizer.zero_grad()
            cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim = self.loss(image_embeddings, text_embeddings, labels)
            image_gen_loss, text_gen_loss = self.generator_loss(image_embeddings, text_embeddings)

            generator_loss = (image_gen_loss + text_gen_loss)/2

            loss = loss + image_gen_loss + text_gen_loss

            # loss = F.nll_loss(output, target)
            # if (self.config.constraints_images or self.config.constraints_text) and step == self.data_loader.train_iterations - 1:
            #     con_images, con_text = constraints_loss(self.data_loader.train_loader, self.model, self.config)
            #     loss += (con_images + con_text)
            #     print('epoch:{}, step:{}, con_images:{:.3f}, con_text:{:.3f}'.format(epoch, step, con_images, con_text))
            
            loss.backward()
            self.optimizer.step()
            if step % 10 == 0:
                print('epoch:{}, step:{}, cmpm_loss:{:.3f}, cmpc_loss:{:.3f}'.format(self.current_epoch, step, cmpm_loss, cmpc_loss))
                self.summary_writer.add_scalars("GAN_loss",{"discriminator_loss":disc_loss.item(), "generator_loss":generator_loss.item()}
                        , self.current_iteration)


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
        max_size_image = self.config.batch_size * len(self.data_loader.test_loader_image)
        max_size_text = self.config.batch_size * len(self.data_loader.test_loader_caption)
        images_bank = torch.zeros((max_size_image, self.config.feature_size)).cuda()
        text_bank = torch.zeros((max_size_text, self.config.feature_size)).cuda()
        images_labels_bank = torch.zeros(max_size_image).cuda()
        text_labels_bank = torch.zeros(max_size_text).cuda()

        

        with torch.no_grad():
            index = 0
            for step, (images, labels) in enumerate(self.data_loader.test_loader_image):
                images = images.cuda()
                # captions = captions.cuda()

                # captions_length = captions_length.cuda()

                interval = images.shape[0]
                image_embeddings= self.model.image_model(images)
                images_bank[index: index + interval] = image_embeddings
                # text_bank[index: index + interval] = text_embeddings
                images_labels_bank[index: index + interval] = labels

                index = index + interval

            images_bank = images_bank[:index]
            # text_bank = text_bank[:index]
            images_labels_bank = images_labels_bank[:index]

            index = 0
            for step, (captions, labels,captions_length) in enumerate(self.data_loader.test_loader_caption):
                # images = images.cuda()
                captions = captions.cuda()

                captions_length = captions_length.cuda()

                interval = captions.shape[0]
                text_embeddings = self.model.bilstm(captions, captions_length)
                # images_bank[index: index + interval] = image_embeddings
                text_bank[index: index + interval] = text_embeddings
                text_labels_bank[index: index + interval] = labels

                index = index + interval


            text_bank = text_bank[:index]
            text_labels_bank = text_labels_bank[:index]

            ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, images_labels_bank, text_labels_bank, [1,5,10], True)
            
            print("ac_top1_i2t: {:.3f}".format(ac_top1_i2t))
            print("ac_top1_t2i: {:.3f}".format(ac_top1_t2i))
            print("ac_top5_i2t: {:.3f}".format(ac_top5_i2t))
            print("ac_top5_t2i: {:.3f}".format(ac_top5_t2i))
            print("ac_top10_i2t: {:.3f}".format(ac_top10_i2t))
            print("ac_top10_t2i: {:.3f}".format(ac_top10_t2i))

            # t2i_cmc = compute_topk_2(text_bank, images_bank, labels_bank, labels_bank)

            # print('R@1: {:.4f}%, R@5: {:.4f}%, R@10: {:.4f}%'.format(
            # t2i_cmc[1 - 1] * 100,
            # t2i_cmc[5 - 1] * 100,
            # t2i_cmc[10 - 1] * 100))

            # t2i_cmc = compute_topk_2(text_bank, images_bank, labels_bank, labels_bank)

            # print('R@1: {:.4f}%, R@5: {:.4f}%, R@10: {:.4f}%'.format(
            # t2i_cmc[0 - 1] * 100,
            # t2i_cmc[5 - 1] * 100,
            # t2i_cmc[10 - 1] * 100))


            
            return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i,ac_top10_t2i






    def test(self):

        # ac_i2t_top1_best = 0.0
        # ac_i2t_top10_best = 0.0
        # ac_t2i_top1_best = 0.0
        # ac_t2i_top10_best = 0.0

        
        # self.model.eval()
        # max_size = self.config.batch_size * len(self.data_loader.test_loader)
        # images_bank = torch.zeros((max_size, self.config.feature_size)).cuda()
        # text_bank = torch.zeros((max_size, self.config.feature_size)).cuda()
        # labels_bank = torch.zeros(max_size).cuda()

        # index = 0

        # with torch.no_grad():
        #     for step, (images, captions, labels, captions_length) in enumerate(self.data_loader.test_loader):
        #         images = images.cuda()
        #         captions = captions.cuda()

        #         captions_length = captions_length.cuda()

        #         interval = images.shape[0]
        #         image_embeddings, text_embeddings = self.model(images, captions, captions_length)
        #         images_bank[index: index + interval] = image_embeddings
        #         text_bank[index: index + interval] = text_embeddings
        #         labels_bank[index: index + interval] = labels

        #         index = index + interval

        #     images_bank = images_bank[:index]
        #     text_bank = text_bank[:index]
        #     labels_bank = labels_bank[:index]

        #     ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,10, 20], True)
            
        #     print("ac_top1_i2t: {:.3f}".format(ac_top1_i2t))
        #     print("ac_top1_t2i: {:.3f}".format(ac_top1_t2i))
        #     print("ac_top5_i2t: {:.3f}".format(ac_top5_i2t))
        #     print("ac_top5_t2i: {:.3f}".format(ac_top5_t2i))
        #     print("ac_top10_i2t: {:.3f}".format(ac_top10_i2t))
        #     print("ac_top10_t2i: {:.3f}".format(ac_top10_t2i))
        

        ac_i2t_top1_best = 0.0
        ac_i2t_top10_best = 0.0
        ac_t2i_top1_best = 0.0
        ac_t2i_top10_best = 0.0

        
        self.model.eval()
        max_size_image = self.config.batch_size * len(self.data_loader.test_loader_image)
        max_size_text = self.config.batch_size * len(self.data_loader.test_loader_caption)
        images_bank = torch.zeros((max_size_image, self.config.feature_size)).cuda()
        text_bank = torch.zeros((max_size_text, self.config.feature_size)).cuda()
        images_labels_bank = torch.zeros(max_size_image).cuda()
        text_labels_bank = torch.zeros(max_size_text).cuda()

        

        with torch.no_grad():
            index = 0
            for step, (images, labels) in enumerate(self.data_loader.test_loader_image):
                images = images.cuda()
                # captions = captions.cuda()

                # captions_length = captions_length.cuda()

                interval = images.shape[0]
                image_embeddings= self.model.image_model(images)
                images_bank[index: index + interval] = image_embeddings
                # text_bank[index: index + interval] = text_embeddings
                images_labels_bank[index: index + interval] = labels

                index = index + interval

            images_bank = images_bank[:index]
            # text_bank = text_bank[:index]
            images_labels_bank = images_labels_bank[:index]

            index = 0
            for step, (captions, labels,captions_length) in enumerate(self.data_loader.test_loader_caption):
                # images = images.cuda()
                captions = captions.cuda()

                captions_length = captions_length.cuda()

                interval = captions.shape[0]
                text_embeddings = self.model.bilstm(captions, captions_length)
                # images_bank[index: index + interval] = image_embeddings
                text_bank[index: index + interval] = text_embeddings
                text_labels_bank[index: index + interval] = labels

                index = index + interval


            text_bank = text_bank[:index]
            text_labels_bank = text_labels_bank[:index]

            ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, images_labels_bank, text_labels_bank, [1,5,10], True)
            
            print("ac_top1_i2t: {:.3f}".format(ac_top1_i2t))
            print("ac_top1_t2i: {:.3f}".format(ac_top1_t2i))
            print("ac_top5_i2t: {:.3f}".format(ac_top5_i2t))
            print("ac_top5_t2i: {:.3f}".format(ac_top5_t2i))
            print("ac_top10_i2t: {:.3f}".format(ac_top10_i2t))
            print("ac_top10_t2i: {:.3f}".format(ac_top10_t2i))

            # t2i_cmc = compute_topk_2(text_bank, images_bank, text_labels_bank, images_labels_bank)

            # print('R@1: {:.4f}%, R@5: {:.4f}%, R@10: {:.4f}%'.format(
            # t2i_cmc[1 - 1] * 100,
            # t2i_cmc[5 - 1] * 100,
            # t2i_cmc[10 - 1] * 100))


            
            # return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i,ac_top10_t2i



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
