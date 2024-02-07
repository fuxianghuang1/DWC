# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main method to train the DWC model."""

# !/usr/bin/python
import json
import os 
import argparse
import logging
import sys
import gc
import time
import PIL.Image
import datasets
import Our_global as composition_g
import Our_local as composition_l
import numpy as np
from torch.autograd import Variable
import test2 
import test
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from copy import deepcopy
import socket
import os
from datetime import datetime

from torch.utils.data import dataloader
from torch.cuda.amp import autocast as autocast, GradScaler
import torch.nn.functional as F

torch.set_num_threads(8)

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--dataset', type=str, default='fashionIQ')# fashionIQ fashion200k  css3d shoes
    parser.add_argument('--dataset_path', type=str, default='/media/cqu/D/HFX/datasets/fashionIQ/')
    #'/media/cqu/D/HFX/datasets/CSS'
    #'/media/cqu/D/HFX/datasets/Fashion200k'
    #'/media/cqu/D/HFX/datasets/shoes/'
    #'/media/cqu/D/HFX/datasets/fashionIQ/'
    parser.add_argument('--model', type=str, default='DWC')# direct_sum Our Our_global Combiner
    parser.add_argument('--backbone', type=str, default='RN50')#'RN50'  'RN101'  'RN50x4'  'RN50x16'  'ViT-B/32'  'ViT-B/16'
    parser.add_argument('--resize_dim', type=int, default=224)#224 288 384
    parser.add_argument('--image_embed_dim', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=599)
    parser.add_argument('--learning_rate', type=float, default=1e-4)#1e-2
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--name', default = 'css3d', help = "data set type")#'all', 'dress', 'shirt', 'toptee'
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--loader_num_workers', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./experiment/')
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--global_model_checkpoint', type=str, default='/media/dlc/ssd_data/HFX/new_31/experiment/dress/best_global_checkpoint.pth')
    parser.add_argument('--local_model_checkpoint', type=str, default='/media/dlc/ssd_data/HFX/new_31/experiment/dress/best_local_checkpoint.pth')
    args = parser.parse_args()
    return args

def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset ', opt.dataset)
    resize_dim = opt.resize_dim
    if opt.dataset == 'css3d':
        trainset = datasets.CSSDataset(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]))
        testset = datasets.CSSDataset(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]))    
    elif opt.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.CenterCrop(resize_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]))
        testset = datasets.Fashion200k(
            path=opt.dataset_path,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.CenterCrop(resize_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]))
            
    elif opt.dataset == 'fashionIQ':
        trainset = datasets.FashionIQ(
            path = opt.dataset_path,
            name = opt.name,
            split = 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.CenterCrop(resize_dim),
                torchvision.transforms.RandomHorizontalFlip(),              
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

            ]))
        testset = datasets.FashionIQ(
            path = opt.dataset_path,
            name = opt.name,
            split = 'val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.CenterCrop(resize_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]))
            
    elif opt.dataset == 'shoes':
        trainset = datasets.Shoes(
            path = opt.dataset_path,
            split = 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.CenterCrop(resize_dim),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]))
        testset = datasets.Shoes(
            path = opt.dataset_path,
            split = 'test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_dim,interpolation=PIL.Image.BICUBIC),
                torchvision.transforms.CenterCrop(resize_dim),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])) 
            
    else:
        print('Invalid dataset', opt.dataset)
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset


def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""
    print("Creating model and optimizer for", opt.model)
    text_embed_dim = opt.image_embed_dim
    if opt.model == 'model': 
        global_model = composition_g.Our_global1(texts,
                                             image_embed_dim=opt.image_embed_dim,
                                             text_embed_dim=text_embed_dim,
                                             backbone = opt.backbone) 
        local_model = composition_l.Our_local1(texts,
                                             image_embed_dim=opt.image_embed_dim,
                                             text_embed_dim=text_embed_dim,
                                             backbone = opt.backbone)

    elif opt.model == 'DWC':    
        global_model = composition_g.Our_global6(texts,
                                             image_embed_dim=opt.image_embed_dim,
                                             text_embed_dim=text_embed_dim,
                                             backbone = opt.backbone) 
        local_model = composition_l.Our_local3(texts,
                                             image_embed_dim=opt.image_embed_dim,
                                             text_embed_dim=text_embed_dim,
                                             backbone = opt.backbone)
                                           
                                                 
    global_model =  global_model.cuda()
    local_model = local_model.cuda()

    '''optimizer = torch.optim.SGD(params,
                                lr=opt.learning_rate,
                                momentum=0.9,
                                weight_decay=opt.weight_decay)'''

                                
    global_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()),
                                lr=opt.learning_rate,
                                eps=opt.eps,
                                weight_decay=opt.weight_decay)  
    local_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()),
                                lr=opt.learning_rate,
                                eps=opt.eps,
                                weight_decay=opt.weight_decay)                                 

    return global_model, local_model, global_optimizer, local_optimizer


def train_loop(opt, loss_weights, logdir, trainset, testset, global_model, local_model, global_optimizer, local_optimizer):
    """Function for train loop"""
    print('Begin training')
    scaler = GradScaler()
    losses_tracking = {}
    epoch = -1
    tic = time.time()
    all_Rmean_max = 0.0
    loc_Rmean_max = 0.0
    while  epoch < opt.epoch: 
        epoch += 1

        # show/log stats
        print('The epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                              4), opt.dataset)
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print('    Loss', loss_name, round(avg_loss, 4))

        if epoch % 1 == 0:
            gc.collect()

        # test
        if epoch % 1 == 0:# and epoch > 0:        
            #test on the all model 
            testsall = []
            all_Rmean = 0.0
            all_a = 0.0
            testsloc = []
            loc_Rmean = 0.0
            loc_a = 0.0            
            #for name, dataset in [('train', trainset), ('test', testset)]:
            for name, dataset in [('test', testset)]:
                if opt.dataset == 'fashionIQ' or opt.dataset == 'shoes':
                    tall = test2.fiq_test(opt, global_model, local_model, dataset)
                    tloc = test.fiq_test(opt, local_model, dataset)
                else:
                    tall = test2.test(opt, global_model, local_model, dataset)
                    tloc = test.test(opt, local_model, dataset)
                testsall += [(name + ' ' + metric_name, metric_value)
                          for metric_name, metric_value in tall]
                testsloc += [(name + ' ' + metric_name, metric_value)
                          for metric_name, metric_value in tloc]                          
            for metric_name, metric_value in testsall:
                all_a += 1
                all_Rmean += metric_value 
                print('    all ', metric_name, round(metric_value, 4))
            all_Rmean /= all_a
            print('The epoch', epoch, 'all_Rmean = ', round(all_Rmean, 4))
            all_is_best = all_Rmean > all_Rmean_max
            if all_is_best:
                all_Rmean_max = all_Rmean
                print ('save all model and all_Rmean_max = ', round(all_Rmean_max, 4))
                print(testsall)
                best_json_path_combine = os.path.join(
                    logdir, "metrics_best_all.txt"
                )
                test_metrics = {}
                for metric_name, metric_value in testsall:
                    test_metrics[metric_name] = metric_value
                save_dict_to_json(test_metrics, best_json_path_combine)
                
                # save checkpoint
                torch.save({
                    'model_state_dict': local_model.state_dict(),
                },
                    logdir + '/all_local_checkpoint.pth')
                torch.save({
                    'model_state_dict': global_model.state_dict(),
                },
                    logdir + '/all_global_checkpoint.pth') 

            for metric_name, metric_value in testsloc:
                loc_a += 1
                loc_Rmean += metric_value 
                print('    local ', metric_name, round(metric_value, 4))
            loc_Rmean /= loc_a
            print('The epoch', epoch, 'loc_Rmean = ', round(loc_Rmean, 4))
            loc_is_best = loc_Rmean > loc_Rmean_max
            if loc_is_best:
                loc_Rmean_max = loc_Rmean
                print ('save local model and loc_Rmean_max = ', round(loc_Rmean_max, 4))
                print(testsloc)
                best_json_path_combine = os.path.join(
                    logdir, "metrics_best_local.txt"
                )
                test_metrics = {}
                for metric_name, metric_value in testsloc:
                    test_metrics[metric_name] = metric_value
                save_dict_to_json(test_metrics, best_json_path_combine)
                
                # save checkpoint
                torch.save({
                    'model_state_dict': local_model.state_dict(),
                },
                    logdir + '/best_local_checkpoint.pth')                    
        # run training for 1 epoch
        global_model.train()
        local_model.train()
        if opt.dataset == 'fashion200k' or opt.dataset == 'css3d':
            trainloader = trainset.get_loader(
                batch_size=opt.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=opt.loader_num_workers)
        else:        
            trainloader = dataloader.DataLoader(trainset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=opt.loader_num_workers) 
                                            

        def training_1_iter(data):
            if opt.dataset == 'fashion200k':
                assert type(data) is list
                img1 = np.stack([d['source_img_data'] for d in data])
                img1 = torch.from_numpy(img1).float()
                img1 = torch.autograd.Variable(img1).cuda()          

                img2 = np.stack([d['target_img_data'] for d in data])
                img2 = torch.from_numpy(img2).float()
                img2 = torch.autograd.Variable(img2).cuda()
                
                text_query = [str(d['mod']['str']) for d in data]
                
                target_label = np.stack([d['target_label'] for d in data])
                target_label = torch.from_numpy(target_label).float()
                target_label = torch.autograd.Variable(target_label).cuda()
               

            else:
                img1 = data['source_img_data'].cuda()
                img2 = data['target_img_data'].cuda()
                text_query = data['mod']['str']
                target_label = data['target_img_id'].cuda()
                
           
            #train global
            global_optimizer.zero_grad()
            with autocast():
                com_local = local_model.compose_img_text(img1, text_query)["repres"].detach()
                tar_local = local_model.extract_tar_feature(img1)["repres"].detach()
                global_losses = global_model.compute_loss(img1, text_query, img2, com_local, tar_local, target_label, loss_weights)
                
                global_total_loss = sum([
                    loss_weight * loss_value
                    for loss_name, loss_weight, loss_value in global_losses
                ])
            scaler.scale(global_total_loss).backward()
            scaler.step(global_optimizer)
            scaler.update()            
 
            # train local
            local_optimizer.zero_grad()
            with autocast():
                com_global = global_model.compose_img_text(img1, text_query)["repres"].detach()
                tar_global = global_model.extract_tar_feature(img1)["repres"].detach()
                local_losses = local_model.compute_loss(img1, text_query, img2, com_global, tar_global, target_label, loss_weights)
                
                local_total_loss = sum([
                    loss_weight * loss_value
                    for loss_name, loss_weight, loss_value in local_losses
                ]) 
            scaler.scale(local_total_loss).backward()
            scaler.step(local_optimizer)
            scaler.update()
            
                
            losses = global_losses
            assert not torch.isnan(global_total_loss)
            losses += [('global total training loss', None, global_total_loss.item())]

            losses += local_losses
            assert not torch.isnan(local_total_loss)
            losses += [('local total training loss', None, local_total_loss.item())]
            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            training_1_iter(data)

    print('Finished training')

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        
        
def main():
    print('      ------start-----')
    opt = parse_opt()
    
    print('Arguments:')
    #torch.cuda.set_device(opt.choose_device) #bert-serving-start -model_dir /media/dlc/ssd_data/HFX/ComposeAE-master_bert/uncased_L-12_H-768_A-12 -device_map 3
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    loss_weights = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    logdir = os.path.join(opt.log_dir, opt.name)


    trainset, testset = load_dataset(opt)
    global_model, local_model, global_optimizer, local_optimizer = create_model_and_optimizer(opt, [t for t in trainset.get_all_texts()])
    if opt.test_only:
        print('Doing test only')

        global_model_checkpoint = opt.global_model_checkpoint
        checkpoint = torch.load(global_model_checkpoint)
        global_model.load_state_dict(checkpoint['model_state_dict'])

        local_model_checkpoint = opt.local_model_checkpoint
        checkpoint1 = torch.load(local_model_checkpoint)
        local_model.load_state_dict(checkpoint1['model_state_dict'])        
        global_model.eval()
        local_model.eval()
        tests = []
            
        for name, dataset in [('test', testset)]: #[('train', trainset), ('test', testset)]:
            if opt.dataset == 'fashionIQ' or opt.dataset == 'shoes':
                t = test2.fiq_test(opt, global_model, local_model, dataset)
            else:
                t = test2.test(opt, global_model, local_model, dataset)
                #t = test_retrieval.test1(opt, model, dataset)#save top 10 images
            tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
        for metric_name, metric_value in tests:
            print('    test all   ', metric_name, round(metric_value, 4))   
            

        return 0
  
        
    train_loop(opt, loss_weights, logdir, trainset, testset, global_model, local_model, global_optimizer, local_optimizer)



if __name__ == '__main__':
    main()
