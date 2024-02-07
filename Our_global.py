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

"""CLIP stream for Text and Image Composition."""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


import clip
from torch.autograd import Variable

from metric_loss import TripletLoss,CircleLoss
#from blocks import *

#model_path = '../CLIP-main/pretrainedmodels/RN50.pt'  #RN50  RN101  RN50x4  RN50x16  ViT-B/32  ViT-B/16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_pretrian_model(model_path):
    model, preprocess = clip.load(model_path, device=device, jit=False)  # jit=false
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess
        
class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super().__init__()
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))


    def extract_img_feature(self, imgs):
        raise NotImplementedError
        
    def extract_text_feature(self, text_query):
        raise NotImplementedError

    def compose_img_text(self, imgs, text_query):
        raise NotImplementedError        
    
    def compute_loss(self,
                     imgs_query,
                     text_query,
                     imgs_target,
                     com_local,
                     tar_local,
                     target_label,
                     loss_weights):
        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        com_img = dct_with_representations["repres"]
        fuseimg = dct_with_representations["fuseimg"]
        fusetxt = dct_with_representations["fusetxt"]
        pre_dynamic_scalar = dct_with_representations["dynamic_scalar"]


        ref_clip, ref_img = self.extract_img_feature(imgs_query)
        tar_clip, tar_img = self.extract_img_feature(imgs_target)
        
        txt_clip, txt_fea = self.extract_text_feature(text_query)
        

        assert (com_img.shape[0] == tar_img.shape[0] and
                com_img.shape[1] == tar_img.shape[1])
                
        all_loss = []
        
        
        compute_contrastive_loss = self.compute_contrastive_loss_(com_img, tar_img, target_label) 
        all_loss += [('global contrastive loss', loss_weights[1], compute_contrastive_loss)]

        
        mutual_learning_loss = self.mutual_learning(com_img, tar_img, com_local, tar_local) 
        all_loss +=[('global mutual_learning loss', loss_weights[3], mutual_learning_loss)]
        
        
        return all_loss
           
    def compute_contrastive_loss_(self, com, img, labels):  
        alpha  = 0.2  
        temp   = 0.07     

        labels = labels.view(-1,1)
        pos = torch.eq(labels, labels.t()).float()

        sim_labels = pos / pos.sum(1,keepdim=True) 
        sim_img = img @ img.t() / temp
        sim_targets = alpha * F.softmax(sim_img, dim=1) + (1 - alpha) * sim_labels
        sim_predict = com @ img.t() / temp
        loss = -torch.sum(F.log_softmax(sim_predict, dim=1)*sim_targets,dim=1).mean()
        return loss  
        
    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))         
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()

        loss = F.cross_entropy(self.loss_weight * x, labels)   # loss_weight temperature
        return loss
        
    def compute_triplet_loss_(self, com_img, tar_img):
        x = torch.mm(com_img, tar_img.transpose(0, 1))
        labels = torch.tensor(range(com_img.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()      
        labels2 = torch.cat([labels,labels])
        features = torch.cat([com_img,tar_img])
        tri_loss = TripletLoss(margin = 0.3, normalize_feature = False, hard_mining = True, use_cosine_dist = False)
        loss = tri_loss(features,labels2)
        return loss
        
    def compute_l2(self, x1, x2):
        l2_loss = torch.nn.MSELoss().cuda()
        return l2_loss(x1, x2)
        
    def mutual_learning(self, com_img1, tar_img1, com_img2, tar_img2):
        #com_img1 = F.normalize(com_img1, p=2, dim=-1)
        #com_img2 = F.normalize(com_img2, p=2, dim=-1)
        #tar_img1 = F.normalize(tar_img1, p=2, dim=-1)
        #tar_img2 = F.normalize(tar_img2, p=2, dim=-1)
        x1 = 10.0 * torch.mm(com_img1, tar_img1.transpose(0, 1)) 
        x2 = 10.0 * torch.mm(com_img2, tar_img2.transpose(0, 1))

        log_soft_x1 = F.log_softmax(x1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
        return kl
        
class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""

    def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
        super().__init__()

        if backbone == 'RN50':
           model, preprocess = clip.load('RN50', device=device, jit=False)#out_dim=1024 input_image:224
        elif backbone =='RN101':   
           model, preprocess = clip.load('RN101', device=device, jit=False)#out_dim=512 input_image:224
        elif backbone =='RN50x4':   
           model, preprocess = clip.load('RN50x4', device=device, jit=False)#out_dim=640 input_image:288
        elif backbone =='RN50x16':   
           model, preprocess = clip.load('RN50x16', device=device, jit=False)#out_dim=768 input_image:384
        elif backbone =='ViT-B/32':   
           model, preprocess = clip.load('ViT-B/32', device=device, jit=False)#out_dim=512 input_image:224
        elif backbone =='ViT-B/16':   
           model, preprocess = clip.load('ViT-B/16', device=device, jit=False)#out_dim=512 input_image:224
        clip_input_dim = model.visual.input_resolution  
        print('clip_input_dim:', clip_input_dim)        
        clip_feature_dim = model.visual.output_dim
        print('clip_feature_dim:', clip_feature_dim)
          
        # freeze weights
        for param in model.parameters():
            param.requires_grad = False
            
          
        self.img_model = model.encode_image
        self.txt_model = model.encode_text

        self.img_fc = torch.nn.Sequential(
            torch.nn.Linear(clip_feature_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            )
        self.txt_fc = torch.nn.Sequential(
            torch.nn.Linear(clip_feature_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            )
    def extract_img_feature(self, imgs): 
        imgclipfea = self.img_model(imgs)
        imgclipfea = imgclipfea.float()
        imgfeature = self.img_fc(imgclipfea)
        return F.normalize(imgclipfea), F.normalize(imgfeature)
       
    def extract_text_feature(self, text_query):
        text_query = clip.tokenize(text_query).to(device)
        txtclipfea = self.txt_model(text_query)
        txtclipfea = txtclipfea.float()
        txtfeature = self.txt_fc(txtclipfea)
        return F.normalize(txtclipfea), F.normalize(txtfeature)
        
        
    def extract_tar_feature(self, imgs): 
        imgclipfea = self.img_model(imgs)
        imgclipfea = imgclipfea.float()
        imgfeature = self.img_fc(imgclipfea)
        representations = {"repres":F.normalize(imgfeature),
                   }
        return representations


class Our_global(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
    super().__init__(text_query, image_embed_dim, text_embed_dim, backbone)
    self.dynamic = nn.Sequential(nn.Linear(image_embed_dim * 2, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(image_embed_dim, image_embed_dim),
                                            nn.Sigmoid())
    self.dynamic_scalar = nn.Sequential(nn.Linear(image_embed_dim, 1),nn.Sigmoid())
    
    self.combine =  nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(image_embed_dim, image_embed_dim) )
    self.gamma = Inception1(image_embed_dim + text_embed_dim, image_embed_dim)  
    self.beta = Inception1(image_embed_dim + text_embed_dim, image_embed_dim)                               
  def compose_img_text(self, imgs, text_query):
    clip_img, image_features = self.extract_img_feature(imgs)
    clip_txt, text_features = self.extract_text_feature(text_query)
    raw_combined_features = torch.cat((image_features, text_features), -1)
    combined_features = self.gamma(raw_combined_features) * self.combine(raw_combined_features) + self.beta(raw_combined_features)
    dynamic = self.dynamic(raw_combined_features)
    com_fearues = combined_features + dynamic * image_features + (1 - dynamic) * text_features
    representations = {"repres":  F.normalize(com_fearues),  #self.normalization_layer(com_fearues),
                       "dynamic_scalar" :  self.dynamic_scalar(dynamic)
                   }
    return representations
    
class Our_global1(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
    super().__init__(text_query, image_embed_dim, text_embed_dim, backbone)
    self.dynamic = nn.Sequential(nn.Linear(image_embed_dim * 2, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(image_embed_dim, image_embed_dim),
                                            nn.Sigmoid())
    self.dynamic_scalar = nn.Sequential(nn.Linear(image_embed_dim, 1),nn.Sigmoid())   
    self.combine =  nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(image_embed_dim, image_embed_dim) )
    self.gamma = nn.Linear(image_embed_dim, image_embed_dim)  
    self.beta = nn.Linear(image_embed_dim, image_embed_dim)                               
  def compose_img_text(self, imgs, text_query):
    clip_img, image_features = self.extract_img_feature(imgs)
    clip_txt, text_features = self.extract_text_feature(text_query)
    raw_combined_features = torch.cat((image_features, text_features), -1)
    combined_features = self.combine(raw_combined_features)
    dynamic = self.dynamic(raw_combined_features)
    com_fearues = combined_features + dynamic * image_features + (1 - dynamic) * text_features
    com_fearues = self.gamma(com_fearues) * com_fearues + self.beta(com_fearues)
    representations = {"repres":  F.normalize(com_fearues),  #self.normalization_layer(com_fearues),
                       "dynamic_scalar" :  self.dynamic_scalar(dynamic),
                   }
    return representations  


class Our_global6(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
    super().__init__(text_query, image_embed_dim, text_embed_dim, backbone)
    self.atten_I = nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(image_embed_dim, image_embed_dim),
                                            nn.Sigmoid())
    self.atten_T = nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, text_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(text_embed_dim, text_embed_dim),
                                            nn.Sigmoid())                                         
    self.dynamic_scalar = nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(image_embed_dim, 1),
                                            nn.Sigmoid()) 
                         
  def compose_img_text(self, imgs, text_query):
    clip_img, image_features = self.extract_img_feature(imgs)
    clip_txt, text_features = self.extract_text_feature(text_query)
    raw_combined_features = torch.cat((image_features, text_features), -1)
    atten_I = self.atten_I(raw_combined_features)
    atten_T = self.atten_T(raw_combined_features)
    image_features = atten_I * image_features
    text_features = atten_T * text_features 
    new_combined_features = torch.cat((image_features, text_features), -1)
    dynamic = self.dynamic_scalar(new_combined_features)
    com_fearues = dynamic * image_features + (1 - dynamic) * text_features
    
    representations = {"repres":  F.normalize(com_fearues),  #self.normalization_layer(com_fearues),
                       "fuseimg" :  image_features,
                       "fusetxt" :  text_features,                       
                       "dynamic_scalar" :  dynamic,
                   }
    return representations    