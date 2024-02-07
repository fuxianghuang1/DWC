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

"""CNN stream for Text and Image Composition."""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


import clip
from torch.autograd import Variable
from metric_loss import TripletLoss,CircleLoss
#from blocks import *
from newblocks import *
from blocks1207 import *
import text_model 

        
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
                     com_global, 
                     tar_global,
                     target_label,
                     loss_weights):
        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        com_img = dct_with_representations["repres"]
        att_mask = dct_with_representations["attention_mask"]
        ref_tensor = dct_with_representations["img_tensor"]
        local_out = dct_with_representations["local_out"]
        retxtfea = dct_with_representations["retxtfea"]
        txtfea = dct_with_representations["txtfea"]
        re_ref_fea = dct_with_representations["reimgfea"]
        ref_img = dct_with_representations["imgfea"]
        fuseimg = dct_with_representations["fuseimg"]
        fusetxt = dct_with_representations["fusetxt"]
        pre_dynamic_scalar = dct_with_representations["dynamic_scalar"]

        tar_tensor, tar_img = self.extract_img_feature(imgs_target)
        
     
        
        assert (com_img.shape[0] == tar_img.shape[0] and
                com_img.shape[1] == tar_img.shape[1])
                
       
        all_loss = []
        
        compute_contrastive_loss = self.compute_contrastive_loss_(com_img, tar_img, target_label) 
        all_loss += [('local contrastive loss', loss_weights[1], compute_contrastive_loss)]
        
        mutual_learning_loss = self.mutual_learning(com_img, tar_img, com_global, tar_global)
        all_loss +=[('local mutual_learning loss', loss_weights[3], mutual_learning_loss)]

                
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
        # img model
        img_model = torchvision.models.resnet50(pretrained=True)  
        img_model.fc = nn.Sequential()  
        self.img_model = img_model
        self.img_pool = GeM(p=3)
        self.img_fc = nn.Linear(2048, image_embed_dim)
        
        self.txt_model = text_model.TextLSTMModel(
            texts_to_build_vocab=text_query,
            word_embed_dim=512,
            lstm_hidden_dim=text_embed_dim)


    def extract_img_feature(self, imgs): 
        imgs = self.img_model.conv1(imgs)
        imgs = self.img_model.bn1(imgs)
        imgs = self.img_model.relu(imgs)
        imgs = self.img_model.maxpool(imgs)
        imgs = self.img_model.layer1(imgs)
        imgs = self.img_model.layer2(imgs)
        imgs = self.img_model.layer3(imgs)
        imgs = self.img_model.layer4(imgs)
        img_feature = self.img_fc(self.img_pool(imgs))
        return imgs, F.normalize(img_feature)
       
        
    def extract_text_feature(self, text_query): 
        txt, txt_feature = self.txt_model(text_query)
        return txt, F.normalize(txt_feature)
        
    def extract_tar_feature(self, imgs): 
        imgs = self.img_model.conv1(imgs)
        imgs = self.img_model.bn1(imgs)
        imgs = self.img_model.relu(imgs)
        imgs = self.img_model.maxpool(imgs)
        imgs = self.img_model.layer1(imgs)
        imgs = self.img_model.layer2(imgs)
        imgs = self.img_model.layer3(imgs)
        imgs = self.img_model.layer4(imgs)
        img_feature = self.img_fc(self.img_pool(imgs))
        representations = {"repres": F.normalize(img_feature),
                   }
        return representations
              
    
class Our_local(ImgEncoderTextEncoderBase):

  def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
    super().__init__(text_query, image_embed_dim, text_embed_dim, backbone)
    self.attention = OurAttention(channel=2048, len_text = text_embed_dim, alpha = 1)
    self.local_pool = GeM(p=3)
    #self.finalimg = Inception1(in_dim=2048, out_dim=1024)
    #self.finaltxt = Inception1(in_dim=1024, out_dim=1024)
    self.finalimg = nn.Linear(2048, 1024)
    #self.finaltxt = nn.Linear(1024, 1024)
  def compose_img_text(self, imgs, text_query):
    img_tensor, image_features = self.extract_img_feature(imgs) #N,2048,7,7
    txt_tensor, txts = self.extract_text_feature(text_query)
    mask, imgs = self.attention(img_tensor, txts)
    local_feature = self.local_pool(imgs) # N,2048
    local_feature = self.finalimg(local_feature)
    #txts = self.finaltxt(txts)
    #print('local_feature:', local_feature.shape)
    
    representations = {"repres":  F.normalize(local_feature + txts),  #self.normalization_layer(com_fearues),
                       #"attention_mask" :  mask
                   }
    return representations 


    
class Our_local1(ImgEncoderTextEncoderBase):
#new local model4
  def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
    super().__init__(text_query, image_embed_dim, text_embed_dim, backbone)
    self.imgconv = Inception3(in_dim=2048, out_dim=image_embed_dim)
    self.txtconv = Inception3(in_dim=text_embed_dim, out_dim=2048)
    self.softmax = nn.Softmax(dim=1)
    self.imgatt = Inception2(in_dim=image_embed_dim+text_embed_dim, out_dim=2048)
    self.gamma = Inception2(in_dim=2048, out_dim=2048)
    self.beta = Inception2(in_dim=2048, out_dim=2048)
    self.txt_pool = GeM(p=3)
    self.txt_fc = nn.Linear(2048, text_embed_dim)

  def compose_img_text(self, img_query, text_query):
    img_tensor, image_features = self.extract_img_feature(img_query) #N,2048,7,7
    img_embed = self.imgconv(img_tensor)
    #print('imgs:', imgs.shape)
    n,c,h,w = img_embed.size()  #N,1024,7,7
    img_embed = img_embed.view(n,c,h*w) # N,1024,49
    text_embed, txts_features = self.extract_text_feature(text_query)#N,L,1024
    #print('txts:', txts.shape)
    dot_product = torch.bmm(text_embed, img_embed)   # N,L,49
    atten = self.softmax(dot_product / 7.0)
    sentence_cat = []
    for i in range(img_tensor.size(2)*img_tensor.size(3)):
        sentence = torch.sum(text_embed * atten[:,:,i].unsqueeze(-1), dim=1)  # N,1024
        sentence_cat.append(sentence)
    sentence_cat = torch.stack(sentence_cat).permute(1, 2, 0).contiguous()  # N,1024,49
    
    x = torch.cat([img_embed, sentence_cat], dim=1).view(n,-1,h,w)  # N,2048,7,7 
    #print('x:', x.shape)    
    imgatt = self.imgatt(x)
    sentence_cat = sentence_cat.view(n,-1,h,w)#N,1024,7,7 
    txtfea_map = self.txtconv(sentence_cat)#N,2048,7,7 
    
    imgfea_map = imgatt * img_tensor#
 
    x = imgfea_map + txtfea_map + x # N,2048,7,7 
    gamma = self.gamma(x)
    beta = self.beta(x)
    local_out = gamma * x + beta
    
    #print('local_out:', local_out.shape)
    local_feature = F.normalize(self.img_fc(self.img_pool(local_out))) # N,2048
    #print('local_feature:', local_feature.shape)
    retxtfea = F.normalize(self.txt_fc(self.txt_pool(txtfea_map)))
    reimgfea = F.normalize(self.img_fc(self.img_pool(imgfea_map)))
    representations = {"repres":  local_feature,  #self.normalization_layer(com_fearues),
                       "attention_mask" :  imgatt,
                       "img_tensor" :  img_tensor,
                       "local_out" :  local_out,
                       "retxtfea": retxtfea,
                       "txtfea": txts_features,
                       "reimgfea": reimgfea,
                       "imgfea": image_features
                   }
    return representations 


class Our_local3(ImgEncoderTextEncoderBase):
  def __init__(self, text_query, image_embed_dim, text_embed_dim, backbone):
    super().__init__(text_query, image_embed_dim, text_embed_dim, backbone)
    self.imgconv = nn.Sequential(Inception2d(in_dim=2048, out_dim=image_embed_dim), nn.ReLU()) 
    
    self.softmax_i = nn.Softmax(dim=1)
    self.softmax_t = nn.Softmax(dim=-1)
    
    self.imgatt = nn.Sequential(Inception2d(in_dim=image_embed_dim + text_embed_dim, out_dim=2048), nn.Sigmoid())
    self.txtatt = nn.Sequential(Inception1d(in_dim=image_embed_dim + text_embed_dim, out_dim=1024), nn.Sigmoid())
    
    self.txt2img = nn.Sequential(Inception2d(in_dim=1024, out_dim=2048), nn.ReLU())
    self.img2txt = nn.Sequential(Inception1d(in_dim=1024, out_dim=1024), nn.ReLU()) 
    
    self.pool_i = GeM(p=3)
    self.pool_t = GeM(p=3)

    self.fc_txt = nn.Linear(1024, text_embed_dim)
    
    self.fc_ret = nn.Linear(2048, image_embed_dim)
    self.fc_rei = nn.Linear(1024, image_embed_dim)
    
    self.combine =  nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(image_embed_dim, image_embed_dim) )
    self.dynamic_scalar =  nn.Sequential(nn.Linear(image_embed_dim + text_embed_dim, image_embed_dim), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(image_embed_dim, 1), nn.Sigmoid())

    
  def compose_img_text(self, img_query, text_query):
    img_tensor, image_features = self.extract_img_feature(img_query) #N,2048,7,7
    img_embed = self.imgconv(img_tensor)
    #print('imgs:', img_embed.shape)
    n,c,h,w = img_embed.size()  #N,1024,7,7
    img_embed = img_embed.view(n,c,h*w) # N,1024,49
    text_embed, txts_features = self.extract_text_feature(text_query)#N,L,1024

    #print('txts:', text_embed.shape)
    dot_product = torch.bmm(text_embed, img_embed)   # N,L,49
    
    atten_i = self.softmax_i(dot_product / 7.0)
    atten_t = self.softmax_t(dot_product / 4.0)
    sentence_cat = []
    for i in range(img_tensor.size(2)*img_tensor.size(3)):
        sentence = torch.sum(text_embed * atten_i[:,:,i].unsqueeze(-1), dim=1)  # N,1024
        sentence_cat.append(sentence)
    sentence_cat = torch.stack(sentence_cat).permute(1, 2, 0).contiguous()  # N,1024,49  
    #print('sentence_cat', sentence_cat.shape)
    x = torch.cat([img_embed, sentence_cat], dim=1).view(n,-1,h,w)  # N,2048,7,7 
    
    imgatt = self.imgatt(x) #N,2048,7,7 
    sentence_cat = sentence_cat.view(n,-1,h,w)#N,1024,7,7
    #print('sentence_cat', sentence_cat.shape)
    txt2img = self.txt2img(sentence_cat)#N,2048,7,7 
    fuse_tensor = imgatt * img_tensor + txt2img
    fuse_i = self.img_fc(self.pool_i(fuse_tensor))
    retxtfea = self.fc_ret(self.pool_i(txt2img))
    
    img_cat = []
    for i in range(text_embed.shape[1]):
        img = torch.sum(img_embed * atten_t[:,i,:].unsqueeze(1), dim=-1) # N,1024
        img_cat.append(img)
    img_cat = torch.stack(img_cat).permute(1, 2, 0).contiguous()   # N,1024,L    
    x = torch.cat([text_embed.permute(0, 2, 1).contiguous(), img_cat], dim=1) # N,2048,L
    
    txtatt = self.txtatt(x) 
    img2txt = self.img2txt(img_cat)
    fuse_t = txtatt * text_embed.permute(0, 2, 1).contiguous() + img2txt # N,1024,L   
    fuse_t = self.fc_txt(self.pool_t(fuse_t))
    reimgfea = self.fc_rei(self.pool_t(img2txt)) 
    
    raw_combined_features = torch.cat((fuse_i, fuse_t), -1)
    dynamic_scalar = self.dynamic_scalar(raw_combined_features)
    com_fearues =  dynamic_scalar * fuse_i + (1 - dynamic_scalar) * fuse_t 
    #com_fearues = self.gamma(com_fearues) * com_fearues + self.beta(com_fearues)

    representations = {"repres":  F.normalize(com_fearues),  #self.normalization_layer(com_fearues),
                       "attention_mask" :  imgatt,
                       "img_tensor" :  img_tensor,
                       "local_out" :  fuse_tensor,
                       "retxtfea": retxtfea,
                       "txtfea": txts_features,
                       "reimgfea": reimgfea,
                       "imgfea": image_features,
                       "fuseimg" :  fuse_i,
                       "fusetxt" :  fuse_t,                        
                       "dynamic_scalar" :  dynamic_scalar,

                   }
    return representations         