# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Evaluates the retrieval model."""
import numpy as np
import torch
import random
import torchvision
from tqdm import tqdm as tqdm
from torchvision import utils as vutils
from collections import OrderedDict
import torch.nn.functional as F


def fiq_test(opt, model, testset):
    model.eval()
    with torch.no_grad():
        test_queries = testset.get_test_queries()
        test_targets = testset.get_test_targets()

        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features
            imgs = []
            mods = []
            for t in tqdm(test_queries):
                imgs += [t['source_img_data']]
                mods += [t['mod']['str']]
                if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float()
                    imgs = torch.autograd.Variable(imgs).cuda()
                    dct_with_representations = model.compose_img_text(imgs, mods)
                    #f = F.normalize(dct_with_representations["repres"]).data.cpu().numpy()
                    f = dct_with_representations["repres"].data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    mods = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            logits = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                if len(imgs) >= opt.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float()
                    imgs = torch.autograd.Variable(imgs).cuda()
                    #imgs = F.normalize(model.extract_img_feature(imgs.cuda())["repres"]).data.cpu().numpy()
                    imgs = model.extract_tar_feature(imgs)["repres"].data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    '''# feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])'''
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    if opt.dataset == 'fashionIQ':
        for k in [10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                    r += 1
            r = 100 * r / len(nn_result)
            out += [('{}_r{}'.format(opt.dataset + ' ' + opt.name, k), r)]
    elif opt.dataset == 'shoes':  
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                    r += 1
            r = 100 * r / len(nn_result)
            out += [('{}_r{}'.format(opt.dataset, k), r)]    
    return out

        
def test(opt, model, testset):
    """Tests a model over the given testset."""
    model.eval()
    test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []
    if test_queries:
        imgs = []
        mods = []
        for t in tqdm(test_queries):
            torch.cuda.empty_cache()
            imgs += [testset.get_img(t['source_img_id'])]
            #print('testset', testset.get_img(t['source_img_id']))
            mods += [t['mod']['str']]

            if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()            
                dct_with_representations = model.compose_img_text(imgs, mods)
                f = dct_with_representations["repres"].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)
        all_target_captions = [t['target_caption'] for t in test_queries]

        # compute all image features
        imgs = []
        for i in tqdm(range(len(testset.imgs))):
            imgs += [testset.get_img(i)]
            if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                imgs = model.extract_tar_feature(imgs)["repres"].data.cpu().numpy()

                all_imgs += [imgs]
                imgs = []
        all_imgs = np.concatenate(all_imgs)
        all_captions = [img['captions'][0] for img in testset.imgs]

    else:
        # use training queries to approximate training retrieval performance
        imgs0 = []
        imgs = []
        mods = []
        training_approx = 9600
        for i in range(training_approx):
            torch.cuda.empty_cache()
            item = testset[i]
            imgs += [item['source_img_data']]
            mods += [item['mod']['str']]

            if len(imgs) >= opt.batch_size or i == training_approx:
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                dct_with_representations = model.compose_img_text(imgs, mods)
                f = dct_with_representations["repres"].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
            imgs0 += [item['target_img_data']]
            if len(imgs0) >= opt.batch_size or i == training_approx:
                imgs0 = torch.stack(imgs0).float()
                imgs0 = torch.autograd.Variable(imgs0).cuda()
                imgs0 = model.extract_tar_feature(imgs0)["repres"].data.cpu().numpy()
                all_imgs += [imgs0]
                imgs0 = []
            all_captions += [item['target_caption']]
            all_target_captions += [item['target_caption']]
        all_imgs = np.concatenate(all_imgs)
        all_queries = np.concatenate(all_queries)

    '''# feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])'''

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            sims[i, t['source_img_id']] = -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    if opt.dataset == 'css3d':
        for k in [1, 5, 10]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i] in nns[:k]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', r)]
    elif opt.dataset == 'fashion200k':
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i] in nns[:k]:
                    r += 1
            #r /= len(nn_result)
            r = 100 * r / len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', r)]    
        


    return out
    
def test1(opt, model, testset):
    """Tests a model over the given testset."""
    model.eval()
    test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []
    if test_queries:
        imgs = []
        mods = []
        imgsid = []
        for t in tqdm(test_queries):
            torch.cuda.empty_cache()
            imgs += [testset.get_img(t['source_img_id'])]
            imgsid += [t['source_img_id']]#get imgid
            mods += [t['mod']['str']]

            if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                dct_with_representations = model.compose_img_text(imgs.cuda(), mods)
                f = dct_with_representations["repres"].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)
        all_target_captions = [t['target_caption'] for t in test_queries]
        if opt.use_complete_text_query:
            if opt.dataset == 'mitstates':
                all_mods = [t['mod']['str'] + " " + t["noun"] for t in test_queries]
            else:
                all_mods = [t['target_caption'] for t in test_queries]
        else:
            all_mods = [t['mod']['str'] for t in test_queries]
        imgid = []
        # compute all image features
        imgs = []
        for i in tqdm(range(len(testset.imgs))):
            imgs += [testset.get_img(i)]
            if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                imgs = model.extract_img_feature(imgs.cuda())["repres"].data.cpu().numpy()

                all_imgs += [imgs]
                imgs = []
        all_imgs = np.concatenate(all_imgs)
        all_captions = [img['captions'][0] for img in testset.imgs]

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            sims[i, t['source_img_id']] = -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
    
    #save image
    savepath = '/media/dlc/ssd_data/HFX/ComposeAE+/logs/visualization'
    for a, i in enumerate(imgsid[30000:]):
       # if a % 10 == 0:
          image = testset.get_img(i, raw_img=True)
          targetcaption = all_target_captions[a]
          mod = all_mods[a]
          image = torchvision.transforms.ToTensor()(image)
          imagepath = '/media/dlc/ssd_data/HFX/ComposeAE+/logs/visualization/query_'  + str(i) + '_' + str(mod) + '.jpg'
          vutils.save_image(image, imagepath)
          for b, j in enumerate(nn_result[a][:10]):
              image = testset.get_img(j, raw_img=True)
              returntargetcaption = all_captions[nn_result[a][b]]
              image = torchvision.transforms.ToTensor()(image)
              imagepath = '/media/dlc/ssd_data/HFX/ComposeAE+/logs/visualization/return_'  + str(i)+ '_' + str(b) + '_' + str(returntargetcaption) + '.jpg'
              vutils.save_image(image, imagepath) 


    # compute recalls
    out = []
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    for k in [1, 5, 10, 50, 100]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
        out += [('recall_top' + str(k) + '_correct_composition', r)]

    return out