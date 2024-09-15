# Dynamic Weighted Combiner for Mixed-Modal Image Retrieval - Accepted at AAAI2024
The paper can be accessed at: https://arxiv.org/pdf/2312.06179.pdf


If you find this code useful in your research then please cite

```bibtex
@article{huang2023dynamic,

  title={Dynamic Weighted Combiner for Mixed-Modal Image Retrieval},
  
  author={Huang, Fuxiang and Zhang, Lei and Fu, Xiaowei and Song, Suqi},
  
  journal={arXiv preprint arXiv:2312.06179},
  
  year={2023}
}


@inproceedings{huang2023dynamic,

  title={Dynamic Weighted Combiner for Mixed-Modal Image Retrieval},
  
  author={Huang, Fuxiang and Zhang, Lei and Fu, Xiaowei and Song, Suqi},
  
  booktitle={Association for the Advance of Artificial Intelligence (AAAI)},
  
  year={2024}
}
```


## Abstract

Mixed-Modal Image Retrieval (MMIR) as a flexible search paradigm has attracted wide attention. However, previous approaches always achieve limited performance, due to two critical factors are seriously overlooked. 1) The contribution of image and text modalities is different, but incorrectly treated equally. 2) There exist inherent labeling noises in describing users' intentions with text in web datasets from diverse real-world scenarios, giving rise to overfitting. We propose a Dynamic Weighted Combiner (DWC) to tackle the above challenges, which includes three merits. First, we propose an Editable Modality De-equalizer (EMD) by taking into account the contribution disparity between modalities, containing two modality feature editors and an adaptive weighted combiner. Second, to alleviate labeling noises and data bias, we propose a dynamic soft-similarity label generator (SSG) to implicitly improve noisy supervision. Finally, to bridge modality gaps and facilitate similarity learning, we propose a CLIP-based mutual enhancement module alternately trained by a mixed-modality contrastive loss. Extensive experiments verify that our proposed model significantly outperforms state-of-the-art methods on real-world datasets.
#### Requirements and Installation
 Python 3.7
```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
## Running the experiments 

### Download the datasets

### CSS3D dataset
- Download the dataset from this [external website](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing).

#### Fashion200k dataset
- Download the dataset via this [link](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) 
- To ensure fair comparison, we employ the same test queries as TIRG. They can be downloaded from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt). 

#### FashionIQ dataset
- Download Fashion-IQ dataset images from [here](https://github.com/hongwang600/fashion-iq-metadata). 
- Download Fashion-IQ dataset annotations from [here](https://github.com/XiaoxiaoGuo/fashion-iq).    
- To ensure fair comparison, we employ the same splits as VAL. They can be downloaded from [here](https://www.tensorflow.org/). 
#### Shoes dataset                                          
- Download Shoes dataset images from [here](http://tamaraberg.com/attributesDataset/attributedata.tar.gz). 
- Download Shoes dataset annotations from [here](https://github.com/yanbeic/VAL/tree/master/datasets/shoes).        


## Running the Code

For training and testing new models, pass the appropriate arguments. 

For instance, for training DWC model on Fashion200k dataset run the following command:

```
python   main.py --dataset=fashion200k --dataset_path=../data/fashion200k/ 
```






