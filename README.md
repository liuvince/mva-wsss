# Weakly Supervised Semantic Segmentation - School Project

The code is a little messy but should work.

1. We download two datasets: Pascal VOC Dataset and Semantic Boundaries Dataset.
```
mkdir data; cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
```

As well as weights provided by https://github.com/Juliachang/SC-CAM. The link is:
```
https://drive.google.com/file/d/1YB3DkHiBeUH5wn6shk93jChvXwfOxwBE/view
```

We need external code in order to perform clustering [7], refinement [2][8] and fmix [6].

```
git clone https://github.com/ssarfraz/FINCH-Clustering.git
git clone https://github.com/jiwoon-ahn/psa.git
git clone https://github.com/ecs-vlc/FMix.git
git clone https://github.com/kazuto1011/deeplab-pytorch.git
```

2. We use the script ```prepare_data.py``` which creates a repository ```experiment``` and creates three files ```data_train.csv```, ```data_train_aug.csv``` and ```data_test.csv```, which contains the absolute path of the images, as well as image-level class labels.
```
./
└── data
    ├── VOCdevkit              # Pascal VOC Dataset
    ├── benchark               # SBD Dataset
    ├── downoad_data.sh
└── experiment
    ├── data_test.csv
    ├── data_train.csv
    ├── data_train_aug.csv
└── FINCH_Clustering
└── FMix
└── psa
└── deeplab-pytorch
└── ...
```
```
python3 prepare_data.py
```

3. Baseline CAM [10].
```
python3 train_cam.py     --log_dir experiment/baseline/log_dir 
python3 compute_cams.py  --load_model experiment/baseline/log_dir/best_model.ckpt --cam_dir experiment/baseline/cam_dir
python3 evaluate_cams.py --cam_dir experiment/baseline/cam_dir
```

4. Sub-category Exploration labels [1].
   
(One round, but we can use the code for more rounds)

Please speficy the number of clusters.

```
python3 train_cam.py             --log_dir experiment/sce/log_dir 
python3 compute_vit_features.py  --load_model experiment/sce/log_dir/best_model.ckpt --compute_vit_features 0 --vit_features_dir experiment/sce/features_dir
python3 compute_clusters.py      --features_dir experiment/sce/vit_features_dir --data_train_out experiment/sce/data_train_clusters.csv --data_val_out experiment/sce/data_test_clusters.csv
python3 train_cam_clusters.py    --knowledge_distillation 1 --n_clusters [clusters] --data_train experiment/sce/data_train_clusters.csv --data_val experiment/sce/data_test_clusters.csv
python3 evaluate_cams.py         --cam_dir experiment/sce/cam_dir
```


5. Sub-category Teaching labels.

(One round, but we can use the code for more rounds)

Please speficy the number of obtained clusters. (TODO Need to change the code in order to automate this)

```
python3 train_cam.py             --log_dir experiment/sct/log_dir  --max_epochs 25
python3 compute_vit_features.py  --load_model experiment/sct/log_dir/best_model.ckpt --compute_vit_features 1 --vit_features_dir experiment/vit/features_dir
python3 compute_clusters.py      --features_dir experiment/sct/vit_features_dir --data_train_out experiment/sct/data_train_clusters.csv --data_val_out experiment/sct/data_test_clusters.csv
python3 train_cam_clusters.py    --knowledge_distillation 1 --n_clusters [clusters] --data_train experiment/sct/data_train_clusters.csv --data_val experiment/sct/data_test_clusters.csv
python3 evaluate_cams.py         --cam_dir experiment/sct/cam_dir
```

6. Mixup-CAM [2].

```
python3 train_cam.py     --log_dir experiment/mixup/log_dir --mixup 1 --ent 1 --concent 1
python3 compute_cams.py  --load_model experiment/mixup/log_dir/best_model.ckpt --cam_dir experiment/mixup/cam_dir
python3 evaluate_cams.py --cam_dir experiment/mixup/cam_dir
```

7. Manifold Mixup-CAM using [9].

```
python3 train_cam.py     --log_dir experiment/manifold_mixup/log_dir --manifold_mixup 1 --ent 1 --concent 1
python3 compute_cams.py  --load_model experiment/manifold_mixup/log_dir/best_model.ckpt --cam_dir experiment/mixup/cam_dir
python3 evaluate_cams.py --cam_dir experiment/manifold_mixup/cam_dir
```

8. Fmix-CAM using [6].

FMix need more epochs in order to converge. 

```
python3 train_cam.py     --max_epochs 25 --log_dir experiment/fmix/log_dir --fmix 1 --ent 1 --concent 1
python3 compute_cams.py  --load_model experiment/fmix/log_dir/best_model.ckpt --cam_dir experiment/fmix/cam_dir
python3 evaluate_cams.py --cam_dir experiment/fmix/cam_dir
```

9. Fmix-CAM with augmentated training images.
```
python3 train_cam.py   --log_dir experiment/fmix_v2/log_dir --fmix 1 --ent 1 --concent 1 --data_train experiment/data_train_aug.csv --lr 0.05 --max_epochs 20
python3 compute_cams.py --load_model experiment/fmix_v2/log_dir/best_model.ckpt --cam_dir experiment/fmix_v2/cam_dir
python3 evaluate_cams.py --cam_dir experiment/fmix_v2/cam_dir
```

## Reference
1. Yu-Ting Chang and Qiaosong Wang and Wei-Chih Hung and Robinson Piramuthu and Yi-Hsuan Tsai and Ming-Hsuan Yang. Weakly-Supervised Semantic Segmentation via Sub-category Exploration. CVPR, 2020.
2. Yu-Ting Chang, Qiaosong Wang, Wei-Chih Hung, Robinson Piramuthu, Yi-Hsuan Tsai, Ming-Hsuan Yang. Mixup-CAM: Weakly-supervised Semantic Segmentation via Uncertainty Regularization. 2020
3. Jiwoon Ahn and Suha Kwak. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation. CVPR, 2018.
4. Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. TPAMI, 2017.
5. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CVPR, 2016.
6. Ethan Harris, Antonia Marcu, Matthew Painter, Mahesan Niranjan, Adam Prügel-Bennett, Jonathon Hare. FMix: Enhancing Mixed Sample Data Augmentation. 2020.
7. M. Saquib Sarfraz, Vivek Sharma and Rainer Stiefelhagen. Efficient Parameter-free Clustering Using First Neighbor Relations. CVPR, 2019.
8. Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. 2017.
9. Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, Aaron Courville, David Lopez-Paz, Yoshua Bengio. Manifold Mixup: Better Representations by Interpolating Hidden States. ICML 2019.
10. Bolei  Zhou,  Aditya  Khosla,  Agata  Lapedriza,  Aude  Oliva,  and  Antonio  Torralba.Learning deep features for discriminative localization.   InCVPR, 2016. 
11. Geoffrey Hinton, Oriol Vinyals, Jeff Dean. Distilling the Knowledge in a Neural Network. NIPS 2014