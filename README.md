# Green-guided Self-Supervised Learning for Diabetic Retinopathy Grading (GSSL)
This is an official implementation of the Green-guided Self-supervised Learning Framework for Diabetic Retinopathy Grading 


## Contents
1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [GSSL Architecture](#gssl-architecture)
4. [Installation](#installation)
5. [Dataset Preprocessing](#datasets-preprocessing-for-pretrain-model-gssl)
6. [Training GSSL model](#training-gssl-model)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Citation](#citation)
11. [Acknowledgements](#acknowledgements)


## Introduction
Contrastive self-supervised learning is an effective training approach for medical imaging tasks, mainly when well-annotated datasets are limited, which allows for extracting robust and reliable representations from unlabeled data without human annotation. To attain exceptional performance in the grading of Diabetic Retinopathy, a substantial quantity of labeled data is necessary. However, this data collection and annotation process can be costly and time-consuming. To tackle this problem, we suggest implementing a contrastive self-supervised framework named Green-guided self-supervised learning to make the most of the representation of fundus images by using a small number of unlabeled fundus images. First, we employ the data stored in the green channel to enhance the visibility of the fundus image characteristics, making them observable using saliency guidance. Furthermore, we provide empirical evidence demonstrating that a restricted quantity of dual-view fundus images yields comparable outcomes to single-view images Moreover, we highlight the potency of the projection head and the significance of the residual connection in extracting intricate features that steer the whole training process, enhancing performance on the subsequent tasks. We assessed our method using three standard diabetic retinopathy datasets. Our approach surpassed the state-of-the-art self-supervised learning techniques, utilizing only a few unlabeled fundus images.

## Datasets

### Pretraining Dataset
#### Optic Disc Macula dataset (ODM) (TODO)
We created a novel dataset by combining two datasets DRTiD and DeepDRiD with dual-view fundus images. Our dataset, ODM, which stands for Optic-disc and Macula-centered, is designed explicitly for self-supervised learning and differs from the original dataset used for supervised learning. In contrast, combining datasets from different sources acquired from various sites can enrich diverse data structures and representations, which is the basis of learning in SSL. This fusion of diverse data can provide a more comprehensive knowledge to the SSL model. Furthermore, it is crucial to provide SSL models with varying difficulty levels of samples, where easy examples can guide the model’s initial learning, while harder ones can push its boundaries and enhance representation learning.
Our constructed ODM dataset comprises 5089 pairs of two-field fundus images, each representing a specific grade. The ODM dataset exhibits imbalances, similar to the original dataset and many benchmark datasets in supervised learning. The development of this extensive dataset and its variations addresses a significant gap in the study community and drives future progress in dual-views diabetic retinopathy diagnosis research using the self-supervised learning paradigm. The fundus images in Figure 1 display several DR samples from the ODM dataset.

#### ODM dataset variants
We create four versions of Optic-disc (OD), Macula (MA), Optic-disc and Macula Small ODM-S, and Optic-disc and Macula Green ODM-G, Table 1 describes briefly each variant. The OD dataset comprises 2545 single- view fundus images obtained with the optic disc (OD) posi- tioned at the center. A new perspective was created using the MA dataset, which consists of 2544 fundus images collected with a focus on the macula. ODM-S is a subset of the original ODM dual-view dataset with the same amount of samples as the OD and MA datasets. It differs from OD and MA datasets in that the ODM-S specifically focuses on examples of dual views. The last variant, ODM-G, includes dual-view fundus images that solely contain the green channel while maintaining the same number of samples as ODM.

Download ODM, [Click here](https://drive.google.com/drive/folders/1Dlheky35lPq7y7q2oj2n_Po2DamLJo-9?usp=sharing)

<p align="center">
  <img algin="center" src="/images/ODM_variants.png" title="title" >
  <figcaption>  </figcaption>
  </p>
  
<p align="center">
  <img algin="center" src="/images/MAOD.png" title="title" >
  <figcaption>Selected dual-view fundus images from ODM dataset. (a) Dual-view fundus images in RGB color space. (b) Corresponding green-channel fundus images.  </figcaption>
  </p>

### Evaluation Datasets
* DDR [homepage](https://github.com/nkicsl/DDR-dataset).
* APTOS 2019 [homepage](https://www.kaggle.com/c/aptos2019-blindness-detection/overview).
* Messidor-2 [images](https://www.adcis.net/en/third-party/messidor2/) , [Labels](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) 



## GSSL Architecture 

<p align="center">
  <img algin="center" src="/images/GSSL.png" title="title" >
  <figcaption> The proposed Green-guided self-supervised learning (GSSL) framework. The Green-channel extraction component extracts green-channel knowledge from RGB dual-view fundus image. Two augmented images are derived from the green dual- view image and fed to two encoders. The query encoder is the base encoder where the SSL learning is achieved. The key encoder is a momentum encoder that supports the query encoder by focusing on green-salient guidance. The projection head design consists of two layers with residual connection, batch normalization and ReLU.  </figcaption>
  </p> 

## Installation
To install the dependencies, run:
https://github.com/Wadha-Almattar/GSSL

1. Frisr, clone the repository 
```shel
git clone https://github.com/Wadha-Almattar/GSSL.git
```
2. Access GSSL

```shell
cd GSSL
```

3. Create an environment and name it gssl

```shell
conda create -n gssl python=3.8.0
conda activate gssl
```

4. Install the requirements at the gssl environment

```shell
pip install -r requirements.txt
```
   



## Dataset Preprocessing for Pretrain model GSSL 

### Green channel extraction

```shell
cd utils
python green.py -n 8 --image-folder <path/to/processed/dataset> --output-folder <path/to/green images/folder>
cd ..
```
Here, `-n` is the number of workers.

<p align="center">
  <img algin="center" src="/images/green_images.png " title="title" >
  <figcaption> Green-channel Extraction: Selected dual-view fundus images from ODM dataset. (a) Dual-view fundus images in RGB color space. (b) Corresponding green-channel fundus images.  </figcaption>
  </p>  



### Crop and Resize
```shell
cd utils
python crop.py -n 8 --crop-size 512 --image-folder <path/to/green images/folder> --output-folder <path/to/processed green images/folder>
cd ..
```
Here, `-n` is the number of workers. The processed green images dataset will be saved in the `--output-folder`.

### Detect saliency on green-channel images 

```shell
cd utils
python saliency_detect.py -n 8 --image-folder <path/to/processed green images/dataset> --output-folder <path/to/saliency green/folder>
cd ..
```
Here, `-n` is the number of workers.

<p align="center">
  <img algin="center" src="/images/salient_green.png " title="title" >
  <figcaption> The visualization of the salient maps of a fundus image. a) original fundus image in RGB color space. b) The salient map is detected from the RGB fundus image. c) The salient map is detected from the green-channel fundus image.  </figcaption>
  </p>  



### Prepare the dataset in a pickle file format
The pickle file contains the pretrain dataset (ODM) along with the salient map of each green-channel image. 
ignore [cd utils] if you are still at the same directory. 

```shell
cd utils
python folder2pkl.py --image-folder <path/to/processed green images/folder> --saliency-folder <path/to/saliency green/folder> --output-file ../data_index/pretraining_dataset.pkl
cd ..
```

## Training GSSL model
Pretraining with ViT-S on a single-GPUs node:

```shell
python main.py \
    --num-workers 5 --arch ViT-S-p16 --batch-size 128 \
    --epochs 300 --warmup-epochs 20 \
    --data-index ./data_index/pretraining_dataset.pkl \
    --save-path <path/to/save/checkpoints> 
```
NVIDIA GeForce 3090 GPU with 19GB memory are used in our experiments
CUDA Version: 12.0

## Evaluation 
We evaluate our pretrain model GSSL on three benchmark datasets:
* APTOS 2019
* Messidor-2 
* DDR

### Evaluation Dataset preparation

1\. Organize each dataset as follows:
```
├── dataset
    ├── train
        ├── DR1
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
        ├── DR2
            ├── image3.jpg
            ├── image4.jpg
            ├── ...
        ├── DR3
        ├── ...
    ├── val
    ├── test
```
Here, `val` and `test` have the same structure of `train`

2\. Crop and resize:
```shell
cd utils
python crop.py -n 8 --crop-size 512 --image-folder <path/to/image/dataset> --output-folder <path/to/processed/dataset>
cd ..
```

### Fine-tuning Evalaution

Fine-tuning evaluation on aptos2019 dataset on one GPU:
```shell
python eval.py \
    --dataset aptos2019 --arch ViT-S-p16 --kappa-prior \
    --data-path <path/to/aptos2019/dataset/folder> \
    --checkpoint <path/to/pretrained/model/epoch_xxx.pt> \
    --save-path <path/to/save/eval/checkpoints>
```
you can change --dataset to any of the three evaluation datasets:
* aptos2019
* messidor2
* ddr

For --arch choice, you have to use as same as architecture used in the pretraining model. 

## Results 
<p align="center">
  <img algin="center" src="/images/results.png " title="title" >
  <figcaption> Comparison results with State-of-the-art self-supervised models on three benchmark datasets: APTOS, Messidor, and DDR. GSSL is trained with limited green dual-view fundus image samples using ViT-s as the backbone feature extractor. Our constructed dataset ODM-G contains 5,100 green dual-view fundus images. Results are reported in Kappa score.  </figcaption>
  </p>  



## Citation 
if you find this code helpful in your research or work, please cite the following paper. 

```
@article{almattar2024gssl,
  title={Green-guided Self-Supervised Learning for Diabetic Retinopathy Grading (GSSL)},
  author={Wadha, Saeed , Fakhri, Hamzah},
  journal={CBM},
  year={2024}
}
```


## Acknowledgements
This work is dunded by SDAIA-KFUPM Joint Research Center for Artificial Intelligence at KinFahd University of Petroleum & Minerals (KFUPM) under grant number (XXX) .

