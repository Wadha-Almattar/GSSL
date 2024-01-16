# Green-guided Self-Supervised Learning for Diabetic Retinopathy Grading (GSSL)
This is an official implementation of the Green-guided Self-supervised Learning Framework for Diabetic Retinopathy Grading 

## Requirements


## Contents
1. [Datasets](#datasets).
3. [GSSL Architecture](#gssl-architecture).
4. [Installation](#installation).
5. [Dataset Preprocessing](#datasets-preprocessing-for-pretrain-model-gssl).
6. [Training GSSL model](#training-gssl-model).
8. [Evaluation](#evaluation).


## Datasets

### Pretraining Dataset
#### Optic Disc Macula dataset (ODM) (link)
We created a novel dataset by combining two datasets DRTiD and DeepDRiD with dual-view fundus images. Our dataset, ODM, which stands for Optic-disc and Macula-centered, is designed explicitly for self-supervised learning and differs from the original dataset used for supervised learning. In contrast, combining datasets from different sources acquired from various sites can enrich diverse data structures and representations, which is the basis of learning in SSL. This fusion of diverse data can provide a more comprehensive knowledge to the SSL model. Furthermore, it is crucial to provide SSL models with varying difficulty levels of samples, where easy examples can guide the model’s initial learning, while harder ones can push its boundaries and enhance representation learning.
Our constructed ODM dataset comprises 5100 pairs of two-field fundus images, each representing a specific grade. The ODM dataset exhibits imbalances, similar to the original dataset and many benchmark datasets in supervised learning. The development of this extensive dataset and its variations addresses a significant gap in the study community and drives future progress in dual-views diabetic retinopathy diagnosis research using the self-supervised learning paradigm. The fundus images in Figure 1 display several DR samples from the ODM dataset.

#### ODM dataset variants
We create four versions of Optic-disc (OD), Macula (MA), Optic-disc and Macula Small ODM-S, and Optic-disc and Macula Green ODM-G, Table 1 describes briefly each variant. The OD dataset comprises 2545 single- view fundus images obtained with the optic disc (OD) posi- tioned at the center. A new perspective was created using the MA dataset, which consists of 2544 fundus images collected with a focus on the macula. ODM-S is a subset of the original ODM dual-view dataset with the same amount of samples as the OD and MA datasets. It differs from OD and MA datasets in that the ODM-S specifically focuses on examples of dual views. The last variant, ODM-G, includes dual-view fundus images that solely contain the green channel while maintaining the same number of samples as ODM.
(Add the table)

<p align="center">
  <img algin="center" src="/images/..png" title="title" >
  <figcaption> caption  </figcaption>
  </p>
  
(Add image to MA , OD) 
<p align="center">
  <img algin="center" src="/images/..png" title="title" >
  <figcaption> caption  </figcaption>
  </p>

### Evaluation Datasets
* DDR [homepage](https://github.com/nkicsl/DDR-dataset).
* APTOS 2019 [homepage](https://www.kaggle.com/c/aptos2019-blindness-detection/overview).
* Messidor-2 [images](https://www.adcis.net/en/third-party/messidor2/) , [Labels](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) 



## GSSL Architecture 

<p align="center">
  <img algin="center" src="/images/..png" title="title" >
  <figcaption> caption  </figcaption>
  </p> 

## Installation
To install the dependencies, run:
https://github.com/Wadha-Almattar/GSSL
```
git clone https://github.com/Wadha-Almattar/GSSL.git
cd GSSL
conda create -n gssl python=3.8.0
conda activate gssl
pip install -r requirements.txt
```

## Datasets Preprocessing for Pretrain model GSSL 

### Green channel extraction

```shell
cd utils
python green.py -n 8 --image-folder <path/to/processed/dataset> --output-folder <path/to/green images/folder>
cd ..
```
Here, `-n` is the number of workers.

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
Add the table 



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


