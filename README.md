# Green-guided Self-Supervised Learning for Diabetic Retinopathy Grading (GSSL)
This is an official implementation of the Green-guided Self-supervised Learning Framework for Diabetic Retinopathy Grading 

## Requirements


## Contents



## Datasets

### Pretraining Dataset
* Optic Disc Macula dataset (ODM) (link)
We created a novel dataset by combining two datasets DRTiD and DeepDRiD with dual-view fundus images. Our dataset, ODM, which stands for Optic-disc and Macula-centered, is designed explicitly for self-supervised learning and differs from the original dataset used for supervised learning. In contrast, combining datasets from different sources acquired from various sites can enrich diverse data structures and representations, which is the basis of learning in SSL. This fusion of diverse data can provide a more comprehensive knowledge to the SSL model. Furthermore, it is crucial to provide SSL models with varying difficulty levels of samples, where easy examples can guide the model’s initial learning, while harder ones can push its boundaries and enhance representation learning.
Our constructed ODM dataset comprises 5100 pairs of two-field fundus images, each representing a specific grade. The ODM dataset exhibits imbalances, similar to the original dataset and many benchmark datasets in supervised learning. The development of this extensive dataset and its variations addresses a significant gap in the study community and drives future progress in dual-views diabetic retinopathy diagnosis research using the self-supervised learning paradigm. The fundus images in Figure 1 display several DR samples from the ODM dataset.

* ODM dataset variants


### Evaluation Datasets
* DDR [homepage](https://github.com/nkicsl/DDR-dataset).
* APTOS 2019 [homepage](https://www.kaggle.com/c/aptos2019-blindness-detection/overview).
* Messidor-2 [images](https://www.adcis.net/en/third-party/messidor2/) , [Labels](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) 



## GSSL Architecture 

Picture 



## Pre-train model - GSSL 

1. Download the train model and place in .../ ../ ///
2.  ...
3.   ....


## Results 
Add the table 



## Citation 
if you find this code helpful in your research or work, please cite the following paper. 

@article{almattar2024gssl,
  title={Green-guided Self-Supervised Learning for Diabetic Retinopathy Grading (GSSL)},
  author={Wadha, Saeed , Fakhri, Hamzah},
  journal={CBM},
  year={2024}
}


## Acknowledgements


