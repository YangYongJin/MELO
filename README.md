# MELO - Meta-Learning with Adaptive Weighted Loss for Imbalanced Cold-Start Recommendation


A replication of the paper "Meta-Learning with Adaptive Weighted Loss for
Imbalanced Cold-Start Recommendation"

## Introduction

This repository includes code for training MELO and MAML with various baselines(BERT4REC, SASREC, GRU4REC, NARM). All baseline models are modified to do regression tasks. 


## References (Codes and Papers)

> **BERT4REC model reference code: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch**

> **NARM model reference code: https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch**

> **Maml++ reference code: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch**


> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**

> **SAS4Rec: Self-Attentive Sequential Recommendation (Kang et al.)**

> **NARM: Neural Attentive Session-based Recommendation (Li et al.)**

> **GRU4Rec: SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS (Hidasi et al.)**

> **METAL: Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning (Baik et al.)** 





## Dependencies  
* `pytorch==1.11.0` 
* `tqdm==4.64.0` 
* `numpy==1.12.5`
* `pandas==1.4.2`
* `tensorboard==2.9.0`
* `wget==3.2`
  
<br/>

## Datasets

We use three datasets; movielens(1m and 10m), amazon grocery, and yelp. Movielens dataset are automatically downloaded if you run an experiment on movielens. Preprocessed amazon grocery data is already in Data folder. Original amazon data can be downloaded from https://jmcauley.ucsd.edu/data/amazon/. For yelp dataset, you need to unzip yelp data since yelp dataset is much bigger than others. Original yelp data can be downloaded from https://www.yelp.com/dataset/documentation/main



<br/>

## Code Structures
"models" folder                      : This folder includes baseline models(Bert4rec, Narm, Sasrec, Gru4rec) with meta learning setting and adaptive loss networks<br/> 
"dataloader" file                    : This file contains data preprocessing and task generation for meta learner<br/>
"main.py" file                       : Main Code. MELO and MAML with sequential recommenders can be trained using this amin file. <br/>
"inner_loop_optimizers.py" file      : This code is same as inner loop optimizer for MAML++.<br/>
"options.py"                         : Configuration file<br/>
"train_original.py"                  : This code is used for training baseline models. With --save_pretrained option, you can save embedding and model parameters and use these parameters for training meta models.<br/>



# Running an experiment
<br/>
Please read <strong>options.py</strong> carefully to adjust configurations
<br/>


## MELO

* Train MELO(BERT4REC baseline) on Amazon dataset
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True 
```

* Test MELO on Amazon with best step of n(e.g. 1750) 
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True --test --checkpoint_step=1750
```

## MAML

* Train MAML(BERT4REC baseline) on Amazon dataset
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True --use_adaptive_loss=False
```

* Test MAML on Amazon with best step of n(e.g. 1750) 
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True --use_adaptive_loss=False 
```

## Basic Model (without meta learning)
* Train BERT4REC on Amazon dataset
```bash 
python train_original.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv --pretrain_epochs=40 --val_size=1000 --num_test_data=5000 --save_pretrained=False
```

* Test BERT4REC on Amazon with best step of n(e.g. 22) 
```bash 
python train_original.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv --val_size=1000 --num_test_data=5000 --save_pretrained=False --test --checkpoint_step=22
```
