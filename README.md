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

We use three datasets; movielens(1m and 10m), amazon grocery, and yelp. Movielens dataset are automatically downloaded if you run an experiment on movielens. After movielens dataset is downloaded, Data folder is created. For amazon dataset, create amazon folder inside Data folder and then upload Amazon Grocery dataset inside  amazon folder which can be downloaded from https://jmcauley.ucsd.edu/data/amazon/. For yelp dataset, create yelp folder inside Data folder and then upload yelp dataset inside yelp folder which can be downloaded from https://www.yelp.com/dataset/documentation/main



<br/>

## Code Structures
"models" folder                      : This folder includes baseline models(Bert4rec, Narm, Sasrec, Gru4rec) with meta learning setting and adaptive loss networks<br/> 
"dataloader" file                    : Data Preprocessing and Task Generation<br/>
"main.py" file                       : MAML with Task Adaptive Loss Implementation <br/>
"inner_loop_optimizers.py" file      : This code is same as inner loop optimizer for MAML++.<br/>
"options.py"                         : Configuration file<br/>
"train_original.py"                  : This code is used for training baseline models. With --save_pretrained option, you can save embedding and model parameters and use these parameters for training meta models.<br/>



# Running an experiment
<br/>
Please read <strong>options.py</strong> carefully to adjust configurations
<br/>


## Training

* MELO(BERT4REC baseline) - Amazon 
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True 
```

* MELO(BERT4REC baseline) - Yelp
```bash 
python main.py --model=bert4rec --mode=yelp --data_path=./Data/yelp/yelp_ratings.csv --val_size=2000 --num_test_data=5000 --num_train_iterations=4000 --load_pretrained_embedding=True   
```

* MELO(BERT4REC baseline) - Movielens
place amazon ratings dataset(csv file) at Data/amazon/
```bash 
python main.py --model=bert4rec --mode=ml-1m --data_path=./Data/ml-1m/ratings.dat  --val_size=600 --num_test_data=1000 --num_train_iterations=2000 --load_pretrained_embedding=True --lstm_input=16 --lstm_hidden=128
```

* MAML(BERT4REC baseline) - Amazon 
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True --use_adaptive_loss=False
```

* MAML(BERT4REC baseline) - Yelp
```bash 
python main.py --model=bert4rec --mode=yelp --data_path=./Data/yelp/yelp_ratings.csv --val_size=2000 --num_test_data=5000 --num_train_iterations=4000 --load_pretrained_embedding=True --use_adaptive_loss=False 
```

* MAML(BERT4REC baseline) - Movielens
place amazon ratings dataset(csv file) at Data/amazon/
```bash 
python main.py --model=bert4rec --mode=ml-1m --data_path=./Data/ml-1m/ratings.dat  --val_size=600 --num_test_data=1000 --num_train_iterations=2000 --load_pretrained_embedding=True --use_adaptive_loss=False
```

## Test

* Test MELO on Amazon with best step of n(e.g. 1750) 
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True --test --checkpoint_step=1750
```

* Test MAML on Amazon with best step of n(e.g. 1750) 
```bash 
python main.py --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv  --val_size=1000 --num_test_data=5000 --num_train_iterations=3000 --load_pretrained_embedding=True --use_adaptive_loss=False --test --checkpoint_step=1750
```

## Train Baseline Model (without meta learning)
* Train on Amazon Data
```bash 
python train_original.py --save_pretrained=False --pretrain_epochs=40 --val_size=1000 --num_test_data=5000 --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv
```

* Test on Amazon Data with best step of n(e.g. 23) 
```bash 
python train_original.py --save_pretrained=False --pretrain_epochs=40 --val_size=1000 --num_test_data=5000 --model=bert4rec --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv --test --checkpoint_step=23
```
