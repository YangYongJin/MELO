# Introduction

BERT4REC_MAML is implementation of MAML with task adaptive loss using BERT4REC model

> **BERT4REC model reference code: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch**

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**

> **METAL: Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning (Baik et al.)** 



# Dependencies  
* pytorch 
* tqdm 
* tensorboard
* wget  
<br/>

# Usage
<br/>
Please read <strong>options.py</strong> carefully to adjust multiple options
<br/>


## Training

* MAML with Adaptive Loss(Proposed Method) on Movielens Data
```bash 
python main.py
```

* FO-MAML on Movielens Data
```bash 
python main.py --use_multi_step=False --use_adaptive_loss=False
```

* Training on Amazon dataset(example)
place amazon ratings dataset(csv file) at Data/amazon/
```bash 
python main.py --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv --min_sub_window_size=2 --max_seq_len=30 --num_samples=25 --num_query_set=1
```

## Test

* Test on Movielens Data using model of nth iterations 
```bash 
python main.py --checkpoint_step=n --test
```

* Test on Amazon Data using model of nth iterations
```bash 
python main.py --mode=amazon --data_path=./Data/amazon/grocery_ratings.csv --min_sub_window_size=2 --max_seq_len=30 --num_samples=25 --num_query_set=1 --checkpoint_step=n --test
```

## Train Bert4rec Model(without meta learning)
* Train on Movielens Data
```bash 
python pretrain.py
```

* Test on Movielens Data using model of nth iterations
```bash 
python pretrain.py --checkpoint_step=n --test
```

<br/>

## Files
"models" folder     : BERT4REC models<br/> 
"dataloader" file   : Data Preprocessing and Task Generation<br/>
"main.py" file      : MAML with Task Adaptive Loss Implementation <br/>
"options.py"        : Adjust Multiple Options(Hyperparmaters, Task Information, etc)<br/>
"pretrain.py"       : Training Single BERT4REC Model<br/>
"utils.py"          : util functions used for preprocessing<br/>
