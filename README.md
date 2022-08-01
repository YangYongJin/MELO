# MELO - MEta-learning-based Sequential Recommendation Framework with Learned LOss Network for Imbalanced Cold-Start Recommendation 

Melo solves user cold start problem of sequential recommendation using meta learning with adaptive loss

# References (Codes and Papers)

> **BERT4REC model reference code: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch**

> **NARM model reference code: https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch**

> **Maml++ reference code: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch**


> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**

> **SAS4Rec: Self-Attentive Sequential Recommendation (Kang et al.)**

> **NARM: Neural Attentive Session-based Recommendation (Li et al.)**

> **GRU4Rec: SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS (Hidasi et al.)**

> **METAL: Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning (Baik et al.)** 





# Dependencies  
* pytorch 
* tqdm 
* tensorboard
* wget
* shutil
* zipfile
* tempfile
  
<br/>

# Usage
<br/>
Please read <strong>options.py</strong> carefully to adjust multiple options
<br/>


## Training

* MAML++ with Adaptive Loss(Proposed Method) on Movielens Data
```bash 
python main.py
```

* MAML++ on Movielens Data
```bash 
python main.py --use_adaptive_loss=False
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

## Train Baseline Model(without meta learning)
* Train on Movielens Data
```bash 
python train_original.py
```

* Test on Movielens Data using model of nth iterations
```bash 
python train_original.py --checkpoint_step=n --test
```

<br/>

## Files
"models" folder                      : Baseline Models(Bert4rec, Narm, Sas4rec, Gru4rec)<br/> 
"dataloader" file                    : Data Preprocessing and Task Generation<br/>
"main.py" file                       : MAML with Task Adaptive Loss Implementation <br/>
"inner_loop_optimizers.py" file      : Inner Loop Optimizers for MAML++<br/>
"options.py"                         : Adjust Multiple Options(Hyperparmaters, Task Information, etc)<br/>
"train_original.py"                  : Training Single Baseline Model<br/>
