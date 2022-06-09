# BERT4REC_MAML  
BERT4REC_MAML is implementation of maml with task adaptive loss using bert4rec model<br/>
BERT4REC model reference code: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch  
<br/>

## Dependencies  
* pytorch 
* tqdm 
* tensorboard
* wget  
<br/>

## Usage
<br/>
<b> Please read options.py carefully to choose correct learner </b>
<br/>
<br/>
Basic options

* maml with adaptive loss & multi step loss & adaptive loss weight
```bash 
python main.py
```
* maml without multi step loss
```bash 
python main.py --use_multi_step=False
```

* FO-MAML
```bash 
python main.py --use_multi_step=False --use_adaptive_loss=False
```

* test on Amazon dataset(example)
place amazon dataset csv file at Data/
```bash 
python main.py --mode=amazon --data_path=./Data/Office_Products.csv --min_window_size=2 --seq_len=10
```
<br/>

## Files
"models" folder     : bert4rec models<br/> 
"dataloader" file   : data preprocessing and task generation<br/>
"main.py" file      : main algorithm for maml<br/>
"options.py"        : hyperparamters control<br/>
"pretrain.py"       : training single bert model<br/>
"utils.py"          : util - no need to look at it<br/>
"pretrained" folder : folder that contains pretrained model<br/>
