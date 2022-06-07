# BERT4REC_MAML  
BERT4REC_MAML is a MAML for BERT4REC model.<br/>
Reference code: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch  

## Dependencies  
* pytorch 
* tqdm 
* tensorboard  

## Usage  
* maml with adaptive loss and multi step loss
```bash 
python main.py
```
* maml without multi step loss
```bash 
python main.py --use_multi_step=False
```

* normal FO-maml(no adaptive loss)
```bash 
python main.py --use_multi_step=False --use_adaptive_loss=False
```

## Files

"models" folder  : bert4rec models<br/> 
"dataloader" file: data preprocessing and task generation<br/>
"main.py" file   : main algorithm for maml<br/>
"options.py"     : hyperparamters control<br/>
