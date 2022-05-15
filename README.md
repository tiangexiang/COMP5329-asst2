# 5329 assignment2
Tiange Xiang 470082274  
Shunqi Mao  
Xinyi Wang  

## Environment
Please clone our environment using the following command:

```
conda env create -f environment.yml
conda activate 5329
```

## Instructions
### Config
Create a new configure file in ```./src/configs/```. All configure files inherent from the ```base.py``` configure.

Please change ```exp_num``` and data path accordingly.

### Pre processing
1. Generate caption embeddings:
```cd src```
```python3 get_caption_embedding.py --config EXP```

2. Generate image features:
```cd ..```
```cd ML_Decoder```
```python3 infer_ours.py --phase train``` for train features (w/o denoise)
```python3 infer_ours.py --phase train --denoise``` for train features (w denoise)
```python3 infer_ours.py --phase test``` for test features (w/o denoise)
```python3 infer_ours.py --phase test --denoise``` for test features (w denoise)
```cd ..```


### Training Pipeline
0. ```cd src```

1. train caption model 
```python3 train_caption_model.py --config EXP```

2. train combine model  
```python3 train_combine_model.py --config EXP```

### Inference pipeline
0. ```cd src``` and make sure model weights are in ```ckpt/EXP```.

1. get best threshold
```python3 find_threshold.py --config EXP```

3. predict on test set  
```python3 predict.py --config EXP```

predicted csv file will be saved in ```predicts/```.