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

### Pipeline
please ```cd src``` first.

1. generate caption embeddings (skip this if you download the embeding file already)  
```python3 get_caption_embedding.py --config 4```

2. train caption model 
```python3 train_caption_model.py --config 4```

3. train image model (specifiy which feature level in YOLOv5 shoud be used)
```python3 train_image_model.py --config 4 --level 4```

5. train combine model  
```python3 train_combine_model.py --config 4```

6. predict on test set  
```python3 predict.py --config 4```