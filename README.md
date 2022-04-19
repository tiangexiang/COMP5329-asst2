# 5329 assignment2
Tiange Xiang 470082274  
Shunqi Mao  
Xinyi Wang  

## Instructions
### Config
Create a new configure file in ```./src/configs/```. All configure files inherent from the ```base.py``` configure.

Please change ```exp_num``` and data path accordingly.

### Pipeline
please ```cd src``` first.

1. generate caption embeddings
```python3 get_caption_embedding.py --config 3```

2. train caption model
```python3 train_caption_model.py --config 3```

3. get image and detection features
```python3 get_image_feature.py --config 3```

4. train image model
```python3 train_image_model.py --config 3```

5. train detection model
```python3 train_detection_model.py --config 3```

6. train combine model
```python3 train_combine_model.py --config 3```

7. predict on test set
```python3 predict.py --config 3```