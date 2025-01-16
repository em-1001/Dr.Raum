# Dr.Raum

<img src="https://github.com/user-attachments/assets/aeff1eb7-01b7-4199-9624-32d6ff8a0fec">

**Dr. Raum** is a brain tumor segmentation project using the **UNETR** model. More details are on my [blog](https://em-1001.github.io/computer%20vision/Raum/).

# How to Train

1. **Install packages**
```sh
$ pip install -r requirements.txt
```

2. **Create kaggle token**  
Download the kaggle token and put it in the Dr.Raum folder
```json
{"username":"?????","key":"????????????????????????????"}
```

3. **Download dataset**  
```sh
$ mkdir -p ~/.kaggle
$ cp kaggle.json ~/.kaggle/
$ chmod 600 ~/.kaggle/kaggle.json
$ kaggle datasets download awsaf49/brats20-dataset-training-validation
$ unzip -qq /content/brats20-dataset-training-validation.zip
```
4. **Set config.py values and Train**
```sh
$ python3 train.py
```

# Dice Coefficient

```ini
# Train
Loss = 0.2376
Dice_Score = 0.7878

# Validation
Loss = 0.2280
Dice_Score = 0.7977
```

# Reference

## Web
1. Preprocess : https://www.kaggle.com/code/zeeshanlatif/brain-tumor-segmentation-using-u-net
2. F1 Score: https://velog.io/@jadon/F1-score%EB%9E%80
3. Dice Loss: https://attagungho.tistory.com/11#index
4. UNETR : https://kimbg.tistory.com/33


## Paper

1. Transformer : https://arxiv.org/abs/1409.0473
2. 3D U-net : https://arxiv.org/abs/1606.06650
3. Vision Transformer : https://arxiv.org/abs/2010.11929
4. UNETR : https://arxiv.org/abs/2103.10504
