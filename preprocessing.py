# preprocessing.py

import config
import os
from sklearn.model_selection import train_test_split

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

def preprocess():
    TRAIN_DATASET_PATH = config.TRAIN_DATASET_PATH
    old_name = TRAIN_DATASET_PATH + "BraTS20_Training_355/W39_1998.09.19_Segm.nii"
    new_name = TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_seg.nii"
    # renaming the file
    try:
        os.rename(old_name, new_name)
        print("File has been re-named successfully!")
    except:
        print("File is already renamed!")


    train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

    train_and_test_ids = pathListIntoIds(train_and_val_directories);

    train_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.1, random_state=123)

    return train_ids, val_ids

