# util.py

import config
from dataset import UnetrDataset
from torch.utils.data import DataLoader
import torch
import numpy as np 
import matplotlib.pyplot as plt



def get_loaders(train_ids, val_ids):
    training_generator = UnetrDataset(list_IDs=train_ids, transform=config.rand_affine)
    valid_generator = UnetrDataset(list_IDs=val_ids)

    train_loader = DataLoader(
        dataset=training_generator,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        dataset=valid_generator,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, valid_loader


def Display_Tumor_Segmenstation(model, x, y, DEVICE, SLICE=70):
    model.eval()
    with torch.no_grad():
        input = x.permute(0, 4, 2, 3, 1).float() # (1, 4, 128, 128, 96)
        input = input.to(DEVICE)
        p = model(input).detach().cpu()
    p = torch.sigmoid(p)
    p = p.squeeze(dim=0) # (4, 128, 128, 96)
    p = p.permute(3, 1, 2, 0).numpy() # (96, 128, 128, 4)

    core = p[:,:,:,1]
    edema = p[:,:,:,2]
    enhancing = p[:,:,:,3]

    x = x.squeeze(dim=0).numpy() # (96, 128, 128, 4)
    y = y.squeeze(dim=0).numpy()


    gt = np.argmax(y, axis=-1)
    gt = gt.astype(float)
    pred_all = np.argmax(p, axis=-1)
    pred_all = pred_all.astype(float)

    gt_core = gt.copy()
    gt_core[gt_core != 1] = np.nan
    gt_edema = gt.copy()
    gt_edema[gt_edema != 2] = np.nan
    gt_enhancing = gt.copy()
    gt_enhancing[gt_enhancing != 3] = np.nan

    pred_zero = pred_all.copy()
    pred_zero[pred_zero != 0] = np.nan
    pred_core = pred_all.copy()
    pred_core[pred_core != 1] = np.nan
    pred_edema = pred_all.copy()
    pred_edema[pred_edema != 2] = np.nan
    pred_enhancing = pred_all.copy()
    pred_enhancing[pred_enhancing != 3] = np.nan

    plt.figure(figsize=(8, 8))
    f, axarr = plt.subplots(3,3, figsize = (10, 10))

    axarr[0][0].imshow(x[SLICE,:,:,0], cmap='gray', interpolation='none')
    axarr[0][0].title.set_text('Original image flair')
    axarr[0][1].imshow(gt[SLICE,:,:], cmap="Greens") # RdPu
    axarr[0][1].title.set_text('Original Segmentation')
    axarr[0][2].imshow(pred_all[SLICE,:,:], cmap="Greens")
    axarr[0][2].title.set_text('Predicted - all classes')
    axarr[1][0].imshow(gt_core[SLICE,:,:], cmap="gray")
    axarr[1][0].title.set_text('Ground truth - Core')
    axarr[1][1].imshow(gt_edema[SLICE,:,:], cmap="gray")
    axarr[1][1].title.set_text('Ground truth - Edema')
    axarr[1][2].imshow(gt_enhancing[SLICE,:,:], cmap="gray")
    axarr[1][2].title.set_text('Ground truth - Enhancing')
    axarr[2][0].imshow(pred_core[SLICE,:,:], cmap="gray")
    axarr[2][0].title.set_text('Predicted - Necrotic/Core')
    axarr[2][1].imshow(pred_edema[SLICE,:,:], cmap="gray")
    axarr[2][1].title.set_text('Predicted - Edema')
    axarr[2][2].imshow(pred_enhancing[SLICE,:,:], cmap="gray")
    axarr[2][2].title.set_text('Predicted - Enhancing')

    plt.show()


def Display(model, optimizer, valid_loader):
    if config.LOAD_MODEL:
        model_filename = config.SAVE_PATH
        checkpoint = torch.load(model_filename, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    for i, (x, y) in enumerate(valid_loader):
        Display_Tumor_Segmenstation(model, x, y, config.DEVICE)




