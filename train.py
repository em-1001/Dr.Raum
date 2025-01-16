# train.py

import config
from preprocessing import preprocess
from loss import LossFuncion, MetricDice, BinaryOutput
from util import get_loaders
from model import UNETR
from tqdm import tqdm
from monai.losses.dice import DiceLoss, one_hot
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch

torch.backends.cudnn.benchmark = True

def train(model, optimizer, train_loader, DEVICE):
    model.train()
    batch_loss = []
    epoch_loss = 0
    epoch_dice_scores = 0

    loop = tqdm(train_loader, leave=True)

    for batch_idx, (x, y) in enumerate(loop):

        x = x.permute(0, 4, 2, 3, 1).float()
        y = y.permute(0, 4, 2, 3, 1).float()
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        loss = LossFuncion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

        bi_output = BinaryOutput(pred)
        MetricDice(pred, y)
        dice_score = MetricDice.aggregate().item()
        epoch_dice_scores += dice_score


        batch_loss.append(loss.item())
        mean_loss = sum(batch_loss) / len(batch_loss)
        loop.set_postfix(loss=mean_loss)

    avg_loss = epoch_loss / len(train_loader)
    avg_dice_score = epoch_dice_scores / len(train_loader)

    return avg_loss, avg_dice_score


def eval(model, optimizer, valid_loader, DEVICE):
    model.eval()
    epoch_loss = 0
    epoch_dice_scores = 0

    loop = tqdm(valid_loader, leave=True)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):

            x = x.permute(0, 4, 2, 3, 1).float()
            y = y.permute(0, 4, 2, 3, 1).float()
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = LossFuncion(pred, y)
            epoch_loss += loss.item()

            bi_output = BinaryOutput(pred)
            MetricDice(pred, y)
            dice_score = MetricDice.aggregate().item()
            epoch_dice_scores += dice_score

    avg_loss = epoch_loss / len(valid_loader)
    avg_dice_score = epoch_dice_scores / len(valid_loader)

    MetricDice.reset()

    return avg_loss, avg_dice_score


def func(epoch):
    if epoch < 50:
        return 1     # 0.0001
    elif epoch < 100:
        return 0.5   # 0.00005
    else:
        return 0.2   # 0.00001


def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def main():
    EPOCHS = config.EPOCHS
    DEVICE = config.DEVICE
    SAVE_MODEL = config.SAVE_MODEL
    model = UNETR().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS+10)
    best_loss = 999
    best_epoch = -1
    best_dice_score = 0
    torch.backends.cudnn.benchmark = True

    train_ids, val_ids = preprocess()
    train_loader, valid_loader = get_loaders(train_ids, val_ids)

    for epoch in range(1, EPOCHS+1):
        print(scheduler._last_lr)
        train_avg_loss, train_avg_dice_score = train(model, optimizer, train_loader, DEVICE)
        print(f"Epoch: {epoch}/{EPOCHS}, Loss: {train_avg_loss:.4f}, Dice Score: {train_avg_dice_score:.4f}")
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            eval_avg_loss, eval_avg_dice_score = eval(model, optimizer, valid_loader, DEVICE)

        if eval_avg_dice_score > best_dice_score:
            best_dice_score = eval_avg_dice_score
            best_epoch = epoch
            print(f"Validation in Epoch: {epoch}/{EPOCHS}, Loss: {eval_avg_loss:.4f}, Dice Score: {eval_avg_dice_score:.4f}")

            if SAVE_MODEL:
                save_checkpoint(model, optimizer, config.SAVE_PATH)

if __name__ == "__main__":
    main()
