import argparse
import torch
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from evaluation import iou
from data.dataset import MilaLogoDataset
import os
from torchvision import transforms
from model.pretrained_baseline import get_baseline
from model.pretrained_segmentation import get_fcn
import time


def train(model, dataloaders, optimizer,
          criterion, device, best, epoch):
    scheduler = lr_scheduler
    best_iou, best_model_wts = best
    print_freq = 100

    model.train()  # Set model to training mode
    running_loss = 0.0
    total_iou = 0

    # Iterate over data.
    for i, (input, mask) in enumerate(dataloaders["train"]):
        input = input.to(device)
        mask = mask.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            out = model(input)['out']
            pred = torch.argmax(out.squeeze(), dim=0).detach().cpu()
            pred = pred.unsqueeze(0)
            loss = criterion(out, mask)

            total_iou += iou(pred, mask)

            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * input.size(0)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Average Loss: ({3})\t'.format(
                epoch, i, len(dataloaders["train"]), running_loss / (i + 1)))
            checkpoint(model.state_dict(), 'loss', running_loss / (i + 1), '{}_{}_{}'.format(
                epoch, i, len(dataloaders["train"])))

    scheduler.step()

    epoch_loss = running_loss / len(dataloaders["train"])
    mean_iou = total_iou / len(dataloaders["train"])
    print('{} Loss: {:.4f} Mean IOU: {:.4f}'.format(
        "train", epoch_loss, mean_iou))

    return best_iou, best_model_wts


def validate(model, dataloaders, optimizer,
             criterion, device, best, epoch):
    best_iou, best_model_wts = best
    print_freq = 100

    model.eval()

    running_loss = 0.0
    total_iou = 0

    for i, (input, mask) in enumerate(dataloaders['val']):
        input = input.to(device)
        mask = mask.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            out = model(input)['out']
            pred = torch.argmax(out.squeeze(), dim=0).detach().cpu()
            pred = pred.unsqueeze(0)
            loss = criterion(out, mask)

            total_iou += iou(pred, mask)

        # statistics
        running_loss += loss.item() * input.size(0)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: ({3})\t'.format(
                epoch, i, len(dataloaders['val']), loss.item()))

    epoch_loss = running_loss / len(dataloaders["val"])
    mean_iou = total_iou / len(dataloaders["val"])
    print('{} Loss: {:.4f} Mean IOU: {:.4f}'.format(
        "val", epoch_loss, mean_iou))

    if mean_iou > best_iou:
        best_iou = mean_iou
        best_model_wts = copy.deepcopy(model.state_dict())
        checkpoint(best_model_wts, 'iou', iou, epoch)

    return best_iou, best_model_wts


def checkpoint(model_wts, metric, iou, epoch):
    print('Saving checkpoints...')

    if metric == 'iou:':
        torch.save(
            {'iou': iou},
            '{}/iou_epoch_{}.pth'.format("./history", epoch))
    elif metric == 'loss':
        torch.save(
            {'loss': iou},
            '{}/loss_epoch_{}.pth'.format("./history", epoch))

    torch.save(
        model_wts,
        '{}/model_epoch_{}.pth'.format("./history", epoch))


def main():
    root = "./datasets/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    datasets = {split: MilaLogoDataset(root=root, split=split,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize]))
                for split in ['train', 'val']}
    dataloaders = {split: torch.utils.data.DataLoader(datasets[split], batch_size=20)
                   for split in ['train', 'val']}

    model_ft = get_fcn()

    model_ft = model_ft.to(device)

    best = (0, model_ft)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    for epoch in range(2):
        train(model_ft, dataloaders, optimizer_ft, criterion, device, best, epoch)
        validate(model_ft, dataloaders, optimizer_ft, criterion, device, best, epoch)


if __name__ == '__main__':
    # argparse.ArgumentParser
    main()
