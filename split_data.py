import numpy as np
import os
import glob
import shutil

def make_img_mask_subdirs(path):
    os.makedirs(os.path.join(path, "mask"))
    os.makedirs(os.path.join(path, "img"))

if __name__ == '__main__':
    root = "./datasets/"
    root_full = os.path.join(root, "full")

    masks = sorted(glob.glob(str(root_full) + str("mask/*")))
    images = sorted(glob.glob(str(root_full) + str("img/*")))

    val_split = .1
    test_split = .1
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(masks)
    indices = list(range(dataset_size))
    val_split_idx = int(np.floor(test_split * dataset_size))
    test_split_idx = int(np.floor((val_split + test_split) * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[test_split_idx:], indices[:val_split_idx], indices[val_split_idx:test_split_idx]

    for split in ["train", "val", "test"]:
        split_path = os.path.join(root, split)
        if not os.path.exists(split_path):
            make_img_mask_subdirs(split_path)

    for n, (mask, img) in enumerate(zip(masks, images)):
        if n in train_indices:
            path = os.path.join(root, "train")
        elif n in test_indices:
            path = os.path.join(root, "test")
        elif n in val_indices:
            path = os.path.join(root, "val")

        shutil.copy(masks[n], os.path.join(path, "mask"))
        shutil.copy(images[n], os.path.join(path, "img"))
