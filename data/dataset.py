import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class MilaLogoDataset(Dataset):

    def __init__(self, root, split, transform, verbose=False):
        self.mask_arr = sorted(glob.glob(str(root) + str(split) + str("/mask/*")))
        self.image_arr = sorted(glob.glob(str(root) + str(split) + str("/img/*")))
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.mask_arr)

    def __getitem__(self, idx):
        single_image_name = self.image_arr[idx]
        img_as_img = Image.open(single_image_name)
        if self.verbose:
            img_as_img.show()
        img_as_np = np.asarray(img_as_img)

        #img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        # img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor


        if self.transform:
            img_as_tensor = self.transform(img_as_np)

        single_mask_name = self.mask_arr[idx]
        msk_as_img = Image.open(single_mask_name)
        if self.verbose:
            msk_as_img.show()
        msk_as_np = np.asarray(msk_as_img)

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np / 255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor

        return img_as_tensor, msk_as_tensor


if __name__ == "__main__":
    data = MilaLogoDataset('datasets/full/', True)

    print(data.mask_arr[0])
    print(data.image_arr[0])

    img, mask = data.__getitem__(0)
    print(img)
