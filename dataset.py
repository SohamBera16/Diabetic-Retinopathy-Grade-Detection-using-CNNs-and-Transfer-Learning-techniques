import pandas as pd
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import time
import torch
from _operator import truediv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, lbl_file, transform=None, cross_num=None, cross_ids=None, norm_per_image=False, norm_per_dataset=False, augment=None):
        # Initialize dataframe
        self.df = pd.read_csv(lbl_file)

        # Initialize list of image names and temporary images and labels list.
        # NumPy vectorization (to_numpy()) is faster than Pandas vectorization.
        img_names = self.df['Image name'].to_numpy().tolist()
        imgs_tmp, lbls_tmp = [], self.df['Retinopathy grade'].to_numpy().tolist()

        # Initialize images and labels list and transform.
        self.imgs, self.lbls = [], []
        self.transform = transform

        # Load images, transform them and store in temporary image list
        for name in img_names:
            img = Image.open(img_dir + name + '.jpg')
            if self.transform:
                img = self.transform(img)

                # Per Image Normalization
                # Every image is normalized based on its own mean and std.
                if norm_per_image:
                    mean = img.mean([1, 2])
                    std = img.std([1, 2])
                    norm = transforms.Normalize(mean, std)
                    img = norm(img)
                
                # Per Dataset Normalization
                # Every image is normalized based on the mean and std of the whole dataset.
                if norm_per_dataset:
                    norm = transforms.Normalize([0.4346, 0.2110, 0.0705], [
                                                0.3068, 0.1637, 0.0834])
                    img = norm(img)

            imgs_tmp.append(img)

        # Split data for cross validation.
        if cross_num is not None:
            total_length = len(imgs_tmp)
            length = int(truediv(total_length, cross_num))

            for cross_id in cross_ids:
                low_r = cross_id - 1
                if cross_id == cross_num:
                    self.imgs.extend(imgs_tmp[(low_r)*length:])
                    self.lbls.extend(lbls_tmp[(low_r)*length:])
                else:
                    self.imgs.extend(
                        imgs_tmp[(low_r)*length:(cross_id)*length])
                    self.lbls.extend(
                        lbls_tmp[(low_r)*length:(cross_id)*length])
        else:
            self.imgs = imgs_tmp
            self.lbls = lbls_tmp

        # Data augmentation
        # Every image is transformed randomly, and added onto the loaded data.
        # augment = 5 would mean that this is repeated 5 times.
        # Important: The images are not replaced, this will increase the size of the data.
        if augment is not None:
            tr_list = [
                transforms.RandomRotation(degrees=(-40, 40)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(128)
            ]

            for a in range(augment):
                for i in range(len(imgs_tmp)):
                    img = transforms.ToPILImage()(img)
                    img = transforms.RandomChoice(tr_list)(img)
                    img = transforms.ToTensor()(img)
                    self.imgs.append(img)
                    self.lbls.append(lbls_tmp[i])

    def __getitem__(self, idx):
        img = self.imgs[idx]
        lbl = self.lbls[idx]
        return img, lbl

    def __len__(self):
        return len(self.lbls)


# For testing purposes
if __name__ == '__main__':

    tr = transforms.Compose([
        transforms.Resize((64, 64))
    ])
    
    # change the directory to the data directory
    dir = '../IDRID_dataset/'
    img_dir = dir + 'images/test/'
    lbl_file = dir + 'labels/test.csv'

    start = time.time()
    CD = Dataset(img_dir, lbl_file, tr, cross_num=None, cross_ids=[5])
    end = time.time()
    imgs, lbls = next(iter(CD))

    print(f'Length: {len(CD)}')
    print(f'Data load time: {end - start}')
    print(f'Image shape: {imgs.shape}')
