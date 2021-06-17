import os
import cv2
import glob
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A

class BrainMriDataset(Dataset):
    def __init__(self, df):
        self.transforms = A.Compose([
            A.augmentations.Resize(width=256, height=256, p=1.0),
            A.augmentations.Normalize(p=1.0),
            # ToTensor(),
        ])

        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']   
        
        return image, mask

def make_data_df(image_path, mask_path):
    train_files = glob.glob(image_path + "*.tif")
    mask_files  = glob.glob(mask_path + "*.tif") 

    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})

    return df


if __name__ == "__main__":
    df = make_data_df(
        "/home/ubuntu/Workspaces/AIPlatformOptimization/Project_optimizing_U-net/src/test/scan/", 
        "/home/ubuntu/Workspaces/AIPlatformOptimization/Project_optimizing_U-net/src/test/mask/"
    )

    print(df)