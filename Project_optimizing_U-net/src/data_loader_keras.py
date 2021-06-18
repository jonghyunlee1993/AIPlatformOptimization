import os
import cv2
import glob
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_generator(data_frame, batch_size, aug_dict,
        image_color_mode = "rgb",
        mask_color_mode = "grayscale",
        image_save_prefix = "image",
        mask_save_prefix = "mask",
        save_to_dir = None,
        target_size = (256, 256),
        seed = 1234):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen  = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        
        yield (img,mask)


def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)


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