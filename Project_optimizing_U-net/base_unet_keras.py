import os 
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model_keras import *
from src.data_loader_keras import *
# from src.data_loader_torch import *



from timeit import default_timer as timer
from datetime import timedelta


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

    model = tf.keras.models.load_model(
        'weights/unet_best_keras', 
        custom_objects={'dice_coef_loss' : dice_coef_loss, 
                        'dice_coef' : dice_coef}
        )
    
    optimizer = Adam(lr=0.0005)

    model.compile(optimizer=optimizer, 
              loss=dice_coef_loss, 
              metrics=["binary_accuracy", tf.keras.metrics.MeanIoU(num_classes=2), dice_coef])

    test_scan_dir = 'src/test/scan/'
    test_mask_dir = 'src/test/mask/'
    test_df = make_data_df(test_scan_dir, test_mask_dir)

    valid_generator_args = dict(rotation_range=0,
                            width_shift_range=0,
                            height_shift_range=0,
                            shear_range=0,
                            zoom_range=0,
                            horizontal_flip=False,
                            fill_mode='nearest')

    test_gen = train_generator(test_df, 1,
                                valid_generator_args,
                                target_size=(256, 256))

    from tqdm import tqdm
    # os.environ["CUDA_VISIBLE_DEVICES"]=''
    epoch_time = 0
    for i in tqdm(range(5)):
        start = timer()
        results = model.evaluate(test_gen, steps=len(test_df))
        end = timer()
        epoch_time += (end - start)

        print(results)
    print(epoch_time / 323 * 5)
