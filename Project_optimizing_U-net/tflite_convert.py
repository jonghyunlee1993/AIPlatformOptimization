import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from src.model_keras import *
from src.data_loader_torch import *
from timeit import default_timer as timer
from datetime import timedelta

# os.environ["CUDA_VISIBLE_DEVICES"]=""

def dice_coef(y_true, y_pred, smooth=100):
    
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)

    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))


model = tf.keras.models.load_model(
    'weights/unet_best_keras', 
    custom_objects={'dice_coef_loss' : dice_coef_loss, 
                    'dice_coef' : dice_coef}
    )


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('weights/model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="weights/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_scan_dir = 'src/test/scan/'
test_mask_dir = 'src/test/mask/'
num_workers = 3 

test_df = make_data_df(test_scan_dir, test_mask_dir)
test_dataset = BrainMriDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
print("Test Loader Len: ", len(test_loader))

total_time = 0
total_dc = 0
n_epoch = 1

for i in range(n_epoch):
    print("Epoch ", i)
    epoch_time = 0
    epoch_dc = 0

    for batch in tqdm(test_loader):
        img = batch[0]
        mask = batch[1]
        
        img = np.array(img).astype(np.float32)
        
        mask = np.array(mask).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], img)

        start = timer()
        
        interpreter.invoke()

        end = timer()
        epoch_time += (end - start)

        output = interpreter.get_tensor(output_details[0]['index'])

        epoch_dc += dice_coef(output, mask)

    epoch_time /= len(test_loader)
    epoch_dc /= len(test_loader)
    
    total_time += epoch_time
    total_dc += epoch_dc

    print("\tAverage Time: ", timedelta(seconds=epoch_time))
    print("\tAverage Dice: ", epoch_dc)


total_time /= n_epoch
total_dc /= n_epoch

print("\n%d Epoch Average Time: "%n_epoch, timedelta(seconds=total_time))
print("%d Epoch Average Dice: "%n_epoch, total_dc)