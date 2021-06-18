import os
import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime
from tvm.contrib.download import download_testdata

from timeit import default_timer as timer
from datetime import timedelta

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model_keras import *
# from src.data_loader_keras import *
from src.data_loader_torch import *


def get_input_shape():
    from PIL import Image
    from matplotlib import pyplot as plt
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img_path = "src/sample_image.tif"
    img = Image.open(img_path).resize((256, 256))

    # input preprocess
    data = np.array(img)[np.newaxis, :].astype("float32")
    data = preprocess_input(data).transpose([0, 3, 1, 2])

    return data.shape

def get_network():
    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 256, 256]
    output_shape = [1, 1, 256, 256]
    
    model = tf.keras.models.load_model(
        'weights/unet_best_keras', 
        custom_objects={'dice_coef_loss' : dice_coef_loss, 
                        'dice_coef' : dice_coef}
        )

    data_shape = get_input_shape()
    shape_dict = {"input_2": data_shape}
    mod, params = relay.frontend.from_keras(model, shape_dict)

    return mod, params, input_shape, output_shape


# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
# num_threads = multiprocessing.cpu_count()
# os.environ["TVM_NUM_THREADS"] = str(num_threads)

def dice_coef(y_true, y_pred, smooth=100):
    
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)

    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

if __name__ == "__main__":
    test_scan_dir = 'src/test/scan/'
    test_mask_dir = 'src/test/mask/'
    dir_checkpoint = 'weights/unet_best_keras'

    batch_size = 1
    dtype = "float32"
    model_name = "unet_keras_cpu"
    log_file = "%s.log" % model_name
    # graph_opt_sch_file = "%s_graph_opt.log" % model_name

    # Set the input name of the graph
    # For ONNX models, it is typically "0".
    input_name = "input_2"

    ## Load Dataset and Model
    num_workers = 3 # System Suggested

    test_df = make_data_df(test_scan_dir, test_mask_dir)

    # valid_generator_args = dict(rotation_range=0,
    #                         width_shift_range=0,
    #                         height_shift_range=0,
    #                         shear_range=0,
    #                         zoom_range=0,
    #                         horizontal_flip=False,
    #                         fill_mode='nearest')

    # test_gen = train_generator(test_df, 1,
    #                             valid_generator_args,
    #                             target_size=(256, 256))

    test_dataset = BrainMriDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    print("Test Loader Len: ", len(test_loader))

    mod, params, input_shape, out_shape = get_network()

    mod, params, input_shape, out_shape = get_network()
    target = "llvm"
    target = tvm.target.Target(target)

    # compile kernels with graph-level best records
    # with autotvm.apply_graph_best(graph_opt_sch_file):
    from tqdm import tqdm

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # upload parameters to device
        dev = tvm.cpu()
        module = runtime.GraphModule(lib["default"](dev))

        total_time = 0
        total_dc = 0
        n_epoch = 5

        for i in range(n_epoch):
            print("Epoch ", i)
            epoch_time = 0
            epoch_dc = 0

            for batch in tqdm(test_loader):
                img = batch[0]
                mask = batch[1]
                # img = img.to(device=device, dtype=torch.float32)
                # img = img.numpy()
                img = np.array(img).astype(np.float32)

                # mask_type = torch.float32
                # mask = mask.to(device=device, dtype=mask_type)
                mask = np.array(mask).astype(np.float32)
                module.set_input(input_name, tvm.nd.array(img.astype(dtype)))

                start = timer()
                module.run()
                end = timer()
                epoch_time += (end - start)

                output = module.get_output(0)
                output = output.numpy()

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