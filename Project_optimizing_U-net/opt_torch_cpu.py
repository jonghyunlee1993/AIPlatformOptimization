import os
import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime
from tvm.contrib.download import download_testdata

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_torch import *


def get_input():
    from PIL import Image
    from matplotlib import pyplot as plt
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img_path = "src/sample_image.tif"
    img = Image.open(img_path).resize((256, 256))

    # input preprocess
    data = np.array(img)[np.newaxis, :].astype("float32")
    data = preprocess_input(data).transpose([0, 3, 1, 2])
    data = torch.from_numpy(np.flip(data, axis=0).copy())
    
    return data.to(device="cpu", dtype=torch.float32)

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""

    input_shape  = (batch_size, 3, 256, 256)
    output_shape = (batch_size, 1, 256, 256)

    input_data = get_input()
    shape_dict = [("input_2", input_data.shape)]

    model = torch.load('weights/unet_best_pytorch').to('cpu')
    model.eval()
    scripted_model = torch.jit.trace(model, input_data).eval()

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)

    return mod, params, input_shape, output_shape


# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("input_2", data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

if __name__ == "__main__":
    #### DEVICE CONFIG ####
    target = "llvm"

    #### TUNING OPTION ####
    network = "unet_torch_cpu"
    log_file = "%s.log" % network
    dtype = "float32"

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 500,
        "early_stopping": 200,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }

    num_threads = 7
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

    tune_and_evaluate(tuning_option)