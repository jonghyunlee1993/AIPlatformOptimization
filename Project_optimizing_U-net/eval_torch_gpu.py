import os
import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_executor as runtime
from timeit import default_timer as timer
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from src.model_torch import *
from src.data_loader_torch import *


def get_network(net):
    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 256, 256]
    output_shape = [1, 1, 256, 256]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(net, input_data).eval()

    input_name = "input_2"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    return mod, params, input_shape, output_shape

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
# num_threads = multiprocessing.cpu_count()
# os.environ["TVM_NUM_THREADS"] = str(num_threads)

def dice_coef(inputs, targets, smooth=1):

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()                            
    dice_coef = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return dice_coef

if __name__ == "__main__":
    test_scan_dir = 'src/test/scan/'
    test_mask_dir = 'src/test/mask/'
    dir_checkpoint = 'weights/unet_best_torch'
    out_threshold = 0.5

    batch_size = 1
    dtype = "float32"
    model_name = "unet_torch_gpu"
    log_file = "%s.log" % model_name
    # graph_opt_sch_file = "%s_graph_opt.log" % model_name

    # Set the input name of the graph
    # For ONNX models, it is typically "0".
    input_name = "input_2"

    ## Load Dataset and Model
    num_workers = 3 # System Suggested
    
    device = torch.device('cpu')
    model = torch.load('weights/unet_best_torch').to(device)
    model.eval()

    test_df = make_data_df(test_scan_dir, test_mask_dir)
    test_dataset = BrainMriDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    print("Test Loader Len: ", len(test_loader))

    mod, params, input_shape, out_shape = get_network(model)

    target = tvm.target.cuda()

    # compile kernels with graph-level best records
    # with autotvm.apply_graph_best(graph_opt_sch_file):
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # upload parameters to device
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))

        total_time = 0
        total_dc = 0
        n_epoch = 5
        for i in range(n_epoch):
            print("Epoch ", i)
            epoch_time = 0
            epoch_dc = 0

            for batch in test_loader:
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