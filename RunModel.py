#! /Users/yannic/opt/anaconda3/envs/tvm/bin/python3
# -*- coding: UTF-8 -*-

import argparse
import os
import logging
import time

import numpy as np

import onnx

import tvm
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler

from tvm import autotvm
from tvm.runtime import profiler_vm
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.relay.testing import mlp
from tvm.contrib import ndk

from PIL import Image

def RunModelOnMobile():
    onnx_model = onnx.load("gemm_v2.onnx")
    input = np.load("CustomNetInput.npy")

    input_name = 'video_frame_input'

    shape_dict = {input_name: input.shape}

    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    target = tvm.target.Target('opencl -device=adreno')

    with tvm.transform.PassContext(opt_level=3):
        module = relay.build_module.build(sym, target, params=params)

    module.export_library('gemm_v2.so', ndk.create_shared)

def RunModel(tune_file_name, profiler, benchmark):
    onnx_model = onnx.load("gemm_v2.onnx")
    input = np.load("CustomNetInput.npy")

    target = tvm.target.Target("llvm", host="llvm")

    input_name = 'video_frame_input'

    shape_dict = {input_name: input.shape}

    # breakpoint()
    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    dev = tvm.cpu(0)

    dtype = 'float32'

    data = tvm.nd.array(input.astype(dtype))

    # benchmark
    if benchmark:
        with auto_scheduler.ApplyHistoryBest(tune_file_name) as apply_history_best:
            # with relay.build_config(opt_level=3):
            with tvm.transform.PassContext(opt_level=3):
                # breakpoint()
                module = relay.build(sym, target, params=params)
                lib = module.get_lib()
                print("build lib: ", lib)

            # Save the optimized graph to a file
            with open("optimized_graph.json", "w") as f:
                f.write(module.get_graph_json())

            # breakpoint()
            graph_module = tvm.contrib.graph_executor.GraphModule(module["default"](dev))

            graph_module.set_input(input_name, data)
            print(graph_module.benchmark(dev, func_name="main", number=100, repeat=3, end_to_end=True))


    # profile_vm
    if profiler:
        with auto_scheduler.ApplyHistoryBest(tune_file_name) as apply_history_best:
            with tvm.transform.PassContext(opt_level=3):
                exe = relay.vm.compile(sym, target, params=params)
                vm = profiler_vm.VirtualMachineProfiler(exe, dev)

                # breakpoint()
                report = vm.profile([data], func_name="main", number=100, repeat=3, end_to_end=True)

                with open("vm_profile_report.txt", "w") as f:
                    f.write(str(report))

                print(report)

    # if profiler:
    #     with tvm.transform.PassContext(opt_level=3):
    #         exe = relay.vm.compile(sym, target, params=params)
    #         vm = profiler_vm.VirtualMachineProfiler(exe, dev)

    #         # breakpoint()
    #         report = vm.profile([data], func_name="main", number=100, repeat=3, end_to_end=True)

    #         with open("vm_profile_report.txt", "w") as f:
    #             f.write(str(report))

    #         print(report)

    # debug
    if False:
        with tvm.transform.PassContext(opt_level=3):
            exe = relay.build(sym, target, params=params)
            gr = debug_executor.create(exe.get_graph_json(), exe.lib, dev)
            report = gr.profile(data=data)
            print(report)

        with open("debug_profile_report.txt", "w") as f:
            f.write(str(report))

        print(report)




def RunModelWithTune(tune_file_name):
    onnx_model = onnx.load("gemm_v2.onnx")
    input=np.load("CustomNetInput.npy")

    target=tvm.target.Target("llvm")

    input_name='video_frame_input'

    shape_dict = {input_name:input.shape}

    mod,params=relay.frontend.from_onnx(onnx_model, shape_dict)

    # Extract tasks from the network
    tasks = auto_scheduler.extract_tasks(mod["main"], params, target)

    # Tune the tasks
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=64,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(),
        runner=auto_scheduler.LocalRunner(number=10, repeat=3, timeout=4),
        measure_callbacks=[auto_scheduler.RecordToFile(tune_file_name)],
    )

    scheduler = auto_scheduler.TaskScheduler(tasks[0])
    scheduler.tune(tune_option)

    # Compile with tuning data
    with auto_scheduler.ApplyHistoryBest(tune_file_name):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": False}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Set inputs
    module.set_input(input_name, tvm.nd.array(input.astype('float32')))

    # Run
    module.run()

    # Get output
    output = module.get_output(0)

    print("Run Finish.")



def main():

    pareser = argparse.ArgumentParser(description="Test TVM.",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    pareser.add_argument('-tune', '--tune_model',
                         help='tvm is tune op get best performance.',
                         default=False, action='store_true')

    pareser.add_argument('-profiler', '--profiler',
                        help='tvm run profiler_vm profiling op.',
                        default=False, action='store_true')

    pareser.add_argument('-benchmark', '--benchmark',
                        help='tvm benchmark model.',
                        default=False, action='store_true')
    
    pareser.add_argument('-mobile', '--mobile',
                        help='tvm mobile model.',
                        default=False, action='store_true')
    
    args = pareser.parse_args()

    # logging.getLogger('tvm').setLevel(logging.DEBUG)

    tune_file_name = 'gemm_v2.json'

    if args.tune_model:
        if os.path.exists(tune_file_name):
            RunModel(tune_file_name, args.profiler, args.benchmark)
        else:
            RunModelWithTune(tune_file_name)

    elif args.mobile:
        RunModelOnMobile()
    else:
        RunModel(tune_file_name, args.profiler, args.benchmark)


if __name__ == '__main__':
    main()