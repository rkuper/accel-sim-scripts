"""""""""""""""""""""""""""""""""""""""""""""""""""
Address Checker Script
===================================================
Filename: address_compare.py
Author: Reese Kuper
Purpose: Compare address between kernel traces and
the simulated addresses
"""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import sys
import argparse
import re
from subprocess import Popen, PIPE
import pprint

"""""""""
GLOBALS
"""""""""
kernel_traces = {}
sim_stats = {}



"""
Larger wrapper for obtaining trace addresses
"""
def get_traces(device_number, cuda_version, benchmark, params, start, end):
    # Find beginning accel-sim-framework directory
    find = Popen(['find', '../', '-maxdepth', '4', '-name', 'accel-sim-framework'], stdout = PIPE)
    accelsim_dir = find.communicate()[0].decode('ascii').rstrip()
    if accelsim_dir == None:
        print("Could not find accel-sim-framework")
        return

    device_dir = accelsim_dir + "/hw_run/traces/device-" + str(device_number)
    if not os.path.exists(device_dir):
        print("Could not find GPU device number in accel-sim-framework/hw_run/traces/device-#")

    cuda_dir = device_dir + "/" + str(cuda_version)
    if not os.path.exists(cuda_dir):
        print("Could not find cuda version in accel-sim-framework/hw_run/traces/device-#/<CUDA>")

    benchmark_dir = cuda_dir + "/" + benchmark
    if not os.path.exists(benchmark_dir):
        print("Could not find benchmark in accel-sim-framework/hw_run/traces/device-#/<CUDA>/<BENCHMARK>")

    # TODO: Get through grepping under each argument in params
    params_dir = benchmark_dir + "/" + params
    if not os.path.exists(params_dir):
        print("Could not find specific test in accel-sim-framework/hw_run/traces/device-#/<CUDA>/<BENCHMARK>/<TEST>")

    traces_dir = params_dir + "/traces"

    # Kernel numbers help get a list of all the kernels traced
    kernel_numbers = []
    kernel_offset = 0
    for subdir, dirs, files in os.walk(traces_dir):

        # Get the kernel numbers and the first traced kernel for the offset
        number_of_kernels = sum('kernel-' in s for s in files)
        for kernel in files:
            if len(re.findall("\d+", kernel)) > 0:
                kernel_numbers.append(re.findall("\d+", kernel)[0])
        kernel_offset = int((sorted(kernel_numbers)[0]))

        # Begin parsing each trace
        for i in range(start + kernel_offset, min(number_of_kernels + kernel_offset, end)):

            # Get kernel info
            temp_kernel_name = ""
            kernel_id = 0
            kernel_name = "kernel-"
            trace_inst_cnt = 0
            current_block = "0,0,0"
            current_warp = "warp-"
            with open((traces_dir + "/kernel-" + str(i) + ".traceg"), 'r', encoding = 'utf-8') as trace:
                for line in trace:
                    # Gather kernel info
                    if "kernel id =" in line:
                        kernel_id = int(line.split(' ')[-1])
                        kernel_name = kernel_name + str(kernel_id)
                        kernel_traces[kernel_name] = {}
                        kernel_traces[kernel_name]["id"] = kernel_id
                        kernel_traces[kernel_name]["mem_addrs"] = []
                        kernel_traces[kernel_name]["num_insts"] = 0
                        kernel_traces[kernel_name]["num_mem_insts"] = 0
                    elif "kernel name =" in line:
                        temp_kernel_name = line.split(' ')[-1]
                    elif "grid dim =" in line:
                        grid_xyz = line[line.index('(') + 1: len(line) - 2]
                        grid_xyz = grid_xyz.split(',')
                        grid_dim = (int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2]))
                        kernel_traces[kernel_name]["grid_dim"] = grid_dim
                    elif "block dim =" in line:
                        block_xyz = line[line.index('(') + 1: len(line) - 2]
                        block_dim = (int(block_xyz[0]), int(block_xyz[1]), int(block_xyz[2]))
                        kernel_traces[kernel_name]["block_dim"] = block_dim
                        kernel_traces[kernel_name]["thread_blocks"] = {}
                    elif "shmem =" in line:
                        kernel_traces[kernel_name]["shmem"] = (line.split(' ')[-1]).rstrip()
                    elif "nregs =" in line:
                        kernel_traces[kernel_name]["nregs"] = (line.split(' ')[-1]).rstrip()
                    elif "shmem base_addr =" in line:
                        kernel_traces[kernel_name]["shmem_base_addr"] = (line.split(' ')[-1]).rstrip()
                    elif "local mem base_addr =" in line:
                        kernel_traces[kernel_name]["local_mem_base_addr"] = (line.split(' ')[-1]).rstrip()

                    # Begin preparing the specific thread block and warp
                    elif "thread block = " in line:
                        current_block = (line.split(' ')[-1]).rstrip()
                        kernel_traces[kernel_name]["thread_blocks"][current_block] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["mem_addrs"] = []
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["num_insts"] = 0
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["num_mem_insts"] = 0
                    elif "warp = " in line:
                        current_warp = "warp-" + (line.split(' ')[-1]).rstrip()
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_addrs"] = []
                    elif "insts = " in line:
                        warp_insts = int(line.split(' ')[-1])
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["num_insts"] = warp_insts
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["num_mem_insts"] = 0
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["num_insts"] += warp_insts
                        kernel_traces[kernel_name]["num_insts"] += warp_insts

                    # Start the actual instruction parsing
                    elif "LD" in line or "ST" in line:
                        # Add line
                        line_fields = line.split(' ')
                        inst_name = kernel_name + "_0x" + line_fields[0]
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["line"] = line.rsplit()

                        # Add the PC and mask values
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["pc"] = hex(int(line_fields[0], 16))
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["mask"] = hex(int(line_fields[1], 16))

                        # Add memory instruction type
                        inst_type = ""
                        if "LD" in line:
                            inst_type = line_fields[4].split('.')[0]
                        else:
                            inst_type = line_fields[3].split('.')[0]
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["type"] = inst_type

                        # Add address info
                        address = hex(0)
                        if inst_type != "LDC":
                            address = hex(int(line_fields[9], 16))
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["addr"] = address
                        if address != 0:
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_addrs"].append(address)
                            kernel_traces[kernel_name]["thread_blocks"][current_block]["mem_addrs"].append(address)
                            kernel_traces[kernel_name]["mem_addrs"].append(address)

                        # Increment instruction counts
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["num_mem_insts"] += 1
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["num_mem_insts"] += 1
                        kernel_traces[kernel_name]["num_mem_insts"] += 1

    return



"""
Parse the arguments, then run through tracer
"""
def arg_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", help = "Specify the benchmark (ex. rnn_bench from Deepbench)")
    parser.add_argument("-p", "--params", help = "Specify the benchmark parameters delimited by '_' (ex. train_half_8_8_1_lstm)")
    parser.add_argument("-s", "--start", help = "Which kernel to start tracing from", default=0)
    parser.add_argument("-e", "--end", help = "Which kernel to end tracing on", default=float('inf'))
    args = parser.parse_args()

    # Get the GPU device number
    lspci = Popen("lspci", stdout=PIPE)
    grep = Popen(['grep', 'VGA'], stdin = lspci.stdout, stdout = PIPE)
    cut_col = Popen(['cut', '-d', ':', '-f', '2'], stdin = grep.stdout, stdout = PIPE)
    cut_space = Popen(['cut', '-d', ' ', '-f', '1'], stdin = cut_col.stdout, stdout = PIPE)
    cut_dec = Popen(['cut', '-d', '.', '-f', '1'], stdin = cut_col.stdout, stdout = PIPE)
    # device_number = int(float(cut_dec.communicate()[0].decode('ascii').rstrip()))
    device_number = cut_dec.communicate()[0].decode('ascii').rstrip()
    device_number = 0 if device_number == '' else int(device_number)

    # Get the cuda version for pathing
    cuda_version = os.getenv('CUDA_INSTALL_PATH').split('-')[-1]

    # Make sure the kernel values are normal
    if args.end < args.start:
        print("End kernel should not be earlier than the starting kernel")

    get_traces(device_number, cuda_version, args.benchmark, args.params, args.start, args.end)
    return



"""
Main state
"""
if __name__=="__main__":

    arg_wrapper()
