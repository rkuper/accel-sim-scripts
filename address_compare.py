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
import glob
import pprint

"""""""""
GLOBALS
"""""""""
kernel_traces = {}
sim_stats = {}



"""
Gets addresses from the simulated stats file
"""
def get_sim_stats(cuda_version, benchmark, params, sass):
    # Find beginning accel-sim-framework directory
    accelsim_dir = get_accel_sim()
    if accelsim_dir == None:
        print("Could not find accel-sim-framework")
        return

    run_dir = accelsim_dir + "/sim_run_" + str(cuda_version)
    if not os.path.exists(run_dir):
        print("Could not find sim_run_<CUDA> in accel-sim-framework/sim_run_<CUDA>/. Did you simulate yet?")
        return

    benchmark_dir = run_dir + "/" + benchmark
    if not os.path.exists(benchmark_dir):
        print("Could not find benchmark in accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>")
        return

    # The actual test is a bit harder to ensure while the params are in any order
    params_dir = get_test(benchmark_dir, params)
    if params_dir == None:
        print("Could not find specific test in accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>/<TEST>")
        return
    print("\nParsing Simulation Output\n=========================")
    print("Using test: " + params_dir[params_dir.rfind('/') + 1:])

    sass_dir = params_dir + "/" + sass + "-SASS"
    if not os.path.exists(sass_dir):
        print("Could not find sass in accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>/<TEST>/<SASS>")
        return

    # Now getting the specific test simulation output
    sim_file = get_test(sass_dir, (benchmark + "_" + params))
    if sim_file == None:
        print("Could not find simulation log in accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>/<TEST>/<SASS>/<LOG>")
        return

    # Begin parsing the sim output
    # Get kernel info
    temp_kernel_name = ""
    kernel_id = 0
    kernel_name = "kernel-"
    with open(sim_file, 'r', encoding = 'utf-8') as sim_file:
        for line in sim_file:
            # Gather kernel info
            if "kernel id =" in line:
                if kernel_name != "kernel-":
                    print(' Done')
                kernel_id = int(line.split(' ')[-1])
                kernel_name = kernel_name + str(kernel_id)
                sim_stats[kernel_name] = {}
                sim_stats[kernel_name]["id"] = kernel_id
                sim_stats[kernel_name]["mem_addrs"] = []
                sim_stats[kernel_name]["num_insts"] = 0
                sim_stats[kernel_name]["num_mem_insts"] = 0
                sim_stats[kernel_name]["mem_insts"] = {}
                print("Parsing kernel " + str(kernel_id) + "...", end = '')
            elif "gpu_sim_cycle" in line:
                sim_stats[kernel_name]["num_insts"] = int(line.split(' ')[-1])
            elif "kernel name =" in line:
                temp_kernel_name = line.split(' ')[-1]
            elif "grid dim =" in line:
                grid_xyz = line[line.index('(') + 1: len(line) - 2]
                grid_xyz = grid_xyz.split(',')
                grid_dim = (int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2]))
                sim_stats[kernel_name]["grid_dim"] = grid_dim
            elif "block dim =" in line:
                block_xyz = line[line.index('(') + 1: len(line) - 2]
                block_xyz = block_xyz.split(',')
                block_dim = (int(block_xyz[0]), int(block_xyz[1]), int(block_xyz[2]))
                sim_stats[kernel_name]["block_dim"] = block_dim
                sim_stats[kernel_name]["thread_blocks"] = {}
            elif "local mem base_addr =" in line:
                sim_stats[kernel_name]["local_mem_base_addr"] = (line.split(' ')[-1]).rstrip()

            # Begin parsing mem instructions
            elif "mf:" in line:

                # Grab only the global memory instructions
                if 'GLOBAL' not in line:
                    continue

                # Clean up the list a little bit
                line_fields = line.strip().replace(',', '').split(' ')
                if '' in line_fields:
                    line_fields.remove('')

                sim_stats[kernel_name]["line"] = line.strip()
                sim_stats[kernel_name]["pc"] = line_fields[13]
                sim_stats[kernel_name]["type"] = line_fields[6]
                sim_stats[kernel_name]["mask"] = line_fields[14][line_fields[14].index('[') + 1:line_fields[14].index(']') - 1]
                sim_stats[kernel_name]["warp"] = line_fields[3][line_fields[3].index('w') + 1:]
                # sim_stats[kernel_name]["address"] = line_fields[]
                # sim_stats["mem_addr"] = line_fields[]


                # sim_stats[kernel_name]["mem_insts"]
        print(' Done')
    return



"""
Gets addresses of all specified trace files
"""
def get_traces(device_number, cuda_version, benchmark, params, start, end):
    # Find beginning accel-sim-framework directory
    accelsim_dir = get_accel_sim()
    if accelsim_dir == None:
        print("Could not find accel-sim-framework")
        return

    device_dir = accelsim_dir + "/hw_run/traces/device-" + str(device_number)
    if not os.path.exists(device_dir):
        print("Could not find GPU device number in accel-sim-framework/hw_run/traces/device-#")
        return

    cuda_dir = device_dir + "/" + str(cuda_version)
    if not os.path.exists(cuda_dir):
        print("Could not find cuda version in accel-sim-framework/hw_run/traces/device-#/<CUDA>")
        return

    benchmark_dir = cuda_dir + "/" + benchmark
    if not os.path.exists(benchmark_dir):
        print("Could not find benchmark in accel-sim-framework/hw_run/traces/device-#/<CUDA>/<BENCHMARK>")
        return

    # The actual test is a bit harder to ensure while the params are in any order
    params_dir = get_test(benchmark_dir, params)
    if params_dir == None:
        print("Could not find specific test in accel-sim-framework/hw_run/traces/device-#/<CUDA>/<BENCHMARK>/<TEST>")
        return
    print("\nParsing Kernel Traces\n=====================")
    print("Using test: " + params_dir[params_dir.rfind('/') + 1:])

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
            current_block = "0,0,0"
            current_warp = "warp-"
            with open((traces_dir + "/kernel-" + str(i) + ".traceg"), 'r', encoding = 'utf-8') as trace:
                print("Parsing kernel " + str(i) + "...", end = '')
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
                        block_xyz = block_xyz.split(',')
                        block_dim = (int(block_xyz[0]), int(block_xyz[1]), int(block_xyz[2]))
                        kernel_traces[kernel_name]["block_dim"] = block_dim
                        kernel_traces[kernel_name]["thread_blocks"] = {}
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
                    elif "LDG" in line or "STG" in line:
                        # Add line
                        line_fields = line.split(' ')
                        inst_name = kernel_name + "_0x" + line_fields[0]
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["line"] = line.rsplit()

                        # Add the PC and mask values
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["pc"] = hex(int(line_fields[0], 16))
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["mask"] = hex(int(line_fields[1], 16))

                        # Add memory instruction type
                        mem_type = ""
                        if "LDG" in line:
                            mem_type = line_fields[4].split('.')[0]
                        else:
                            mem_type = line_fields[3].split('.')[0]
                        kernel_traces[kernel_name]["thread_blocks"][current_block]["warps"][current_warp]["mem_insts"][inst_name]["type"] = mem_type

                        # Add address info
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
                print(' Done')

    return



"""
Get accel-sim directory path
"""
def get_accel_sim():
    find = Popen(['find', '../', '-maxdepth', '4', '-name', 'accel-sim-framework'], stdout = PIPE)
    accelsim_dir = find.communicate()[0].decode('ascii').rstrip()
    return accelsim_dir



"""
Get the test name given the subdirectories (depth = 1) and the params
"""
def get_test(path, params_str):
    subdirs = os.listdir(path)
    if subdirs == []:
        return None

    # Sort by size of file
    subdirs_full = []
    for subdir in subdirs:
        subdirs_full.append((path + '/' + subdir))
    subdirs_full = sorted(subdirs_full, key=os.path.getsize, reverse=True)
    subdirs = [s[s.rfind('/') + 1:] for s in subdirs_full]

    # Default values then search each file/subdir
    params = re.split('_|-', params_str)
    best_match = path + "/" + subdirs[0]
    best_num = len(re.split('_|-',subdirs[0]))
    for subdir in subdirs:
        subdir_list = re.split('_|-', subdir)

        # Make sure that all parameters are at least in the possible test dir
        if not all(param in subdir_list for param in params):
            continue
        for param in params:
            if param not in subdir_list:

                # See if this test is better than the current best one
                if len(subdir_list) < best_num:
                    best_match = path + "/" + subdir
                    best_num = len(subdir_list)
                break
            subdir_list.remove(param)

            # This would be a perfect match
            if subdir_list == []:
                return (path + "/" + subdir)

    return best_match



"""
Parse the arguments, then run through tracer
"""
def arg_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", help = "Specify the benchmark (ex. rnn_bench from Deepbench)")
    parser.add_argument("-p", "--params", help = "Specify the benchmark parameters delimited by '_' (ex. train_half_8_8_1_lstm)")
    parser.add_argument("-a", "--sass", help = "Specify the SASS that the traces used (ex. QV100)")
    parser.add_argument("-s", "--start", help = "Which kernel to start parsing from", default=0)
    parser.add_argument("-e", "--end", help = "Which kernel to end parsing on", default=float('inf'))
    args = parser.parse_args()

    # Get the GPU device number
    lspci = Popen("lspci", stdout=PIPE)
    grep = Popen(['grep', 'VGA'], stdin = lspci.stdout, stdout = PIPE)
    cut_col = Popen(['cut', '-d', ':', '-f', '2'], stdin = grep.stdout, stdout = PIPE)
    cut_space = Popen(['cut', '-d', ' ', '-f', '1'], stdin = cut_col.stdout, stdout = PIPE)
    cut_dec = Popen(['cut', '-d', '.', '-f', '1'], stdin = cut_col.stdout, stdout = PIPE)
    device_number = cut_dec.communicate()[0].decode('ascii').rstrip()
    device_number = 0 if device_number == '' else int(device_number)

    # Get the cuda version for pathing
    cuda_version = os.getenv('CUDA_INSTALL_PATH').split('-')[-1]

    # Get the SASS ISA
    sass = 'QV100' if (args.sass == None) else args.sass


    # Make sure the kernel values are normal
    if args.end < args.start:
        print("End kernel should not be earlier than the starting kernel")

    get_traces(device_number, cuda_version, args.benchmark, args.params, args.start, args.end)
    get_sim_stats(cuda_version, args.benchmark, args.params, sass)
    return



"""
Print the info to see what it looks like
"""
def print_structure():
    pprint.pprint(kernel_traces)
    return



"""
Main state
"""
if __name__=="__main__":
    arg_wrapper()
