"""""""""""""""""""""""""""""""""""""""""""""""""""
Dependency Tracker Script
===================================================
Filename: dependency_tracker.py
Author: Reese Kuper
Purpose: Compare address between kernel traces and
the simulated addresses to find and graph kernel
and thread block dependencies
"""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import sys
import time
import multiprocessing as mp
from functools import partial
import argparse
import re
from subprocess import Popen, PIPE
import glob
import pprint
import json
from graphviz import Digraph

"""""""""
 GLOBALS
"""""""""
kernel_traces = {}
sim_stats = {}
start_kernel = 0
end_kernel = float('inf')
tbd_graph = Digraph(comment='Kernel Trace Dependencies', strict=True)



"""""""""""""""""""""

    Main Functions

"""""""""""""""""""""

def arg_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", help = \
            "Specify the benchmark (ex. rnn_bench from Deepbench)")
    parser.add_argument("-t", "--test", help = \
            ("Specify the benchmark parameters delimited by '_' " + \
            "(ex. train_half_8_8_1_lstm)"))
    parser.add_argument("-a", "--sass", help = \
            "Specify the SASS that the traces used (ex. QV100)")
    parser.add_argument("-s", "--start", help = \
            "Which kernel to start parsing from", default=0)
    parser.add_argument("-e", "--end", help = \
            "Which kernel to end parsing on", default=float('inf'))
    parser.add_argument("-d", "--depth", help = \
            "Data contains line the data was obtained from", default=1)
    parser.add_argument("-l", "--line_debug", help = \
            "Data contains line the data was obtained from", action='store_true')
    parser.add_argument("-o", "--open", help = \
            "Open and use json data given in json file named: kernel_traces.json", \
            action='store_true')
    parser.add_argument("-j", "--json", help = \
            "Output kernel_traces to json file (kernel_traces.json)", \
            action='store_true')
    parser.add_argument("-g", "--graph", help = \
            "Output a graph of all kernel dependencies found", action='store_true')
    args = parser.parse_args()

    # Get the GPU device number
    lspci = Popen("lspci", stdout=PIPE)
    grep = Popen(['grep', 'VGA'], stdin = lspci.stdout, stdout = PIPE)
    cut_col = Popen(['cut', '-d', ':', '-f', '2'], stdin = grep.stdout, stdout = PIPE)
    cut_space = Popen(['cut', '-d', ' ', '-f', '1'], stdin = cut_col.stdout, stdout = PIPE)
    cut_dec = Popen(['cut', '-d', '.', '-f', '1'], stdin = cut_col.stdout, stdout = PIPE)
    device_number = cut_dec.communicate()[0].decode('ascii').rstrip()

    # FIXME Set to 0 for debugging, NOT for normal use :/
    # device_number = 0 if device_number == '' else int(device_number)
    device_number = 0

    # Get the cuda version for pathing
    cuda_version = '11.1' if os.getenv('CUDA_INSTALL_PATH') == None \
            else os.getenv('CUDA_INSTALL_PATH').split('-')[-1]

    # Get the SASS ISA
    sass = 'QV100' if (args.sass == None) else args.sass

    # Make sure depth is not bad
    depth = 1 if int(args.depth) < 1 else int(args.depth)

    # Get global values for starting and ending kernel traces
    global start_kernel, end_kernel
    start_kernel = int(args.start)
    end_kernel = float('inf') if (args.end == float('inf')) else int(args.end)

    # Make sure the kernel values are normal
    if (args.end != float('inf')) and (int(args.end) < int(args.start)):
        print("End kernel should not be earlier than the starting kernel\n")
        end_kernel = float('inf')

    # If data exists and want to use, skip getting it again
    if args.open and os.path.isfile('kernel_traces.json'):
        kernel_trace_title = "=   Getting Kernel Traces   ="
        kernel_trace_title = ("=" * len(kernel_trace_title)) + "\n"  + \
                kernel_trace_title + "\n" + ("=" * len(kernel_trace_title))
        print(kernel_trace_title)
        global kernel_traces
        print("Gathering kernel_traces.json data...", end = ' ')
        kernel_traces = json.load(open('kernel_traces.json', 'r'))
        print("Done")
        kernels = sorted(kernel_traces.keys())
        start_kernel = int(kernels[0].split('-')[1])
        end_kernel = int(kernels[-1].split('-')[1])
        print("Using kernels " + str(start_kernel) + "-" + str(end_kernel))

    else:
        # Manage kernel traces
        get_traces(device_number, cuda_version, args.benchmark, \
                args.test, args.line_debug, depth)

        # Grab kernel trace dependencies
        print('Grabbing dependencies...', end = ' ')
        sys.stdout.flush()
        pool = mp.Pool(mp.cpu_count())
        specific_dependencies = partial(trace_dependencies, depth=depth)
        all_kernel_dependencies = pool.map(specific_dependencies, \
                kernel_traces.keys())
        for kernel_dependencies in all_kernel_dependencies:
            kernel_name = list(kernel_dependencies.keys())[0]
            kernel_traces[kernel_name]["dependencies"] = \
                    kernel_dependencies[kernel_name]
        print('Done')

    # Manage sim output
    get_sim_stats(cuda_version, args.benchmark, args.test, sass, args.line_debug)

    # Manage trace stats
    print_trace_stats(args.graph)

    # Print kernel names
    print_kernel_names()

    # Output to .json file
    if args.json:
        print("Writing file 'kernel_traces.json...'", end = ' ')
        with open('kernel_traces.json','w') as fp:
            json.dump(kernel_traces, fp)
        print("Done")

    # Note for info
    print("\n*** NOTE: Add third arguement 'view' in " + \
            "graph_dependencies(kernels=[], thread_blocks=[], view=...) to show:")
    print("\t'all': everything")
    print("\t'kernel': kernels and the kernels they depend on (no shown thread_block)")
    print("\t'thread-block': kernels and the selected thread blocks, " + \
            "along with the kernels and thread blocks that depend on them\n")

    return



def get_traces(device_number, cuda_version, benchmark, test, line_debug, depth):
    # Find beginning accel-sim-framework directory
    accelsim_dir = get_accel_sim()
    if accelsim_dir == None:
        print("Could not find accel-sim-framework")
        return

    device_dir = accelsim_dir + "/hw_run/traces/device-" + str(device_number)
    if not os.path.exists(device_dir):
        print("Could not find GPU device number in " + \
                "accel-sim-framework/hw_run/traces/device-#")
        return

    cuda_dir = device_dir + "/" + str(cuda_version)
    if not os.path.exists(cuda_dir):
        print("Could not find cuda version in " + \
                "accel-sim-framework/hw_run/traces/device-#/<CUDA>")
        return

    benchmark_dir = cuda_dir + "/" + benchmark
    if not os.path.exists(benchmark_dir):
        print("Could not find benchmark in " + \
                "accel-sim-framework/hw_run/traces/device-#/<CUDA>/<BENCHMARK>")
        return

    # The actual test is a bit harder to ensure while the test are in any order
    test_dir = get_test(benchmark_dir, test)
    if test_dir == None:
        print("Could not find specific test in " + \
                "accel-sim-framework/hw_run/traces/device-#/<CUDA>/<BENCHMARK>/<TEST>")
        return

    parse_trace_title = "=   Parsing Kernel Traces   ="
    parse_trace_title = "\n" + ("=" * len(parse_trace_title)) + "\n" + \
            parse_trace_title + "\n" + ("=" * len(parse_trace_title))
    print(parse_trace_title)
    print("Using test: " + test_dir[test_dir.rfind('/') + 1:])

    traces_dir = test_dir + "/traces"

    # Kernel numbers help get a list of all the kernels traced
    kernel_numbers = []
    kernel_offset = 0
    for subdir, dirs, files in os.walk(traces_dir):

        # Get the kernel numbers and the first traced kernel for the offset
        number_of_kernels = sum('kernel-' in s for s in files)
        for kernel in files:
            if len(re.findall("\d+", kernel)) > 0:
                kernel_numbers.append(int(re.findall("\d+", kernel)[0]))

        if len(kernel_numbers) == 0:
            print('No Traces Found')
            return 0, 0

        # Get kernel ranges
        global start_kernel, end_kernel
        kernel_offset = int((sorted(kernel_numbers)[0]))
        start_kernel = max(start_kernel, kernel_offset)
        last_kernel = number_of_kernels + kernel_offset
        end_kernel = last_kernel if end_kernel == float('inf') else \
                min(last_kernel, (end_kernel + 1))

        # Begin parsing each trace
        for i in range(start_kernel, end_kernel):

            # Get kernel info
            temp_kernel_name = ""
            new_thread_block = True
            kernel_id = 0
            kernel_name = "kernel-"
            current_block = "0,0,0"
            current_warp = "warp-"
            with open((traces_dir + "/kernel-" + str(i) + ".traceg"), 'r', encoding = 'utf-8')\
                    as trace:
                print("Parsing kernel " + str(i) + "...", end = ' ')
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
                        kernel_traces[kernel_name]["dependencies"] = {}
                        kernel_traces[kernel_name]["kernel_name"] = temp_kernel_name
                    elif "kernel name =" in line:
                        temp_kernel_name = line.split(' ')[-1].rstrip()
                    elif "grid dim =" in line:
                        grid_xyz = line[line.index('(') + 1: len(line) - 2]
                        grid_xyz = grid_xyz.split(',')
                        grid_dim = (int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2]))
                        kernel_traces[kernel_name]["grid_dim"] = grid_dim
                    elif "block dim =" in line:
                        block_xyz = line[line.index('(') + 1: len(line) - 2]
                        block_xyz = block_xyz.split(',')
                        block_dim = (int(block_xyz[0]), int(block_xyz[1]), \
                                int(block_xyz[2]))
                        kernel_traces[kernel_name]["block_dim"] = block_dim
                        kernel_traces[kernel_name]["thread_blocks"] = {}
                    elif "local mem base_addr =" in line:
                        kernel_traces[kernel_name]["local_mem_base_addr"] = \
                                (line.split(' ')[-1]).rstrip()

                    # Begin preparing the specific thread block and warp
                    elif "thread block = " in line:
                        current_block = (line.split(' ')[-1]).rstrip()
                        kernel_traces[kernel_name]["thread_blocks"][current_block] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["mem_addrs"] = []
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["num_insts"] = 0
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["num_mem_insts"] = 0
                        new_thread_block = True
                    elif "warp = " in line:

                        # Remove previous warp if nothing exists, UNLESS IN DEBUG
                        if (not line_debug) and (not new_thread_block) and (len(\
                                kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_addrs"]) == 0):
                           del kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                   ["warps"][current_warp]

                        current_warp = "warp-" + (line.split(' ')[-1]).rstrip()
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_insts"] = {}
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_addrs"] = []
                        new_thread_block = False
                    elif "#END_TB" in line:

                        # Remove previous warp if nothing exists, UNLESS IN DEBUG
                        if (not line_debug) and (not new_thread_block) and (len(\
                                kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_addrs"]) == 0):
                           del kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                   ["warps"][current_warp]
                        if (not line_debug) and (len(kernel_traces[kernel_name]\
                                ["thread_blocks"][current_block]["warps"]) == 0):
                           del kernel_traces[kernel_name]["thread_blocks"][current_block]
                    elif "insts = " in line:
                        warp_insts = int(line.split(' ')[-1])
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["num_insts"] = warp_insts
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["num_mem_insts"] = 0
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["num_insts"] += warp_insts
                        kernel_traces[kernel_name]["num_insts"] += warp_insts

                    # Start the actual instruction parsing
                    elif "LDG" in line or "STG" in line:
                        # Add line
                        line_fields = line.split(' ')
                        inst_name = kernel_name + "_0x" + line_fields[0]

                        # Skip if address is somehow 0
                        address = hex(int(line_fields[9], 16))
                        if str(address) == '0x0':
                            continue

                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_insts"][inst_name] = {}
                        if line_debug:
                            kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                    ["warps"][current_warp]["mem_insts"][inst_name]\
                                    ["line"] = line.rsplit()

                        # Add the PC and mask values
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_insts"][inst_name]\
                                ["pc"] = hex(int(line_fields[0], 16))
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_insts"][inst_name]\
                                ["mask"] = hex(int(line_fields[1], 16))

                        # Add memory instruction type
                        mem_type = ""
                        if "LDG" in line:
                            # mem_type = line_fields[4].split('.')[0]
                            mem_type = 'load'
                        else:
                            # mem_type = line_fields[3].split('.')[0]
                            mem_type = 'store'
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_insts"][inst_name]\
                                ["type"] = mem_type

                        # Add address info
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["mem_insts"][inst_name]\
                                ["addr"] = address
                        if address != 0:
                            kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                    ["warps"][current_warp]["mem_addrs"].append(address)
                            kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                    ["mem_addrs"].append(address)
                            kernel_traces[kernel_name]["mem_addrs"].append(address)

                        # Increment instruction counts
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["warps"][current_warp]["num_mem_insts"] += 1
                        kernel_traces[kernel_name]["thread_blocks"][current_block]\
                                ["num_mem_insts"] += 1
                        kernel_traces[kernel_name]["num_mem_insts"] += 1

                print('Done')
    return



# def trace_dependencies(depth, kernel_name, dependencies):
def trace_dependencies(kernel_name, depth):
    if start_kernel == (end_kernel - 1):
        return

    dependencies = {}
    dependencies[kernel_name] = {}
    kernel = int(kernel_name.split('-')[1])
    for current_block in kernel_traces[kernel_name]["thread_blocks"]:

        # Set up the current block dependency list
        current_block_name = kernel_name + '_' + str(current_block)
        dependencies[kernel_name][current_block_name] = []

        for current_address in kernel_traces[kernel_name]["thread_blocks"]\
                [current_block]["mem_addrs"]:

            cur_addr_check = int(current_address, 16) & 0xFFFFFFB0
            # Covers all subsequent kernels - takes FOREVER
            for future_kernel in range(kernel + 1, min(kernel + depth + 1, end_kernel)):
                future_kernel_name = 'kernel-' + str(future_kernel)
                for future_block in kernel_traces[future_kernel_name]["thread_blocks"]:
                    future_block_name = future_kernel_name + '_' + str(future_block)

                    # If current address matches any address in future thread block, add
                    if cur_addr_check in list(map(lambda x: (int(x, 16) & 0xFFFFFFB0), \
                            kernel_traces[future_kernel_name]["thread_blocks"]\
                            [future_block]["mem_addrs"])):
                        dependencies[kernel_name][current_block_name].append(\
                                future_block_name)

        # Remove independent blocks
        if len(dependencies[kernel_name][current_block_name]) == 0:
            del dependencies[kernel_name][current_block_name]

    return dependencies



def get_sim_stats(cuda_version, benchmark, test, sass, line_debug):
    # Find beginning accel-sim-framework directory
    accelsim_dir = get_accel_sim()
    if accelsim_dir == None:
        print("Could not find accel-sim-framework")
        return

    run_dir = accelsim_dir + "/sim_run_" + str(cuda_version)
    if not os.path.exists(run_dir):
        print("Could not find sim_run_<CUDA> in " + \
                "accel-sim-framework/sim_run_<CUDA>/. Did you simulate yet?")
        return

    benchmark_dir = run_dir + "/" + benchmark
    if not os.path.exists(benchmark_dir):
        print("Could not find benchmark in " + \
                "accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>")
        return

    # The actual test is a bit harder to ensure while the test are in any order
    test_dir = get_test(benchmark_dir, test)
    if test_dir == None:
        print("Could not find specific test in " + \
                "accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>/<TEST>")
        return
    parse_sim_title = "=   Parsing Simulation Output   ="
    parse_sim_title = "\n" + ("=" * len(parse_sim_title)) + "\n" + \
            parse_sim_title + "\n" + ("=" * len(parse_sim_title))
    print(parse_sim_title)
    print("Using test: " + test_dir[test_dir.rfind('/') + 1:])

    sass_dir = test_dir + "/" + sass + "-SASS"
    if not os.path.exists(sass_dir):
        print("Could not find sass in " + \
                "accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>/<TEST>/<SASS>")
        return

    # Now getting the specific test simulation output
    sim_file = get_test(sass_dir, (benchmark + "_" + test))
    if sim_file == None:
        print("Could not find simulation log in " + \
                "accel-sim-framework/sim_run_<CUDA>/<BENCHMARK>/<TEST>/<SASS>/<LOG>")
        return

    # Begin parsing the sim output
    # Get kernel info
    temp_kernel_name = ""
    kernel_id = 0
    kernel_name = "kernel-"
    began_print = False
    skipping_kernel = False
    with open(sim_file, 'r', encoding = 'utf-8') as sim_file:
        for line in sim_file:
            # Gather kernel info
            if "kernel id =" in line:
                kernel_id = int(line.split(' ')[-1])
                if (kernel_id < start_kernel) or (kernel_id > end_kernel):
                    skipping_kernel = True
                    continue
                else:
                    skipping_kernel = False

                if kernel_name != "kernel-":
                    print('Done')
                kernel_name = "kernel-" + str(kernel_id)
                sim_stats[kernel_name] = {}
                sim_stats[kernel_name]["id"] = kernel_id
                sim_stats[kernel_name]["mem_addrs"] = []
                sim_stats[kernel_name]["num_mem_insts"] = 0
                sim_stats[kernel_name]["mem_insts"] = {}
                began_print = True
                print("Parsing kernel " + str(kernel_id) + "...", end = ' ')
                sys.stdout.flush()
            elif not skipping_kernel and "gpu_sim_cycle" in line:
                sim_stats[kernel_name]["num_insts"] = int(line.split(' ')[-1])
            elif not skipping_kernel and "kernel name =" in line:
                temp_kernel_name = line.split(' ')[-1]
            elif not skipping_kernel and "grid dim =" in line:
                grid_xyz = line[line.index('(') + 1: len(line) - 2]
                grid_xyz = grid_xyz.split(',')
                grid_dim = (int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2]))
                sim_stats[kernel_name]["grid_dim"] = grid_dim
            elif not skipping_kernel and "block dim =" in line:
                block_xyz = line[line.index('(') + 1: len(line) - 2]
                block_xyz = block_xyz.split(',')
                block_dim = (int(block_xyz[0]), int(block_xyz[1]), int(block_xyz[2]))
                sim_stats[kernel_name]["block_dim"] = block_dim
            elif not skipping_kernel and "local mem base_addr =" in line:
                sim_stats[kernel_name]["local_mem_base_addr"] = (line.split(' ')[-1]).rstrip()

            # Begin parsing mem instructions
            elif not skipping_kernel and "mf:" in line:
                # Grab only the global memory instructions
                if 'GLOBAL' not in line:
                    continue

                # Clean up the list a little bit
                line_fields = line.strip().replace(',', '').split(' ')
                if '' in line_fields:
                    line_fields.remove('')

                inst_id = kernel_name + '_' + str(line_fields[13])
                sim_stats[kernel_name]["mem_insts"][inst_id] = {}

                # Only include the line in debug mode
                if line_debug:
                    sim_stats[kernel_name]["mem_insts"][inst_id]["line"] = line.strip()

                # Add all important fields
                sim_stats[kernel_name]["mem_insts"][inst_id]["pc"] = hex(int(line_fields[13], 16))
                sim_stats[kernel_name]["mem_insts"][inst_id]["type"] = line_fields[6]
                sim_stats[kernel_name]["mem_insts"][inst_id]["mask"] = line_fields[14]\
                        [line_fields[14].index('[') + 1:line_fields[14].index(']') - 1]
                sim_stats[kernel_name]["mem_insts"][inst_id]["warp"] = line_fields[3]\
                        [line_fields[3].index('w') + 1:]

                # Add the address and set to hex
                address = hex(int(line_fields[5][line_fields[5].index('=') + 1:], 16))
                sim_stats[kernel_name]["mem_insts"][inst_id]["address"] = address
                sim_stats[kernel_name]["mem_addrs"].append(address)

                # Increment counters
                sim_stats[kernel_name]["num_mem_insts"] += 1

        # Print that the sim trace for ending kernel is done
        if began_print:
            print('Done')
    return



"""""""""""""""

  Output Funcs

"""""""""""""""

def graph_dependencies(kernels=[], thread_blocks=[], view='all'):

    # Grab all needed info from the dependency section of stats/traces
    needed_info = {}
    for kernel in range(start_kernel, end_kernel):
        kernel_name = 'kernel-' + str(kernel)
        kernel_match = (kernel in kernels) or (len(kernels) == 0)

        if kernel_match:
            if kernel_name not in needed_info:
                needed_info[kernel_name] = {}
                needed_info[kernel_name]["dependencies"] = {}
                needed_info[kernel_name]["thread_blocks"] = []
                needed_info[kernel_name]["kernels"] = []

            # Add all dependent blocks to kernel info
            for block_depend in kernel_traces[kernel_name]["dependencies"]:
                thread_block = block_depend.split('_')[1]

                # If looking only at specific thread blocks, ignore the irrelevant information
                if (view == 'thread_block') and ((thread_block not in thread_blocks)\
                        and (len(thread_blocks) != 0)):
                    continue

                if block_depend not in needed_info[kernel_name]["dependencies"]:
                    needed_info[kernel_name]["dependencies"][block_depend] = []

                # Add all other kernel block dependencies to their kernel info
                for dependency in kernel_traces[kernel_name]["dependencies"][block_depend]:
                    kernel_dependency = dependency.split('_')[0]
                    thread_block_dependency = dependency.split('_')[1]

                    # Add all thread block dependencies for easy access
                    needed_info[kernel_name]["dependencies"][block_depend].append(dependency)

                    # In order: add kernel to the current kernel dependency list,
                    # then add the new other kernel info to the structure
                    if kernel_dependency not in needed_info[kernel_name]["kernels"]:
                        needed_info[kernel_name]["kernels"].append(kernel_dependency)
                    if kernel_dependency not in needed_info:
                        needed_info[kernel_dependency] = {}
                        needed_info[kernel_dependency]["dependencies"] = {}
                        needed_info[kernel_dependency]["thread_blocks"] = []
                        needed_info[kernel_dependency]["kernels"] = []
                    if thread_block_dependency not in needed_info[kernel_dependency]\
                            ["thread_blocks"]:
                        needed_info[kernel_dependency]["thread_blocks"].append(\
                                thread_block_dependency)


    # Begin creating the graph
    tbd_graph.clear()
    sys.stdout.flush()

    # For 'thread_block' or 'all' mode
    if view != 'kernel':
        tbd_graph.attr(ranksep="3")
        for kernel_name in needed_info:
            kernel = int(kernel_name.split('-')[1])
            kernel_match = (kernel in kernels) or (len(kernels) == 0)

            # Add/Change kernels and internal nodes
            with tbd_graph.subgraph(name=("cluster" + str(kernel))) as current_kernel:

                # If on a requested kernel
                fill_color = '#f72116bb' if (kernel in kernels or len(kernels) == 0) \
                        else '#f7211640'
                kernel_width = '5' if (kernel in kernels) else '3'
                short_kernel_name = kernel_traces[kernel_name]["kernel_name"].split('_')
                short_kernel_name = short_kernel_name[0] + short_kernel_name[1] + \
                        '-' + short_kernel_name[2]
                kernel_label = '<<br/><font point-size="20"><b>' + kernel_name + \
                        '</b></font>'
                kernel_label += '<br/><font point-size="14">' + short_kernel_name + \
                        '</font>>'
                current_kernel.attr(margin="20", style="rounded,filled", \
                        color="black", fillcolor=fill_color, penwidth=kernel_width, \
                        label=kernel_label, pad="2")

                # Add/Change thread block nodes
                for block_depend in needed_info[kernel_name]["dependencies"]:
                    thread_block = block_depend.split('_')[1]
                    thread_block_match = thread_block in thread_blocks
                    node_color = 'gray' if (kernel_match and thread_block_match) \
                            else 'white'
                    node_width = '3' if (kernel_match and thread_block_match) else '2'
                    thread_block_label = ('<<b>' + thread_block + '</b>>') if \
                            (kernel_match and thread_block_match) else thread_block
                    thread_block_id = kernel_name + '_' + thread_block
                    current_kernel.node(thread_block_id, thread_block_label, \
                            style="rounded,filled", color="black", \
                            fillcolor=node_color, penwidth=node_width)

                if (len(needed_info[kernel_name]["dependencies"]) == 0) or \
                        (view == 'thread_block'):
                    for thread_block in needed_info[kernel_name]["thread_blocks"]:
                        thread_block_match = thread_block in thread_blocks
                        node_color = 'gray' if (kernel_match and thread_block_match) \
                                else 'white'
                        node_width = '3' if (kernel_match and thread_block_match) \
                                else '2'
                        thread_block_label = ('<<b>' + thread_block + '</b>>') if \
                                (kernel_match and thread_block_match) else thread_block
                        thread_block_id = kernel_name + '_' + thread_block
                        current_kernel.node(thread_block_id, thread_block_label, \
                                style="rounded,filled", color="black", \
                                fillcolor=node_color, penwidth=node_width)

            # Change oppacities of edges if necessary
            for block_depend in needed_info[kernel_name]["dependencies"]:
                thread_block = block_depend.split('_')[1]
                thread_block_match = thread_block in thread_blocks

                edge_weight = '3' if (kernel_match and thread_block_match) else '1'
                edge_color = '#000000ff' if ((len(thread_blocks) == 0) or \
                        (kernel_match and thread_block_match)) else '#00000006'

                # Draw/Redraw the edges or for the first time
                for dependency in needed_info[kernel_name]["dependencies"][block_depend]:
                    tbd_graph.edge(block_depend, dependency, color=edge_color, \
                            penwidth=edge_weight)

    # For 'kernel' mode
    else:
        tbd_graph.attr(ranksep="1")
        for kernel_name in needed_info:
            kernel = int(kernel_name.split('-')[1])
            kernel_match = (kernel in kernels) or (len(kernels) == 0)
            short_kernel_name = kernel_traces[kernel_name]["kernel_name"].split('_')
            short_kernel_name = short_kernel_name[0] + short_kernel_name[1] + \
                    '-' + short_kernel_name[2]
            kernel_label = '<<br/><font point-size="20"><b>' + kernel_name + \
                    '</b></font>'
            kernel_label += '<br/><font point-size="14">' + short_kernel_name + \
                    '<br/></font>>'
            node_color = '#f72116bb' if (kernel_match) else '#f7211640'
            node_width = '3' if (kernel_match) else '2'
            tbd_graph.node(kernel_name, kernel_label, shape="box", \
                    style="bold,rounded,filled", color="black", fillcolor=node_color, \
                    penwidth=node_width)
        for kernel_name in needed_info:
            kernel = int(kernel_name.split('-')[1])
            kernel_match = (kernel in kernels) or (len(kernels) == 0)
            edge_weight = '3' if (kernel_match) else '1'
            edge_color = '#000000ff' if (kernel_match) else '#00000006'
            for kernel_dependency in needed_info[kernel_name]["kernels"]:
                tbd_graph.edge(kernel_name, kernel_dependency, color=edge_color, \
                        penwidth=edge_weight)

    create_graph_pdf()
    return



def create_graph_pdf():
    print('Creating kernel trace dependency graph...', end = ' ')
    sys.stdout.flush()
    tbd_graph.render('kernel_dependencies.gv')
    print('Done')
    return



"""""""""""""""

  Getter Info

"""""""""""""""

def get_accel_sim():
    """
    Get accel-sim directory path
    """

    find = Popen(['find', '../', '-maxdepth', '4', '-name', 'accel-sim-framework'], \
            stdout = PIPE)
    accelsim_dir = find.communicate()[0].decode('ascii').rstrip()
    return accelsim_dir



def get_test(path, test_str):
    """
    Get the test name given the subdirectories (depth = 1) and the test
    """

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
    test = re.split('_|-', test_str)
    best_match = path + "/" + subdirs[0]
    best_num = len(re.split('_|-',subdirs[0]))
    for subdir in subdirs:
        subdir_list = re.split('_|-', subdir)

        # Make sure that all parameters are at least in the possible test dir
        if not all(param in subdir_list for param in test):
            continue
        for param in test:
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



"""""""""""""""

 Printing Info

"""""""""""""""

def print_inst(kernel, pc):
    kernel_name = "kernel-" + str(kernel)
    inst_name = kernel_name + "_" + str(hex(pc))

    kernel_inst_title = "=   Kernel Trace for:" + inst_name + "   ="
    kernel_inst_title = "\n" + ("=" * len(kernel_inst_title)) + "\n" + \
            kernel_inst_title + "\n" + ("=" * len(kernel_inst_title))
    print(kernel_inst_title)
    for thread_block in kernel_traces[kernel_name]["thread_blocks"]:
        for warp in kernel_traces[kernel_name]["thread_blocks"][thread_block]["warps"]:
            for mem_inst in kernel_traces[kernel_name]["thread_blocks"][thread_block]\
                    ["warps"][warp]["mem_insts"]:
                if mem_inst == inst_name:
                    print(str(warp) + " ", end = '')
                    pprint.pprint(kernel_traces[kernel_name]["thread_blocks"]\
                            [thread_block]["warps"][warp]["mem_insts"][mem_inst])

    sim_inst_title = "=   Simulation Output for:" + inst_name + "   ="
    sim_inst_title = "\n" + ("=" * len(sim_inst_title)) + "\n"  + sim_inst_title + \
            "\n" + ("=" * len(sim_inst_title))
    print(sim_inst_title)
    for mem_inst in sim_stats[kernel_name]["mem_insts"]:
        if mem_inst == inst_name:
            pprint.pprint(sim_stats[kernel_name]["mem_insts"][mem_inst])
    return


def print_trace_stats(graph):
    dependency_stats_title = "=   Dependency Stats   ="
    dependency_stats_title = "\n" + ("=" * len(dependency_stats_title)) + "\n" + \
            dependency_stats_title + "\n" + ("=" * len(dependency_stats_title))
    print(dependency_stats_title)
    if graph:
        global tbd_graph
        graph_dependencies()

    # TODO:
    # Find number of indendent thread blocks per kernel
    # Find most dependent kernel AND threadblock
    for kernel in range(start_kernel, end_kernel):
        kernel_name = 'kernel-' + str(kernel)
        print("Dependent thread_blocks in " + kernel_name + ":", end = ' ')
        print(str(len(kernel_traces[kernel_name]["dependencies"])), end = '')
        print("/" + str(len(kernel_traces[kernel_name]["thread_blocks"])))
    indendent_blocks = 0
    return


def print_dependencies():
    kernel_list = []
    for kernel in kernel_traces:
        kernel_list.append(int(kernel.split('-')[1]))
    kernel_list = sorted(kernel_list)

    for kernel in range(kernel_list[0], kernel_list[-1]):
        kernel_name = 'kernel-' + str(kernel)
        if len(kernel_traces[kernel_name]['dependencies']) == 0:
            continue
        print('\'' + kernel_name + '\':', end = '\n\t')
        pprint.pprint(kernel_traces[kernel_name]['dependencies'])
    return


def print_kernel_names():
    kernel_names_title = "=   Kernel Names   ="
    kernel_names_title = "\n" + ("=" * len(kernel_names_title)) + "\n" + \
            kernel_names_title + "\n" + ("=" * len(kernel_names_title))
    print(kernel_names_title)
    for kernel in range(start_kernel, end_kernel):
        kernel_name = 'kernel-' + str(kernel)
        print(kernel_name + " - " + kernel_traces[kernel_name]["kernel_name"])


def print_sim():
    pprint.pprint(sim_stats)
    return


def print_trace():
    pprint.pprint(kernel_traces)
    return


if __name__=="__main__":
    begin = time.time()
    arg_wrapper()
    print("Runtime: " + str((time.time() - begin)) + "s\n")
