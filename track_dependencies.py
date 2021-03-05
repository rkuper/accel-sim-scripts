"""""""""""""""""""""""""""""""""""""""""""""""""""
Dependency Tracker Script
===================================================
Filename: track_dependencies.py
Author: Reese Kuper
Purpose: Compare address between kernel traces and
the simulated addresses to find and graph kernel
and thread block dependencies
"""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import sys
import gc
import time
import string
from collections import deque
import multiprocessing as mp
from functools import partial
import argparse
import re
from subprocess import Popen, PIPE
import glob
import pprint
import json
import networkx as nx
from graphviz import Digraph

"""""""""
 GLOBALS
"""""""""
sim_stats = {}
start_kernel = 0
end_kernel = float('inf')
trace_tbd_graph = Digraph(comment='Kernel Trace Dependencies')
sim_tbd_graph = Digraph(comment='Kernel Sim Dependencies')
CACHE_LINE_SIZE = 0xFFFFFFFFFFFFFF80



"""""""""""""""""""""

    Main Functions

"""""""""""""""""""""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", help = \
            "Specify the benchmark (ex. rnn_bench from Deepbench)")
    parser.add_argument("-t", "--test", help = \
            ("Specify the benchmark parameters delimited by '_' " + \
            "(ex. train_half_8_8_1_lstm)"))
    parser.add_argument("-a", "--sass", help = \
            "Specify the SASS that the traces used (ex. QV100)")
    parser.add_argument("-c", "--compressed", help = \
            "Use uncompressed addresses from the traces when parsing", action='store_true')
    parser.add_argument("-s", "--start", help = \
            "Which kernel to start parsing from", default=0)
    parser.add_argument("-e", "--end", help = \
            "Which kernel to end parsing on", default=float('inf'))
    parser.add_argument("-d", "--depth", help = \
            "Data contains line the data was obtained from", default=1)
    parser.add_argument("-l", "--line_debug", help = \
            "Data contains line the data was obtained from", action='store_true')
    parser.add_argument("-o", "--open", help = \
            "Open and use json data given in json file named: sim_stats.json", \
            action='store_true')
    parser.add_argument("-j", "--json", help = \
            "Output sim_stats to json file (sim_stats.json)", \
            action='store_true')
    parser.add_argument("-g", "--graph", help = \
            "Output a graph of all kernel dependencies found", action='store_true')
    args = parser.parse_args()

    # Set timing variables
    total_begin = time.time()
    parse_sim_begin = 0
    parse_sim_end = 0
    sim_dependencies_begin = 0
    sim_dependencies_end = 0
    graph_begin = 0
    graph_end = 0
    ideal_kernel_begin = 0
    ideal_kernel_end = 0
    ideal_thread_block_begin = 0
    ideal_thread_block_end = 0

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
    if args.open and os.path.isfile('sim_stats.json'):
        sim_stats_title = "=   Getting Simulation Info From sim_stats.json   ="
        sim_stats_title = ("=" * len(sim_stats_title)) + "\n"  + \
                sim_stats_title + "\n" + ("=" * len(sim_stats_title))
        print(sim_stats_title)
        parse_sim_begin = time.time()
        sim_dependencies_begin = time.time()
        global sim_stats
        print("Gathering sim_stats.json data...", end = ' ')
        sim_stats = json.load(open('sim_stats.json', 'r'))
        print("Done")
        kernels = sorted(sim_stats.keys())
        start_kernel = int(kernels[0].split('-')[1])
        end_kernel = int(kernels[-1].split('-')[1])
        sim_dependencies_end = time.time()
        parse_sim_end = time.time()
        print("Using kernels " + str(start_kernel) + "-" + str(end_kernel))

    else:
        # Manage kernel traces
        parse_sim_begin = time.time()
        parse_sim_output(cuda_version, args.benchmark, args.test, sass, \
                args.line_debug)
        parse_sim_end = time.time()

        # Grab kernel trace dependencies
        print('Grabbing dependencies...', end = ' ')
        sys.stdout.flush()
        sim_dependencies_begin = time.time()
        pool = mp.Pool(mp.cpu_count())
        specific_dependencies = partial(find_dependencies, depth=depth, \
                info=sim_stats)
        all_kernel_dependencies = pool.map(specific_dependencies, \
                sim_stats.keys())
        for kernel_dependencies in all_kernel_dependencies:
            kernel_name = list(kernel_dependencies.keys())[0]
            sim_stats[kernel_name]["dependencies"] = \
                    kernel_dependencies[kernel_name]
        sim_dependencies_end = time.time()
        print('Done')

    # Output to .json file
    if args.json:
        print("Writing file 'sim_stats.json...'", end = ' ')
        with open('sim_stats.json','w') as fp:
            json.dump(sim_stats, fp)
        print("Done")

    # Manage trace stats
    graph_begin = time.time()
    print_dependency_stats(args.graph)
    graph_end = time.time()

    # Print kernel names
    # print_kernel_names()

    # Print kernel level estimated cycle time
    ideal_kernel_begin = time.time()
    # get_kernel_estimated_time(int(args.depth))
    get_kernel_estimated_time()
    ideal_kernel_end = time.time()

    # Print thread_block level estimated cycle time
    ideal_thread_block_begin = time.time()
    get_thread_block_estimated_time()
    ideal_thread_block_end = time.time()

    # Timing information
    print('')
    timing_title = "=   Notable Timings   ="
    timing_title = ("=" * len(timing_title)) + "\n"  + \
        timing_title + "\n" + ("=" * len(timing_title))
    print(timing_title)
    print("Parse Simulation Output Time: " + str(parse_sim_end - parse_sim_begin))
    print("Get Simulation Dependencies Time: " + str(sim_dependencies_end - \
            sim_dependencies_begin))
    if args.graph:
        print("Graph Time: " + str(graph_end - graph_begin))
    print("Ideal Kernel Estimate Time: " + str(ideal_kernel_end - ideal_kernel_begin))
    print("Ideal Thread Block Estimate Time: " + str(ideal_thread_block_end - ideal_thread_block_begin))
    print('---------------------------------')
    print("Total Runtime: " + str((time.time() - total_begin)) + "s\n")

    # Note for info
    print("\n\n*** NOTE: Third arguement 'view' in " + \
            "graph_dependencies(kernels=[], thread_blocks=[], view=..., source=...)" + \
            " can show:")
    print("\t'all': everything")
    print("\t'kernel': kernels and the kernels they depend on (no shown thread_block)")
    print("\t'thread-block': selected kernels and thread blocks, " + \
            "along with dependent kernels and thread blocks\n")
    print("Fourth argument 'source' specifies either:")
    print("\t'all': dependencies from the both data sets")
    print("\t'sim': dependencies from the simulation output\n")
    return



def parse_sim_output(cuda_version, benchmark, test, sass, line_debug):
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

    global end_kernel

    # Begin parsing the sim output
    # Get kernel info
    temp_kernel_name = ""
    kernel_id = 0
    kernel_name = "kernel-"
    began_print = False
    skipping_kernel = False
    temp_thread_block = ""
    temp_end_kernel = 0
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

                temp_end_kernel = kernel_id
                if kernel_name != "kernel-":
                    print('Done')
                kernel_name = "kernel-" + str(kernel_id)
                sim_stats[kernel_name] = {}
                sim_stats[kernel_name]['info_name'] = 'sim'
                sim_stats[kernel_name]["id"] = kernel_id
                sim_stats[kernel_name]["mem_addrs"] = []
                sim_stats[kernel_name]["num_mem_insts"] = 0
                sim_stats[kernel_name]["total_time"] = 0
                sim_stats[kernel_name]["thread_blocks"] = {}
                sim_stats[kernel_name]["dependencies"] = {}
                sim_stats[kernel_name]["kernel_name"] = temp_kernel_name
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
            elif not skipping_kernel and "thread block = " in line:
                temp_thread_block = line.split(' ')[3].rstrip()

            # Get start and end times for ctas
            elif not skipping_kernel and "Started CTA" in line:
                line_fields = line.split(' ')
                thread_block = temp_thread_block
                if thread_block not in sim_stats[kernel_name]["thread_blocks"]:
                    sim_stats[kernel_name]["thread_blocks"][thread_block] = {}
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"] = {}
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["mem_addrs"] = []
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["num_mem_insts"] = 0
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["start_time"] = \
                            line_fields[6][line_fields[6].index('(') + 1:\
                            line_fields[6].index(',')]
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["end_time"] = 0
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["time"] = 0

            elif not skipping_kernel and "gpu_tot_sim_cycle = " in line:
                line_fields = line.split(' ')
                sim_stats[kernel_name]['total_time'] = int(line_fields[-1])

            elif not skipping_kernel and "Finished CTA" in line:
                line_fields = line.split(' ')
                thread_block = line_fields[4][line_fields[4].index('(') + 1:\
                        line_fields[4].index(')')]
                sim_stats[kernel_name]["thread_blocks"][thread_block]["end_time"] = \
                            line_fields[6][line_fields[6].index('(') + 1:\
                            line_fields[6].index(',')]
                sim_stats[kernel_name]["thread_blocks"][thread_block]["time"] = \
                            str(int(sim_stats[kernel_name]["thread_blocks"][thread_block]\
                            ["end_time"]) - int(sim_stats[kernel_name]["thread_blocks"]\
                            [thread_block]["start_time"]))

            # Begin parsing mem instructions
            elif not skipping_kernel and "mf:" in line:
                # Grab only the global memory instructions
                if 'GLOBAL' not in line:
                    continue

                # Clean up the list a little bit
                line_fields = line.strip().split(' ')
                if '' in line_fields:
                    line_fields.remove('')
                if ',' in line_fields:
                    line_fields.remove(',')

                # Add the thread block info to put info into
                thread_block = line_fields[10].split('=')[1]

                # Add the warp info to put info into
                warp = line_fields[3].split('=')[1]
                if warp not in sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]:
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp] = {}
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_insts"] = {}
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_addrs"] = []

                # inst = line_fields[1].split('=')[1]
                inst = hex(int(line_fields[9], 16))
                if inst not in sim_stats[kernel_name]["thread_blocks"][thread_block]\
                        ["warps"][warp]["mem_insts"]:
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_insts"][inst] = {}
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_insts"][inst]["line_addr"] = []
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_insts"][inst]["addr"] = []
                    if line_debug:
                        sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                                [warp]["mem_insts"][inst]["line"] = []

                # Add all important fields
                mem_type = line_fields[6].replace(',', '')

                # Add the address and set to hex
                address = int(line_fields[5].split('=')[1].replace(',', ''), 16)
                line_address = address & CACHE_LINE_SIZE
                sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                        [warp]["mem_insts"][inst]["addr"].append(hex(address))
                if hex(line_address) not in sim_stats[kernel_name]["thread_blocks"]\
                        [thread_block]["warps"][warp]["mem_insts"][inst]["line_addr"]:
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_insts"][inst]["line_addr"].append(hex(line_address))

                type_address = hex(line_address | 0x1) if (mem_type == 'load')\
                        else hex(line_address | 0x2)
                if type_address not in sim_stats[kernel_name]["thread_blocks"]\
                        [thread_block]["warps"][warp]["mem_addrs"]:
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                        [warp]["mem_addrs"].append(type_address)
                if type_address not in sim_stats[kernel_name]["thread_blocks"]\
                        [thread_block]["mem_addrs"]:
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["mem_addrs"]\
                        .append(type_address)
                if type_address not in sim_stats[kernel_name]["mem_addrs"]:
                    sim_stats[kernel_name]["mem_addrs"].append(type_address)

                # Only include the line in debug mode
                if line_debug:
                    sim_stats[kernel_name]["thread_blocks"][thread_block]["warps"]\
                            [warp]["mem_insts"][inst]["line"].append(line.strip())

                # Increment counters
                sim_stats[kernel_name]["thread_blocks"][thread_block]["num_mem_insts"] += 1
                sim_stats[kernel_name]["num_mem_insts"] += 1

        # Print that the sim trace for ending kernel is done
        if began_print:
            print('Done')

    if (end_kernel == float('inf')) or (end_kernel > temp_end_kernel):
        end_kernel = temp_end_kernel
    return



def find_dependencies(kernel_name, depth, info):
    if (start_kernel == end_kernel):
        return

    dependencies = {}
    dependencies[kernel_name] = {}
    kernel = int(kernel_name.split('-')[1])
    for current_block in info[kernel_name]["thread_blocks"]:

        # Set up the current block dependency list
        current_block_name = kernel_name + '_' + str(current_block)
        dependencies[kernel_name][current_block_name] = []

        for current_address in info[kernel_name]["thread_blocks"]\
                [current_block]["mem_addrs"]:

            cur_line_check = int(current_address, 16) & CACHE_LINE_SIZE
            before_type = 'R' if (int(current_address, 16) & 0x1) else 'W'

            # Covers all subsequent kernels - takes FOREVER
            for future_kernel in range(kernel + 1, min(kernel + depth + 1, end_kernel + 1)):
                future_kernel_name = 'kernel-' + str(future_kernel)
                for future_block in info[future_kernel_name]["thread_blocks"]:
                    future_block_name = future_kernel_name + '_' + str(future_block)

                    # No duplicates
                    if future_block_name in dependencies[kernel_name][current_block_name]:
                        continue

                    # If current address matches any address in future thread block, add
                    if ((cur_line_check | 0x2) in list(map(lambda x: int(x, 16), info\
                            [future_kernel_name]["thread_blocks"][future_block]\
                            ["mem_addrs"]))):
                        full_name = future_block_name + '_' + 'WA' + before_type
                        if full_name not in dependencies[kernel_name][current_block_name]:
                            dependencies[kernel_name][current_block_name].append(\
                                    future_block_name + '_' + 'WA' + before_type)
                    if ((cur_line_check | 0x1) in list(map(lambda x: int(x, 16), info\
                            [future_kernel_name]["thread_blocks"][future_block]\
                            ["mem_addrs"]))) and (before_type == 'W'):
                        full_name = future_block_name + '_' + 'RA' + before_type
                        if full_name not in dependencies[kernel_name][current_block_name]:
                            dependencies[kernel_name][current_block_name].append(\
                                    future_block_name + '_' + 'RA' + before_type)

        # Remove independent blocks, otherwise, sort the dependencies
        if len(dependencies[kernel_name][current_block_name]) == 0:
            del dependencies[kernel_name][current_block_name]
        else:
            dependencies[kernel_name][current_block_name] = sorted(dependencies\
                    [kernel_name][current_block_name])

    return dependencies



"""""""""""""""

  Output Funcs

"""""""""""""""

def graph_dependencies(kernels=[], thread_blocks=[], view='all', source='all', time_report=True, path=[]):
    graph_begin = time.time()
    graph_dependencies_helper(kernels=kernels, thread_blocks=thread_blocks, \
        view=view, info=sim_stats, graph=sim_tbd_graph, info_name="sim", path=path)

    # Combine the two graphs
    os.system("rm -f dependencies.gv.pdf")
    os.system("mv sim_dependencies.gv.pdf dependencies.gv.pdf")

    if time_report:
        print("Graph Time: " + str(time.time() - graph_begin) + '\n')
    return



def graph_dependencies_helper(kernels, thread_blocks, view, info, graph, info_name, path):

    # Grab all needed info from the dependency section of stats/traces
    needed_info = {}
    for kernel in range(start_kernel, end_kernel + 1):
        kernel_name = 'kernel-' + str(kernel)
        kernel_match = (kernel in kernels) or (len(kernels) == 0)

        if kernel_match:
            if kernel_name not in needed_info:
                needed_info[kernel_name] = {}
                needed_info[kernel_name]["dependencies"] = {}
                needed_info[kernel_name]["thread_blocks"] = []
                needed_info[kernel_name]["kernels"] = []

            # Add all dependent blocks to kernel info
            for block_depend in info[kernel_name]["dependencies"]:
                thread_block = block_depend.split('_')[1]

                # If looking only at specific thread blocks, ignore the irrelevant information
                if (view == 'thread_block') and ((thread_block not in thread_blocks)\
                        and (len(thread_blocks) != 0)):
                    continue

                if block_depend not in needed_info[kernel_name]["dependencies"]:
                    needed_info[kernel_name]["dependencies"][block_depend] = []

                # Add all other kernel block dependencies to their kernel info
                for dependency in info[kernel_name]["dependencies"][block_depend]:
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
    graph.clear()
    sys.stdout.flush()
    kernel_description = 'all' if len(kernels) == 0 else str(kernels)
    thread_blocks_description = 'all' if len(thread_blocks) == 0 else str(thread_blocks)
    title = '<<font point-size="100"><br/><b>'
    title += 'Simulation Output' if (info_name == 'sim') else 'Trace File'
    title += ' Dependencies</b><br/></font><font point-size="80">'
    title += 'kernels=' + kernel_description + ', thread_blocks=' + \
            thread_blocks_description + ', view=' + view
    title += '<br/><br/><br/><br/></font>>'
    graph.attr(labelloc="t", label=title)

    # For 'thread_block' or 'all' mode
    if view != 'kernel':
        graph.attr(ranksep="3")
        for kernel_name in needed_info:
            kernel = int(kernel_name.split('-')[1])
            kernel_match = (kernel in kernels) or (len(kernels) == 0)

            # Add/Change kernels and internal nodes
            with graph.subgraph(name=("cluster" + str(kernel))) as current_kernel:

                # If on a requested kernel
                fill_color = '#f72116bb' if (kernel in kernels or len(kernels) == 0) \
                        else '#f7211640'
                kernel_width = '5' if (kernel in kernels) else '3'
                short_kernel_name = info[kernel_name]["kernel_name"].split('_')
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
                    test_block_name = kernel_name + '_' + thread_block
                    thread_block_match = ((thread_block in thread_blocks) and (len(path) == 0)) \
                            or (test_block_name in path)
                    node_color = 'gray' if (kernel_match and thread_block_match) \
                            else 'white'
                    node_width = '3' if (kernel_match and thread_block_match) else '2'
                    thread_block_label = ('<<b>' + thread_block + '</b>>') if \
                            (kernel_match and thread_block_match) else thread_block

                    # Add time info for sim graph
                    if info_name == 'sim':
                        time = info[kernel_name]["thread_blocks"][thread_block]["time"]
                        thread_block_label = ('<<b>' + thread_block + '<br/>' + \
                                time + '</b>>') if (kernel_match and thread_block_match)\
                                else ('<' + thread_block + '<br/>' + time + '>')

                    thread_block_id = kernel_name + '_' + thread_block
                    current_kernel.node(thread_block_id, thread_block_label, \
                            style="rounded,filled", color="black", \
                            fillcolor=node_color, penwidth=node_width)

                if (len(needed_info[kernel_name]["dependencies"]) == 0) or \
                        (view == 'thread_block'):
                    for thread_block in needed_info[kernel_name]["thread_blocks"]:
                        test_block_name = kernel_name + '_' + thread_block
                        thread_block_match = ((thread_block in thread_blocks) and \
                                (len(path) == 0)) or (test_block_name in path)
                        node_color = 'gray' if (kernel_match and thread_block_match) \
                                else 'white'
                        node_width = '3' if (kernel_match and thread_block_match) \
                                else '2'
                        thread_block_label = ('<<b>' + thread_block + '</b>>') if \
                                (kernel_match and thread_block_match) else thread_block

                        # Add time info for sim graph
                        if info_name == 'sim':
                            time = info[kernel_name]["thread_blocks"][thread_block]["time"]
                            thread_block_label = ('<<b>' + thread_block + '<br/>' + \
                                    time + '</b>>') if (kernel_match and thread_block_match)\
                                    else ('<' + thread_block + '<br/>' + time + '>')

                        thread_block_id = kernel_name + '_' + thread_block
                        current_kernel.node(thread_block_id, thread_block_label, \
                                style="rounded,filled", color="black", \
                                fillcolor=node_color, penwidth=node_width)

                # FIXME remove later
                for thread_block in info[kernel_name]["thread_blocks"]:
                    if kernel_name == 'kernel-34' or kernel_name == 'kernel-31':
                        break
                    thread_block_id = kernel_name + '_' + thread_block
                    time = info[kernel_name]["thread_blocks"][thread_block]["time"]
                    thread_block_label = ('<' + thread_block + '<br/>' + time + '>')
                    current_kernel.node(thread_block_id, thread_block_label, \
                            style="rounded,filled", color="black", \
                            fillcolor='white', penwidth='3')


            # Change oppacities of edges if necessary
            for block_depend in needed_info[kernel_name]["dependencies"]:
                thread_block = block_depend.split('_')[1]
                thread_block_match = ((thread_block in thread_blocks) and (len(path) == 0)) \
                        or (block_depend in path)

                # Draw/Redraw the edges or for the first time
                for dependency in needed_info[kernel_name]["dependencies"][block_depend]:
                    dependency_info = dependency.split('_')
                    dependency_id = dependency_info[0] + '_' + dependency_info[1]
                    dependency_type = dependency_info[2]

                    # Get graph colors and weights
                    edge_match = thread_block_match and (dependency_id in path)
                    edge_checks = ((len(thread_blocks) == 0) and len(path) == 0) or \
                            (edge_match)
                    edge_op = 'ff' if edge_checks else '0b'
                    edge_weight = '3' if (kernel_match and edge_match) else '1'

                    # Blue
                    if  dependency_type == 'RAW':
                        edge_color = '#00529a'
                    # Green
                    elif dependency_type == 'WAW':
                        edge_color = '#0f9a00'
                    # Rurple
                    elif dependency_type == 'WAR':
                        edge_color = '#9a0045'
                    else:
                        edge_color = '#000000'

                    graph.edge(block_depend, dependency_id, color=(edge_color + edge_op), \
                            penwidth=edge_weight)

    # For 'kernel' mode
    else:
        graph.attr(ranksep="1")
        for kernel_name in needed_info:
            kernel = int(kernel_name.split('-')[1])
            kernel_match = (kernel in kernels) or (len(kernels) == 0)
            short_kernel_name = info[kernel_name]["kernel_name"].split('_')
            short_kernel_name = short_kernel_name[0] + short_kernel_name[1] + \
                    '-' + short_kernel_name[2]
            kernel_label = '<<br/><font point-size="20"><b>' + kernel_name + \
                    '</b></font>'
            kernel_label += '<br/><font point-size="14">' + short_kernel_name + \
                    '<br/></font>>'
            node_color = '#f72116bb' if (kernel_match) else '#f7211640'
            node_width = '3' if (kernel_match) else '2'
            graph.node(kernel_name, kernel_label, shape="box", \
                    style="bold,rounded,filled", color="black", fillcolor=node_color, \
                    penwidth=node_width)
        for kernel_name in needed_info:
            kernel = int(kernel_name.split('-')[1])
            kernel_match = (kernel in kernels) or (len(kernels) == 0)
            edge_weight = '3' if (kernel_match) else '1'
            edge_color = '#000000ff' if (kernel_match) else '#00000006'
            for kernel_dependency in needed_info[kernel_name]["kernels"]:
                graph.edge(kernel_name, kernel_dependency, color=edge_color, \
                        penwidth=edge_weight)

    create_graph_pdf(info_name, graph)
    return



def create_graph_pdf(info_name, graph):
    print('Creating ' + info_name + ' dependency graph...', end = ' ')
    sys.stdout.flush()
    graph.render((info_name + '_dependencies.gv'))
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


def get_kernel_dependencies(info=sim_stats):
    kernel_dependencies = {}
    for kernel_name in info:
        kernel_num = int(kernel_name.split("-")[1])
        kernel_dependencies[kernel_num] = []
        for block_dependency in info[kernel_name]["dependencies"]:
            for dependency in info[kernel_name]["dependencies"][block_dependency]:
                dependent_kernel = int((dependency.split("-")[1]).split("_")[0])
                if dependent_kernel not in kernel_dependencies[kernel_num]:
                    kernel_dependencies[kernel_num].append(dependent_kernel)
    return kernel_dependencies


def get_max_kernel_time_old(kernel, early_offset):
    max_cost = 0
    for thread_block in kernel["thread_blocks"]:
        time = int(kernel["thread_blocks"][thread_block]["time"])
        current_offset = int(kernel["thread_blocks"][thread_block]["start_time"])
        time_offset = current_offset - early_offset
        time += time_offset
        max_cost = time if (time > max_cost) else max_cost
    return max_cost


# TODO KEEP - IS THIS THE KEY???
def get_kernel_estimated_time_old(depth):
    kernel_time_title = "=   " + "Ideal Kernel Cycle Times" + "   ="
    kernel_time_title = "\n" + ("=" * len(kernel_time_title)) + "\n" + \
            kernel_time_title + "\n" + ("=" * len(kernel_time_title))
    print(kernel_time_title)

    kernel_dependencies = get_kernel_dependencies(sim_stats)
    current_kernel = 'kernel-' + str(start_kernel)
    current_cost = 0
    total_cost = 0

    kernel_list = []
    for kernel in range(start_kernel, end_kernel + 1):
        kernel_list.append('kernel-' + str(kernel))

    time_offset = float('inf')
    while current_kernel != ('kernel-' + str(end_kernel)):

        # Get earliest start time for kernel
        for thread_block in sim_stats[current_kernel]["thread_blocks"]:
            start_time = int(sim_stats[current_kernel]["thread_blocks"][thread_block]["start_time"])
            time_offset = start_time if (start_time < time_offset) else time_offset

        current_kernel_num = int(current_kernel.split('-')[1])
        current_cost = get_max_kernel_time(sim_stats[current_kernel], time_offset)
        kernel_list.remove(current_kernel)

        # Check future kernels (in depth) for independent times
        for test_kernel in range(1, depth + 1):
            test_kernel_name = 'kernel-' + str(int(current_kernel.split('-')[1]) + test_kernel)
            if test_kernel_name not in kernel_list:
                continue

            # Check for independent kernels for max cycle count (horizontally in graph)
            if (current_kernel_num + test_kernel) not in kernel_dependencies[current_kernel_num]:
                if (int(current_kernel.split('-')[1]) + test_kernel) < end_kernel:
                    kernel_list.remove(test_kernel_name)
                test_cost = get_max_kernel_time(sim_stats[test_kernel_name], time_offset)
                current_cost = test_cost if (test_cost > current_cost) else current_cost
            else:
                break

        total_cost += current_cost
        current_kernel = kernel_list[0]

    print("Total Cycle Time: " + str(total_cost))
    return


def get_max_kernel_time(kernel):
    max_cost = 0
    for thread_block in kernel["thread_blocks"]:
        time = int(kernel["thread_blocks"][thread_block]["time"])
        max_cost = time if (time > max_cost) else max_cost
    return max_cost


def get_kernel_estimated_time(graph=True):
    kernel_time_title = "=   " + "Ideal Kernel Cycle Times" + "   ="
    kernel_time_title = "\n" + ("=" * len(kernel_time_title)) + "\n" + \
            kernel_time_title + "\n" + ("=" * len(kernel_time_title))
    print(kernel_time_title)

    # Make DAG
    nx_graph = nx.DiGraph()
    kernel_dependencies = get_kernel_dependencies(sim_stats)

    # Add all kernel nodes
    for kernel_name in sim_stats:
        nx_graph.add_node(kernel_name)

    # Add all edges for each kernel
    for kernel in kernel_dependencies:
        for dependent in kernel_dependencies[kernel]:
            edge_weight = get_max_kernel_time(sim_stats[('kernel-' + str(dependent))])
            nx_graph.add_edge(('kernel-' + str(kernel)), ('kernel-' + str(dependent)), \
                    weight=edge_weight)

    # Add start and finish nodes
    nx_graph.add_node('Start')
    nx_graph.add_node('Finish')
    for kernel in list(nx_graph.nodes()):
        if (kernel == 'Start') or (kernel == 'Finish'):
            continue
        num_in = nx_graph.in_degree(kernel)
        num_out = nx_graph.out_degree(kernel)
        edge_weight = get_max_kernel_time(sim_stats[kernel])

        # Add in needed start and finish nodes for the full paths
        if num_in == 0:
            nx_graph.add_edge('Start', kernel, weight=edge_weight)
        if num_out == 0:
            nx_graph.add_edge(kernel, 'Finish', weight=0)

    # DFS for longest path
    best_found = ['', 0]
    kernel_queue = deque()
    kernel_queue.append([['Start'], 0])
    while (len(kernel_queue) != 0):
        current_path = kernel_queue.pop()
        if (current_path[0][-1] == 'Finish'):
            if (current_path[1] > best_found[1]):
                best_found = [current_path[0].copy(), current_path[1]]
        for dependent_kernel in nx_graph.neighbors(current_path[0][-1]):
            next_path = [current_path[0].copy(), current_path[1]]
            next_path[0].append(dependent_kernel)
            next_path[1] = current_path[1] + nx_graph.get_edge_data(next_path[0][-2], next_path[0][-1],
                    default=0)['weight']
            kernel_queue.append(next_path)
    path = best_found[0]
    total_cost = best_found[1]

    # Print path
    kernel_list = []
    path.remove('Start')
    path.remove('Finish')
    for kernel_index in range(len(path)):
        kernel_name = path[kernel_index]
        kernel_list.append(int(kernel_name.split('-')[1]))
        print(str(kernel_index + 1) + ': ' + kernel_name)

    # Graph the path
    if graph:
        graph_dependencies(kernels=kernel_list, view='kernel', time_report=False)

    print("Ideal Total Cycle Time: " + str(total_cost))
    print("Sim Total Cycle Time: " + str(sim_stats[('kernel-' + str(end_kernel))]['total_time']))
    return


def get_thread_block_estimated_time(graph=True):
    cta_time_title = "=   " + "Ideal Thread Block Cycle Path and Time" + "   ="
    cta_time_title = "\n" + ("=" * len(cta_time_title)) + "\n" + \
            cta_time_title + "\n" + ("=" * len(cta_time_title))
    print(cta_time_title)

    # Make DAG and average kernel timing info
    nx_graph = nx.DiGraph()
    kernel_averages = {}


    # Add the nodes for all thread blocks
    # keep average kernel cost for removing unnecessary nodes later
    for kernel_name in sim_stats:
        total_kernel_time = 0
        total_thread_blocks = 0
        for thread_block in sim_stats[kernel_name]["thread_blocks"]:
            thread_block_id = kernel_name + '_' + thread_block
            nx_graph.add_node(thread_block_id)
            total_kernel_time += int(sim_stats[kernel_name]["thread_blocks"]\
                    [thread_block]["time"])
            total_thread_blocks += 1
        kernel_averages[kernel_name] = (total_kernel_time / total_thread_blocks)

    # Add edges with their weights for all thread block dependencies
    for kernel_name in sim_stats:
        for block_depend in sim_stats[kernel_name]["dependencies"]:
            for dependency in sim_stats[kernel_name]["dependencies"][block_depend]:
                dependency_info = dependency.split('_')
                dependency_id = dependency_info[0] + '_' + dependency_info[1]
                edge_weight = int(sim_stats[dependency_info[0]]\
                        ["thread_blocks"][dependency_info[1]]["time"])
                nx_graph.add_edge(block_depend, dependency_id, weight=edge_weight)

    # Set start and finish attachments to each node that needs it
    nx_graph.add_node('Start')
    nx_graph.add_node('Finish')
    for block in list(nx_graph.nodes()):
        if (block == 'Start') or (block == 'Finish'):
            continue
        num_in = nx_graph.in_degree(block)
        num_out = nx_graph.out_degree(block)
        kernel = block.split('_')[0]
        thread_block = block.split('_')[1]
        edge_weight = int(sim_stats[kernel]["thread_blocks"][thread_block]["time"])

        # Don't add uneccesary nodes (small, no in our out degrees)
        if (num_in == 0) and (num_out == 0) and ((edge_weight / \
                kernel_averages[kernel]) <= 1.10):
            continue

        # Add in needed start and finish nodes for the full paths
        if num_in == 0:
            nx_graph.add_edge('Start', block, weight=edge_weight)
        if num_out == 0:
            nx_graph.add_edge(block, 'Finish', weight=0)

    # DFS for finding biggest path
    best_found = ['', 0]
    thread_block_queue = deque()
    thread_block_queue.append([['Start'], 0])
    while (len(thread_block_queue) != 0):
        current_path = thread_block_queue.pop()
        if (current_path[0][-1] == 'Finish'):
            if (current_path[1] > best_found[1]):
                best_found = [current_path[0].copy(), current_path[1]]
        for dependent_thread_block in nx_graph.neighbors(current_path[0][-1]):
            next_path = [current_path[0].copy(), current_path[1]]
            next_path[0].append(dependent_thread_block)
            next_path[1] = current_path[1] + nx_graph.get_edge_data(next_path[0][-2], next_path[0][-1],
                    default=0)['weight']
            thread_block_queue.append(next_path)
    path = best_found[0]
    total_cost = best_found[1]

    # Print path
    kernel_list = []
    path.remove('Start')
    path.remove('Finish')
    for block_index in range(len(path)):
        block = path[block_index]
        kernel_name = block.split('_')[0]
        kernel_list.append(int(kernel_name.split('-')[1]))
        thread_block = block.split('_')[1]
        print(str(block_index + 1) + ': ' + kernel_name + ', thread block-' + thread_block)

    # Graph the path
    if graph:
        graph_dependencies(time_report=False, path=path)

    print("Ideal Total Cycle Time: " + str(total_cost))
    print("Sim Total Cycle Time: " + str(sim_stats[('kernel-' + str(end_kernel))]['total_time']))
    return



"""""""""""""""

 Printing Info

"""""""""""""""

def print_dependency_stats(graph):
    dependency_stats_title = "=   Dependency Stats   ="
    dependency_stats_title = "\n" + ("=" * len(dependency_stats_title)) + "\n" + \
            dependency_stats_title + "\n" + ("=" * len(dependency_stats_title))
    print(dependency_stats_title)
    if graph:
        global trace_tbd_graph, sim_tbd_graph
        graph_dependencies(time_report=False)

    # Sim stat info
    for kernel in range(start_kernel, end_kernel + 1):
        kernel_name = 'kernel-' + str(kernel)
        print("Dependent sim thread_blocks in " + kernel_name + ":", end = ' ')
        print(str(len(sim_stats[kernel_name]["dependencies"])), end = '')
        print("/" + str(len(sim_stats[kernel_name]["thread_blocks"])))
    return


def print_dependencies(kernels=[], thread_blocks=[], info='sim'):
    print_dependencies_helper(kernels=kernels, thread_blocks=thread_blocks, \
            info=sim_stats, info_name='sim')
    return


def print_dependencies_helper(kernels=[], thread_blocks=[], info=sim_stats, \
        info_name='sim'):
    dep_title = "Kernel Trace" if (info_name == 'trace') else "Simulation Output"
    kernel_dep_title = "=   " + dep_title + " for kernels: " + str(kernels) + \
            " and CTA(s): " + str(thread_blocks) + "   ="
    kernel_dep_title = "\n" + ("=" * len(kernel_dep_title)) + "\n" + \
            kernel_dep_title + "\n" + ("=" * len(kernel_dep_title))
    print(kernel_dep_title)

    needed_info = {}
    for kernel in info:
        kernel_num = kernel.split('-')[1]
        if (len(kernels) != 0) and (kernel_num not in kernels):
            continue
        needed_info[kernel] = {}
        for block_depend in info[kernel]['dependencies']:
            thread_block = block_depend.split('_')[1]
            if (len(thread_blocks) == 0) or (thread_block in thread_blocks):
                needed_info[kernel][thread_block] = []
                block_name = kernel + '_' + thread_block
                if len(info[kernel]['dependencies'][block_name]) != 0:
                    needed_info[kernel][thread_block] = \
                            info[kernel]['dependencies'][block_name].copy()

    pprint.pprint(needed_info)
    return

def print_kernel_names():
    kernel_names_title = "=   Kernel Names   ="
    kernel_names_title = "\n" + ("=" * len(kernel_names_title)) + "\n" + \
            kernel_names_title + "\n" + ("=" * len(kernel_names_title))
    print(kernel_names_title)
    for kernel in range(start_kernel, end_kernel + 1):
        kernel_name = 'kernel-' + str(kernel)
        print(kernel_name + " - " + sim_stats[kernel_name]["kernel_name"])


def print_sim():
    pprint.pprint(sim_stats)
    return


if __name__=="__main__":
    main()
