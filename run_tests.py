"""""""""""""""""""""""""""""""""""""""""""""""""""
Run Many Tests
===================================================
Filename: run_tests.py
Author: Reese Kuper
Purpose: Run multiple track_depdencies.py per each
test specified in the test_list.yml file
"""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import sys
import time
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--depth", help = \
            "Specify depth applied to all benchmark (default = 2)", default="2")
    parser.add_argument("-s", "--start", help = \
            "Specify start kernel applied to all benchmark (default = 1)", default="1")
    parser.add_argument("-e", "--end", help = \
            "Specify last kernel applied to all benchmark (default = inf)", default=float('inf'))
    parser.add_argument("-u", "--update", help = \
            "Update each test (default = False)", action='store_true')
    parser.add_argument("-i", "--independent", help = \
            "Update each test (default = False)", action='store_true')
    args = parser.parse_args()

    log_file = open('run_tests.log', 'wt')
    sys.stdout = log_file

    if not os.path.exists('test_list.yml'):
        print("[ERROR] test_list.yml was not found.")
        return

    yml_file = open('test_list.yml', 'r')
    tests = yaml.safe_load(yml_file)

    base_args = ' -d ' + args.depth + ' -s ' + args.start
    if args.end != float('inf'):
        base_args += ' -e ' + args.end
    if args.update:
        base_args += ' -u '
    if args.independent:
        base_args += ' -i '

    total_time = time.time()
    for benchmark in tests:
        benchmark_title = '=   ' + benchmark + '   ='
        benchmark_title = "=" * len(benchmark_title) + '\n' + benchmark_title + \
                '\n' + "=" * len(benchmark_title)
        print(benchmark_title)
        bench_args = base_args + ' -b ' + benchmark
        for sass in tests[benchmark]:
            sass_args = bench_args + ' -a ' + sass
            for test in tests[benchmark][sass]:
                test_args = sass_args + ' -t ' + test
                print('Running: [' + sass + '] ' + test + '...', end='')
                test_time = time.time()
                os.system("python3 track_dependencies.py" + test_args)
                print(' Done in ' + str(round((time.time() - test_time), 4)) + "s")

                test_log_file = "./benchmarks/" + benchmark + "/" + test + "/" + test + ".log"
                if not os.path.exists(test_log_file):
                    print("[ERROR] test log file not found\n")
                    continue

                # Print timings from test
                times = []
                with open(test_log_file, 'r', encoding = 'utf-8') as test_log_file_fp:
                    for line in test_log_file_fp:
                        if "Cycle Time: " in line:
                            times.insert(0, line.rstrip())

                # Print times in order
                for time_stat in times:
                    print(time_stat)
                print()
                sys.stdout.flush()


    print_total_time = 'Total Time: ' + str(round((time.time() - total_time), 1)) + "s"
    print_total_time = '\n' + "-" * len(print_total_time) + '\n' + print_total_time
    print(print_total_time)
    log_file.close()
    return

if __name__=="__main__":
    main()
