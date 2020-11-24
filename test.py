import os
import sys
import glob
from subprocess import Popen, PIPE


things = ['a_b_c', 'a_b_b', 'a_g_e']
looking = ['b', 'a', 'b']
def find(things, looking):
    for thing in things:
        thing_list = thing.split('_')
        if not all(look in thing_list for look in looking):
            continue
        for item in looking:
            if item not in thing_list:
                break
            thing_list.remove(item)
            if thing_list == []:
                return True
    return False
print(find(things, looking))
