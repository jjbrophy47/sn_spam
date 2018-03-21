import os
import sys


def out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
