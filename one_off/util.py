import os
import sys


def out(message='', newline=1):
    msg = '\n' + message if newline == 1 else message
    sys.stdout.write(msg)
    sys.stdout.flush()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
