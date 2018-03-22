import os
import sys
import time as t


def out(message='', newline=1):
    msg = '\n' + message if newline == 1 else message
    sys.stdout.write(msg)
    sys.stdout.flush()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def time(t1, suffix='m'):
    elapsed = t.time() - t1

    if suffix == 'm':
        elapsed /= 60.0
    if suffix == 'h':
        elapsed /= 3600.0

    out('%.2f%s' % (elapsed, suffix), 0)
