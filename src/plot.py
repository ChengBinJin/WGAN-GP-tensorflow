import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections


_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush(save_folder):
    prints = []
    for name, vals in _since_last_flush.items():
        sum_ = 0
        keys = vals.keys()
        values = vals.values()
        num_keys = len(list(keys))
        for val in values:
            sum_ += val

        prints.append("{}\t{}".format(name, sum_/num_keys))
        _since_beginning[name].update(vals)

        x_vals = _since_beginning[name].keys()
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(save_folder, name.replace(' ', '_')+'.jpg'))

    # print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()
