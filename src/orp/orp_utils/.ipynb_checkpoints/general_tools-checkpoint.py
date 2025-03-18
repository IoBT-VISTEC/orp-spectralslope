from itertools import combinations
import numpy as np


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def combination(arr, r):
    return list(combinations(arr, r))


def map_to_class_label(CLASS_MAP, stages):
    return np.array([CLASS_MAP[str(s)] for s in stages])


def map_to_class_index(epoch_classes_map, yt):
    epoch_classes_map_converted = {
        v: int(k) for k, v in epoch_classes_map.items()
    }
    return np.array([epoch_classes_map_converted[y] for y in yt])
        
