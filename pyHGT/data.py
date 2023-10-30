import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd

import math
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp
import torch.nn.functional as F

class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
        self.t_r_id_p_dict_train-->time:relaton:quadid:path list(2维，第一行为正确的triple之间的所有path)
        self.t_max_num-->每个时隙内所有triple之间的最大路径数
        '''
        '''

        self.t_r_id_p_dict_train = defaultdict(lambda: {})
        self.t_r_id_p_dict_valid = defaultdict(lambda: {})
        self.t_r_id_p_dict_test = defaultdict(lambda: {})

        self.t_paths_train = defaultdict(lambda: [])
        self.t_paths_valid = defaultdict(lambda: [])
        self.t_paths_test = defaultdict(lambda: [])

        self.t_paths_len_train = defaultdict(lambda: [])
        self.t_paths_len_valid = defaultdict(lambda: [])
        self.t_paths_len_test = defaultdict(lambda: [])

        self.t_max_num_train = {}
        self.t_max_num_valid = {}
        self.t_max_num_test = {}
        '''

        self.t_r_id_p_dict = defaultdict(lambda: {})
        self.t_r_id_target_dict = defaultdict(lambda: {})

        self.r_copy = defaultdict(lambda: {})
        #self.r_copy_t = defaultdict(lambda: {})


        self.t_paths = defaultdict(lambda: [])

        self.t_paths_len = defaultdict(lambda: [])

        self.t_paths_time = defaultdict(lambda: [])
        self.t_paths_m_time = defaultdict(lambda: [])
    
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
