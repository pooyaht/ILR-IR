from collections import defaultdict

import dill


class Graph():
    def __init__(self):
        super(Graph, self).__init__()

        self.t_r_id_p_dict = defaultdict(lambda: {})
        self.t_r_id_target_dict = defaultdict(lambda: {})

        self.r_copy = defaultdict(lambda: {})

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
