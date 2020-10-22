import os
import sys
import numpy as np
import torch

sys.path.append("/home/lyf/projects/benchmarks/cec2013")
import cec13
sys.path.append("/home/lyf/projects/benchmarks/cec2017")
import cec17
sys.path.append("/home/lyf/projects/benchmarks/cec2020")
import cec20

def func_wrapper(func, func_id):

    def wrapped(x):
        
        origin_type = type(x)
        if origin_type is not list:
            origin_shape = x.shape
            dim = origin_shape[-1]
        
            if origin_type is torch.Tensor:
                x = x.cpu().numpy()
            x = x.reshape((-1, dim)).tolist()

        if func == "cec13":
            tmp = cec13.eval(x, func_id+1)
        elif func == "cec17":
            tmp = cec17.eval(x, func_id+1)
        elif func == "cec20":
            tmp = cec20.eval(x, func_id+1)
        else:
            raise Exception("No such benchmark!")
        
        if origin_type is np.ndarray:
            return np.array(tmp).reshape(origin_shape[:-1])
        elif origin_type is torch.Tensor:
            return torch.tensor(tmp).reshape(origin_shape[:-1])
        elif type(x) is list and type(x[0]) is list:
            return tmp
        else:
            return tmp[0]

    return wrapped


class Evaluator(object):

    def __init__(self, benchmark):
        
        if benchmark == 'CEC13':
            self.func = [func_wrapper("cec13", func_id) for func_id in range(28)]
            self.func_num = 28
        elif benchmark == 'CEC17':
            self.func = [func_wrapper("cec17", func_id) for func_id in range(30)]
            self.func_num = 30
        elif benchmark == 'CEC20':
            self.func = [func_wrapper("cec20", func_id) for func_id in range(10)]
            self.func_num = None
        else:
            raise Exception("Benchmark")

    def init(self, func_id=None, trace=False):
        self.func_id = func_id
        self.num_eval = 0
        self.num_batch = 0
        self.best_idv = None
        self.best_fit = None
        
        self.record_trace = trace
        self.trace = []
    
    def __call__(self, x, func_id=None):
        if self.func_id is None:
            raise Exception("No function id is given")
        
        ans = self.func[self.func_id](x)
        
        # record update
        self.num_eval += x.shape[0]
        self.num_batch += 1
        
        if self.best_fit is None or min(ans) < self.best_fit:
            min_idx = np.argmin(ans)
            self.best_idv = x[min_idx, :]
            self.best_fit = ans[min_idx]
        
        if self.record_trace:
            self.trace.append([self.num_eval, self.best_idv, self.best_fit])
        
        return ans
