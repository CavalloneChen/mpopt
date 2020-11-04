import os
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "cec17.h":
    void ceval(const double *x, const int nx, const int mx, double *f, int func_num, char *path);

cpdef eval(np.ndarray X, int func_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = current_dir
    lib_dir = os.path.join(current_dir, 'lib')
    path_string_as_byte = lib_dir.encode('UTF-8')
    row_size, col_size = np.shape(X)
    cdef np.ndarray y = np.zeros(row_size, dtype=np.float64)
    cdef double *X_pointer = <double *> X.data
    cdef double *y_pointer = <double *> y.data
    cdef char *path_pointer = path_string_as_byte

    #print(lib_dir)
    ceval(X_pointer, col_size, row_size, y_pointer, func_id, path_pointer)
    return y
