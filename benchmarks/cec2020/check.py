import numpy as np
import cec20

a = np.random.uniform(-100, 100, [10, 30])
tmp = cec20.eval(a, 1)
print(tmp)
