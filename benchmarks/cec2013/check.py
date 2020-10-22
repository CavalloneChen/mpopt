import numpy as np
import cec13

a = np.random.uniform(-100, 100, [10, 30]).tolist()
tmp = cec13.eval(a, 1)
print(tmp)
