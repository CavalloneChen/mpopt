import numpy as np
import cec17

a = np.random.uniform(-100, 100, [10, 30])
tmp = cec17.eval(a, 1)
print(tmp)
