# Cartesian Product for NumPy - 40x faster than NumPy + itertools.product 

## pip install cythoncartesian2

### Tested against Windows / Python 3.11 / Anaconda

## Cython (and a C/C++ compiler) must be installed




```python

cartesian_product(*args, outputdtype=np.uint32, dtype=np.uint32):
    Calculate the Cartesian product of input arrays.

    Parameters:
    - *args: Variable number of input arrays.
    - outputdtype (numpy.dtype): Data type of the output array.
    - dtype (numpy.dtype): Data type used for intermediate calculations. # be careful!

    Returns:
    - numpy.ndarray: Cartesian product of input arrays.

	
import random
from cythoncartesian2 import cartesian_product
import numpy as np

# Strings are NOT supported!

args=[[h*random.uniform(1,4) for h in (range(random.randint(2,9)))] for x in range(9)]
q=cartesian_product(*args,outputdtype=np.float32,dtype=np.uint32)

# array([[0.       , 0.       , 0.       , ..., 0.       , 0.       ,
#         0.       ],
#        [3.529998 , 0.       , 0.       , ..., 0.       , 0.       ,
#         0.       ],
#        [0.       , 3.715651 , 0.       , ..., 0.       , 0.       ,
#         0.       ],
#        ...,
#        [3.529998 , 7.956308 , 5.9014587, ..., 1.0379078, 7.9018135,
#         8.816498 ],
#        [0.       , 9.456019 , 5.9014587, ..., 1.0379078, 7.9018135,
#         8.816498 ],
#        [3.529998 , 9.456019 , 5.9014587, ..., 1.0379078, 7.9018135,
#         8.816498 ]], dtype=float32)

args=[[h for h in (range(8))] for x in range(9)]
q=cartesian_product(*args,outputdtype=np.uint8,dtype=np.uint32)

# %timeit q=cartesian_product(*args,outputdtype=np.uint8,dtype=np.uint32)
# 1.63 s ± 36.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %timeit (list(itertools.product(*args)))
# 11.3 s ± 180 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %timeit q=np.array(list(itertools.product(*args)),dtype=np.uint8)
# 1min 6s ± 282 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# q
# Out[3]:
# array([[0, 0, 0, ..., 0, 0, 0],
#        [1, 0, 0, ..., 0, 0, 0],
#        [2, 0, 0, ..., 0, 0, 0],
#        ...,
#        [5, 7, 7, ..., 7, 7, 7],
#        [6, 7, 7, ..., 7, 7, 7],
#        [7, 7, 7, ..., 7, 7, 7]], dtype=uint8)
```