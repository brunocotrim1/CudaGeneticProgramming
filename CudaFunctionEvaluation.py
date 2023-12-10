import numpy as np
from numba import cuda, float32,int32
import math
import numpy as np
import pandas as pd
import string
import time


template = """
@cuda.jit
def dynamic_kernel_gpu(input, result,y_values):
    tx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    blockDim_x, blockDim_y = cuda.blockDim.x, cuda.blockDim.y
    shared_memory = cuda.shared.array({funLen}, dtype=float32)

    i = cuda.threadIdx.x
    while(i<len(shared_memory)):
       shared_memory[i] = 0
       i+=cuda.blockDim.x
    i = tx
    while(i<len(result)):
      result[i] = 0
      i+=cuda.blockDim.x
    {inputs}
    cuda.syncthreads()
    if tx < input.shape[0]:
      {expressions}
    cuda.syncthreads()
    i = cuda.threadIdx.x
    while(i<len(result)):
       cuda.atomic.add(result,i,shared_memory[i])
       i+=cuda.blockDim.x
    cuda.syncthreads()
"""


def score(line):
    for u in ["sinf", "cosf", "tanf", "sqrtf", "expf"]:
        if u == "sqrtf":
          line = line.replace(u, "math.sqrt")
          continue
        if u == "expf":
          line = line.replace(u, f"(2.718281828459045 ** )")
          continue
        line = line.replace(u, f"np.{u[:-1]}")
    for c in df.columns[1:]:
        index = string.ascii_lowercase.index(c) + 1
        line = line.replace(f"_{c}_", f"input_{index}")
    return line


def run_kernel(device_input_array,expression,blocks,threads_block,df,funs,
               y_values,device_result_array):
  exec(expression, globals())
  start = time.time()
  print("Starting Kernel")
  eval("dynamic_kernel_gpu[num_blocks,block_size](device_input_array,device_result_array,y_values)")
  end = time.time()
  print("Kernel Took",end - start)
  result = device_result_array.copy_to_host(host_result_array)
  return result


df = pd.read_csv("data.csv")
funs = [ line.strip() for line in open("functions.txt").readlines() ]
host_input_array = df.to_numpy()
block_size = 512
num_blocks = math.ceil((len(host_input_array)) / block_size)
print("Number of blocks",num_blocks)
print("Number of threads: ",num_blocks*block_size)
print("Number of elements: ",len(host_input_array))
gpu = cuda.get_current_device()
print("Using GPU:", gpu.name)
print("GPU memory total:", gpu.compute_capability)
result = np.zeros(len(funs), dtype=np.float32)

expressions = []
funs = funs
for fun in funs:
    expressions.append(score(fun))


device_input_array = cuda.to_device(host_input_array)
print(host_input_array.shape)
start = time.time()
result = []

best_expression = ""
best_mean = 2
values =  df["y"].to_numpy()
device_y_values = cuda.to_device(values)
chunkSize = min(len(funs),50)
print("ChunkSize: ",chunkSize)
indexAssignments = "";
host_result_array = np.zeros((chunkSize),)
device_result_array = cuda.to_device(host_result_array)
for i in range(0,len(df.columns)):
  indexAssignments += f"input_{i} = input[tx,{i}];"

for i in range(0, len(funs), chunkSize):
    chunk = funs[i:i + chunkSize]
    expressions=[]
    for fun in chunk:
      expressions.append(score(fun))

    kernel_string = ""
    for i in range(len(expressions)):
      kernel_string += (f"cuda.atomic.add(shared_memory,{i},(({expressions[i]})-y_values[tx])**2);")#[{i}][tx]
    kernel_string= template.format(expressions=kernel_string,funLen=len(chunk),inputs = indexAssignments)
    resultTemp = run_kernel(device_input_array,kernel_string,num_blocks,
                            block_size,df,chunk,device_y_values,
                            device_result_array)
    for i in range(0,len(resultTemp)):
      if i >= len(chunk):
        break;
      mean = resultTemp[i]/len(host_input_array)
      if(i == 0):
        print(f"{mean} {chunk[i]}")
      if mean < best_mean:
        best_mean = mean
        best_expression = chunk[i]
    del resultTemp

end = time.time()
print(f"{best_mean} {best_expression}")
print("Parallel Processing took in total: ",end - start)
del device_y_values
del values
