from sys import has_accelerator
from gpu.host import DeviceContext
from gpu.id import block_dim, grid_dim, thread_idx

from src import iamax_device
from random import rand, seed
from math import ceildiv
from python import Python

alias dtype = DType.float32
alias size = 51
alias TBsize = 512

def main():
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU:", ctx.name())

        # Allocate GPU and CPU memory
        d_v = ctx.enqueue_create_buffer[dtype](size)
        v = ctx.enqueue_create_host_buffer[dtype](size)

        # Generate an array of random numbers on CPU
        seed()
        rand[dtype](v.unsafe_ptr(), size)

        # Copy random vector from CPU to GPU memory
        ctx.enqueue_copy(d_v, v)

        # Allocate memory for a single int on GPU to store result, initialize to -1
        d_res = ctx.enqueue_create_buffer[DType.int64](1).enqueue_fill(-1)

        # Launch Mojo GPU kernel
        ctx.enqueue_function[iamax_device[TBsize]](
            size, d_v.unsafe_ptr(),
            1, d_res.unsafe_ptr(),
            grid_dim=ceildiv(size, TBsize),     # total thread blocks
            block_dim=TBsize)                   # threads per block

        # Move result from CPU to GPU and print
        with d_res.map_to_host() as res_gpu:
            print("GPU result:", res_gpu[0])

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move values in v to a SciPy-compatible array
        py_list = Python.list()
        for i in range(size):
            py_list.append(v[i])
        np_v = np.array(py_list, dtype=np.float32)

        # Run SciPy BLAS routine
        sp_res = sp_blas.isamax(np_v)

        print("SciPy result:")
        print(sp_res)
