from sys import has_accelerator
from gpu.host import DeviceContext
from gpu.id import block_dim, grid_dim, thread_idx

from src import iamax, iamax_device
from random import rand, seed
from math import ceildiv

alias dtype = DType.float32
alias size = 51
alias TBsize = 512

def main():
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU:", ctx.name())

        d_v = ctx.enqueue_create_buffer[dtype](size)
        v = ctx.enqueue_create_host_buffer[dtype](size)

        seed()
        rand[dtype](v.unsafe_ptr(), size)

        # print("v = [", end="")
        # for i in range(size):
        #     print(" ", v[i], end="")
        # print("]")

        var res_cpu = iamax(size, v.unsafe_ptr(), 1)
        print("CPU result:", res_cpu)

        ctx.enqueue_copy(d_v, v)
        d_res = ctx.enqueue_create_buffer[DType.int64](1).enqueue_fill(-1)

        ctx.enqueue_function[iamax_device[TBsize]](
            size, d_v.unsafe_ptr(),
            1, d_res.unsafe_ptr(),
            grid_dim=ceildiv(size, TBsize),
            block_dim=TBsize)

        # print("GPU kernel done")
        with d_res.map_to_host() as res_gpu:
            print("GPU result:", res_gpu[0])


