from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import thread_idx, block_dim, block_idx, barrier

# level1.iamax
# finds the index of the first element having maximum absolute value
fn iamax_device[BLOCK: Int](n: Int,
                            sx: UnsafePointer[Float32, MutAnyOrigin],
                            incx: Int,
                            result: UnsafePointer[Scalar[DType.int64], MutAnyOrigin]):
    # current issues:
    #                   1. Only supports n <= TBsize
    #                   2. Doesn't support incx > 1
    result[0] = -1
    if n < 1 or incx <= 0:
        return

    result[0] = 0
    if n == 1:
        return

    shared = stack_allocation[
        BLOCK,
        Int64, address_space = AddressSpace.SHARED
    ]()
    local_tid = thread_idx.x
    global_tid = block_dim.x * block_idx.x + thread_idx.x

    if global_tid < n:
        shared[local_tid] = Int64(global_tid)

    smax = abs(sx[shared[local_tid]])
    stride = block_dim.x // 2
    while stride > 0:
        if local_tid < stride:
            tmp = abs(sx[shared[local_tid + stride]])
            if tmp > smax:
                smax = tmp
                shared[local_tid] = shared[local_tid + stride]

        barrier()
        stride //= 2

    if local_tid == 0:
        result[0] = shared[0]

    # if incx == 1:
    #     # code for increment equal to 1
    #     smax = abs(sx[0])
    #     for i in range(1, n):
    #         if abs(sx[i]) > smax:
    #             result = i
    #             smax = abs(sx[i])
    # else:
    #     ix = 1
    #     smax = abs(sx[0])
    #     ix += incx
    #     for i in range(1, n):
    #         if abs(sx[ix]) > smax:
    #             result = i
    #             smax = abs(sx[ix])
    #         ix += incx

