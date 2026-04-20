from gpu import thread_idx, block_idx, block_dim, grid_dim
from os.atomic import Atomic
from memory import stack_allocation
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# Pass 1: each block finds its partial sum
fn dot_device_partial[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    partial_results: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x
    var local_i = thread_idx.x

    shared_res = stack_allocation[
        BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED
    ]()

    var thread_sum = Scalar[dtype](0)
    for i in range(global_i, n, n_threads):
        thread_sum += x[i * incx] * y[i * incy]

    shared_res[local_i] = thread_sum
    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if local_i < stride:
            shared_res[local_i] += shared_res[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        partial_results[block_idx.x] = shared_res[0]

# Pass 2: Final reduction
fn dot_device_reduce[
    BLOCK: Int,
    dtype: DType
](
    n_blocks: Int,
    partial_results: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    shared_res = stack_allocation[
        BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED
    ]()

    var local_i = thread_idx.x

    if local_i < n_blocks:
        shared_res[local_i] = partial_results[local_i]
    else:
        shared_res[local_i] = 0

    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if local_i < stride:
            shared_res[local_i] += shared_res[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        d_out[0] = shared_res[0]

# level1.dot
# computes the dot product of two vectors
fn blas_dot[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext
) raises:

    blas_error_if["blas_dot", "n < 0"](n < 0)
    blas_error_if["blas_copy", "incx == 0"](incx == 0)
    blas_error_if["blas_copy", "incy == 0"](incy == 0)

    # Limit number of blocks to TBsize max for second pass
    var n_blocks = min(ceildiv(n, TBsize), TBsize)
    var partial_results = ctx.enqueue_create_buffer[dtype](n_blocks)

    comptime kernel1 = dot_device_partial[TBsize, dtype]
    ctx.enqueue_function[kernel1, kernel1](
        n, d_x, incx,
        d_y, incy, partial_results.unsafe_ptr(),
        grid_dim=n_blocks,
        block_dim=TBsize,
    )

    comptime kernel2 = dot_device_reduce[TBsize, dtype]
    ctx.enqueue_function[kernel2, kernel2](
        n_blocks,
        partial_results.unsafe_ptr(),
        d_out,
        grid_dim=1,
        block_dim=TBsize
    )
    ctx.synchronize()
