from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import thread_idx, block_dim, block_idx, grid_dim, barrier
from os.atomic import Atomic
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# Pass 1: each block finds its local max_val and max_idx
fn iamax_device_partial[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    sx: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    partial_vals: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    partial_idxs: UnsafePointer[Int64, MutAnyOrigin]
):
    # Quick return if possible
    if n == 1:
        return

    shared_indices = stack_allocation[
        BLOCK,
        Int,
        address_space = AddressSpace.SHARED
    ]()
    shared_values = stack_allocation[
        BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED
    ]()

    var local_tid = thread_idx.x
    var global_tid = block_idx.x * BLOCK + local_tid
    var n_threads = grid_dim.x * BLOCK

    # Each thread finds its local max
    var local_max_id = -1
    var local_max_val = Scalar[dtype](-1)

    for i in range(global_tid, n, n_threads):
        var idx = i * incx
        var val = abs(sx[idx])

        if val > local_max_val:
            local_max_val = val
            local_max_id = i

    shared_indices[local_tid] = local_max_id
    shared_values[local_tid] = local_max_val

    barrier()

    # Parallel reduction to find max within the block
    var stride = BLOCK // 2
    while stride > 0:
        if local_tid < stride:
            var other_idx = shared_indices[local_tid + stride]
            var other_val = shared_values[local_tid + stride]
            var my_idx = shared_indices[local_tid]
            var my_val = shared_values[local_tid]

            if other_val > my_val or (other_val == my_val and other_idx != -1
                                      and (my_idx == -1 or other_idx < my_idx)):
                shared_indices[local_tid] = other_idx
                shared_values[local_tid] = other_val
        barrier()
        stride //= 2

    if local_tid == 0:
        partial_vals[block_idx.x] = shared_values[0]
        partial_idxs[block_idx.x] = shared_indices[0]


# Pass 2: Final reduction
fn iamax_device_reduce[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    n_blocks: Int,
    partial_vals: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    partial_idxs: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    d_res: UnsafePointer[Scalar[DType.int64], MutAnyOrigin]
):
    # Quick return if possible
    d_res[0] = 0
    if n == 1:
        return

    var shared_vals = stack_allocation[
        BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED
    ]()
    var shared_idxs = stack_allocation[
        BLOCK,
        Int,
        address_space = AddressSpace.SHARED
    ]()

    var local_tid = thread_idx.x

    if local_tid < n_blocks:
        shared_vals[local_tid] = partial_vals[local_tid]
        shared_idxs[local_tid] = Int(partial_idxs[local_tid])
    else:
        shared_vals[local_tid] = Scalar[dtype](-1)
        shared_idxs[local_tid] = -1

    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if local_tid < stride:
            var other_val = shared_vals[local_tid + stride]
            var other_idx = shared_idxs[local_tid + stride]
            var my_val = shared_vals[local_tid]
            var my_idx = shared_idxs[local_tid]

            if other_val > my_val or (other_val == my_val and other_idx != -1
                                      and (my_idx == -1 or other_idx < my_idx)):
                shared_vals[local_tid] = other_val
                shared_idxs[local_tid] = other_idx
        barrier()
        stride //= 2

    if local_tid == 0:
        d_res[0] = shared_idxs[0]

# level1.iamax
# finds the index of the first element having maximum absolute value
fn blas_iamax[dtype: DType](
    n: Int,
    d_v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_res: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    ctx: DeviceContext
) raises:
    blas_error_if["blas_iamax", "n < 0"](n<=0)
    blas_error_if["blas_iamax", "incx <= 0"](incx <= 0)

    # Limit number of blocks to TBsize max for second pass
    var n_blocks = min(ceildiv(n, TBsize), TBsize)

    var partial_vals = ctx.enqueue_create_buffer[dtype](n_blocks)
    var partial_idxs = ctx.enqueue_create_buffer[DType.int64](n_blocks)

    comptime kernel1 = iamax_device_partial[TBsize, dtype]
    ctx.enqueue_function[kernel1, kernel1](
        n, d_v, incx,
        partial_vals.unsafe_ptr(), partial_idxs.unsafe_ptr(),
        grid_dim=n_blocks,
        block_dim=TBsize
    )

    comptime kernel2 = iamax_device_reduce[TBsize, dtype]
    ctx.enqueue_function[kernel2, kernel2](
        n, n_blocks,
        partial_vals.unsafe_ptr(), partial_idxs.unsafe_ptr(),
        d_res,
        grid_dim=1,
        block_dim=TBsize
    )

    ctx.synchronize()
