from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import thread_idx, block_idx, grid_dim, barrier
from os.atomic import Atomic

# level1.asum
# computes the sum of absolute values of vector elements
fn asum_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    sx: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):
    if n < 1 or incx <= 0:
        result[0] = 0
        return

    var local_tid = thread_idx.x

    var shared_sums = stack_allocation[
        BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED
    ]()

    var global_tid = block_idx.x * BLOCK + local_tid
    var n_threads = grid_dim.x * BLOCK

    # Each thread computes partial sum
    var local_sum = Scalar[dtype](0)

    for i in range(global_tid, n, n_threads):
        var idx = i * incx
        local_sum += abs(sx[idx])

    shared_sums[local_tid] = local_sum

    barrier()

    # Parallel reduction within block
    var stride = BLOCK // 2
    while stride > 0:
        if local_tid < stride:
            shared_sums[local_tid] += shared_sums[local_tid + stride]

        barrier()
        stride //= 2

    # Thread 0 atomically adds block result to global sum
    if local_tid == 0:
        _ = Atomic[dtype].fetch_add(result, shared_sums[0])
