from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import thread_idx, block_dim, block_idx, barrier
from os.atomic import Atomic

# level1.iamax
# finds the index of the first element having maximum absolute value
fn iamax_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    sx: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    result: UnsafePointer[Scalar[DType.int64], MutAnyOrigin]
):
    result[0] = -1
    if n < 1 or incx <= 0:
        return

    result[0] = 0
    if n == 1:
        return

    # Shared memory for indices and values
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
    var global_tid = block_idx.x * block_dim.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

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
    var stride = block_dim.x // 2
    while stride > 0:
        if local_tid < stride:
            var other_idx = local_tid + stride
            if other_idx < BLOCK and shared_indices[other_idx] >= 0:
                if shared_values[other_idx] > shared_values[local_tid]:
                    shared_values[local_tid] = shared_values[other_idx]
                    shared_indices[local_tid] = shared_indices[other_idx]
                # Resolve ties by selecting smaller index
                elif shared_values[other_idx] == shared_values[local_tid] and
                     shared_indices[other_idx] < shared_indices[local_tid]:
                    shared_indices[local_tid] = shared_indices[other_idx]

        barrier()
        stride //= 2

    # Thread 0 atomically updates the global result
    # TODO: complete this
    if local_tid == 0:
        result[0] = shared_indices[0]
