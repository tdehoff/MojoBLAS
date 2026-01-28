from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import thread_idx, block_dim, block_idx, barrier

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

    local_tid = thread_idx.x

    # Handle incx: each thread processes elements at intervals of incx
    var thread_global_idx = Int(local_tid * incx)

    # Initialize shared memory with index and absolute value
    if thread_global_idx < n:
        shared_indices[local_tid] = thread_global_idx
        shared_values[local_tid] = abs(sx[thread_global_idx])
    else:
        # Initialize out-of-bounds threads with minimum values
        shared_indices[local_tid] = -1
        shared_values[local_tid] = Scalar[dtype](0)

    barrier()

    # Parallel reduction to find maximum absolute value
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

    # Thread 0 writes the final result
    if local_tid == 0:
        result[0] = shared_indices[0]
