from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from gpu.primitives.warp import sum as warp_sum, WARP_SIZE
from math import ceildiv
from buffer import NDBuffer, DimList
from algorithm import sum
from layout import Layout, LayoutTensor
from os.atomic import Atomic

fn dot_device[
    in_layout: Layout, size: Int, dtype: DType
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # Each thread computes one partial product using vectorized approach as values in Mojo are SIMD based
    var partial_product: Scalar[dtype] = 0
    if global_i < UInt(size):
        partial_product = (a[global_i] * b[global_i]).reduce_add()

    # warp_sum() replaces all the shared memory + barriers + tree reduction
    total = warp_sum(partial_product)

    # Only lane 0 writes the result (all lanes have the same total)
    if lane_id() == 0:
        _ = Atomic[dtype].fetch_add(output, total)
