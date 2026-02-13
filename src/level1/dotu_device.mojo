from gpu import thread_idx, block_idx, block_dim, grid_dim
from os.atomic import Atomic
from memory import stack_allocation
from gpu.host import DeviceContext
from math import ceildiv
from complex import ComplexScalar, ComplexFloat64

comptime TBsize = 512

# level1.dotu
# computes the dot product of two vectors of complex numbers x' * y
fn dotu_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    if n < 1:
        return

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x
    var local_i = thread_idx.x

    shared_res = stack_allocation[
        BLOCK,
        ComplexScalar[DType.float64],
        address_space = AddressSpace.SHARED
    ]()

    var thread_sum = ComplexFloat64(0, 0)
    for i in range(global_i, n, n_threads):
        x_val = ComplexFloat64(Scalar[DType.float64](x[i*incx*2]), Scalar[DType.float64](x[i*incx*2+1]))
        y_val = ComplexFloat64(Scalar[DType.float64](y[i*incy*2]), Scalar[DType.float64](y[i*incy*2+1]))

        thread_sum = thread_sum + (x_val * y_val)

    shared_res[local_i] = thread_sum
    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if local_i < stride:
            shared_res[local_i] = shared_res[local_i] + shared_res[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        _ = Atomic[dtype].fetch_add(
            UnsafePointer(to=output[0]),
            Scalar[dtype](shared_res[0].re)
        )
        _ = Atomic[dtype].fetch_add(
            UnsafePointer(to=output[1]),
            Scalar[dtype](shared_res[0].im)
        )


fn blas_dotu[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext
) raises:
    comptime kernel = dotu_device[TBsize, dtype]
    ctx.enqueue_function[kernel, kernel](
        n, d_x, incx,
        d_y, incy, d_out,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
