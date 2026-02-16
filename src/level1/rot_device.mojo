from gpu import grid_dim, block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level1.rot
# applies a plane rotation to vectors x and y
fn rot_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    c: Scalar[dtype],
    s: Scalar[dtype]
):
    if (n < 1):
        return

    var global_tid = block_idx.x * block_dim.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # Each thread processes multiple elements with stride
    for i in range(global_tid, n, n_threads):
        var ix = i * incx
        var iy = i * incy

        var tmp = c * x[ix] + s * y[iy]
        y[iy] = c * y[iy] - s * x[ix]
        x[ix] = tmp


fn blas_rot[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    c: Scalar[dtype],
    s: Scalar[dtype],
    ctx: DeviceContext
) raises:
    comptime kernel = rot_device[TBsize, dtype]
    ctx.enqueue_function[kernel, kernel](
        n,
        d_x, incx,
        d_y, incy,
        c, s,
        grid_dim=ceildiv(n, TBsize),     # total thread blocks
        block_dim=TBsize                    # threads per block
    )
    ctx.synchronize()


fn rot_device_complex[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[ComplexScalar[dtype], MutAnyOrigin],
    incx: Int,
    y: UnsafePointer[ComplexScalar[dtype], MutAnyOrigin],
    incy: Int,
    c: Scalar[dtype],
    s: UnsafePointer[ComplexScalar[dtype], MutAnyOrigin]
):
    if (n < 1):
        return

    var global_tid = block_idx.x * block_dim.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # Each thread processes multiple elements with stride
    for i in range(global_tid, n, n_threads):
        var ix = i * incx
        var iy = i * incy

        var tmp = c * x[ix] + s[] * y[iy]
        y[iy] = c * y[iy] - s[] * x[ix]
        x[ix] = tmp


fn blas_rot_complex[dtype: DType](
    n: Int,
    d_x: UnsafePointer[ComplexScalar[dtype], MutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[ComplexScalar[dtype], MutAnyOrigin],
    incy: Int,
    c: Scalar[dtype],
    s: ComplexScalar[dtype],
    ctx: DeviceContext
) raises:
    comptime kernel = rot_device[TBsize, dtype]
    ctx.enqueue_function[kernel, kernel](
        n,
        d_x, incx,
        d_y, incy,
        c, UnsafePointer(to=s),
        grid_dim=ceildiv(n, TBsize),     # total thread blocks
        block_dim=TBsize                    # threads per block
    )
    ctx.synchronize()
