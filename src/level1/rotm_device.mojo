from gpu import grid_dim, block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level1.rotm
# applies the modified Givens transformation, H, to the 2 by n matrix
fn rotm_device[
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    param: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):
    var flag = param[0]
    if (n < 1):
        return

    var idx = block_idx.x * block_dim.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    var sh11 = Scalar[dtype](1.0)
    var sh12 = Scalar[dtype](1.0)
    var sh21 = Scalar[dtype](-1.0)
    var sh22 = Scalar[dtype](1.0)

    if (flag < 0.0):
        sh11 = param[1]
        sh12 = param[3]
        sh21 = param[2]
        sh22 = param[4]
    elif (flag == 0) :
        sh12 = param[3]
        sh21 = param[2]
    elif (flag == 1):
        sh11 = param[1]
        sh22 = param[4]
    else:
        return

    # Each thread processes multiple elements with stride
    for i in range(idx, n, n_threads):
        var ix = i * incx
        var iy = i * incy

        var w = x[ix]
        var z = y[iy]

        x[ix] = w * sh11 + z * sh12
        y[iy] = w * sh21 + z * sh22


fn blas_rotm[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    d_param: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext
) raises:
    comptime kernel = rotm_device[dtype]
    ctx.enqueue_function[kernel, kernel](
        n,
        d_x, incx,
        d_y, incy,
        d_param,
        grid_dim=ceildiv(n, TBsize),     # total thread blocks
        block_dim=TBsize                    # threads per block
    )
    ctx.synchronize()
