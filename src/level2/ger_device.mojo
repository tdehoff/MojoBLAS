from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.ger
# Computes rank-1 update of given matrix: A := A + αxy'
fn sger_device[
    BLOCK: Int,
](
    m: Int,
    n: Int,
    alpha: Scalar[DType.float32],
    x: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    incy: Int,
    A: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    lda: Int,
):
    if m < 1 or n < 1:
        return

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    var total = m * n

    for idx in range(global_i, total, n_threads):
        var row = idx // n
        var col = idx % n

        var x_val = x[row * incx]
        var y_val = y[col * incy]

        A[row * lda + col] += alpha * x_val * y_val

fn dger_device[
    BLOCK: Int,
](
    m: Int,
    n: Int,
    alpha: Scalar[DType.float64],
    x: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    incy: Int,
    A: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    lda: Int,
):
    if m < 1 or n < 1:
        return

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    var total = m * n

    for idx in range(global_i, total, n_threads):
        var row = idx // n
        var col = idx % n

        var x_val = x[row * incx]
        var y_val = y[col * incy]

        A[row * lda + col] += alpha * x_val * y_val

fn blas_ger[dtype: DType](
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    d_A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lda: Int,
    ctx: DeviceContext
) raises:
    if m < 1 or n < 1:
        return
    
    var total = m * n

    if dtype == DType.float32:
        ctx.enqueue_function[sger_device[TBsize], sger_device[TBsize]](
            m, n, alpha,
            d_x, incx,
            d_y, incy,
            d_A, lda,
            grid_dim=ceildiv(total, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dger_device[TBsize], dger_device[TBsize]](
            m, n, alpha,
            d_x, incx,
            d_y, incy,
            d_A, lda,
            grid_dim=ceildiv(total, TBsize),
            block_dim=TBsize,
        )
    else:
        return

    ctx.synchronize()
