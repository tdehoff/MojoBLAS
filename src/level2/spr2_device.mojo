from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.spr2
# Performs symmetric packed rank-2 update:
# AP := alpha*x*y**T + alpha*y*x**T + AP
# uplo: 0 = upper triangle, 1 = lower triangle
# AP is stored in packed format (n*(n+1)/2 elements, column-major)
#
# Upper triangular packing (uplo=0):
#   AP[i + j*(j+1)//2] = A[i,j]  for 0 <= i <= j < n
#
# Lower triangular packing (uplo=1):
#   AP[j*n - j*(j-1)//2 + (i-j)] = A[i,j]  for 0 <= j <= i < n
#
# Each GPU thread handles one row i, writing to distinct AP positions,
# so no data races occur and the kernel is fully parallel.
fn sspr2_device(
    uplo: Int,
    n: Int,
    alpha: Float32,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int,
    AP: UnsafePointer[Float32, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # upper triangle: AP[i + j*(j+1)//2] for 0 <= i <= j
    if not uplo:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(i, n):
                AP[i + j * (j + 1) // 2] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]
    # lower triangle: AP[j*n - j*(j-1)//2 + (i-j)] for 0 <= j <= i
    else:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(0, i + 1):
                AP[j * n - j * (j - 1) // 2 + (i - j)] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]

fn dspr2_device(
    uplo: Int,
    n: Int,
    alpha: Float64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int,
    AP: UnsafePointer[Float64, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # upper triangle: AP[i + j*(j+1)//2] for 0 <= i <= j
    if not uplo:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(i, n):
                AP[i + j * (j + 1) // 2] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]
    # lower triangle: AP[j*n - j*(j-1)//2 + (i-j)] for 0 <= j <= i
    else:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(0, i + 1):
                AP[j * n - j * (j - 1) // 2 + (i - j)] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]

fn blas_spr2[dtype: DType](
    uplo: Int,
    n: Int,
    alpha: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    d_AP: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[sspr2_device, sspr2_device](
            uplo, n,
            alpha,
            d_x, incx,
            d_y, incy,
            d_AP,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dspr2_device, dspr2_device](
            uplo, n,
            alpha,
            d_x, incx,
            d_y, incy,
            d_AP,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_spr2: Unsupported type")

    ctx.synchronize()
