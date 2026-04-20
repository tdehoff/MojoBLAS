from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.tpmv
# Performs matrix-vector multiplication of form
#    x := A*x,   or   x := A**T*x
# where x is a vector and A is an n by n unit or non-unit, upper or lower
# triangular band matrix with (k+1) diagonals.
fn stpmv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    AP: UnsafePointer[Float32, ImmutAnyOrigin],
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    workspace: UnsafePointer[Float32, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    for i in range(global_i, n, n_threads):
        var sum = x[i * incx]
        if trans:
            if uplo:
                var col_start = i * (i + 1) / 2
                if not diag:
                    sum *= AP[col_start + i]
                for j in range(0, i):
                    sum += AP[col_start + j] * x[j * incx]
            else:
                var col_start = i * n - ((i - 1) * i) / 2
                if not diag:
                    sum *= AP[col_start]
                for j in range(i + 1, n):
                    sum += AP[col_start + (j - i)] * x[j * incx]
        else:
            if uplo:
                if not diag:
                    sum *= AP[i * (i + 1) / 2 + i]
                for j in range(i + 1, n):
                    var index = j * (j + 1) / 2 + i
                    sum += AP[index] * x[j * incx]
            else:
                if not diag:
                    sum *= AP[i * n - ((i - 1) * i) / 2]
                for j in range(0, i):
                    var index = j * n - ((j - 1) * j) / 2 + (i - j)
                    sum += AP[index] * x[j * incx]
        workspace[i * incx] = sum


fn dtpmv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    AP: UnsafePointer[Float64, ImmutAnyOrigin],
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    workspace: UnsafePointer[Float64, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    for i in range(global_i, n, n_threads):
        var sum = x[i * incx]
        if trans:
            if uplo:
                var col_start = i * (i + 1) / 2
                if not diag:
                    sum *= AP[col_start + i]
                for j in range(0, i):
                    sum += AP[col_start + j] * x[j * incx]
            else:
                var col_start = i * n - ((i - 1) * i) / 2
                if not diag:
                    sum *= AP[col_start]
                for j in range(i + 1, n):
                    sum += AP[col_start + (j - i)] * x[j * incx]
        else:
            if uplo:
                if not diag:
                    sum *= AP[i * (i + 1) / 2 + i]
                for j in range(i + 1, n):
                    var index = j * (j + 1) / 2 + i
                    sum += AP[index] * x[j * incx]
            else:
                if not diag:
                    sum *= AP[i * n - ((i - 1) * i) / 2]
                for j in range(0, i):
                    var index = j * n - ((j - 1) * j) / 2 + (i - j)
                    sum += AP[index] * x[j * incx]
        workspace[i * incx] = sum


fn blas_tpmv[dtype: DType](
    uplo: Int,
    trans: Bool,
    diag: Int,
    n: Int,
    d_AP: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    ctx: DeviceContext,
) raises:
    # NOTE: add error checking here?
    
    var trans_i = 1 if trans else 0

    var workspace = ctx.enqueue_create_buffer[dtype](n)

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[stpmv_device, stpmv_device](
            uplo, trans_i, diag,
            n, d_AP,
            d_x, incx,
            workspace,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dtpmv_device, dtpmv_device](
            uplo, trans_i, diag,
            n, d_AP,
            d_x, incx,
            workspace,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_tpmv: Unsupported type")

    ctx.enqueue_copy(d_x, workspace)

    ctx.synchronize()
