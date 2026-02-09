from gpu import grid_dim, block_dim, global_idx

fn swap_device[dtype: DType](
    n: Int,
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int
):
    if (n <= 0):
        return
    if (incx == 0 or incy == 0):
        return

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    if (n <= n_threads):
        # Standard case: each thread gets 1 cell
        if (global_i < n):
            var tmp = x[global_i * incx]
            x[global_i * incx] = y[global_i * incy]
            y[global_i * incy] = tmp
    
    else:
        # Multiple cells per thread
        for i in range(global_i, n, n_threads):
            var tmp = x[i * incx]
            x[i * incx] = y[i * incy]
            y[i * incy] = tmp
