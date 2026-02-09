from gpu import grid_dim, block_dim, global_idx

fn scal_device[dtype: DType](
    n: Int, 
    a: Scalar[dtype],
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
):
    if (n <= 0):
        return
    if (a == 0):
        return
    if (incx == 0):
        return

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    if (n <= n_threads):
        # Standard case: each thread gets 1 cell
        if (global_i < n):
            x[global_i*incx] *= a
    
    else:
        # Multiple cells per thread
        for i in range(global_i, n, n_threads):
            x[i*incx] *= a

