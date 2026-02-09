from gpu import grid_dim, block_dim, global_idx

fn copy_device[dtype: DType](
    n: Int,
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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

    # Multiple cells per thread
    for i in range(global_i, n, n_threads):
        y[i*incy] = x[i*incx]
