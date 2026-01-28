from gpu import thread_idx

fn axpy_device[dtype: DType](
    n: Int, 
    sa: SIMD[dtype, 1],
    sx: UnsafePointer[SIMD[dtype, 1], ImmutAnyOrigin],
    incx: Int,
    sy: UnsafePointer[SIMD[dtype, 1], MutAnyOrigin],
    incy: Int
):
    var thread_id: UInt = thread_idx.x

    if (n <= 0):
        return
    if (sa == 0):
        return
    if (incx == 0 or incy == 0):
        return

    if (thread_id < n):
        sy[thread_id*incy] += sa*sx[thread_id*incx]
