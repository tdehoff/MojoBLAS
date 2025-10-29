# level1.iamax
# finds the index of the first element having maximum absolute value
fn iamax(n: Int, sx: UnsafePointer[Float32], incx: Int) -> Int:
    var result = -1
    if n < 1 or incx <= 0:
        return result

    result = 0
    if n == 1:
        return result

    var smax: Float32

    if incx == 1:
        # code for increment equal to 1
        smax = abs(sx[0])
        for i in range(1, n):
            if abs(sx[i]) > smax:
                result = i
                smax = abs(sx[i])
    else:
        ix = 1
        smax = abs(sx[0])
        ix += incx
        for i in range(1, n):
            if abs(sx[ix]) > smax:
                result = i
                smax = abs(sx[ix])
            ix += incx

    return result
