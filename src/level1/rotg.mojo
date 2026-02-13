from math import abs, sqrt, copysign


fn blas_rotg[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    s: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):
    a_value = a[0]
    b_value = b[0]

    if a_value == 0.0 and b_value == 0.0:
        c[0] = 1.0
        s[0] = 0.0
        return

    roe = b_value if abs(b_value) > abs(a_value) else a_value
    scale = abs(a_value) + abs(b_value)

    if scale == 0.0:
        c[0] = 1.0
        s[0] = 0.0
        return

    var r_value = scale * sqrt((a_value / scale)**2 + (b_value / scale)**2)
    r_value = copysign(r_value, roe)

    c_value = a_value / r_value
    s_value = b_value / r_value

    var z_value: Scalar[dtype]
    if abs(a_value) > abs(b_value):
        z_value = s_value
    elif c_value != 0.0:
        z_value = 1.0 / c_value
    else:
        z_value = 1.0

    a[0] = r_value
    b[0] = z_value
    c[0] = c_value
    s[0] = s_value
