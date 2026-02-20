fn blas_rotmg[dtype: DType](
    d1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d2: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    y1: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param: UnsafePointer[SIMD[dtype, 5], MutAnyOrigin]
):

    var flag: Scalar[dtype]
    var h11: Scalar[dtype] = 0
    var h12: Scalar[dtype] = 0
    var h21: Scalar[dtype] = 0
    var h22: Scalar[dtype] = 0
    var p1: Scalar[dtype]
    var p2: Scalar[dtype]
    var q1: Scalar[dtype]
    var q2: Scalar[dtype]
    var temp: Scalar[dtype]
    var u: Scalar[dtype]
    var gam: Scalar[dtype] = 4096.0
    var gamsq: Scalar[dtype] = 16777216.0
    var rgamsq: Scalar[dtype] = 5.9604645e-8

    if (d1[] < 0):
        # GO 0-H-D-AND-x1..
        flag = -1
        h11 = 0
        h12 = 0
        h21 = 0
        h22 = 0

        d1[] = 0
        d2[] = 0
        x1[] = 0
    else:
        # CASE-d1-NONNEGATIVE
        p2 = d2[]*y1[]
        if (p2 == 0):
            flag = -2
            param[0] = flag
            return

        # REGULAR-CASE..
        p1 = d1[]*x1[]
        q2 = p2*y1[]
        q1 = p1*x1[]
        #
        if (abs(q1) > abs(q2)):
            h21 = -y1[]/x1[]
            h12 = p2/p1
            #
            u = 1 - h12*h21
            #
            if (u > 0):
                flag = 0
                d1[] /= u
                d2[] /= u
                x1[] *= u
            else:
                # This code path if here for safety. We do not expect this
                # condition to ever hold except in edge cases with rounding
                # errors. See DOI: 10.1145/355841.355847
                flag = -1
                h11 = 0
                h12 = 0
                h21 = 0
                h22 = 0
                #
                d1[] = 0
                d2[] = 0
                x1[] = 0
        else:
            if (q2 < 0):
                # GO 0-H-D-AND-x1..
                flag = -1
                h11 = 0
                h12 = 0
                h21 = 0
                h22 = 0
                #
                d1[] = 0
                d2[] = 0
                x1[] = 0
            else:
                flag = 1
                h11 = p1/p2
                h22 = x1[]/y1[]
                u = 1 + h11*h22
                temp = d2[]/u
                d2[] = d1[]/u
                d1[] = temp
                x1[] = y1[]*u

        # PROCEDURE..SCALE-CHECK
        if (d1[] != 0):
            while ((d1[] <= rgamsq)  or  (d1[] >= gamsq)):
                if (flag == 0):
                    h11 = 1
                    h22 = 1
                    flag = -1
                else:
                    h21 = -1
                    h12 = 1
                    flag = -1

                if (d1[] <= rgamsq):
                    d1[] *= gam**2
                    x1[] /= gam
                    h11 /= gam
                    h12 /= gam
                else:
                    d1[] /= gam**2
                    x1[] *= gam
                    h11 *= gam
                    h12 *= gam

        if (d2[] != 0):
            while ( (abs(d2[]) <= rgamsq) or (abs(d2[]) >= gamsq) ):
                if (flag == 0):
                    h11 = 1
                    h22 = 1
                    flag = -1
                else:
                    h21 = -1
                    h12 = 1
                    flag = -1
               
                if (abs(d2[]) <= rgamsq):
                    d2[] *= gam**2
                    h21 /= gam
                    h22 /= gam
                else:
                    d2[] *= gam**2
                    h21 *= gam
                    h22 *= gam

    if (flag < 0):
        param[1] = h11
        param[2] = h21
        param[3] = h12
        param[4] = h22
    elif (flag == 0):
        param[2] = h21
        param[3] = h12
    else:
        param[1] = h11
        param[4] = h22
      
    param[0] = flag
    return
