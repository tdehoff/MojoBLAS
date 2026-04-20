from sys import argv
from testing import assert_equal, assert_almost_equal, assert_true, TestSuite
from gpu.host import DeviceContext

from src import *
from python import Python, PythonObject

comptime atol = 1.0E-4


def gemv_test[
    dtype: DType,
    m: Int,
    n: Int,
    trans: Bool,
]():
    # x_len and y_len depend on transpose:
    comptime x_len = n if not trans else m
    comptime y_len = m if not trans else n

    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](m * n)
        A = ctx.enqueue_create_host_buffer[dtype](m * n)
        x_d = ctx.enqueue_create_buffer[dtype](x_len)
        x = ctx.enqueue_create_host_buffer[dtype](x_len)
        y_d = ctx.enqueue_create_buffer[dtype](y_len)
        y = ctx.enqueue_create_host_buffer[dtype](y_len)

        generate_random_arr[dtype](m * n, A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](x_len, x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](y_len, y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta = generate_random_scalar[dtype](-100, 100)

        # Compute norms for error checks
        var norm_A = frobenius_norm[dtype](A.unsafe_ptr(), m * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), x_len)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), y_len)

        blas_gemv[dtype](
            trans,
            m, n,
            alpha,
            A_d.unsafe_ptr(), n,
            x_d.unsafe_ptr(), 1,
            beta,
            y_d.unsafe_ptr(), 1,
            ctx,
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()
        for i in range(m * n):
            py_A.append(A[i])
        for i in range(x_len):
            py_x.append(x[i])
        for i in range(y_len):
            py_y.append(y[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.sgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)
        else:
            print("Unsupported type: ", dtype)
            return

        # Referred to BLAS++ for an alternative error computation
        # https://github.com/icl-utk-edu/blaspp/blob/master/test/check_gemm.hh
        # NOTE: might use this for dot, gemv, ger, geru, gemm, symv, hemv, symm, trmv, trsv?, trmm, trsm?
        with y_d.map_to_host() as res_mojo:
            # Compute norm of (y - y_ref) vector
            var norm_diff = Scalar[dtype](0)
            for i in range(y_len):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)
            # From BLAS++: treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
            var ok = check_gemm_error[dtype](1, y_len, x_len, alpha, beta, norm_A, norm_x, norm_y, norm_diff)
            assert_true(ok)


def ger_test[
    dtype: DType,
    m:  Int,
    n: Int,
]():
    with DeviceContext() as ctx:
        A_device = ctx.enqueue_create_buffer[dtype](m*n)
        A = ctx.enqueue_create_host_buffer[dtype](m*n)
        x_device = ctx.enqueue_create_buffer[dtype](m)
        x = ctx.enqueue_create_host_buffer[dtype](m)
        y_device = ctx.enqueue_create_buffer[dtype](n)
        y = ctx.enqueue_create_host_buffer[dtype](n)

        # Generate three arrays of random numbers on CPU
        generate_random_arr[dtype](m * n, A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](m, x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_device, A)
        ctx.enqueue_copy(x_device, x)
        ctx.enqueue_copy(y_device, y)

        var alpha = generate_random_scalar[dtype](0.0, 1.0)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move a and b to a SciPy-compatible array and run SciPy BLAS routine
        py_a = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(m*n):
            py_a.append(A[i])
        for i in range(m):
            py_x.append(x[i])
        for i in range(n):
            py_y.append(y[i])

        var sp_res: PythonObject
        # ger - float32
        if dtype == DType.float32:
            np_a = np.array(py_a, dtype=np.float32).reshape(m,n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.sger(alpha, np_x, np_y, 1, 1, np_a)
        elif dtype == DType.float64:
            np_a = np.array(py_a, dtype=np.float64).reshape(m,n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dger(alpha, np_x, np_y, 1, 1, np_a)
        else:
            print("Unsupported type: ", dtype)
            return

        blas_ger[dtype](
            m,
            n,
            Scalar[dtype](alpha),
            x_device.unsafe_ptr(), 1,
            y_device.unsafe_ptr(), 1,
            A_device.unsafe_ptr(), n,
            ctx)

        with A_device.map_to_host() as res_mojo:
            for i in range(m):
                for j in range(n):
                    assert_almost_equal(Scalar[dtype](py=sp_res[i][j]), res_mojo[(i*n)+j], atol=atol)

def syr_test[
    dtype: DType,
    n: Int,
    uplo: Int,
]():
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](n * n)
        A = ctx.enqueue_create_host_buffer[dtype](n * n)
        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, x.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)

        blas_syr[dtype](uplo, n, alpha, x_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        for i in range(n * n):
            py_A.append(A[i])
        for i in range(n):
            py_x.append(x[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            sp_res = sp_blas.ssyr(alpha, np_x, lower=uplo, a=np_A, overwrite_a=False)
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            sp_res = sp_blas.dsyr(alpha, np_x, lower=uplo, a=np_A, overwrite_a=False)
        else:
            print("Unsupported type: ", dtype)
            return

        # NOTE: Error('only 0-dimensional arrays can be converted to Python scalars')
        sp_flat = sp_res.flatten()
        with A_d.map_to_host() as res_mojo:
            for i in range(n * n):
                assert_almost_equal(res_mojo[i], Scalar[dtype](py=sp_flat[i]), atol=atol)

def spr_test[
    dtype: DType,
    n: Int,
    uplo: Int,
]():
    comptime ap_len = n * (n + 1) // 2

    with DeviceContext() as ctx:
        AP_d = ctx.enqueue_create_buffer[dtype](ap_len)
        AP = ctx.enqueue_create_host_buffer[dtype](ap_len)
        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype](ap_len, AP.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, x.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(AP_d, AP)
        ctx.enqueue_copy(x_d, x)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)

        blas_spr[dtype](uplo, n, alpha, x_d.unsafe_ptr(), 1, AP_d.unsafe_ptr(), ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_AP = Python.list()
        py_x = Python.list()
        for i in range(ap_len):
            py_AP.append(AP[i])
        for i in range(n):
            py_x.append(x[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_AP = np.array(py_AP, dtype=np.float32)
            np_x = np.array(py_x, dtype=np.float32)
            sp_res = sp_blas.sspr(n, alpha, np_x, lower=uplo, ap=np_AP, overwrite_ap=False)
        elif dtype == DType.float64:
            np_AP = np.array(py_AP, dtype=np.float64)
            np_x = np.array(py_x, dtype=np.float64)
            sp_res = sp_blas.dspr(n, alpha, np_x, lower=uplo, ap=np_AP, overwrite_ap=False)
        else:
            print("Unsupported type: ", dtype)
            return

        with AP_d.map_to_host() as res_mojo:
            for i in range(ap_len):
                assert_almost_equal(res_mojo[i], Scalar[dtype](py=sp_res[i]), atol=atol)

def syr2_test[
    dtype: DType,
    n: Int,
    uplo: Int,
]():
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](n * n)
        A = ctx.enqueue_create_host_buffer[dtype](n * n)

        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        y_d = ctx.enqueue_create_buffer[dtype](n)
        y = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm_symmetric[dtype](A.unsafe_ptr(), n, n, uplo)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), n)

        blas_syr2[dtype](
            uplo,
            n,
            alpha,
            x_d.unsafe_ptr(), 1,
            y_d.unsafe_ptr(), 1,
            A_d.unsafe_ptr(),
            n,
            ctx,
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(n * n):
            py_A.append(A[i])
        for i in range(n):
            py_x.append(x[i])
            py_y.append(y[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.ssyr2(alpha, np_x, np_y, lower=uplo, a=np_A, overwrite_a=False)

        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dsyr2(alpha, np_x, np_y, lower=uplo, a=np_A, overwrite_a=False)
        else:
            print("Unsupported type: ", dtype)
            return

        sp_flat = sp_res.flatten()

        with A_d.map_to_host() as res_mojo:
            var error = InlineArray[Scalar[dtype], n*n](fill=Scalar[dtype](0))
            for i in range(n * n):
                error[i] = res_mojo[i] - Scalar[dtype](py=sp_flat[i])

            var error_norm = frobenius_norm_symmetric[dtype](
                error.unsafe_ptr(),
                n,
                n,
                uplo
            )

            var passed = check_syr_error[dtype](
                n,
                alpha,
                norm_x,
                norm_y,
                norm_A,
                error_norm
            )

            assert_true(passed)

def gbmv_test[
    dtype: DType,
    m: Int,
    n: Int,
    kl: Int,
    ku: Int,
    trans: Bool,
]():
    comptime x_len = n if not trans else m
    comptime y_len = m if not trans else n
    comptime lda = kl + ku + 1

    with DeviceContext() as ctx:
        # Dense host matrix (for reference + band extraction)
        A_dense = ctx.enqueue_create_host_buffer[dtype](m * n)

        # Band storage
        A_band = ctx.enqueue_create_host_buffer[dtype](lda * m)
        A_d = ctx.enqueue_create_buffer[dtype](lda * m)

        x = ctx.enqueue_create_host_buffer[dtype](x_len)
        x_d = ctx.enqueue_create_buffer[dtype](x_len)

        y = ctx.enqueue_create_host_buffer[dtype](y_len)
        y_d = ctx.enqueue_create_buffer[dtype](y_len)

        generate_random_arr[dtype](m * n, A_dense.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype](x_len, x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](y_len, y.unsafe_ptr(), -100, 100)

        dense_to_band(A_dense.unsafe_ptr(), A_band.unsafe_ptr(), m, n, kl, ku)

        ctx.enqueue_copy(A_d, A_band)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta  = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm[dtype](A_dense.unsafe_ptr(), m * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), x_len)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), y_len)

        blas_gbmv[dtype](
            trans,
            m, n,
            kl, ku,
            alpha,
            A_d.unsafe_ptr(), lda,
            x_d.unsafe_ptr(), 1,
            beta,
            y_d.unsafe_ptr(), 1,
            ctx,
        )

        # Python reference
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(m * n):
            py_A.append(A_dense[i])
        for i in range(x_len):
            py_x.append(x[i])
        for i in range(y_len):
            py_y.append(y[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)

            sp_res = sp_blas.sgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)

        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)

            sp_res = sp_blas.dgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)
        else:
            print("Unsupported type")
            return

        with y_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)
            for i in range(y_len):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])

                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)

            var ok = check_gemm_error[dtype](
                1, y_len, x_len,
                alpha, beta,
                norm_A, norm_x, norm_y,
                norm_diff
            )
            assert_true(ok)

def sbmv_test[
    dtype: DType,
    n: Int,
    k: Int,
    upper: Bool,
]():
    comptime lda = k + 1

    with DeviceContext() as ctx:
        # Dense symmetric host matrix (reference)
        A_dense = ctx.enqueue_create_host_buffer[dtype](n * n)

        # Band storage (symmetric)
        A_band_rm = ctx.enqueue_create_host_buffer[dtype](lda * n)
        A_band_cm = ctx.enqueue_create_host_buffer[dtype](n * lda)
        A_d = ctx.enqueue_create_buffer[dtype](lda * n)

        x = ctx.enqueue_create_host_buffer[dtype](n)
        x_d = ctx.enqueue_create_buffer[dtype](n)

        y = ctx.enqueue_create_host_buffer[dtype](n)
        y_d = ctx.enqueue_create_buffer[dtype](n)

        # --- Generate symmetric dense matrix ---
        generate_random_arr[dtype](n * n, A_dense.unsafe_ptr(), -1, 1)

        # Force symmetry: A = 0.5 * (A + A^T)
        for i in range(n):
            for j in range(i, n):
                var aij = A_dense[i * n + j]
                var aji = A_dense[j * n + i]
                var sym = (aij + aji) * Scalar[dtype](0.5)
                A_dense[i * n + j] = sym
                A_dense[j * n + i] = sym

        generate_random_arr[dtype](n, x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, y.unsafe_ptr(), -100, 100)

        # Convert dense -> symmetric row-major band for MojoBLAS routine
        dense_to_sym_band_rm(
            A_dense.unsafe_ptr(),
            A_band_rm.unsafe_ptr(),
            n,
            k,
            upper
        )

        # Convert dense -> symmetric col-major band for SciPy routine
        dense_to_sym_band_cm(
            A_dense.unsafe_ptr(),
            A_band_cm.unsafe_ptr(),
            n,
            k,
            upper
        )

        ctx.enqueue_copy(A_d, A_band_rm)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta  = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm[dtype](A_dense.unsafe_ptr(), n * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), n)

        blas_sbmv[dtype](
            1 if upper else 0,
            n,
            k,
            alpha,
            A_d.unsafe_ptr(), lda,
            x_d.unsafe_ptr(), 1,
            beta,
            y_d.unsafe_ptr(), 1,
            ctx,
        )

        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(n * lda):
            py_A.append(A_band_cm[i])
        for i in range(n):
            py_x.append(x[i])
            py_y.append(y[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            # order='F' is col-major, 'C' is row-major
            np_A = np.array(py_A, dtype=np.float32).reshape(lda, n, order='F')
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)

            sp_res = sp_blas.ssbmv(
                k,
                alpha,
                np_A,
                np_x,
                beta=beta,
                y=np_y,
                lower=0 if upper else 1
            )

        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(lda, n, order='F')
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)

            sp_res = sp_blas.dsbmv(
                k,
                alpha,
                np_A,
                np_x,
                beta=beta,
                y=np_y,
                lower=0 if upper else 1
            )
        else:
            print("Unsupported type")
            return

        with y_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)

            for i in range(n):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff

            norm_diff = sqrt(norm_diff)

            var ok = check_gemm_error[dtype](
                1, n, n,
                alpha, beta,
                norm_A, norm_x, norm_y,
                norm_diff
            )
            assert_true(ok)

def symv_test[
    dtype: DType,
    n: Int,
    upper: Bool,
]():
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](n * n)
        A = ctx.enqueue_create_host_buffer[dtype](n * n)
        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)
        y_d = ctx.enqueue_create_buffer[dtype](n)
        y = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A.unsafe_ptr(), -1, 1)

        # Force symmetry: A = 0.5 * (A + A^T)
        for i in range(n):
            for j in range(i, n):
                var sym = (A[i * n + j] + A[j * n + i]) * Scalar[dtype](0.5)
                A[i * n + j] = sym
                A[j * n + i] = sym

        generate_random_arr[dtype](n, x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](n, y.unsafe_ptr(), -100, 100)

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta  = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm[dtype](A.unsafe_ptr(), n * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), n)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        blas_symv[dtype](
            upper,
            n,
            alpha,
            A_d.unsafe_ptr(), n,
            x_d.unsafe_ptr(), 1,
            beta,
            y_d.unsafe_ptr(), 1,
            ctx,
        )

        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()
        for i in range(n * n):
            py_A.append(A[i])
        for i in range(n):
            py_x.append(x[i])
            py_y.append(y[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.ssymv(alpha, np_A, np_x, beta=beta, y=np_y, lower=0 if upper else 1)
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dsymv(alpha, np_A, np_x, beta=beta, y=np_y, lower=0 if upper else 1)
        else:
            print("Unsupported type: ", dtype)
            return

        with y_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)
            for i in range(n):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)
            var ok = check_gemm_error[dtype](1, n, n, alpha, beta, norm_A, norm_x, norm_y, norm_diff)
            assert_true(ok)

def trsv_test[
    dtype: DType,
    n: Int,
    trans: Bool,
    uplo: Int,
    diag: Int,
]():
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](n * n)
        A = ctx.enqueue_create_host_buffer[dtype](n * n)
        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype](n, x.unsafe_ptr(), -1, 1)

        # Zero out off-triangle elements to make A strictly triangular
        for i in range(n):
            for j in range(n):
                # upper triangular
                if uplo == 0:
                    if i > j:
                        A[i * n + j] = 0
                # lower triangular
                else:
                    if i < j:
                        A[i * n + j] = 0

        # Handle diagonal based on diag parameter
        for i in range(n):
            # unit diagonal
            if diag == 1:
                A[i * n + i] = 1
            # non-unit diagonal: make diagonally dominant to reduce numerical instability
            else:
                A[i * n + i] += 1000

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.synchronize()

        # Compute norms for error checks
        var norm_A = frobenius_norm[dtype](A.unsafe_ptr(), n * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)

        blas_trsv[dtype](
            uplo,
            trans,
            diag,
            n,
            A_d.unsafe_ptr(), n,
            x_d.unsafe_ptr(), 1,
            ctx,
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        for i in range(n * n):
            py_A.append(A[i])
        for i in range(n):
            py_x.append(x[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            sp_res = sp_blas.strsv(np_A, np_x,
                lower=0 if uplo else 1,
                trans=1 if trans else 0,
                diag=1 if diag else 0
            )
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            sp_res = sp_blas.dtrsv(np_A, np_x,
                lower=0 if uplo else 1,
                trans=1 if trans else 0,
                diag=1 if diag else 0
            )
        else:
            print("Unsupported type: ", dtype)
            return

        with x_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)
            for i in range(n):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)

            var ok = check_gemm_error[dtype](
                1, n, n,
                Scalar[dtype](1), Scalar[dtype](0),
                norm_A, norm_x, Scalar[dtype](0),
                norm_diff
            )
            assert_true(ok)

def tbmv_test[
    dtype: DType,
    n: Int,
    k: Int,
    uplo: Int,
    trans: Bool,
    diag: Int,
]():
    comptime lda = k + 1

    with DeviceContext() as ctx:
        A_dense = ctx.enqueue_create_host_buffer[dtype](n * n)
        A_band = ctx.enqueue_create_host_buffer[dtype](lda * n)
        A_d = ctx.enqueue_create_buffer[dtype](lda * n)
        x = ctx.enqueue_create_host_buffer[dtype](n)
        x_d = ctx.enqueue_create_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A_dense.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype](n, x.unsafe_ptr(), -100, 100)

        # Zero out elements outside the triangle and outside the band
        for i in range(n):
            for j in range(n):
                if uplo:  # upper triangular
                    if j < i or j > i + k:
                        A_dense[i * n + j] = 0
                else:  # lower triangular
                    if j > i or j < i - k:
                        A_dense[i * n + j] = 0
            # Set unit diagonal to 1 so the numpy reference is consistent
            if diag:
                A_dense[i * n + i] = 1

        # Pack dense triangular band into row-major band storage
        dense_to_tri_band_rm(A_dense.unsafe_ptr(), A_band.unsafe_ptr(), n, k, uplo)

        ctx.enqueue_copy(A_d, A_band)
        ctx.enqueue_copy(x_d, x)
        ctx.synchronize()

        var norm_A = frobenius_norm[dtype](A_dense.unsafe_ptr(), n * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)

        blas_tbmv[dtype](
            uplo, trans, diag,
            n, k,
            A_d.unsafe_ptr(), lda,
            x_d.unsafe_ptr(), 1,
            UnsafePointer[Scalar[dtype], MutAnyOrigin](), # Emtpy temp (forces auto allocation)
            ctx,
        )

        np = Python.import_module("numpy")

        py_A = Python.list()
        py_x = Python.list()
        for i in range(n * n):
            py_A.append(A_dense[i])
        for i in range(n):
            py_x.append(x[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            sp_res = np.dot(np_A.T, np_x) if trans else np.dot(np_A, np_x)
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            sp_res = np.dot(np_A.T, np_x) if trans else np.dot(np_A, np_x)
        else:
            print("Unsupported type: ", dtype)
            return

        with x_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)
            for i in range(n):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)

            var ok = check_gemm_error[dtype](
                1, n, n,
                Scalar[dtype](1), Scalar[dtype](0),
                norm_A, norm_x, Scalar[dtype](0),
                norm_diff
            )
            assert_true(ok)

def tbsv_test[
    dtype: DType,
    n: Int,
    k: Int,
    uplo: Int,
    trans: Bool,
    diag: Int,
]():
    comptime lda = k + 1

    with DeviceContext() as ctx:
        A_dense = ctx.enqueue_create_host_buffer[dtype](n * n)
        A_band = ctx.enqueue_create_host_buffer[dtype](lda * n)
        A_d = ctx.enqueue_create_buffer[dtype](lda * n)
        b = ctx.enqueue_create_host_buffer[dtype](n)
        x_d = ctx.enqueue_create_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A_dense.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype](n, b.unsafe_ptr(), -1, 1)

        # Zero out elements outside the triangle and band; ensure diagonal
        # dominance for a well-conditioned system
        for i in range(n):
            for j in range(n):
                if uplo:  # upper triangular
                    if j < i or j > i + k:
                        A_dense[i * n + j] = 0
                else:  # lower triangular
                    if j > i or j < i - k:
                        A_dense[i * n + j] = 0
            if diag:
                A_dense[i * n + i] = 1
            else:
                # Make diagonally dominant to keep the system well-conditioned
                A_dense[i * n + i] += 1000

        # Pack dense triangular band into row-major band storage
        dense_to_tri_band_rm(A_dense.unsafe_ptr(), A_band.unsafe_ptr(), n, k, uplo)

        ctx.enqueue_copy(A_d, A_band)
        ctx.enqueue_copy(x_d, b)
        ctx.synchronize()

        var norm_A = frobenius_norm[dtype](A_dense.unsafe_ptr(), n * n)
        var norm_b = frobenius_norm[dtype](b.unsafe_ptr(), n)

        blas_tbsv[dtype](
            uplo, trans, diag,
            n, k,
            A_d.unsafe_ptr(), lda,
            x_d.unsafe_ptr(), 1,
            ctx,
        )

        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")

        py_A = Python.list()
        py_b = Python.list()
        for i in range(n * n):
            py_A.append(A_dense[i])
        for i in range(n):
            py_b.append(b[i])

        var is_lower = 0 if uplo else 1
        var trans_mode = 1 if trans else 0
        var unit_diag = True if diag else False

        var np_A: PythonObject
        var np_b: PythonObject
        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_b = np.array(py_b, dtype=np.float32)
            sp_res = sp.linalg.solve_triangular(
                np_A, np_b,
                lower=is_lower,
                trans=trans_mode,
                unit_diagonal=unit_diag,
            )
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_b = np.array(py_b, dtype=np.float64)
            sp_res = sp.linalg.solve_triangular(
                np_A, np_b,
                lower=is_lower,
                trans=trans_mode,
                unit_diagonal=unit_diag,
            )
        else:
            print("Unsupported type: ", dtype)
            return

        # Primary check: will pass for well-conditioned A
        with x_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)
            for i in range(n):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)

            var ok = check_gemm_error[dtype](
                1, n, n,
                Scalar[dtype](1), Scalar[dtype](0),
                norm_A, norm_b, Scalar[dtype](0),
                norm_diff
            )

            # A is ill-conditioned, perform backward error-check.
            # Compute x * A and check if it's close enough to b.
            if not ok:
                # Copy x into temp so we don't overwrite our solution
                temp_d = ctx.enqueue_create_buffer[dtype](n)
                ctx.enqueue_copy(temp_d, x_d)
                ctx.synchronize()

                # Compute temp = A * x with tbmv
                work = ctx.enqueue_create_buffer[dtype](n)
                blas_tbmv[dtype](
                    uplo, trans, diag,
                    n, k,
                    A_d.unsafe_ptr(), lda,
                    temp_d.unsafe_ptr(), 1,
                    work.unsafe_ptr(),
                    ctx
                )

                with temp_d.map_to_host() as ax:
                    # Compute residual r = A*x - b, and norm of x for scaling
                    var norm_r = Scalar[dtype](0)
                    var norm_x = Scalar[dtype](0)
                    for i in range(n):
                        var r_i = ax[i] - b[i]
                        norm_r += r_i * r_i
                        norm_x += res_mojo[i] * res_mojo[i]
                    norm_r = sqrt(norm_r)
                    norm_x = sqrt(norm_x)

                    # Check residual
                    ok = check_gemm_error[dtype](
                        1, n, n,
                        Scalar[dtype](1), Scalar[dtype](0),
                        norm_A, norm_x, Scalar[dtype](0),
                        norm_r,
                    )

            assert_true(ok)


def tpmv_test[
    dtype: DType,
    n: Int,
    trans: Bool,
    uplo: Int,
    diag: Int,
]():
    with DeviceContext() as ctx:
        A_dense = ctx.enqueue_create_host_buffer[dtype](n * n)

        AP_d = ctx.enqueue_create_buffer[dtype](n * n)
        AP = ctx.enqueue_create_host_buffer[dtype](n * n)
        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype](n * n, A_dense.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype](n, x.unsafe_ptr(), -1, 1)

        # Zero out off-triangle elements to make A strictly triangular
        for i in range(n):
            for j in range(n):
                # upper triangular
                if uplo == 0:
                    if i > j:
                        A_dense[i * n + j] = 0
                # lower triangular
                else:
                    if i < j:
                        A_dense[i * n + j] = 0

        # Handle diagonal based on diag parameter
        for i in range(n):
            # unit diagonal
            if diag == 1:
                A_dense[i * n + i] = 1
            # non-unit diagonal: make diagonally dominant to reduce numerical instability
            else:
                A_dense[i * n + i] += 1000

        # Pack into column-major packed storage
        dense_to_tri_packed(A_dense.unsafe_ptr(), AP.unsafe_ptr(), n, uplo)

        ctx.enqueue_copy(AP_d, AP)
        ctx.enqueue_copy(x_d, x)
        ctx.synchronize()

        # Compute norms for error checks
        var norm_A = frobenius_norm[dtype](A_dense.unsafe_ptr(), n * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)

        blas_tpmv[dtype](
            uplo,
            trans,
            diag,
            n,
            AP_d.unsafe_ptr(),
            x_d.unsafe_ptr(), 1,
            ctx,
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        for i in range(n * n):
            py_A.append(AP[i])
        for i in range(n):
            py_x.append(x[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32)
            np_x = np.array(py_x, dtype=np.float32)
            sp_res = sp_blas.stpmv(n, np_A, np_x,
                lower=0 if uplo else 1,
                trans=1 if trans else 0,
                diag=1 if diag else 0
            )
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64)
            np_x = np.array(py_x, dtype=np.float64)
            sp_res = sp_blas.dtpmv(n, np_A, np_x,
                lower=0 if uplo else 1,
                trans=1 if trans else 0,
                diag=1 if diag else 0
            )
        else:
            print("Unsupported type: ", dtype)
            return

        with x_d.map_to_host() as res_mojo:

            var norm_diff = Scalar[dtype](0)
            for i in range(n):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                print(res_mojo[i], Scalar[dtype](py=sp_res[i]))
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)
            print(norm_diff)

            var ok = check_gemm_error[dtype](
                1, n, n,
                Scalar[dtype](1), Scalar[dtype](0),
                norm_A, norm_x, Scalar[dtype](0),
                norm_diff
            )
            assert_true(ok)


def test_gemv():
    gemv_test[DType.float32,  64,  64, False]()
    gemv_test[DType.float32,  64,  64, True]()
    gemv_test[DType.float64,  64,  64, False]()
    gemv_test[DType.float64,  64,  64, True]()
    gemv_test[DType.float32, 1024,  64, False]()
    gemv_test[DType.float32, 1024,  64, True]()
    gemv_test[DType.float64, 1024,  64, False]()
    gemv_test[DType.float64, 1024,  64, True]()

def test_ger():
    ger_test[DType.float32, 64, 64]()
    ger_test[DType.float32, 256, 256]()
    ger_test[DType.float64, 64, 64]()
    ger_test[DType.float64, 256, 256]()

def test_spr():
    spr_test[DType.float32,  256, 1]()
    spr_test[DType.float32, 1024, 0]()
    spr_test[DType.float64,  256, 0]()
    spr_test[DType.float64, 1024, 1]()

def test_syr():
    syr_test[DType.float32,  256, 1]()
    syr_test[DType.float32, 1024, 0]()
    syr_test[DType.float64,  256, 0]()
    syr_test[DType.float64, 1024, 1]()

def test_syr2():
    syr2_test[DType.float32,  512, 1]()
    syr2_test[DType.float32, 512, 0]()
    syr2_test[DType.float64,  512, 0]()
    syr2_test[DType.float64, 512, 1]()

def test_gbmv():
    gbmv_test[DType.float32,  64,  64, 1, 2, False]()
    gbmv_test[DType.float32,  64,  64, 2, 2, True]()
    gbmv_test[DType.float64,  64,  64, 1, 2, False]()
    gbmv_test[DType.float64,  64,  64, 2, 2, True]()
    gbmv_test[DType.float32, 512,  64, 1, 2, False]()
    gbmv_test[DType.float32, 512,  64, 2, 2, True]()
    gbmv_test[DType.float64, 512,  64, 1, 2, False]()
    gbmv_test[DType.float64, 512,  64, 2, 2, True]()

def test_sbmv():
    sbmv_test[DType.float32,  64,  1, False]()
    sbmv_test[DType.float32,  64,  2, True]()
    sbmv_test[DType.float32,  64,  16, False]()
    sbmv_test[DType.float32,  64,  16, True]()
    sbmv_test[DType.float64,  64,  1, False]()
    sbmv_test[DType.float64,  64,  2, True]()
    sbmv_test[DType.float64,  64,  16, False]()
    sbmv_test[DType.float64,  64,  16, True]()
    sbmv_test[DType.float32, 512,  1, False]()
    sbmv_test[DType.float32, 512,  2, True]()
    sbmv_test[DType.float32, 512,  32, False]()
    sbmv_test[DType.float32, 512,  32, True]()
    sbmv_test[DType.float64, 512,  1, False]()
    sbmv_test[DType.float64, 512,  2, True]()
    sbmv_test[DType.float64, 512,  32, False]()
    sbmv_test[DType.float64, 512,  32, True]()

def test_symv():
    symv_test[DType.float32,   64, True]()
    symv_test[DType.float32,   64, False]()
    symv_test[DType.float64,   64, True]()
    symv_test[DType.float64,   64, False]()
    symv_test[DType.float32, 1024, True]()
    symv_test[DType.float32, 1024, False]()
    symv_test[DType.float64, 1024, True]()
    symv_test[DType.float64, 1024, False]()

def test_trsv():
    trsv_test[DType.float32,  64,  True, 0, 0]()
    trsv_test[DType.float32,  64,  True, 1, 0]()
    trsv_test[DType.float32,  64,  True, 0, 1]()
    trsv_test[DType.float32,  64,  True, 1, 1]()
    trsv_test[DType.float32,  64,  False, 0, 0]()
    trsv_test[DType.float32,  64,  False, 1, 0]()
    trsv_test[DType.float32,  64,  False, 0, 1]()
    trsv_test[DType.float32,  64,  False, 1, 1]()
    trsv_test[DType.float64,  64,  True, 0, 0]()
    trsv_test[DType.float64,  64,  True, 1, 0]()
    trsv_test[DType.float64,  64,  True, 0, 1]()
    trsv_test[DType.float64,  64,  True, 1, 1]()
    trsv_test[DType.float64,  64,  False, 0, 0]()
    trsv_test[DType.float64,  64,  False, 1, 0]()
    trsv_test[DType.float64,  64,  False, 0, 1]()
    trsv_test[DType.float64,  64,  False, 1, 1]()

def test_tbmv():
    # uplo: 0=lower, 1=upper   trans: False/True   diag: 0=non-unit, 1=unit
    tbmv_test[DType.float32,  64,  1, 1, False, 0]()
    tbmv_test[DType.float32,  64,  1, 0, False, 0]()
    tbmv_test[DType.float32,  64,  2, 1, True,  0]()
    tbmv_test[DType.float32,  64,  2, 0, True,  0]()
    tbmv_test[DType.float32,  64,  4, 1, False, 1]()
    tbmv_test[DType.float32,  64,  4, 0, False, 1]()
    tbmv_test[DType.float32,  64,  4, 1, True,  1]()
    tbmv_test[DType.float32,  64,  4, 0, True,  1]()
    tbmv_test[DType.float64,  64,  1, 1, False, 0]()
    tbmv_test[DType.float64,  64,  1, 0, False, 0]()
    tbmv_test[DType.float64,  64,  2, 1, True,  0]()
    tbmv_test[DType.float64,  64,  2, 0, True,  0]()
    tbmv_test[DType.float64,  64,  4, 1, False, 1]()
    tbmv_test[DType.float64,  64,  4, 0, False, 1]()
    tbmv_test[DType.float64,  64,  4, 1, True,  1]()
    tbmv_test[DType.float64,  64,  4, 0, True,  1]()
    tbmv_test[DType.float32, 512, 16, 1, False, 0]()
    tbmv_test[DType.float32, 512, 16, 0, False, 0]()
    tbmv_test[DType.float32, 512, 16, 1, True,  0]()
    tbmv_test[DType.float32, 512, 16, 0, True,  0]()
    tbmv_test[DType.float64, 512, 16, 1, False, 0]()
    tbmv_test[DType.float64, 512, 16, 0, False, 0]()
    tbmv_test[DType.float64, 512, 16, 1, True,  0]()
    tbmv_test[DType.float64, 512, 16, 0, True,  0]()

def test_tbsv():
    # uplo: 0=lower, 1=upper   trans: False/True   diag: 0=non-unit, 1=unit
    tbsv_test[DType.float32,  64,  1, 1, False, 0]()
    tbsv_test[DType.float32,  64,  1, 0, False, 0]()
    tbsv_test[DType.float32,  64,  2, 1, True,  0]()
    tbsv_test[DType.float32,  64,  2, 0, True,  0]()
    tbsv_test[DType.float32,  64,  4, 1, False, 1]()
    tbsv_test[DType.float32,  64,  4, 0, False, 1]()
    tbsv_test[DType.float32,  64,  4, 1, True,  1]()
    tbsv_test[DType.float32,  64,  4, 0, True,  1]()
    tbsv_test[DType.float64,  64,  1, 1, False, 0]()
    tbsv_test[DType.float64,  64,  1, 0, False, 0]()
    tbsv_test[DType.float64,  64,  2, 1, True,  0]()
    tbsv_test[DType.float64,  64,  2, 0, True,  0]()
    tbsv_test[DType.float64,  64,  4, 1, False, 1]()
    tbsv_test[DType.float64,  64,  4, 0, False, 1]()
    tbsv_test[DType.float64,  64,  4, 1, True,  1]()
    tbsv_test[DType.float64,  64,  4, 0, True,  1]()
    tbsv_test[DType.float32, 512, 16, 1, False, 0]()
    tbsv_test[DType.float32, 512, 16, 0, False, 0]()
    tbsv_test[DType.float32, 512, 16, 1, True,  0]()
    tbsv_test[DType.float32, 512, 16, 0, True,  0]()
    tbsv_test[DType.float64, 512, 16, 1, False, 0]()
    tbsv_test[DType.float64, 512, 16, 0, False, 0]()
    tbsv_test[DType.float64, 512, 16, 1, True,  0]()
    tbsv_test[DType.float64, 512, 16, 0, True,  0]()

def test_tpmv():
    tpmv_test[DType.float32,  64,  True,  0, 0]()
    tpmv_test[DType.float32,  64,  True,  1, 0]()
    tpmv_test[DType.float32,  64,  True,  0, 1]()
    tpmv_test[DType.float32,  64,  True,  1, 1]()
    tpmv_test[DType.float32,  64,  False, 0, 0]()
    tpmv_test[DType.float32,  64,  False, 1, 0]()
    tpmv_test[DType.float32,  64,  False, 0, 1]()
    tpmv_test[DType.float32,  64,  False, 1, 1]()
    tpmv_test[DType.float64,  64,  True,  0, 0]()
    tpmv_test[DType.float64,  64,  True,  1, 0]()
    tpmv_test[DType.float64,  64,  True,  0, 1]()
    tpmv_test[DType.float64,  64,  True,  1, 1]()
    tpmv_test[DType.float64,  64,  False, 0, 0]()
    tpmv_test[DType.float64,  64,  False, 1, 0]()
    tpmv_test[DType.float64,  64,  False, 0, 1]()
    tpmv_test[DType.float64,  64,  False, 1, 1]()

def main():
    print("--- MojoBLAS Level 2 routines testing ---")
    var args = argv()
    if (len(args) < 2):
        TestSuite.discover_tests[__functions_in_module()]().run()
        return

    var suite = TestSuite(cli_args=List[StaticString]())
    for i in range(1, len(args)):
        if   args[i] == "gemv":  suite.test[test_gemv]()
        elif args[i] == "ger":   suite.test[test_ger]()
        elif args[i] == "sbmv":  suite.test[test_sbmv]()
        elif args[i] == "spr":   suite.test[test_spr]()
        elif args[i] == "syr":   suite.test[test_syr]()
        elif args[i] == "syr2":  suite.test[test_syr2]()
        elif args[i] == "gbmv":  suite.test[test_gbmv]()
        elif args[i] == "trsv":  suite.test[test_trsv]()
        elif args[i] == "tbmv":  suite.test[test_tbmv]()
        elif args[i] == "tbsv":  suite.test[test_tbsv]()
        elif args[i] == "tpmv":  suite.test[test_tpmv]()
        elif args[i] == "symv":  suite.test[test_symv]()
        else: print("unknown routine:", args[i])
    suite^.run()

