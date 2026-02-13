from testing import assert_equal, assert_almost_equal, TestSuite
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, grid_dim, thread_idx
from layout import Layout, LayoutTensor
from math import sqrt
from complex import ComplexSIMD

from src import *
from random import rand, seed, random_float64
from math import ceildiv, sin, cos
from python import Python, PythonObject

comptime TBsize = 512
comptime atol = 1.0E-6

def generate_random_arr[
    dtype: DType,
    size:  Int
](
    a:   UnsafePointer[Scalar[dtype], MutAnyOrigin],
    min_value: Scalar[dtype],
    max_value: Scalar[dtype]
):
    # Generate random values in [0, 1]
    seed()
    rand[dtype](a, size)

    # Scale to [min, max]
    var rng = max_value - min_value
    for i in range(size):
        a[i] = min_value + a[i] * rng


def asum_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ asum test:", dtype, "]")

        d_v = ctx.enqueue_create_buffer[dtype](size)
        v = ctx.enqueue_create_host_buffer[dtype](size)
        generate_random_arr[dtype, size](v.unsafe_ptr(), -10000, 10000)
        ctx.enqueue_copy(d_v, v)

        d_res = ctx.enqueue_create_buffer[dtype](1)
        d_res.enqueue_fill(Scalar[dtype](-1))

        # Launch Mojo BLAS kernel
        blas_asum[dtype](size, d_v.unsafe_ptr(), 1, d_res.unsafe_ptr(), ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move values in v to a SciPy-compatible array and run SciPy BLAS routine
        py_list = Python.list()
        for i in range(size):
            py_list.append(v[i])
        var sp_res: PythonObject
        # sasum - float32, dasum - float64
        if dtype == DType.float32:
            np_v = np.array(py_list, dtype=np.float32)
            sp_res = sp_blas.sasum(np_v)
        elif dtype == DType.float64:
            np_v = np.array(py_list, dtype=np.float64)
            sp_res = sp_blas.dasum(np_v)
        else:
            print(dtype , " is not supported by SciPy")
            return

        sp_res_mojo = Scalar[dtype](py=sp_res)
        with d_res.map_to_host() as res_mojo:
            assert_almost_equal(sp_res_mojo, res_mojo[0], atol=atol)


def axpy_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ axpy test:", dtype, "]")

        a: SIMD[dtype, 1] = 0
        x = ctx.enqueue_create_host_buffer[dtype](size)
        y = ctx.enqueue_create_host_buffer[dtype](size)
        mojo_res = ctx.enqueue_create_host_buffer[dtype](size)

        generate_random_arr[dtype, 1](UnsafePointer[SIMD[dtype, 1]](to=a), -10000, 10000)
        generate_random_arr[dtype, size](x.unsafe_ptr(), -10000, 10000)
        generate_random_arr[dtype, size](y.unsafe_ptr(), -10000, 10000)
        # print("a = ", a)
        # print("x = ", x)
        # print("y = ", y)

        d_x = ctx.enqueue_create_buffer[dtype](size)
        d_y = ctx.enqueue_create_buffer[dtype](size)

        ctx.enqueue_copy(d_x, x)
        ctx.enqueue_copy(d_y, y)

        blas_axpy[dtype](size, a, d_x.unsafe_ptr(), 1, d_y.unsafe_ptr(), 1, ctx)

        sp_blas = Python.import_module("scipy.linalg.blas")
        builtins = Python.import_module("builtins")

        x_py = Python.list()
        y_py = Python.list()
        for i in range(size):
            x_py.append(x[i])
            y_py.append(y[i])

        if dtype == DType.float32:
            sp_result = sp_blas.saxpy(x_py, y_py, a=a)
        elif dtype == DType.float64:
            sp_result = sp_blas.daxpy(x_py, y_py, a=a)
        else:
            print(dtype , " is not supported by SciPy")
            return

        ctx.enqueue_copy(mojo_res, d_y)
        ctx.synchronize()

        # Prints too much for large vectors. May want to add a verbose option
        # print("out:", mojo_res)
        # print("expected", sp_result)

        for i in range(size):
            assert_almost_equal(Scalar[dtype](py=sp_result[i]), mojo_res[i], atol=atol)


def copy_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ copy test:", dtype, "]")

        x = ctx.enqueue_create_host_buffer[dtype](size)
        y = ctx.enqueue_create_host_buffer[dtype](size)

        generate_random_arr[dtype, size](x.unsafe_ptr(), -10000, 10000)
        generate_random_arr[dtype, size](y.unsafe_ptr(), -10000, 10000)
        # print("x = ", x)
        # print("y = ", y)

        d_x = ctx.enqueue_create_buffer[dtype](size)
        d_y = ctx.enqueue_create_buffer[dtype](size)

        ctx.enqueue_copy(d_x, x)
        ctx.enqueue_copy(d_y, y)

        blas_copy[dtype](size, d_x.unsafe_ptr(), 1, d_y.unsafe_ptr(), 1, ctx)

        ctx.enqueue_copy(y, d_y)
        ctx.synchronize()

        # Prints too much for large vectors. May want to add a verbose option
        # print("out:", mojo_res)
        # print("expected", sp_result)

        for i in range(size):
            assert_equal(x[i], y[i])


def dot_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ dot test", dtype, "]")
        out = ctx.enqueue_create_buffer[dtype](1)
        out.enqueue_fill(0)
        a_device = ctx.enqueue_create_buffer[dtype](size)
        a = ctx.enqueue_create_host_buffer[dtype](size)
        b_device = ctx.enqueue_create_buffer[dtype](size)
        b = ctx.enqueue_create_host_buffer[dtype](size)

        # Generate two arrays of random numbers on CPU
        generate_random_arr[dtype, size](a.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, size](b.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(a_device, a)
        ctx.enqueue_copy(b_device, b)

        blas_dot[dtype](
            size,
            a_device.unsafe_ptr(), 1,
            b_device.unsafe_ptr(), 1,
            out.unsafe_ptr(),
            ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move a and b to a SciPy-compatible array and run SciPy BLAS routine
        py_a = Python.list()
        py_b = Python.list()
        for i in range(size):
            py_a.append(a[i])
            py_b.append(b[i])
        var sp_res: PythonObject
        # sdot - float32, ddot - float64
        if dtype == DType.float32:
            np_a = np.array(py_a, dtype=np.float32)
            np_b = np.array(py_b, dtype=np.float32)
            sp_res = sp_blas.sdot(np_a, np_b)
        elif dtype == DType.float64:
            np_a = np.array(py_a, dtype=np.float64)
            np_b = np.array(py_b, dtype=np.float64)
            sp_res = sp_blas.ddot(np_a, np_b)
        else:
            print(dtype , " is not supported by SciPy")
            return

        sp_res_mojo = Scalar[dtype](py=sp_res)
        with out.map_to_host() as res_mojo:
            # print("out:", res_mojo[0])
            # print("expected:", sp_res)
            # may want to use assert_almost_equal with tolerance specified
            assert_almost_equal(res_mojo[0], sp_res_mojo, atol=atol)


def dotc_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ dotc test", dtype, "]")
        out = ctx.enqueue_create_buffer[dtype](2)
        out.enqueue_fill(0)
        a_device = ctx.enqueue_create_buffer[dtype](size*2)
        a = ctx.enqueue_create_host_buffer[dtype](size*2)
        b_device = ctx.enqueue_create_buffer[dtype](size*2)
        b = ctx.enqueue_create_host_buffer[dtype](size*2)

        # Generate two arrays of random numbers on CPU
        generate_random_arr[dtype, size*2](a.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype, size*2](b.unsafe_ptr(), -1, 1)

        ctx.enqueue_copy(a_device, a)
        ctx.enqueue_copy(b_device, b)

        blas_dotc[dtype](
            size,
            a_device.unsafe_ptr(), 1,
            b_device.unsafe_ptr(), 1,
            out.unsafe_ptr(),
            ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move a and b to a SciPy-compatible array and run SciPy BLAS routine
        py_a = Python.list()
        py_b = Python.list()
        for i in range(size):
            py_a.append(np.complex64(a[i*2], a[i*2+1]))
            py_b.append(np.complex64(b[i*2], b[i*2+1]))

        var sp_res: PythonObject
        # Scipy only supports one precision - cdotc
        if dtype == DType.float32:
            np_a = np.array(py_a, dtype=np.complex64)
            np_b = np.array(py_b, dtype=np.complex64)
            sp_res = sp_blas.cdotc(np_a, np_b)
        elif dtype == DType.float64:
            np_a = np.array(py_a, dtype=np.complex64)
            np_b = np.array(py_b, dtype=np.complex64)
            sp_res = sp_blas.cdotc(np_a, np_b)
        else:
            print(dtype , " is not supported by SciPy")
            return

        sp_res_mojo_real = Scalar[dtype](py=sp_res.real)
        sp_res_mojo_imag = Scalar[dtype](py=sp_res.imag)
        with out.map_to_host() as res_mojo:
            # print("out:", res_mojo[0])
            # print("expected:", sp_res)
            # may want to use assert_almost_equal with tolerance specified
            assert_almost_equal(res_mojo[0], sp_res_mojo_real, atol=atol)
            assert_almost_equal(res_mojo[1], sp_res_mojo_imag, atol=atol)

def dotu_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ dotu test", dtype, "]")
        out = ctx.enqueue_create_buffer[dtype](2)
        out.enqueue_fill(0)
        a_device = ctx.enqueue_create_buffer[dtype](size*2)
        a = ctx.enqueue_create_host_buffer[dtype](size*2)
        b_device = ctx.enqueue_create_buffer[dtype](size*2)
        b = ctx.enqueue_create_host_buffer[dtype](size*2)

        # Generate two arrays of random numbers on CPU
        generate_random_arr[dtype, size*2](a.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype, size*2](b.unsafe_ptr(), -1, 1)

        ctx.enqueue_copy(a_device, a)
        ctx.enqueue_copy(b_device, b)

        blas_dotu[dtype](
            size,
            a_device.unsafe_ptr(), 1,
            b_device.unsafe_ptr(), 1,
            out.unsafe_ptr(),
            ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move a and b to a SciPy-compatible array and run SciPy BLAS routine
        py_a = Python.list()
        py_b = Python.list()
        for i in range(size):
            py_a.append(np.complex64(a[i*2], a[i*2+1]))
            py_b.append(np.complex64(b[i*2], b[i*2+1]))

        var sp_res: PythonObject
        # Scipy only supports one precision - cdotu
        if dtype == DType.float32:
            np_a = np.array(py_a, dtype=np.complex64)
            np_b = np.array(py_b, dtype=np.complex64)
            sp_res = sp_blas.cdotu(np_a, np_b)
        elif dtype == DType.float64:
            np_a = np.array(py_a, dtype=np.complex64)
            np_b = np.array(py_b, dtype=np.complex64)
            sp_res = sp_blas.cdotu(np_a, np_b)
        else:
            print(dtype , " is not supported by SciPy")
            return

        sp_res_mojo_real = Scalar[dtype](py=sp_res.real)
        sp_res_mojo_imag = Scalar[dtype](py=sp_res.imag)
        with out.map_to_host() as res_mojo:
            # print("out:", res_mojo[0])
            # print("expected:", sp_res)
            # may want to use assert_almost_equal with tolerance specified
            assert_almost_equal(res_mojo[0], sp_res_mojo_real, atol=atol)
            assert_almost_equal(res_mojo[1], sp_res_mojo_imag, atol=atol)


def iamax_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ iamax test:", dtype, "]")

        # Allocate GPU and CPU memory
        d_v = ctx.enqueue_create_buffer[dtype](size)
        v = ctx.enqueue_create_host_buffer[dtype](size)

        # Generate an array of random numbers on CPU
        generate_random_arr[dtype, size](v.unsafe_ptr(), -10000, 10000)

        # Copy random vector from CPU to GPU memory
        ctx.enqueue_copy(d_v, v)

        # Allocate memory for a single int on GPU to store result, initialize to -1
        d_res = ctx.enqueue_create_buffer[DType.int64](1)
        d_res.enqueue_fill(Int64(-1))

        # Launch Mojo BLAS kernel
        blas_iamax[dtype](size, d_v.unsafe_ptr(), 1, d_res.unsafe_ptr(), ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move values in v to a SciPy-compatible array and run SciPy BLAS routine
        py_list = Python.list()
        for i in range(size):
            py_list.append(v[i])
        var sp_res: PythonObject
        # isamax - float32, idamax - float64
        if dtype == DType.float32:
            np_v = np.array(py_list, dtype=np.float32)
            sp_res = sp_blas.isamax(np_v)
        elif dtype == DType.float64:
            np_v = np.array(py_list, dtype=np.float64)
            sp_res = sp_blas.idamax(np_v)
        else:
            print(dtype , " is not supported by SciPy")
            return

        # Move Mojo result from CPU to GPU and compare to SciPy
        sp_res_mojo = Int(py=sp_res)             # cast Python int into Mojo int
        with d_res.map_to_host() as res_mojo:
            # print("out:", res_mojo[0])
            # print("expected:", sp_res)
            assert_equal(res_mojo[0], sp_res_mojo)


def nrm2_test[
    dtype:DType,
    size: Int
]():
    with DeviceContext() as ctx:
        x = ctx.enqueue_create_host_buffer[dtype](size)
        d_x = ctx.enqueue_create_buffer[dtype](size)
        d_res = ctx.enqueue_create_buffer[dtype](1)

        generate_random_arr[dtype, size](x.unsafe_ptr(), -1000, 1000)
        ctx.enqueue_copy(d_x, x)

        d_res.enqueue_fill(-1) # set result to -1 for now
        blas_nrm2[dtype](size, d_x.unsafe_ptr(), 1, d_res.unsafe_ptr(), ctx)

         # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move values in v to a SciPy-compatible array and run SciPy BLAS routine
        py_list = Python.list()
        for i in range(size):
            py_list.append(x[i])

        var sp_res: PythonObject
        # snrm2 - float32, dnrm2 - float64
        if dtype == DType.float32:
            np_x = np.array(py_list, dtype=np.float32)
            sp_res = sp_blas.snrm2(np_x)
        elif dtype == DType.float64:
            np_x = np.array(py_list, dtype=np.float64)
            sp_res = sp_blas.dnrm2(np_x)
        else:
            print(dtype , " is not supported by SciPy")
            return

        # Move Mojo result from CPU to GPU and compare to SciPy
        sp_res_mojo = Scalar[dtype](py=sp_res)
        with d_res.map_to_host() as res_mojo:
            res_mojo[0] = sqrt(res_mojo[0])
            # print("out:", res_mojo[0])
            # print("expected:", sp_res)
            assert_almost_equal(res_mojo[0], sp_res_mojo)


def rot_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
    #     print("[ rot test:", dtype, "]")

        d_x = ctx.enqueue_create_buffer[dtype](size)
        x = ctx.enqueue_create_host_buffer[dtype](size)
        d_y = ctx.enqueue_create_buffer[dtype](size)
        y = ctx.enqueue_create_host_buffer[dtype](size)

        generate_random_arr[dtype, size](x.unsafe_ptr(), -10000, 10000)
        generate_random_arr[dtype, size](y.unsafe_ptr(), -10000, 10000)

        ctx.enqueue_copy(d_x, x)
        ctx.enqueue_copy(d_y, y)

        # Generate random angle for sin and cos
        var angle = random_float64(0, 2 * 3.14159265359)
        var c = Scalar[dtype](cos(angle))
        var s = Scalar[dtype](sin(angle))

        blas_rot[dtype](size, d_x.unsafe_ptr(), 1, d_y.unsafe_ptr(), 1, c, s, ctx)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_x = Python.list()
        py_y = Python.list()

        for i in range(size):
            py_x.append(x[i])
            py_y.append(y[i])

        # srot - float32, drot - float64
        if dtype == DType.float32:
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            var res = sp_blas.srot(np_x, np_y, c=c, s=s)
            # rot returns updated x and y arrays in a single variable
            np_x = res[0]
            np_y = res[1]
        elif dtype == DType.float64:
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            var res = sp_blas.drot(np_x, np_y, c=c, s=s)
            np_x = res[0]
            np_y = res[1]
        else:
            print(dtype , " is not supported by SciPy")
            return

        with d_x.map_to_host() as x_result:
            with d_y.map_to_host() as y_result:
                # Check x vector
                for i in range(size):
                    var expected_x = Scalar[dtype](py=np_x[i])
                    assert_almost_equal(x_result[i], expected_x, atol=atol)

                # Check y vector
                for i in range(size):
                    var expected_y = Scalar[dtype](py=np_y[i])
                    assert_equal(y_result[i], expected_y)


def rotg_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        d_x = ctx.enqueue_create_buffer[dtype](size)
        x = ctx.enqueue_create_host_buffer[dtype](size)
        d_y = ctx.enqueue_create_buffer[dtype](size)
        y = ctx.enqueue_create_host_buffer[dtype](size)

        generate_random_arr[dtype, size](x.unsafe_ptr(), -10000, 10000)
        generate_random_arr[dtype, size](y.unsafe_ptr(), -10000, 10000)

        ctx.enqueue_copy(d_x, x)
        ctx.enqueue_copy(d_y, y)

        d_c = ctx.enqueue_create_buffer[dtype](1)
        d_s = ctx.enqueue_create_buffer[dtype](1)

        # TODO: implement this
        d_c.enqueue_fill(-1)    # remove when done
        d_s.enqueue_fill(-1)
        # blas_rotg[dtype](
        #     d_x.unsafe_ptr(),
        #     d_y.unsafe_ptr(),
        #     d_c.unsafe_ptr(),
        #     d_s.unsafe_ptr(),
        #     ctx
        # )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_x = Python.list()
        py_y = Python.list()

        for i in range(size):
            py_x.append(x[i])
            py_y.append(y[i])

        # srotg - float32, drotg - float64
        if dtype == DType.float32:
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            var res = sp_blas.srotg(np_x, np_y)
            # rotg returns updated cos and sin in a single variable
            np_c = res[0]
            np_s = res[1]
        elif dtype == DType.float64:
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            var res = sp_blas.drotg(np_x, np_y)
            np_c = res[0]
            np_s = res[1]
        else:
            print(dtype , " is not supported by SciPy")
            return

        with d_c.map_to_host() as c_result:
            with d_s.map_to_host() as s_result:
                # Check cos & sin
                assert_equal(c_result[0], Scalar[dtype](py=np_c))
                assert_equal(s_result[0], Scalar[dtype](py=np_s))


def scal_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ scal test:", dtype, "]")

        a: SIMD[dtype, 1] = 0
        x = ctx.enqueue_create_host_buffer[dtype](size)
        mojo_res = ctx.enqueue_create_host_buffer[dtype](size)

        generate_random_arr[dtype, 1](UnsafePointer[SIMD[dtype, 1]](to=a), -10000, 10000)
        generate_random_arr[dtype, size](x.unsafe_ptr(), -10000, 10000)
        # print("a = ", a)
        # print("x = ", x)

        d_x = ctx.enqueue_create_buffer[dtype](size)

        ctx.enqueue_copy(d_x, x)

        blas_scal[dtype](size, a, d_x.unsafe_ptr(), 1, ctx)

        sp_blas = Python.import_module("scipy.linalg.blas")
        builtins = Python.import_module("builtins")

        x_py = Python.list()
        for i in range(size):
            x_py.append(x[i])

        if dtype == DType.float32:
            sp_result = sp_blas.sscal(a, x_py)
        elif dtype == DType.float64:
            sp_result = sp_blas.dscal(a, x_py)
        else:
            print(dtype , " is not supported by SciPy")
            return

        ctx.enqueue_copy(mojo_res, d_x)
        ctx.synchronize()

        # Prints too much for large vectors. May want to add a verbose option
        # print("out:", mojo_res)
        # print("expected", sp_result)

        for i in range(size):
            assert_almost_equal(Scalar[dtype](py=sp_result[i]), mojo_res[i], atol=atol)


def swap_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        # print("[ swap test:", dtype, "]")

        x = ctx.enqueue_create_host_buffer[dtype](size)
        y = ctx.enqueue_create_host_buffer[dtype](size)
        x2 = ctx.enqueue_create_host_buffer[dtype](size)
        y2 = ctx.enqueue_create_host_buffer[dtype](size)

        generate_random_arr[dtype, size](x.unsafe_ptr(), -10000, 10000)
        generate_random_arr[dtype, size](y.unsafe_ptr(), -10000, 10000)

        d_x = ctx.enqueue_create_buffer[dtype](size)
        d_y = ctx.enqueue_create_buffer[dtype](size)

        ctx.enqueue_copy(d_x, x)
        ctx.enqueue_copy(d_y, y)

        blas_swap[dtype](size, d_x.unsafe_ptr(), 1, d_y.unsafe_ptr(), 1, ctx)

        ctx.enqueue_copy(x2, d_x)
        ctx.enqueue_copy(y2, d_y)
        ctx.synchronize()

        for i in range(size):
            assert_equal(x[i], y2[i])
            assert_equal(y[i], x2[i])


def test_asum():
    asum_test[DType.float32, 256]()
    asum_test[DType.float32, 4096]()
    asum_test[DType.float64, 256]()
    asum_test[DType.float64, 4096]()

def test_axpy():
    axpy_test[DType.float32, 256]()
    axpy_test[DType.float32, 4096]()
    axpy_test[DType.float64, 256]()
    axpy_test[DType.float64, 4096]()

def test_copy():
    copy_test[DType.float32, 256]()
    copy_test[DType.float32, 4096]()
    copy_test[DType.float64, 256]()
    copy_test[DType.float64, 4096]()

def test_dot():
    dot_test[DType.float32, 256]()
    dot_test[DType.float32, 4096]()
    dot_test[DType.float64, 256]()
    dot_test[DType.float64, 4096]()

def test_dotc():
    dotc_test[DType.float32, 256]()
    dotc_test[DType.float32, 4096]()
    dotc_test[DType.float64, 256]()
    dotc_test[DType.float64, 4096]()

def test_dotu():
    dotc_test[DType.float32, 256]()
    dotc_test[DType.float32, 4096]()
    dotc_test[DType.float64, 256]()
    dotc_test[DType.float64, 4096]()

def test_iamax():
    iamax_test[DType.float32, 256]()
    iamax_test[DType.float32, 4096]()
    iamax_test[DType.float64, 256]()
    iamax_test[DType.float64, 4096]()

def test_nrm2():
    nrm2_test[DType.float32, 256]()
    nrm2_test[DType.float32, 4096]()
    nrm2_test[DType.float64, 256]()
    nrm2_test[DType.float64, 4096]()

def test_rot():
    rot_test[DType.float32, 256]()
    rot_test[DType.float32, 4096]()
    rot_test[DType.float64, 256]()
    rot_test[DType.float64, 4096]()

def test_rotg():
    rotg_test[DType.float32, 256]()
    rotg_test[DType.float32, 4096]()
    rotg_test[DType.float64, 256]()
    rotg_test[DType.float64, 4096]()

def test_scal():
    scal_test[DType.float32, 256]()
    scal_test[DType.float32, 4096]()
    scal_test[DType.float64, 256]()
    scal_test[DType.float64, 4096]()

def test_swap():
    swap_test[DType.float32, 256]()
    swap_test[DType.float32, 4096]()
    swap_test[DType.float64, 256]()
    swap_test[DType.float64, 4096]()

def main():
    print("--- MojoBLAS Level 1 routines testing ---")
    TestSuite.discover_tests[__functions_in_module()]().run()
