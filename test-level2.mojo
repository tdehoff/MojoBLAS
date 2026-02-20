from testing import assert_equal, assert_almost_equal, TestSuite
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, grid_dim, thread_idx
from layout import Layout, LayoutTensor
from math import sqrt

from src import *
from random import rand, seed, randn_float64
from math import ceildiv, sin, cos
from python import Python, PythonObject

comptime TBsize = 512
comptime atol = 1.0E-5


def sger_test[
    m:  Int,
    n: Int,
]():
    with DeviceContext() as ctx:
        A_device = ctx.enqueue_create_buffer[DType.float32](m*n)
        A = ctx.enqueue_create_host_buffer[DType.float32](m*n)
        x_device = ctx.enqueue_create_buffer[DType.float32](m)
        x = ctx.enqueue_create_host_buffer[DType.float32](m)
        y_device = ctx.enqueue_create_buffer[DType.float32](n)
        y = ctx.enqueue_create_host_buffer[DType.float32](n)

        # Generate three arrays of random numbers on CPU
        generate_random_arr[DType.float32, m*n](A.unsafe_ptr(), -100, 100)
        generate_random_arr[DType.float32, m](x.unsafe_ptr(), -100, 100)
        generate_random_arr[DType.float32, n](y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_device, A)
        ctx.enqueue_copy(x_device, x)
        ctx.enqueue_copy(y_device, y)

        var alpha = randn_float64(0.0, 1.0)

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
        # sger - float32
        np_a = np.array(py_a, dtype=np.float32).reshape(m,n)
        np_x = np.array(py_x, dtype=np.float32)
        np_y = np.array(py_y, dtype=np.float32)
        sp_res = sp_blas.sger(alpha, np_x, np_y, 1, 1, np_a)

        blas_sger(
            m,
            n,
            Scalar[DType.float32](alpha),
            x_device.unsafe_ptr(), 1,
            y_device.unsafe_ptr(), 1,
            A_device.unsafe_ptr(), n,
            ctx)

        with A_device.map_to_host() as res_mojo:
            for i in range(m):
                for j in range(n):
                    assert_almost_equal(Scalar[DType.float32](py=sp_res[i][j]), res_mojo[(i*n)+j], atol=atol)

def dger_test[
    m:  Int,
    n: Int,
]():
    with DeviceContext() as ctx:
        A_device = ctx.enqueue_create_buffer[DType.float64](m*n)
        A = ctx.enqueue_create_host_buffer[DType.float64](m*n)
        x_device = ctx.enqueue_create_buffer[DType.float64](m)
        x = ctx.enqueue_create_host_buffer[DType.float64](m)
        y_device = ctx.enqueue_create_buffer[DType.float64](n)
        y = ctx.enqueue_create_host_buffer[DType.float64](n)

        # Generate three arrays of random numbers on CPU
        generate_random_arr[DType.float64, m*n](A.unsafe_ptr(), -100, 100)
        generate_random_arr[DType.float64, m](x.unsafe_ptr(), -100, 100)
        generate_random_arr[DType.float64, n](y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_device, A)
        ctx.enqueue_copy(x_device, x)
        ctx.enqueue_copy(y_device, y)

        var alpha = randn_float64(0.0, 1.0)

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
        # dger - float64
        np_a = np.array(py_a, dtype=np.float64).reshape(m,n)
        np_x = np.array(py_x, dtype=np.float64)
        np_y = np.array(py_y, dtype=np.float64)
        sp_res = sp_blas.dger(alpha, np_x, np_y, 1, 1, np_a)
        blas_dger(
            m,
            n,
            Scalar[DType.float64](alpha),
            x_device.unsafe_ptr(), 1,
            y_device.unsafe_ptr(), 1,
            A_device.unsafe_ptr(), n,
            ctx)

        with A_device.map_to_host() as res_mojo:
            for i in range(m):
                for j in range(n):
                    assert_almost_equal(Scalar[DType.float64](py=sp_res[i][j]), res_mojo[(i*n)+j], atol=atol)

def test_sger():
    sger_test[64, 64]()
    sger_test[256, 256]()

def test_dger():
    dger_test[64, 64]()
    dger_test[256, 256]()

def main():
    print("--- MojoBLAS Level 2 routines testing ---")
    TestSuite.discover_tests[__functions_in_module()]().run()
