from testing import assert_equal, assert_almost_equal, TestSuite
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, grid_dim, thread_idx
from layout import Layout, LayoutTensor

from src import axpy_device, iamax_device, dot_device
from random import rand, seed
from math import ceildiv
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

comptime atol = 1.0E-6

def dot_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        print("[ dot test", dtype, "]")
        out = ctx.enqueue_create_buffer[dtype](1)
        out.enqueue_fill(0)
        a_device = ctx.enqueue_create_buffer[dtype](size)
        a = ctx.enqueue_create_host_buffer[dtype](size)
        b_device = ctx.enqueue_create_buffer[dtype](size)
        b = ctx.enqueue_create_host_buffer[dtype](size)

        comptime in_layout = Layout.row_major(size)

        # Generate two arrays of random numbers on CPU
        generate_random_arr[dtype, size](a.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, size](b.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(a_device, a)
        ctx.enqueue_copy(b_device, b)

        a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a_device.unsafe_ptr())
        b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b_device.unsafe_ptr())

        comptime kernel = dot_device[in_layout, size, dtype]
        ctx.enqueue_function[
            kernel, kernel
        ](
            out,
            a_tensor,
            b_tensor,
            grid_dim=ceildiv(size, TBsize),
            block_dim=TBsize,
        )

        ctx.synchronize()

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
            print(dtype , "is not supported by SciPy")
            return

        sp_res_mojo = Scalar[dtype](py=sp_res)
        with out.map_to_host() as res_mojo:
            print("out:", res_mojo[0])
            print("expected:", sp_res_mojo)
            # may want to use assert_almost_equal with tolerance specified
            assert_almost_equal(res_mojo[0], sp_res_mojo, atol=atol)


def iamax_test[
    dtype: DType,
    size:  Int
]():
    with DeviceContext() as ctx:
        print("[ iamax test:", dtype, "]")

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

        # Launch Mojo GPU kernel
        comptime kernel = iamax_device[TBsize, dtype]
        ctx.enqueue_function[kernel, kernel](
            size, d_v,
            1, d_res,
            grid_dim=ceildiv(size, TBsize),     # total thread blocks
            block_dim=TBsize                    # threads per block
        )

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
            print(dtype , "is not supported by SciPy")
            return

        # Move Mojo result from CPU to GPU and compare to SciPy
        sp_res_mojo = Int(py=sp_res)             # cast Python int into Mojo int
        with d_res.map_to_host() as res_mojo:
            print("out:", res_mojo[0])
            print("expected:", sp_res_mojo)
            assert_equal(res_mojo[0], sp_res_mojo)


def test_axpy():
    with DeviceContext() as ctx:

        a: SIMD[dtype, 1] = 0
        x = ctx.enqueue_create_host_buffer[dtype](size)
        y = ctx.enqueue_create_host_buffer[dtype](size)
        mojo_res = ctx.enqueue_create_host_buffer[dtype](size)

        seed()
        rand[dtype](UnsafePointer[SIMD[dtype, 1]](to=a), 1)
        rand[dtype](x.unsafe_ptr(), size)
        rand[dtype](y.unsafe_ptr(), size)
        print("a = ", a)
        print("x = ", x)
        print("y = ", y)

        d_x = ctx.enqueue_create_buffer[dtype](size)
        d_y = ctx.enqueue_create_buffer[dtype](size)

        ctx.enqueue_copy(d_x, x)
        ctx.enqueue_copy(d_y, y)

        axpy_kernel = ctx.compile_function[axpy_device[dtype], axpy_device[dtype]]()

        ctx.enqueue_function(
            axpy_kernel,
            size, a, d_x, 1, d_y, 1,
            grid_dim=1,
            block_dim=size
        )

        sp_blas = Python.import_module("scipy.linalg.blas")
        builtins = Python.import_module("builtins")

        x_py = Python.list()
        y_py = Python.list()
        for i in range(size):
            x_py.append(x[i])
            y_py.append(y[i])
        sp_result = sp_blas.saxpy(x_py, y_py, a=a)
        print(sp_result)

        ctx.enqueue_copy(mojo_res, d_y)
        ctx.synchronize()

        print("out:", mojo_res)
        print("expected", sp_result)

        for i in range(size):
            var f: Float32 = Float32(py=sp_result[i])
            assert_almost_equal(Float32(py=sp_result[i]), mojo_res[i], atol=1.0E-6)

def test_iamax():
    iamax_test[DType.float32, 256]()
    iamax_test[DType.float64, 256]()

def test_dot():
    dot_test[DType.float32, 256]()
    # It looks like warp_sum doesn't support float64
    # dot_test[DType.float64, 256]()

def main():
    print("--- MojoBLAS Level 1 routines testing ---")
    TestSuite.discover_tests[__functions_in_module()]().run()
