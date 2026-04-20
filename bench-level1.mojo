from gpu.host import DeviceContext
from sys import has_accelerator, argv
from time import monotonic
from math import sin, cos
from src import *

# Reference: https://github.com/icl-utk-edu/blaspp/blob/master/test/run_tests.py

comptime WARMUP = 5


def bytes_per_elem(dtype: DType) -> Int:
    if dtype == DType.float32:
        return 4
    if dtype == DType.float64:
        return 8
    return 0


struct RunParams:
    var routines: List[String]
    var dtype_str: String
    var sizes: List[Int]
    var iters: Int
    var dim_str: String

    fn __init__(out self):
        self.routines = List[String]()
        self.dtype_str = String("all")
        self.sizes = List[Int]()
        self.iters = 100
        self.dim_str = String("")

# dim parameter usage: --dim size
#                   or --dim min_size:max_size (doubling size with each step)
#                   or --dim min_size:max_size:step
def parse_dim(dim_str: String, mut sizes: List[Int]):
    if dim_str.find(":") != -1:
        var parts = dim_str.split(":")
        var start = Int(parts[0])
        var stop = Int(parts[1])
        if len(parts) == 3:
            # linear range
            var step = Int(parts[2])
            var n = start
            while n <= stop:
                sizes.append(n)
                n += step
        else:
            # doubling each step or single input size
            var n = start
            while n <= stop:
                sizes.append(n)
                n *= 2
    elif dim_str.find(",") != -1:
        var parts = dim_str.split(",")
        for i in range(len(parts)):
            sizes.append(Int(parts[i]))
    else:
        sizes.append(Int(dim_str))


def parse_args(mut params: RunParams) -> Bool:
    var args = argv()

    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--type":
            if i + 1 < len(args):
                params.dtype_str = String(args[i + 1])
                i += 2
            else:
                print("--type requires a value")
                return False
        elif arg == "--dim":
            if i + 1 < len(args):
                params.dim_str = String(args[i + 1])
                i += 2
            else:
                print("--dim requires a value")
                return False
        elif arg == "--iters":
            if i + 1 < len(args):
                params.iters = Int(args[i + 1])
                i += 2
            else:
                print("--iters requires a value")
                return False
        elif not arg.startswith("-"):
            params.routines.append(arg)
            i += 1
        else:
            i += 1

    if len(params.dim_str) > 0:
        parse_dim(params.dim_str, params.sizes)
    else:
        # Defaults:
        params.sizes.append(1 << 10)  # 2^10
        params.sizes.append(1 << 14)  # 2^14
        params.sizes.append(1 << 18)  # 2^18
        params.sizes.append(1 << 22)  # 2^22
        params.sizes.append(1 << 26)  # 2^26

    if len(params.routines) == 0:
        params.routines = ["asum", "axpy", "copy", "dot", "dotc",
                           "dotu", "iamax", "nrm2", "rot", "rotg", "rotm", "rotmg", "scal", "swap"]

    return True


def bench_asum[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_asum[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_asum[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: n reads
    var bw_gbs = Float32(n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("asum," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_axpy[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](2.0)

    for _ in range(WARMUP):
        blas_axpy[dtype](n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_axpy[dtype](n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2n reads + n writes = 3n
    var bw_gbs = Float32(3 * n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("axpy," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))


def bench_copy[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: n reads + n writes = 2n
    var bw_gbs = Float32(2 * n * bytes_per_elem(dtype)) / min_max_mean[0]
    print("copy," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_dot[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_dot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_dot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2n reads
    var bw_gbs = Float32(2 * n * bytes_per_elem(dtype)) / min_max_mean[0]
    print("dot," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_dotc[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    y_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    generate_random_arr[dtype](2 * n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](2 * n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](2 * n)
    y_d = ctx.enqueue_create_buffer[dtype](2 * n)
    res_d = ctx.enqueue_create_buffer[dtype](2)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_dotc[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_dotc[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2 vectors * 2n floats = 4n reads
    var bw_gbs = Float32(4 * n * bytes_per_elem(dtype)) / min_max_mean[0]
    print("dotc," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_dotu[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    y_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    generate_random_arr[dtype](2 * n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](2 * n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](2 * n)
    y_d = ctx.enqueue_create_buffer[dtype](2 * n)
    res_d = ctx.enqueue_create_buffer[dtype](2)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_dotu[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_dotu[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2 vectors * 2n floats = 4n reads
    var bw_gbs = Float32(4 * n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("dotu," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))


def bench_iamax[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[DType.int64](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_iamax[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_iamax[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: n reads
    var bw_gbs = Float32(n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("iamax," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_nrm2[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_nrm2[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_nrm2[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: n reads
    var bw_gbs = Float32(n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("nrm2," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_rot[dtype: DType](n: Int, iters: Int, ctx: DeviceContext) where dtype.is_floating_point():
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var angle = generate_random_scalar[dtype](0, 2 * 3.14159265359)
    var c = Scalar[dtype](cos(angle))
    var s = Scalar[dtype](sin(angle))

    for _ in range(WARMUP):
        blas_rot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, c, s, ctx)
    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_rot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, c, s, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2n reads
    var bw_gbs = Float32(2 * n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("rot," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_rotg[dtype: DType](iters: Int, ctx: DeviceContext):
    var a = generate_random_scalar[dtype](-100, 100)
    var b = generate_random_scalar[dtype](-100, 100)
    var c = Scalar[dtype](0)
    var s = Scalar[dtype](0)

    for _ in range(WARMUP):
        blas_rotg[dtype](UnsafePointer(to=a), UnsafePointer(to=b), UnsafePointer(to=c), UnsafePointer(to=s))

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_rotg[dtype](UnsafePointer(to=a), UnsafePointer(to=b), UnsafePointer(to=c), UnsafePointer(to=s))
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)
    var bw_gbs: Float32 = 0.0

    print("rotg,cpu," + String(dtype) + ",0," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_rotm[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)

    # d1 and d2 must be positive
    var d1 = generate_random_scalar[dtype](1, 100)
    var d2 = generate_random_scalar[dtype](1, 100)
    var x1 = generate_random_scalar[dtype](-100, 100)
    var y1 = generate_random_scalar[dtype](-100, 100)
    var param = List[Scalar[dtype]](length=5, fill=0.0)
    blas_rotmg[dtype](UnsafePointer(to=d1), UnsafePointer(to=d2), UnsafePointer(to=x1), UnsafePointer(to=y1), param)
    param_h = ctx.enqueue_create_host_buffer[dtype](5)
    param_d = ctx.enqueue_create_buffer[dtype](5)
    var param_ptr = param_h.unsafe_ptr()
    for i in range(5):
        param_ptr[i] = param[i]
    ctx.enqueue_copy(param_d, param_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_rotm[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, param_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_rotm[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, param_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2n reads
    var bw_gbs = Float32(2 * n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("rotm," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))


def bench_rotmg[dtype: DType](iters: Int, ctx: DeviceContext):
    var d1 = generate_random_scalar[dtype](1, 100)
    var d2 = generate_random_scalar[dtype](1, 100)
    var x1 = generate_random_scalar[dtype](-100, 100)
    var y1 = generate_random_scalar[dtype](-100, 100)
    var param = List[Scalar[dtype]](length=5, fill=0.0)

    for _ in range(WARMUP):
        blas_rotmg[dtype](UnsafePointer(to=d1), UnsafePointer(to=d2), UnsafePointer(to=x1), UnsafePointer(to=y1), param)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_rotmg[dtype](UnsafePointer(to=d1), UnsafePointer(to=d2), UnsafePointer(to=x1), UnsafePointer(to=y1), param)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)
    var bw_gbs: Float32 = 0.0

    print("rotmg,cpu," + String(dtype) + ",0," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))


def bench_scal[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](2.0)

    for _ in range(WARMUP):
        blas_scal[dtype](n, alpha, x_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_scal[dtype](n, alpha, x_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: n reads + n writes = 2n
    var bw_gbs = Float32(2 * n * bytes_per_elem(dtype)) / min_max_mean[0]

    print("scal," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_swap[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_swap[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_swap[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: 2n reads + 2n writes = 4n
    var bw_gbs = Float32(4 * n * bytes_per_elem(dtype)) / min_max_mean[0]
    print("swap," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def run_dtype[
    dtype: DType
](
    routine: String,
    params: RunParams,
    ctx: DeviceContext
) where dtype.is_floating_point():
    for i in range(len(params.sizes)):
        var n = params.sizes[i]
        if   (routine == "asum"):  bench_asum[dtype](n, params.iters, ctx)
        elif (routine == "axpy"):  bench_axpy[dtype](n, params.iters, ctx)
        elif (routine == "copy"):  bench_copy[dtype](n, params.iters, ctx)
        elif (routine == "dot"):   bench_dot[dtype](n, params.iters, ctx)
        elif (routine == "dotc"):  bench_dotc[dtype](n, params.iters, ctx)
        elif (routine == "dotu"):  bench_dotu[dtype](n, params.iters, ctx)
        elif (routine == "iamax"): bench_iamax[dtype](n, params.iters, ctx)
        elif (routine == "nrm2"):  bench_nrm2[dtype](n, params.iters, ctx)
        elif (routine == "rot"):   bench_rot[dtype](n, params.iters, ctx)
        elif (routine == "rotg"):
            # rotg and rotmg don't take n -- run once and return
            bench_rotg[dtype](params.iters, ctx)
            return
        elif (routine == "rotm"):  bench_rotm[dtype](n, params.iters, ctx)
        elif (routine == "rotmg"):
            bench_rotmg[dtype](params.iters, ctx)
            return
        elif (routine == "scal"):  bench_scal[dtype](n, params.iters, ctx)
        elif (routine == "swap"):  bench_swap[dtype](n, params.iters, ctx)
        else:
            print("Unknown routine:", routine, "for", dtype)
            return


def main():
    if not has_accelerator():
        print("No accelerator detected")
        return

    var params = RunParams()
    if not parse_args(params):
        return

    print("op,device,dtype,n,iters,best_ms,avg_ms,best_bandwidth_GBs")

    with DeviceContext() as ctx:
        for routine in(params.routines):
            if params.dtype_str == "float32" or params.dtype_str == "all":
                run_dtype[DType.float32](routine, params, ctx)

            if params.dtype_str == "float64" or params.dtype_str == "all":
                run_dtype[DType.float64](routine, params, ctx)
