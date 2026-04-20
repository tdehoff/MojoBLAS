from gpu.host import DeviceContext
from sys import has_accelerator, argv
from time import monotonic
from src import *

# All matrix routines use square matrices (m = n).

comptime WARMUP = 1


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
            var step = Int(parts[2])
            var n = start
            while n <= stop:
                sizes.append(n)
                n += step
        else:
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
        # Defaults: matrix dimension n (matrix has n*n elements)
        params.sizes.append(1 << 5)   # 32
        params.sizes.append(1 << 7)   # 128
        params.sizes.append(1 << 9)   # 512
        params.sizes.append(1 << 11)  # 2048
        params.sizes.append(1 << 13)  # 8192

    if len(params.routines) == 0:
        params.routines = ["gemv", "gbmv", "ger", "sbmv", "symv",
                           "syr", "syr2", "spr", "tbmv", "tbsv",
                           "trsv", "tpmv"]

    return True


def bench_gemv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)
    var beta = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_gemv[dtype](False, n, n, alpha, A_d.unsafe_ptr(), n, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_gemv[dtype](False, n, n, alpha, A_d.unsafe_ptr(), n, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read A (n*n) + read x (n) + read y (n) + write y (n) = n*n + 3n
    var bw_gbs = Float32((n * n + 3 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("gemv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_gbmv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    var kl = 10
    var ku = 10
    var lda = kl + ku + 1
    A_h = ctx.enqueue_create_host_buffer[dtype](lda * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](lda * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](lda * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)
    var beta = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_gbmv[dtype](False, n, n, kl, ku, alpha, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_gbmv[dtype](False, n, n, kl, ku, alpha, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read band storage (lda*n) + read x (n) + read y (n) + write y (n) = lda*n + 3n
    var bw_gbs = Float32((lda * n + 3 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("gbmv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_ger[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_ger[dtype](n, n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_ger[dtype](n, n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read x (n) + read y (n) + read A (n*n) + write A (n*n) = 2n*n + 2n
    var bw_gbs = Float32((2 * n * n + 2 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("ger," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_sbmv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    var k = 10
    var lda = k + 1
    A_h = ctx.enqueue_create_host_buffer[dtype](lda * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](lda * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](lda * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)
    var beta = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_sbmv[dtype](1, n, k, alpha, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_sbmv[dtype](1, n, k, alpha, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read band storage (lda*n) + read x (n) + read y (n) + write y (n) = lda*n + 3n
    var bw_gbs = Float32((lda * n + 3 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("sbmv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_symv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)
    var beta = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_symv[dtype](True, n, alpha, A_d.unsafe_ptr(), n, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_symv[dtype](True, n, alpha, A_d.unsafe_ptr(), n, x_d.unsafe_ptr(), 1, beta, y_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read A (n*n/2) + read x (n) + read y (n) + write y (n) = n*n/2 + 3n
    var bw_gbs = Float32((n * n / 2 + 3 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("symv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_syr[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_syr[dtype](1, n, alpha, x_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_syr[dtype](1, n, alpha, x_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read x (n) + read triangle (n*n/2) + write triangle (n*n/2) = n*n + n
    var bw_gbs = Float32((n * n + n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("syr," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_syr2[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_syr2[dtype](1, n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_syr2[dtype](1, n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read x (n) + read y (n) + read triangle (n*n/2) + write triangle (n*n/2) = n*n + 2n
    var bw_gbs = Float32((n * n + 2 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("syr2," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_spr[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    var ap_size = n * (n + 1) // 2
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    AP_h = ctx.enqueue_create_host_buffer[dtype](ap_size)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](ap_size, AP_h.unsafe_ptr(), -1, 1)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    AP_d = ctx.enqueue_create_buffer[dtype](ap_size)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(AP_d, AP_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](1.0)

    for _ in range(WARMUP):
        blas_spr[dtype](1, n, alpha, x_d.unsafe_ptr(), 1, AP_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_spr[dtype](1, n, alpha, x_d.unsafe_ptr(), 1, AP_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read x (n) + read AP (n*(n+1)/2) + write AP (n*(n+1)/2) = n*(n+1) + n
    var bw_gbs = Float32((ap_size * 2 + n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("spr," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_tbmv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    var k = 10
    var lda = k + 1
    A_h = ctx.enqueue_create_host_buffer[dtype](lda * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](lda * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](lda * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    temp_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_tbmv[dtype](1, False, 0, n, k, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, temp_d.unsafe_ptr(), ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_tbmv[dtype](1, False, 0, n, k, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, temp_d.unsafe_ptr(), ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read band (lda*n) + read x (n) + write x (n) = lda*n + 2n
    var bw_gbs = Float32((lda * n + 2 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("tbmv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_tbsv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    var k = 10
    var lda = k + 1
    A_h = ctx.enqueue_create_host_buffer[dtype](lda * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](lda * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](lda * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_tbsv[dtype](1, False, 0, n, k, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_tbsv[dtype](1, False, 0, n, k, A_d.unsafe_ptr(), lda, x_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read band (lda*n) + read/write x (2n) = lda*n + 2n
    var bw_gbs = Float32((lda * n + 2 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("tbsv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_tpmv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    var ap_size = n * (n + 1) // 2
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    AP_h = ctx.enqueue_create_host_buffer[dtype](ap_size)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](ap_size, AP_h.unsafe_ptr(), -1, 1)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    AP_d = ctx.enqueue_create_buffer[dtype](ap_size)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(AP_d, AP_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_tpmv[dtype](1, False, 0, n, AP_d.unsafe_ptr(), x_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_tpmv[dtype](1, False, 0, n, AP_d.unsafe_ptr(), x_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read AP (ap_size) + read/write x (2n) = ap_size + 2n
    var bw_gbs = Float32((ap_size + 2 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("tpmv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))

def bench_trsv[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_trsv[dtype](1, False, 0, n, A_d.unsafe_ptr(), n, x_d.unsafe_ptr(), 1, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters):
        start = monotonic()
        blas_trsv[dtype](1, False, 0, n, A_d.unsafe_ptr(), n, x_d.unsafe_ptr(), 1, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min_max_mean = arr_min_max_mean(timings)

    # bandwidth: read triangle (n*n/2) + read/write x (2n) = n*n/2 + 2n
    var bw_gbs = Float32((n * n / 2 + 2 * n) * bytes_per_elem(dtype)) / min_max_mean[0]

    print("trsv," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
          "," + String(min_max_mean[0] * 1e-6) +
          "," + String(min_max_mean[2] * 1e-6) + "," + String(bw_gbs))


def run_dtype[
    dtype: DType
](
    routine: String,
    params: RunParams,
    ctx: DeviceContext,
) where dtype.is_floating_point():
    for i in range(len(params.sizes)):
        var n = params.sizes[i]
        if   (routine == "gemv"): bench_gemv[dtype](n, params.iters, ctx)
        elif (routine == "gbmv"): bench_gbmv[dtype](n, params.iters, ctx)
        elif (routine == "ger"):  bench_ger[dtype](n, params.iters, ctx)
        elif (routine == "sbmv"): bench_sbmv[dtype](n, params.iters, ctx)
        elif (routine == "symv"): bench_symv[dtype](n, params.iters, ctx)
        elif (routine == "syr"):  bench_syr[dtype](n, params.iters, ctx)
        elif (routine == "syr2"): bench_syr2[dtype](n, params.iters, ctx)
        elif (routine == "spr"):  bench_spr[dtype](n, params.iters, ctx)
        elif (routine == "tbmv"): bench_tbmv[dtype](n, params.iters, ctx)
        elif (routine == "tbsv"): bench_tbsv[dtype](n, params.iters, ctx)
        elif (routine == "tpmv"): bench_tpmv[dtype](n, params.iters, ctx)
        elif (routine == "trsv"): bench_trsv[dtype](n, params.iters, ctx)
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
