from gpu.host import DeviceContext
from sys import has_accelerator, argv
from time import monotonic
from src import *

# All matrix routines use square matrices (m = n).

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
        # Defaults:
        params.sizes.append(256)
        params.sizes.append(512)
        params.sizes.append(1024)
        params.sizes.append(2048)
        params.sizes.append(4096)

    if len(params.routines) == 0: # TODO: Add other level 3 routines as they are implemented
        params.routines = ["gemm"]

    return True

def bench_gemm[dtype: DType](n: Int, iters: Int, ctx: DeviceContext) :
    A_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    B_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    C_h = ctx.enqueue_create_host_buffer[dtype](n * n)
    generate_random_arr[dtype](n * n, A_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n * n, B_h.unsafe_ptr(), -1, 1)
    generate_random_arr[dtype](n * n, C_h.unsafe_ptr(), -1, 1)
    A_d = ctx.enqueue_create_buffer[dtype](n * n)
    B_d = ctx.enqueue_create_buffer[dtype](n * n)
    C_d = ctx.enqueue_create_buffer[dtype](n * n)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    ctx.synchronize()

    var alpha = generate_random_scalar[dtype](-1,1)
    var beta  = generate_random_scalar[dtype](-1,1)

    for _ in range(WARMUP) :
        blas_gemm(False, False, n , n , n, alpha, A_d.unsafe_ptr(), n, B_d.unsafe_ptr(), n, beta, C_d.unsafe_ptr(), n, ctx)

    var timings = List[Float32](length=iters, fill=0.0)

    for i in range(iters) :
        start = monotonic()
        blas_gemm(False, False, n , n , n, alpha, A_d.unsafe_ptr(), n, B_d.unsafe_ptr(), n, beta, C_d.unsafe_ptr(), n, ctx)
        end = monotonic()
        timings[i] = Float32(end - start)

    var min, max, mean = arr_min_max_mean(timings)
    #bandwidth: read A (n * n) + read B (n * n) + read C (n * n) + write C (n * n)
    var bw_gbs = Float32(4 * n * n * bytes_per_elem(dtype)) / mean

    print("gemm," + ctx.name() + "," + String(dtype) + "," + String(n) + "," + String(iters) +
        "," + String(min * 1e-9) + "," + String(max * 1e-9) +
        "," + String(mean * 1e-9) + "," + String(bw_gbs))


def run_dtype[
    dtype: DType
](
    routine: String,
    params: RunParams,
    ctx: DeviceContext,
) where dtype.is_floating_point():
    for i in range(len(params.sizes)):
        var n = params.sizes[i]
        if   (routine == "gemm"): bench_gemm[dtype](n, params.iters, ctx)
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

    print("op,device,dtype,n,iters,avg_ns,bandwidth_GBs")

    with DeviceContext() as ctx:
        for routine in(params.routines):
            if params.dtype_str == "float32" or params.dtype_str == "all":
                run_dtype[DType.float32](routine, params, ctx)

            if params.dtype_str == "float64" or params.dtype_str == "all":
                run_dtype[DType.float64](routine, params, ctx)

