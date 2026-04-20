from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv
from memory import stack_allocation, memset_zero

comptime TBsize = 1024
comptime Blocksize = 32

fn sgemm_device[trans_a: Int, trans_b: Int](
    m: Int,
    n: Int,
    k: Int,
    alpha: Float32,
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int,
    B: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int,
    beta: Float32,
    C: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int,
) :

    var row = block_idx.x
    var col = block_idx.y
    var threadRow = thread_idx.y
    var threadCol = thread_idx.x

    var A_base: UInt
    var A_kstep: UInt
    var B_base: UInt
    var B_kstep: UInt

    @parameter
    if trans_a:
        A_base = row * Blocksize
        A_kstep = Blocksize * lda
    else:
        A_base = row * Blocksize * lda
        A_kstep = Blocksize

    @parameter
    if trans_b:
        B_base = col * Blocksize * ldb
        B_kstep = Blocksize
    else:
        B_base = col * Blocksize
        B_kstep = Blocksize * ldb

    var C_base = row * Blocksize * ldc + col * Blocksize

    var As = stack_allocation[Blocksize * Blocksize, DType.float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[Blocksize * Blocksize, DType.float32, address_space=AddressSpace.SHARED]()

    var tmp = Scalar[DType.float32](0)
    for bk in range(0, k, Blocksize):
        @parameter
        if trans_a:
            As[threadRow * Blocksize + threadCol] = A[A_base + threadCol * lda + threadRow]
        else:
            As[threadRow * Blocksize + threadCol] = A[A_base + threadRow * lda + threadCol]

        @parameter
        if trans_b:
            Bs[threadRow * Blocksize + threadCol] = B[B_base + threadCol * ldb + threadRow]
        else:
            Bs[threadRow * Blocksize + threadCol] = B[B_base + threadRow * ldb + threadCol]

        barrier()

        A_base += A_kstep
        B_base += B_kstep

        for dotIdx in range(Blocksize):
            tmp += As[threadRow * Blocksize + dotIdx] * Bs[dotIdx * Blocksize + threadCol]

        barrier()

    var C_idx = C_base + threadRow * ldc + threadCol
    C[C_idx] = alpha * tmp + beta * C[C_idx]


fn dgemm_device[trans_a: Int, trans_b: Int](
    m: Int,
    n: Int,
    k: Int,
    alpha: Float64,
    A: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int,
    B: UnsafePointer[Float64, ImmutAnyOrigin],
    ldb: Int,
    beta: Float64,
    C: UnsafePointer[Float64, MutAnyOrigin],
    ldc: Int,
) :
    var row = block_idx.x
    var col = block_idx.y
    var threadRow = thread_idx.y
    var threadCol = thread_idx.x

    var A_base: UInt
    var A_kstep: UInt
    var B_base: UInt
    var B_kstep: UInt

    @parameter
    if trans_a:
        A_base = row * Blocksize
        A_kstep = Blocksize * lda
    else:
        A_base = row * Blocksize * lda
        A_kstep = Blocksize

    @parameter
    if trans_b:
        B_base = col * Blocksize * ldb
        B_kstep = Blocksize
    else:
        B_base = col * Blocksize
        B_kstep = Blocksize * ldb

    var C_base = row * Blocksize * ldc + col * Blocksize

    var As = stack_allocation[Blocksize * Blocksize, DType.float64, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[Blocksize * Blocksize, DType.float64, address_space=AddressSpace.SHARED]()

    var tmp = Scalar[DType.float64](0)
    for bk in range(0, k, Blocksize):
        @parameter
        if trans_a:
            As[threadRow * Blocksize + threadCol] = A[A_base + threadCol * lda + threadRow]
        else:
            As[threadRow * Blocksize + threadCol] = A[A_base + threadRow * lda + threadCol]

        @parameter
        if trans_b:
            Bs[threadRow * Blocksize + threadCol] = B[B_base + threadCol * ldb + threadRow]
        else:
            Bs[threadRow * Blocksize + threadCol] = B[B_base + threadRow * ldb + threadCol]

        barrier()

        A_base += A_kstep
        B_base += B_kstep

        for dotIdx in range(Blocksize):
            tmp += As[threadRow * Blocksize + dotIdx] * Bs[dotIdx * Blocksize + threadCol]

        barrier()

    var C_idx = C_base + threadRow * ldc + threadCol
    C[C_idx] = alpha * tmp + beta * C[C_idx]

def launch_gemm[dtype: DType, trans_a: Int, trans_b: Int](
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    ldb: Int,
    beta: Scalar[dtype],
    d_C: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ldc: Int,
    ctx: DeviceContext
) :
    @parameter 
    if dtype == DType.float32 :
        ctx.enqueue_function[sgemm_device[trans_a, trans_b], sgemm_device[trans_a, trans_b]](
        m, n, k,
        alpha, 
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc,
        grid_dim=(ceildiv(m, Blocksize), ceildiv(n, Blocksize)),
        block_dim= (Blocksize, Blocksize))
    elif dtype == DType.float64 :
        ctx.enqueue_function[dgemm_device[trans_a, trans_b], dgemm_device[trans_a, trans_b]](
        m, n, k,
        alpha, 
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc,
        grid_dim=(ceildiv(m, Blocksize), ceildiv(n, Blocksize)),
        block_dim=(Blocksize, Blocksize))
    else :
        raise Error("blas_gemm: Unsupported type")

        



fn blas_gemm[dtype: DType](
    trans_a: Bool, trans_b: Bool,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    ldb: Int,
    beta: Scalar[dtype],
    d_C: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ldc: Int,
    ctx: DeviceContext
) raises :
    """
    Performs Matrix multiplication of from:
    C := alpha*op( A )*op( B ) + beta*C
    where op ( X ) is one of 
    op( X ) = X or op ( X ) = X**T
    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
    """

    blas_error_if["blas_gemm" , "m < 0"](m < 0)
    blas_error_if["blas_gemm" , "n < 0"](n < 0)
    blas_error_if["blas_gemm" , "k < 0"](k < 0)

    if trans_a :
        blas_error_if["blas_gemm" , "lda < m"](lda < m)
    else :
        blas_error_if["blas_gemm" , "lda < k"](lda < k)
    if trans_b : 
        blas_error_if["blas_gemm" , "ldb < k"](ldb < k)
    else :
        blas_error_if["blas_gemm" , "ldb < n"](ldb < n)

    blas_error_if["blas_gemm" , "ldc < n"](ldc < n)

    # quick returns
    if m == 0 or n == 0: return

 

    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    comptime scal_kernel = scal_device.scal_device[dtype]

    if alpha == zero or k == 0 : # No Matrix multiplication, use scale kernel
        if beta == one :
            return
        else :
            ctx.enqueue_function[scal_kernel, scal_kernel](m*n, beta, d_C, 1, grid_dim=ceildiv(m*n, TBsize), block_dim=TBsize)
        ctx.synchronize()
        return
    
    #convert trans flags to comptime parameters
    if trans_a and trans_b :
        launch_gemm[dtype, 1, 1](m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc, ctx)
    elif trans_a :
        launch_gemm[dtype, 1, 0](m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc, ctx)
    elif trans_b :
        launch_gemm[dtype, 0, 1](m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc, ctx)
    else:
        launch_gemm[dtype, 0, 0](m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc, ctx)

    ctx.synchronize()
