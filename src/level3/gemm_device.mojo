from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv
from memory import memset_zero, memcpy

comptime TBsize = 512
comptime TBx = 32
comptime TBy = 16
fn sgemm_device(
    trans_a: Int, trans_b: Int,
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

    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    if row < m and col < n :
        var sum = Scalar[DType.float32](0)
        if trans_a and trans_b:
            for i in range(k) :
                sum += A[i * lda + row] * B[col * ldb + i]
        elif trans_a :
            for i in range(k) :
                sum += A[i * lda + row] * B[i * ldb + col]
        elif trans_b :
            for i in range(k) :
                sum += A[row * lda + i] * B[col * ldb + i]
        else :
            for i in range(k) :
                sum += A[row * lda + i] * B[i * ldb + col]
        
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col]
        

    # var global_row = block_dim.y * block_idx.y + thread_idx.y
    # var global_col = block_dim.x * block_idx.x + thread_idx.x
    # var n_threads_row = grid_dim.y * block_dim.y
    # var n_threads_col = grid_dim.x * block_dim.x

    # for i in range(global_row, m, n_threads_row) :
    #     for j in range(global_col, n, n_threads_col) :
    #         var sum = Scalar[DType.float32](0)
    #         if trans_a and trans_b :
    #             for kk in range(k) :
    #                 sum += A[kk * lda + i] * B[j * ldb + kk]
    #         elif trans_a :
    #             for kk in range(k) :
    #                 sum += A[kk * lda + i] * B[kk * ldb + j]
    #         elif trans_b :
    #             for kk in range(k) :
    #                 sum += A[i * lda + kk] * B[j * ldb + kk]
    #         else :
    #             for kk in range(k) :
    #                 sum += A[i * lda + kk] * B[kk * ldb + j]
    #         C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j]


fn dgemm_device(
    trans_a: Int, trans_b: Int,
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
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    if row < m and col < n :
        var sum = Scalar[DType.float64](0)
        if trans_a and trans_b:
            for i in range(k) :
                sum += A[i * lda + row] * B[col * ldb + i]
        elif trans_a :
            for i in range(k) :
                sum += A[i * lda + row] * B[i * ldb + col]
        elif trans_b :
            for i in range(k) :
                sum += A[row * lda + i] * B[col * ldb + i]
        else :
            for i in range(k) :
                sum += A[row * lda + i] * B[i * ldb + col]
        
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col]
    # var global_row = block_dim.y * block_idx.y + thread_idx.y
    # var global_col = block_dim.x * block_idx.x + thread_idx.x
    # var n_threads_row = grid_dim.y * block_dim.y
    # var n_threads_col = grid_dim.x * block_dim.x

    # for i in range(global_row, m, n_threads_row) :
    #     for j in range(global_col, n, n_threads_col) :
    #         var sum = Scalar[DType.float64](0)
    #         if trans_a and trans_b :
    #             for kk in range(k) :
    #                 sum += A[kk * lda + i] * B[j * ldb + kk]
    #         elif trans_a :
    #             for kk in range(k) :
    #                 sum += A[kk * lda + i] * B[kk * ldb + j]
    #         elif trans_b :
    #             for kk in range(k) :
    #                 sum += A[i * lda + kk] * B[j * ldb + kk]
    #         else :
    #             for kk in range(k) :
    #                 sum += A[i * lda + kk] * B[kk * ldb + j]
    #         C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j]


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
    var trans_a_i = 0
    var trans_b_i = 0

    if trans_a :
        blas_error_if["blas_gemm" , "lda < m"](lda < m)
        trans_a_i = 1
    else :
        blas_error_if["blas_gemm" , "lda < k"](lda < k)
    if trans_b : 
        blas_error_if["blas_gemm" , "ldb < k"](ldb < k)
        trans_b_i = 1
    else :
        blas_error_if["blas_gemm" , "ldb < n"](ldb < n)

    blas_error_if["blas_gemm" , "ldc < n"](ldc < n)

    #quick return
    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    if m == 0 or n == 0 or k == 0 : return
    if alpha == zero and beta == one : return


    @parameter
    if dtype == DType.float32:
        if alpha == zero and beta == zero :
            ctx.enqueue_function[szero_kernel, szero_kernel](d_C, m*n, grid_dim=ceildiv(m*n, TBsize), block_dim=TBsize)
        else :
            ctx.enqueue_function[sgemm_device, sgemm_device](
            trans_a_i, trans_b_i,
            m, n, k,
            alpha, 
            d_A, lda,
            d_B, ldb,
            beta,
            d_C, ldc,
            grid_dim=(ceildiv(n, TBx), ceildiv(m, TBy)),
            block_dim=(TBx, TBy))
    elif dtype == DType.float64:
        if alpha == zero and beta == zero :
            ctx.enqueue_function[dzero_kernel, dzero_kernel](d_C, m*n, grid_dim=ceildiv(m*n, TBsize), block_dim=TBsize)
        else :
            ctx.enqueue_function[dgemm_device, dgemm_device](
            trans_a_i, trans_b_i,
            m, n, k,
            alpha, 
            d_A, lda,
            d_B, ldb,
            beta,
            d_C, ldc,
            grid_dim=(ceildiv(m, TBx), ceildiv(n, TBy)),
            block_dim=(TBx, TBy)
        )
    else:
        raise Error("blas_gemm: Unsupported type")

    ctx.synchronize()
