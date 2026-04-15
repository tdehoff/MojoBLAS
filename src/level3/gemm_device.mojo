from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv
from memory import memset_zero

comptime TBsize = 512
comptime TBx = 32
comptime TBy = 16
fn sgemm_device[trans_a: Flag, trans_b: Flag, ak : ScalarKind, bk : ScalarKind](
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
    var global_row = block_dim.y * block_idx.y + thread_idx.y
    var global_col = block_dim.x * block_idx.x + thread_idx.x
    var n_threads_row = grid_dim.y * block_dim.y
    var n_threads_col = grid_dim.x * block_dim.x

    for i in range(global_row, m, n_threads_row) :
        for j in range(global_col, n, n_threads_col) :
            var sum = Scalar[DType.float32](0)
            @parameter
            if ak == ScalarKind.zero :
                @parameter 
                if bk == ScalarKind.zero :
                    C[i * ldc + j] = Scalar[DType.float32](0)
                elif bk == ScalarKind.gen :
                    C[i * ldc + j] = beta * C[i * ldc + j]
            else :
                @parameter
                if trans_a == Flag.true and trans_b == Flag.true :
                    for kk in range(k) :
                        sum += A[kk * lda + i] * B[j * ldb + kk]
                elif trans_a == Flag.true:
                    for kk in range(k) :
                        sum += A[kk * lda + i] * B[kk * ldb + j]
                elif trans_b == Flag.true:
                    for kk in range(k) :
                        sum += A[i * lda + kk] * B[j * ldb + kk]
                else :
                    for kk in range(k) :
                        sum += A[i * lda + kk] * B[kk * ldb + j]
                @parameter
                if ak == ScalarKind.one :
                    @parameter
                    if bk == ScalarKind.zero :
                        C[i * ldc + j] = sum
                    elif bk == ScalarKind.one :
                        C[i * ldc + j] = sum + C[i * ldc + j]
                    else :
                        C[i * ldc + j] = sum + beta * C[i * ldc + j]
                else:
                    @parameter
                    if bk == ScalarKind.zero : 
                        C[i * ldc + j] = alpha * sum
                    elif bk == ScalarKind.one :
                        C[i * ldc + j] = alpha * sum + C[i * ldc + j]
                    else :
                        C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j]
            


fn dgemm_device[trans_a: Flag, trans_b: Flag, ak: ScalarKind, bk: ScalarKind](
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
    var global_row = block_dim.y * block_idx.y + thread_idx.y
    var global_col = block_dim.x * block_idx.x + thread_idx.x
    var n_threads_row = grid_dim.y * block_dim.y
    var n_threads_col = grid_dim.x * block_dim.x

    for i in range(global_row, m, n_threads_row) :
        for j in range(global_col, n, n_threads_col) :
            var sum = Scalar[DType.float64](0)
            @parameter
            if ak == ScalarKind.zero :
                @parameter 
                if bk == ScalarKind.zero :
                    C[i * ldc + j] = Scalar[DType.float64](0)
                elif bk == ScalarKind.gen :
                    C[i * ldc + j] = beta * C[i * ldc + j]
            else :
                @parameter
                if trans_a == Flag.true and trans_b == Flag.true :
                    for kk in range(k) :
                        sum += A[kk * lda + i] * B[j * ldb + kk]
                elif trans_a == Flag.true:
                    for kk in range(k) :
                        sum += A[kk * lda + i] * B[kk * ldb + j]
                elif trans_b == Flag.true:
                    for kk in range(k) :
                        sum += A[i * lda + kk] * B[j * ldb + kk]
                else :
                    for kk in range(k) :
                        sum += A[i * lda + kk] * B[kk * ldb + j]
                @parameter
                if ak == ScalarKind.one :
                    @parameter
                    if bk == ScalarKind.zero :
                        C[i * ldc + j] = sum
                    elif bk == ScalarKind.one :
                        C[i * ldc + j] = sum + C[i * ldc + j]
                    else :
                        C[i * ldc + j] = sum + beta * C[i * ldc + j]
                else:
                    @parameter
                    if bk == ScalarKind.zero : 
                        C[i * ldc + j] = alpha * sum
                    elif bk == ScalarKind.one :
                        C[i * ldc + j] = alpha * sum + C[i * ldc + j]
                    else :
                        C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j]

def launch_gemm[dtype: DType, ak: ScalarKind, bk: ScalarKind, ta: Flag, tb: Flag](    
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
    ctx: DeviceContext) :

    @parameter
    if dtype == DType.float32: 
        ctx.enqueue_function[sgemm_device[ta, tb, ak, bk], sgemm_device[ta, tb, ak, bk]](
        m, n, k,
        alpha, 
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc,
        grid_dim=(ceildiv(n, TBx), ceildiv(m, TBy)),
        block_dim=(TBx, TBy)
        )
    elif dtype == DType.float64 :
        ctx.enqueue_function[dgemm_device[ta, tb, ak, bk], dgemm_device[ta, tb, ak, bk]](
        m, n, k,
        alpha, 
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc,
        grid_dim=(ceildiv(n, TBx), ceildiv(m, TBy)),
        block_dim=(TBx, TBy)
        )
    else:
        raise Error("blas_gemm: Unsupported type")


def dispatch_transpose[dtype: DType, ak: ScalarKind, bk: ScalarKind](
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
    ctx: DeviceContext) :

    if trans_a and trans_b :
        launch_gemm[dtype, ak, bk, Flag.true, Flag.true](m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif trans_a :
        launch_gemm[dtype, ak, bk,Flag.true, Flag.false](m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif trans_b :
        launch_gemm[dtype, ak, bk,Flag.false, Flag.true](m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    else :
        launch_gemm[dtype, ak, bk, Flag.false, Flag.false](m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)



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


    #quick return
    if m == 0 or n == 0 or k == 0 : return

    var zero = Scalar[dtype](0)
    var one = Scalar[dtype](1)

    if alpha == zero and beta == zero : # C gets filled with zeros, but cant memset unsafe pointer to Device Buffer
        dispatch_transpose[dtype, ScalarKind.zero, ScalarKind.zero](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif alpha == zero and beta == one: # no op
        return
    elif alpha == zero :
        dispatch_transpose[dtype, ScalarKind.zero, ScalarKind.gen](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif alpha == one and beta == zero :
        dispatch_transpose[dtype, ScalarKind.one, ScalarKind.zero](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif alpha == one and beta == one :
        dispatch_transpose[dtype, ScalarKind.one, ScalarKind.one](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif alpha == one: 
        dispatch_transpose[dtype, ScalarKind.one, ScalarKind.gen](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif beta == zero :
        dispatch_transpose[dtype, ScalarKind.gen, ScalarKind.zero](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    elif beta == one :
        dispatch_transpose[dtype, ScalarKind.gen, ScalarKind.one](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)
    else :
        dispatch_transpose[dtype, ScalarKind.gen, ScalarKind.gen](trans_a, trans_b, m, n, k,alpha, d_A, lda,d_B, ldb, beta, d_C, ldc, ctx)


    # elif dtype == DType.float64:
    #     ctx.enqueue_function[dgemm_device, dgemm_device](
    #     trans_a_i, trans_b_i,
    #     m, n, k,
    #     alpha, 
    #     d_A, lda,
    #     d_B, ldb,
    #     beta,
    #     d_C, ldc,
    #     grid_dim=(ceildiv(n, TBx), ceildiv(m, TBy)),
    #     block_dim=(TBx, TBy)
    #     )
    # else:
    #     raise Error("blas_gemm: Unsupported type")

    ctx.synchronize()
