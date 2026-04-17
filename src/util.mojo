
fn blas_error_if[caller: String, cond_str: String](cond: Bool) raises: 
    """
    Function raises an error describing the bad paramters passed to caller.
    """
    if(cond) :
        raise Error("Error: {} in {}".format(cond_str, caller))

fn zero_device[dtype: DType](count: Int,arr: UnsafePointer[Scalar[dtype], MutAnyOrigin],) :
    """
    Kernel sets count elements of arr to 0
    Used when scalars == 0.
    """
    var global_i = global_idx.x
    var n_threads = grid_dim.x * block_dim.x
    for i in range(global_i, count, n_threads):
        arr[i] = Scalar[dtype](0)






    

