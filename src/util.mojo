
fn blas_error_if[caller: String, cond_str: String](cond: Bool) raises: 
    """
    Function raises an error describing the bad paramters passed to caller.
    """
    if(cond) :
        raise Error("Error: {} in {}".format(cond_str, caller))
    
