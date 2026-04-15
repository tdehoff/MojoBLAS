
fn blas_error_if[caller: String, cond_str: String](cond: Bool) raises: 
    """
    Function raises an error describing the bad paramters passed to caller.
    """
    if(cond) :
        raise Error("Error: {} in {}".format(cond_str, caller))
    
@fieldwise_init
struct ScalarKind(Copyable, Movable) :
    comptime zero = ScalarKind(0)
    comptime one = ScalarKind(1)
    comptime gen = ScalarKind(2)
    var _val: Int
    fn __eq__(self, other: ScalarKind) -> Bool :
        return self._val == other._val
    fn __ne__(self, other: ScalarKind) -> Bool :
        return self._val != other._val

@fieldwise_init
struct Flag(Copyable, Movable) :
    comptime false = Flag(0)
    comptime true = Flag(1)
    var _val : Int
    fn __eq__(self, other: Flag) -> Bool :
        return self._val == other._val
    fn __ne__(self, other: Flag) -> Bool :
        return self._val != other._val
