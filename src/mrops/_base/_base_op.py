"""Base operator."""

__all__ = ["BaseOperator", "IdentityOperator"]


class BaseOperator:
    """Base class to encapsulate methods."""
    
    def __init__(self):
        self.adj = None
        self.normal = None
    
    def __call__(self, input):
        return self._apply(input)
    
    def _apply(self, input):
        raise NotImplementedError

    def _adjoint_op(self):
        raise NotImplementedError

    def _normal_op(self):
        return _NormalOperator(self, self.H)
    
    @property
    def H(self):
        if self.adj is None:
            self.adj = self._adjoint_op()
        return self.adj

    @property
    def N(self):
        if self.normal is None:
            self.normal = self._normal_op()
        return self.normal
    

class _NormalOperator(BaseOperator):
    def __init__(self, fwd_op, adj_op):
        super().__init__()
        self._fwd_op = fwd_op
        self._adj_op = adj_op
        
    def _apply(self, input):
        return self._adj_op(self._fwd_op(input))
    
    def _adjoint_op(self):
        return self
    
    def _normal_op(self):
        return self


class IdentityOperator(BaseOperator):

    def __init__(self):
        super().__init__()
    
    def _apply(self, input):
        return input
    
    def _adjoint_op(self):
        return self
    
    def _normal_op(self):
        return self
