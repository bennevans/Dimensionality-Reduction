

class DimensionReducer:
    def __init__(self, d, d_prime):
        self.d = d
        self.d_prime = d_prime
    
    def load(self, dir):
        """
        Loads a pre-made reducer
        """
        raise NotImplementedError
    
    def save(self, dir):
        """
        Saves a reducer
        """
        raise NotImplementedError

    def construct(self, dataset):
        """
        Constructs the reducer given a dataset
        """
        raise NotImplementedError

    def reduce_dim(self, x):
        """
        Reduces the dimension of a given point. x: [batch, d]
        """
        raise NotImplementedError
