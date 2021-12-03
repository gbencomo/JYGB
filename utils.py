import numpy as np

class AttrDict(dict):
    """ subclass dict and define getter-setter. This behaves as both dict and obj"""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

def shuffle(D, axis=0):
    #fisher–Yates shuffle algorithm
    assert isinstance(D, np.ndarray), "Not a numpy array!"
    assert axis <= D.ndim, "Axis out-of-range."

    # TODO: fix so that you can shuffle on any dimension,
    # we will assert for now!
    assert axis == 0, "Shuffle is only compatiable with first dimension ... for now"
    
    for i in range(D.shape[axis]):
        j = np.random.randint(0, i + 1)
        D[i], D[j] = D[j], D[i]

    return D
        
    
