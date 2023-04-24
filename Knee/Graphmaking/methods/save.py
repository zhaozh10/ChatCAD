import pickle

import numpy as np


def saveData(data):

    with open(data.save + ".pkl", "wb") as f:
        pickle.dump(data.graph, f)
        pickle.dump(data.v, f)

    np.savez_compressed(data.save + ".npz", patch=data.patch, bones=data.bones, pos=data.pos)

    return
