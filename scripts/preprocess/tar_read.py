import gzip
import pickle
import tarfile
import time
from joblib import Parallel, delayed
import os
from tqdm import tqdm


def read_one_from_tar(key, pkl_dir, tar=True):
    if tar:
        tar_handler = tarfile.open(tar_f, "r")
        pkl = tar_handler.extractfile(key)
    else:
        pkl = os.path.join(pkl_dir, f"{key}")
    f = gzip.open(pkl, "rb")
    data = pickle.load(f)
    return data


if __name__ == "__main__":
    tar_f = "./train.tar"
    # 1. read one cif
    t = time.time()
    one_cif = "ggaggtcgtaaatcattacc_1327.cif.pkl.gz"
    tar_handler = tarfile.open(tar_f, "r")
    pkl = tar_handler.extractfile(one_cif)
    with gzip.open(pkl, "rb") as f:
        data = pickle.load(f)
    print(f"data type:{type(data)}, time:{time.time() - t}")
    # 2. read all names
    t = time.time()
    names = tar_handler.getnames()
    print(f"Cost: {time.time() - t} seconds")
    print("n_names:", len(names))
    res = Parallel(n_jobs=60, verbose=1)(delayed(read_one_from_tar)(x, tar=True) for x in names[:300])
    print(f"Multi-process, n:{len(res)}")
    #
    tar_handler.close()
    pass
