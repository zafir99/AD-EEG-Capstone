import sklearn
import numpy as np
from os import getcwd, makedirs, execv
from pathlib import Path


cwd = Path(getcwd())
data_root = cwd.parent / "processed"
alz_root = data_root / "alz"
con_root = data_root / "con"
ftd_root = data_root / "ftd"

if not (alz_root.exists() and con_root.exists() and ftd_root.exists()) :
    print("Processed dataset does not exist.\nNow running dsprocess.py...\n")
    execv("./dsprocess.py", [" "])

alz_index = (1,36)
con_index = (37,65)
ftd_index = (66,88)

num_a = alz_index[1]-alz_index[0]+1
num_c = con_index[1]-con_index[0]+1
num_f = ftd_index[1]-ftd_index[0]+1
num_subjects = 88
