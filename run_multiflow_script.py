from genericpath import exists
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from turb2d import RunMultiFlows
import time
import shutil
import pdb

# pdb.set_trace()

path = "C:\\Users\\Seiya\\Desktop\\test_turb2d"
dirname = "4eq_spacing10"
dirpath = os.path.join(path, dirname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

shutil.copy("C:\\Users\\Seiya\\Documents\\Python_Scripts\\turb2d_mixed_grain\\turb2d\\run_multiflow_script.py", dirpath)
shutil.copy("C:\\Users\\Seiya\\Documents\\Python_Scripts\\turb2d_mixed_grain\\turb2d\\turb2d\\run_flows.py", dirpath)

# import pdb
# ipdb.set_trace()
grain_class_num = 4
proc = 1  # number of processors to be used
num_runs = 10
Cmin, Cmax = [0.0001, 0.01]
rmin, rmax = [50., 200.]
# hmin, hmax = [0., .]
hmin, hmax = [25., 150.]
# pdb.set_trace()
C_ini=[]
for i in range(num_runs):
    C_ini.append(np.random.uniform(Cmin, Cmax, grain_class_num))
# C_ini = np.random.uniform(Cmin, Cmax, num_runs)
r_ini = np.random.uniform(rmin, rmax, num_runs)
h_ini = np.random.uniform(hmin, hmax, num_runs)

rmf = RunMultiFlows(
    dirpath,
    os.path.join(dirpath, 'num_3000.nc'),
    C_ini=C_ini,
    r_ini=r_ini,
    h_ini=r_ini,
    grain_class_num=grain_class_num,
    processors=proc,
    endtime=4000.0,
    timelimit=None
)
# pdb.set_trace()
rmf.create_datafile()
start = time.time()
print("start calculation")
rmf.run_multiple_flows()
print("elapsed time: {} sec.".format(time.time() - start))
