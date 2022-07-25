from genericpath import exists
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from turb2d import RunMultiFlows
import time
import shutil

path = "/home/biosphere/fujishima/training_data/mixed_grain/4eq"
dirname = "4eq_spacing10"
dirpath = os.path.join(path, dirname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

shutil.copy("/home/biosphere/fujishima/turb2d/run_multiflow_script.py", dirpath)
shutil.copy("/home/biosphere/fujishima/turb2d/turb2d/run_flows.py", dirpath)

# import pdb
# ipdb.set_trace()
grain_class_num = 4
proc = 15  # number of processors to be used
num_runs = 3000
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
    C_ini,
    r_ini,
    h_ini,
    dirpath,
    os.path.join(dirpath, 'num_3000.nc'),
    grain_class_num,
    processors=proc,
    endtime=4000.0,
    timelimit=None
)
rmf.create_datafile()
start = time.time()
rmf.run_multiple_flows()
print("elapsed time: {} sec.".format(time.time() - start))
