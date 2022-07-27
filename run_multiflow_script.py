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

path = "/home/biosphere/fujishima/expscale_test/data"
dirname = "exp"
dirpath = os.path.join(path, dirname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

shutil.copy("/home/biosphere/fujishima/expscale_test/turb2d/run_multiflow_script.py", dirpath)
shutil.copy("/home/biosphere/fujishima/expscale_test/turb2d/turb2d/run_flows.py", dirpath)

# import pdb
# ipdb.set_trace()
grain_class_num = 4
proc = 1  # number of processors to be used
num_runs = 3000
Cmin, Cmax = [0.0001, 0.01]
rmin, rmax = [50., 200.]
# hmin, hmax = [0., .]
hmin, hmax = [0.01, 0.3]
Umin, Umax = [0.01, 0.2]
# pdb.set_trace()
C_ini=[]
for i in range(num_runs):
    C_ini.append(np.random.uniform(Cmin, Cmax, grain_class_num))
# C_ini = np.random.uniform(Cmin, Cmax, num_runs)
# r_ini = np.random.uniform(rmin, rmax, num_runs)
h_ini = np.random.uniform(hmin, hmax, num_runs)
U_ini =np.random.uniform(Umin, Umax, num_runs)

rmf = RunMultiFlows(
    dirpath,
    os.path.join(dirpath, 'num_3000.nc'),
    C_ini=C_ini,
    # r_ini=r_ini,
    U_ini = U_ini,
    h_ini=h_ini,
    grain_class_num=grain_class_num,
    processors=proc,
    endtime=8000.0,
    flow_type='continuous',
    timelimit=None
)
# pdb.set_trace()
rmf.create_datafile()
start = time.time()
print("start calculation")
rmf.run_multiple_flows()
print("elapsed time: {} sec.".format(time.time() - start))
