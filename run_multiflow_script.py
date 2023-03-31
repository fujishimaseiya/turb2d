from genericpath import exists
import os
from timeit import repeat
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from turb2d import RunMultiFlows
import time
import shutil
import pdb
import yaml

# pdb.set_trace()
if __name__ ==  '__main__':
    dirpath = "."
    dirname = "yaml_test"
    savedir = os.path.join(dirpath, dirname)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    config_path = "config.yml"

    shutil.copy("run_multiflow_script.py", savedir)
    shutil.copy("turb2d/run_flows.py", savedir)


    rmf = RunMultiFlows(
        dirpath = savedir,
        filename = os.path.join(savedir, 'num_3000.nc'),
        config_file = 'config.yml'
    )

    rmf.create_datafile()
    start = time.time()
    print("start calculation")
    rmf.run_multiple_flows()
    print("elapsed time: {} sec.".format(time.time() - start))
    calc_time = time.time() - start
    # np.savetxt(os.path.join(dirpath, 'elapsed_time.txt'), calc_time)
