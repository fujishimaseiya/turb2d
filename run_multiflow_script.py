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

# pdb.set_trace()
if __name__ ==  '__main__':
    path = "../training_data/exp2/3eq"
    dirname = "0.004_0.6_1.5_erosion"
    dirpath = os.path.join(path, dirname)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    shutil.copy("run_multiflow_script.py", dirpath)
    shutil.copy("turb2d/run_flows.py", dirpath)

    # import pdb
    # ipdb.set_trace()
    grain_class_num = 5 # melamin + salt
    proc = 10  # number of processors to be used
    num_runs = 5000
    Cmin, Cmax = [0.0001, 0.005]
    # rmin, rmax = [50., 200.]
    # hmin, hmax = [0., .]
    # hmin, hmax = [0.05, 0.5]
    Umin, Umax = [0.01, 0.5]
    endmin,endmax = [30, 300]
    saltmin, saltmax = [0.001, 0.05]
    # salt_conc = 0.0248
    C_ini=[]
    # pdb.set_trace()
    for i in range(num_runs):
        conc = np.random.uniform(Cmin, Cmax, 4)
        salt = np.array([np.random.uniform(saltmin, saltmax)])
        C_ini.append(np.append(conc, salt))
    # C_ini = np.random.uniform(Cmin, Cmax, num_runs)
    # r_ini = np.random.uniform(rmin, rmax, num_runs)
    # h_ini = np.random.uniform(hmin, hmax, num_runs)
    h_ini = np.full(num_runs, 0.2)
    U_ini =np.random.uniform(Umin, Umax, num_runs)
    endtime = np.random.randint(endmin, endmax, num_runs)
    np.savetxt(os.path.join(dirpath, 'C_ini.csv'), C_ini, delimiter=',')
    np.savetxt(os.path.join(dirpath, 'U_ini.csv'), U_ini, delimiter=',')
    np.savetxt(os.path.join(dirpath, 'h_ini.csv'), h_ini, delimiter=',')
    np.savetxt(os.path.join(dirpath, 'endtime.csv'), endtime, delimiter=',')

    rmf = RunMultiFlows(
        dirpath,
        os.path.join(dirpath, 'num_3000.nc'),
        C_ini=C_ini,
        # r_ini=r_ini,
        U_ini = U_ini,
        h_ini=h_ini,
        grain_class_num=grain_class_num,
        processors=proc,
        endtime=endtime,
        flow_type='continuous',
        timelimit=None,
        repeat=2
    )

    rmf.create_datafile()
    start = time.time()
    print("start calculation")
    rmf.run_multiple_flows()
    print("elapsed time: {} sec.".format(time.time() - start))
    calc_time = time.time() - start
    # np.savetxt(os.path.join(dirpath, 'elapsed_time.txt'), calc_time)
