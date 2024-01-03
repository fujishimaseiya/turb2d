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
import netCDF4 as nc
import matplotlib.pyplot as plt

if __name__ ==  '__main__':
    config_file = "config_runmulti.yml"
    with open(config_file, 'r') as yml:
        config = yaml.safe_load(yml)
        dirpath = config['multi_param']['dirpath']
        filename = config['multi_param']['filename']
    savedir = os.path.join(dirpath, filename)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    shutil.copy("run_multiflow_script.py", savedir)
    shutil.copy("turb2d/run_flows.py", savedir)
    shutil.copy(config_file, savedir)
    shutil.copy("config_grid.yml", savedir)
    shutil.copy("config_turb2d.yml", savedir)

    rmf = RunMultiFlows(
        dirpath = savedir,
        filename = os.path.join(savedir, filename),
        turb2d_config_file='config_turb2d.yml',
        run_multi_config_file = 'config_runmulti.yml',
        grid_config_file='config_grid.yml'
    )

    rmf.create_datafile()
    start = time.time()
    print("start calculation")
    rmf.run_multiple_flows()
    print("elapsed time: {} sec.".format(time.time() - start))
    calc_time = time.time() - start
    # np.savetxt(os.path.join(dirpath, 'elapsed_time.txt'), calc_time)

    savedir_name = ["img_sed0", "img_sed1", "img_sed2", "img_sed3", "img_bedthick"]
    var_list = ["sed_volume_per_unit_area_0", "sed_volume_per_unit_area_1", "sed_volume_per_unit_area_2", "sed_volume_per_unit_area_3", "bed_thick"]
    savedir_list = []
    for i in savedir_name:
        imgdir = os.path.join(savedir, i)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)
        savedir_list.append(imgdir)
    read_num = 5
    fn = os.path.join(savedir, filename)
    ds = nc.Dataset(fn)
    for j in range(len(var_list)):
        for k in range(read_num):
            fig, ax = plt.subplots()
            im = ax.imshow(ds[var_list[j]][k], cmap="viridis")
            fig.colorbar(im, ax=ax)
            plt.savefig(os.path.join(savedir_list[j], "series_{}.jpg".format(k)))
            plt.close()