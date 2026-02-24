"""This is a script to run the model of TurbidityCurrent2D
"""
import os
import shutil
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
import numpy as np
from turb2d.utils import create_topography, initialize_grid_fields, set_inlet_condition, set_boundary_condition, set_inlet_region
from turb2d.utils import create_init_flow_region
from landlab import RasterModelGrid
from turb2d import TurbidityCurrent2D
import time
from landlab import FieldError
# from landlab import FIXED_GRADIENT_BOUNDARY, FIXED_VALUE_BOUNDARY
import pdb
from tqdm import tqdm
import yaml

grid = create_topography(
    config_file="config_runturb2d.yml"
        )

# initialize grid fields
initialize_grid_fields(grid, config_file="config_runturb2d.yml")

inlet, inlet_link = set_inlet_region(grid, config_file="config_runturb2d.yml")

# set inlet condition
set_inlet_condition(grid=grid, 
                    inlet=inlet,
                    inlet_link=inlet_link,
                    config_file="config_runturb2d.yml")

# set boundary condition of parent grid
set_boundary_condition(grid=grid, 
                       config_file="config_runturb2d.yml",
                       top_edge_bc=None,
                       bottom_edge_bc=None,
                       left_edge_bc=None, 
                       right_edge_bc=None)


tc = TurbidityCurrent2D(grid, config_path="config_runturb2d.yml")

# start calculation
t = time.time()
tc.save_nc('tc{:04d}.nc'.format(0))
Ch_init = np.sum(tc.C * tc.h)
last = 100
num = 1
for j in range(1):
    for i in tqdm(range(1, last + 1), disable=False):
        tc.run_one_step(dt=1.0, repeat=j, last=i)
        tc.save_nc('tc{:04d}.nc'.format(num))
        if np.sum(tc.C * tc.h) / Ch_init < 0.01:
            break
        num = num + 1
    tc.save_grid('tc{:04d}.nc'.format(num))
print('elapsed time: {} sec.'.format(time.time() - t))

