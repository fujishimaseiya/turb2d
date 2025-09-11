"""This is a script to run the model of TurbidityCurrent2D
"""
import os
import shutil
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
import numpy as np
from turb2d.utils import create_topography, create_child_topography
from turb2d.utils import create_nested_grid, initialize_grid_fields, set_inlet_condition, set_boundary_condition
from landlab import RasterModelGrid
from turb2d import TurbidityCurrent2D
import time
from landlab import FieldError
# from landlab import FIXED_GRADIENT_BOUNDARY, FIXED_VALUE_BOUNDARY
import pdb
from tqdm import tqdm
import yaml
from turb2d.nesting import OneWayNesting, TwoWayNesting
import matplotlib.pyplot as plt

pdb.set_trace()
# create initial topography
parent_grid = create_topography(config_file="config_runturb2d_parent.yml")
child_grid, nested_region_idx = create_child_topography(parent_grid=parent_grid, config_file="config_runturb2d_child.yml")

# initialize grid fields
initialize_grid_fields(parent_grid, config_file="config_runturb2d_parent.yml")
initialize_grid_fields(child_grid, config_file="config_runturb2d_child.yml")

# set inlet condition
set_inlet_condition(grid=parent_grid, config_file="config_runturb2d_parent.yml", inlet_edge=None)

# set boundary condition of parent grid
set_boundary_condition(grid=parent_grid, 
                       config_file="config_runturb2d_parent.yml")
set_boundary_condition(grid=child_grid, 
                       config_file="config_runturb2d_child.yml")

# TurbidityCurrent2D objects
tc_parent = TurbidityCurrent2D(parent_grid, config_path="config_runturb2d_parent.yml",  parent_grid=True, child_grid=False)
tc_child = TurbidityCurrent2D(child_grid, config_path="config_runturb2d_child.yml", parent_grid=False, child_grid=True)

# plot nested region
fig, ax = plt.subplots()
topo = np.flipud(parent_grid.at_node['topographic__elevation'].reshape(parent_grid.shape))
im = ax.imshow(topo, cmap='viridis', extent=(0, np.max(parent_grid.node_x), np.max(parent_grid.node_y), 0))
fig.colorbar(im, ax=ax, label='Elevation [m]')
nested_region_x = parent_grid.node_x[nested_region_idx]
nested_region_y = parent_grid.node_y[nested_region_idx]
xmin = np.min(nested_region_x)
xmax = np.max(nested_region_x)
ymin = np.max(parent_grid.node_y) - np.min(nested_region_y)
ymax = np.max(parent_grid.node_y) - np.max(nested_region_y)
ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='red')
ax.set_xlabel('Width [m]')
ax.set_ylabel('Distance from inlet [m]')
plt.savefig('nested_region.png')

# start calculation
# In this calculation, only the calculation results of the child grid is saved.
t = time.time()
tc_parent.save_nc('tc{:04d}_parent.nc'.format(0))
tc_child.save_nc('tc{:04d}.nc'.format(0))
Ch_init = np.sum(tc_child.C * tc_child.h)
last = 432000
num = 1
num_repeat = 1
save_interval = 100
one_way_nesting = OneWayNesting(tc_parent=tc_parent, tc_child=tc_child, dt=1.0, num_relaxation_grid=5)

for j in range(num_repeat):
    for i in tqdm(range(1, last + 1), disable=False):
        # parent grid calculation 
        tc_parent.run_one_step(dt=1.0, repeat=j, last=i)
        # tc_parent.save_nc('tc{:04d}_parent.nc'.format(num))
        # calculate boundary conditions of a child grid from a parent grid
        one_way_nesting.compute_child_grid_condition_from_parent_grid(time_step=0.0001)
        tc_child.run_one_step(dt=1.0, repeat=j, last=i)
        if i % save_interval == 0:
            tc_child.save_nc('tc{:04d}.nc'.format(num))
        num = num + 1
    tc_child.save_grid('tc{:04d}.nc'.format(num-2))
print('elapsed time: {} sec.'.format(time.time() - t))
