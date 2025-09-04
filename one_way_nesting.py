"""This is a script to run the model of TurbidityCurrent2D
"""
import os
import shutil
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
import numpy as np
from turb2d.utils import create_topography
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


# import config file
with open ('config_runturb2d.yml', 'r') as f:
    config = yaml.safe_load(f)
nested_region = [config['grid_param']['nested_region_xmin'], 
                config['grid_param']['nested_region_xmax'],
                config['grid_param']['nested_region_ymin'], 
                config['grid_param']['nested_region_ymax']]

# create initial topography
parent_grid, child_grid, nested_region_idx = create_nested_grid(config_file="config_runturb2d.yml")

# initialize grid fields
initialize_grid_fields(parent_grid, config_file="config_runturb2d.yml")
initialize_grid_fields(child_grid, config_file="config_runturb2d.yml")

# set inlet condition
set_inlet_condition(grid=parent_grid, config_file="config_runturb2d.yml", inlet_edge=None)

# set boundary condition of parent grid
set_boundary_condition(grid=parent_grid, 
                       top_edge_bc='fixed_value',
                       bottom_edge_bc='fixed_gradient',
                       left_edge_bc='fixed_gradient', 
                       right_edge_bc='fixed_gradient')
set_boundary_condition(grid=child_grid, 
                       top_edge_bc='fixed_value',
                       bottom_edge_bc='fixed_value',
                       left_edge_bc='fixed_value', 
                       right_edge_bc='fixed_value')

# TurbidityCurrent2D objects
tc_parent = TurbidityCurrent2D(parent_grid, config_path="config_runturb2d.yml",  parent_grid=True, child_grid=False)
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
last = 8640
num = 1
num_repeat = 1
save_interval = 100
one_way_nesting = OneWayNesting(tc_parent=tc_parent, tc_child=tc_child, dt=1.0, num_relaxation_grid=5)

for j in range(num_repeat):
    for i in tqdm(range(1, last + 1), disable=False):
        # parent grid calculation 
        tc_parent.run_one_step(dt=1.0, repeat=j, last=i)
        tc_parent.save_nc('tc{:04d}_parent.nc'.format(num))
        # calculate boundary conditions of a child grid from a parent grid
        one_way_nesting.compute_child_grid_condition_from_parent_grid(time_step=0.0001)
        tc_child.run_one_step(dt=1.0, repeat=j, last=i)
        if i % save_interval == 0:
            tc_child.save_nc('tc{:04d}.nc'.format(num))
        num = num + 1
    tc_child.save_grid('tc{:04d}.nc'.format(num-2))
print('elapsed time: {} sec.'.format(time.time() - t))
