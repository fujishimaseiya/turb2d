"""This is a script to run the model of TurbidityCurrent2D
"""
import os
import shutil
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
import numpy as np
from turb2d.utils import create_topography
from turb2d.utils import create_init_flow_region, create_topography_from_geotiff
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
        
grid.status_at_node[grid.nodes_at_top_edge] = grid.BC_NODE_IS_FIXED_VALUE
grid.status_at_node[grid.nodes_at_bottom_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
grid.status_at_node[grid.nodes_at_left_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
grid.status_at_node[grid.nodes_at_right_edge] = grid.BC_NODE_IS_FIXED_GRADIENT


C_ini = [0.00005, 0.00005, 0.00005, 0.00005, 0.00005]
h_ini = 0.2
U_ini = 0.2
with open ('config_runturb2d.yml', 'r') as f:
    config = yaml.safe_load(f)
    flume_length = config['grid_param']['flume_length']
    flume_width = config['grid_param']['flume_width']
    inlet_width = config['grid_param']['inlet_width']
inlet_edge = [(flume_width - inlet_width) / 2.0, (flume_width + inlet_width) / 2.0]
# set inlet
inlet = np.where((grid.x_of_node >= inlet_edge[0])
                        & (grid.x_of_node <= inlet_edge[1]) & (grid.y_of_node == flume_length))
inlet_link = np.where((grid.midpoint_of_link[:,0] >= inlet_edge[0]) & (grid.midpoint_of_link[:,0] <= inlet_edge[1])
                        & (grid.midpoint_of_link[:,1] == flume_length))
grid.status_at_node[inlet] = grid.BC_NODE_IS_FIXED_VALUE

# check number of grain size classes
if type(C_ini) is float or type(C_ini) is np.float64:
    C_ini_i = np.array([C_ini])
else:
    C_ini_i = np.array(C_ini).reshape(
    len(C_ini), 1
)
# initialize flow parameters
for i in range(len(C_ini_i)):
    try:
        grid.add_zeros("flow__sediment_concentration_{}".format(i), at="node")
    except FieldError:
        grid.at_node["flow__sediment_concentration_{}".format(i)][:] = 0.0
    try:
        grid.add_zeros("bed__sediment_volume_per_unit_area_{}".format(i), at="node")
    except FieldError:
        grid.at_node["bed__sediment_volume_per_unit_area_{}".format(i)][:] = 0.0

try:
    grid.add_zeros("flow__sediment_concentration_total", at="node")
except FieldError:
    grid.at_node["flow__sediment_concentration_total"][:] = 0.0
try:
    grid.add_zeros("flow__depth", at="node")
except FieldError:
    grid.at_node["flow__depth"][:] = 0.0
try:
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
except FieldError:
    grid.at_node["flow__horizontal_velocity_at_node"][:] = 0.0
try:
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")
except FieldError:
    grid.at_node["flow__vertical_velocity_at_node"][:] = 0.0
try:
    grid.add_zeros("flow__horizontal_velocity", at="link")
except FieldError:
    grid.at_link["flow__horizontal_velocity"][:] = 0.0
try:
    grid.add_zeros("flow__vertical_velocity", at="link")
except FieldError:
    grid.at_link["flow__vertical_velocity"][:] = 0.0
    
# set condition at inlet
grid.at_node['flow__depth'][inlet] = h_ini
for i in range(len(C_ini_i)):
    grid.at_node['flow__sediment_concentration_{}'.format(i)][inlet] = C_ini_i[i]
grid.at_node['flow__horizontal_velocity_at_node'][inlet] = 0.0
grid.at_node['flow__vertical_velocity_at_node'][inlet] = -U_ini
grid.at_link['flow__horizontal_velocity'][inlet_link] = 0.0
grid.at_link['flow__vertical_velocity'][inlet_link] = -U_ini
grid.at_node["flow__sediment_concentration_total"][inlet] = np.sum(C_ini_i)

# grid = create_topography_from_geotiff('depth500.tif',
#                                       xlim=[200, 800],
#                                       ylim=[400, 1200],
#                                       spacing=500,
#                                       filter_size=[5, 5])

#grid.set_status_at_node_on_edges(top=grid.BC_NODE_IS_FIXED_GRADIENT,
#                                 bottom=grid.BC_NODE_IS_FIXED_GRADIENT,
#                                 right=grid.BC_NODE_IS_FIXED_GRADIENT,
#                                 left=grid.BC_NODE_IS_FIXED_GRADIENT)

# create_init_flow_region(
#     grid,
#     initial_flow_concentration=[0.01, 0.01, 0.01, 0.01],
#     initial_flow_thickness=0.5,
#     initial_region_radius=0.3,
#     initial_region_center=[1.1, 4.0],  # 1000, 4000
# )

# create_init_flow_region(
#     grid,
#     initial_flow_concentration=0.01,
#     initial_flow_thickness=200,
#     initial_region_radius=30000,
#     initial_region_center=[100000, 125000],
# )
# import pdb
# pdb.set_trace()
# making turbidity current object
# last element of Ds has no setting velocity (salt).
tc = TurbidityCurrent2D(grid,
                        config_path="config_runturb2d.yml")

# start calculation
t = time.time()
tc.save_nc('tc{:04d}.nc'.format(0))
Ch_init = np.sum(tc.C * tc.h)
last = 200
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

