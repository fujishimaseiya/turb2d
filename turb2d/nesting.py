import numpy as np
import pdb
import re
import netCDF4 as nc
from decimal import Decimal
from scipy.interpolate import interp1d
from turb2d._links import top_edge_vertical_ids, bottom_edge_vertical_ids, left_edge_horizontal_ids, right_edge_horizontal_ids, top_edge_horizontal_ids, bottom_edge_horizontal_ids, left_edge_vertical_ids, right_edge_vertical_ids
from landlab import RasterModelGrid
from scipy.interpolate import griddata
from .gridutils import map_mean_of_link_nodes_to_link
from ._links import vertical_link_ids, horizontal_link_ids
import time
import sys

class OneWayNesting():
    """Class for one-way nesting."""

    def __init__(self, tc_parent, tc_child, nested_region, parent_grid_file=None, child_grid_file=None, dt=1.0):
        """Initialize the OneWayNesting class.

        Parameters
        ----------
        tc_parent : TurbidityCurrent2D
            Parent grid object.
        tc_child : TurbidityCurrent2D
            Child grid object.
        nested_region : list
            List containing the coordinates of the nested region in the format [xmin, xmax, ymin, ymax].
        parent_grid_file : str, optional
            Path to the NetCDF file of the parent grid. If None, nesting is performed using the tc_parent object.
            Default is None.
        child_grid_file : str, optional
            Path to the NetCDF file of the child grid. If None, nesting is performed using the tc_parent object.
            Default is None.
        dt : float, optional
            Total duration of the data to interpolate, in seconds. Default is 1.0.
        """
        self.tc_parent = tc_parent
        self.tc_child = tc_child
        self.nested_region = nested_region
        self.parent_grid_file = parent_grid_file
        self.child_grid_file = child_grid_file
        self.dt = dt
        
    def interp_griddata(self, x, y, parent_values, x_new, y_new, interp_method='linear'):
        """Interpolate the parent grid data for the initial and boundary conditions of the child grid.

        Parameters
        ----------
        x : ndarray
            x-coordinates of nested region in the parent grid data
        y : ndarray
            y-coodinates of nested region in the parent grid data
        parent_values : ndarray
            values at the nested region in the parent grid data
        x_new : ndarray
            x-coordiates of the child grid data
        intep_method : str, optional
            Interpolation method to use. Defualt is 'linear'. Other options include 'nearest' and 'cubic'.
        
        Returns
        -------
        y_new : ndarray
            Interpolated values at the child grid.

        """
        xy_cood = np.array([x, y]).T
        xx, yy = np.meshgrid(x_new, y_new)
        interp_values = griddata(points=xy_cood, values=parent_values, xi=(xx, yy), method=interp_method)

        y_new = interp_values.flatten()

        return y_new
    
    def temporal_interp(self, node_values, link_values, time_interp):
        """Interpolate the node and link values linearly over time steps.

        Parameters
        ----------
        node_values : ndarray
            Array of node values at different time steps. Shape should be (time_steps, number_of_nodes).
        link_values : ndarray, optional
            Array of link values at different time steps. Shape should be (time_steps, number_of_links). Default is None.
        time_interp : ndarray
            Array of time steps for interpolation. Should be of shape (time_steps,).

        Returns
        -------
        node_values : ndarray
            Interpolated node values at the specified time steps.
        link_values : ndarray, optional
            Interpolated link values at the specified time steps. Will be None if link_values is None.
        """

        t0, t1 = time_interp[0], time_interp[-1]
        t = time_interp[1:-1]
        alpha = (t - t0) / (t1 - t0)

        node_start, node_end = node_values[0, :], node_values[-1, :]
        node_values[1:-1, :] = (1 - alpha[:, None]) * node_start + alpha[:, None] * node_end

        if link_values is not None:
            link_start, link_end = link_values[0, :], link_values[-1, :]
            link_values[1:-1, :] = (1 - alpha[:, None]) * link_start + alpha[:, None] * link_end

        return node_values, link_values
    

    def round_values(self, x, decimals=0):
        """Round a number to a specified number of decimal places.
        Parameters
        ----------
        x : float or ndarray
            The number or array of numbers to round.
        decimals : int, optional
            The number of decimal places to round to. Default is 0.

        Returns
        -------
        rounded_value float or ndarray
            The rounded number or array of numbers.
        """
        rounded_value = np.floor(x * 10**decimals + 0.5) / 10**decimals

        return rounded_value

    def interpolate_parent_to_child_grid(self, time_interp, variable_start, variable_end, nested_idx, spatial_interp_method='linear'):
        """Interpolate the result of the parent calculation to obtain the initial and boundary conditions for the child grid.
        
        Parameters
        ----------
        time_interp : ndarray
            Array of time steps for interpolation.
        variable_start : ndarray
            Variable at initial time to be interpolated to obtain the boundary and initial conditions.
        variable_end : ndarray
            Variable at final time to be interpolated to obtain the boundary and initial conditions.
        nested_idx : ndarray
            Indices of the nested region in the parent grid.
        spatial_interp_method : str, optional
            Interpolation method to use for spatial interpolation. Default is 'linear'. Other options include 'nearest' and 'cubic'.

        Returns
        -------
        interp_values : ndarray
            Interpolated values at the child grid nodes for the specified variable at the initial and final time steps.
        """

        values_at_start = variable_start[nested_idx]
        values_at_end = variable_end[nested_idx]
        interp_values = np.zeros((time_interp.size, self.tc_child.grid.number_of_nodes))

        # remove the error due to floating point precision
        parent_grid_spacing = self.tc_parent.grid.spacing[0]
        s = str(parent_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        parent_x = round_values(self.tc_parent.grid.node_x[nested_idx], decimals=decimal)
        parent_y = round_values(self.tc_parent.grid.node_y[nested_idx], decimals=decimal)

        child_grid_spacing = self.tc_child.grid.spacing[0]
        s = str(child_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        child_x = round_values(self.tc_child.grid.node_x, decimals=decimal)
        child_y = round_values(self.tc_child.grid.node_y, decimals=decimal)

        # get indices of child grid nodes that correspond to the nested region in the parent grid
        idx_match_parent_child_grid = np.zeros_like(parent_x, dtype=int)
        for i in range(parent_x.size):
            idx_match_parent_child_grid[i] = np.where((child_x == parent_x[i]) & (child_y == parent_y[i]))[0]

        # spatial intepolation of the values at the nested region to compute the conditions at the initial and final time
        x = child_x[idx_match_parent_child_grid]
        y = child_y[idx_match_parent_child_grid]
        x_new = np.unique(child_x)
        y_new = np.unique(child_y)
        interp_values[0, :] = interp_griddata(x=x, y=y, parent_values=values_at_start, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)
        interp_values[-1, :] = interp_griddata(x=x, y=y, parent_values=values_at_end, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)
        
        # temporal interpolation to calculate the initial and boundary conditions at each time steps
        interp_values, _ = temporal_interp(node_values=interp_values, link_values=None, time_interp=time_interp)

        return interp_values

    def compute_child_grid_condition_from_parent_grid(self, time_step):
        """Compute initial and boundary conditions of a child calculation from a parent calculation.

        Parameters
        ----------

        time_step : float
            Time step for interpolation. Time steps must be given in seconds.
        first_calc : bool, optional
            Flag to indicate if this is the first calculation. Default is True.
        """

        xmin, xmax, ymin, ymax = self.nested_region
        # The extraction of regions by self.nested_region does not work well due to floating point precision.
        # Therefore, the coordinates of the parent grid are rounded to a specified number of decimal places.
        parent_node_x = self.round_values(self.tc_parent.grid.node_x, decimals=5)
        parent_node_y = self.round_values(self.tc_parent.grid.node_y, decimals=5)
        nested_region_idx = np.where(
                                    (parent_node_x >= xmin) & 
                                    (parent_node_x <= xmax) & 
                                    (parent_node_y >= ymin) & 
                                    (parent_node_y <= ymax)
                                    )
        
        # time step for interpolation
        self.tc_child.time_interp = np.arange(0.0, self.dt, time_step)

        # initialize arrays for the conditions of the child grid
        self.tc_child.u_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        # tc_child.u_link_child_grid_condition = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))
        self.tc_child.v_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        # tc_child.v_link_child_grid_condition = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))

        self.tc_child.h_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        # tc_child.h_link_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))

        gsize = self.tc_parent.C_i.shape[0]
        self.tc_child.C_i_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, gsize, self.tc_child.grid.number_of_nodes))

        self.tc_child.Kh_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        # tc_child.Kh_link_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))

        self.tc_child.bed_thick_i_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, gsize, self.tc_child.grid.number_of_nodes))
        self.tc_child.bed_thick_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))

        # interpolate the values of the parent grid to generated the initial and boundary conditions for the child grid
        # NOTE: The link values of u, v, and Kh are calculated within the function `map_values()`, which is called during `run_one_step()`.
        self.tc_child.u_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                                variable_start=self.tc_parent.u_node_ini,
                                                                                                variable_end=self.tc_parent.u_node,     
                                                                                                nested_idx=nested_region_idx[0],
                                                                                                spatial_interp_method='linear')

        self.tc_child.v_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp, 
                                                                                                variable_start=self.tc_parent.v_node_ini,
                                                                                                variable_end=self.tc_parent.v_node,     
                                                                                                nested_idx=nested_region_idx[0],
                                                                                                spatial_interp_method='linear')

        self.tc_child.h_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                      variable_start=self.tc_parent.h_ini,
                                                                                      variable_end=self.tc_parent.h,     
                                                                                      nested_idx=nested_region_idx[0],
                                                                                      spatial_interp_method='linear')

        self.tc_child.bed_thick_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                              variable_start=self.tc_parent.bed_thick_ini,
                                                                                              variable_end=self.tc_parent.bed_thick,     
                                                                                              nested_idx=nested_region_idx[0],
                                                                                              spatial_interp_method='linear')

        for i in range(gsize):
            self.tc_child.C_i_node_child_grid_condition[:, i, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                               variable_start=self.tc_parent.C_i_ini[i, :],
                                                                                               variable_end=self.tc_parent.C_i[i, :],     
                                                                                               nested_idx=nested_region_idx[0],
                                                                                               spatial_interp_method='linear')

            self.tc_child.bed_thick_i_node_child_grid_condition[:, i, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                                       variable_start=self.tc_parent.bed_thick_i[i, :],
                                                                                                       variable_end=self.tc_parent.bed_thick_i[i, :],     
                                                                                                       nested_idx=nested_region_idx[0],
                                                                                                       spatial_interp_method='linear')

        if self.tc_child.model == '4eq' and self.tc_parent.model == '4eq':
            self.tc_child.Kh_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                           variable_start=self.tc_parent.Kh_node_ini,
                                                                                           variable_end=self.tc_parent.Kh_node,     
                                                                                           nested_idx=nested_region_idx[0],
                                                                                           spatial_interp_method='linear')
        elif (self.tc_child.model == '4eq' and self.tc_parent.model != '4eq') or (self.tc_child.model != '4eq' and self.tc_parent.model == '4eq'):
            raise ValueError("The parent grid model must be '4eq' to compute the Kh values for the child grid.")

def interp_griddata(x, y, parent_values, x_new, y_new, interp_method='linear'):
    """Interpolate the parent grid data for the initial and boundary conditions of the child grid.

    Parameters
    ----------
    x : ndarray
        x-coordinates of nested region in the parent grid data
    y : ndarray
        y-coodinates of nested region in the parent grid data
    parent_values : ndarray
        values at the nested region in the parent grid data
    x_new : ndarray
        x-coordiates of the child grid data
    intep_method : str, optional
        Interpolation method to use. Defualt is 'linear'. Other options include 'nearest' and 'cubic'.
    
    Returns
    -------
    y_new : ndarray
        Interpolated values at the child grid.

    """
    xy_cood = np.array([x, y]).T
    xx, yy = np.meshgrid(x_new, y_new)
    interp_values = griddata(points=xy_cood, values=parent_values, xi=(xx, yy), method=interp_method)

    y_new = interp_values.flatten()

    return y_new

def temporal_interp(node_values, link_values, time_interp):
    """Interpolate the node and link values linearly over time steps using the property of internal division.

    Parameters
    ----------
    node_values : ndarray
        Array of node values at different time steps. Shape should be (time_steps, number_of_nodes).
    link_values : ndarray, optional
        Array of link values at different time steps. Shape should be (time_steps, number_of_links). Default is None.
    time_interp : ndarray
        Array of time steps for interpolation. Should be of shape (time_steps,).

    Returns
    -------
    node_values : ndarray
        Interpolated node values at the specified time steps.
    link_values : ndarray, optional
        Interpolated link values at the specified time steps. Will be None if link_values is None.
    """

    t0, t1 = time_interp[0], time_interp[-1]
    t = time_interp[1:-1]
    alpha = (t - t0) / (t1 - t0)

    node_start, node_end = node_values[0, :], node_values[-1, :]
    node_values[1:-1, :] = (1 - alpha[:, None]) * node_start + alpha[:, None] * node_end

    if link_values is not None:
        link_start, link_end = link_values[0, :], link_values[-1, :]
        link_values[1:-1, :] = (1 - alpha[:, None]) * link_start + alpha[:, None] * link_end

    return node_values, link_values

def round_values(x, decimals=0):
    """Round a number to a specified number of decimal places.
    Parameters
    ----------
    x : float or ndarray
        The number or array of numbers to round.
    decimals : int, optional
        The number of decimal places to round to. Default is 0.

    Returns
    -------
    rounded_value float or ndarray
        The rounded number or array of numbers.
    """
    rounded_value = np.floor(x * 10**decimals + 0.5) / 10**decimals

    return rounded_value

def compute_child_grid_conditions(grid_start, grid_end, time_interp, child_grid, variable_name, nested_idx, spatial_interp_method='linear'):
    """Compute the initial and boundary conditions for the child grid from the parent grid data.
    
    Parameters
    ----------
    grid_start : RasterModelGrid
        Parent grid at the initial time step.
    grid_end : RasterModelGrid
        Parent grid at the final time step.
    time_interp : ndarray
        Array of time steps for interpolation.
    child_grid : RasterModelGrid
        Child grid for which the conditions are to be computed.
    variable_name : str
        Name of the variable to be interpolated (e.g., 'flow__horizontal_velocity_at_node').
    nested_idx : ndarray
        Indices of the nested region in the parent grid.
    spatial_interp_method : str, optional
        Interpolation method to use for spatial interpolation. Default is 'linear'. Other options include 'nearest' and 'cubic'.

    Returns
    -------
    interp_values : ndarray
        Interpolated values at the child grid nodes for the specified variable at the initial and final time steps.
    """
    values_at_start = grid_start.at_node[variable_name][nested_idx]
    values_at_end = grid_end.at_node[variable_name][nested_idx]
    interp_values = np.zeros((time_interp.size, child_grid.number_of_nodes))

    parent_x = grid_start.node_x[nested_idx]
    parent_y = grid_start.node_y[nested_idx]

    child_grid_spacing = child_grid.spacing[0]
    s = str(child_grid_spacing)
    _, s_d = s.split('.')
    decimal = len(s_d)
    child_x = round_values(child_grid.node_x, decimals=decimal)
    child_y = round_values(child_grid.node_y, decimals=decimal)

    # get indices of child grid nodes that correspond to the nested region in the parent grid
    idx_match_parent_child_grid = np.zeros_like(parent_x, dtype=int)
    for i in range(parent_x.size):
        idx_match_parent_child_grid[i] = np.where(
            (child_x == parent_x[i]) & (child_y == parent_y[i])
        )[0]
    
    # replace the child grid values by the values of parent grid at nodes matching the parent and child grid coordinates
    # interp_values[0, idx_match_parent_child_grid] = values_at_start
    # interp_values[-1, idx_match_parent_child_grid] = values_at_end

    # spatial intepolation of the values at the nested region to compute the conditions at the initial and final time
    x = child_x[idx_match_parent_child_grid]
    y = child_y[idx_match_parent_child_grid]
    x_new = np.unique(child_x)
    y_new = np.unique(child_y)
    interp_values[0, :] = interp_griddata(x=x, y=y, parent_values=values_at_start, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)
    interp_values[-1, :] = interp_griddata(x=x, y=y, parent_values=values_at_end, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)
    
    # temporal interpolation to calculate the initial and boundary conditions at each time steps
    interp_values, _ = temporal_interp(node_values=interp_values, link_values=None, time_interp=time_interp)

    return interp_values

def compute_child_grid_condition_from_parent_grid(parent_grid_file_start, parent_grid_file_end, tc_parent, tc_child, nested_region, dt, first_calc=True):
    """compute boundary conditions from calculation resut of parent grid to child grid

    Parameters
    ----------
    parent_grid_file_start : str or None
        Path to the NetCDF file of the parent grid at the start time.
        If you want to compute the boundary and initial conditions of child grid at a user-specified time,
        parent_grid_file_start and parent_grid_file_end should be set to None.

    parent_grid_file_end : str, None
        Path to the NetCDF file of the parent grid at the end time.
        If you want to compute the boundary and initial conditions of child grid at a user-specified time,
        parent_grid_file_start and parent_grid_file_end should be set to None.

    tc_parent : Object or None
        TurbidityCurrent2D object for parent grid.
        If you want to compute the child grid conditions from nc files, this should be set to None.

    tc_child : Object
        TurbidityCurrent2D object for child grid.

    nested_region : list
        List containing the coordinates of the nested region in the format [xmin, xmax, ymin, ymax].

    dt : float
        Number of seconds over which to interpolate the data. This should match the storage interval of the parent grid.

    first_calc : bool, optional
        Flag to indicate if this is the first calculation. Default is True.
    """
    pdb.set_trace()
    start = time.time()
        
    # Extract nested region indices
    xmin, xmax, ymin, ymax = nested_region
    parent_grid_start = nc_to_landlab(parent_grid_file_start)
    parent_grid_end = nc_to_landlab(parent_grid_file_end)

    nested_region_idx = np.where(
                                (parent_grid_start.node_x >= xmin) & 
                                (parent_grid_start.node_x <= xmax) & 
                                (parent_grid_start.node_y >= ymin) & 
                                (parent_grid_start.node_y <= ymax)
                                )
    
    # time step for interpolation
    # TODO: time step should be determied by local_dt
    tc_child.time_interp = np.arange(0.0, dt, 1.0*10**-5)

    # initialize arrays for the conditions of the child grid
    tc_child.u_node_child_grid_condition = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_nodes))
    # tc_child.u_link_child_grid_condition = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))
    tc_child.v_node_child_grid_condition = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_nodes))
    # tc_child.v_link_child_grid_condition = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))

    tc_child.h_node_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_nodes))
    # tc_child.h_link_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))

    # count number of sediment concentration variables
    pattern = re.compile(r'^flow__sediment_concentration_\d+$')
    gsize = 0
    for name in parent_grid_start.at_node:
        if pattern.match(name):
            gsize += 1
    tc_child.C_i_node_child_grid = np.zeros((tc_child.time_interp.size, gsize, tc_child.grid.number_of_nodes))

    tc_child.Kh_node_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_nodes))
    # tc_child.Kh_link_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_links))

    tc_child.bed_thick_i_node_child_grid = np.zeros((tc_child.time_interp.size, gsize, tc_child.grid.number_of_nodes))
    tc_child.bed_thick_node_child_grid = np.zeros((tc_child.time_interp.size, tc_child.grid.number_of_nodes))

    # interpolate the values of the parent grid to generated the initial and boundary conditions for the child grid
    # NOTE: The link values of u, v, and Kh are calculated within the function `map_values()`, which is called during `run_one_step()`.
    tc_child.u_node_child_grid_condition[:, :] = compute_child_grid_conditions(grid_start=parent_grid_start, 
                                                                               grid_end=parent_grid_end, 
                                                                               time_interp=tc_child.time_interp, 
                                                                               child_grid=tc_child.grid, 
                                                                               variable_name='flow__horizontal_velocity_at_node', 
                                                                               nested_idx=nested_region_idx[0])

    tc_child.v_node_child_grid_condition[:, :] = compute_child_grid_conditions(grid_start=parent_grid_start, 
                                                                               grid_end=parent_grid_end, 
                                                                               time_interp=tc_child.time_interp, 
                                                                               child_grid=tc_child.grid, 
                                                                               variable_name='flow__vertical_velocity_at_node', 
                                                                               nested_idx=nested_region_idx[0])

    tc_child.h_node_child_grid_condition[:, :] = compute_child_grid_conditions(grid_start=parent_grid_start, 
                                                                               grid_end=parent_grid_end, 
                                                                               time_interp=tc_child.time_interp, 
                                                                               child_grid=tc_child.grid, 
                                                                               variable_name='flow__depth', 
                                                                               nested_idx=nested_region_idx[0])

    tc_child.bed_thick_node_child_grid_condition[:, :] = compute_child_grid_conditions(grid_start=parent_grid_start,
                                                                                       grid_end=parent_grid_end, 
                                                                                       time_interp=tc_child.time_interp, 
                                                                                       child_grid=tc_child.grid, 
                                                                                       variable_name='bed__thickness', 
                                                                                       nested_idx=nested_region_idx[0])

    for i in range(gsize):
        tc_child.C_i_node_child_grid_condition[:, i, :] = compute_child_grid_conditions(grid_start=parent_grid_start, 
                                                                                        grid_end=parent_grid_end, 
                                                                                        time_interp=tc_child.time_interp, 
                                                                                        child_grid=tc_child.grid, 
                                                                                        variable_name=f'flow__sediment_concentration_{i}', 
                                                                                        nested_idx=nested_region_idx[0])

        tc_child.bed_thick_i_node_child_grid_condition[:, i, :] = compute_child_grid_conditions(grid_start=parent_grid_start, 
                                                                                                grid_end=parent_grid_end, 
                                                                                                time_interp=tc_child.time_interp, 
                                                                                                child_grid=tc_child.grid, 
                                                                                                variable_name=f'bed__sediment_volume_per_unit_area_{i}', 
                                                                                                nested_idx=nested_region_idx[0])

    if tc_child.model == '4eq':
        tc_child.Kh_node_child_grid_condition[:, :] = compute_child_grid_conditions(grid_start=parent_grid_start, 
                                                                                    grid_end=parent_grid_end, 
                                                                                    time_interp=tc_child.time_interp, 
                                                                                    child_grid=tc_child.grid, 
                                                                                    variable_name='flow__TKE_at_node', 
                                                                                    nested_idx=nested_region_idx[0])
    end = time.time()
    print(f"calculation time is {end - start} seconds")

    # linkの計算はrun_one_stepで行われるのでここでは必要ないが，u, v, khのみはnodeからlinkの値を計算するようにする必要あり．
    # map_node_to_linkでu, v, khも取り扱えるようにして，ここで計算する．run one stepでやると，link to nodeの計算があるので，ややこしい
    # もしかしたら，map_mean_of_link_nodes_to_linkをここに持ってくるだけでいいかも． 
    # get indices of nested boundary nodes in the parent grid
    # nested_right_boundary_nodes_parent = np.where((parent_grid_start.node_x == xmax) &
    #                                (parent_grid_start.node_y >= ymin) & 
    #                                (parent_grid_start.node_y <= ymax)
    #                                )
    # nested_left_boundary_nodes_parent = np.where((parent_grid_start.node_x == xmin) &
    #                                (parent_grid_start.node_y >= ymin) & 
    #                                (parent_grid_start.node_y <= ymax)
    #                                )
    # nested_top_boundary_nodes_parent = np.where((parent_grid_start.node_y == ymax) &
    #                                (parent_grid_start.node_x >= xmin) & 
    #                                (parent_grid_start.node_x <= xmax)
    #                                )
    # nested_bottom_boundary_nodes_parent = np.where((parent_grid_start.node_y == ymin) &
    #                                (parent_grid_start.node_x >= xmin) & 
    #                                (parent_grid_start.node_x <= xmax)
    #                                )
    # get indices of nested boundary nodes in the child grid
    # top_boundary_nodes_child= tc_child.grid.nodes_at_top_edge
    # bottom_boundary_nodes_child = tc_child.grid.nodes_at_bottom_edge
    # left_boundary_nodes_child = tc_child.grid.nodes_at_left_edge
    # right_boundary_nodes_child = tc_child.grid.nodes_at_right_edge
    
    if first_calc is True:
        # get indices of nested boundary links in the child grid
        tc_child.top_edge_horizontal_links_child = top_edge_horizontal_ids(tc_child.grid.shape)
        tc_child.bottom_edge_horizontal_links_child = bottom_edge_horizontal_ids(tc_child.grid.shape)
        tc_child.left_edge_vertical_links_child = left_edge_vertical_ids(tc_child.grid.shape)
        tc_child.right_edge_vertical_links_child = right_edge_vertical_ids(tc_child.grid.shape)

    # time step for interpolation
    # NOTE: Is time step correct?
    tc_child.time_interp = np.arange(0.0, dt, 1.0*10**-5)

    # initialize arrays for boundary conditions
    tc_child.u_node_top = np.zeros((tc_child.time_interp.size, top_boundary_nodes_child.size))
    tc_child.u_node_bottom = np.zeros((tc_child.time_interp.size, bottom_boundary_nodes_child.size))
    tc_child.u_node_left = np.zeros((tc_child.time_interp.size, left_boundary_nodes_child.size))
    tc_child.u_node_right = np.zeros((tc_child.time_interp.size, right_boundary_nodes_child.size))
    tc_child.u_link_top = np.zeros((tc_child.time_interp.size, tc_child.top_edge_horizontal_links_child.size))
    tc_child.u_link_bottom = np.zeros((tc_child.time_interp.size, tc_child.bottom_edge_horizontal_links_child.size))
    tc_child.u_link_left = np.zeros((tc_child.time_interp.size, tc_child.left_edge_vertical_links_child.size))
    tc_child.u_link_right = np.zeros((tc_child.time_interp.size, tc_child.right_edge_vertical_links_child.size))

    tc_child.v_node_top = np.zeros((tc_child.time_interp.size, top_boundary_nodes_child.size))
    tc_child.v_node_bottom = np.zeros((tc_child.time_interp.size, bottom_boundary_nodes_child.size))
    tc_child.v_node_left = np.zeros((tc_child.time_interp.size, left_boundary_nodes_child.size))
    tc_child.v_node_right = np.zeros((tc_child.time_interp.size, right_boundary_nodes_child.size))
    tc_child.v_link_top = np.zeros((tc_child.time_interp.size, tc_child.top_edge_horizontal_links_child.size))
    tc_child.v_link_bottom = np.zeros((tc_child.time_interp.size, tc_child.bottom_edge_horizontal_links_child.size))
    tc_child.v_link_left = np.zeros((tc_child.time_interp.size, tc_child.left_edge_vertical_links_child.size))
    tc_child.v_link_right = np.zeros((tc_child.time_interp.size, tc_child.right_edge_vertical_links_child.size))

    tc_child.h_node_top = np.zeros((tc_child.time_interp.size, top_boundary_nodes_child.size))
    tc_child.h_node_bottom = np.zeros((tc_child.time_interp.size, bottom_boundary_nodes_child.size))
    tc_child.h_node_left = np.zeros((tc_child.time_interp.size, left_boundary_nodes_child.size))
    tc_child.h_node_right = np.zeros((tc_child.time_interp.size, right_boundary_nodes_child.size))
    tc_child.h_link_top = np.zeros((tc_child.time_interp.size, tc_child.top_edge_horizontal_links_child.size))
    tc_child.h_link_bottom = np.zeros((tc_child.time_interp.size, tc_child.bottom_edge_horizontal_links_child.size))
    tc_child.h_link_left = np.zeros((tc_child.time_interp.size, tc_child.left_edge_vertical_links_child.size))
    tc_child.h_link_right = np.zeros((tc_child.time_interp.size, tc_child.right_edge_vertical_links_child.size))
    # count number of sediment concentration variables
    pattern = re.compile(r'^flow__sediment_concentration_\d+$')
    gsize = 0
    for name in parent_grid_start.at_node:
        if pattern.match(name):
            gsize += 1
    tc_child.C_i_node_top = np.zeros((tc_child.time_interp.size, gsize, top_boundary_nodes_child.size))
    tc_child.C_i_node_bottom = np.zeros((tc_child.time_interp.size, gsize, bottom_boundary_nodes_child.size))
    tc_child.C_i_node_left = np.zeros((tc_child.time_interp.size, gsize, left_boundary_nodes_child.size))
    tc_child.C_i_node_right = np.zeros((tc_child.time_interp.size, gsize, right_boundary_nodes_child.size))

    tc_child.bed_thick_i_node_top = np.zeros((tc_child.time_interp.size, gsize, top_boundary_nodes_child.size))
    tc_child.bed_thick_i_node_bottom = np.zeros((tc_child.time_interp.size, gsize, bottom_boundary_nodes_child.size))
    tc_child.bed_thick_i_node_left = np.zeros((tc_child.time_interp.size, gsize, left_boundary_nodes_child.size))
    tc_child.bed_thick_i_node_right = np.zeros((tc_child.time_interp.size, gsize, right_boundary_nodes_child.size))

    tc_child.bed_thick_node_top = np.zeros((tc_child.time_interp.size, top_boundary_nodes_child.size))
    tc_child.bed_thick_node_bottom = np.zeros((tc_child.time_interp.size, bottom_boundary_nodes_child.size))
    tc_child.bed_thick_node_left = np.zeros((tc_child.time_interp.size, left_boundary_nodes_child.size))
    tc_child.bed_thick_node_right = np.zeros((tc_child.time_interp.size, right_boundary_nodes_child.size))

    tc_child.Kh_node_top = np.zeros((tc_child.time_interp.size, top_boundary_nodes_child.size))
    tc_child.Kh_node_bottom = np.zeros((tc_child.time_interp.size, bottom_boundary_nodes_child.size))
    tc_child.Kh_node_left = np.zeros((tc_child.time_interp.size, left_boundary_nodes_child.size))
    tc_child.Kh_node_right = np.zeros((tc_child.time_interp.size, right_boundary_nodes_child.size))
    tc_child.Kh_link_top = np.zeros((tc_child.time_interp.size, tc_child.top_edge_horizontal_links_child.size))
    tc_child.Kh_link_bottom = np.zeros((tc_child.time_interp.size, tc_child.bottom_edge_horizontal_links_child.size))
    tc_child.Kh_link_left = np.zeros((tc_child.time_interp.size, tc_child.left_edge_vertical_links_child.size))
    tc_child.Kh_link_right = np.zeros((tc_child.time_interp.size, tc_child.right_edge_vertical_links_child.size))

    # calculate boundary conditions for child grid from parent grid
    # calculate boundary conditions at start and end of calculation
    (tc_child.u_node_top[:, :], 
    tc_child.u_node_bottom[:, :], 
    tc_child.u_node_left[:, :], 
    tc_child.u_node_right[:, :], 
    tc_child.u_link_top[:, :], 
    tc_child.u_link_bottom[:, :],
    tc_child.u_link_left[:, :],
    tc_child.u_link_right[:, :]) = interp_boundary_condition(node_values_top=tc_child.u_node_top, 
                                                    node_values_bottom=tc_child.u_node_bottom,
                                                    node_values_left=tc_child.u_node_left,
                                                    node_values_right=tc_child.u_node_right,
                                                    link_values_top=tc_child.u_link_top,
                                                    link_values_bottom=tc_child.u_link_bottom,
                                                    link_values_left=tc_child.u_link_left,
                                                    link_values_right=tc_child.u_link_right,
                                                    parent_grid_start=parent_grid_start,
                                                    parent_grid_end=parent_grid_end, 
                                                    time_interp=tc_child.time_interp,
                                                    tc_child=tc_child, 
                                                    variable_name='flow__horizontal_velocity_at_node', 
                                                    nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                                    nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                                    nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                                    nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)
    
    (tc_child.v_node_top[:, :],
    tc_child.v_node_bottom[:, :],
    tc_child.v_node_left[:, :],
    tc_child.v_node_right[:, :],
    tc_child.v_link_top[:, :],
    tc_child.v_link_bottom[:, :],
    tc_child.v_link_left[:, :],
    tc_child.v_link_right[:, :]) = interp_boundary_condition(node_values_top=tc_child.v_node_top,
                                                    node_values_bottom=tc_child.v_node_bottom,
                                                    node_values_left=tc_child.v_node_left,
                                                    node_values_right=tc_child.v_node_right,
                                                    link_values_top=tc_child.v_link_top,
                                                    link_values_bottom=tc_child.v_link_bottom,
                                                    link_values_left=tc_child.v_link_left,
                                                    link_values_right=tc_child.v_link_right,
                                                    parent_grid_start=parent_grid_start,
                                                    parent_grid_end=parent_grid_end, 
                                                    time_interp=tc_child.time_interp,
                                                    tc_child=tc_child, 
                                                    variable_name='flow__vertical_velocity_at_node', 
                                                    nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                                    nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                                    nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                                    nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)
    
    (tc_child.h_node_top[:, :], 
    tc_child.h_node_bottom[:, :], 
    tc_child.h_node_left[:, :], 
    tc_child.h_node_right[:, :], 
    tc_child.h_link_top[:, :],
    tc_child.h_link_bottom[:, :],
    tc_child.h_link_left[:, :],
    tc_child.h_link_right[:, :]) = interp_boundary_condition(node_values_top=tc_child.h_node_top,
                                                    node_values_bottom=tc_child.h_node_bottom,
                                                    node_values_left=tc_child.h_node_left,
                                                    node_values_right=tc_child.h_node_right,
                                                    link_values_top=tc_child.h_link_top,
                                                    link_values_bottom=tc_child.h_link_bottom,
                                                    link_values_left=tc_child.h_link_left,
                                                    link_values_right=tc_child.h_link_right,
                                                    parent_grid_start=parent_grid_start,
                                                    parent_grid_end=parent_grid_end, 
                                                    time_interp=tc_child.time_interp,
                                                    tc_child=tc_child, 
                                                    variable_name='flow__depth', 
                                                    nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                                    nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                                    nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                                    nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)

    (tc_child.bed_thick_node_top[:, :],
    tc_child.bed_thick_node_bottom[:, :],
    tc_child.bed_thick_node_left[:, :],
    tc_child.bed_thick_node_right[:, :], 
    _, 
    _, 
    _, 
    _) = interp_boundary_condition(node_values_top=tc_child.bed_thick_node_top,
                                   node_values_bottom=tc_child.bed_thick_node_bottom,
                                   node_values_left=tc_child.bed_thick_node_left,
                                   node_values_right=tc_child.bed_thick_node_right,
                                   link_values_top=None,
                                   link_values_bottom=None,
                                   link_values_left=None,
                                   link_values_right=None,
                                   parent_grid_start=parent_grid_start,
                                   parent_grid_end=parent_grid_end,
                                   time_interp=tc_child.time_interp,
                                   tc_child=tc_child, 
                                   variable_name='bed__thickness', 
                                   nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                   nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                   nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                   nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)
    
    for i in range(gsize):
        (tc_child.C_i_node_top[:, i, :], 
        tc_child.C_i_node_bottom[:, i, :], 
        tc_child.C_i_node_left[:, i, :], 
        tc_child.C_i_node_right[:, i, :], 
        _, 
        _, 
        _, 
        _) = interp_boundary_condition(node_values_top=tc_child.C_i_node_top[:, i, :],
                                       node_values_bottom=tc_child.C_i_node_bottom[:, i, :],
                                       node_values_left=tc_child.C_i_node_left[:, i, :],
                                       node_values_right=tc_child.C_i_node_right[:, i, :],
                                       link_values_top=None,
                                       link_values_bottom=None,
                                       link_values_left=None,
                                       link_values_right=None,
                                       parent_grid_start=parent_grid_start,
                                       parent_grid_end=parent_grid_end, 
                                       time_interp=tc_child.time_interp,
                                       tc_child=tc_child, 
                                       variable_name='flow__sediment_concentration_%d' % (i), 
                                       nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                       nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                       nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                       nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)

        (tc_child.bed_thick_i_node_top[:, i, :], 
        tc_child.bed_thick_i_node_bottom[:, i, :],
        tc_child.bed_thick_i_node_left[:, i, :],
        tc_child.bed_thick_i_node_right[:, i, :],
        _,
        _,
        _,
        _) = interp_boundary_condition(node_values_top=tc_child.bed_thick_i_node_top[:, i, :],
                                       node_values_bottom=tc_child.bed_thick_i_node_bottom[:, i, :],
                                       node_values_left=tc_child.bed_thick_i_node_left[:, i, :],
                                       node_values_right=tc_child.bed_thick_i_node_right[:, i, :],
                                       link_values_top=None,
                                       link_values_bottom=None,
                                       link_values_left=None,
                                       link_values_right=None,
                                       parent_grid_start=parent_grid_start,
                                       parent_grid_end=parent_grid_end, 
                                       time_interp=tc_child.time_interp,
                                       tc_child=tc_child, 
                                       variable_name='bed__sediment_volume_per_unit_area_%d' % (i), 
                                       nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                       nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                       nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                       nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)
    if tc_child.model == '4eq':
        # calculate boundary conditions of Kh using other boundary conditions
        (tc_child.Kh_node_top[:, :], 
        tc_child.Kh_node_bottom[:, :],
        tc_child.Kh_node_left[:, :],
        tc_child.Kh_node_right[:, :],
        tc_child.Kh_link_top[:, :],
        tc_child.Kh_link_bottom[:, :],
        tc_child.Kh_link_left[:, :], 
        tc_child.Kh_link_right[:, :]) = interp_boundary_condition(node_values_top=tc_child.Kh_node_top,
                                                        node_values_bottom=tc_child.Kh_node_bottom,
                                                        node_values_left=tc_child.Kh_node_left,
                                                        node_values_right=tc_child.Kh_node_right,
                                                        link_values_top=tc_child.Kh_link_top,
                                                        link_values_bottom=tc_child.Kh_link_bottom,
                                                        link_values_left=tc_child.Kh_link_left,
                                                        link_values_right=tc_child.Kh_link_right,
                                                        parent_grid_start=parent_grid_start,
                                                        parent_grid_end=parent_grid_end, 
                                                        time_interp=tc_child.time_interp,
                                                        tc_child=tc_child, 
                                                        variable_name='flow__TKE_at_node', 
                                                        nested_top_boundary_nodes_parent=nested_top_boundary_nodes_parent, 
                                                        nested_bottom_boundary_nodes_parent=nested_bottom_boundary_nodes_parent, 
                                                        nested_left_boundary_nodes_parent=nested_left_boundary_nodes_parent,
                                                        nested_right_boundary_nodes_parent=nested_right_boundary_nodes_parent)


def nc_to_landlab(nc_file):
    """Convert netcdf file to landlab grid

    Parameters
    ----------
    nc_file : str
        Path to the netcdf file

    Returns
    -------
    grid : RasterModelGrid
        Landlab grid object
    """
    # Read netcdf file
    ds = nc.Dataset(nc_file)
    # Create a RasterModelGrid object
    dx = Decimal(str(ds.variables['x'][0][1])) - Decimal(str(ds.variables['x'][0][0]))
    dy = Decimal(str(ds.variables['y'][1][0])) - Decimal(str(ds.variables['y'][0][0]))
    dx = float(dx)
    dy = float(dy)
    grid = RasterModelGrid((ds.dimensions['nj'].size, ds.dimensions['ni'].size), xy_spacing=(dx, dy))
    grid.node_x = np.around(grid.node_x, decimals=2)
    grid.node_y = np.around(grid.node_y, decimals=2)
    # Add data to the grid
    for var in ds.variables:
        if var not in ['x', 'y', 'ni', 'nj']:
            grid.add_field(var, ds.variables[var][:], at='node')
    
    return grid