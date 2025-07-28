import numpy as np
import pdb
import re
import netCDF4 as nc
from decimal import Decimal
from scipy.interpolate import interp1d
from turb2d._links import top_edge_vertical_ids, bottom_edge_vertical_ids, left_edge_horizontal_ids, right_edge_horizontal_ids, top_edge_horizontal_ids, bottom_edge_horizontal_ids, left_edge_vertical_ids, right_edge_vertical_ids
from landlab import RasterModelGrid
from scipy.interpolate import griddata, RegularGridInterpolator
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

        xmin, xmax, ymin, ymax = self.nested_region
        # The extraction of regions by self.nested_region does not work well due to floating point precision.
        # Therefore, the coordinates of the parent grid are rounded to a specified number of decimal places.
        parent_node_x = self.round_values(self.tc_parent.grid.node_x, decimals=5)
        parent_node_y = self.round_values(self.tc_parent.grid.node_y, decimals=5)
        self.nested_region_idx = np.where(
                                    (parent_node_x >= xmin) & 
                                    (parent_node_x <= xmax) & 
                                    (parent_node_y >= ymin) & 
                                    (parent_node_y <= ymax)
                                    )
        self.nested_region_idx_except_boundary = np.where(
                                    (parent_node_x > xmin) & 
                                    (parent_node_x < xmax) & 
                                    (parent_node_y > ymin) & 
                                    (parent_node_y < ymax)
                                    )
        ### This is the debugging code to check the nested region indices ###
        self.tc_parent.nested_region_idx = self.nested_region_idx
        
    def interp_griddata(self, x, y, parent_values, x_new, y_new, interp_method='linear'):
        """
        Interpolate (with extrapolation) the parent grid data for the initial 
        and boundary conditions of the child grid using RegularGridInterpolator.

        Parameters
        ----------
        x : ndarray
            1D array of unique x-coordinates (grid-aligned).
        y : ndarray
            1D array of unique y-coordinates (grid-aligned).
        parent_values : ndarray
            1D array of values of shape (len(x) * len(y)), ordered to match np.meshgrid(x, y, indexing='ij').
        x_new : ndarray
            1D array of x-coordinates for the child grid.
        y_new : ndarray
            1D array of y-coordinates for the child grid.

        Returns
        -------
        z_new : ndarray
            1D array of interpolated (and extrapolated) values at the (x_new, y_new) grid.
        """

        x_unique = np.unique(x)
        y_unique = np.unique(y)

        # RegularGridInterpolatorのために正しい形状にreshape
        # parent_valuesは1D配列で、(y, x)の順序で並んでいると仮定
        z_grid = parent_values.reshape(len(y_unique), len(x_unique))

        interpolator = RegularGridInterpolator(
            (y_unique, x_unique),  
            z_grid,
            method=interp_method,
            bounds_error=False,
            fill_value=None # allow linear extrapolation
        )

        xx, yy = np.meshgrid(x_new, y_new, indexing='xy')  
        points_new = np.column_stack([yy.ravel(), xx.ravel()])  

        z_new = interpolator(points_new)

        return z_new
    
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
        parent_x = self.round_values(self.tc_parent.grid.node_x[nested_idx], decimals=decimal)
        parent_y = self.round_values(self.tc_parent.grid.node_y[nested_idx], decimals=decimal)

        child_grid_spacing = self.tc_child.grid.spacing[0]
        s = str(child_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        child_x = self.round_values(self.tc_child.grid.node_x, decimals=decimal)
        child_y = self.round_values(self.tc_child.grid.node_y, decimals=decimal)

        # get indices of child grid nodes that correspond to the nested region in the parent grid
        idx_match_parent_child_grid = np.zeros_like(parent_x, dtype=int)
        for i in range(parent_x.size):
            idx_match_parent_child_grid[i] = np.where((child_x == parent_x[i]) & (child_y == parent_y[i]))[0]

        # spatial intepolation of the values at the nested region to compute the conditions at the initial and final time
        x = child_x[idx_match_parent_child_grid]
        y = child_y[idx_match_parent_child_grid]
        x_new = np.unique(child_x)
        y_new = np.unique(child_y)
        interp_values[0, :] = self.interp_griddata(x=x, y=y, parent_values=values_at_start, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)
        interp_values[-1, :] = self.interp_griddata(x=x, y=y, parent_values=values_at_end, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)
        
        # temporal interpolation to calculate the initial and boundary conditions at each time steps
        interp_values, _ = self.temporal_interp(node_values=interp_values, link_values=None, time_interp=time_interp)

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
        
        # time step for interpolation
        self.tc_child.time_interp = np.arange(0.0, self.dt+time_step, time_step)

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
                                                                                                nested_idx=self.nested_region_idx[0],
                                                                                                spatial_interp_method='linear')

        self.tc_child.v_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp, 
                                                                                                variable_start=self.tc_parent.v_node_ini,
                                                                                                variable_end=self.tc_parent.v_node,     
                                                                                                nested_idx=self.nested_region_idx[0],
                                                                                                spatial_interp_method='linear')

        self.tc_child.h_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                      variable_start=self.tc_parent.h_ini,
                                                                                      variable_end=self.tc_parent.h,     
                                                                                      nested_idx=self.nested_region_idx[0],
                                                                                      spatial_interp_method='linear')

        self.tc_child.bed_thick_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                              variable_start=self.tc_parent.bed_thick_ini,
                                                                                              variable_end=self.tc_parent.bed_thick,     
                                                                                              nested_idx=self.nested_region_idx[0],
                                                                                              spatial_interp_method='linear')

        for i in range(gsize):
            self.tc_child.C_i_node_child_grid_condition[:, i, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                               variable_start=self.tc_parent.C_i_ini[i, :],
                                                                                               variable_end=self.tc_parent.C_i[i, :],     
                                                                                               nested_idx=self.nested_region_idx[0],
                                                                                               spatial_interp_method='linear')

            self.tc_child.bed_thick_i_node_child_grid_condition[:, i, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                                       variable_start=self.tc_parent.bed_thick_i_ini[i, :],
                                                                                                       variable_end=self.tc_parent.bed_thick_i[i, :],     
                                                                                                       nested_idx=self.nested_region_idx[0],
                                                                                                       spatial_interp_method='linear')

        if self.tc_child.model == '4eq' and self.tc_parent.model == '4eq':
            self.tc_child.Kh_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                           variable_start=self.tc_parent.Kh_node_ini,
                                                                                           variable_end=self.tc_parent.Kh_node,     
                                                                                           nested_idx=self.nested_region_idx[0],
                                                                                           spatial_interp_method='linear')
        elif (self.tc_child.model == '4eq' and self.tc_parent.model != '4eq') or (self.tc_child.model != '4eq' and self.tc_parent.model == '4eq'):
            raise ValueError("The parent grid model must be '4eq' to compute the Kh values for the child grid.")

class TwoWayNesting(OneWayNesting):
    """Class for two-way nesting."""

    def interpolate_parent_to_child_grid(self, variable, nested_idx, spatial_interp_method='linear'):
        """Interpolate the result of the parent calculation to obtain the initial and boundary conditions for the child grid.
        
        Parameters
        ----------
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

        values = variable[nested_idx]
        interp_values = np.zeros((self.tc_child.grid.number_of_nodes))

        # remove the error due to floating point precision
        parent_grid_spacing = self.tc_parent.grid.spacing[0]
        s = str(parent_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        parent_x = self.round_values(self.tc_parent.grid.node_x[nested_idx], decimals=decimal)
        parent_y = self.round_values(self.tc_parent.grid.node_y[nested_idx], decimals=decimal)

        child_grid_spacing = self.tc_child.grid.spacing[0]
        s = str(child_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        child_x = self.round_values(self.tc_child.grid.node_x, decimals=decimal)
        child_y = self.round_values(self.tc_child.grid.node_y, decimals=decimal)

        # get indices of child grid nodes that correspond to the nested region in the parent grid
        idx_match_parent_child_grid = np.zeros_like(parent_x, dtype=int)
        for i in range(parent_x.size):
            idx_match_parent_child_grid[i] = np.where((child_x == parent_x[i]) & (child_y == parent_y[i]))[0]

        # spatial intepolation of the values at the nested region to compute the conditions at the initial and final time
        x = child_x[idx_match_parent_child_grid]
        y = child_y[idx_match_parent_child_grid]
        x_new = np.unique(child_x)
        y_new = np.unique(child_y)
        interp_values[:] = self.interp_griddata(x=x, y=y, parent_values=values, x_new=x_new, y_new=y_new, interp_method=spatial_interp_method)

        return interp_values

    def compute_child_grid_condition_from_parent_grid(self):
        """Compute initial and boundary conditions of a child calculation from a parent calculation.

        Parameters
        ----------

        time_step : float
            Time step for interpolation. Time steps must be given in seconds.
        first_calc : bool, optional
            Flag to indicate if this is the first calculation. Default is True.
        """
        
        # time step for interpolation
        # self.tc_child.time_interp = np.arange(0.0, self.dt+time_step, time_step)

        # initialize arrays for the conditions of the child grid
        # self.tc_child.u_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        # self.tc_child.v_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        self.tc_child.u_node_child_grid_condition = np.zeros((self.tc_child.grid.number_of_nodes))
        self.tc_child.v_node_child_grid_condition = np.zeros((self.tc_child.grid.number_of_nodes))

        # self.tc_child.h_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        self.tc_child.h_node_child_grid_condition = np.zeros((self.tc_child.grid.number_of_nodes))

        gsize = self.tc_parent.C_i.shape[0]
        # self.tc_child.C_i_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, gsize, self.tc_child.grid.number_of_nodes))
        self.tc_child.C_i_node_child_grid_condition = np.zeros((gsize, self.tc_child.grid.number_of_nodes))

        # self.tc_child.Kh_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        self.tc_child.Kh_node_child_grid_condition = np.zeros((self.tc_child.grid.number_of_nodes))


        # self.tc_child.bed_thick_i_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, gsize, self.tc_child.grid.number_of_nodes))
        # self.tc_child.bed_thick_node_child_grid_condition = np.zeros((self.tc_child.time_interp.size, self.tc_child.grid.number_of_nodes))
        self.tc_child.bed_thick_i_node_child_grid_condition = np.zeros((gsize, self.tc_child.grid.number_of_nodes))
        self.tc_child.bed_thick_node_child_grid_condition = np.zeros((self.tc_child.grid.number_of_nodes))

        # interpolate the values of the parent grid to generated the initial and boundary conditions for the child grid
        # NOTE: The link values of u, v, and Kh are calculated within the function `map_values()`, which is called during `run_one_step()`.
        self.tc_child.u_node_child_grid_condition[:] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.u_node,     
                                                                                                nested_idx=self.nested_region_idx[0],
                                                                                                spatial_interp_method='linear')

        self.tc_child.v_node_child_grid_condition[:] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.v_node,     
                                                                                                nested_idx=self.nested_region_idx[0],
                                                                                                spatial_interp_method='linear')
        self.tc_child.h_node_child_grid_condition[:] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.h,     
                                                                                                nested_idx=self.nested_region_idx[0],
                                                                                                spatial_interp_method='linear')

        self.tc_child.bed_thick_node_child_grid_condition[:] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.bed_thick,     
                                                                                                        nested_idx=self.nested_region_idx[0],
                                                                                                        spatial_interp_method='linear')

        for i in range(gsize):
            self.tc_child.C_i_node_child_grid_condition[i, :] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.C_i[i, :],     
                                                                                                        nested_idx=self.nested_region_idx[0],
                                                                                                        spatial_interp_method='linear')

            self.tc_child.bed_thick_i_node_child_grid_condition[i, :] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.bed_thick_i[i, :],     
                                                                                                                nested_idx=self.nested_region_idx[0],
                                                                                                                spatial_interp_method='linear')

        if self.tc_child.model == '4eq' and self.tc_parent.model == '4eq':
            self.tc_child.Kh_node_child_grid_condition[:] = self.interpolate_parent_to_child_grid(variable=self.tc_parent.Kh_node,     
                                                                                                    nested_idx=self.nested_region_idx[0],
                                                                                                    spatial_interp_method='linear')
        elif (self.tc_child.model == '4eq' and self.tc_parent.model != '4eq') or (self.tc_child.model != '4eq' and self.tc_parent.model == '4eq'):
            raise ValueError("The parent grid model must be '4eq' to compute the Kh values for the child grid.")

    def compute_parent_grid_condition_from_child_grid(self):
        # round values
        parent_grid_spacing = self.tc_parent.grid.spacing[0]
        s = str(parent_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        parent_x = self.round_values(self.tc_parent.grid.node_x[self.nested_region_idx], decimals=decimal)
        parent_y = self.round_values(self.tc_parent.grid.node_y[self.nested_region_idx], decimals=decimal)

        child_grid_spacing = self.tc_child.grid.spacing[0]
        s = str(child_grid_spacing)
        _, s_d = s.split('.')
        decimal = len(s_d)
        child_x = self.round_values(self.tc_child.grid.node_x, decimals=decimal)
        child_y = self.round_values(self.tc_child.grid.node_y, decimals=decimal)
        
        # the index of the child node that matches the coordinates of the parent node
        # OPTIMIZE: vectorize this operation
        match_child_grid_idx = np.zeros_like(parent_x, dtype=int)
        for i in range(parent_x.size):
            match_child_grid_idx[i] = np.where((child_x == parent_x[i]) & (child_y == parent_y[i]))[0]
        # remove boundary nodes idx from match_child_grid_idx
        boundary_nodes_idx = self.tc_child.grid.node_is_boundary(match_child_grid_idx)
        match_child_grid_idx = match_child_grid_idx[~boundary_nodes_idx]

        # get idx of neighboring child nodes
        neighbor_nodes = self.tc_child.grid.adjacent_nodes_at_node.copy()
        node_east = neighbor_nodes[:, 0][match_child_grid_idx]
        node_north = neighbor_nodes[:, 1][match_child_grid_idx]
        node_west = neighbor_nodes[:, 2][match_child_grid_idx]
        node_south = neighbor_nodes[:, 3][match_child_grid_idx]

        # calculate the parent node values using the neighbor child node values
        # u, v, ci, kh, bedthick_i, bedthick
        self.tc_parent.u_node[self.nested_region_idx_except_boundary] = np.mean(
            [self.tc_child.u_node[node_east],
             self.tc_child.u_node[node_north],
             self.tc_child.u_node[node_west],
             self.tc_child.u_node[node_south]], axis=0)
        
        self.tc_parent.v_node[self.nested_region_idx_except_boundary] = np.mean(
            [self.tc_child.v_node[node_east],
             self.tc_child.v_node[node_north],
             self.tc_child.v_node[node_west],
             self.tc_child.v_node[node_south]], axis=0)
        
        self.tc_parent.h[self.nested_region_idx_except_boundary] = np.mean(
            [self.tc_child.h[node_east],
             self.tc_child.h[node_north],
             self.tc_child.h[node_west],
             self.tc_child.h[node_south]], axis=0)
        
        self.tc_parent.bed_thick[self.nested_region_idx_except_boundary] = np.mean(
            [self.tc_child.bed_thick[node_east],
             self.tc_child.bed_thick[node_north],
             self.tc_child.bed_thick[node_west],
             self.tc_child.bed_thick[node_south]], axis=0)
        
        for i in range(self.tc_child.C_i.shape[0]):
            self.tc_parent.C_i[i, self.nested_region_idx_except_boundary] = np.mean(
                [self.tc_child.C_i[i, node_east],
                 self.tc_child.C_i[i, node_north],
                 self.tc_child.C_i[i, node_west],
                 self.tc_child.C_i[i, node_south]], axis=0)
            
            self.tc_parent.bed_thick_i[i, self.nested_region_idx_except_boundary] = np.mean(
                [self.tc_child.bed_thick_i[i, node_east],
                 self.tc_child.bed_thick_i[i, node_north],
                 self.tc_child.bed_thick_i[i, node_west],
                 self.tc_child.bed_thick_i[i, node_south]], axis=0)
            
        if self.tc_child.model == '4eq' and self.tc_parent.model == '4eq':
            self.tc_parent.Kh_node[self.nested_region_idx_except_boundary] = np.mean(
                [self.tc_child.Kh_node[node_east],
                 self.tc_child.Kh_node[node_north],
                 self.tc_child.Kh_node[node_west],
                 self.tc_child.Kh_node[node_south]], axis=0)
        elif (self.tc_child.model == '4eq' and self.tc_parent.model != '4eq') or (self.tc_child.model != '4eq' and self.tc_parent.model == '4eq'):
            raise ValueError("The parent grid model must be '4eq' to compute the Kh values for the child grid.")