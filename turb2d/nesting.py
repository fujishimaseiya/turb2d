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
from numba import jit

@jit(nopython=True, cache=True)
def temporal_interp_numba(node_values, link_values, time_interp):
    """Numba最適化された時間補間関数"""
    if time_interp.size <= 2:
        return node_values, link_values
    
    t0, t1 = time_interp[0], time_interp[-1]
    dt = t1 - t0
    if dt == 0:
        return node_values, link_values
    
    # 効率的な線形補間
    for i in range(1, time_interp.size - 1):
        alpha = (time_interp[i] - t0) / dt
        for j in range(node_values.shape[1]):
            node_values[i, j] = node_values[0, j] + alpha * (node_values[-1, j] - node_values[0, j])
        
        if link_values is not None:
            for j in range(link_values.shape[1]):
                link_values[i, j] = link_values[0, j] + alpha * (link_values[-1, j] - link_values[0, j])
    
    return node_values, link_values

class OneWayNesting():
    """Class for one-way nesting."""

    def __init__(self, tc_parent, tc_child, nested_region, parent_grid_file=None, child_grid_file=None, dt=1.0, num_relaxation_grid=1):
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
        self.num_relaxation_grid = num_relaxation_grid

        xmin = self.tc_child.grid.x_of_node.min()
        xmax = self.tc_child.grid.x_of_node.max()
        ymin = self.tc_child.grid.y_of_node.min()
        ymax = self.tc_child.grid.y_of_node.max()
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
        # self.tc_parent.nested_region_idx = self.nested_region_idx
        
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
        """This is a wrapper function for the numba-optimized temporal interpolation."""
        return temporal_interp_numba(node_values, link_values, time_interp)
    # @jit(nopython=True, cache=True)
    # def temporal_interp(self, node_values, link_values, time_interp):
    #     """Interpolate the node and link values linearly over time steps.

    #     Parameters
    #     ----------
    #     node_values : ndarray
    #         Array of node values at different time steps. Shape should be (time_steps, number_of_nodes).
    #     link_values : ndarray, optional
    #         Array of link values at different time steps. Shape should be (time_steps, number_of_links). Default is None.
    #     time_interp : ndarray
    #         Array of time steps for interpolation. Should be of shape (time_steps,).

    #     Returns
    #     -------
    #     node_values : ndarray
    #         Interpolated node values at the specified time steps.
    #     link_values : ndarray, optional
    #         Interpolated link values at the specified time steps. Will be None if link_values is None.
    #     """

    #     t0, t1 = time_interp[0], time_interp[-1]
    #     t = time_interp[1:-1]
    #     alpha = (t - t0) / (t1 - t0)

    #     node_start, node_end = node_values[0, :], node_values[-1, :]
    #     node_values[1:-1, :] = node_start + alpha[:, None] * (node_end - node_start)

    #     if link_values is not None:
    #         link_start, link_end = link_values[0, :], link_values[-1, :]
    #         link_values[1:-1, :] = link_start + alpha[:, None] * (link_end - link_start)

    #     return node_values, link_values
    

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

        # self.tc_child.bed_thick_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
        #                                                                                       variable_start=self.tc_parent.bed_thick_ini,
        #                                                                                       variable_end=self.tc_parent.bed_thick,     
        #                                                                                       nested_idx=self.nested_region_idx[0],
        #                                                                                       spatial_interp_method='linear')

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
        # self.tc_child.bed_thick_node_child_grid_condition[:, :] = np.sum(self.tc_child.bed_thick_i_node_child_grid_condition, axis=1)

        if self.tc_child.model == '4eq' and self.tc_parent.model == '4eq':
            self.tc_child.Kh_node_child_grid_condition[:, :] = self.interpolate_parent_to_child_grid(time_interp=self.tc_child.time_interp,
                                                                                           variable_start=self.tc_parent.Kh_node_ini,
                                                                                           variable_end=self.tc_parent.Kh_node,     
                                                                                           nested_idx=self.nested_region_idx[0],
                                                                                           spatial_interp_method='linear')
        elif (self.tc_child.model == '4eq' and self.tc_parent.model != '4eq') or (self.tc_child.model != '4eq' and self.tc_parent.model == '4eq'):
            raise ValueError("The parent grid model must be '4eq' to compute the Kh values for the child grid.")
        
    def flow_relaxation_scheme(self):
        """Flow relaxation scheme (FRS) for the flow variables in the child grid.
        """
        num_grid_NS = self.tc_child.grid.shape[0]
        num_grid_EW = self.tc_child.grid.shape[1]

        # calculate the relaxation parameters if not already calculated
        has_var = (
                    hasattr(self, 'relax_param_N_to_S') 
                    and hasattr(self, 'relax_param_S_to_N') 
                    and hasattr(self, 'relax_param_E_to_W') 
                    and hasattr(self, 'relax_param_W_to_E')
                    )
        if has_var is False:
            self.calc_relax_param()

        # initialize the variables for the flow relaxation scheme
        u_node_relax = self.tc_child.u_node.reshape(self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        v_node_relax = self.tc_child.v_node.reshape(self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        h_node_relax = self.tc_child.h.reshape(self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        bed_thick_node_relax = self.tc_child.bed_thick.reshape(self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        C_i_node_relax = self.tc_child.C_i.reshape(self.tc_child.C_i.shape[0], self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        bed_thick_i_node_relax = self.tc_child.bed_thick_i.reshape(self.tc_child.bed_thick_i.shape[0], self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        if self.tc_child.model == '4eq':
            Kh_node_relax = self.tc_child.Kh_node.reshape(self.tc_child.grid.shape[0], self.tc_child.grid.shape[1]).copy()
        
        # calculate the variables using Flow Relaxation Scheme
        # ここから，relax valueを求めるところをカプセル化してすべての変数に適用．
        end_internal_region_x = self.tc_child.grid.shape[1] - (self.num_relaxation_grid + 1)

        u_node_relax[:, :] = self.apply_flow_relaxation_scheme(u_node_relax, 
                                                          end_internal_region_x)
        v_node_relax[:, :] = self.apply_flow_relaxation_scheme(v_node_relax,
                                                          end_internal_region_x)
        h_node_relax[:, :] = self.apply_flow_relaxation_scheme(h_node_relax,
                                                          end_internal_region_x)
        # bed_thick_node_relax[:, :] = self.apply_flow_relaxation_scheme(bed_thick_node_relax,
        #                                                   end_internal_region_x)
        for i in range(self.tc_child.C_i.shape[0]):
            C_i_node_relax[i, :, :] = self.apply_flow_relaxation_scheme(C_i_node_relax[i, :, :],
                                                          end_internal_region_x)
            bed_thick_i_node_relax[i, :, :] = self.apply_flow_relaxation_scheme(bed_thick_i_node_relax[i, :, :],
                                                          end_internal_region_x)
        if self.tc_child.model == '4eq':
            Kh_node_relax[:, :] = self.apply_flow_relaxation_scheme(Kh_node_relax,
                                                          end_internal_region_x)
            
        # mask the FRS zone in the child grid
        mask_frs_zone = np.ones((num_grid_NS, num_grid_EW), dtype=bool)
        mask_frs_zone[self.num_relaxation_grid:-self.num_relaxation_grid, self.num_relaxation_grid:-self.num_relaxation_grid] = False
        mask_frs_zone = mask_frs_zone.flatten()
        # update the child grid variables in the FRS zone
        self.tc_child.u_node[mask_frs_zone] = u_node_relax.flatten()[mask_frs_zone]
        self.tc_child.v_node[mask_frs_zone] = v_node_relax.flatten()[mask_frs_zone]
        self.tc_child.h[mask_frs_zone] = h_node_relax.flatten()[mask_frs_zone]
        self.tc_child.C_i[:, mask_frs_zone] = C_i_node_relax.reshape(self.tc_child.C_i.shape[0], -1)[:, mask_frs_zone]
        self.tc_child.C[:] = np.sum(self.tc_child.C_i, axis=0)
        self.tc_child.bed_thick_i[:, mask_frs_zone] = bed_thick_i_node_relax.reshape(self.tc_child.bed_thick_i.shape[0], -1)[:, mask_frs_zone]
        self.tc_child.bed_thick[:] = np.sum(self.tc_child.bed_thick_i, axis=0)
        if self.tc_child.model == '4eq':
            self.tc_child.Kh_node[mask_frs_zone] = Kh_node_relax.flatten()[mask_frs_zone]


    def calc_relaxed_values(self, 
                            values,
                            values_east, 
                            values_west, 
                            values_south, 
                            values_north, 
                            start_x_idx, 
                            end_x_idx, 
                            start_y_idx, 
                            end_y_idx, 
                            relax_param_NS, 
                            relax_param_SN, 
                            relax_param_EW, 
                            relax_param_WE):

        """Calculate values at flow relaxation zone

        Parameters
        ----------
        values : np.ndarray
            Values to be updated.

        values_east : np.ndarray
            Values at east end for relaxation calculation.

        values_west : np.ndarray
            Values at west end for relaxation calculation.

        values_south : np.ndarray
            Values at south end for relaxation calculation.

        values_north : np.ndarray
            Values at north end for relaxation calculation.

        start_x_idx : int
            Starting x-index of the calculation region.

        end_x_idx : int
            Ending x-index of the calculation region.

        start_y_idx : int
            Starting y-index of the calculation region.

        end_y_idx : int
            Ending y-index of the calculation region.
        """

        values_in_frs_zone = (relax_param_NS*values_north + (1 - relax_param_NS)*values[(start_y_idx):end_y_idx, (start_x_idx):end_x_idx] 
                              + relax_param_SN*values_south + (1 - relax_param_SN)*values[(start_y_idx):end_y_idx, (start_x_idx):end_x_idx] 
                              + relax_param_EW*values_east + (1 - relax_param_EW)*values[(start_y_idx):end_y_idx, (start_x_idx):end_x_idx] 
                              + relax_param_WE*values_west + (1 - relax_param_WE)*values[(start_y_idx):end_y_idx, (start_x_idx):end_x_idx]
                              ) / 4

        # values_in_frs_zone_shape = (end_y_idx - start_y_idx, end_x_idx - start_x_idx)
        # values_in_frs_zone = np.zeros(values_in_frs_zone_shape)

        # for i in range(values_in_frs_zone_shape[0]):
        #     for j in range(values_in_frs_zone_shape[1]):
        #         y_idx = start_y_idx + i
        #         x_idx = start_x_idx + j
        #         term1 = relax_param_NS[i, j] * values_north[j] + (1 - relax_param_NS[i, j]) * values[y_idx, x_idx]
        #         term2 = relax_param_SN[i, j] * values_south[j] + (1 - relax_param_SN[i, j]) * values[y_idx, x_idx]
        #         term3 = relax_param_EW[i, j] * values_east[i] + (1 - relax_param_EW[i, j]) * values[y_idx, x_idx]
        #         term4 = relax_param_WE[i, j] * values_west[i] + (1 - relax_param_WE[i, j]) * values[y_idx, x_idx]
        #         values_in_frs_zone[i, j] = (
        #                                     relax_param_NS[i, j] * values_north[j] + (1 - relax_param_NS[i, j]) * values[start_y_idx + i, start_x_idx + j] +
        #                                     relax_param_SN[i, j] * values_south[j] + (1 - relax_param_SN[i, j]) * values[start_y_idx + i, start_x_idx + j] +
        #                                     relax_param_EW[i, j] * values_east[i] + (1 - relax_param_EW[i, j]) * values[start_y_idx + i, start_x_idx + j] +
        #                                     relax_param_WE[i, j] * values_west[i] + (1 - relax_param_WE[i, j]) * values[start_y_idx + i, start_x_idx + j]
        #                                    ) / 4

        return values_in_frs_zone

    def apply_flow_relaxation_scheme(
            self, 
            hyd_value, 
            end_internal_region_x
    ):
        
        """Calculate the values at the FRS zone.
        # ---- # ---- $ ---- $ ---- $ ---- $ ---- $ ---- $ ---- @ ---- @
        |      |      |      |      |      |      |      |      |      |
        #             $                                  $             @
        |   Region1   |            Region 2              |   Region 3  |
        #             $                                  $             @ 
        |      |      |      |      |      |      |      |      |      |
        ^ ---- ^ ---- % ~~~~ % ~~~~ % ~~~~ % ~~~~ % ~~~~ % ---- & ---- &
        |      |      ~      ~      ~      ~      ~      ~      |      |
        *             % ---- % ---- % ---- % ---- % ---- %             &
        |             ~      ~      ~      ~      ~      ~             |
        *   Region 4  % ----  internal zone  ---- % ---- %   Region 5  &
        |             ~      ~      ~      ~      ~      ~             |
        *             % ---- % ---- % ---- % ---- % ---- %             &
        |      |      ~      ~      ~      ~      ~      ~      |      |
        ^ ---- ^ ---- % ~~~~ % ~~~~ % ~~~~ % ~~~~ % ~~~~ % ---- & ---- &
        |      |      |      |      |      |      |      |      |      |
        +             ?                                  ?             >
        |   Region 6  |            Region 7              |   Region 8  |
        +             ?                                  ?             >
        |      |      |      |      |      |      |      |      |      |
        + ---- + ---- ? ---- ? ---- ? ---- ? ---- ? ---- ? ---- > ---- >

        """

        values_in_frs_zone = hyd_value.copy()

        # relaxation using the boundary values only
        # values_in_frs_zone[:, :] = self.calc_relaxed_values(values=hyd_value,
        #                                                     values_east=hyd_value[:, -1],
        #                                                     values_west=hyd_value[:, 0],
        #                                                     values_south=hyd_value[-1, :],
        #                                                     values_north=hyd_value[0, :],
        #                                                     start_x_idx=0,
        #                                                     end_x_idx=None,
        #                                                     start_y_idx=0,
        #                                                     end_y_idx=None,
        #                                                     relax_param_NS=self.relax_param_NS_external_external,
        #                                                     relax_param_SN=self.relax_param_SN_external_external,
        #                                                     relax_param_EW=self.relax_param_EW_external_external,
        #                                                     relax_param_WE=self.relax_param_WE_external_external
        #                                                     )
        # mask = np.ones_like(values_in_frs_zone, dtype=bool)
        # mask[self.num_relaxation_grid:-self.num_relaxation_grid, self.num_relaxation_grid:-self.num_relaxation_grid] = False
        # hyd_value[mask] = values_in_frs_zone[mask]

        # Region 1
        # region1_start_x = 0
        # region1_end_x = self.num_relaxation_grid
        # region1_start_y = 0
        # region1_end_y = self.num_relaxation_grid
        # # Region 1
        # values_in_frs_zone[(region1_start_y+1):region1_end_y, 
        #                    (region1_start_x+1):region1_end_x] = self.calc_relaxed_values(values=hyd_value,
        #                                                                                  values_east=hyd_value[(region1_start_y+1):region1_end_y, -1],
        #                                                                                  values_west=hyd_value[(region1_start_y+1):region1_end_y, 0],
        #                                                                                  values_south=hyd_value[-1, (region1_start_x+1):region1_end_x],
        #                                                                                  values_north=hyd_value[0, (region1_start_x+1):region1_end_x],
        #                                                                                  start_x_idx=(region1_start_x+1),
        #                                                                                  end_x_idx=region1_end_x,
        #                                                                                  start_y_idx=(region1_start_y+1),
        #                                                                                  end_y_idx=region1_end_y,
        #                                                                                  relax_param_NS=self.relax_param_NS_external_external[(region1_start_y+1):region1_end_y, 
        #                                                                                                                                       (region1_start_x+1):region1_end_x],
        #                                                                                  relax_param_SN=self.relax_param_SN_external_external[(region1_start_y+1):region1_end_y, 
        #                                                                                                                                       (region1_start_x+1):region1_end_x],
        #                                                                                  relax_param_EW=self.relax_param_EW_external_external[(region1_start_y+1):region1_end_y, 
        #                                                                                                                                       (region1_start_x+1):region1_end_x],
        #                                                                                  relax_param_WE=self.relax_param_WE_external_external[(region1_start_y+1):region1_end_y, 
        #                                                                                                                                       (region1_start_x+1):region1_end_x]
        #                                                                                 )
        # values_in_frs_zone[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x] = \
        # (
        #     self.relax_param_NS_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #         * hyd_value[0, (region1_start_x+1):region1_end_x] \
        #             + (1 - self.relax_param_NS_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x])
        #             * hyd_value[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #     + self.relax_param_SN_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #         *hyd_value[-1, (region1_start_x+1):region1_end_x] \
        #             + (1 - self.relax_param_SN_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x])
        #             *hyd_value[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #     + self.relax_param_EW_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #         *hyd_value[(region1_start_y+1):region1_end_y, 0] \
        #             + (1 - self.relax_param_EW_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x])
        #             *hyd_value[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #     + self.relax_param_WE_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]
        #         *hyd_value[(region1_start_y+1):region1_end_y, -1] \
        #             + (1 - self.relax_param_WE_external_external[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x])
        #             *hyd_value[(region1_start_y+1):region1_end_y, (region1_start_x+1):region1_end_x]         
        # ) / 4
        # Region 2
        region2_start_x = self.num_relaxation_grid
        region2_end_x = end_internal_region_x + 1
        region2_start_y = 0
        region2_end_y = self.num_relaxation_grid
        values_in_frs_zone[(region2_start_y+1):region2_end_y, 
                           region2_start_x:region2_end_x] = self.calc_relaxed_values(values=hyd_value,
                                                                                     values_east=hyd_value[(region2_start_y+1):region2_end_y, -1][:, None],
                                                                                     values_west=hyd_value[(region2_start_y+1):region2_end_y, 0][:, None],
                                                                                     values_south=hyd_value[region2_end_y+1, region2_start_x:region2_end_x],
                                                                                     values_north=hyd_value[0, region2_start_x:region2_end_x],
                                                                                     start_x_idx=region2_start_x,
                                                                                     end_x_idx=region2_end_x,
                                                                                     start_y_idx=(region2_start_y+1),
                                                                                     end_y_idx=region2_end_y,
                                                                                     relax_param_NS=self.relax_param_NS_external_internal[(region2_start_y+1):, :],
                                                                                     relax_param_SN=self.relax_param_SN_external_internal[(region2_start_y+1):, :],
                                                                                     relax_param_EW=self.relax_param_EW_external_external[(region2_start_y+1):region2_end_y, 
                                                                                                                                          region2_start_x:region2_end_x],
                                                                                     relax_param_WE=self.relax_param_WE_external_external[(region2_start_y+1):region2_end_y, 
                                                                                                                                          region2_start_x:region2_end_x]
                                                                                    )
        
        # region2_start_x = self.num_relaxation_grid
        # region2_end_x = end_internal_region_x
        # region2_start_y = 0
        # region2_end_y = self.num_relaxation_grid
        # values_in_frs_zone[(region2_start_y+1):region2_end_y, region2_start_x:region2_end_x] = (
        #     self.relax_param_NS_external_internal[(region2_start_y+1):, :]*hyd_value[0, region2_start_x:region2_end_x] \
        #         + (1 - self.relax_param_NS_external_internal[(region2_start_y+1):, :])*hyd_value[region2_end_y, region2_start_x:region2_end_x]
        #     + self.relax_param_EW_external_external[(region2_start_y+1):region2_end_y, region2_start_x:region2_end_x]*hyd_value[(region2_start_y+1):region2_end_y, 0][:, None] \
        #         + (1 - self.relax_param_EW_external_external[(region2_start_y+1):region2_end_y, region2_start_x:region2_end_x])*hyd_value[(region2_start_y+1):region2_end_y, -1][:, None]
        # ) / 2
        # Region 3
        # region3_start_x = end_internal_region_x + 1
        # region3_end_x = -1
        # region3_start_y = 0
        # region3_end_y = self.num_relaxation_grid        
        # values_in_frs_zone[(region3_start_y+1):region3_end_y, region3_start_x:region3_end_x] = (
        #         self.relax_param_NS_external_external[(region3_start_y+1):region3_end_y, region3_start_x:region3_end_x]*hyd_value[0, region3_start_x:region3_end_x] \
        #             + (1 - self.relax_param_NS_external_external[(region3_start_y+1):region3_end_y, region3_start_x:region3_end_x])*hyd_value[-1, region3_start_x:region3_end_x]
        #         + self.relax_param_EW_external_external[(region3_start_y+1):region3_end_y, region3_start_x:region3_end_x]*hyd_value[(region3_start_y+1):region3_end_y, 0] \
        #             + (1 - self.relax_param_EW_external_external[(region3_start_y+1):region3_end_y, region3_start_x:region3_end_x])*hyd_value[(region3_start_y+1):region3_end_y, -1]
        #     ) / 2
        # values_in_frs_zone[(region3_start_y+1):region3_end_y, 
        #                    region3_start_x:region3_end_x] = self.calc_relaxed_values(values=hyd_value,
        #                                                                              values_east=hyd_value[(region3_start_y+1):region3_end_y, -1],
        #                                                                              values_west=hyd_value[(region3_start_y+1):region3_end_y, 0],
        #                                                                              values_south=hyd_value[-1, region3_start_x:region3_end_x],
        #                                                                              values_north=hyd_value[0, region3_start_x:region3_end_x],
        #                                                                              start_x_idx=region3_start_x,
        #                                                                              end_x_idx=region3_end_x,
        #                                                                              start_y_idx=(region3_start_y+1),
        #                                                                              end_y_idx= region3_end_y,
        #                                                                              relax_param_NS=self.relax_param_NS_external_external[(region3_start_y+1):region3_end_y, 
        #                                                                                                                                   region3_start_x:region3_end_x],
        #                                                                              relax_param_SN=self.relax_param_SN_external_external[(region3_start_y+1):region3_end_y,
        #                                                                                                                                   region3_start_x:region3_end_x],
        #                                                                              relax_param_EW=self.relax_param_EW_external_external[(region3_start_y+1):region3_end_y,
        #                                                                                                                                   region3_start_x:region3_end_x],
        #                                                                              relax_param_WE=self.relax_param_WE_external_external[(region3_start_y+1):region3_end_y,
        #                                                                                                                                   region3_start_x:region3_end_x]
        #                                                                             )
        # Region 4
        region4_start_x = 0
        region4_end_x = self.num_relaxation_grid
        region4_start_y = self.num_relaxation_grid
        region4_end_y = self.tc_child.grid.shape[0] - self.num_relaxation_grid      
        # values_in_frs_zone[region4_start_y:region4_end_y, (region4_start_x+1):region4_end_x] = ( 
        #         self.relax_param_NS_external_external[region4_start_y:region4_end_y, (region4_start_x+1):region4_end_x]*hyd_value[0, (region4_start_x+1):region4_end_x] \
        #             + (1 - self.relax_param_NS_external_external[region4_start_y:region4_end_y, (region4_start_x+1):region4_end_x])*hyd_value[-1, (region4_start_x+1):region4_end_x]
        #         + self.relax_param_WE_external_internal[:, (region4_start_x+1):]*hyd_value[region4_start_y:region4_end_y, 0][:, None] \
        #             + (1 - self.relax_param_WE_external_internal[:, (region4_start_x+1):])*hyd_value[region4_start_y:region4_end_y, region4_end_x][:, None]
        #     ) / 2
        values_in_frs_zone[region4_start_y:region4_end_y, 
                           (region4_start_x+1):region4_end_x] = self.calc_relaxed_values(values=hyd_value,
                                                                                         values_east=hyd_value[region4_start_y:region4_end_y, region4_end_x+1][:, None],
                                                                                         values_west=hyd_value[region4_start_y:region4_end_y, 0][:, None],
                                                                                         values_south=hyd_value[-1, (region4_start_x+1):region4_end_x],
                                                                                         values_north=hyd_value[0, (region4_start_x+1):region4_end_x],
                                                                                         start_x_idx=(region4_start_x+1),
                                                                                         end_x_idx=region4_end_x,
                                                                                         start_y_idx=region4_start_y,
                                                                                         end_y_idx= region4_end_y,
                                                                                         relax_param_NS=self.relax_param_NS_external_external[region4_start_y:region4_end_y,
                                                                                                                                              (region4_start_x+1):region4_end_x],
                                                                                         relax_param_SN=self.relax_param_SN_external_external[region4_start_y:region4_end_y,
                                                                                                                                              (region4_start_x+1):region4_end_x],
                                                                                         relax_param_EW=self.relax_param_EW_external_internal[:, (region4_start_x+1):],
                                                                                         relax_param_WE=self.relax_param_WE_external_internal[:, (region4_start_x+1):],
                                                                                        )
        # Region 5
        # pdb.set_trace()
        region5_start_x = end_internal_region_x + 1
        region5_end_x = -1
        region5_start_y = self.num_relaxation_grid
        region5_end_y = self.tc_child.grid.shape[0] - self.num_relaxation_grid  
        # values_in_frs_zone[region5_start_y:region5_end_y, region5_start_x:region5_end_x] = (
        #         self.relax_param_NS_external_external[region5_start_y:region5_end_y, region5_start_x:region5_end_x]*hyd_value[0, region5_start_x:region5_end_x] \
        #             + (1 - self.relax_param_NS_external_external[region5_start_y:region5_end_y, region5_start_x:region5_end_x])*hyd_value[-1, region5_start_x:region5_end_x]
        #         + (1 - self.relax_param_EW_external_internal[:, :region5_end_x])*hyd_value[region5_start_y:region5_end_y, region5_start_x][:, None] \
        #             + self.relax_param_EW_external_internal[:, :region5_end_x]*hyd_value[region5_start_y:region5_end_y, -1][:, None]
        #     ) / 2
        values_in_frs_zone[region5_start_y:region5_end_y, 
                           region5_start_x:region5_end_x] = self.calc_relaxed_values(values=hyd_value,
                                                                                     values_east=hyd_value[region5_start_y:region5_end_y, -1][:, None],
                                                                                     values_west=hyd_value[region5_start_y:region5_end_y, region5_start_x-1][:, None],
                                                                                     values_south=hyd_value[-1, region5_start_x:region5_end_x],
                                                                                     values_north=hyd_value[0, region5_start_x:region5_end_x],
                                                                                     start_x_idx=region5_start_x,
                                                                                     end_x_idx=region5_end_x,
                                                                                     start_y_idx=region5_start_y,
                                                                                     end_y_idx= region5_end_y,
                                                                                     relax_param_NS=self.relax_param_NS_external_external[region5_start_y:region5_end_y, 
                                                                                                                                          region5_start_x:region5_end_x],
                                                                                     relax_param_SN=self.relax_param_SN_external_external[region5_start_y:region5_end_y,
                                                                                                                                          region5_start_x:region5_end_x],
                                                                                     relax_param_EW=self.relax_param_EW_external_internal[:, :region5_end_x],
                                                                                     relax_param_WE= self.relax_param_WE_external_internal[:, 1:]
                                                                                    )
        # Region 6
        # region6_start_x = 0
        # region6_end_x = self.num_relaxation_grid
        # region6_start_y = (self.tc_child.grid.shape[0] - self.num_relaxation_grid)
        # region6_end_y = -1    
        # values_in_frs_zone[region6_start_y:region6_end_y, (region6_start_x+1):region6_end_x] = (   
        #         self.relax_param_NS_external_external[region6_start_y:region6_end_y, (region6_start_x+1):region6_end_x]*hyd_value[0, (region6_start_x+1):region6_end_x]\
        #              + (1 - self.relax_param_NS_external_external[region6_start_y:region6_end_y, (region6_start_x+1):region6_end_x])*hyd_value[-1, (region6_start_x+1):region6_end_x]
        #         + self.relax_param_EW_external_external[region6_start_y:region6_end_y, (region6_start_x+1):region6_end_x]*hyd_value[region6_start_y:region6_end_y, 0]\
        #              + (1 - self.relax_param_EW_external_external[region6_start_y:region6_end_y, (region6_start_x+1):region6_end_x])*hyd_value[region6_start_y:region6_end_y, -1]
        #     ) / 2
        # values_in_frs_zone[region6_start_y:region6_end_y, 
        #                    (region6_start_x+1):region6_end_x] = self.calc_relaxed_values(values=hyd_value,
        #                                                                                  values_east=hyd_value[region6_start_y:region6_end_y, -1],
        #                                                                                  values_west=hyd_value[region6_start_y:region6_end_y, 0],
        #                                                                                  values_south=hyd_value[-1, (region6_start_x+1):region6_end_x],
        #                                                                                  values_north=hyd_value[0, (region6_start_x+1):region6_end_x],
        #                                                                                  start_x_idx=(region6_start_x+1),
        #                                                                                  end_x_idx=region6_end_x,
        #                                                                                  start_y_idx= region6_start_y,
        #                                                                                  end_y_idx= region6_end_y,
        #                                                                                  relax_param_NS=self.relax_param_NS_external_external[region6_start_y:region6_end_y, 
        #                                                                                                                                       (region6_start_x+1):region6_end_x],
        #                                                                                  relax_param_SN=self.relax_param_SN_external_external[region6_start_y:region6_end_y, 
        #                                                                                                                                       (region6_start_x+1):region6_end_x],
        #                                                                                  relax_param_EW=self.relax_param_EW_external_external[region6_start_y:region6_end_y,
        #                                                                                                                                       (region6_start_x+1):region6_end_x],
        #                                                                                  relax_param_WE=self.relax_param_WE_external_external[region6_start_y:region6_end_y,
        #                                                                                                                                       (region6_start_x+1):region6_end_x]
        #                                                                                 )
        # Region 7
        region7_start_x = self.num_relaxation_grid
        region7_end_x = end_internal_region_x + 1
        region7_start_y = (self.tc_child.grid.shape[0] - self.num_relaxation_grid)
        region7_end_y = -1
        # values_in_frs_zone[region7_start_y:region7_end_y, region7_start_x:region7_end_x] = (
        #     (1 - self.relax_param_SN_external_internal[:region7_end_y, :])*hyd_value[region7_start_y, region7_start_x:region7_end_x]\
        #           + self.relax_param_SN_external_internal[:region7_end_y, :]*hyd_value[-1, region7_start_x:region7_end_x]
        #     + self.relax_param_EW_external_external[region7_start_y:region7_end_y, region7_start_x:region7_end_x]*hyd_value[region7_start_y:region7_end_y, 0][:, None]\
        #           + (1 - self.relax_param_EW_external_external[region7_start_y:region7_end_y, region7_start_x:region7_end_x])*hyd_value[region7_start_y:region7_end_y, -1][:, None]
        # ) / 2
        values_in_frs_zone[region7_start_y:region7_end_y, 
                           region7_start_x:region7_end_x] = self.calc_relaxed_values(values=hyd_value,
                                                                                     values_east=hyd_value[region7_start_y:region7_end_y, -1][:, None],
                                                                                     values_west=hyd_value[region7_start_y:region7_end_y, 0][:, None],
                                                                                     values_south=hyd_value[-1, region7_start_x:region7_end_x],
                                                                                     values_north=hyd_value[region7_start_y-1, region7_start_x:region7_end_x],
                                                                                     start_x_idx=region7_start_x,
                                                                                     end_x_idx=region7_end_x,
                                                                                     start_y_idx=region7_start_y,
                                                                                     end_y_idx= region7_end_y,
                                                                                     relax_param_NS=self.relax_param_NS_external_internal[:region7_end_y, :],
                                                                                     relax_param_SN=self.relax_param_SN_external_internal[:region7_end_y, :],
                                                                                     relax_param_EW=self.relax_param_EW_external_external[region7_start_y:region7_end_y,
                                                                                                                                          region7_start_x:region7_end_x],
                                                                                     relax_param_WE=self.relax_param_WE_external_external[region7_start_y:region7_end_y,
                                                                                                                                          region7_start_x:region7_end_x]
                                                                                    )

        # Region 8
        # region8_start_x = end_internal_region_x + 1
        # region8_end_x = -1
        # region8_start_y = (self.tc_child.grid.shape[0] - self.num_relaxation_grid)
        # region8_end_y = -1
        # values_in_frs_zone[region8_start_y:region8_end_y, region8_start_x:region8_end_x] = (
        #         self.relax_param_NS_external_external[region8_start_y:region8_end_y, region8_start_x:region8_end_x]*hyd_value[0, region8_start_x:region8_end_x]\
        #               + (1 - self.relax_param_NS_external_external[region8_start_y:region8_end_y, region8_start_x:region8_end_x])*hyd_value[-1, region8_start_x:region8_end_x]
        #         + self.relax_param_EW_external_external[region8_start_y:region8_end_y, region8_start_x:region8_end_x]*hyd_value[region8_start_y:region8_end_y, 0]\
        #               + (1 - self.relax_param_EW_external_external[region8_start_y:region8_end_y, region8_start_x:region8_end_x])*hyd_value[region8_start_y:region8_end_y, -1]
        #     ) / 2
        # values_in_frs_zone[region8_start_y:region8_end_y, 
        #                    region8_start_x:region8_end_x] = self.calc_relaxed_values(values=hyd_value,
        #                                                                              values_east=hyd_value[region8_start_y:region8_end_y, -1],
        #                                                                              values_west= hyd_value[region8_start_y:region8_end_y, 0],
        #                                                                              values_south=hyd_value[-1, region8_start_x:region8_end_x],
        #                                                                              values_north=hyd_value[0, region8_start_x:region8_end_x],
        #                                                                              start_x_idx=region8_start_x,
        #                                                                              end_x_idx=region8_end_x,
        #                                                                              start_y_idx=region8_start_y,
        #                                                                              end_y_idx=region8_end_y,
        #                                                                              relax_param_NS=self.relax_param_NS_external_external[region8_start_y:region8_end_y, 
        #                                                                                                                                   region8_start_x:region8_end_x],
        #                                                                              relax_param_SN=self.relax_param_SN_external_external[region8_start_y:region8_end_y,
        #                                                                                                                                   region8_start_x:region8_end_x],
        #                                                                              relax_param_EW=self.relax_param_EW_external_external[region8_start_y:region8_end_y,
        #                                                                                                                                   region8_start_x:region8_end_x],
        #                                                                              relax_param_WE=self.relax_param_WE_external_external[region8_start_y:region8_end_y,
        #                                                                                                                                   region8_start_x:region8_end_x]
        #                                                                             )
        
        # intepolate data at corners of the relaxation zone
        x = np.arange(self.tc_child.grid.shape[1])
        y = np.arange(self.tc_child.grid.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = values_in_frs_zone.copy()

        # mask cornars of the relaxation zone
        region1 = (1 <= X) & (X < self.num_relaxation_grid) & (1 <= Y) & (Y < self.num_relaxation_grid)
        region3 = (end_internal_region_x < X) & (X < self.tc_child.grid.shape[1]-1) & (1 <= Y) & (Y < self.num_relaxation_grid)
        region6 = (1 <= X) & (X < self.num_relaxation_grid) & ((self.tc_child.grid.shape[0]-1 - self.num_relaxation_grid) < Y) & (Y < self.tc_child.grid.shape[0]-1)
        region8 = (end_internal_region_x < X) & (X < self.tc_child.grid.shape[1]-1) & ((self.tc_child.grid.shape[0]-1 - self.num_relaxation_grid) < Y) & (Y < self.tc_child.grid.shape[0]-1)
        mask = region1 | region3 | region6 | region8
        # mask = (
        #     (1 <= X) & (X <= self.num_relaxation_grid) & (1 <= Y) & (Y <= self.num_relaxation_grid) # region 1
        #     & (end_internal_region_x + 1 < X) & (X < self.tc_child.grid.shape[1]) & (1 <= Y) & (Y <= self.num_relaxation_grid) # region 3
        #     & (1 <= X) & (X <= self.num_relaxation_grid) & ((self.tc_child.grid.shape[0] - self.num_relaxation_grid) <= Y) & (Y < self.tc_child.grid.shape[0]) # region 6
        #     & (end_internal_region_x + 1 <= X) & (X < self.tc_child.grid.shape[1]) & ((self.tc_child.grid.shape[0] - self.num_relaxation_grid) <= Y) & (Y < self.tc_child.grid.shape[0]) # region 8
        # )
        Z[mask] = np.nan  # Set corners to NaN for interpolation
        # pdb.set_trace()
        known_mask = ~np.isnan(Z)
        x_known = X[known_mask]
        y_known = Y[known_mask]
        z_known = Z[known_mask]
        points_all = np.column_stack([X.ravel(), Y.ravel()])

        Z_filled = griddata(points=np.column_stack([x_known, y_known]),
                            values=z_known,
                            xi=points_all,
                            method='linear'
                            ).reshape(values_in_frs_zone.shape[0], values_in_frs_zone.shape[1])
        frs_interpolated = Z_filled

        if np.any(np.isnan(frs_interpolated)):
            raise ValueError("Interpolation failed, NaN values found in the relaxation zone.")
        
        return frs_interpolated

    def calc_relax_param(self):
        """Calculate the relaxation parameter based on Equation (4) in Martinsen and Engedahl (1987).
        This method calculates the relaxation parameter based on the number of grid nodes in the child grid.
        """
        num_grid_NS = self.tc_child.grid.shape[0]
        num_grid_EW = self.tc_child.grid.shape[1]
        num_xgrid_NS_external_internal = self.tc_child.grid.shape[1] - self.num_relaxation_grid*2
        num_ygrid_NS_external_internal = self.num_relaxation_grid
        num_xgrid_EW_external_internal = self.num_relaxation_grid
        num_ygrid_EW_external_internal = self.tc_child.grid.shape[0] - self.num_relaxation_grid*2

        # relaxation parameter based on Martinsen and Engedahl (1987)
        self.relax_param_NS_external_external = np.zeros((num_grid_NS, num_grid_EW))
        self.relax_param_SN_external_external = np.zeros_like(self.relax_param_NS_external_external)
        self.relax_param_NS_external_internal = np.zeros((num_ygrid_NS_external_internal, num_xgrid_NS_external_internal))
        self.relax_param_SN_external_internal = np.zeros_like(self.relax_param_NS_external_internal)
        self.relax_param_EW_external_external = np.zeros((num_grid_NS, num_grid_EW))
        self.relax_param_WE_external_external = np.zeros_like(self.relax_param_EW_external_external)
        self.relax_param_EW_external_internal = np.zeros((num_ygrid_EW_external_internal, num_xgrid_EW_external_internal))
        self.relax_param_WE_external_internal = np.zeros_like(self.relax_param_EW_external_internal)

        # calculate the relaxation parameter for grid nodes between the external boundaries of a child grid
        for i in range(1, num_grid_NS+1):
            # self.relax_param_NS_external_external[i-1, :] = ((num_grid_NS - i + 1)/num_grid_NS)**2
            self.relax_param_NS_external_external[i-1, :] = 1 - np.tanh((i-1)/2)
        for j in range(1, num_grid_EW+1):
            # self.relax_param_EW_external_external[:, j-1] = ((num_grid_EW - j + 1)/num_grid_EW)**2
            self.relax_param_EW_external_external[:, j-1] = 1 - np.tanh((j-1)/2)
        self.relax_param_SN_external_external[:, :] = np.flipud(self.relax_param_NS_external_external)
        self.relax_param_WE_external_external[:, :] = np.fliplr(self.relax_param_EW_external_external)

        # calculate the relaxation parameter for grid nodes between the external boundary and the interior of the child grid
        for i in range(1, self.num_relaxation_grid+1):
            # self.relax_param_NS_external_internal[i-1, :] = ((self.num_relaxation_grid - i + 1)/self.num_relaxation_grid)**2
            self.relax_param_NS_external_internal[i-1, :] = 1 - np.tanh((i-1)/2)
        for j in range(1, self.num_relaxation_grid+1):
            # self.relax_param_EW_external_internal[:, j-1] = ((self.num_relaxation_grid - j + 1)/self.num_relaxation_grid)**2
            self.relax_param_WE_external_internal[:, j-1] = 1 - np.tanh((j-1)/2)
        self.relax_param_SN_external_internal[:, :] = np.flipud(self.relax_param_NS_external_internal)
        self.relax_param_EW_external_internal[:, :] = np.fliplr(self.relax_param_WE_external_internal)

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