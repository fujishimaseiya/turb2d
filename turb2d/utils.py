"""A module for TurbidityCurrent2D to produce a grid object from a geotiff
   file or from scratch.

   codeauthor: : Hajime Naruse
"""

from landlab import RasterModelGrid
import numpy as np
from scipy.ndimage import median_filter, zoom
from landlab import FieldError
from decimal import Decimal
import os
import yaml
import rasterio
from scipy.interpolate import LinearNDInterpolator
from fractions import Fraction

def create_topography(
    config_file=None,
    length=8000,
    width=2000,
    spacing=20,
    slope_outside=0.1,
    slope_inside=0.05,
    slope_basin=0.02,
    slope_basin_break=2000,
    canyon_basin_break=2200,
    canyon_center=1000,
    canyon_half_width=100,
    canyon="parabola",
    noise=0.01,
):
    """create an artificial topography where a turbidity current flow down
       A slope and a flat basin plain are set in calculation domain, and a
       parabola or v-shaped canyon is created in the slope.

       Parameters
       ------------------
        length: float, optional
           length of calculation domain [m]

        width: float, optional
           width of calculation domain [m]

        spacing: float, optional
           grid spacing [m]

        slope_outside: float, optional
           topographic inclination in the region outside the canyon

        slope_inside: float, optional
           topographic inclination in the region inside the thalweg of
           the canyon

        slope_basin: float, optional
           topographic inclination of the basin plain

        slope_basin_break: float, optional
           location of slope-basin break

        canyon_basin_break: float, optional
           location of canyon-basin break. This value must be
           larger than slope-basin break.

        canyon_center: float, optional
           location of center of the canyon

        canyon_half_width: float, optional
           half width of the canyon

        canyon: String, optional
           Style of the canyon. 'parabola' or 'V' can be chosen.

        random: float, optional
           Range of random noise to be added on generated topography

        Return
        -------------------------
        grid: RasterModelGrid
           a landlab grid object. Topographic elevation is stored as
           grid.at_node['topographic__elevation']


    """
    if os.path.exists(config_file):
        with open(config_file) as yml:
            config = yaml.safe_load(yml)
        length=config['grid_param']['flume_length']
        width=config['grid_param']['flume_width']
        spacing=config['grid_param']['spacing']
        slope_outside=config['grid_param']['slope_outside']
        slope_inside=config['grid_param']['slope_inside']
        slope_basin=config['grid_param']['slope_basin']
        slope_basin_break=config['grid_param']['slope_basin_break']
        canyon_basin_break=config['grid_param']['canyon_basin_break']
        canyon_center=config['grid_param']['canyon_center']
        canyon_half_width=config['grid_param']['canyon_half_width']
        canyon=config['grid_param']['canyon']
        noise=config['grid_param']['noise']
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m
    length = Decimal(str(length))
    width = Decimal(str(width))
    spacing = Decimal(str(spacing))
    lgrids = length / spacing
    wgrids = width / spacing
    grid = RasterModelGrid((lgrids+1, wgrids+1), xy_spacing=[spacing, spacing])
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")

    # making topography
    # set the slope
    grid.at_node["topographic__elevation"] = (
        grid.node_y - slope_basin_break
    ) * slope_outside

    if canyon == "parabola":
        # set canyon
        d0 = slope_inside * (canyon_basin_break - slope_basin_break)
        d = slope_inside * (grid.node_y - canyon_basin_break) - d0
        a = d0 / canyon_half_width ** 2
        canyon_elev = a * (grid.node_x - canyon_center) ** 2 + d
        inside = np.where(canyon_elev < grid.at_node["topographic__elevation"])
        grid.at_node["topographic__elevation"][inside] = canyon_elev[inside]

    elif canyon == "rectangular":
        # Set canyon (rectangular shape with constant width)
        d0 = slope_inside * (canyon_basin_break - slope_basin_break)
        d = slope_inside * (grid.node_y - canyon_basin_break) - d0

        # Define canyon lateral limits
        x_min = canyon_center - canyon_half_width
        x_max = canyon_center + canyon_half_width

        # Logical mask for canyon area
        canyon_mask = (grid.node_x >= x_min) & (grid.node_x <= x_max)
        # Elevation inside canyon region
        canyon_elev = d

        # Apply canyon elevation where it is lower than current surface
        current_elev = grid.at_node["topographic__elevation"]
        new_elev = np.where(canyon_mask, canyon_elev, current_elev)
        grid.at_node["topographic__elevation"] = np.minimum(current_elev, new_elev)

    # set basin
    basin_height = (grid.node_y - slope_basin_break) * slope_basin
    basin_region = grid.at_node["topographic__elevation"] < basin_height
    grid.at_node["topographic__elevation"][basin_region] = basin_height[basin_region]

    # add random value on topographic elevation (+- noise)
    grid.at_node["topographic__elevation"] += (
        2.0 * noise * (np.random.rand(grid.number_of_nodes) - 0.5)
    )

    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    return grid

def create_nested_grid(config_file=None,
                       length=8000,
                       width=2000,
                       spacing=20,
                       slope_outside=0.1,
                       slope_inside=0.05,
                       slope_basin=0.02,
                       slope_basin_break=2000,
                       canyon_basin_break=2200,
                       canyon_center=1000,
                       canyon_half_width=100,
                       canyon="parabola",
                       noise=0.01,
                       nested_region=[0.5, 1.5, 0.5, 1.5], 
                       child_grid_spacing=0.01
                       ):
    """
    Create a nested grid in a region of interest

    Parameters
    ----------------------
    config_file: String, optional
        path to a configuration file

    length: float, optional
        length of calculation domain of parent grid [m]
    
    width: float, optional
        width of calculation domain of parent grid [m]
    
    spacing: float, optional
        grid spacing of parent grid [m]
    
    slope_outside: float, optional
        topographic inclination in the region outside the canyon
    
    slope_inside: float, optional
        topographic inclination in the region inside the thalweg of the canyon

    slope_basin: float, optional
        topographic inclination of the basin plain
    
    slope_basin_break: float, optional
        location of slope-basin break point

    canyon_basin_break: float, optional
        location of canyon-basin break point. This value must be larger than slope-basin break point
    
    canyon_center: float, optional
        location of center of the canyon
    
    canyon_half_width: float, optional
        half width of the canyon

    canyon: String, optional
        Style of the canyon. 'parabola' or 'V' can be chosen.
    
    noise: float, optional
        Range of random noise to be added on generated topography

    nested_region: list, optional
        [xmin, xmax, ymin, ymax] of the region of interest in the parent grid
    
    child_grid_spacing: float, optional
        grid spacing of the child grid [m]
    
    Returns
    ----------------------

    parent_grid: RasterModelGrid
        a parent grid object
    
    child_grid: RasterModelGrid
        a child grid object
    """

    # create parent grid
    parent_grid = create_topography(
        config_file=config_file,
        length=length,
        width=width,
        spacing=spacing,
        slope_outside=slope_outside,
        slope_inside=slope_inside,
        slope_basin=slope_basin,
        slope_basin_break=slope_basin_break,
        canyon_basin_break=canyon_basin_break,
        canyon_center=canyon_center,
        canyon_half_width=canyon_half_width,
        canyon=canyon,
        noise=noise
    )

    # open configuration file
    if config_file is not None:
        with open(config_file) as yml:
            config = yaml.safe_load(yml)
        child_grid_spacing = config['grid_param']['child_grid_spacing']
        xmin = config['grid_param']['nested_region_xmin']
        xmax = config['grid_param']['nested_region_xmax']
        ymin = config['grid_param']['nested_region_ymin']
        ymax = config['grid_param']['nested_region_ymax']
    else:
        # Get indices of the region of interest
        xmin, xmax, ymin, ymax = nested_region

    # Initialize the child grid
    # lgrids = (ymax - ymin) / child_grid_spacing
    # wgrids = (xmax - xmin) / child_grid_spacing
    ymax = Decimal(str(ymax))
    ymin = Decimal(str(ymin))
    xmax = Decimal(str(xmax))
    xmin = Decimal(str(xmin))
    length = (ymax - ymin)
    width = (xmax - xmin)
    child_grid_length = Decimal(str(length))
    child_grid_width = Decimal(str(width))
    child_grid_spacing = Decimal(str(child_grid_spacing))
    lgrids = child_grid_length / child_grid_spacing
    wgrids = child_grid_width / child_grid_spacing
    child_grid = RasterModelGrid(shape=(lgrids+1, wgrids+1), xy_spacing=[child_grid_spacing, child_grid_spacing], xy_of_lower_left=[xmin, ymin])
    
    # Initialize fields of the child grid
    child_grid.add_zeros("flow__depth", at="node")
    child_grid.add_zeros("topographic__elevation", at="node")
    child_grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    child_grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    child_grid.add_zeros("flow__horizontal_velocity", at="link")
    child_grid.add_zeros("flow__vertical_velocity", at="link")
    child_grid.add_zeros("bed__thickness", at="node")

    # Extract topographic elevation from the parent grid
    nested_region_idx = np.where(
                                (parent_grid.node_x >= float(xmin)) & 
                                (parent_grid.node_x <= float(xmax)) & 
                                (parent_grid.node_y >= float(ymin)) & 
                                (parent_grid.node_y <= float(ymax))
                                )

    parent_topo = parent_grid.at_node["topographic__elevation"][nested_region_idx]
    parent_x = parent_grid.node_x[nested_region_idx]
    parent_y = parent_grid.node_y[nested_region_idx]

    # Interpolate and assign topographic elevation to the child grid
    interp = LinearNDInterpolator(list(zip(parent_x, parent_y)), parent_topo)
    child_topo = interp(list(zip(child_grid.node_x, child_grid.node_y)))
    child_grid.at_node["topographic__elevation"] = child_topo

    return parent_grid, child_grid

def create_child_grid_from_npy(config_file=None, parent_grid_file=None, parent_spacing=0.01, nested_region=[0.5, 1.5, 0.5, 1.5], child_grid_spacing=0.01):
    # open configuration file
    if config_file is not None:
        with open(config_file) as yml:
            config = yaml.safe_load(yml)
        child_grid_spacing = config['grid_param']['child_grid_spacing']
        xmin = config['grid_param']['nested_region_xmin']
        xmax = config['grid_param']['nested_region_xmax']
        ymin = config['grid_param']['nested_region_ymin']
        ymax = config['grid_param']['nested_region_ymax']
    else:
        # Get indices of the region of interest
        xmin, xmax, ymin, ymax = nested_region

    parent_grid = create_topography_from_npy(filename=parent_grid_file, spacing=parent_spacing)

    # Initialize the child grid
    ymax = Decimal(str(ymax))
    ymin = Decimal(str(ymin))
    xmax = Decimal(str(xmax))
    xmin = Decimal(str(xmin))
    length = (ymax - ymin)
    width = (xmax - xmin)
    child_grid_length = Decimal(str(length))
    child_grid_width = Decimal(str(width))
    child_grid_spacing = Decimal(str(child_grid_spacing))
    lgrids = child_grid_length / child_grid_spacing
    wgrids = child_grid_width / child_grid_spacing
    child_grid = RasterModelGrid(shape=(lgrids+1, wgrids+1), xy_spacing=[child_grid_spacing, child_grid_spacing], xy_of_lower_left=[xmin, ymin])
    
    # Initialize fields of the child grid
    child_grid.add_zeros("flow__depth", at="node")
    child_grid.add_zeros("topographic__elevation", at="node")
    child_grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    child_grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    child_grid.add_zeros("flow__horizontal_velocity", at="link")
    child_grid.add_zeros("flow__vertical_velocity", at="link")
    child_grid.add_zeros("bed__thickness", at="node")

    # Extract topographic elevation from the parent grid
    nested_region_idx = np.where(
                                (parent_grid.node_x >= float(xmin)) & 
                                (parent_grid.node_x <= float(xmax)) & 
                                (parent_grid.node_y >= float(ymin)) & 
                                (parent_grid.node_y <= float(ymax))
                                )

    parent_topo = parent_grid.at_node["topographic__elevation"][nested_region_idx]
    parent_x = parent_grid.node_x[nested_region_idx]
    parent_y = parent_grid.node_y[nested_region_idx]

    # Interpolate and assign topographic elevation to the child grid
    interp = LinearNDInterpolator(list(zip(parent_x, parent_y)), parent_topo)
    child_topo = interp(list(zip(child_grid.node_x, child_grid.node_y)))
    child_grid.at_node["topographic__elevation"] = child_topo

    return child_grid

def create_init_flow_region(
    grid,
    initial_flow_concentration=0.02,
    initial_flow_thickness=200,
    initial_region_radius=200,
    initial_region_center=[1000, 7000],
):
    """ making initial flow region in a grid, assuming lock-exchange type initiation
         of a turbidity current. Plan-view morphology of a suspended cloud is a circle,

         Parameters
         ----------------------
         grid: RasterModelGrid
            a landlab grid object

         initial_flow_concentration: float, optional
            initial flow concentration

         initial_flow_thickness: float, optional
            initial flow thickness

         initial_region_radius: float, optional
            radius of initial flow region

         initial_region_center: list, optional
            [x, y] coordinates of center of initial flow region
    """
    # check number of grain size classes
    if type(initial_flow_concentration) is float or type(initial_flow_concentration) is np.float64:
        initial_flow_concentration_i = np.array([initial_flow_concentration])
    else:
        initial_flow_concentration_i = np.array(initial_flow_concentration).reshape(
            len(initial_flow_concentration), 1
        )

    # initialize flow parameters
    for i in range(len(initial_flow_concentration_i)):
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

    # set initial flow region
    initial_flow_region = (
        (grid.node_x - initial_region_center[0]) ** 2
        + (grid.node_y - initial_region_center[1]) ** 2
    ) < initial_region_radius ** 2
    grid.at_node["flow__depth"][initial_flow_region] = initial_flow_thickness
    grid.at_node["flow__depth"][~initial_flow_region] = 0.0
    for i in range(len(initial_flow_concentration_i)):
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            initial_flow_region
        ] = initial_flow_concentration_i[i]
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            ~initial_flow_region
        ] = 0.0
    grid.at_node["flow__sediment_concentration_total"][initial_flow_region] = np.sum(
        initial_flow_concentration_i
    )


def create_topography_from_geotiff(
    geotiff_filename, xlim=None, ylim=None, spacing=500, filter_size=[1, 1]
):
    """create a landlab grid file from a geotiff file

       Parameters
       -----------------------
       geotiff_filename: String
          name of a geotiff-format file to import

       xlim: list, optional
          list [xmin, xmax] to specify x coordinates of a region of interest
             in a geotiff file to import

       ylim: list, optional
          list [ymin, ymax] to specify y coordinates of a region of interest
             in a geotiff file to import

       spacing: float, optional
          grid spacing

       filter_size: list, optional
          [x, y] size of a window used in a median filter.
            This filter is applied for smoothing DEM data.

       Return
       ------------------------
       grid: RasterModelGrid
          a landlab grid object to be used in TurbidityCurrent2D

    """

    # read a geotiff file into ndarray
    with rasterio.open(geotiff_filename) as src:
        topo_data = src.read(1)[::-1, :]
        profile = src.profile
        width = profile["width"]
        height = profile["height"]
        transform = src.transform
        dx = transform[0]
        min_x, max_y = transform * (0, 0)
        max_x, min_y = transform * (width, height)
        xy_of_lower_left = (min_x, min_y)

    # print(topo_data.shape)
    if (xlim is not None) and (ylim is not None):
        topo_data = topo_data[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    # Smoothing by median filter
    topo_data = median_filter(topo_data, size=filter_size)

    # change grid size if the parameter spacing is specified
    if spacing is not None and spacing != dx:
        zoom_factor = dx / spacing
        topo_data = zoom(topo_data, zoom_factor)
        dx = spacing

    grid = RasterModelGrid(
        topo_data.shape, xy_spacing=[dx, dx], xy_of_lower_left=xy_of_lower_left
    )
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")
    grid.at_node["topographic__elevation"][grid.nodes] = topo_data
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")

    return grid

def create_topography_from_npy(filename, spacing):
    ds = np.load(filename)
    topo_data = np.rot90(ds, 1)
    grid = RasterModelGrid(topo_data.shape, xy_spacing=[spacing, spacing])
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")
    grid.at_node["topographic__elevation"][grid.nodes] = topo_data

    return grid