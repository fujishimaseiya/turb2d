"""Tests for the bedload transport in calc_deposition.

The tests cover the following bugs:

1. The divergence of bedload flux was added to the bed change with the wrong
   sign, so that the bed aggraded where the bedload flux increased downstream.
2. The bedload flux at nodes that were not wet kept the values of the previous
   steps, and the central difference across the wet/dry front did not
   conserve bedload.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from landlab import RasterModelGrid

import turb2d.gridutils as gridutils
import turb2d.turb2d as turb2d_module
from turb2d import TurbidityCurrent2D
from turb2d.utils import initialize_grid_fields, set_boundary_condition

DX = 10.0
SHAPE = (8, 12)

MODEL_PARAM = {
    "h_init": 0.0,
    "Ch_w": 1.0e-9,
    "h_w": 0.001,
    "alpha": 0.08,
    "Cf": 0.004,
    "g": 9.81,
    "R": 1.65,
    "Ds": [1.0e-3],
    "lambda_p": 0.4,
    "r0": 2.0,
    "nu": 1.010e-6,
    "kappa": 0.05,
    "nu_a": 0.75,
    "implicit_num": 500,
    "implicit_threshold": 1.0e-12,
    "C_init": 0.0,
    "gamma": 0.35,
    "la": 0.003,
    "water_entrainment": True,
    "water_detrainment": True,
    "detrainment_coef": 1.0,
    "suspension": True,
    "sed_entrainment_func": "GP1991exp",
    "no_erosion": False,
    "bedload_transport": True,
    "model": "3eq",
    "salt": False,
    "alpha_4eq": 0.6,
    "karman": 0.4,
    "p": 1.0,
    "flow_type": "continuous",
    "multi_flow": False,
    "active_layer_fraction": None,
    "one_way_nesting": False,
    "two_way_nesting": False,
}


def make_grid():
    grid = RasterModelGrid(SHAPE, xy_spacing=DX)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("bed__thickness", at="node")
    return grid


def make_stub(grid):
    """Make an object that has the attributes used by the divergence."""
    neighbors = grid.adjacent_nodes_at_node
    return SimpleNamespace(
        grid=grid,
        node_east=neighbors[:, 0],
        node_north=neighbors[:, 1],
        node_west=neighbors[:, 2],
        node_south=neighbors[:, 3],
    )


def divergence(stub, qx, qy, nodes):
    return TurbidityCurrent2D._calc_bedload_divergence(stub, qx, qy, nodes)


def test_divergence_is_central_difference_in_wet_region():
    grid = make_grid()
    stub = make_stub(grid)
    nodes = grid.core_nodes
    qx = (grid.x_of_node ** 2)[np.newaxis, :]
    qy = (2.0 * grid.y_of_node)[np.newaxis, :]

    div = divergence(stub, qx.copy(), qy.copy(), nodes)

    east, west = stub.node_east[nodes], stub.node_west[nodes]
    north, south = stub.node_north[nodes], stub.node_south[nodes]
    # Interior nodes whose neighbors are all core nodes
    interior = np.isin(east, nodes) & np.isin(west, nodes) & np.isin(
        north, nodes) & np.isin(south, nodes)
    expected = (qx[0, east] - qx[0, west]) / (2 * DX) + (
        qy[0, north] - qy[0, south]) / (2 * DX)
    np.testing.assert_allclose(div[0, interior], expected[interior])


def test_divergence_conserves_bedload_at_wet_dry_front():
    grid = make_grid()
    stub = make_stub(grid)
    core = grid.core_nodes
    # wet patch away from the grid edges
    in_patch = (
        (grid.x_of_node[core] >= 2 * DX) & (grid.x_of_node[core] <= 7 * DX)
        & (grid.y_of_node[core] >= 2 * DX) & (grid.y_of_node[core] <= 5 * DX)
    )
    nodes = core[in_patch]
    rng = np.random.default_rng(0)
    qx = rng.uniform(-1.0, 1.0, (2, grid.number_of_nodes))
    qy = rng.uniform(-1.0, 1.0, (2, grid.number_of_nodes))

    div = divergence(stub, qx, qy, nodes)

    # No bedload is lost or created through the wet/dry front
    np.testing.assert_allclose(div.sum(axis=1), 0.0, atol=1.0e-12)


def test_divergence_ignores_values_at_dry_nodes():
    grid = make_grid()
    stub = make_stub(grid)
    core = grid.core_nodes
    nodes = core[grid.x_of_node[core] <= 5 * DX]
    dry = np.setdiff1d(core, nodes)
    qx = np.ones((1, grid.number_of_nodes))
    qy = np.zeros((1, grid.number_of_nodes))

    div_clean = divergence(stub, np.where(np.isin(np.arange(qx.size), dry),
                                          0.0, qx), qy.copy(), nodes)
    qx_stale = qx.copy()
    qx_stale[:, dry] = 100.0
    div_stale = divergence(stub, qx_stale, qy.copy(), nodes)

    np.testing.assert_allclose(div_stale, div_clean)


def test_uniform_flux_leaves_through_edges_without_bed_change():
    grid = make_grid()
    stub = make_stub(grid)
    nodes = grid.core_nodes
    qx = np.ones((1, grid.number_of_nodes))
    qy = np.full((1, grid.number_of_nodes), 0.5)
    qx[:, grid.boundary_nodes] = 0.0
    qy[:, grid.boundary_nodes] = 0.0

    div = divergence(stub, qx, qy, nodes)

    np.testing.assert_allclose(div, 0.0, atol=1.0e-12)


@pytest.fixture
def tc(tmp_path):
    config = {
        "inlet_condition": {
            "inlet_edge": None,
            "inlet_width": 10.0,
            "h_ini": 1.0,
            "C_ini": [0.0],
            "U_ini": 1.0,
        },
        "boundary_condition": {
            "top_edge": "fixed_gradient",
            "bottom_edge": "fixed_gradient",
            "left_edge": "fixed_gradient",
            "right_edge": "fixed_gradient",
        },
        "model_param": MODEL_PARAM,
    }
    path = tmp_path / "config.yml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    grid = make_grid()
    initialize_grid_fields(grid, str(path))
    set_boundary_condition(grid=grid, config_file=str(path))
    return TurbidityCurrent2D(
        grid, config_path=str(path), parent_grid=False, child_grid=False
    )


def test_bed_erodes_where_bedload_increases_downstream(tc, monkeypatch):
    # Remove suspended sediment exchange and bed diffusion to see bedload only
    def no_entrainment(Ch, **kwargs):
        zeros = np.zeros_like(Ch)
        return zeros, zeros, zeros

    monkeypatch.setattr(turb2d_module, "get_es", no_entrainment)
    monkeypatch.setattr(tc, "_bed_diffusion_at_high_slope", lambda: None)

    grid = tc.grid
    n = grid.number_of_nodes
    gridutils.set_up_neighbor_arrays(tc)
    tc.wet_nodes = grid.core_nodes
    tc.dt_local = 1.0
    tc.Cf_node[:] = 0.004
    tc.bed_active_layer[:, :] = 1.0
    # flow accelerates toward the east
    u_node = 1.0 + grid.x_of_node / grid.x_of_node.max()
    v_node = np.zeros(n)
    h = np.ones(n)
    Ch_i = np.zeros((1, n))
    eta = np.zeros(n)
    bed_thick_i = np.zeros((1, n))
    # stale bedload from previous steps must not affect the result
    tc.bedload_x[:, :] = 100.0
    tc.bedload_y[:, :] = 100.0

    out_eta = np.zeros(n)
    tc.calc_deposition(
        h, Ch_i, u_node, v_node, eta, bed_thick_i, np.abs(u_node),
        out_Ch_i=np.zeros((1, n)), out_eta=out_eta,
        out_bed_thick_i=np.zeros((1, n)),
    )

    assert np.all(tc.bed_change_i[0, tc.wet_nodes] < 0.0)
    assert np.all(out_eta[tc.wet_nodes] < 0.0)
