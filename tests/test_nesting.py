"""Regression tests for one-way and two-way nesting.

The tests cover the following bugs:

1. Updating the boundary of a child grid overwrote Ch_i at the interior nodes
   with C_i * h, although C_i is not updated until the end of run_one_step().
2. The flow relaxation scheme (FRS) did not update Ch_i, which is the
   prognostic variable of the model.
3. Enabling nesting changed the solution of a parent grid, because link values
   were replaced with the mean of the node values at every run_one_step().
"""
import numpy as np
import pytest
import yaml
from landlab import RasterModelGrid

import turb2d.gridutils as gridutils
from turb2d import TurbidityCurrent2D
from turb2d.nesting import OneWayNesting
from turb2d.utils import (
    initialize_grid_fields,
    set_boundary_condition,
    set_inlet_condition,
    set_inlet_region,
)

DX_PARENT = 100.0
DX_CHILD = 50.0
SLOPE = 0.05
DT = 2.0
GSIZE = 2

MODEL_PARAM = {
    "h_init": 0.0,
    "Ch_w": 1.0e-9,
    "h_w": 0.001,
    "alpha": 0.08,
    "Cf": 0.004,
    "g": 9.81,
    "R": 1.65,
    "Ds": [1.0e-4, 2.0e-5],
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
    "bedload_transport": False,
    "model": "4eq",
    "salt": False,
    "alpha_4eq": 0.6,
    "karman": 0.4,
    "p": 1.0,
    "flow_type": "continuous",
    "multi_flow": False,
    "active_layer_fraction": None,
}


def write_config(path, boundary, one_way=True, two_way=False):
    """Write a configuration file, and return its path as a string."""
    config = {
        "inlet_condition": {
            "inlet_edge": None,
            "inlet_width": 1000.0,
            "h_ini": 30.0,
            "C_ini": [0.005, 0.005],
            "U_ini": 5.0,
        },
        "boundary_condition": boundary,
        "model_param": dict(
            MODEL_PARAM, one_way_nesting=one_way, two_way_nesting=two_way
        ),
    }
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


def make_grid(shape, spacing, lower_left, y_top):
    """Make a grid on a plane that slopes down toward the bottom edge."""
    grid = RasterModelGrid(shape, xy_spacing=spacing, xy_of_lower_left=lower_left)
    grid.add_field(
        "topographic__elevation",
        1000.0 - SLOPE * (y_top - grid.y_of_node),
        at="node",
    )
    grid.add_zeros("bed__thickness", at="node")
    return grid


def make_parent(tmp_path, shape=(41, 21), one_way=True):
    """Make a parent grid with a continuous inflow from the top edge."""
    config = write_config(
        tmp_path / "parent_{}.yml".format(one_way),
        {
            "top_edge": "fixed_value",
            "bottom_edge": "fixed_gradient",
            "left_edge": "fixed_gradient",
            "right_edge": "fixed_gradient",
        },
        one_way=one_way,
    )
    y_top = (shape[0] - 1) * DX_PARENT
    grid = make_grid(shape, DX_PARENT, (0.0, 0.0), y_top)
    initialize_grid_fields(grid, config)
    inlet, inlet_link = set_inlet_region(grid, config)
    set_inlet_condition(grid=grid, inlet=inlet, inlet_link=inlet_link, config_file=config)
    set_boundary_condition(grid=grid, config_file=config)
    return TurbidityCurrent2D(
        grid, config_path=config, parent_grid=True, child_grid=False
    )


def make_child(tmp_path, region, y_top, one_way=True, two_way=False):
    """Make a child grid that covers region = [xmin, xmax, ymin, ymax]."""
    config = write_config(
        tmp_path / "child.yml",
        {
            "top_edge": "fixed_value",
            "bottom_edge": "fixed_value",
            "left_edge": "fixed_value",
            "right_edge": "fixed_value",
        },
        one_way=one_way,
        two_way=two_way,
    )
    xmin, xmax, ymin, ymax = region
    shape = (int((ymax - ymin) / DX_CHILD) + 1, int((xmax - xmin) / DX_CHILD) + 1)
    grid = make_grid(shape, DX_CHILD, (xmin, ymin), y_top)
    initialize_grid_fields(grid, config)
    set_boundary_condition(grid=grid, config_file=config)
    return TurbidityCurrent2D(
        grid, config_path=config, parent_grid=False, child_grid=True
    )


def make_nesting(tmp_path, num_relaxation_grid=0):
    """Make a pair of parent and child grids and their nesting object."""
    parent = make_parent(tmp_path)
    child = make_child(tmp_path, [500.0, 1500.0, 1500.0, 3500.0], y_top=4000.0)
    nesting = OneWayNesting(
        tc_parent=parent,
        tc_child=child,
        dt=DT,
        num_relaxation_grid=num_relaxation_grid,
    )
    return parent, child, nesting


def perimeter_mask(tc):
    """Return a boolean array that is True at the nodes on the grid edges."""
    mask = np.zeros(tc.grid.number_of_nodes, dtype=bool)
    mask[tc.grid.nodes_at_top_edge] = True
    mask[tc.grid.nodes_at_bottom_edge] = True
    mask[tc.grid.nodes_at_left_edge] = True
    mask[tc.grid.nodes_at_right_edge] = True
    return mask


def index_distance_to_edge(tc):
    """Return the number of nodes between each node and the nearest edge."""
    rows, cols = tc.grid.shape
    row = np.arange(rows)[:, None]
    col = np.arange(cols)[None, :]
    return np.minimum(
        np.minimum(row, rows - 1 - row), np.minimum(col, cols - 1 - col)
    ).flatten()


def set_child_conditions(child, h=2.0):
    """Set the one-level conditions given from the parent grid."""
    n = child.grid.number_of_nodes
    child.h_node_child_grid_condition = np.full((1, n), h)
    child.u_node_child_grid_condition = np.full((1, n), 0.5)
    child.v_node_child_grid_condition = np.full((1, n), -1.0)
    child.Kh_node_child_grid_condition = np.full((1, n), 0.1)
    child.bed_thick_i_node_child_grid_condition = np.zeros((1, GSIZE, n))
    child.C_i_node_child_grid_condition = np.zeros((1, GSIZE, n))
    for j in range(GSIZE):
        child.C_i_node_child_grid_condition[0, j, :] = 0.01 * (j + 1)


def set_child_external_values(child, h, C_i):
    """Set the values interpolated from the parent grid for the FRS."""
    n = child.grid.number_of_nodes
    child.h_node_external = h
    child.u_node_external = np.full(n, 0.5)
    child.v_node_external = np.full(n, -1.0)
    child.Kh_node_external = np.full(n, 0.1)
    child.C_i_node_external = C_i
    child.bed_thick_i_external = np.zeros((GSIZE, n))


class TestUpdateBoundaryConditions:
    """Bug 1: the boundary update must not change the interior Ch_i."""

    def test_interior_Ch_i_is_kept(self, tmp_path):
        _, child, nesting = make_nesting(tmp_path)
        set_child_conditions(child)
        interior = ~perimeter_mask(child)

        # C_i is intentionally inconsistent with Ch_i, as it is in the local
        # time steps after the first one of run_one_step().
        child.h[:] = 3.0
        child.C_i[:, :] = 0.5
        expected = np.array([[0.1], [0.2]]) * np.ones_like(child.Ch_i)
        child.Ch_i[:, :] = expected

        nesting.update_boundary_conditions(child, 0)

        np.testing.assert_array_equal(child.Ch_i[:, interior], expected[:, interior])

    def test_boundary_Ch_i_is_given_by_parent(self, tmp_path):
        _, child, nesting = make_nesting(tmp_path)
        set_child_conditions(child, h=2.0)
        boundary = perimeter_mask(child)
        child.h[:] = 3.0
        child.C_i[:, :] = 0.5
        child.Ch_i[:, :] = 0.1

        nesting.update_boundary_conditions(child, 0)

        for j in range(GSIZE):
            np.testing.assert_allclose(child.Ch_i[j, boundary], 2.0 * 0.01 * (j + 1))

    def test_Ch_is_sum_of_Ch_i(self, tmp_path):
        _, child, nesting = make_nesting(tmp_path)
        set_child_conditions(child)
        child.Ch_i[:, :] = 0.1

        nesting.update_boundary_conditions(child, 0)

        np.testing.assert_allclose(child.Ch, np.sum(child.Ch_i, axis=0))


class TestFlowRelaxationScheme:
    """Bug 2: the FRS must relax Ch_i toward the values of the parent grid."""

    NUM_RELAXATION_GRID = 3

    def relax(self, tmp_path):
        """Apply the FRS, and return the child grid with input values."""
        _, child, nesting = make_nesting(
            tmp_path, num_relaxation_grid=self.NUM_RELAXATION_GRID
        )
        n = child.grid.number_of_nodes
        h_ext = np.full(n, 2.0)
        C_i_ext = np.array([[0.01], [0.02]]) * np.ones((GSIZE, n))
        set_child_external_values(child, h_ext, C_i_ext)

        h_int = np.full(n, 5.0)
        Ch_i_int = np.array([[0.03], [0.04]]) * np.ones((GSIZE, n))
        child.h[:] = h_int
        child.Ch_i[:, :] = Ch_i_int
        child.C_i[:, :] = 0.5  # outdated values must not affect Ch_i

        nesting.flow_relaxation_scheme()
        return child, h_ext, C_i_ext, h_int, Ch_i_int

    def test_perimeter_Ch_i_equals_external_value(self, tmp_path):
        child, h_ext, C_i_ext, _, _ = self.relax(tmp_path)
        perimeter = perimeter_mask(child)
        np.testing.assert_allclose(
            child.Ch_i[:, perimeter], (C_i_ext * h_ext)[:, perimeter]
        )

    def test_zone_Ch_i_is_blended(self, tmp_path):
        child, h_ext, C_i_ext, _, Ch_i_int = self.relax(tmp_path)
        distance = index_distance_to_edge(child)
        zone = distance < self.NUM_RELAXATION_GRID
        alpha = 1.0 - np.tanh(distance / 2.0)
        expected = alpha * (C_i_ext * h_ext) + (1.0 - alpha) * Ch_i_int
        np.testing.assert_allclose(child.Ch_i[:, zone], expected[:, zone])

    def test_interior_Ch_i_is_kept(self, tmp_path):
        child, _, _, _, Ch_i_int = self.relax(tmp_path)
        interior = index_distance_to_edge(child) >= self.NUM_RELAXATION_GRID
        np.testing.assert_array_equal(child.Ch_i[:, interior], Ch_i_int[:, interior])

    def test_Ch_is_sum_of_Ch_i(self, tmp_path):
        child, _, _, _, _ = self.relax(tmp_path)
        np.testing.assert_allclose(child.Ch, np.sum(child.Ch_i, axis=0))

    def test_zone_C_i_is_consistent_with_Ch_i(self, tmp_path):
        child, _, _, _, _ = self.relax(tmp_path)
        zone = index_distance_to_edge(child) < self.NUM_RELAXATION_GRID
        np.testing.assert_allclose(
            child.C_i[:, zone], child.Ch_i[:, zone] / child.h[zone]
        )

    def test_dry_zone_C_i_is_initial_value(self, tmp_path):
        _, child, nesting = make_nesting(
            tmp_path, num_relaxation_grid=self.NUM_RELAXATION_GRID
        )
        n = child.grid.number_of_nodes
        set_child_external_values(child, np.zeros(n), np.zeros((GSIZE, n)))
        child.h[:] = 0.0
        child.Ch_i[:, :] = 0.0
        child.C_i[:, :] = 0.5

        nesting.flow_relaxation_scheme()

        zone = index_distance_to_edge(child) < self.NUM_RELAXATION_GRID
        np.testing.assert_array_equal(child.C_i[:, zone], child.C_init)


class TestNodeToLinkMapping:
    """Bug 3: the mapping for nesting must be applied to a child grid only."""

    @staticmethod
    def spy_on_mapping(monkeypatch):
        """Record the calls that map the node velocities and Kh to links."""
        calls = []
        original = gridutils.map_nodes_to_links

        def spy(tc, **kwargs):
            if kwargs.get("Kh_node") is not None:
                calls.append(tc)
            return original(tc, **kwargs)

        monkeypatch.setattr(gridutils, "map_nodes_to_links", spy)
        return calls

    def test_parent_solution_does_not_depend_on_nesting_flag(self, tmp_path):
        nested = make_parent(tmp_path, one_way=True)
        plain = make_parent(tmp_path, one_way=False)

        for i in range(15):
            nested.run_one_step(dt=DT, repeat=0, last=i + 1)
            plain.run_one_step(dt=DT, repeat=0, last=i + 1)

        assert np.count_nonzero(plain.h > plain.h_w) > 0
        np.testing.assert_array_equal(nested.h, plain.h)
        np.testing.assert_array_equal(nested.u, plain.u)
        np.testing.assert_array_equal(nested.v, plain.v)
        np.testing.assert_array_equal(nested.Ch_i, plain.Ch_i)
        np.testing.assert_array_equal(nested.Kh, plain.Kh)

    def test_mapping_is_not_applied_to_parent(self, tmp_path, monkeypatch):
        parent = make_parent(tmp_path, one_way=True)
        calls = self.spy_on_mapping(monkeypatch)

        parent.run_one_step(dt=DT, repeat=0, last=1)

        assert calls == []

    def test_mapping_is_applied_to_child(self, tmp_path, monkeypatch):
        parent, child, nesting = make_nesting(tmp_path)
        parent.run_one_step(dt=DT, repeat=0, last=1)
        nesting.compute_child_grid_condition_from_parent_grid(time_step=DT / 10)
        calls = self.spy_on_mapping(monkeypatch)

        child.run_one_step(dt=DT, repeat=0, last=1, one_way_nesting_method=nesting)

        assert child in calls


class TestTwoWayNestingBoundaryUpdate:
    """Bug 1 in the boundary update for two-way nesting in run_one_step()."""

    class Stop(Exception):
        """Raised to leave run_one_step() after the boundary update."""

    def test_interior_Ch_i_is_kept(self, tmp_path):
        child = make_child(
            tmp_path,
            [500.0, 1500.0, 1500.0, 3500.0],
            y_top=4000.0,
            one_way=False,
            two_way=True,
        )
        n = child.grid.number_of_nodes
        child.first_nesting = False
        child.h_node_child_grid_condition = np.full(n, 2.0)
        child.u_node_child_grid_condition = np.zeros(n)
        child.v_node_child_grid_condition = np.full(n, -1.0)
        child.Kh_node_child_grid_condition = np.zeros(n)
        child.bed_thick_node_child_grid_condition = np.zeros(n)
        child.bed_thick_i_node_child_grid_condition = np.zeros((GSIZE, n))
        child.C_i_node_child_grid_condition = np.array([[0.01], [0.02]]) * np.ones(
            (GSIZE, n)
        )
        child.grid.at_node["flow__depth"][:] = 5.0
        for j in range(GSIZE):
            child.grid.at_node["flow__sediment_concentration_{}".format(j)][:] = 0.01
        child.grid.at_link["flow__vertical_velocity"][:] = -1.0

        # copy_values_to_temp() is called just before the time loop, and just
        # after the boundary update. The first call is used to give Ch_i that
        # differs from C_i * h as it does after the first local time step,
        # and the second call is used to obtain Ch_i after the boundary update.
        interior = ~perimeter_mask(child)
        original = child.copy_values_to_temp
        expected = 0.123
        result = {}

        def spy():
            original()
            if "count" not in result:
                result["count"] = 1
                child.Ch_i[:, interior] = expected
            else:
                result["Ch_i"] = child.Ch_i.copy()
                raise self.Stop

        child.copy_values_to_temp = spy
        with pytest.raises(self.Stop):
            child.run_one_step(dt=DT, repeat=0, last=1)

        np.testing.assert_array_equal(result["Ch_i"][:, interior], expected)
        np.testing.assert_allclose(result["Ch_i"][:, ~interior][0], 2.0 * 0.01)
        np.testing.assert_allclose(result["Ch_i"][:, ~interior][1], 2.0 * 0.02)


@pytest.mark.parametrize("num_relaxation_grid", [0, 3])
def test_child_sediment_follows_parent(tmp_path, num_relaxation_grid):
    """All the fixes together: the total of Ch in a child grid follows that in
    the parent grid at the common nodes in the core of the child grid.
    """
    parent, child, nesting = make_nesting(
        tmp_path, num_relaxation_grid=num_relaxation_grid
    )

    for i in range(150):
        parent.run_one_step(dt=DT, repeat=0, last=i + 1)
        nesting.compute_child_grid_condition_from_parent_grid(time_step=DT / 100)
        child.run_one_step(
            dt=DT, repeat=0, last=i + 1, one_way_nesting_method=nesting
        )

    parent_nodes = nesting.nested_region_idx[0]
    child_nodes = np.array(
        [
            np.where(
                (child.grid.x_of_node == parent.grid.x_of_node[node])
                & (child.grid.y_of_node == parent.grid.y_of_node[node])
            )[0][0]
            for node in parent_nodes
        ]
    )
    core = ~perimeter_mask(child)[child_nodes]
    Ch_parent = np.sum(parent.Ch[parent_nodes[core]])
    Ch_child = np.sum(child.Ch[child_nodes[core]])

    assert Ch_parent > 0.0
    assert Ch_child == pytest.approx(Ch_parent, rel=0.2)
