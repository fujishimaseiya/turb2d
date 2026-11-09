"""Regression tests for one-way and two-way nesting.

The tests cover the following bugs:

1. Updating the boundary of a child grid overwrote Ch_i at the interior nodes
   with C_i * h, although C_i is not updated until the end of run_one_step().
2. The flow relaxation scheme (FRS) did not update Ch_i, which is the
   prognostic variable of the model.
3. Enabling nesting changed the solution of a parent grid, because link values
   were replaced with the mean of the node values at every run_one_step().
"""
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from landlab import RasterModelGrid

import turb2d.gridutils as gridutils
from turb2d import TurbidityCurrent2D
from turb2d.nesting import OneWayNesting
from turb2d.utils import (
    create_nested_grid,
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
    child.bed_thick_node_child_grid_condition = np.zeros((1, n))
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
    child.eta_external = child.eta_init.copy()


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


class TestBoundaryKh:
    """Kh at the boundary links of a child grid must be given by the parent grid."""

    def test_child_boundary_links_use_Kh_node(self, tmp_path):
        parent, child, nesting = make_nesting(tmp_path)
        parent.run_one_step(dt=DT, repeat=0, last=1)
        nesting.compute_child_grid_condition_from_parent_grid(time_step=DT / 10)
        child.run_one_step(dt=DT, repeat=0, last=1, one_way_nesting_method=nesting)
        child.Kh_node[:] = 2.0
        child.Kh[:] = 1.0

        child.update_boundary_conditions(Kh=child.Kh)

        # The links next to the corners are also edge links. As for the velocities,
        # the value for the edge links takes priority at those links.
        bound_links = np.setdiff1d(child.fixed_value_links, child.fixed_value_edge_links)
        assert bound_links.size > 0
        np.testing.assert_allclose(child.Kh[bound_links], 2.0 / 3.0 * 2.0 + 1.0 / 3.0 * 1.0)
        np.testing.assert_allclose(child.Kh[child.fixed_value_edge_links], 2.0)

    def test_parent_boundary_links_are_unchanged(self, tmp_path):
        parent = make_parent(tmp_path)
        parent.run_one_step(dt=DT, repeat=0, last=1)
        parent.Kh_node[:] = 2.0
        parent.Kh[:] = 1.0

        parent.update_boundary_conditions(Kh=parent.Kh)

        np.testing.assert_array_equal(parent.Kh[parent.fixed_value_links], 1.0)


class TestFlowRelaxationSchemeAtLinks:
    """The FRS must relax u, v and Kh at links, where they are solved."""

    NUM_RELAXATION_GRID = 3

    def relax(self, tmp_path, num_relaxation_grid):
        """Apply the FRS with uniform external and internal link values."""
        _, child, nesting = make_nesting(
            tmp_path, num_relaxation_grid=num_relaxation_grid
        )
        n = child.grid.number_of_nodes
        set_child_external_values(child, np.full(n, 2.0), np.zeros((GSIZE, n)))
        child.h[:] = 2.0
        child.Ch_i[:, :] = 0.0
        child.u[:] = 3.0
        child.v[:] = 4.0
        child.Kh[:] = 5.0
        nesting.flow_relaxation_scheme()
        node_distance = index_distance_to_edge(child)
        link_distance = np.mean(node_distance[child.grid.nodes_at_link], axis=1)
        return child, link_distance

    def test_zone_links_are_blended(self, tmp_path):
        child, distance = self.relax(tmp_path, self.NUM_RELAXATION_GRID)
        zone = distance < self.NUM_RELAXATION_GRID
        alpha = 1.0 - np.tanh(distance[zone] / 2.0)
        np.testing.assert_allclose(child.u[zone], alpha * 0.5 + (1 - alpha) * 3.0)
        np.testing.assert_allclose(child.v[zone], alpha * -1.0 + (1 - alpha) * 4.0)
        np.testing.assert_allclose(child.Kh[zone], alpha * 0.1 + (1 - alpha) * 5.0)

    def test_links_on_edges_equal_external_value(self, tmp_path):
        child, distance = self.relax(tmp_path, self.NUM_RELAXATION_GRID)
        edge = distance == 0
        assert np.count_nonzero(edge) > 0
        np.testing.assert_allclose(child.v[edge], -1.0)

    def test_interior_links_are_kept(self, tmp_path):
        child, distance = self.relax(tmp_path, self.NUM_RELAXATION_GRID)
        interior = distance >= self.NUM_RELAXATION_GRID
        assert np.count_nonzero(interior) > 0
        np.testing.assert_array_equal(child.u[interior], 3.0)
        np.testing.assert_array_equal(child.v[interior], 4.0)
        np.testing.assert_array_equal(child.Kh[interior], 5.0)

    def test_no_link_changes_without_relaxation_zone(self, tmp_path):
        child, _ = self.relax(tmp_path, 0)
        np.testing.assert_array_equal(child.u, 3.0)
        np.testing.assert_array_equal(child.v, 4.0)
        np.testing.assert_array_equal(child.Kh, 5.0)


class TestFirstNestingLinks:
    """The links of a child grid must be initialized at the first nesting."""

    class Stop(Exception):
        """Raised to leave run_one_step() after the initialization."""

    def run_until_nesting(self, child, **kwargs):
        """Return (u, v, Kh) at the links just after the first nesting."""
        child.grid.at_node["flow__depth"][:] = 5.0
        for j in range(GSIZE):
            child.grid.at_node["flow__sediment_concentration_{}".format(j)][:] = 0.01
        child.grid.at_link["flow__vertical_velocity"][:] = -3.0

        original = child.copy_values_to_temp
        result = {}

        def spy():
            original()
            result["count"] = result.get("count", 0) + 1
            if result["count"] == 2:
                result["values"] = (child.u.copy(), child.v.copy(), child.Kh.copy())
                raise self.Stop

        child.copy_values_to_temp = spy
        with pytest.raises(self.Stop):
            child.run_one_step(dt=DT, repeat=0, last=1, **kwargs)
        return result["values"]

    def test_one_way(self, tmp_path):
        _, child, nesting = make_nesting(tmp_path)
        set_child_conditions(child)
        child.time_interp = np.array([0.0])

        u, v, Kh = self.run_until_nesting(child, one_way_nesting_method=nesting)

        assert child.first_nesting is False
        np.testing.assert_allclose(u, 0.5)
        np.testing.assert_allclose(v, -1.0)
        np.testing.assert_allclose(Kh, 0.1)

    def test_two_way(self, tmp_path):
        child = make_child(
            tmp_path,
            [500.0, 1500.0, 1500.0, 3500.0],
            y_top=4000.0,
            one_way=False,
            two_way=True,
        )
        n = child.grid.number_of_nodes
        child.first_nesting = True
        child.h_node_child_grid_condition = np.full(n, 2.0)
        child.u_node_child_grid_condition = np.full(n, 0.5)
        child.v_node_child_grid_condition = np.full(n, -1.0)
        child.Kh_node_child_grid_condition = np.full(n, 0.1)
        child.bed_thick_node_child_grid_condition = np.zeros(n)
        child.bed_thick_i_node_child_grid_condition = np.zeros((GSIZE, n))
        child.C_i_node_child_grid_condition = np.full((GSIZE, n), 0.01)

        u, v, Kh = self.run_until_nesting(child)

        np.testing.assert_allclose(u, 0.5)
        np.testing.assert_allclose(v, -1.0)
        np.testing.assert_allclose(Kh, 0.1)


class TestBedElevation:
    """The bed elevation of a child grid must follow the parent grid."""

    BED_CHANGE = 0.3

    def test_boundary_gets_change_of_parent_bed(self, tmp_path):
        _, child, nesting = make_nesting(tmp_path)
        set_child_conditions(child)
        child.bed_thick_node_child_grid_condition[:] = self.BED_CHANGE
        eta_before = child.eta.copy()

        nesting.update_boundary_conditions(child, 0)

        perimeter = perimeter_mask(child)
        np.testing.assert_allclose(
            child.eta[perimeter], child.eta_init[perimeter] + self.BED_CHANGE
        )
        np.testing.assert_array_equal(child.eta[~perimeter], eta_before[~perimeter])

    def test_frs_relaxes_eta(self, tmp_path):
        num_relaxation_grid = 3
        _, child, nesting = make_nesting(
            tmp_path, num_relaxation_grid=num_relaxation_grid
        )
        n = child.grid.number_of_nodes
        set_child_external_values(
            child, np.full(n, 2.0), np.full((GSIZE, n), 0.01)
        )
        child.eta_external = child.eta_init + self.BED_CHANGE

        nesting.flow_relaxation_scheme()

        distance = index_distance_to_edge(child)
        change = child.eta - child.eta_init
        np.testing.assert_allclose(change[distance == 0], self.BED_CHANGE)
        blended = (distance > 0) & (distance < num_relaxation_grid)
        assert np.all(change[blended] > 0.0)
        assert np.all(change[blended] < self.BED_CHANGE)
        np.testing.assert_array_equal(change[distance >= num_relaxation_grid], 0.0)

    def test_boundary_update_keeps_eta_of_parent(self, tmp_path):
        _, child, _ = make_nesting(tmp_path)
        gridutils.set_up_neighbor_arrays(child)
        child.eta_external = child.eta_init + self.BED_CHANGE

        child.update_boundary_conditions(eta=child.eta)

        perimeter = perimeter_mask(child)
        np.testing.assert_allclose(
            child.eta[perimeter], child.eta_init[perimeter] + self.BED_CHANGE
        )

    def test_first_nesting_sets_eta_everywhere(self, tmp_path):
        _, child, nesting = make_nesting(tmp_path)
        set_child_conditions(child)
        child.bed_thick_node_child_grid_condition[:] = self.BED_CHANGE
        child.time_interp = np.array([0.0])

        TestFirstNestingLinks().run_until_nesting(
            child, one_way_nesting_method=nesting
        )

        np.testing.assert_allclose(child.eta - child.eta_init, self.BED_CHANGE)

    @pytest.mark.parametrize("num_relaxation_grid", [0, 3])
    def test_child_boundary_bed_follows_parent(self, tmp_path, num_relaxation_grid):
        """At the end of a step, bed_thick at the edges of a child grid equals
        the bed change of the parent grid.
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

        perimeter = perimeter_mask(child)
        expected = child.bed_thick_node_child_grid_condition[-1, perimeter]
        assert np.max(np.abs(expected)) > 0.0
        np.testing.assert_allclose(
            child.bed_thick[perimeter],
            expected,
            atol=0.02 * np.max(np.abs(expected)),
        )


class TestNestedRegionRounding:
    """The nodes of a parent grid on the edges of a child grid must be selected,
    even if the coordinates of the child grid have floating point errors.
    """

    @staticmethod
    def make_nesting_of_region(parent_spacing, ratio, region):
        """Make a nesting object in the same way as create_nested_grid().

        The region is given as [xmin, xmax, ymin, ymax] in parent nodes.
        """
        dx_parent = Decimal(str(parent_spacing))
        dx_child = dx_parent / ratio
        xmin, xmax, ymin, ymax = [dx_parent * index for index in region]
        parent = RasterModelGrid((121, 121), xy_spacing=[dx_parent, dx_parent])
        child = RasterModelGrid(
            ((ymax - ymin) / dx_child + 1, (xmax - xmin) / dx_child + 1),
            xy_spacing=[dx_child, dx_child],
            xy_of_lower_left=[xmin, ymin],
        )
        return OneWayNesting(
            tc_parent=SimpleNamespace(grid=parent),
            tc_child=SimpleNamespace(grid=child),
            dt=DT,
        )

    def test_node_on_edge_with_rounding_error_is_selected(self):
        """xmax of the child grid is 3.5999999999999996 in this region."""
        nesting = self.make_nesting_of_region(0.1, 5, [19, 36, 2, 15])

        assert nesting.tc_child.grid.x_of_node.max() < 3.6
        assert nesting.nested_region_idx[0].size == 18 * 14
        assert nesting.nested_region_idx_except_boundary[0].size == 16 * 12

    @pytest.mark.parametrize(
        "parent_spacing, ratio", [(0.1, 5), (0.01, 2), (0.01, 5), (0.02, 2)]
    )
    def test_all_nodes_in_region_are_selected(self, parent_spacing, ratio):
        for xmin in range(1, 25, 3):
            for width in (10, 17, 30):
                for ymin in (2, 7, 20):
                    for height in (13, 40, 50):
                        region = [xmin, xmin + width, ymin, ymin + height]
                        nesting = self.make_nesting_of_region(
                            parent_spacing, ratio, region
                        )

                        assert nesting.nested_region_idx[0].size == (width + 1) * (
                            height + 1
                        ), region
                        assert nesting.nested_region_idx_except_boundary[0].size == (
                            width - 1
                        ) * (height - 1), region


class TestChildGridRange:
    """A child grid outside its parent grid must be rejected.

    The parent grid covers x = [0, 2000] and y = [0, 4000].
    """

    OUTSIDE = [
        pytest.param([-500.0, 500.0, 1500.0, 3500.0], id="left"),
        pytest.param([1500.0, 2500.0, 1500.0, 3500.0], id="right"),
        pytest.param([500.0, 1500.0, -500.0, 1500.0], id="bottom"),
        pytest.param([500.0, 1500.0, 3000.0, 4500.0], id="top"),
    ]
    SAME_AS_PARENT = [0.0, 2000.0, 0.0, 4000.0]

    @staticmethod
    def write_grid_configs(tmp_path, region):
        """Write the grid parameters, and return (parent path, child path)."""
        parent = tmp_path / "grid_parent.yml"
        child = tmp_path / "grid_child.yml"
        with open(parent, "w") as f:
            yaml.safe_dump(
                {
                    "grid_param": {
                        "grid_file": None,
                        "flume_length": 4000,
                        "flume_width": 2000,
                        "spacing": DX_PARENT,
                        "slope_outside": SLOPE,
                        "slope_inside": SLOPE,
                        "slope_basin": SLOPE,
                        "slope_basin_break": 500,
                        "canyon_basin_break": 500,
                        "canyon_center": 1000,
                        "canyon_half_width": 500,
                        "canyon": None,
                        "noise": 0.0,
                    }
                },
                f,
            )
        with open(child, "w") as f:
            yaml.safe_dump(
                {
                    "grid_param": {
                        "spacing": DX_CHILD,
                        "nested_region_xmin": region[0],
                        "nested_region_xmax": region[1],
                        "nested_region_ymin": region[2],
                        "nested_region_ymax": region[3],
                    }
                },
                f,
            )
        return str(parent), str(child)

    @pytest.mark.parametrize("region", OUTSIDE)
    def test_nesting_rejects_child_outside_parent(self, tmp_path, region):
        parent = make_parent(tmp_path)
        child = make_child(tmp_path, region, y_top=4000.0)

        with pytest.raises(ValueError, match="inside the parent grid"):
            OneWayNesting(tc_parent=parent, tc_child=child, dt=DT)

    def test_nesting_accepts_child_same_as_parent(self, tmp_path):
        parent = make_parent(tmp_path)
        child = make_child(tmp_path, self.SAME_AS_PARENT, y_top=4000.0)

        OneWayNesting(tc_parent=parent, tc_child=child, dt=DT)

    def test_create_nested_grid_accepts_same_as_parent(self, tmp_path):
        parent, child = self.write_grid_configs(tmp_path, self.SAME_AS_PARENT)

        _, child_grid, _ = create_nested_grid(
            config_file_parent=parent, config_file_child=child
        )

        assert not np.any(np.isnan(child_grid.at_node["topographic__elevation"]))

    def test_create_nested_grid_checks_rounded_region(self, tmp_path):
        """ymax = 4040 is outside the parent grid, but rounded to 4000."""
        region = [500.0, 1500.0, 3000.0, 4040.0]
        parent, child = self.write_grid_configs(tmp_path, region)

        _, child_grid, _ = create_nested_grid(
            config_file_parent=parent, config_file_child=child
        )

        assert child_grid.y_of_node.max() == 4000.0


class TestParentChildNodeAlignment:
    """Every parent node in a nested region must be a node of the child grid."""

    MESSAGE = "do not coincide"

    @staticmethod
    def make_nesting_of_spacing(parent_spacing, child_spacing, num_cells, offset=0):
        """Make a nesting object whose child grid covers num_cells parent cells.

        The child grid starts at the node 3 of the parent grid, shifted by offset.
        """
        dx_parent = Decimal(str(parent_spacing))
        dx_child = Decimal(str(child_spacing))
        lower_left = dx_parent * 3 + Decimal(str(offset))
        width = dx_parent * num_cells
        parent = RasterModelGrid((61, 61), xy_spacing=[dx_parent, dx_parent])
        child = RasterModelGrid(
            (width / dx_child + 1, width / dx_child + 1),
            xy_spacing=[dx_child, dx_child],
            xy_of_lower_left=[lower_left, lower_left],
        )
        return OneWayNesting(
            tc_parent=SimpleNamespace(grid=parent),
            tc_child=SimpleNamespace(grid=child),
            dt=DT,
        )

    @pytest.mark.parametrize(
        "parent_spacing, child_spacing, num_cells",
        [
            (100.0, 40.0, 4),
            (90.0, 60.0, 4),
            (100.0, 30.0, 3),
            (50.0, 100.0, 4),
            (1.0, 0.3333333333, 3),
        ],
    )
    def test_nesting_rejects_non_coinciding_nodes(
        self, parent_spacing, child_spacing, num_cells
    ):
        with pytest.raises(ValueError, match=self.MESSAGE):
            self.make_nesting_of_spacing(parent_spacing, child_spacing, num_cells)

    @pytest.mark.parametrize(
        "parent_spacing, child_spacing",
        [(100.0, 100.0), (100.0, 50.0), (100.0, 25.0), (1.0, 0.25), (0.1, 0.02)],
    )
    def test_nesting_accepts_coinciding_nodes(self, parent_spacing, child_spacing):
        self.make_nesting_of_spacing(parent_spacing, child_spacing, 4)

    def test_nesting_rejects_child_shifted_from_parent_nodes(self):
        with pytest.raises(ValueError, match=self.MESSAGE):
            self.make_nesting_of_spacing(100.0, 100.0, 4, offset=50)

    def test_error_reports_spacings_and_coordinates(self):
        with pytest.raises(ValueError) as error:
            self.make_nesting_of_spacing(100.0, 40.0, 4)

        message = str(error.value)
        assert "Parent grid spacing: 100.0" in message
        assert "child grid spacing: 40.0" in message
        assert "400.0" in message


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
