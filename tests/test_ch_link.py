"""Tests for the sediment volume at links (Ch_link and Ch_link_i).

The tests cover the following bug:

1. map_nodes_to_links() rebound the local name Ch_link instead of writing to
   the array given by the caller, so TurbidityCurrent2D.Ch_link was not
   updated by map_values(). In addition, map_values() was called without
   Ch_link_i after the wet/dry processing, the non-advection phase, the
   advection phase and the shock dissipation phase, so Ch_link_i also kept
   the values of the beginning of the step. At links that had just become
   wet at the flow front, Ch_link stayed zero.
"""
import warnings

import numpy as np
import pytest
from landlab import RasterModelGrid

import turb2d.gridutils as gridutils
from turb2d import TurbidityCurrent2D
from turb2d.utils import create_init_flow_region

SHAPE = (21, 21)
DX = 10.0
C_INI = [0.006, 0.004]


@pytest.fixture
def tc():
    grid = RasterModelGrid(SHAPE, xy_spacing=DX)
    grid.add_zeros("topographic__elevation", at="node")
    create_init_flow_region(
        grid,
        initial_flow_concentration=C_INI,
        initial_flow_thickness=20.0,
        initial_region_radius=35.0,
        initial_region_center=[100.0, 100.0],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TurbidityCurrent2D(
            grid,
            Cf=0.004,
            alpha=0.4,
            kappa=0.1,
            nu_a=0.75,
            Ds=[80e-6, 40e-6],
            h_init=0.0,
            Ch_w=1e-5,
            h_w=0.01,
            C_init=0.0,
            implicit_num=100,
            implicit_threshold=1e-10,
            r0=1.5,
            water_entrainment=True,
            suspension=True,
            no_erosion=True,
            bedload_transport=False,
            model="3eq",
        )


def interior_wet_links(tc):
    """Wet and partially wet links whose both nodes are core nodes."""
    links = tc.wet_pwet_links
    nodes = tc.grid.nodes_at_link[links]
    return links[np.all(np.isin(nodes, tc.grid.core_nodes), axis=1)]


def expected_Ch_link_i(tc, links):
    nodes = tc.grid.nodes_at_link[links]
    return 0.5 * (tc.Ch_i[:, nodes[:, 0]] + tc.Ch_i[:, nodes[:, 1]])


def run_steps(tc, dt=1.0, n=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            tc.run_one_step(dt=dt)


def test_Ch_link_i_is_mean_of_nodes_after_step(tc):
    run_steps(tc)
    links = interior_wet_links(tc)

    np.testing.assert_allclose(
        tc.Ch_link_i[:, links], expected_Ch_link_i(tc, links), atol=1e-12
    )


def test_Ch_link_is_sum_of_Ch_link_i_after_step(tc):
    run_steps(tc)

    np.testing.assert_allclose(tc.Ch_link, np.sum(tc.Ch_link_i, axis=0), atol=1e-12)


def test_Ch_link_at_new_front_links_follows_nodes(tc):
    run_steps(tc, n=1)
    wet_before = set(tc.wet_pwet_links)
    run_steps(tc, n=1)
    links = interior_wet_links(tc)
    new_links = np.array([link for link in links if link not in wet_before])
    new_links = new_links[np.sum(expected_Ch_link_i(tc, new_links), axis=0) > 0]

    assert new_links.size > 0
    np.testing.assert_allclose(
        tc.Ch_link[new_links],
        np.sum(expected_Ch_link_i(tc, new_links), axis=0),
        atol=1e-12,
    )


def test_map_values_updates_Ch_link_in_place(tc):
    run_steps(tc, n=1)
    Ch_link = np.zeros_like(tc.Ch_link)

    gridutils.map_values(
        tc,
        h=tc.h,
        Ch_i=tc.Ch_i,
        Ch=tc.Ch,
        eta=tc.eta,
        h_link=tc.h_link,
        Ch_link_i=tc.Ch_link_i,
        Ch_link=Ch_link,
    )

    np.testing.assert_allclose(Ch_link, np.sum(tc.Ch_link_i, axis=0))
    assert np.any(Ch_link > 0)


def test_map_values_requires_Ch_link_i_for_Ch_link(tc):
    run_steps(tc, n=1)
    with pytest.raises(ValueError):
        gridutils.map_values(tc, Ch_link=np.zeros_like(tc.Ch_link))
