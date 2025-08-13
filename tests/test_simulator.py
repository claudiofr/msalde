import context  # noqa: F401
import pytest
from msalde.simulator import DESimulator


def test_compute_metrics_col_name_map_include_ve_sources(
        de_simulator: DESimulator):
    de_simulator.run_simulation()
    pass


