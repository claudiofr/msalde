import pytest
import context  # noqa: F401
from msalde.container import ALDEContainer


def get_alde_container():
    return ALDEContainer()


@pytest.fixture
def alde_container():
    return get_alde_container()


@pytest.fixture
def de_simulator(alde_container):
    return alde_container.simulator
