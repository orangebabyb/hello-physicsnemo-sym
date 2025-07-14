import numpy as np
from sympy import Symbol, Function, Number, pi, sin

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D, Line1D
from physicsnemo.sym.domain.constraint import (
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.eq.pde import PDE


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # make list of nodes to unroll graph on
    u_net = FullyConnectedArch(
        input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32
    )

    nodes = [u_net.make_node(name="u_network")]

    # add constraints to solver
    # make geometry
    x = Symbol("x")
    geo = Line1D(0, 1)

    # make domain
    domain = Domain()

    # integral
    integral = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=1,
        integral_batch_size=100,
    )
    domain.add_constraint(integral, "integral")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()