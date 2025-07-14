import numpy as np
from sympy import Symbol, Function, Number, sin, pi

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import PointwiseInteriorConstraint, PointwiseBoundaryConstraint
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.eq.pde import PDE

# 自訂 2D heat PDE
class HeatPDE(PDE):
    def __init__(self, alpha=1.0, f=0):
        x = Symbol("x")
        y = Symbol("y")
        input_vars = {"x": x, "y": y}
        u = Function("u")(*input_vars)

        alpha = Number(alpha)

        # 預設 source term f(x,y) = 0
        if f is None:
            f = Number(0)
        elif isinstance(f, str):
            f = Function(f)(*input_vars)
        elif isinstance(f, (float, int)):
            f = Number(f)

        self.equations = {}
        self.equations["heat_eq"] = -alpha * (u.diff(x, 2) + u.diff(y, 2)) - f

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # PDE 與 NN 結合
    pde = HeatPDE(alpha=1.0, f=0)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        nr_layers=4,
        layer_size=64,
    )
    nodes = pde.make_nodes() + [net.make_node(name="u_net")]

    # make geometry
    geo = Rectangle((0, 0), (1, 1))
    domain = Domain()

    # add PDE interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"heat_eq": 0},
        batch_size=512,
    )
    domain.add_constraint(interior, "interior")

    # Dirichlet boundary: u=0
    bc = PointwiseBoundaryConstraint(
        nodes=[net.make_node(name="u_net")],
        geometry=geo,
        outvar={"u": 0},
        batch_size=128,
    )
    domain.add_constraint(bc, "boundary")

    # 建立求解器並訓練
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
