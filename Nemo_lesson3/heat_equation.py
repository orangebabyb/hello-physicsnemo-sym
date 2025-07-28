# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sympy import Symbol, sin, Number, Function
from physicsnemo.sym.eq.pde import PDE

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig, to_absolute_path
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key

from scipy.io import loadmat
import time
from custom_plotter import CustomValidatorPlotter

@physicsnemo.sym.main(config_path="conf", config_name="config_heat")
def run(cfg: PhysicsNeMoConfig) -> None:
    class HeatPDE(PDE):
        def __init__(self, alpha=1.0, beta=1.0, Q=0):
            x = Symbol("x")
            y = Symbol("y")
            t = Symbol("t")
            input_vars = {"x": x, "y": y, "t": t}
            u = Function("u")(*input_vars)
            alpha = Number(alpha)
            beta = Number(beta)

            # source term
            if isinstance(Q, str):
                f = Function(Q)(*input_vars)
            elif isinstance(Q, (float, int)):
                f = Number(Q)

            # set equation
            self.equations = {
                "diffusion_u": beta * u.diff(t) - alpha * (u.diff(x, 2) + u.diff(y, 2)) - f
            }

    # Setup PDE and network
    diffusion_eq = HeatPDE(alpha=0.01, beta=1, Q=0)
    FC = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = diffusion_eq.make_nodes() + [FC.make_node(name="FC")]

    # make geometry
    alpha=0.01
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    geo = Rectangle((0,0), (1,1))
    time_range = {t_symbol: (0, 3.0)}

    # make domain
    domain = Domain()

    # add constraints to solver
    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": sin(2*np.pi*x)*sin(2*np.pi*y)},
        batch_size=cfg.batch_size.IC,
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"diffusion_u": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")

    # Load .mat file
    file_path = "heat_ground_truth.mat"
    mat_data = loadmat(to_absolute_path(file_path))

    # Extract data from .mat file
    X_flat = mat_data["X_flat"]
    Y_flat = mat_data["Y_flat"]
    T_flat = mat_data["T_flat"]
    U_flat = mat_data["U_flat"]

    # Build invar and outvar for Validator
    invar_numpy = {
        "x": X_flat,
        "y": Y_flat,
        "t": T_flat
    }
    outvar_numpy = {
        "u": U_flat
    }

    _plotter = None
    if cfg.run_mode == 'eval':
        _plotter = CustomValidatorPlotter(D_val=alpha)

    validator = PointwiseValidator(
        nodes=nodes, 
        invar=invar_numpy, 
        true_outvar=outvar_numpy, 
        batch_size=10000,
        plotter=_plotter,
    )

    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    start_time = time.time()  # record start time
    run()
    end_time = time.time()    # record end time
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
