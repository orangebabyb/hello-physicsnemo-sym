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
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.diffusion import Diffusion

import time
from custom_plotter import CustomValidatorPlotter
import pynvml

def get_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_MB = meminfo.used / 1024**2
    total_MB = meminfo.total / 1024**2
    pynvml.nvmlShutdown()
    return used_MB, total_MB


@physicsnemo.sym.main(config_path="conf", config_name="config_heat")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Custom PDE
    class HeatPDE(PDE):
        def __init__(self, beta=1.0, Q=0):
            x = Symbol("x")
            y = Symbol("y")
            t = Symbol("t")
            D = Symbol("D") # alpha be a learnable variable 
            input_vars = {"x": x, "y": y, "t": t, "D":D}
            u = Function("u")(*input_vars)
            #alpha = Number(alpha)
            beta = Number(beta)

            # source term
            if isinstance(Q, str):
                f = Function(f)(*input_vars)
            elif isinstance(Q, (float, int)):
                f = Number(Q)

            # set equation
            self.equations = {
                "diffusion_u": beta * u.diff(t) - D * (u.diff(x, 2) + u.diff(y, 2)) - f
            }

    # make list of nodes to unroll graph on
    diffusion_eq = HeatPDE(beta=1, Q=0)
    net_u = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t"), Key("D")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = diffusion_eq.make_nodes() + [net_u.make_node(name="net_u")]

    # add constraints to solver
    # make geometry
    x, y, t_symbol, d_symbol = Symbol("x"), Symbol("y"), Symbol("t"), Symbol("D")
    L = 1
    geo = Rectangle((0,0), (L,L))

    # make domain
    domain = Domain()

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": sin(2*np.pi*x)*sin(2*np.pi*y)},
        batch_size=cfg.batch_size.IC,
        #lambda_weighting={"u": 10.0},
        parameterization={t_symbol: 0.0, d_symbol: (0.01, 0.1)},
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        parameterization={t_symbol: (0, 3.0), d_symbol: (0.01, 0.1)},
    )
    domain.add_constraint(BC, "BC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"diffusion_u": 0},
        batch_size=cfg.batch_size.interior,
        parameterization={t_symbol: (0, 3.0), d_symbol: (0.01, 0.1)},
        fixed_dataset=True,
    )
    domain.add_constraint(interior, "interior")

    # add validation data

    L = 1
    x = np.linspace(0, L, 64)
    y = np.linspace(0, L, 64)

    D_values = [0.01, 0.05, 0.1]
    D_to_Nt = {
        0.01: 481,   # 300 intervals
        0.05: 481,  # 1500 intervals
        0.1: 481    # 3000 intervals
    }

    for D_val in D_values:
        Nt = D_to_Nt[D_val]
        t = np.linspace(0, 3.0, Nt)
        
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")
        X_flat = X.flatten()[:, None]
        Y_flat = Y.flatten()[:, None]
        T_flat = T.flatten()[:, None]
        D_flat = np.full_like(X_flat, D_val)

        u_D = np.sin(2*np.pi*X_flat) * np.sin(2*np.pi*Y_flat) * np.exp(-2 * D_val * (2*np.pi)**2 * T_flat)

        invar_numpy_D = {
            "x": X_flat,
            "y": Y_flat,
            "t": T_flat,
            "D": D_flat
        }
        outvar_numpy_D = {"u": u_D}

        _plotter = None
        if cfg.run_mode == 'eval':
            _plotter = CustomValidatorPlotter(D_val=D_val)

        name = f"validator_D_{str(D_val).replace('.', '')}"
        validator = PointwiseValidator(
            nodes=nodes,
            invar=invar_numpy_D,
            true_outvar=outvar_numpy_D,
            batch_size=1000,
            plotter=_plotter,
        )
        domain.add_validator(validator, name=name)
   

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    used, total = get_gpu_memory()
    print(f"[INIT] GPU memory used: {used:.0f} / {total:.0f} MB")


if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
