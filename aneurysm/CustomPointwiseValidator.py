import numpy as np
import torch

from typing import List, Dict
from pathlib import Path

from physicsnemo.sym.domain.validator import Validator
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk, VTKBase
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.constants import TF_SUMMARY
from physicsnemo.sym.dataset import DictPointwiseDataset
from physicsnemo.sym.distributed import DistributedManager

class CustomPointwiseValidator(Validator):
    """
    Pointwise Validator with extended metrics (MSE, RMSE, L1, L2)
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        batch_size: int = 1024,
        plotter: ValidatorPlotter = None,
        requires_grad: bool = False,
    ):
        self.dataset = DictPointwiseDataset(invar=invar, outvar=true_outvar)
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            distributed=False,
            infinite=False,
        )

        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd
        self.plotter = plotter

    # save result
    def save_results(self, name, results_dir, writer, save_filetypes, step):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}

        # loop through mini-batches
        for _, (invar0, true_outvar0, _) in enumerate(self.dataloader):
            invar = Constraint._set_device(invar0, device=self.device, requires_grad=self.requires_grad)
            true_outvar = Constraint._set_device(true_outvar0, device=self.device, requires_grad=self.requires_grad)
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            for key in invar_cpu:
                invar_cpu[key].append(invar[key].cpu().detach())
            for key in true_outvar_cpu:
                true_outvar_cpu[key].append(true_outvar[key].cpu().detach())
            for key in pred_outvar_cpu:
                pred_outvar_cpu[key].append(pred_outvar[key].cpu().detach())

        invar_cpu = {k: torch.cat(v) for k, v in invar_cpu.items()}
        true_outvar_cpu = {k: torch.cat(v) for k, v in true_outvar_cpu.items()}
        pred_outvar_cpu = {k: torch.cat(v) for k, v in pred_outvar_cpu.items()}

        # compute losses on cpu
        # TODO add metrics specific for validation
        metrics = {}
        for key in true_outvar_cpu.keys():
            if key not in pred_outvar_cpu:
                continue
            pred = pred_outvar_cpu[key]
            true = true_outvar_cpu[key]

            mse = torch.mean((pred - true) ** 2)
            rmse = torch.sqrt(mse)
            l1 = torch.mean(torch.abs(pred - true))
            l2 = torch.mean((pred - true) ** 2)

            metrics[f"{key}_MSE"] = mse.item()
            metrics[f"{key}_RMSE"] = rmse.item()
            metrics[f"{key}_L1"] = l1.item()
            metrics[f"{key}_L2"] = l2.item()

        # ----- only print validation result -----
        print("\n=== Validation Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.2e}")
        print("==========================\n")

        # TODO: add potential support for lambda_weighting
        losses = CustomPointwiseValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:
            var_to_polyvtk(
                {**invar, **named_true_outvar, **named_pred_outvar}, results_dir + name
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True
                )
        return losses