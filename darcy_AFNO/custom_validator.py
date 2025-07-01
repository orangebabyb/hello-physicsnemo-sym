import os
import numpy as np
import torch
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.utils.io.vtk import grid_to_vtk
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.constants import TF_SUMMARY

class CustomGridValidator(GridValidator):
    def save_results(self, name, results_dir, writer, save_filetypes, step):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}

        for i, (invar0, true_outvar0, _) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(invar0, self.device)
            true_outvar = Constraint._set_device(true_outvar0, self.device)
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            for key in invar_cpu:
                invar_cpu[key].append(invar[key].cpu().detach())
            for key in true_outvar_cpu:
                true_outvar_cpu[key].append(true_outvar[key].cpu().detach())
            for key in pred_outvar_cpu:
                pred_outvar_cpu[key].append(pred_outvar[key].cpu().detach())

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {key: torch.cat(value) for key, value in true_outvar_cpu.items()}
        pred_outvar_cpu = {key: torch.cat(value) for key, value in pred_outvar_cpu.items()}

        # compute losses on cpu
        losses = GridValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # Loop through mini-batches
        os.makedirs(results_dir, exist_ok=True)
        n_examples = self.plotter.n_examples if self.plotter is not None else 5
        for i in range(min(n_examples, next(iter(invar.values())).shape[0])):
            single_invar = {k: v[i:i+1] for k, v in invar.items()}
            single_true = {k: v[i:i+1] for k, v in true_outvar.items()}
            single_pred = {k: v[i:i+1] for k, v in pred_outvar.items()}
            sol_diff = {k: np.abs(single_true[k] - single_pred[k]) for k in single_true} # abs loss

            # MSE column
            mse_fields = {}
            for k in sol_diff:
                mse_val = np.mean(np.square(sol_diff[k]))
                mse_array = np.zeros_like(sol_diff[k])
                mse_array.flat[0] = mse_val
                mse_fields[f"mse_{k}"] = mse_array
                print(f"[{name}_{i}] sol_diff MSE ({k}): {mse_val:.6e}")
            
            # Save individual VTI file
            vtk_dict = {
                **single_invar,
                **{f"true_{k}": v for k, v in single_true.items()},
                **{f"pred_{k}": v for k, v in single_pred.items()},
                **{f"diff_{k}": v for k, v in sol_diff.items()},
                **mse_fields,
            }
            grid_to_vtk(vtk_dict, os.path.join(results_dir, f"{name}_{i}"))

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
