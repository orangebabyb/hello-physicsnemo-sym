from physicsnemo.sym.utils.io import GridValidatorPlotter
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

class CustomGridValidatorPlotter(GridValidatorPlotter):
    def __init__(self, n_examples: int = 1, vmin_vmax_dict: Dict[int, Dict[str, tuple]] = None):
        super().__init__(n_examples)
        self.vmin_vmax_dict = vmin_vmax_dict or {}  # 結構: { example_idx: { "sol": (vmin, vmax) } }

    def __call__(
        self,
        invar: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        pred_outvar: Dict[str, np.array],
    ):
        ndim = next(iter(invar.values())).ndim - 2
        if ndim > 3:
            print("Default plotter can only handle <=3 input dimensions, passing")
            return []
        
        # abs loss
        diff_outvar = {
            k: np.abs(true_outvar[k] - pred_outvar[k]) for k in true_outvar
        }

        fs = []
        for ie in range(self.n_examples):
            f = self._make_plot(ndim, ie, invar, true_outvar, pred_outvar, diff_outvar)
            fs.append((f, f"prediction_{ie}"))
        return fs
    
    # make plot
    def _make_plot(self, ndim, ie, invar, true_outvar, pred_outvar, diff_outvar):
        nrows = max(len(invar), len(true_outvar))
        f = plt.figure(figsize=(4 * 5, nrows * 4), dpi=100)
        for ic, (d, tag) in enumerate(
            zip(
                [invar, true_outvar, pred_outvar, diff_outvar],
                ["in", "true", "pred", "diff"],
            )
        ):
            for ir, k in enumerate(d):
                # setting vmin, vmax
                vmin, vmax = None, None
                if tag == "diff":
                    if ie in self.vmin_vmax_dict and k in self.vmin_vmax_dict[ie]:
                        vmin, vmax = self.vmin_vmax_dict[ie][k]
                    else:
                        vmin = 0

                plt.subplot2grid((nrows, 4), (ir, ic))
                if ndim == 1:
                    plt.plot(d[k][ie, 0, :])
                elif ndim == 2:
                    if tag == "diff":
                        plt.imshow(d[k][ie, 0, :, :].T, origin="lower", vmin=vmin, vmax=vmax)
                    else:
                        plt.imshow(d[k][ie, 0, :, :].T, origin="lower")
                else:
                    z = data.shape[-1] // 2
                    if tag == "diff":
                        plt.imshow(d[k][ie, 0, :, :, z], origin="lower", vmin=vmin, vmax=vmax)
                    else:
                        plt.imshow(slice_2d.T, origin="lower")
                plt.colorbar()

                plt.title(f"{k}_{tag}")
        plt.tight_layout()
        return f
