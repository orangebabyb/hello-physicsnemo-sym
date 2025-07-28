import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

class CustomValidatorPlotter(ValidatorPlotter):
    def __init__(self, D_val):
        super().__init__()
        self.D_val = D_val  # alpha
    def __call__(self, invar, true_outvar, pred_outvar):
        x_all = invar["x"][:, 0]
        y_all = invar["y"][:, 0]
        t_all = invar["t"][:, 0]
        u_true_all = true_outvar["u"][:, 0]
        u_pred_all = pred_outvar["u"][:, 0]

        overall_mse = np.mean((u_true_all - u_pred_all) ** 2)
        print(f"[Overall] Spatio-temporal MSE = {overall_mse:.4e}")

        # -------------------------------
        # 1. Find all time steps
        # -------------------------------
        t_unique = np.unique(np.round(t_all, decimals=6))
        print("FC  time grid :", np.unique(t_all)[:10], "...")

        # -------------------------------
        # 2. Check the time steps are enough
        # -------------------------------
        valid_times = []
        for t in t_unique:
            mask = np.isclose(t_all, t, atol=1e-6)
            if np.sum(mask) >= 100:
                valid_times.append(t)

        print(f"Total {len(valid_times)} time steps to visualize.")

        # -------------------------------
        # 3. Create spaital-temporal grids
        # -------------------------------
        extent = (x_all.min(), x_all.max(), y_all.min(), y_all.max())
        x_grid = np.linspace(extent[0], extent[1], 64)
        y_grid = np.linspace(extent[2], extent[3], 64)
        xyi = np.meshgrid(x_grid, y_grid, indexing="ij")

        # -------------------------------
        # 4. Create 3D subplot
        # -------------------------------
        fig = plt.figure(figsize=(18, 5))
        axs = []
        for i in range(3):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            axs.append(ax)

        axs[0].set_title("True u(x,y,t)")
        axs[1].set_title("Predicted u(x,y,t)")
        axs[2].set_title("Absolute Error")

        for ax in axs:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u")
            ax.set_zlim(-1, 1)

        fig.suptitle("Time t = 0.000")
        fig.tight_layout()

        # -------------------------------
        # 5. Update function
        # -------------------------------
        mse_history = []
        
        def update(t):
            fig.clf()  # clear the figure

            axs = []
            for i in range(3):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                axs.append(ax)

            mask = np.isclose(t_all, t, atol=1e-6)
            x = x_all[mask]
            y = y_all[mask]
            u_true = u_true_all[mask]
            u_pred = u_pred_all[mask]
            u_error = np.abs(u_true - u_pred)

            u_true_interp = scipy.interpolate.griddata((x, y), u_true, tuple(xyi), method="linear")
            u_pred_interp = scipy.interpolate.griddata((x, y), u_pred, tuple(xyi), method="linear")
            u_error_interp = scipy.interpolate.griddata((x, y), u_error, tuple(xyi), method="linear")

            axs[0].plot_surface(xyi[0], xyi[1], u_true_interp, cmap='viridis', vmin=-1, vmax=1)
            axs[1].plot_surface(xyi[0], xyi[1], u_pred_interp, cmap='viridis', vmin=-1, vmax=1)
            axs[2].plot_surface(xyi[0], xyi[1], u_error_interp, cmap='viridis', vmin=0, vmax=0.05)

            axs[0].set_title("True u(x,y,t)")
            axs[1].set_title("Predicted u(x,y,t)")
            axs[2].set_title("Absolute Error")

            for ax in axs:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("u")
                ax.set_zlim(-1, 1)
                ax.set_xlim(ax.get_xlim()[::-1])  #inverse x coordinate

            axs[2].set_zlim(0, 0.05) #set fixed range
            # Calculate MSE
            mse = np.nanmean((u_true_interp - u_pred_interp)**2)
            mse_history.append(mse)
            max_mse_so_far = np.max(mse_history)

            fig.suptitle(f"D = {self.D_val:.3f}    Time t = {t:.3f}    MSE = {mse:.4e}(max MSE: {max_mse_so_far:.4e})")

            return axs

        # -------------------------------
        # 6. Create animation
        # -------------------------------
        ani = animation.FuncAnimation(
            fig, update, frames=valid_times, blit=False, interval=150, repeat_delay=1000
        )

        gif_path = f"./animation_{self.D_val:.3f}.gif"
        ani.save(gif_path, writer="pillow")
        print(f"Saved animation to {gif_path}")
        return [(fig, "animation")]

        