import numpy as np
import tkinter as tk
import matplotlib as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
from abc import ABC, abstractmethod

plt.use("TkAgg")


class PlotterBase(ABC):
    def __init__(
        self,
        update_freq: int | None = 100,
        fig_size: tuple[int, int] = (10, 10),
        projection: str = None,
    ):
        self._update_freq = update_freq
        self._fig_size = fig_size
        self._call_counter = 0

        self._data = defaultdict(list)

        self._window = tk.Tk()
        self._figure = Figure(figsize=fig_size, dpi=100)
        self._axis = self._figure.add_subplot(1, 1, 1, projection=projection)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self._window)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def reset(self):
        self._data.clear()
        self._call_counter = 0

    def draw_plot(self):
        self._axis.clear()
        self._make_plot(self._axis, self._data)
        self._canvas.draw()
        self._window.update()

    def close(self):
        try:
            self._window.update()
            self._window.destroy()
            self._window.quit()
        except:
            pass

    def save(self, filename: str):
        self._figure.savefig(filename, bbox_inches="tight")

    def add_data(self, **kwargs):
        for k, v in kwargs.items():
            self._data[k].append(v)

        if self._update_freq is None:
            return

        if self._call_counter == 0:
            self.draw_plot()

        self._call_counter = (self._call_counter + 1) % self._update_freq

    @abstractmethod
    def _make_plot(self, axis: Axes, data: dict[str, list]):
        raise NotImplementedError()


class SearchPlotter(PlotterBase):
    def __init__(
        self,
        update_freq: int = 100,
        fig_size: tuple[int, int] = (10, 10),
        alpha: float = 0.25,
        point_size: float = 40,
    ):
        super().__init__(update_freq, fig_size, projection="3d")

        self._alpha = alpha
        self._point_size = point_size
        self.alphas = [1]

    def _make_plot(self, axis: Axes, data: dict[str, list]):
        count = len(data["loss"])
        alphas = self._alpha
        if count > self._update_freq:
            exponents = np.flip(np.arange(self._update_freq))
            alphas1 = np.power(self._alpha*np.ones(self._update_freq), exponents)
            alphas = np.ones(count)*alphas1[0]
            alphas[-self._update_freq:] = alphas1
        
        axis.scatter(
            data["x"],
            data["y"],
            data["z"],
            c=data["loss"],
            cmap="inferno",
            s=self._point_size,
            alpha=alphas,
        )

        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
