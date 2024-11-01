import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        # Create an ellipse of the desired size and color
        ellipse = mpatches.Ellipse(xy=(x0 + width / 2, y0 + height / 2),
                                   width=width, height=height,
                                   angle=0, color=orig_handle.get_facecolor())
        self.update_prop(ellipse, orig_handle, legend)
        ellipse.set_transform(trans)
        return [ellipse]


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        The facecolor of the ellipse.
    **kwargs : dict
        Additional arguments passed on to the `Ellipse` patch.

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor=facecolor, **kwargs)
    # Calculating the stdandard deviation of x from
    # the square root of the variance and multiplying
    # with the number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ellipse.set_zorder(-1)
    return ax.add_patch(ellipse)
