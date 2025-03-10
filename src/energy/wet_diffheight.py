import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

fenergy = np.array(
    [
        [0.77912, 0.78833, 0.74698],
        [0.71800, 0.73023, 0.68891],
        [0.69124, 0.70509, 0.66377],
        [0.66077, 0.67622, 0.63489],
        [0.63359, 0.65050, 0.60926],
    ]
)

label = "height"
item = [0.1, 0.12, 0.13, 0.14, 0.15]

display_barier(x=item, y=fenergy, xlabel=label, ylabel="barier")
plt.title("wetting different bottom height")
plt.savefig("wet_diffheight.png", dpi=300)
plt.show()
