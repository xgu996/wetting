import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

fenergy = np.array(
    [
        [1.01052, 1.03477, 0.92200, 1.05942, 1.00991],
        [0.99462, 1.02722, 0.91083, 1.04635, 0.99465],
        [0.99224, 1.03170, 0.91019, 1.04425, 0.99220],
        [0.98902, 1.03371, 0.90749, 1.03996, 0.98853],
        [0.98982, (1.03948 + 1.03904)/2, 0.90392, ( 1.03948+1.03904 )/2, 0.98907],
    ]
)

average = fenergy[:, 0] + fenergy[:, -1]
average /= 2
fenergy[:, 0] = average
fenergy[:, -1] = average

label = "bottom angle"
item = [50, 60, 70, 80, 90]

display_barier(
    x=item, y=fenergy, xlabel=label, ylabel="energy barier", label="Move to left"
)
display_barier(
    x=item,
    y=fenergy[:, ::-1],
    xlabel=label,
    ylabel="energy barier",
    label="Move to right",
)

plt.title("self-clean small young angle different bottom angle")
plt.legend()
plt.savefig("./sclean2_diffbott.png", dpi=300)
plt.show()
