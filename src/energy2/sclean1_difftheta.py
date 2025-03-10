import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

# bottom angle = 60

# fenergy = np.array(
#     [
#         [0.87928, 0.91010, 0.88512, 0.88961, 0.87928],
#         [0.80990, 0.83530, 0.80991, 0.81677, 0.80990],
#         [0.78662, 0.81032, 0.78476, 0.79248, 0.78613],
#         [0.71662, 0.73551, 0.70942, 0.72004, 0.71617],
#         [0.66996, 0.68587, 0.65941, 0.67198, 0.66953],
#         [0.62340, 0.63653, 0.60968, 0.62437, 0.62298],
#         [0.55384, 0.56325, 0.53578, 0.55372, 0.55344],
#     ]
# )
#
# fenergy = np.array(
#         [
#              [0.88512,0.91010,0.87928,0.88961,0.88512],
#              [0.80991,0.83530,0.80990,0.81677,0.80991],
#              [0.78476,0.81032,0.78662,0.79248,0.78476],
#              [0.70942,0.73551,0.71662,0.72004,0.70942],
#              [0.65941,0.68587,0.66996,0.67198,0.65941],
#              [0.60968,0.63653,0.62340,0.62437,0.60968],
#              [0.53578,0.56325,0.55384,0.55372,0.53578],
#         ]
# )

fenergy = np.array(
    [
        [0.86575, 0.89290, 0.88610, 0.89131, 0.86631],
        [0.79611, 0.82417, 0.81987, 0.82334, 0.79667],
        [0.74973, 0.77850, 0.77559, 0.77790, 0.75028],
        [0.70358, 0.73316, 0.73141, 0.73276, 0.70421],
        [0.63505, 0.66599, 0.66553, 0.66586, 0.63558],
    ]
)

average = fenergy[:, 0] + fenergy[:, -1]
average /= 2
fenergy[:, 0] = average
fenergy[:, -1] = average

label = "young angle"
item = [102, 105, 107, 109, 112]
# item = [98, 101, 102, 105, 107, 109, 112]

display_barier(
    x=item, y=fenergy, xlabel=label, ylabel="energy barier", label="Move to right"
)
display_barier(
    x=item,
    y=fenergy[:, ::-1],
    xlabel=label,
    ylabel="energy barier",
    label="Move to left",
)

plt.title("self-clean large young angle different young angle")
plt.legend()
# plt.savefig("./sclean1_difftheta.png", dpi=300)
plt.show()
