import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

# fenergy = np.array(
#     [
#         [1.21347, 1.23004, 1.04326, 1.22078, 1.21305],
#         [1.16052, 1.18334, 1.01145, 1.20227, 1.16005],
#         [1.10517, 1.14113, 0.97813, 1.15025, 1.10469],
#         [1.04786, 1.08829, 0.94345, 1.09598, 1.04737],
#         [0.98902, 1.03371, 0.90749, 1.03996, 0.98853],
#         [0.86846, 0.92130, 0.83065, 0.92465, 0.86797],
#     ]
# )

fenergy = np.array(
    [
        [1.04326, 1.23004, 1.21347, 1.22078, 1.04326],
        [1.01145, 1.18334, 1.16052, 1.20227, 1.01145],
        [0.97813, 1.14113, 1.10517, 1.15025, 0.97813],
        [0.94345, 1.08829, 1.04786, 1.09598, 0.94345],
        [0.90749, 1.03371, 0.98902, 1.03996, 0.90749],
        [0.83065, 0.92130, 0.86846, 0.92465, 0.83065],
    ]
)

average = fenergy[:, 0] + fenergy[:, -1]
average /= 2
fenergy[:, 0] = average
fenergy[:, -1] = average

label = "Young angle"
item = [60, 65, 70, 75, 80, 90]

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

plt.title("self-clean small young angle different young angle")
plt.legend()
# plt.savefig("./sclean2_difftheta.png", dpi=300)
plt.show()
