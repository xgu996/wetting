import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

# 000_DDD -> 1000_UDDD_L -> 0000_DDDD_L -> 0001_DDDU_L -> 000_DDD_L

# Transition new:

# 0000_DDDD_L -> 1000_UDDD_L -> 000_DDD -> 0001_DDDU -> 0000_DDDD


# fenergy = np.array(
#     [
#         [1.01052, 1.03477, 0.92200, 1.05942, 1.00991],
#         [0.99462, 1.02722, 0.91083, 1.04635, 0.99465],
#         [0.99224, 1.03170, 0.91019, 1.04425, 0.99220],
#         [0.98902, 1.03371, 0.90749, 1.03996, 0.98853],
#         [0.98982, 1.03926, 0.90392, 1.03926, 0.98907],
#     ]
# )

fenergy = np.array(
    [
        [0.92200, 1.03477, 1.01052, 1.05942, 0.92200],
        [0.91083, 1.02722, 0.99462, 1.04635, 0.91083],
        [0.91019, 1.03170, 0.99224, 1.04425, 0.91019],
        [0.90749, 1.03371, 0.98902, 1.03996, 0.90749],
        [0.90392, 1.03926, 0.98982, 1.03926, 0.90392],
    ]
)


average = fenergy[:, 0] + fenergy[:, -1]
average /= 2
fenergy[:, 0] = average
fenergy[:, -1] = average

label = "bottom angle"
item = [50, 60, 70, 80, 90]

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

plt.title("self-clean small young angle different bottom angle")
plt.legend()
# plt.savefig("./sclean2_diffbott.png", dpi=300)
plt.show()
