import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

### 000_UUU -> 001_UUU -> 00_UU_L -> 100_UUU_L -> 000_UUU_L
### Move to right new:
### 00_UU_L -> 001_UUU -> 000_UUU -> 100_UUU -> 00_UU


# fenergy = np.array(
#     [
#         [0.78662, 0.81032, 0.78476, 0.79248, 0.78613],
#         [0.80157, 0.82015, 0.79388, 0.80718, 0.80124],
#         [0.83865, 0.85178, 0.82502, 0.84432, 0.83869],
#         [0.87006, 0.87944, 0.85323, 0.87572, 0.87005],
#         [0.88610, 0.89290, 0.86575, 0.89089, 0.88566],
#         [(0.89317+0.89217)/2, (0.89887+0.89847)/2 + 5e-5, 0.87300, (0.89887+0.89847)/2 - 5e-5, (0.89317+0.89217)/2],
#     ]
# )

fenergy = np.array(
        [
            [0.79388, 0.82015, 0.80157, 0.80718, 0.79388],
            [0.82502, 0.85178, 0.83865, 0.84432, 0.82502],
            [0.85323, 0.87944, 0.87006, 0.87572, 0.85323],
            [0.86575, 0.89290, 0.88610, 0.89089, 0.86575],
            [0.87300, 0.89872, 0.89267, 0.89872, 0.87300],
        ]
)


average = fenergy[:, 0] + fenergy[:, -1]
average /= 2
fenergy[:, 0] = average
fenergy[:, -1] = average

label = "bottom angle"
item = [50, 60, 70, 80, 90]

display_barier(x=item, y=fenergy, xlabel=label, label="Move to right")
display_barier(x=item, y=fenergy[:, ::-1], xlabel=label, label="Move to left")

plt.title("self-clean large young angle different bottom angle")
plt.legend()

# plt.savefig("./sclean1_diffbott.png", dpi=300)
plt.show()
