import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

fenergy = np.array(
    [
        [0.74601, 0.75827, 0.72503],
        [0.77912, 0.78833, 0.74698],
        [0.81447, 0.82064, 0.77095],
        [0.83879, 0.84219, 0.78390],
        [0.85523, 0.85527, 0.78845],
    ]
)

label = "gap"
item = [0.05, 0.06, 0.07, 0.08, 0.09]

display_barier(x=item, y=fenergy, xlabel=label, ylabel="barier")
plt.title("wetting different bottom gaps")
plt.savefig("wet_diffgap.png", dpi=300)
plt.show()
