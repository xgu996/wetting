import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import display_barier

fenergy = np.array(
    [
        [0.63359, 0.65060, 0.60926],
        [0.58943, 0.61018, 0.50750],
        [0.50135, 0.52964, 0.49320],
        [0.45748, 0.48956, 0.45470],
        [0.41374, 0.44964, 0.41633],
    ]
)

label = "Young angle"
item = [102, 103, 105, 106, 107]


display_barier(x=item, y=fenergy, xlabel=label, ylabel="barier")
plt.title("wetting different young angle")
plt.savefig("wet_difftheta.png", dpi=300)
plt.show()
