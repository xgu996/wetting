import numpy as np
import pandas as pd

#  ────────────────────────────────────────────────────────────
#  Wetting different height
#  ────────────────────────────────────────────────────────────
data = {
    "gap": np.ones(5) * 0.06,
    "young": np.ones(5) * 102,
    "height": [0.1, 0.12, 0.13, 0.14, 0.15],
    "barier": [0.00921, 0.01223, 0.01385, 0.01545, 0.01691],
}

df = pd.DataFrame(data)
print(df)
df.to_csv("./wet_diffheight.csv")

#  ────────────────────────────────────────────────────────────
#  Wetting different gap
#  ────────────────────────────────────────────────────────────
data = {
    "gap": [0.05, 0.06, 0.07, 0.08, 0.09],
    "young": np.ones(5) * 102,
    "height": np.ones(5) * 0.1,
    "barier": [1.226e-02, 9.210e-03, 6.170e-03, 3.400e-03, 4.000e-05],
}
df = pd.DataFrame(data)
print(df)
df.to_csv("./wet_diffgap.csv")

#  ────────────────────────────────────────────────────────────
#  Wetting different young
#  ────────────────────────────────────────────────────────────
data = {
    "gap": np.ones(5) * 0.06,
    "young": [102, 103, 105, 106, 107],
    "height": np.ones(5) * 0.15,
    "barier": [
        0.01701,
        0.02075,
        0.02829,
        0.03208,
        0.0359,
    ],
}
df = pd.DataFrame(data)
print(df)
df.to_csv("./wet_diffyoung.csv")

#  ────────────────────────────────────────────────────────────
#  Self clean large young diff bottom
#  ────────────────────────────────────────────────────────────

data = {
    "gap": np.ones(5) * 0.08,
    "young": np.ones(5) * 102,
    "height": np.ones(5) * 0.08,
    "bottom_heta": np.arange(50, 100, 10),
    "barier_move_to_left": [0.018745, 0.0193, 0.02249, 0.02514, 0.02562],
    "barier_move_to_right": [0.02627, 0.02676, 0.02621, 0.02715, 0.02572],
}
df = pd.DataFrame(data)
print(df)
df.to_csv("./sclean1_diffbottom.csv")


#  ────────────────────────────────────────────────────────────
#  Self clean small young diff bottom
#  ────────────────────────────────────────────────────────────

data = {
    "gap": np.ones(5) * 0.08,
    "young": np.ones(5) * 80,
    "height": np.ones(5) * 0.08,
    "bottom_heta": np.arange(50, 100, 10),
    "barier_move_to_left": [0.13742, 0.13552, 0.13406, 0.13247, 0.13534],
    "barier_move_to_right": [0.11277, 0.11639, 0.12151, 0.12622, 0.13534],
}
df = pd.DataFrame(data)
print(df)
df.to_csv("./sclean2_diffbottom.csv")

#  ────────────────────────────────────────────────────────────
#  Self clean large young diff young
#  ────────────────────────────────────────────────────────────
data = {
    "gap": np.ones(7) * 0.08,
    "young": [98, 101, 102, 105, 107, 109, 112],
    "height": np.ones(7) * 0.08 * 0.8,
    "bottom_heta": np.ones(7) * 40,
    "barier_move_to_left": [
        0.03082,
        0.0254,
        0.023945,
        0.019115,
        0.016125,
        0.01469,
        0.01794,
    ],
    "barier_move_to_right": [
        0.02498,
        0.02539,
        0.02556,
        0.02609,
        0.02646,
        0.02685,
        0.02747,
    ],
}
df = pd.DataFrame(data)
print(df)
df.to_csv("./sclean1_diffyoung.csv")

#  ────────────────────────────────────────────────────────────
#  Self clean small young diff young
#  ────────────────────────────────────────────────────────────
data = {
    "gap": np.ones(5) * 0.08,
    "young": [60, 65, 70, 75, 80],
    "height": np.ones(5) * 0.08 * 0.8,
    "bottom_heta": np.ones(5) * 40,
    "barier_move_to_left": [0.17752, 0.19082, 0.17212, 0.15253, 0.13247],
    "barier_move_to_right": [0.18678, 0.17189, 0.163, 0.14484, 0.12622],
}
df = pd.DataFrame(data)
print(df)
df.to_csv("./sclean2_diffyoung.csv")
