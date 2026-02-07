import pandas as pd
import matplotlib.pyplot as plt

# ---- load data ----
file_a = "LTS_SimData/tayloe-detector_lpf.txt"
file_b = "LTS_SimData/tayloe-detector_sw.txt"

df1 = pd.read_csv(file_a, sep=r"\s+")
df2 = pd.read_csv(file_b, sep=r"\s+")

t1 = df1["time"]
t2 = df2["time"]

signals = ["V(i+)", "V(i-)", "V(q+)", "V(q-)"]

# ---- plotting ----
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

for ax, sig in zip(axes, signals):
    ax.plot(t1, df1[sig], label="lpf")
    ax.plot(t2, df2[sig], linestyle="--", label="raw")
    ax.set_ylabel(sig)
    ax.set_xlim(2e-5, 3e-5)
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
