import pandas as pd
import matplotlib.pyplot as plt

# ---- load data ----
file_a = "LTS_SimData/tayloe-detector_diff_amp.txt"

df1 = pd.read_csv(file_a, sep=r"\s+")

t = df1["time"]
signals = ["V(am_sig)", "V(i)", "V(q)", "V(i+)", "V(q+)", "V(i-)", "V(q-)"]
sig = list()

for i in signals:
    sig.append(df1[i])

# ---- plotting ----
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
fig.suptitle("Tayloe Detector Waveforms", fontsize=14)
axes[0].plot(t, sig[0], label="RF", color="orange")
axes[0].set_ylabel("AM Signal (V)")
axes[0].set_xlim(0, 5e-3)
axes[1].plot(t, sig[1], label="I", color="blue")
axes[1].plot(t, sig[2], label="Q", color="green")
axes[1].set_ylabel("Voltage (V)")
axes[1].legend(loc="lower right")
axes[1].set_xlim(0, 5e-3)
axes[2].plot(t, sig[3], label="i+", color="blue")
axes[2].plot(t, sig[4], label="q+", color="green")
axes[2].plot(t, sig[5], label="i-", color="blue", alpha=0.5)
axes[2].plot(t, sig[6], label="q-", color="green", alpha=0.5)
axes[2].set_ylabel("Voltage (V)")
axes[2].legend(loc="lower right")
axes[2].set_xlim(0, 5e-3)
axes[-1].set_xlabel("Time (s)")
for ax in axes:
    ax.grid(True, alpha=0.7)

plt.tight_layout()
plt.show()
