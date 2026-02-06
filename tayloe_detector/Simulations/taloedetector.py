import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Moving average (simulates capacitors)
def moving_average(x, n):
    return np.convolve(x, np.ones(n)/n, mode="same")

from scipy.signal import lfilter

def lowpass_filter(x, dt, tau):
    alpha = dt / (tau + dt)
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, x)

# def lowpass_filter(x, dt, tau, initial=None):
#     """
#     RC (capacitor) low-pass filter
#     y[n] = alpha*x[n] + (1-alpha)*y[n-1]
#     """
#     x = np.asarray(x, dtype=np.float64)

#     alpha = dt / (tau + dt)
#     decay = 1 - alpha

#     y = np.empty_like(x)

#     if initial is None:
#         y[0] = x[0]
#     else:
#         y[0] = initial

#     for n in range(1, len(x)):
#         y[n] = alpha * x[n] + decay * y[n-1]

#     return y



# some constants
t = np.linspace(0, 50e-6, 1000) # desired time division
N = 10 # used for moving avg, would vary with capacitance/ time divisions

dt = t[1]-t[0]

# ----- Setup Plots ----- (can ignore)
fig, axs = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
for ax in axs:
    ax.grid(True)
axs[-1].set_xlim(0, 50e-6)
axs[0].set_ylim(-1.5, 1.5)
for i in range(1, 6):
    axs[i].set_ylim(-1.2, 1.2)
axs[5].set_ylim(-1, 1)
# axs[6].set_ylim(-1, 1.5)

# setup specific line plots
rf, = axs[0].plot([], [], label="RF")
axs[0].set_ylabel("RF (2 Signals)")
l0_lo,  = axs[1].plot([], [], label="LO")
l0_raw, = axs[1].plot([], [], label="Raw")
l0_f,   = axs[1].plot([], [], label="Filt")
axs[1].legend(loc="upper right")
axs[1].set_ylabel("I+")

l180_lo,  = axs[2].plot([], [], label="LO")
l180_raw, = axs[2].plot([], [], label="Raw")
l180_f,   = axs[2].plot([], [], label="Filt")
axs[2].legend(loc="upper right")
axs[2].set_ylabel("I-")

l90_lo,  = axs[3].plot([], [], label="LO")
l90_raw, = axs[3].plot([], [], label="Raw")
l90_f,   = axs[3].plot([], [], label="Filt")
axs[3].legend(loc="upper right")
axs[3].set_ylabel("Q+")

l270_lo,  = axs[4].plot([], [], label="LO")
l270_raw, = axs[4].plot([], [], label="Raw")
l270_f,   = axs[4].plot([], [], label="Filt")
axs[4].legend(loc="upper right")
axs[4].set_ylabel("Q-")

l_i_raw, = axs[5].plot([], [], label="I raw")
l_i_f,   = axs[5].plot([], [], label="I filt")
l_q_raw, = axs[5].plot([], [], label="Q raw")
l_q_f,   = axs[5].plot([], [], label="Q filt")
aud_sig, = axs[5].plot([], [], label="Aud Sig")
axs[5].legend(loc="upper right")
axs[5].set_ylabel("I and Q")

# # reconstructed wave
# reconst, = axs[6].plot([], [], label="reconstructed")
# expected, = axs[6].plot([], [], label="input audio")
# axs[6].legend(loc="upper right")
# axs[6].set_ylabel("sqrt(I^2+Q^2)")



f_start  = 660_000 - 100_000
f_stop   = 660_000 + 100_000
n_frames = 300    # total frames

half = n_frames // 2
df = (f_stop - f_start) / half



# ---- IMPORTANT TO UNDERSTAND -----
def update(frame):
    if frame < half:
        f_car = f_start + df * frame
    else:
        f_car = f_stop - df * (frame - half)
    f_audio = 15_000 # + frame * 100 # audio frequency (50Hz to 20 kHz)
    aud = np.cos(2 * np.pi * f_audio * t) # audio signal
    car = np.cos(2 * np.pi * f_car * t) # carier signal
    m = 0.7 # modulation index
    ant = car * (1 + m * aud) # + 0.5 * (np.cos(2 * np.pi * 660_000 * t) * (1 + m * np.cos(2 * np.pi * 5000 * t))) # modulated signal (from antenna)

    calc = round(125_000_000 / 4 / f_car)

    f_lo = f_car #660_000 # f_car + 2000 # 125_000_000 / 4 / calc # ***** we create this signal,
                          # ***** it it 10kHz offset from carrier frequency
    
    # The following generates 4 selects (like an analog mux!!)
    # they are each 90 degrees offset such that only one is active at a time at frequency f_lo
    # 1000 -> 0100 -> 0010 -> 0001 -> 1000 -> ...
    phase = (t * f_lo) % 1
    # 4-phase LO
    lo0   = ((phase < 0.25)).astype(float)
    lo90  = ((phase >= 0.25) & (phase < 0.50)).astype(float)
    lo180 = ((phase >= 0.50) & (phase < 0.75)).astype(float)
    lo270 = ((phase >= 0.75)).astype(float)

    # this multiplies the four 1-hot signals with the antenna
    # this gives 4 seperate signals named as followed
    ip = lo0 * ant
    in_ = lo180 * ant
    qp = lo90 * ant
    qn = lo270 * ant

    # filtered signals (with capacitors)
    ip_f = lowpass_filter(ip, dt, 1e-5)
    in_f = lowpass_filter(in_, dt, 1e-5)
    qp_f = lowpass_filter(qp, dt, 1e-5)
    qn_f = lowpass_filter(qn, dt, 1e-5)
    
    # ip_f = moving_average(ip, N)
    # in_f = moving_average(in_, N)
    # qp_f = moving_average(qp, N)
    # qn_f = moving_average(qn, N)

    # this simulates the differential amplifier
    i_raw = ip_f - in_f
    q_raw = qp_f - qn_f

    # diff amp with capacitor
    i = 2 * lowpass_filter(i_raw, dt, 1e-6)
    q = 2 * lowpass_filter(q_raw, dt, 1e-6)

    # i = moving_average(i_raw, N)
    # q = moving_average(q_raw, N)

    # i and q are technically phasors and we can simply
    # take the sqrt of i^2+q^2
    A = 1.5 # the gain
    aud_out = A * np.sqrt(i**2 + q**2)








    # the rest isn't too important...
    # now we just plot the data
    rf.set_data(t, ant)

    l0_lo.set_data(t, lo0)
    l0_raw.set_data(t, ip)
    l0_f.set_data(t, ip_f)

    l180_lo.set_data(t, lo180)
    l180_raw.set_data(t, in_)
    l180_f.set_data(t, in_f)

    l90_lo.set_data(t, lo90)
    l90_raw.set_data(t, qp)
    l90_f.set_data(t, qp_f)

    l270_lo.set_data(t, lo270)
    l270_raw.set_data(t, qn)
    l270_f.set_data(t, qn_f)

    l_i_raw.set_data(t, i_raw)
    l_i_f.set_data(t, i)

    l_q_raw.set_data(t, q_raw)
    l_q_f.set_data(t, q)
    aud_sig.set_data(t, np.cos(2 * np.pi * f_audio * t - np.pi / 2))
    
    # reconst.set_data(t, aud_out)
    # expected.set_data(t, aud)

    # axs[0].set_ylabel(f"aud_freq: {f_audio/1000:.1f} kHz")
    axs[0].set_title(f"Carrier Frequency: {f_car/1000:0.1f} kHz")
    # axs[4].set_ylabel(f"lo_freq: {f_lo/1000:.1f} kHz")

    return (
        rf,
        l0_lo, l0_raw, l0_f,
        l180_lo, l180_raw, l180_f,
        l90_lo, l90_raw, l90_f,
        l270_lo, l270_raw, l270_f,
        l_i_raw, l_i_f,
        l_q_raw, l_q_f,
        # reconst
    )

ani = FuncAnimation(
    fig,
    update,
    frames=n_frames, # 2400,
    interval=1,
    blit=False
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# def progress(current, total):
#     percent = 100 * current / total
#     print(f"\rSaving: {percent:6.2f}%", end="", flush=True)

# ani.save(
#     "animation5.gif",
#     writer="pillow",
#     fps=20,
#     progress_callback=progress
# )

# print("\nDone.")