"""
VERSE Pulse Optimizer
=====================
Takes a standard RF pulse (with constant gradient) and compresses it to
minimum duration using Variable-Rate Selective Excitation (VERSE).

Based on: Hargreaves et al., "Variable-Rate Selective Excitation for Rapid
MRI Sequences", Magn Reson Med 52:590-597 (2004).

The algorithm computes a time-dilation function tau_dot(t) such that the
VERSE-transformed pulse has the same excitation profile but shorter duration.
At every moment, at least one hardware constraint (RF amplitude, gradient
amplitude, or gradient slew rate) is at its limit.

Written in NumPy array style for easy translation to MATLAB.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# PULSE LIBRARY: premade test pulses
# ============================================================================
# Each function returns (B1, G_orig, dt, name) where:
#   B1      = RF waveform array (uT)
#   G_orig  = constant gradient amplitude (mT/m)
#   dt      = sample interval (s)
#   name    = descriptive string for plot titles

# Gyromagnetic ratio for 1H (hydrogen)
# gamma_bar = 42.577 MHz/T = 42.577e6 Hz/T
GAMMA = 42.577e6        # Hz/T
GAMMA_RAD = 2 * np.pi * 42.577   # rad/s per uT (for flip angle calculation)


def make_sinc_pulse(TB=10, T_pulse=2.9e-3, flip_angle=60, slab_width=40e-3,
                    dt=4e-6):
    """Hamming-windowed sinc pulse. The standard test case from the paper.

    Parameters
    ----------
    TB         : Time-bandwidth product. Higher = sharper slab profile.
    T_pulse    : Duration of the standard pulse (s).
    flip_angle : Desired flip angle (degrees).
    slab_width : Slab width (m).
    dt         : Sample interval (s).
    """
    # Constant gradient for the standard pulse
    G_orig = TB / (GAMMA * slab_width * T_pulse) * 1e3   # mT/m

    N = int(np.round(T_pulse / dt))
    if N % 2 == 0:
        N += 1   # Odd number of samples so sinc is centered

    t = np.arange(N) * dt
    t_centered = t - t[-1] / 2

    # Windowed sinc
    # np.sinc(x) computes sin(pi*x)/(pi*x), same as MATLAB's sinc
    B1 = np.sinc(TB * t_centered / T_pulse)
    B1 = B1 * np.hamming(N)   # np.hamming(N) = MATLAB's hamming(N)

    # Scale to desired flip angle
    flip_rad = np.deg2rad(flip_angle)
    B1 = B1 * flip_rad / (GAMMA_RAD * np.sum(B1) * dt)

    name = f"Sinc (TB={TB}, {flip_angle}°, {T_pulse*1e3:.1f}ms)"
    return B1, G_orig, dt, name


def make_gaussian_pulse(T_pulse=2.0e-3, flip_angle=60, slab_width=40e-3,
                        dt=4e-6, truncation=2.5):
    """Gaussian RF pulse, truncated at 'truncation' standard deviations.

    Parameters
    ----------
    T_pulse    : Duration of the standard pulse (s).
    flip_angle : Desired flip angle (degrees).
    slab_width : Slab width (m).
    dt         : Sample interval (s).
    truncation : Number of standard deviations before truncation.
    """
    N = int(np.round(T_pulse / dt))
    if N % 2 == 0:
        N += 1

    t = np.arange(N) * dt
    t_centered = t - t[-1] / 2
    sigma = T_pulse / (2 * truncation)

    # Gaussian envelope: exp(-t^2 / (2*sigma^2))
    B1 = np.exp(-t_centered**2 / (2 * sigma**2))

    # Bandwidth ~ 1/sigma (approximate for Gaussian)
    BW = 1.0 / sigma
    G_orig = BW / (GAMMA * slab_width) * 1e3   # mT/m

    # Scale to desired flip angle
    flip_rad = np.deg2rad(flip_angle)
    B1 = B1 * flip_rad / (GAMMA_RAD * np.sum(B1) * dt)

    name = f"Gaussian ({flip_angle}°, {T_pulse*1e3:.1f}ms)"
    return B1, G_orig, dt, name


def make_block_pulse(T_pulse=0.5e-3, flip_angle=90, slab_width=40e-3,
                     dt=4e-6):
    """Rectangular (block/hard) pulse. Constant amplitude, abrupt edges.

    Parameters
    ----------
    T_pulse    : Duration of the standard pulse (s).
    flip_angle : Desired flip angle (degrees).
    slab_width : Slab width (m).
    dt         : Sample interval (s).
    """
    N = int(np.round(T_pulse / dt))

    # Flat RF amplitude
    B1 = np.ones(N)

    # For a block pulse the bandwidth is roughly 1/T_pulse
    BW = 1.0 / T_pulse
    G_orig = BW / (GAMMA * slab_width) * 1e3   # mT/m

    # Scale to desired flip angle
    flip_rad = np.deg2rad(flip_angle)
    B1 = B1 * flip_rad / (GAMMA_RAD * np.sum(B1) * dt)

    name = f"Block ({flip_angle}°, {T_pulse*1e3:.1f}ms)"
    return B1, G_orig, dt, name


# ============================================================================
# SELECT WHICH PULSE TO USE
# ============================================================================
# Uncomment one of the following lines to try different pulses:

B1, G_orig, dt, pulse_name = make_sinc_pulse(TB=10, T_pulse=2.9e-3, flip_angle=60)
# B1, G_orig, dt, pulse_name = make_sinc_pulse(TB=5, T_pulse=1.5e-3, flip_angle=60)
# B1, G_orig, dt, pulse_name = make_gaussian_pulse(T_pulse=2.0e-3, flip_angle=60)
# B1, G_orig, dt, pulse_name = make_block_pulse(T_pulse=0.5e-3, flip_angle=90)


# ============================================================================
# HARDWARE LIMITS (configurable)
# ============================================================================
B1_max = 15.0       # Maximum RF amplitude, uT
G_max = 40.0        # Maximum gradient amplitude, mT/m
S_max = 150e3       # Maximum gradient slew rate, mT/m/s  (= 150 T/m/s)


# ============================================================================
# VERSE ALGORITHM
# ============================================================================
N = len(B1)
t_orig = np.arange(N) * dt
T_orig = N * dt
epsilon = 1e-10

print("=" * 60)
print(f"ORIGINAL PULSE: {pulse_name}")
print("=" * 60)
print(f"  Duration:        {T_orig * 1e3:.3f} ms")
print(f"  Samples:         {N}")
print(f"  Sample interval: {dt * 1e6:.1f} us")
print(f"  Max |B1|:        {np.max(np.abs(B1)):.3f} uT")
print(f"  Gradient:        {G_orig:.2f} mT/m (constant)")

# ---------- Step 1: Pointwise tau_dot (ignoring slew) ----------
print("\n" + "=" * 60)
print("STEP 1: Pointwise tau_dot (ignoring slew rate)")
print("=" * 60)

# RF constraint: |B1[n] * tau_dot| <= B1_max
tau_dot_rf = B1_max / np.maximum(np.abs(B1), epsilon)

# Gradient constraint: G * tau_dot <= G_max (constant across all samples)
tau_dot_grad = G_max / G_orig

# Pointwise tau_dot: the tighter (smaller) of the two at each sample
tau_dot_pw = np.minimum(tau_dot_rf, tau_dot_grad)

rf_active = tau_dot_rf < tau_dot_grad
print(f"  RF is the bottleneck at   {np.sum(rf_active)} / {N} samples")
print(f"  Grad is the bottleneck at {np.sum(~rf_active)} / {N} samples")
print(f"  tau_dot range: {np.min(tau_dot_pw):.2f} to {np.max(tau_dot_pw):.2f}")

# --- Diagnostic plot: Step 1 ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(f"Step 1: Pointwise tau_dot — {pulse_name}")

axes[0].plot(t_orig * 1e3, B1, 'b-', linewidth=1)
axes[0].axhline(y=B1_max, color='r', linestyle='--', label=f'B1_max = {B1_max} uT')
axes[0].axhline(y=-B1_max, color='r', linestyle='--')
axes[0].set_ylabel('B1 (uT)')
axes[0].set_title('Original RF pulse')
axes[0].legend()

axes[1].plot(t_orig * 1e3, tau_dot_rf, 'b-', alpha=0.5, label='tau_dot from RF limit')
axes[1].axhline(y=tau_dot_grad, color='g', linestyle='--',
                label=f'tau_dot from gradient limit = {tau_dot_grad:.2f}')
axes[1].plot(t_orig * 1e3, tau_dot_pw, 'k-', linewidth=2, label='tau_dot pointwise (min)')
axes[1].set_ylabel('tau_dot')
axes[1].set_xlabel('Original time (ms)')
axes[1].set_title('Maximum tau_dot at each sample')
axes[1].set_ylim([0, tau_dot_grad * 1.5])
axes[1].legend()

plt.tight_layout()
plt.savefig('step1_pointwise_tau_dot.png', dpi=150)
plt.show(block=False)
plt.pause(0.1)


# ---------- Step 2: Enforce slew rate (forward + backward pass) ----------
print("\n" + "=" * 60)
print("STEP 2: Enforce slew rate (forward + backward pass)")
print("=" * 60)

# Starting value: one slew-rate step from zero gradient.
# G_verse starts at S_max * dt instead of exactly 0 — negligible in practice.
tau_dot_start = S_max * dt / G_orig
print(f"  tau_dot start: {tau_dot_start:.4f}  (G_verse start = {G_orig * tau_dot_start:.2f} mT/m)")

# Forward pass: ramp up from near-zero gradient, limited by slew.
# Recurrence: tau_dot[n+1] <= tau_dot[n] + S_max * dt / (G * tau_dot[n])
# Cannot be vectorized — each step depends on the previous.
tau_dot_fwd = np.zeros(N)
tau_dot_fwd[0] = tau_dot_start
for n in range(N - 1):
    max_increase = S_max * dt / (G_orig * tau_dot_fwd[n])
    tau_dot_fwd[n + 1] = min(tau_dot_pw[n + 1],
                              tau_dot_fwd[n] + max_increase)

# Backward pass: ramp down to near-zero gradient.
tau_dot_bwd = np.zeros(N)
tau_dot_bwd[-1] = tau_dot_start
for n in range(N - 2, -1, -1):
    max_increase = S_max * dt / (G_orig * tau_dot_bwd[n + 1])
    tau_dot_bwd[n] = min(tau_dot_pw[n],
                          tau_dot_bwd[n + 1] + max_increase)

# Final tau_dot: minimum of forward and backward envelopes
tau_dot = np.minimum(tau_dot_fwd, tau_dot_bwd)

# Which constraint is active at each point?
is_fwd = (tau_dot_fwd <= tau_dot_bwd) & (tau_dot_fwd <= tau_dot_pw)
is_bwd = (tau_dot_bwd < tau_dot_fwd) & (tau_dot_bwd <= tau_dot_pw)
is_pw = ~is_fwd & ~is_bwd

print(f"  Forward slew active at:  {np.sum(is_fwd)} samples (ramp-up)")
print(f"  Backward slew active at: {np.sum(is_bwd)} samples (ramp-down)")
print(f"  Pointwise limit active:  {np.sum(is_pw)} samples (plateau)")
print(f"  Final tau_dot range: {np.min(tau_dot):.4f} to {np.max(tau_dot):.4f}")

# --- Diagnostic plot: Step 2 ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(f"Step 2: Slew rate enforcement — {pulse_name}")

axes[0].plot(t_orig * 1e3, tau_dot_pw, 'b--', alpha=0.5, linewidth=1,
             label='Pointwise tau_dot (Step 1)')
axes[0].plot(t_orig * 1e3, tau_dot_fwd, 'g-', alpha=0.3, label='Forward pass')
axes[0].plot(t_orig * 1e3, tau_dot_bwd, 'r-', alpha=0.3, label='Backward pass')
axes[0].plot(t_orig * 1e3, tau_dot, 'k-', linewidth=2, label='Final tau_dot')
axes[0].set_ylabel('tau_dot')
axes[0].set_title('tau_dot: pointwise vs slew-limited')
axes[0].legend()

constraint_map = np.zeros(N)
constraint_map[is_fwd] = 1
constraint_map[is_bwd] = 2
constraint_map[is_pw] = 3
axes[1].plot(t_orig * 1e3, constraint_map, 'k-', linewidth=1)
axes[1].set_yticks([1, 2, 3])
axes[1].set_yticklabels(['Fwd slew\n(ramp-up)', 'Bwd slew\n(ramp-down)',
                          'RF or Grad\n(plateau)'])
axes[1].set_xlabel('Original time (ms)')
axes[1].set_title('Active constraint at each sample')

plt.tight_layout()
plt.savefig('step2_slew_enforcement.png', dpi=150)
plt.show(block=False)
plt.pause(0.1)


# ---------- Step 3: Compute VERSE waveforms ----------
print("\n" + "=" * 60)
print("STEP 3: Compute VERSE waveforms")
print("=" * 60)

B1_verse = B1 * tau_dot
G_verse = G_orig * tau_dot

# New (non-uniform) time axis
dt_new = dt / tau_dot
t_new = np.concatenate([[0], np.cumsum(dt_new[:-1])])
T_new = t_new[-1] + dt_new[-1]

print(f"  Original duration: {T_orig * 1e3:.3f} ms")
print(f"  VERSE duration:    {T_new * 1e3:.3f} ms")
print(f"  Compression:       {T_orig / T_new:.1f}x")
print(f"  Max |B1_verse|:    {np.max(np.abs(B1_verse)):.3f} / {B1_max} uT")
print(f"  Max |G_verse|:     {np.max(np.abs(G_verse)):.3f} / {G_max} mT/m")

# --- Diagnostic plot: Step 3 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle(f"Step 3: Original vs VERSE — {pulse_name}")

axes[0, 0].plot(t_orig * 1e3, B1, 'b-')
axes[0, 0].axhline(y=B1_max, color='r', linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=-B1_max, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_title(f'Original RF ({T_orig*1e3:.2f} ms)')
axes[0, 0].set_ylabel('B1 (uT)')
axes[0, 0].set_xlabel('Time (ms)')

axes[0, 1].plot(t_new * 1e3, B1_verse, 'b-')
axes[0, 1].axhline(y=B1_max, color='r', linestyle='--', alpha=0.5,
                    label=f'B1_max = {B1_max} uT')
axes[0, 1].axhline(y=-B1_max, color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_title(f'VERSE RF ({T_new*1e3:.2f} ms)')
axes[0, 1].set_ylabel('B1 (uT)')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].legend()

axes[1, 0].plot(t_orig * 1e3, np.full(N, G_orig), 'g-')
axes[1, 0].axhline(y=G_max, color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Original gradient (constant)')
axes[1, 0].set_ylabel('G (mT/m)')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylim([0, G_max * 1.2])

axes[1, 1].plot(t_new * 1e3, G_verse, 'g-')
axes[1, 1].axhline(y=G_max, color='r', linestyle='--', alpha=0.5,
                    label=f'G_max = {G_max} mT/m')
axes[1, 1].set_title('VERSE gradient (with ramps)')
axes[1, 1].set_ylabel('G (mT/m)')
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylim([0, G_max * 1.2])
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('step3_verse_waveforms.png', dpi=150)
plt.show(block=False)
plt.pause(0.1)


# ---------- Step 4: Resample onto uniform time grid ----------
print("\n" + "=" * 60)
print("STEP 4: Resample onto uniform time grid")
print("=" * 60)

t_uniform = np.arange(0, T_new, dt)
N_final = len(t_uniform)

# np.interp = linear interpolation, equivalent to MATLAB's interp1
B1_final = np.interp(t_uniform, t_new, B1_verse)
G_final = np.interp(t_uniform, t_new, G_verse)

print(f"  Resampled to {N_final} uniform samples (from {N} original)")
print(f"  Sample interval: {dt * 1e6:.1f} us")

# Resampling can introduce slew violations (interpolation doesn't preserve
# derivative constraints). Fix by enforcing slew on the uniform grid,
# then recomputing B1 to maintain the B1/G ratio.
max_dG = S_max * dt   # Maximum gradient change per sample (mT/m)

G_slew_fwd = np.copy(G_final)
for n in range(N_final - 1):
    if G_slew_fwd[n + 1] - G_slew_fwd[n] > max_dG:
        G_slew_fwd[n + 1] = G_slew_fwd[n] + max_dG
    elif G_slew_fwd[n + 1] - G_slew_fwd[n] < -max_dG:
        G_slew_fwd[n + 1] = G_slew_fwd[n] - max_dG

G_slew_bwd = np.copy(G_slew_fwd)
for n in range(N_final - 2, -1, -1):
    if G_slew_bwd[n] - G_slew_bwd[n + 1] > max_dG:
        G_slew_bwd[n] = G_slew_bwd[n + 1] + max_dG
    elif G_slew_bwd[n] - G_slew_bwd[n + 1] < -max_dG:
        G_slew_bwd[n] = G_slew_bwd[n + 1] - max_dG

G_final = G_slew_bwd

# Recompute B1 from the B1/G ratio (invariant under VERSE: B1'/G' = B1/G)
ratio_verse = B1_verse / np.maximum(G_verse, epsilon)
ratio_uniform = np.interp(t_uniform, t_new, ratio_verse)
B1_final = ratio_uniform * G_final
B1_final = np.clip(B1_final, -B1_max, B1_max)

print(f"  Slew enforcement on uniform grid: done")


# ---------- Step 5: Verify all constraints ----------
print("\n" + "=" * 60)
print("STEP 5: Verify constraints on final waveforms")
print("=" * 60)

max_B1 = np.max(np.abs(B1_final))
max_G = np.max(np.abs(G_final))
slew = np.diff(G_final) / dt
max_slew = np.max(np.abs(slew))

print(f"  Max |B1|:  {max_B1:.3f} / {B1_max} uT", end="")
print("  OK" if max_B1 <= B1_max * 1.01 else "  *** VIOLATED ***")
print(f"  Max |G|:   {max_G:.3f} / {G_max} mT/m", end="")
print("  OK" if max_G <= G_max * 1.01 else "  *** VIOLATED ***")
print(f"  Max slew:  {max_slew:.0f} / {S_max:.0f} mT/m/s", end="")
print("  OK" if max_slew <= S_max * 1.01 else "  *** VIOLATED ***")

# --- Diagnostic plot: constraint verification ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle(f"Step 5: Constraint verification — {pulse_name}")

axes[0].plot(t_uniform * 1e3, B1_final, 'b-', linewidth=1)
axes[0].axhline(y=B1_max, color='r', linestyle='--', label=f'B1_max = {B1_max} uT')
axes[0].axhline(y=-B1_max, color='r', linestyle='--')
axes[0].set_ylabel('B1 (uT)')
axes[0].set_title('RF waveform')
axes[0].legend()

axes[1].plot(t_uniform * 1e3, G_final, 'g-', linewidth=1)
axes[1].axhline(y=G_max, color='r', linestyle='--', label=f'G_max = {G_max} mT/m')
axes[1].axhline(y=-G_max, color='r', linestyle='--')
axes[1].set_ylabel('G (mT/m)')
axes[1].set_title('Gradient waveform')
axes[1].legend()

t_slew = (t_uniform[:-1] + t_uniform[1:]) / 2
axes[2].plot(t_slew * 1e3, slew, 'm-', linewidth=1)
axes[2].axhline(y=S_max, color='r', linestyle='--', label=f'S_max = {S_max:.0f} mT/m/s')
axes[2].axhline(y=-S_max, color='r', linestyle='--')
axes[2].set_ylabel('dG/dt (mT/m/s)')
axes[2].set_xlabel('Time (ms)')
axes[2].set_title('Gradient slew rate')
axes[2].legend()

plt.tight_layout()
plt.savefig('step5_constraint_verification.png', dpi=150)
plt.show(block=False)
plt.pause(0.1)


# ============================================================================
# COMPARISON PLOT: original and VERSE pulse on the same time axis
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(f"Before & After VERSE — {pulse_name}")

# RF comparison
axes[0].plot(t_orig * 1e3, B1, 'b-', alpha=0.6, linewidth=1.5,
             label=f'Original ({T_orig*1e3:.2f} ms)')
axes[0].plot(t_uniform * 1e3, B1_final, 'r-', linewidth=1.5,
             label=f'VERSE ({T_new*1e3:.2f} ms)')
axes[0].axhline(y=B1_max, color='k', linestyle=':', alpha=0.3)
axes[0].axhline(y=-B1_max, color='k', linestyle=':', alpha=0.3)
axes[0].set_ylabel('B1 (uT)')
axes[0].set_title('RF waveform')
axes[0].legend()

# Gradient comparison
axes[1].plot(t_orig * 1e3, np.full(N, G_orig), 'b-', alpha=0.6, linewidth=1.5,
             label=f'Original (constant {G_orig:.1f} mT/m)')
axes[1].plot(t_uniform * 1e3, G_final, 'r-', linewidth=1.5,
             label=f'VERSE (time-varying)')
axes[1].axhline(y=G_max, color='k', linestyle=':', alpha=0.3)
axes[1].set_ylabel('G (mT/m)')
axes[1].set_xlabel('Time (ms)')
axes[1].set_title('Gradient waveform')
axes[1].legend()

plt.tight_layout()
plt.savefig('comparison_before_after.png', dpi=150)
plt.show(block=False)
plt.pause(0.1)


# ============================================================================
# OUTPUT: save VERSE pulse to text file
# ============================================================================
# Format: tab-separated columns, easy to load in MATLAB (load('file.txt'))
# or Python (np.loadtxt('file.txt'))
output_file = 'verse_pulse_output.txt'

# Build output array: time (us) | B1 (uT) | gradient (mT/m)
output_data = np.column_stack([
    t_uniform * 1e6,    # time in us
    B1_final,           # RF amplitude in uT
    G_final             # gradient in mT/m
])

# Header with all parameters for reproducibility
header_lines = [
    f"VERSE Pulse Output",
    f"Source pulse: {pulse_name}",
    f"Original duration: {T_orig*1e3:.3f} ms  |  VERSE duration: {T_new*1e3:.3f} ms  |  Compression: {T_orig/T_new:.1f}x",
    f"Hardware limits: B1_max={B1_max} uT, G_max={G_max} mT/m, S_max={S_max/1e3:.0f} T/m/s",
    f"Sample interval: {dt*1e6:.1f} us  |  Samples: {N_final}",
    f"",
    f"time_us\tB1_uT\tG_mTm",
]
header = "\n".join(header_lines)

# np.savetxt writes a text file, equivalent to MATLAB's dlmwrite or save -ascii
np.savetxt(output_file, output_data, fmt='%.6f', delimiter='\t', header=header)

print("\n" + "=" * 60)
print("OUTPUT")
print("=" * 60)
print(f"  Saved to: {output_file}")
print(f"  Columns:  time (us) | B1 (uT) | gradient (mT/m)")
print(f"  Samples:  {N_final}")
print(f"")
print(f"  Load in MATLAB:  data = load('{output_file}');")
print(f"  Load in Python:  data = np.loadtxt('{output_file}')")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  {pulse_name}")
print(f"  {T_orig*1e3:.2f} ms  -->  {T_new*1e3:.2f} ms  ({T_orig/T_new:.1f}x compression)")

plt.show()   # Keep all figures open at the end
