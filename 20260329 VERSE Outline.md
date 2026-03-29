# 1. Project Plan: The VERSE Pulse Optimizer

**Objective:** A Python utility that takes an arbitrary MRI RF pulse and "time-smashes" it into the shortest possible duration while adhering to hardware safety limits.

## Introduction & The Formula

The **Variable-Rate Selective Excitation (VERSE)** technique relies on the principle that the flip angle (excitation profile) depends on the integral of the RF pulse over time. If you "speed up" time, you must "scale up" the amplitude to maintain the same effect.

The core transformation involves a time-mapping function $\tau(t)$, where:

$$B_1'(\tau) = \frac{B_1(t)}{d\tau/dt} \quad \text{and} \quad G'(\tau) = \frac{G(t)}{d\tau/dt}$$

By varying $d\tau/dt$ (the rate of time), we can reshape the pulse.

## The Three Safety Constraints

At every point in time, the VERSE-transformed pulse must respect:

- **$|B_1'(t)| \le B_{1,\text{max}}$:** Maximum RF amplifier amplitude.
- **$|G'(t)| \le G_{\text{max}}$:** Maximum gradient strength.
- **$|dG'/dt| \le S_{\text{max}}$:** The "Slew Rate" (maximum change in gradient per time).

These are configurable parameters — different scanners have different limits.

## The Algorithm

### Input

The starting point is a "standard" RF pulse played with a **constant** gradient.
This is the original pulse *before* VERSE — it has no ramps, just a flat gradient
throughout. VERSE will transform this into a time-varying gradient (with natural
ramp-up and ramp-down), which is one of the key benefits.

```python
B1 = np.array([...])      # Original RF waveform (N samples), in uT
G = 40.0                   # Constant gradient during original pulse, mT/m
dt = 4e-6                  # Sample interval, 4 us (as in Hargreaves et al.)

# Hardware limits
B1_max = 15.0              # uT
G_max = 40.0               # mT/m
S_max = 150e3              # mT/m/s (= 150 T/m/s)

N = len(B1)
```

### Step 1: Pointwise tau_dot (RF and gradient limits, ignoring slew)

At each sample, compute the maximum $\dot{\tau}$ allowed by the RF limit and gradient
limit independently, then take the stricter one.

Since the original gradient is constant, the gradient constraint gives the same
limit at every point. The RF constraint varies: where the RF is strong (peaks of
the sinc), it's the bottleneck. Where the RF is weak (zero-crossings), the gradient
constraint takes over.

```python
epsilon = 1e-10

# RF constraint: |B1[n] * tau_dot| <= B1_max
# Where B1 ~ 0 (zero-crossings), this allows tau_dot -> infinity, so cap it
tau_dot_rf = B1_max / np.maximum(np.abs(B1), epsilon)

# Gradient constraint: G * tau_dot <= G_max  (same at every point)
tau_dot_grad = G_max / G

# Pointwise maximum tau_dot: take the tighter constraint at each point
tau_dot_pw = np.minimum(tau_dot_rf, tau_dot_grad)
```

**Diagnostic plot:** Show `tau_dot_pw` overlaid with `tau_dot_rf` and the constant
`tau_dot_grad` line. This should show tau_dot_rf dipping at each RF peak and
tau_dot_grad as a flat ceiling. Print `max(tau_dot_pw)` and `min(tau_dot_pw)`.

### Step 2: Enforce slew rate (forward + backward pass)

The slew constraint couples adjacent samples — you can't just set each point
independently. The gradient change between consecutive samples is:

$$\text{slew} = \frac{G \cdot \dot{\tau}[n+1] - G \cdot \dot{\tau}[n]}{\Delta t'[n]}$$

where $\Delta t'[n] = dt / \dot{\tau}[n]$ is the time step in the new (compressed) domain.
Rearranging, the maximum allowed increase in $\dot{\tau}$ per step is:

$$\dot{\tau}[n+1] \leq \dot{\tau}[n] + \frac{S_{\text{max}} \cdot dt}{G \cdot \dot{\tau}[n]}$$

This is a recurrence: each value depends on the previous one.
**This cannot be vectorized with NumPy** — it requires a for-loop. (Same would
be true in MATLAB.)

The forward pass handles the ramp-up from $G'=0$ at the start.
The backward pass handles the ramp-down to $G'=0$ at the end.

```python
# Forward pass: starting from gradient = 0, how fast can tau_dot grow?
tau_dot_fwd = np.zeros(N)
tau_dot_fwd[0] = epsilon
for n in range(N - 1):
    # Max allowed increase given slew rate and current time step
    max_increase = S_max * dt / (G * tau_dot_fwd[n])
    tau_dot_fwd[n + 1] = min(tau_dot_pw[n + 1],
                              tau_dot_fwd[n] + max_increase)

# Backward pass: ending at gradient = 0, how fast can tau_dot shrink?
tau_dot_bwd = np.zeros(N)
tau_dot_bwd[-1] = epsilon
for n in range(N - 2, -1, -1):
    max_increase = S_max * dt / (G * tau_dot_bwd[n + 1])
    tau_dot_bwd[n] = min(tau_dot_pw[n],
                          tau_dot_bwd[n + 1] + max_increase)

# Final tau_dot: minimum of forward and backward envelopes
tau_dot = np.minimum(tau_dot_fwd, tau_dot_bwd)
```

**What this produces:** A smooth $\dot{\tau}$ curve that ramps up from zero (gradient
ramp-up), reaches a plateau where RF or gradient is the active constraint, and
ramps back down to zero (gradient ramp-down). The ramp shape resembles
$\sqrt{t}$ because when $\dot{\tau}$ is small, the allowed increase per step is large
(more new-time available per original sample).

**Diagnostic plot:** Show `tau_dot_pw` (from Step 1) as a dashed line, and the
final `tau_dot` as a solid line overlaid. The difference shows where the slew rate
"ate into" the pointwise optimum. Print the number of samples where slew is the
active constraint vs RF vs gradient.

### Step 3: Compute the VERSE waveforms

Apply the VERSE transformation using $\dot{\tau}$:

```python
# VERSE RF waveform: amplitude scales with tau_dot
B1_verse = B1 * tau_dot

# VERSE gradient waveform: G * tau_dot at each point
G_verse = G * tau_dot

# New time axis (non-uniform spacing)
dt_new = dt / tau_dot                                    # Time per sample in new domain
t_new = np.concatenate([[0], np.cumsum(dt_new[:-1])])    # Cumulative time
T_new = t_new[-1] + dt_new[-1]                           # Total new duration
T_orig = N * dt                                          # Original duration

print(f"Original duration: {T_orig * 1e3:.2f} ms")
print(f"VERSE duration:    {T_new * 1e3:.2f} ms")
print(f"Compression:       {T_orig / T_new:.1f}x")
```

**Diagnostic plot:** Side-by-side comparison of original B1 vs VERSE B1, and
flat gradient vs VERSE gradient, both plotted against their respective time axes.

### Step 4: Resample onto uniform time grid

Scanners need uniformly sampled waveforms. The VERSE waveforms from Step 3
sit on a non-uniform time grid, so we interpolate them onto a uniform grid.

```python
# Uniform time grid at the original sample rate
t_uniform = np.arange(0, T_new, dt)

# Linear interpolation (equivalent to MATLAB's interp1)
B1_final = np.interp(t_uniform, t_new, B1_verse)
G_final = np.interp(t_uniform, t_new, G_verse)
```

### Step 5: Verify all constraints

Check that the final resampled waveforms respect all three limits.
Resampling (interpolation) can introduce small violations, so this step is
essential.

```python
# RF amplitude
print(f"Max |B1|: {np.max(np.abs(B1_final)):.2f} / {B1_max} uT")

# Gradient amplitude
print(f"Max |G|:  {np.max(np.abs(G_final)):.2f} / {G_max} mT/m")

# Slew rate: computed on the uniform grid
slew = np.diff(G_final) / dt
print(f"Max slew: {np.max(np.abs(slew)):.0f} / {S_max} mT/m/s")
```

**Diagnostic plot (3 panels):**

1. **RF waveform** with horizontal lines at $\pm B_{1,\text{max}}$
2. **Gradient waveform** with horizontal lines at $\pm G_{\text{max}}$
3. **Slew rate** (`np.diff(G_final) / dt`) with horizontal lines at $\pm S_{\text{max}}$

These plots are the main visual proof that the algorithm works: at every moment,
at least one constraint should be close to its limit, and none should be exceeded.

## The Relativity Thought Experiment

This is a brilliant conceptual bridge. In Physics, **Covariance** suggests that the laws of physics should remain the same for observers in different reference frames.

**The Experiment:** Imagine an observer moving at a relativistic speed relative to the MRI scanner. To them, time appears to slow down or speed up.

**The Result:** Since the Bloch equations (which describe MRI) are essentially a set of rotations, the "total rotation" (excitation) is invariant so long as the fields ($B_1$ and $G$) are scaled to compensate for the "stretched" or "compressed" time. The VERSE formula is effectively a mathematical way of simulating this "local time dilation" for the atoms.
