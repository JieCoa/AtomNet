import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters
d0 = 1.0
d1 = 5.0
d = np.linspace(0, 5, 2000)

# Define psi(d) - smoothstep cutoff as before
psi = np.zeros_like(d)
mask_left = d <= d0
mask_right = d >= d1
mask_mid = (d > d0) & (d < d1)
psi[mask_left] = 1.0
psi[mask_right] = 0.0
t = (d[mask_mid] - d0) / (d1 - d0)
S = 6*t**5 - 15*t**4 + 10*t**3
psi[mask_mid] = 1.0 - S

# Define f(x) line WITHOUT value at x==1 (set NaN to avoid connecting across)
f_line = np.full_like(d, np.nan)
mask_f_left = (d >= 0) & (d <= 1.0)     # strictly less than 1.0 for the left continuous segment
mask_f_right = (d > 1.0) & (d <= 5.0)  # strictly greater than 1.0 for the right segment
f_line[mask_f_left] = 1.0
f_line[mask_f_right] = 1.0 - 0.2 * d[mask_f_right]
# Note: f_line at exactly d == 1.0 remains NaN -> no connecting line

# Marker y-values at x = 1.0
y_top = 1.0
y_bottom = 1.0 - 0.2 * 1.0  # 0.8

# Plot
fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
ax.plot(d, psi, label=r'$\psi(x)$ (Cubic Smooth)', linewidth=2.0, color='orange')
ax.plot(d, f_line, label=r'$f(x)$ (Simply)', linewidth=2.0, color='lightgreen')

# Draw exactly two circles at x=1.0: filled (top) and hollow (bottom)
ax.plot(1.0, y_top, marker='o', markersize=8, markeredgewidth=1.2,
        markerfacecolor='lightgreen', markeredgecolor='lightgreen')  # filled top point
ax.plot(1.0, y_bottom, marker='o', markersize=8, markeredgewidth=1.6,
        markerfacecolor='none', markeredgecolor='lightgreen')    # hollow bottom point

# Only horizontal grid lines
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

ax.set_xlim(0, 5)
ax.set_xlabel('Distance')
ax.set_ylabel('Weight')
# ax.set_title(r'$\psi(d)$ (orange) and $f(x)$ (light green) — only two circles at $x=1$ (no vertical line)')
ax.legend()

# Save files
png_path = 'psi_f_plot.png'
svg_path = 'psi_f_plot.svg'
fig.savefig(png_path, dpi=500, bbox_inches='tight')
fig.savefig(svg_path, bbox_inches='tight')

plt.show()
