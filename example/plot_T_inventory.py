"""Plot tritium inventory over time from simulation results."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("results_bin0_W_high_wetted/results.json") as f:
    data = json.load(f)

t = np.array(data["t"])
T_mobile = np.array(data["T"]["data"])
T_trap1 = np.array(data["trap1_T"]["data"])
T_trap2 = np.array(data["trap2_T"]["data"])
T_total = T_mobile + T_trap1 + T_trap2

# Define shaded regions (time span in seconds)
# Edit these to match your scenario phases
regions = [
    {"name": "1st pulse Ramp-up", "t_start": 0, "t_end": 429, "color": "red", "alpha": 0.18},
    {"name": "1st pulse Flat top", "t_start": 429, "t_end": 429+650, "color": "green", "alpha": 0.18},
    {"name": "1st pulse Ramp-down", "t_start": 429+650, "t_end": 429+650+455, "color": "yellow", "alpha": 0.18},
    {"name": "STM (4 days waiting)", "t_start": 51340, "t_end": 396940, "color": "cyan", "alpha": 0.18},
    {"name": "Baking", "t_start": 396940, "t_end": 1001740, "color": "purple", "alpha": 0.18},
]

fig, ax = plt.subplots(figsize=(14, 6))

# Add shaded regions (labels added after plotting)
for region in regions:
    ax.axvspan(region["t_start"], region["t_end"], alpha=region["alpha"], 
               color=region["color"])

# Primary axis: trapped T and total
ax.plot(t, T_trap1, label="T trap 1", linewidth=2)
ax.plot(t, T_trap2, label="T trap 2", linewidth=2)
ax.plot(t, T_total, linewidth=2.5, label="T total", color="black")
ax.set_xlabel("Time (s)")
ax.set_xscale("log")
ax.set_xlim(1e2, t[-1])
ax.set_ylabel("Trapped Inventory (atoms/m²)")
ax.set_title("Tritium inventory — Bin 0 (W, high_wetted, FW)")
ax.tick_params(axis="y", labelcolor="C0")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, t[-1])

# Secondary axis: mobile T
ax2 = ax.twinx()
ax2.plot(t, T_mobile, label="T mobile", color="gray", linestyle="-", linewidth=0.8)
ax2.set_ylabel("Mobile Inventory (atoms/m²)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

# Add region labels with colored boxes below x-axis
from matplotlib.patches import Patch

# Inventory legend (traces) - upper left
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Region legend (shaded areas) - below plot
region_patches = [Patch(facecolor=region["color"], alpha=region["alpha"], 
                        edgecolor="black", linewidth=0.5, label=region["name"]) 
                  for region in regions]
fig.legend(handles=region_patches, loc="lower center", bbox_to_anchor=(0.5, -0.05),
           ncol=5, frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("results_bin0_W_high_wetted/T_inventory.png", dpi=150, bbox_inches="tight")
print("Saved T_inventory.png")
plt.show()
