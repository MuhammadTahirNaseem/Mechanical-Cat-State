import numpy as np
import matplotlib.pyplot as plt
import csv

# ------------------------------------------------------------
# 1) Load data from CSV file
# ------------------------------------------------------------
csv_path = r"E:\Research\Submitted\CatState\Codes\Python\GitHub\Fig3\detuning_scan_results.csv"

data = {}  # will hold {scan_name: {"x": [...], "F": [...]}}

with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["scan"]                 # e.g. "Omega", "Delta2", ...
        x_val = float(row["x_value"])
        F_val = float(row["Fidelity"])

        if name not in data:
            data[name] = {"x": [], "F": []}

        data[name]["x"].append(x_val)
        data[name]["F"].append(F_val)

# Sort each scan by x and convert to numpy arrays
for name in data:
    x = np.array(data[name]["x"], dtype=float)
    F = np.array(data[name]["F"], dtype=float)
    idx = np.argsort(x)
    data[name]["x"] = x[idx]
    data[name]["F"] = F[idx]

# ------------------------------------------------------------
# 2) Plot fidelities: one figure, six panels (3x2)
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2, 3,
    figsize=(6.9, 4.0),
    dpi=600,
    sharey=True
)
axes = axes.ravel()

scan_order = ["Omega", "Delta2", "gx", "gz", "omega_m", "gamma_m"]
axis_labels = {
    "Omega":    r"$\Omega/\kappa$",
    "Delta2":   r"$\Delta_{2-}/\kappa$",
    "gx":       r"$g_x/\kappa$",
    "gz":       r"$g_z/\kappa$",
    "omega_m":  r"$\omega_m/\kappa$",
    "gamma_m":  r"$\gamma/\kappa$",
}
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for i, name in enumerate(scan_order):
    ax = axes[i]
    x = data[name]["x"]
    y = data[name]["F"]   # directly plot fidelity

    ax.plot(x, y, "-", lw=1.2, marker="o", ms=3)

    # Zoom near unity to highlight deviations
    ax.set_ylim(0.98, 1.0005)

    # X-axis label
    ax.set_xlabel(axis_labels.get(name, name), fontsize=9)

    # Y-axis label only on left column
    if i % 3 == 0:
        ax.set_ylabel(r"$\mathcal{F}$", fontsize=10)

    # Panel label
    ax.text(
        0.06, 0.15, panel_labels[i],
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left", va="top"
    )

    # Styling
    ax.grid(True, which="both", alpha=0.12)
    ax.minorticks_on()
    ax.tick_params(
        axis="both", which="major",
        labelsize=7, length=3.0, width=0.6,
        direction="in", top=True, right=True
    )
    ax.tick_params(
        axis="both", which="minor",
        labelsize=7, length=2.0, width=0.4,
        direction="in", top=True, right=True
    )
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_linewidth(0.6)

fig.tight_layout()

# Save if desired
fig.savefig("fidelity_scans.png", dpi=600, bbox_inches="tight")
fig.savefig("fidelity_scans.pdf", dpi=600, bbox_inches="tight")

plt.show()
