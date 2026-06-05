import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-style formatting
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# ==========================================
# 1. HARDCODED DATA (Overdiagnosed Class + Accuracy)
# ==========================================
years = ['5 Yr', '6 Yr', '7 Yr', '8 Yr', '9 Yr', '10 Yr']
x = np.arange(len(years))  # Label locations
width = 0.2                # Width of the bars

# --- EXPERIMENT 1: Max Clinical Safety (UPDATED WITH DYNAMIC ZERO-FP RESULTS) ---
safe_p   = [100, 100, 100, 100, 100, 100]
safe_r   = [21, 33, 27, 21, 16, 27]
safe_f1  = [35, 49, 43, 35, 27, 43]
safe_acc = [82, 87, 88, 88, 89, 91]

# --- EXPERIMENT 2: 50-50 Balanced Cost (UNCHANGED BASELINE) ---
bal_p    = [85, 86, 81, 71, 79, 86]
bal_r    = [46, 50, 50, 54, 40, 37]
bal_f1   = [60, 63, 62, 62, 53, 52]
bal_acc  = [86, 88, 89, 90, 91, 92]

# --- EXPERIMENT 3: Max OD Detection (UNCHANGED BASELINE) ---
maxod_p  = [46, 49, 53, 42, 42, 38]
maxod_r  = [82, 82, 80, 77, 70, 77]
maxod_f1 = [59, 61, 63, 54, 52, 51]
maxod_acc= [73, 79, 84, 80, 83, 83]

# ==========================================
# 2. GENERATE 1x3 GROUPED BAR CHARTS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle("Performance Metrics by Cost Optimization (Target: Overdiagnosed Cohort)", 
             fontsize=18, fontweight='bold', y=0.98)

# Colors matching standard ML metric visual styles (Blue, Orange, Green, Red)
c_prec = '#1f77b4'
c_rec  = '#ff7f0e'
c_f1   = '#2ca02c'
c_acc  = '#d62728'

def plot_four_bars(ax, prec, rec, f1, acc, title):
    # Plotting the 4 bars per year group
    ax.bar(x - 1.5*width, prec, width, label='Precision', color=c_prec, edgecolor='white')
    ax.bar(x - 0.5*width, rec,  width, label='Recall',    color=c_rec,  edgecolor='white')
    ax.bar(x + 0.5*width, f1,   width, label='F1 Score',  color=c_f1,   edgecolor='white')
    ax.bar(x + 1.5*width, acc,  width, label='Accuracy',  color=c_acc,  edgecolor='white')

    # Formatting axes
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Score (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Clean gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)
    
    # Despine for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Plot the three separate graphs (Title for axes[0] updated)
plot_four_bars(axes[0], safe_p, safe_r, safe_f1, safe_acc, "1. Max Clinical Safety")
plot_four_bars(axes[1], bal_p, bal_r, bal_f1, bal_acc, "2. 50-50 Balanced Cost")
plot_four_bars(axes[2], maxod_p, maxod_r, maxod_f1, maxod_acc, "3. Max OD Detection")

# Add a single master legend to the right of the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=12, frameon=True)

plt.tight_layout(rect=[0, 0.02, 0.92, 0.92]) # Leave room on the right for the master legend
plt.show()
