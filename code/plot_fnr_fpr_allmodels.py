import matplotlib.pyplot as plt
import numpy as np

# FPR values (common x-axis for all curves)
fpr = np.array([0.000, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.500])

# Data extracted from LaTeX table
models = {
    "LUMOS-Qwen 2.5-7B (Logistic Regression)": [0.045, 0.033, 0.020, 0.019, 0.014, 0.013, 0.012, 0.007],
    "LUMOS-Qwen 2.5-7B (Random Forest)":     [0.052, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, 0.007],
    "LUMOS-Qwen 2.5-7B (XGBoost)":    [0.041, 0.018, 0.017, 0.015, 0.013, 0.013, 0.012, 0.007],
    
    "LUMOS-Llama 3.1-8B (Logistic Regression)": [0.121, 0.093, 0.061, 0.048, 0.032, 0.026, 0.0149, 0.008],
    "LUMOS-Llama 3.1-8B (Random Forest)":     [0.038, 0.026, 0.022, 0.019, 0.017, 0.015, 0.015, 0.008],
    "LUMOS-Llama 3.1-8B (XGBoost)":    [0.043, 0.042, 0.030, 0.027, 0.022, 0.017, 0.015, 0.008],
    
    "LUMOS-Ministral-8B (Logistic Regression)": [0.227, 0.162, 0.071, 0.050, 0.033, 0.019, 0.015, 0.008],
    "LUMOS-Ministral-8B (Random Forest)":     [0.033, 0.028, 0.026, 0.024, 0.022, 0.015, 0.014, 0.007],
    "LUMOS-Ministral-8B (XGBoost)":    [0.039, 0.032, 0.028, 0.025, 0.022, 0.015, 0.015, 0.008],
}

# Plot
plt.figure(figsize=(10, 6))
for model, fnr in models.items():
    plt.plot(fpr, fnr, marker='o', linewidth=3.5, label=model)

# Labels and title
plt.xlabel("False Positive Rate (FPR)",fontsize=22)
plt.ylabel("False Negative Rate (FNR)",fontsize=22)
plt.title("FNR vs FPR for LUMOS Models and Classifiers",fontsize=22)
plt.xscale("log")  # since FPR varies exponentially
#plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
#plt.legend(fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.7)
plt.legend(fontsize=18, loc="best")
plt.tight_layout()

plt.show()


