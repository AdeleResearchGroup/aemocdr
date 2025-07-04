import pandas as pd
import matplotlib.pyplot as plt

# Données brutes
data = {
    "nbr of TCGA samples used for fine-tuning": [0, 10, 20, 50, 100, 200, 400, 500, 600, 700],
    "Accuracy": [0.504, 0.536, 0.507, 0.484, 0.526, 0.664, 0.706, 0.709, 0.708, 0.759],
    "Precision": [0.432, 0.526, 0.506, 0.481, 0.561, 0.637, 0.661, 0.673, 0.709, 0.793],
    "Recall": [0.505, 0.728, 0.591, 0.395, 0.212, 0.757, 0.841, 0.829, 0.753, 0.767],
    "F1 score": [0.466, 0.611, 0.545, 0.434, 0.308, 0.692, 0.740, 0.743, 0.731, 0.780],
    "AUC": [0.560, 0.532, 0.533, 0.543, 0.672, 0.699, 0.762, 0.785, 0.830, 0.849],
    "AUPRC": [0.580, 0.523, 0.530, 0.549, 0.603, 0.659, 0.713, 0.765, 0.856, 0.889]
}

# Conversion en DataFrame
df = pd.DataFrame(data)

# Création du graphique
plt.figure(figsize=(10, 6))
metrics = ["Accuracy", "F1 score", "AUC", "AUPRC"]

for metric in metrics:
    plt.plot(df["nbr of TCGA samples used for fine-tuning"], df[metric], marker='o', label=metric)

plt.title("Performance vs Number of TCGA samples used for fine-tuning")
plt.xlabel("Number of TCGA samples used for fine-tuning")
plt.ylabel("Metric value")
plt.grid(True)
plt.legend()
plt.ylim(0.4, 1)
plt.tight_layout()
plt.show()