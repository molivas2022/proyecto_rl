import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_reward_curve(csv_path, out_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df, x="Iteration", y="Avg reward per episode", marker="o", lw=2
    )
    ax.set_xlim(left=1)
    ax.set_title("Recompensas de Entrenamiento", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel("Recompensa promedio por episodio", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)
