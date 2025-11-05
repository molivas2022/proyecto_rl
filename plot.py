import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_reward_curve(rewards, out_path):
    df = pd.DataFrame(
        {"Iteración": range(len(rewards)), "Recompensa promedio por episodio": rewards}
    )
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df, x="Iteración", y="Recompensa promedio por episodio", marker="o", lw=2
    )
    ax.set_xlim(left=0)
    ax.set_title("Recompensas de Entrenamiento", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel("Recompensa promedio por episodio", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)
