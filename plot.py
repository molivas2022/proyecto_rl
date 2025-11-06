import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def get_best_checkpoint(csv_path, freq):
    df = pd.read_csv(csv_path)
    filtered_df = df[df["Iteration"] % freq == 0]
    max_reward_idx = filtered_df["Avg reward per episode"].idxmax()
    epoch = filtered_df.loc[max_reward_idx]["Iteration"]
    reward = filtered_df.loc[max_reward_idx]["Avg reward per episode"]
    return epoch, reward


# TODO: Generalizar dos funciones de ploteo
def plot_reward_curve(csv_path, out_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x="Iteration", y="Avg reward per episode", lw=2)
    ax.set_xlim(left=1)
    ax.set_title("Recompensas de Entrenamiento", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel("Recompensa promedio por episodio", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)


def plot_route_completion_curve(csv_path, out_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df, x="Iteration", y="Average route completion", marker="o", lw=2
    )
    ax.set_xlim(left=1)
    ax.set_title("Completación de Ruta Promedio por Iteración", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel("Completación de ruta promedio", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    current_dir = Path.cwd()
    exp_dir = current_dir / "experimentos" / "exp2"
    rewards_csv_path = exp_dir / "rewards_log.csv"
    route_completion_csv_path = exp_dir / "checkpoint_metrics.csv"

    plot_reward_curve(rewards_csv_path, exp_dir / "resultados.png")
    plot_route_completion_curve(
        route_completion_csv_path, exp_dir / "route_completion_checkpoints.png"
    )
    best_checkpoint, best_reward = get_best_checkpoint(rewards_csv_path, 50)

    print(f"best reward at checkpoint {best_checkpoint} with reward {best_reward}")
