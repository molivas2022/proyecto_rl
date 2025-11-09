import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def get_best_checkpoint(csv_path, freq):
    df = pd.read_csv(csv_path)
    filtered_df = df[df["training_iteration"] % freq == 0]
    max_reward_idx = filtered_df["reward_mean"].idxmax()
    epoch = filtered_df.loc[max_reward_idx]["training_iteration"]
    reward = filtered_df.loc[max_reward_idx]["reward_mean"]
    return epoch, reward


# TODO: Generalizar dos funciones de ploteo
def plot_reward_curve(csv_path, out_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x="training_iteration", y="reward_mean", lw=2)
    ax.set_xlim(left=1)
    ax.set_title("Recompensas de Entrenamiento", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel("Recompensa promedio por episodio", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)


def plot_policy_stat(csv_path, out_path, var, policy):
    df = pd.read_csv(csv_path)
    df = df[df["policy"] == policy]
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x="training_iteration", y=var, lw=2)
    ax.set_xlim(left=1)
    ax.set_title("", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel(var, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)


def plot_route_completion_curve(csv_path, out_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df, x="Iteration", y="Average route completion", marker="o", lw=2
    )
    ax.set_xlim(left=20)
    ax.set_title("Completación de Ruta Promedio por Iteración", fontsize=16)
    ax.set_xlabel("Iteración de Entrenamiento (Epoch)", fontsize=12)
    ax.set_ylabel("Completación de ruta promedio", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(ax.figure)


if __name__ == "__main__":
    current_dir = Path.cwd()
    exp_dir = current_dir / "experimentos" / "exp5"
    rewards_csv_path = exp_dir / "rewards_log.csv"
    policy_logs = exp_dir / "policy_log.csv"
    route_completion_csv_path = exp_dir / "checkpoint_metrics.csv"

    plot_reward_curve(rewards_csv_path, exp_dir / "resultados.png")

    for i in range(5):
        policy = f"policy_{i}"
        policy_plots = exp_dir / "policy_plots" / policy
        plot_policy_stat(
            policy_logs,
            policy_plots / f"resultados_policy_loss_{policy}.png",
            "policy_loss",
            policy,
        )
        plot_policy_stat(
            policy_logs,
            policy_plots / f"resultados_vf_loss_{policy}.png",
            "vf_loss",
            policy,
        )
        plot_policy_stat(
            policy_logs,
            policy_plots / f"resultados_total_loss_{policy}.png",
            "total_loss",
            policy,
        )
        plot_policy_stat(
            policy_logs,
            policy_plots / f"resultados_mean_kl_loss_{policy}.png",
            "mean_kl_loss",
            policy,
        )
    plot_route_completion_curve(
        route_completion_csv_path,
        exp_dir / "route_completion_checkpoints_with_traffic.png",
    )
    best_checkpoint, best_reward = get_best_checkpoint(rewards_csv_path, 20)
    print(f"best reward at checkpoint {best_checkpoint} with reward {best_reward}")
