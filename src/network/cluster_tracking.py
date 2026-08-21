import matplotlib.pyplot as plt
import os

def plot_modularity_and_clusters(df, modularity_col, n_clusters_col, window_col, net_type, save_dir):
    plt.figure(figsize=(10,6))
    plt.plot(df[window_col], df[modularity_col], label="Modularidade", color='b')
    plt.xlabel("Janela")
    plt.ylabel("Modularidade")
    plt.title(f"Modularidade ao longo do tempo - {net_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{net_type}_modularidade.png'))
    plt.close()
    plt.figure(figsize=(10,6))
    plt.plot(df[window_col], df[n_clusters_col], label="N° Grupos", color='r')
    plt.xlabel("Janela")
    plt.ylabel("N° de Grupos")
    plt.title(f"Número de Grupos (Clusters) - {net_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{net_type}_num_clusters.png'))
    plt.close()