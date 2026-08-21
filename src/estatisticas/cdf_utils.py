import numpy as np
import matplotlib.pyplot as plt
import os  # Adicionado para ser coerente nos scripts!

def plot_cdf_3d_over_time(df, value_col, window_col, save_path, title="CDF 3D"):
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # S├│ para garantir compatibilidade
    unique_windows = sorted(df[window_col].unique())
    values_all = []
    x_vals = None

    # Garantia de pelo menos um valor para plotar, evita Z vazio
    has_data = False

    for win in unique_windows:
        vals = df[df[window_col] == win][value_col].dropna().values
        if len(vals) == 0:
            continue  # Evita CDFs vazios
        if x_vals is None:
            x_vals = np.linspace(np.nanmin(df[value_col]), np.nanmax(df[value_col]), 100)
        cdf = np.array([np.mean(vals <= x) for x in x_vals])
        values_all.append(cdf)
        has_data = True

    if not has_data or x_vals is None or len(values_all) == 0:
        print(f"Nenhum dado dispon├¡vel para CDF 3D de {value_col}.")
        return

    values_all = np.array(values_all)
    X, Y = np.meshgrid(x_vals, unique_windows[:len(values_all)])  # Ajusta janela em caso de skips

    Z = values_all

    fig = plt.figure(figsize=(22, 14), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax.set_xlabel(value_col)
    ax.set_ylabel('Janela')
    ax.set_zlabel('CDF')
    ax.set_title(title)
    fig.colorbar(surf)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
