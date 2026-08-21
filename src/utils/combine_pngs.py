import matplotlib.pyplot as plt
import math

def combinar_imagens_em_mosaico(figs, output_path, nomes=None, n_cols=5):
    n_imgs = len(figs)
    n_rows = math.ceil(n_imgs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for idx, fig_sub in enumerate(figs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        # Desenha a figura no subplot
        for artist in fig_sub.get_children():
            try:
                artist.draw(ax.figure.canvas.get_renderer())
            except Exception:
                pass
        if nomes:
            ax.set_title(nomes[idx], fontsize=8)
        ax.axis('off')
    for idx in range(len(figs), n_rows*n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Mosaico salvo: {output_path}")