import glob
from pathlib import Path

def gerar_blocos_figuras_latex(fig_dir, section_title="Automated Visualizations"):
    """
    Gera blocos de figuras LaTeX automaticamente a partir dos PNGs/JPGs no diretório.
    """
    fig_dir = Path(fig_dir)
    figuras = sorted(fig_dir.glob("*.png")) + sorted(fig_dir.glob("*.jpg"))
    blocos = []
    if figuras:
        blocos.append(f"\\section*{{{section_title}}}\n")
    for i, fpath in enumerate(figuras, 1):
        caption = f"AUTO: {fpath.stem.replace('_', ' ').title()}"
        blocos.append(
            fr"""
\begin{{figure}}[h]
\centering
\includegraphics[width=0.9\textwidth]{{{fpath.as_posix()}}}
\caption{{{caption}}}
\label{{fig:auto-{i}}}
\end{{figure}}
            """)
    return "\n".join(blocos)