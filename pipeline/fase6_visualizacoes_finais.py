"""
FASE 6: Visualizações Finais das Métricas (evolução, CDF, modularidade)
"""

from pathlib import Path
import sys

from pipeline._runner import run_python

# Garante que src/ seja importável mesmo quando rodado via pipeline/
sys.path.append(str(Path(__file__).resolve().parent.parent))

def fase6_visualizacoes_finais(force: bool = False):
    print(f"\n{'#'*80}\nFASE 6: VISUALIZAÇÕES FINAIS\n{'#'*80}")

    # Arquivo sentinela para controle da fase 6 visualizações
    sentinela = Path('pipeline/.fase6_viz_done')

    figures_paths = [
        Path('figures/networks'),
        Path('figures/statistics')
    ]
    validation_csv = Path('figures/validation/crossmethod_summary.csv')
    validation_expected = [
        Path('figures/validation/silhouette_evolucao.png'),
        Path('figures/validation/bestk_evolucao.png'),
        Path('figures/validation/modularidade_evolucao.png'),
    ]
    completed = sentinela.exists() and all(
        fig_dir.exists() and len(list(fig_dir.glob('*.png'))) > 0
        for fig_dir in figures_paths
    ) and (
        (not validation_csv.exists()) or all(path.exists() for path in validation_expected)
    )

    if not completed or force:
        print("\n▶️  Gerando visualizações finais ...")
        run_python(["main/main_plot_network_metrics.py"])
        run_python(["main/main_plot_network_centralities.py"])
        run_python(["main/main_plot_network_cdf.py"])
        run_python(["main/main_plot_modularidade_temporal.py"])
        run_python(["main/main_plot_cluster_validation_metrics.py"])
        # Cria/atualiza o .done
        sentinela.parent.mkdir(parents=True, exist_ok=True)
        sentinela.write_text("Fase 6 (visualizações finais) executada com sucesso.\n")
        print("\n✅ Visualizações finais geradas nas pastas figures/networks/, figures/statistics/ e figures/validation/")
    else:
        print("\n⏭️  6. Visualizações de Redes - JÁ CONCLUÍDO")

if __name__ == "__main__":
    fase6_visualizacoes_finais(force=False)