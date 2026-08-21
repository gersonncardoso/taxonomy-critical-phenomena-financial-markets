"""
FASE 3: Visualizações Iniciais (heatmaps, dendrogramas, zipagem, checkpoint, SEM .png remanescente)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import os
import shutil
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from tqdm import tqdm

from src.visualization.plots import plot_dendrogram, plot_dendrogram_with_groups, plot_heatmap

def fase3_visualizacoes_iniciais(force: bool = False, salvar_cada_n=10):
    print(f"\n{'#'*80}\nFASE 3: VISUALIZAÇÕES INICIAIS (heatmaps, dendrogramas, zip unico)\n{'#'*80}")

    done_file = Path("pipeline/.fase3_done")
    if done_file.exists() and not force:
        print("⚠️ Fase 3 já concluída, pulando execução.")
        return

    matrix_file = Path('data/processed/correlacoes_matrizes_long.csv')
    png_temp_dir = Path('figures/_temp_fase3_pngs')
    png_temp_dir.mkdir(parents=True, exist_ok=True)  # Criado apenas para gerar/remover os pngs

    if not matrix_file.exists():
        print(f"❌ Arquivo de matrizes não encontrado: {matrix_file}")
        return

    df_long = pd.read_csv(matrix_file)
    if 'Window_Start' not in df_long or 'Window_End' not in df_long:
        print("❌ 'Window_Start' e/ou 'Window_End' não encontrados no arquivo!")
        return

    janela_ids = sorted(df_long["Janela_ID"].unique())
    total_janelas = len(janela_ids)
    # Mantém saída enxuta: barra tqdm já informa progresso completo.

    png_files = []

    primeiro_heatmap = None
    ultimo_heatmap = None
    primeiro_dendro = None
    ultimo_dendro = None
    primeiro_dendro_grupos = None
    ultimo_dendro_grupos = None

    for idx, janela_id in enumerate(tqdm(janela_ids, desc="Processando janelas (Fase 3)", unit="win")):
        if not (idx == 0 or idx == total_janelas-1 or idx % salvar_cada_n == 0):
            continue
        janela = df_long[df_long["Janela_ID"] == janela_id]
        win_start = pd.to_datetime(janela["Window_Start"].iloc[0]).strftime("%d-%m-%Y")
        win_end = pd.to_datetime(janela["Window_End"].iloc[0]).strftime("%d-%m-%Y")
        janela_nome = f"janela_{win_start}_ate_{win_end}"

        heatmap_path = png_temp_dir / f"heatmap_{janela_nome}.png"
        dendro_path  = png_temp_dir / f"dendrograma_{janela_nome}.png"
        dendro_groups_path = png_temp_dir / f"dendrograma_grupos_{janela_nome}.png"

        mat = janela.pivot(index="Ticker1", columns="Ticker2", values="Correlacao")
        mat_full = mat.combine_first(mat.transpose()).fillna(1)

        # Heatmap
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                plot_heatmap(mat_full, output_path=str(heatmap_path), title=f"Heatmap\n{win_start} a {win_end}")
        except Exception as e:
            print(f"❌ Erro ao salvar heatmap {heatmap_path}: {e}")

        png_files.append(heatmap_path)
        
        # Dendrograma
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                plot_dendrogram(mat_full, output_path=str(dendro_path), title=f"Dendrograma\n{win_start} a {win_end}")
        except Exception as e:
            print(f"❌ Erro ao salvar dendrograma {dendro_path}: {e}")

        png_files.append(dendro_path)

        if idx == 0 or idx == total_janelas - 1:
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    plot_dendrogram_with_groups(
                        mat_full,
                        output_path=str(dendro_groups_path),
                        title=f"Dendrograma com grupos\n{win_start} a {win_end}",
                    )
            except Exception as e:
                print(f"❌ Erro ao salvar dendrograma com grupos {dendro_groups_path}: {e}")
            else:
                png_files.append(dendro_groups_path)

        # Guarda referências explícitas para primeira e última janela
        if idx == 0:
            primeiro_heatmap = heatmap_path
            primeiro_dendro = dendro_path
            if dendro_groups_path.exists():
                primeiro_dendro_grupos = dendro_groups_path
        if idx == total_janelas - 1:
            ultimo_heatmap = heatmap_path
            ultimo_dendro = dendro_path
            if dendro_groups_path.exists():
                ultimo_dendro_grupos = dendro_groups_path

    zip_all = Path("figures/visualizacoes_fase3.zip")
    with zipfile.ZipFile(zip_all, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in png_files:
            if f.exists():
                try:
                    zipf.write(f, arcname=f.name)
                except Exception as e:
                    print(f"❌ Erro ao adicionar {f} ao zip: {e}")

    print(f"✅ PNGs zipados em: {zip_all}")

    # Copia heatmaps/dendrogramas da primeira e última janela para pastas permanentes
    dest_heat_dir = Path('figures/heatmaps')
    dest_dendro_dir = Path('figures/dendrogramas')
    dest_heat_dir.mkdir(parents=True, exist_ok=True)
    dest_dendro_dir.mkdir(parents=True, exist_ok=True)

    try:
        if primeiro_heatmap and primeiro_heatmap.exists():
            shutil.copy(primeiro_heatmap, dest_heat_dir / "heatmap_primeira_janela.png")
        if ultimo_heatmap and ultimo_heatmap.exists():
            shutil.copy(ultimo_heatmap, dest_heat_dir / "heatmap_ultima_janela.png")
        if primeiro_dendro and primeiro_dendro.exists():
            shutil.copy(primeiro_dendro, dest_dendro_dir / "dendrograma_primeira_janela.png")
        if ultimo_dendro and ultimo_dendro.exists():
            shutil.copy(ultimo_dendro, dest_dendro_dir / "dendrograma_ultima_janela.png")
        if primeiro_dendro_grupos and primeiro_dendro_grupos.exists():
            shutil.copy(primeiro_dendro_grupos, dest_dendro_dir / "dendrograma_grupos_primeira_janela.png")
        if ultimo_dendro_grupos and ultimo_dendro_grupos.exists():
            shutil.copy(ultimo_dendro_grupos, dest_dendro_dir / "dendrograma_grupos_ultima_janela.png")
        print("✅ Heatmaps/Dendrogramas da primeira e última janela salvos em figures/heatmaps e figures/dendrogramas")
    except Exception as e:
        print(f"❌ Erro ao copiar heatmaps/dendrogramas finais: {e}")

    # Apaga todos os PNGs temporários (garante que não há png sobrando no diretório temporário)
    for f in png_files:
        try:
            os.remove(f)
        except Exception:
            pass  # Silencia erros ao remover arquivos temporários
    try:
        png_temp_dir.rmdir()
    except Exception:
        pass  # Caso não esteja vazio, ignore

    print(f"\nVisualizações zipadas em: {zip_all.resolve()} (apenas primeiras/últimas janelas são mantidas em figures/)")

    # Marca a fase como concluída
    done_file.touch()

if __name__ == "__main__":
    fase3_visualizacoes_iniciais(force=False, salvar_cada_n=10)