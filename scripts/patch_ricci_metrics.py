"""
patch_ricci_metrics.py
======================
Patch cirúrgico: lê os grafo_planar_janela_N.csv já gerados e adiciona as
colunas de curvatura Forman-Ricci ao network_metrics_planar_long.csv existente.

Não reprocessa planarity/MST/etc. — apenas lê edge-lists e computa Ricci.

Uso:
    python patch_ricci_metrics.py
    python patch_ricci_metrics.py --dry-run
"""

import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.network.metrics import calcular_forman_ricci_stats

PROCESSED = ROOT / "data" / "processed"
METRICS_CSV = PROCESSED / "network_metrics_planar_long.csv"

RICCI_COLS = [
    "ricci_fr_mean", "ricci_fr_var", "ricci_fr_std", "ricci_fr_skew",
    "ricci_fr_kurt", "ricci_fr_q10", "ricci_fr_q50", "ricci_fr_q90",
    "ricci_fr_iqr", "ricci_fr_min", "ricci_fr_max",
]
OBSOLETE_RICCI_COLS = ["ricci_fr_neg_share"]


def load_planar_graph(window_id: int) -> nx.Graph | None:
    path = PROCESSED / f"grafo_planar_janela_{window_id}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return nx.Graph()
        G = nx.from_pandas_edgelist(
            df, source="source", target="target",
            edge_attr=["weight", "distance"] if "distance" in df.columns else ["weight"],
        )
        return G
    except Exception as e:
        print(f"  [WARN] Janela {window_id}: erro ao carregar grafo — {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Adiciona colunas Forman-Ricci ao network_metrics_planar_long.csv")
    parser.add_argument("--dry-run", action="store_true", help="Apenas simula, não salva")
    args = parser.parse_args()

    if not METRICS_CSV.exists():
        print(f"[ERRO] {METRICS_CSV} não encontrado. Execute fase5 primeiro.")
        sys.exit(1)

    df = pd.read_csv(METRICS_CSV)
    print(f"Carregado: {METRICS_CSV.name} — {len(df)} linhas")

    # Verificar se Ricci já existe
    already_ricci = [c for c in RICCI_COLS if c in df.columns]
    if already_ricci:
        non_null = df[already_ricci[0]].notna().sum()
        print(f"[INFO] Colunas Ricci já presentes ({non_null} não-nulas). Patch redundante mas harmless.")

    missing_ricci = [c for c in RICCI_COLS if c not in df.columns]
    if missing_ricci:
        print(f"[INFO] Colunas faltantes: {missing_ricci}")
    else:
        print("[INFO] Todas as colunas Ricci já existem. Verificando se têm valores...")

    window_ids = sorted(df["Janela_ID"].unique())
    print(f"Processando {len(window_ids)} janelas...")

    ricci_rows = []
    errors = 0

    for wid in tqdm(window_ids, desc="Ricci patch", unit="jan"):
        G = load_planar_graph(int(wid))
        if G is None:
            # Arquivo não existe — preenche com NaN
            row = {"Janela_ID": int(wid)}
            row.update({c: np.nan for c in RICCI_COLS})
        else:
            try:
                stats = calcular_forman_ricci_stats(G)
                row = {"Janela_ID": int(wid)}
                row.update(stats)
            except Exception as e:
                print(f"  [WARN] Janela {int(wid)}: erro Ricci — {e}")
                row = {"Janela_ID": int(wid)}
                row.update({c: np.nan for c in RICCI_COLS})
                errors += 1
        ricci_rows.append(row)

    df_ricci = pd.DataFrame(ricci_rows)

    # Remove colunas Ricci atuais e métricas obsoletas antes do merge.
    for c in RICCI_COLS + OBSOLETE_RICCI_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Merge por Janela_ID
    df_out = df.merge(df_ricci, on="Janela_ID", how="left")

    # Verificação rápida
    non_null_count = df_out["ricci_fr_mean"].notna().sum()
    print(f"\nResultado: {non_null_count}/{len(df_out)} janelas com Ricci calculado.")
    if errors:
        print(f"[WARN] {errors} janelas com erro no cálculo de Ricci (NaN preenchido).")

    # Amostra
    sample_wins = [1, 90, 180, 270, 360]
    print("\nAmostra janelas chave:")
    cols_show = ["Janela_ID", "n_edges", "ricci_fr_mean", "ricci_fr_std", "ricci_fr_iqr"]
    print(df_out[df_out["Janela_ID"].isin(sample_wins)][cols_show].to_string(index=False))

    if args.dry_run:
        print("\n[DRY-RUN] Nenhum arquivo salvo.")
        return

    # Backup do original
    backup = METRICS_CSV.with_suffix(".csv.bak")
    import shutil
    shutil.copy(METRICS_CSV, backup)
    print(f"\nBackup salvo em: {backup.name}")

    # Salva CSV atualizado
    df_out.to_csv(METRICS_CSV, index=False)
    print(f"✅ {METRICS_CSV.name} atualizado com colunas Ricci.")
    print(f"   Colunas novas: {RICCI_COLS}")


if __name__ == "__main__":
    main()
