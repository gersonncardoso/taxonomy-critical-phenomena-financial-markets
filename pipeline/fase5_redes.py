"""
FASE 5: Construção de Redes com controle por arquivo .done (sentinela)
"""

import os
import json
from pathlib import Path

from pipeline._runner import run_python


NETWORKS_DIR = Path('data/networks')
PROCESSED_DIR = Path('data/processed')
REQUIRED_METRICS = [
    # Pipeline planar-only: apenas métricas planares são obrigatórias.
    # network_metrics_filtrado é gerado se calc_metricas_filtrado=True (opcional).
    # network_metrics_completo e network_metrics_mst NÃO são mais gerados.
    Path('data/processed/network_metrics_planar_long.csv'),
]
PHASE5_INPUTS = [
    Path('data/processed/dados_consolidados.csv'),
    Path('data/processed/ticker_context_by_window.csv'),
    Path('data/processed/correlacoes_matrizes_long.csv'),
    Path('data/correlation/pvalues_long.csv'),
    Path('data/processed/janelas_metadata.json'),
]


def _phase5_graph_count() -> int:
    # Conta grafos planares gerados (um por janela) — métrica canônica do pipeline planar-only.
    return sum(1 for _ in PROCESSED_DIR.glob('grafo_planar_janela_*.csv'))


def _expected_window_count() -> int:
    meta = PROCESSED_DIR / 'janelas_metadata.json'
    if not meta.exists():
        return 0
    try:
        obj = json.loads(meta.read_text(encoding='utf-8'))
        windows = obj.get('windows', []) if isinstance(obj, dict) else []
        return int(len(windows))
    except Exception:
        return 0


def _fase5_outputs_ok() -> bool:
    graph_count = _phase5_graph_count()
    expected = _expected_window_count()
    has_window_graphs = graph_count > 0 and (expected == 0 or graph_count >= expected)
    return has_window_graphs and all(p.exists() for p in REQUIRED_METRICS)


def _latest_mtime(paths: list[Path]) -> float:
    mtimes = []
    for path in paths:
        try:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        except Exception:
            continue
    return max(mtimes) if mtimes else 0.0


def _phase5_outputs_stale() -> bool:
    output_candidates = [
        *REQUIRED_METRICS,
        Path('data/processed/network_centralities_mst_long.csv'),
        Path('data/processed/network_centralities_planar_long.csv'),
        Path('data/processed/network_metrics_filtrado_long.csv'),
        Path('data/processed/network_centralities_filtrado_long.csv'),
        Path('pipeline/.fase5_done'),
    ]
    upstream_mtime = _latest_mtime(PHASE5_INPUTS)
    output_mtime = _latest_mtime(output_candidates)
    return upstream_mtime > 0 and output_mtime > 0 and upstream_mtime > output_mtime


def _phase5_checkpoint_count() -> int:
    temp_dir = PROCESSED_DIR / '_temp_network_metrics'
    if not temp_dir.exists():
        return 0
    return sum(1 for _ in temp_dir.glob('done_*.flag'))


def _clear_phase5_outputs() -> int:
    patterns = [
        'grafo_completo_janela_*.csv',
        'grafo_filtrado_pval_janela_*.csv',
        'grafo_mst_janela_*.csv',
        'grafo_planar_janela_*.csv',
        'network_metrics_*_long.csv',
        'network_centralities_*_long.csv',
    ]
    files_to_remove: set[Path] = set()
    for pattern in patterns:
        files_to_remove.update(PROCESSED_DIR.glob(pattern))
    files_to_remove.update(NETWORKS_DIR.glob('grafo_*_final.graphml'))
    files_to_remove.update(NETWORKS_DIR.glob('grafo_*_final.csv'))
    files_to_remove.update((PROCESSED_DIR / '_temp_network_metrics').glob('*')) if (PROCESSED_DIR / '_temp_network_metrics').exists() else set()

    removed = 0
    for path in files_to_remove:
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except Exception:
            pass

    temp_dir = PROCESSED_DIR / '_temp_network_metrics'
    if temp_dir.exists():
        for path in sorted(temp_dir.glob('*')):
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
            except Exception:
                pass

    sentinela = Path('pipeline/.fase5_done')
    try:
        sentinela.unlink(missing_ok=True)
    except Exception:
        pass

    return removed


def fase5_redes(force: bool = False):
    print(f"\n{'#'*80}\nFASE 5: REDES (grafos, métricas, comunidades)\n{'#'*80}")

    # Agora o sentinela fica em pipeline/.fase5_done
    sentinela = Path('pipeline/.fase5_done')

    protect_existing = os.getenv('FASE5_PROTECT_EXISTING', '1') == '1'
    allow_force_rebuild = os.getenv('FASE5_ALLOW_FORCE_REBUILD', '0') == '1'
    min_graphs_to_protect = int(os.getenv('FASE5_MIN_GRAPHS_TO_PROTECT', '50'))

    graph_count = _phase5_graph_count()
    outputs_ok = _fase5_outputs_ok()
    has_expensive_existing = sentinela.exists() and graph_count >= min_graphs_to_protect
    outputs_stale = _phase5_outputs_stale()

    # Regra operacional: --force precisa forçar rebuild de fato.
    if force:
        protect_existing = False
        allow_force_rebuild = True
        print("[INFO] --force ativo: Fase 5 vai reconstruir artefatos sem proteção de existentes.")
        removed = _clear_phase5_outputs()
        print(f"[INFO] Fase 5: artefatos antigos removidos={removed}.")

    if outputs_stale and not force:
        print("[WARN] Fase 5 detectou mudanças nas bases anteriores. Limpando derivados stale e reexecutando.")
        ckpt_count = _phase5_checkpoint_count()
        removed = _clear_phase5_outputs()
        print(
            "[INFO] Limpeza Fase 5 concluída: "
            f"arquivos_removidos={removed}, checkpoints_fase5_removidos={ckpt_count}."
        )
        completed = False
    else:
        completed = sentinela.exists() and outputs_ok

    if protect_existing and has_expensive_existing and (not outputs_ok) and not force and not allow_force_rebuild:
        print(
            "[WARN] Outputs da Fase 5 parcialmente ausentes, mas preservando artefatos caros existentes "
            f"(grafos_detectados={graph_count})."
        )
        print("[INFO] Mantendo fase como concluída para evitar perda de processamento longo.")
        print("[INFO] Se quiser reconstruir tudo, use FASE5_ALLOW_FORCE_REBUILD=1 e --force.")
        return

    if force or not completed:
        if sentinela.exists() and not outputs_ok:
            print("[WARN] Sentinela da Fase 5 encontrada, mas outputs estão incompletos. Reexecutando fase.")
        print("\n▶️  Construindo redes e salvando métricas ...")
        # Execute a análise e só marca sentinela se completar sem erro.
        run_python(["main/main_network_analysis.py"])
        # Cria/atualiza o arquivo sentinela
        sentinela.parent.mkdir(parents=True, exist_ok=True)
        sentinela.write_text("Fase 5 executada com sucesso.\n", encoding='utf-8')
    else:
        print("\n⏭️  5.1-5.4. Análise de Redes - JÁ CONCLUÍDO")

if __name__ == "__main__":
    fase5_redes(force=False)