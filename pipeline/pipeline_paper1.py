"""Run the reproducible Paper 1 B3 rolling-network pipeline only.

The runner deliberately stops at phase 6. It does not import or execute the
GDELT, news-sentiment, MRQAP, or Paper 2 stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.fase1_dados import fase1_dados
from pipeline.fase2_correlacoes import fase2_correlacoes
from pipeline.fase3_visualizacoes_iniciais import fase3_visualizacoes_iniciais
from pipeline.fase4_validacao import fase4_validacao
from pipeline.fase5_redes import fase5_redes
from pipeline.fase6_visualizacoes_finais import fase6_visualizacoes_finais


ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(force: bool = False) -> None:
    """Execute the six stages required to reproduce Paper 1 outputs."""
    for stage in (
        fase1_dados,
        fase2_correlacoes,
        fase3_visualizacoes_iniciais,
        fase4_validacao,
        fase5_redes,
        fase6_visualizacoes_finais,
    ):
        stage(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Paper 1 B3 pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute existing intermediate outputs",
    )
    args = parser.parse_args()
    run_pipeline(force=args.force)
