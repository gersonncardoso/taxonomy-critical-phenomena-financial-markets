import warnings
import sys
import argparse
from pathlib import Path

# Garante que o diretório raiz do projeto está no sys.path
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import shutil
import csv
import subprocess
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import zipfile
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from src.estatisticas.correlation_pvalues import calcular_pvalues_from_corr
from src.estatisticas.ks_test import calcular_ks
from src.estatisticas.adf_test import calcular_adf
from src.estatisticas.marchenko_pastur import compute_mp_stats
from src.correlation.clustering import run_multiple_clustering_methods
from src.utils.gpu_utils import GPU_AVAILABLE, cp, to_numpy

warnings.filterwarnings("ignore")

PVALUES_COLUMNS = [
    "Janela_ID", "Window_Start", "Window_End",
    "Ticker1", "Ticker2", "Correlacao", "pvalue"
]
KS_COLUMNS = ["Janela_ID", "Window_Start", "Window_End", "ks_stat", "ks_pvalue"]
MP_COLUMNS = ["Janela_ID", "Window_Start", "Window_End",
              "n_assets", "n_obs", "q", "lambda_plus", "lambda_max_emp",
              "n_signal", "n_noise", "var_signal_frac", "market_mode_frac"]
ADF_BASE_COLUMNS = ["Janela_ID", "Window_Start", "Window_End", "Ticker", "adf_stat", "adf_pvalue", "adf_stationary"]
CROSS_COLUMNS = ["Janela_ID", "Window_Start", "Window_End", "method", "best_k", "silhouette", "modularity"]


# Diretórios padronizados para outputs
DEFAULT_OUTPUT_PROCESSED = "data/processed"
DEFAULT_OUTPUT_FIGURES = "figures"

# Defaults alinhados ao runtime Spark da Fase 7 (src/utils/spark_runtime.py).
# O objetivo aqui é manter diagnóstico, reparticionamento interno e escrita
# coerentes com o mesmo envelope de memória usado pelo runtime funcional.
DEFAULT_FASE4_SPARK_MASTER = "local[*]"
DEFAULT_FASE4_SPARK_DRIVER_MEM = "8g"
DEFAULT_FASE4_SPARK_EXECUTOR_MEM = "4g"
DEFAULT_FASE4_SPARK_OFFHEAP_SIZE = "1g"
DEFAULT_FASE4_SPARK_MAX_RESULT_SIZE = "4g"
DEFAULT_FASE4_SPARK_MAX_PARTITION_BYTES = "67108864"  # 64 MB, igual ao runtime da fase 7
DEFAULT_FASE4_SPARK_SHUFFLE_PARTITIONS = 400
DEFAULT_FASE4_SPARK_DEFAULT_PARALLELISM = 64
DEFAULT_FASE4_PVALUES_REPARTITIONS = 400
DEFAULT_FASE4_MAX_RECORDS_PER_FILE = 150000  # Reduzido para mais arquivos menores
DEFAULT_FASE4_MAX_RECORDS_PER_FILE_FALLBACK = 75000  # Fallback menor

FASE4_USE_OS_MAIN_ENV = "PAPER1_PHASE4_USE_OS_MAIN"
FASE4_INTERNAL_CALL_ENV = "PAPER1_INTERNAL_PHASE4_CALL"


def _get_total_system_memory_gb() -> Optional[float]:
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) / (1024.0 * 1024.0)
        except Exception:
            return None
    return None


def _apply_fase4_memory_safeguards() -> None:
    """Aplica limites conservadores para reduzir OOM em Linux/WSL.

    Pode ser desativado com FASE4_SAFE_MODE=0.
    """
    if os.name == "nt":
        return
    if os.getenv("FASE4_SAFE_MODE", "1") != "1":
        return

    total_mem_gb = _get_total_system_memory_gb()
    cpu_count = max(2, int(os.cpu_count() or 2))

    if total_mem_gb is None:
        total_mem_gb = 24.0

    if total_mem_gb <= 16:
        target_threads = min(cpu_count, 6)
        target_shuffle = 120
        target_driver = "4g"
        target_executor = "2g"
        target_max_result = "2g"
        target_max_records = "60000"
    elif total_mem_gb <= 24:
        target_threads = min(cpu_count, 10)
        target_shuffle = 200
        target_driver = "6g"
        target_executor = "3g"
        target_max_result = "3g"
        target_max_records = "90000"
    else:
        target_threads = min(cpu_count, 30)
        target_shuffle = 300
        target_driver = "8g"
        target_executor = "4g"
        target_max_result = "4g"
        target_max_records = "120000"

    def _apply_if_empty_or_default(var_name: str, target_value: str, default_values: Sequence[str]) -> None:
        raw_current = str(os.getenv(var_name, "")).strip()
        if (not raw_current) or (raw_current in set(default_values)):
            os.environ[var_name] = target_value

    _apply_if_empty_or_default(
        "SPARK_LOCAL_MASTER",
        f"local[{target_threads}]",
        [DEFAULT_FASE4_SPARK_MASTER, "local[24]", "local[30]", "local[*]"],
    )
    _apply_if_empty_or_default(
        "SPARK_SQL_SHUFFLE_PARTITIONS",
        str(target_shuffle),
        [str(DEFAULT_FASE4_SPARK_SHUFFLE_PARTITIONS), "400"],
    )
    _apply_if_empty_or_default(
        "SPARK_DEFAULT_PARALLELISM",
        str(max(8, target_shuffle // 2)),
        [str(DEFAULT_FASE4_SPARK_DEFAULT_PARALLELISM), "64"],
    )
    _apply_if_empty_or_default(
        "FASE4_PVALUES_REPARTITIONS",
        str(target_shuffle),
        [str(DEFAULT_FASE4_PVALUES_REPARTITIONS), "400"],
    )
    _apply_if_empty_or_default(
        "SPARK_DRIVER_MEMORY",
        target_driver,
        [DEFAULT_FASE4_SPARK_DRIVER_MEM, "8g"],
    )
    _apply_if_empty_or_default(
        "SPARK_EXECUTOR_MEMORY",
        target_executor,
        [DEFAULT_FASE4_SPARK_EXECUTOR_MEM, "4g"],
    )
    _apply_if_empty_or_default(
        "SPARK_DRIVER_MAX_RESULT_SIZE",
        target_max_result,
        [DEFAULT_FASE4_SPARK_MAX_RESULT_SIZE, "4g"],
    )
    _apply_if_empty_or_default("RAPIDS_CONCURRENT_TASKS", "2", ["", "1", "4"])
    _apply_if_empty_or_default(
        "FASE4_MAX_RECORDS_PER_FILE",
        target_max_records,
        [str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE), "150000"],
    )
    _apply_if_empty_or_default(
        "FASE4_MAX_RECORDS_PER_FILE_FALLBACK",
        str(min(50000, max(25000, int(target_max_records) // 2))),
        [str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE_FALLBACK), "75000"],
    )
    _apply_if_empty_or_default("FASE4_STORAGE_LEVEL", "DISK_ONLY", ["", "MEMORY_AND_DISK"])

    force_rapids = os.getenv("FASE4_FORCE_RAPIDS", "0") == "1"
    if (not force_rapids) and total_mem_gb <= 24:
        # Em máquinas até 24GB, forçar CPU reduz chance de kill por pressão conjunta JVM+GPU.
        os.environ["ENABLE_RAPIDS"] = "0"

    print(
        "[INFO] Fase 4 safe mode ativo: "
        f"RAM~{int(total_mem_gb)}GB, master={os.getenv('SPARK_LOCAL_MASTER')}, "
        f"shuffle={os.getenv('SPARK_SQL_SHUFFLE_PARTITIONS')}, "
        f"driver={os.getenv('SPARK_DRIVER_MEMORY')}, executor={os.getenv('SPARK_EXECUTOR_MEMORY')}, "
        f"maxResult={os.getenv('SPARK_DRIVER_MAX_RESULT_SIZE')}, "
        f"storage={os.getenv('FASE4_STORAGE_LEVEL')}, rapids={os.getenv('ENABLE_RAPIDS')}, "
        f"rapids_tasks={os.getenv('RAPIDS_CONCURRENT_TASKS')}"
    )


def _apply_emergency_cpu_profile() -> None:
    """Perfil de emergência para retry após falha Spark em Linux/WSL."""
    if os.name == "nt":
        return

    os.environ["ENABLE_RAPIDS"] = "0"
    os.environ["SPARK_LOCAL_MASTER"] = os.getenv("FASE4_EMERGENCY_MASTER", "local[4]")
    os.environ["SPARK_SQL_SHUFFLE_PARTITIONS"] = os.getenv("FASE4_EMERGENCY_SHUFFLE", "96")
    os.environ["SPARK_DEFAULT_PARALLELISM"] = os.getenv("FASE4_EMERGENCY_PARALLELISM", "48")
    os.environ["SPARK_DRIVER_MEMORY"] = os.getenv("FASE4_EMERGENCY_DRIVER_MEM", "4g")
    os.environ["SPARK_EXECUTOR_MEMORY"] = os.getenv("FASE4_EMERGENCY_EXECUTOR_MEM", "2g")
    os.environ["SPARK_DRIVER_MAX_RESULT_SIZE"] = os.getenv("FASE4_EMERGENCY_MAX_RESULT", "2g")
    os.environ["FASE4_PVALUES_REPARTITIONS"] = os.getenv("FASE4_EMERGENCY_PVALUES_REPARTITIONS", "96")
    os.environ["FASE4_MAX_RECORDS_PER_FILE"] = os.getenv("FASE4_EMERGENCY_MAX_RECORDS", "50000")
    os.environ["FASE4_MAX_RECORDS_PER_FILE_FALLBACK"] = os.getenv("FASE4_EMERGENCY_MAX_RECORDS_FALLBACK", "25000")
    os.environ["FASE4_STORAGE_LEVEL"] = "DISK_ONLY"

    print(
        "[INFO] Fase 4 emergency profile ativo: "
        f"master={os.getenv('SPARK_LOCAL_MASTER')}, "
        f"shuffle={os.getenv('SPARK_SQL_SHUFFLE_PARTITIONS')}, "
        f"driver={os.getenv('SPARK_DRIVER_MEMORY')}, executor={os.getenv('SPARK_EXECUTOR_MEMORY')}, "
        f"maxResult={os.getenv('SPARK_DRIVER_MAX_RESULT_SIZE')}, "
        f"storage={os.getenv('FASE4_STORAGE_LEVEL')}, rapids={os.getenv('ENABLE_RAPIDS')}"
    )


def _apply_fase4_runtime_defaults() -> None:
    """Alinha a Fase 4 com as mesmas variáveis canônicas usadas na Fase 7."""
    project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("SPARK_LOG_LEVEL", "WARN")
    os.environ.setdefault("SPARK_DRIVER_MEMORY", DEFAULT_FASE4_SPARK_DRIVER_MEM)
    os.environ.setdefault("SPARK_EXECUTOR_MEMORY", DEFAULT_FASE4_SPARK_EXECUTOR_MEM)
    os.environ.setdefault("SPARK_DRIVER_MEMORY_OVERHEAD", DEFAULT_FASE4_SPARK_OFFHEAP_SIZE)
    os.environ.setdefault("SPARK_DRIVER_MAX_RESULT_SIZE", DEFAULT_FASE4_SPARK_MAX_RESULT_SIZE)
    os.environ.setdefault("SPARK_SQL_SHUFFLE_PARTITIONS", str(DEFAULT_FASE4_SPARK_SHUFFLE_PARTITIONS))
    os.environ.setdefault("SPARK_DEFAULT_PARALLELISM", str(DEFAULT_FASE4_SPARK_DEFAULT_PARALLELISM))
    os.environ.setdefault("SPARK_SQL_FILES_MAX_PARTITION_BYTES", DEFAULT_FASE4_SPARK_MAX_PARTITION_BYTES)

    legacy_to_runtime = {
        "FASE4_SPARK_MASTER": "SPARK_LOCAL_MASTER",
        "FASE4_SPARK_DRIVER_MEM": "SPARK_DRIVER_MEMORY",
        "FASE4_SPARK_EXECUTOR_MEM": "SPARK_EXECUTOR_MEMORY",
        "FASE4_SPARK_OFFHEAP_SIZE": "SPARK_DRIVER_MEMORY_OVERHEAD",
        "FASE4_SPARK_SHUFFLE_PARTITIONS": "SPARK_SQL_SHUFFLE_PARTITIONS",
        "FASE4_SPARK_MAX_PARTITION_BYTES": "SPARK_SQL_FILES_MAX_PARTITION_BYTES",
    }
    for legacy_name, runtime_name in legacy_to_runtime.items():
        legacy_value = os.getenv(legacy_name)
        if legacy_value and not os.getenv(runtime_name):
            os.environ[runtime_name] = legacy_value

    if os.name == "nt":
        os.environ.setdefault("ENABLE_RAPIDS", "0")
        for python_env_var in ("PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
            current_value = str(os.getenv(python_env_var, "")).strip().strip('"')
            current_path = Path(current_value.replace("/", "\\")) if current_value else None
            invalid_windows_path = (
                not current_value
                or current_value.startswith("/usr/")
                or current_value.startswith("/bin/")
                or current_value.startswith("/mnt/")
                or current_path is None
                or not current_path.exists()
            )
            if invalid_windows_path:
                if current_value:
                    print(
                        f"[WARN] {python_env_var} inválido para Windows: {current_value!r}. "
                        f"Usando {sys.executable}."
                    )
                os.environ[python_env_var] = sys.executable
    else:
        wsl_venv_python = project_root / "venv" / "bin" / "python"
        if wsl_venv_python.exists():
            os.environ.setdefault("PYSPARK_PYTHON", str(wsl_venv_python))
            os.environ.setdefault("PYSPARK_DRIVER_PYTHON", str(wsl_venv_python))
        os.environ.setdefault("ENABLE_RAPIDS", "1")
        os.environ.setdefault("SPARK_LOCAL_MASTER", "local[30]")
        os.environ.setdefault("RAPIDS_CONCURRENT_TASKS", "2")
        os.environ.setdefault("SPARK_RAPIDS_SQL_CONCURRENTGPUTASKS", "2")
        os.environ.setdefault("SPARK_DRIVER_HOST", "127.0.0.1")
        os.environ.setdefault("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1")
        os.environ.setdefault("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "120s")
        os.environ.setdefault("SPARK_NETWORK_TIMEOUT", "600s")
        os.environ.setdefault("SPARK_RPC_ASK_TIMEOUT", "300s")
        os.environ.setdefault("SPARK_RPC_LOOKUP_TIMEOUT", "300s")

    _apply_fase4_memory_safeguards()


def _get_fase4_runtime_settings() -> Dict[str, str]:
    return {
        "master": os.getenv("SPARK_LOCAL_MASTER", DEFAULT_FASE4_SPARK_MASTER),
        "driver_memory": os.getenv("SPARK_DRIVER_MEMORY", DEFAULT_FASE4_SPARK_DRIVER_MEM),
        "executor_memory": os.getenv("SPARK_EXECUTOR_MEMORY", DEFAULT_FASE4_SPARK_EXECUTOR_MEM),
        "driver_memory_overhead": os.getenv("SPARK_DRIVER_MEMORY_OVERHEAD", DEFAULT_FASE4_SPARK_OFFHEAP_SIZE),
        "driver_max_result_size": os.getenv("SPARK_DRIVER_MAX_RESULT_SIZE", DEFAULT_FASE4_SPARK_MAX_RESULT_SIZE),
        "shuffle_partitions": os.getenv(
            "FASE4_SPARK_SHUFFLE_PARTITIONS",
            os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", str(DEFAULT_FASE4_SPARK_SHUFFLE_PARTITIONS)),
        ),
        "pvalues_repartitions": os.getenv("FASE4_PVALUES_REPARTITIONS", str(DEFAULT_FASE4_PVALUES_REPARTITIONS)),
        "max_records_per_file": os.getenv("FASE4_MAX_RECORDS_PER_FILE", str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE)),
        "rapids_enabled": os.getenv("ENABLE_RAPIDS", "0"),
    }


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = resolved.as_posix()[2:]
    return f"/mnt/{drive}{suffix}"


def _detect_wsl_gpu_runtime(project_root: Path) -> Tuple[bool, str]:
    if os.name != "nt":
        return False, "SO atual não é Windows; execução WSL não é necessária."

    wsl_project_root = _windows_path_to_wsl(project_root)
    checks = [
        "command -v python >/dev/null 2>&1",
        f"test -x '{wsl_project_root}/venv/bin/python'",
        "command -v nvidia-smi >/dev/null 2>&1 || ls /dev/nvidiactl >/dev/null 2>&1",
        "ls /mnt/c/spark/jars/*rapids*spark*.jar >/dev/null 2>&1",
    ]
    cmd = " ; ".join(checks)
    try:
        result = subprocess.run(
            ["wsl", "bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        return False, "WSL não está disponível neste Windows."
    except Exception as exc:
        return False, f"Falha ao sondar WSL/GPU: {exc}"

    if result.returncode == 0:
        return True, "GPU+WSL+RAPIDS detectados; usando runtime Linux da fase 7."

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"código={result.returncode}"
    return False, f"WSL/GPU/RAPIDS indisponível: {detail}"


def _run_fase4_via_wsl(project_root: Path, force: bool) -> None:
    wsl_project_root = _windows_path_to_wsl(project_root)
    python_cmd = "python -u pipeline/fase4_validacao.py"
    if force:
        python_cmd += " --force"

    wsl_cmd = (
        f"cd '{wsl_project_root}'"
        " ; source venv/bin/activate"
        " ; export SPARK_LOCAL_IP=127.0.0.1"
        " ; export SPARK_LOCAL_HOSTNAME=localhost"
        " ; export PAPER1_INTERNAL_PHASE4_CALL=1"
        " ; export PAPER1_PHASE4_USE_OS_MAIN=0"
        " ; export ENABLE_RAPIDS=1"
        f" ; {python_cmd}"
    )

    subprocess.run(["wsl", "bash", "-lc", wsl_cmd], check=True)


def _maybe_dispatch_fase4_runtime(project_root: Path, force: bool) -> bool:
    if os.getenv(FASE4_USE_OS_MAIN_ENV, "1") != "1":
        print(f"[WARN] {FASE4_USE_OS_MAIN_ENV}=0: fase4 executará localmente.")
        return False

    if os.getenv(FASE4_INTERNAL_CALL_ENV, "0") == "1":
        print("[INFO] Fase 4 em chamada interna; sem redespacho de runtime.")
        return False

    if os.name != "nt":
        print("[INFO] Runtime Linux/WSL ativo: fase4 seguirá localmente com configuração RAPIDS/CPU da fase 7.")
        return False

    can_use_wsl_gpu, reason = _detect_wsl_gpu_runtime(project_root)
    if can_use_wsl_gpu:
        print(f"[INFO] {reason}")
        print("[INFO] Delegando fase4 para WSL (Linux + RAPIDS).")
        _run_fase4_via_wsl(project_root, force=force)
        return True

    print(f"[INFO] {reason}")
    print("[INFO] Executando fase4 localmente no Windows em CPU, como fallback da fase 7.")
    return False


def gerar_matriz_aleatoria(n_linhas, n_colunas, seed=None):
    """Gera matriz aleatória normal padrão. Usa GPU (CuPy) se disponível, senão CPU (NumPy)."""
    if GPU_AVAILABLE and cp is not None:
        # GPU: CuPy
        try:
            rng = cp.random.default_rng(seed)
            gpu_matrix = rng.normal(loc=0, scale=1, size=(n_linhas, n_colunas))
            # Converte para NumPy se necessário (geralmente retorna GPU array direto)
            return to_numpy(gpu_matrix)
        except Exception:
            # Fallback para CPU se algo der errado com GPU
            pass
    
    # CPU: NumPy
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0, scale=1, size=(n_linhas, n_colunas))


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(v))
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "sim"}:
        return True
    if s in {"0", "false", "f", "no", "n", "nao", "não"}:
        return False
    return None


def _mean_numeric(values: Iterable[Any]) -> float:
    nums: List[float] = []
    for v in values:
        fv = _safe_float(v)
        if fv is None or np.isnan(fv):
            continue
        nums.append(fv)
    return float(np.mean(nums)) if nums else np.nan


def _infer_fieldnames(rows: Sequence[Dict[str, Any]], preferred: Optional[Sequence[str]] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    if preferred:
        for c in preferred:
            if c not in seen:
                out.append(c)
                seen.add(c)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                out.append(k)
                seen.add(k)
    return out


def _write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    cols = list(fieldnames) if fieldnames else _infer_fieldnames(rows_list)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not cols:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({c: row.get(c, "") for c in cols})


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _scan_unique_int_values(path: Path, column: str) -> List[int]:
    # Usa pandas com usecols para ler apenas a coluna necessária — muito mais
    # rápido que csv.DictReader para arquivos de centenas de MB em WSL/NTFS.
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=[column], engine="c", dtype={column: "Int64"})
        ids = sorted(int(v) for v in df[column].dropna().unique())
        return ids
    except Exception:
        # Fallback para leitura linha a linha
        ids = set()
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                jid = _safe_int(row.get(column))
                if jid is not None:
                    ids.add(jid)
        return sorted(ids)


def _scan_ticker_stats(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    # Usa pandas com usecols para ler apenas a coluna Ticker — muito mais
    # rápido que csv.DictReader para arquivo de 2GB+ em WSL/NTFS.
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["Ticker"], engine="c", dtype={"Ticker": "str"})
        total_rows = len(df)
        n_tickers = df["Ticker"].nunique()
        return n_tickers, total_rows
    except Exception:
        tickers = set()
        total_rows = 0
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                t = str(row.get("Ticker", "")).strip()
                if t:
                    tickers.add(t)
        return len(tickers), total_rows


def _dedupe_keep_last(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    od: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
    for row in rows:
        k = tuple(row.get(c) for c in key_fields)
        if k in od:
            od.pop(k)
        od[k] = row
    return list(od.values())


def _escape_latex(s: Any) -> str:
    t = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in repl.items():
        t = t.replace(a, b)
    return t


def _latex_cell(v: Any, float_decimals: int = 4) -> str:
    fv = _safe_float(v)
    if fv is not None and not np.isnan(fv):
        return f"{fv:.{float_decimals}f}"
    if v is None or str(v).strip() == "":
        return "--"
    return _escape_latex(v)


def _write_simple_latex_table(
    path: Path,
    caption: str,
    label: str,
    columns: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    float_decimals: int = 4,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" * max(1, len(columns))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\hline\n")
        if columns:
            f.write(" & ".join(_escape_latex(c) for c in columns) + " \\\\\n")
            f.write("\\hline\n")
            if rows:
                for r in rows:
                    f.write(" & ".join(_latex_cell(r.get(c, ""), float_decimals) for c in columns) + " \\\\\n")
            else:
                f.write(" & ".join("--" for _ in columns) + " \\\\\n")
        else:
            f.write("-- \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _merge_chunks_to_csv(chunk_files, output_file: Path):
    """Concatena arquivos CSV chunk em streaming para output_file.

    Nunca carrega mais de um chunk em memória: ideal como fallback leve.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as out:
        wrote_header = False
        for csv_file in chunk_files:
            with open(csv_file, "r", encoding="utf-8", newline="") as inp:
                for line_num, line in enumerate(inp):
                    if line_num == 0:
                        if wrote_header:
                            continue
                        wrote_header = True
                    out.write(line)


def _write_pvalues_csv_partitioned(chunks_dir: Path, output_file: Path):
    """Lê os chunks CSV de p-values via Spark, reparticiona em pedaços pequenos
    e grava múltiplos CSV part-files; depois os une em streaming para um único
    CSV final sem carregar tudo em memória.

    Parâmetros via variáveis de ambiente:
    - FASE4_SPARK_MASTER        : compat legado para SPARK_LOCAL_MASTER
    - FASE4_PVALUES_REPARTITIONS: partições Spark de saída (padrão: 400)
    - FASE4_MAX_RECORDS_PER_FILE: linhas máximas por part-file (padrão: 150_000)
    - FASE4_SPARK_DRIVER_MEM    : compat legado para SPARK_DRIVER_MEMORY
    - FASE4_SPARK_EXECUTOR_MEM  : compat legado para SPARK_EXECUTOR_MEMORY
    - FASE4_SPARK_OFFHEAP_SIZE  : compat legado para SPARK_DRIVER_MEMORY_OVERHEAD
    - SPARK_*                   : variáveis canônicas do runtime da Fase 7

    Fallback automático se Spark falhar: merge streaming direto dos chunks
    sem nenhum shuffle.
    """
    chunk_files = sorted(chunks_dir.glob("*.csv"))
    if not chunk_files:
        _write_csv_rows(output_file, [], PVALUES_COLUMNS)
        return

    repartitions = int(os.getenv("FASE4_PVALUES_REPARTITIONS", str(DEFAULT_FASE4_PVALUES_REPARTITIONS)))
    repartitions = max(1, repartitions)
    max_records = int(os.getenv("FASE4_MAX_RECORDS_PER_FILE", str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE)))
    max_records = max(1, max_records)
    driver_mem = os.getenv("SPARK_DRIVER_MEMORY", DEFAULT_FASE4_SPARK_DRIVER_MEM)
    executor_mem = os.getenv("SPARK_EXECUTOR_MEMORY", DEFAULT_FASE4_SPARK_EXECUTOR_MEM)
    master = os.getenv("SPARK_LOCAL_MASTER", DEFAULT_FASE4_SPARK_MASTER)
    spark_tmp_dir = output_file.parent / f"{output_file.stem}_spark_parts"
    fallback_max_records = int(
        os.getenv("FASE4_MAX_RECORDS_PER_FILE_FALLBACK", str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE_FALLBACK))
    )
    fallback_max_records = max(1, fallback_max_records)

    print(f"[INFO] Fase 4 pvalues: usando Spark master={master} driver_mem={driver_mem} executor_mem={executor_mem}")
    print(f"[INFO] Fase 4 pvalues: repartitionando em {repartitions} partições com máx {max_records} registros/arquivo")

    try:
        from pyspark.sql.types import (
            StructType, StructField, IntegerType, StringType, DoubleType
        )

        schema = StructType([
            StructField("Janela_ID", IntegerType(), True),
            StructField("Window_Start", StringType(), True),
            StructField("Window_End", StringType(), True),
            StructField("Ticker1", StringType(), True),
            StructField("Ticker2", StringType(), True),
            StructField("Correlacao", DoubleType(), True),
            StructField("pvalue", DoubleType(), True),
        ])

        spark = _build_spark_session_for_fase4("Fase4PvaluesCSV")
        spark.sparkContext.setLogLevel("WARN")
        try:
            df = (
                spark.read
                .schema(schema)
                .option("header", True)
                .csv(str(chunks_dir))
            )

            write_attempts = [max_records]
            if fallback_max_records not in write_attempts:
                write_attempts.append(fallback_max_records)

            last_write_error = None
            wrote_ok = False
            for mr in write_attempts:
                try:
                    if spark_tmp_dir.exists():
                        shutil.rmtree(spark_tmp_dir, ignore_errors=True)
                    (
                        df.repartition(repartitions, "Janela_ID")
                        .write
                        .mode("overwrite")
                        .option("header", True)
                        .option("maxRecordsPerFile", str(mr))
                        .csv(str(spark_tmp_dir))
                    )
                    print(
                        f"[INFO] Fase 4: pvalues reparticionados em {spark_tmp_dir} "
                        f"({repartitions} partições, max {mr} linhas/arquivo)."
                    )
                    wrote_ok = True
                    break
                except Exception as e_write:
                    last_write_error = e_write
                    print(f"[WARN] Escrita Spark falhou com maxRecordsPerFile={mr}: {e_write}")

            if not wrote_ok:
                raise RuntimeError(
                    f"Falha na escrita Spark para p-values após tentativas "
                    f"{write_attempts}. Ultimo erro: {last_write_error}"
                )
        finally:
            spark.stop()

        part_files = sorted(spark_tmp_dir.glob("part-*.csv"))
        if not part_files:
            raise RuntimeError("Spark não produziu nenhum part-file CSV.")
        _merge_chunks_to_csv(part_files, output_file)
        print(f"[INFO] Fase 4: pvalues_long.csv gerado em {output_file}.")

    except Exception as e:
        print(f"[WARN] Fase 4: Spark falhou ({e}). Fazendo merge streaming dos chunks.")
        _merge_chunks_to_csv(chunk_files, output_file)
    finally:
        if spark_tmp_dir.exists():
            shutil.rmtree(spark_tmp_dir, ignore_errors=True)


def _create_logret_combiner(v: Tuple[str, float]) -> Dict[str, List[float]]:
    ticker, ret = v
    return {ticker: [ret]}


def _merge_logret_value(acc: Dict[str, List[float]], v: Tuple[str, float]) -> Dict[str, List[float]]:
    ticker, ret = v
    acc.setdefault(ticker, []).append(ret)
    return acc


def _merge_logret_combiners(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> Dict[str, List[float]]:
    if len(b) > len(a):
        a, b = b, a
    for ticker, vals in b.items():
        a.setdefault(ticker, []).extend(vals)
    return a


def _compute_window_outputs(
    janela_id: int,
    janela_rows: Sequence[Tuple[str, str, Optional[float], str, str]],
    logret_map: Dict[str, List[float]],
    adf_done: set,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not janela_rows:
        return ({}, [])

    win_start = str(janela_rows[0][3])
    win_end = str(janela_rows[0][4])

    ticker_set = set()
    for t1, t2, _, _, _ in janela_rows:
        if t1:
            ticker_set.add(t1)
        if t2:
            ticker_set.add(t2)
    tickers = sorted(ticker_set)

    label = f"{win_start[:10]} a {win_end[:10]}"
    n_tickers = len(tickers)

    if n_tickers < 2:
        summary = {
            "Janela_ID": janela_id,
            "label": label,
            "ks_row": {
                "Janela_ID": janela_id,
                "Window_Start": win_start,
                "Window_End": win_end,
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
            },
            "adf_rows": [],
            "cross_rows": [],
            "ward_best_k": np.nan,
            "ward_silhouette": np.nan,
            "ward_modularity": np.nan,
        }
        return (summary, [])

    idx = {t: i for i, t in enumerate(tickers)}
    mat_real = np.eye(n_tickers, dtype=float)
    for t1, t2, corr, _, _ in janela_rows:
        i = idx.get(t1)
        j = idx.get(t2)
        if i is None or j is None:
            continue
        c = corr if (corr is not None and not np.isnan(corr)) else 1.0
        mat_real[i, j] = c
        mat_real[j, i] = c

    # Número real de observações da janela (de logret_map; fallback 252)
    n_obs = max((len(v) for v in logret_map.values()), default=252)

    # p-valores analíticos H0: rho_ij=0 via estatística t-Student
    pval_mat = calcular_pvalues_from_corr(mat_real, n_obs)

    pval_rows: List[Dict[str, Any]] = []
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            pval_rows.append(
                {
                    "Janela_ID": int(janela_id),
                    "Window_Start": win_start,
                    "Window_End": win_end,
                    "Ticker1": ticker1,
                    "Ticker2": ticker2,
                    "Correlacao": float(mat_real[i, j]),
                    "pvalue": float(pval_mat[i, j]),
                }
            )

    # KS test: correlações empíricas vs correlações de matriz aleatória N(0,1)
    rng_null = np.random.default_rng(seed=int(janela_id))
    mat_null_raw = rng_null.normal(0.0, 1.0, size=(n_obs, n_tickers))
    corr_null = np.corrcoef(mat_null_raw.T)
    ks_stat, ks_pval = calcular_ks(mat_real, corr_null)
    ks_row = {
        "Janela_ID": int(janela_id),
        "Window_Start": win_start,
        "Window_End": win_end,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pval,
    }

    adf_rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        if (int(janela_id), str(ticker)) in adf_done:
            continue
        try:
            serie = np.asarray(logret_map.get(ticker, []), dtype=float)
            if serie.size == 0:
                continue
            serie = serie[~np.isnan(serie)]
            if serie.size == 0:
                continue
            adf_result = calcular_adf(serie)
            adf_rows.append(
                {
                    "Janela_ID": int(janela_id),
                    "Window_Start": win_start,
                    "Window_End": win_end,
                    "Ticker": ticker,
                    **adf_result,
                }
            )
        except Exception:
            continue

    cross_rows: List[Dict[str, Any]] = []
    cross_results = run_multiple_clustering_methods(mat_real, max_k=10)
    for method, res in cross_results.get("resumo", {}).items():
        cross_rows.append(
            {
                "Janela_ID": int(janela_id),
                "Window_Start": win_start,
                "Window_End": win_end,
                "method": method,
                "best_k": res.get("best_k"),
                "silhouette": res.get("best_silhouette"),
                "modularity": res.get("modularity"),
            }
        )

    # Marchenko-Pastur por janela
    mp_stats = compute_mp_stats(mat_real, n_obs)
    mp_row = {
        "Janela_ID": int(janela_id),
        "Window_Start": win_start,
        "Window_End": win_end,
        **{k: mp_stats[k] for k in [
            "n_assets", "n_obs", "q", "lambda_plus", "lambda_max_emp",
            "n_signal", "n_noise", "var_signal_frac", "market_mode_frac",
        ]},
    }

    ward_res = cross_results.get("resumo", {}).get("ward")
    summary = {
        "Janela_ID": int(janela_id),
        "label": label,
        "ks_row": ks_row,
        "mp_row": mp_row,
        "adf_rows": adf_rows,
        "cross_rows": cross_rows,
        "ward_best_k": ward_res.get("best_k") if ward_res else np.nan,
        "ward_silhouette": ward_res.get("best_silhouette") if ward_res else np.nan,
        "ward_modularity": ward_res.get("modularity") if ward_res else np.nan,
    }
    return (summary, pval_rows)


def _build_spark_session_for_fase4(app_name: str = "Fase4Validacao"):
    """Cria SparkSession usando o runtime Spark apropriado para o SO atual."""
    from pyspark.sql import SparkSession
    from src.utils.spark_runtime import (
        configurar_ambiente_linux,
        configurar_ambiente_windows,
        configurar_spark_linux,
        configurar_spark_windows,
    )

    is_windows = os.name == "nt"

    _apply_fase4_runtime_defaults()

    if is_windows:
        hadoop_home = os.getenv("HADOOP_HOME")
        if hadoop_home is not None:
            normalized_hadoop_home = hadoop_home.strip().strip('"')
            if not normalized_hadoop_home:
                os.environ.pop("HADOOP_HOME", None)
            else:
                normalized_path = Path(normalized_hadoop_home)
                if normalized_path.is_absolute() and normalized_path.exists():
                    os.environ["HADOOP_HOME"] = str(normalized_path)
                else:
                    print(
                        f"[WARN] Ignorando HADOOP_HOME inválido no Windows: {hadoop_home!r}. "
                        "Usando runtime sem winutils explícito."
                    )
                    os.environ.pop("HADOOP_HOME", None)

        _log4j_path, log4j_uri = configurar_ambiente_windows()
    else:
        _log4j_path, log4j_uri = configurar_ambiente_linux()

    log4j_java_opt = f"-Dlog4j2.configurationFile={log4j_uri}"

    require_gpu = os.getenv("PAPER1_REQUIRE_GPU", "0") == "1"

    def _make_session(conf):
        conf.setAppName(app_name)
        return (
            SparkSession.builder
            .config(conf=conf)
            .config("spark.driver.extraJavaOptions", log4j_java_opt)
            .config("spark.executor.extraJavaOptions", log4j_java_opt)
            .getOrCreate()
        )

    conf = configurar_spark_windows(log4j_uri) if is_windows else configurar_spark_linux(log4j_uri)
    try:
        return _make_session(conf)
    except Exception as _e:
        if require_gpu:
            raise
        print(f"[WARN] Falha ao criar SparkSession (RAPIDS?): {_e}. Tentando CPU fallback...")
        os.environ["ENABLE_RAPIDS"] = "0"
        conf_cpu = configurar_spark_windows(log4j_uri) if is_windows else configurar_spark_linux(log4j_uri)
        return _make_session(conf_cpu)


def _run_fase4_distributed(
    matrix_file: Path,
    logret_file: Path,
    adf_done_list: Sequence[Tuple[int, str]],
    pvalues_output_file: Path,
    expected_windows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    from pyspark.storagelevel import StorageLevel
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        StringType,
        DoubleType,
    )

    shuffle_parts = max(
        1,
        int(
            os.getenv(
                "FASE4_SPARK_SHUFFLE_PARTITIONS",
                os.getenv("FASE4_PVALUES_REPARTITIONS", str(DEFAULT_FASE4_SPARK_SHUFFLE_PARTITIONS)),
            )
        ),
    )
    repartitions = max(
        1,
        int(os.getenv("FASE4_PVALUES_REPARTITIONS", str(DEFAULT_FASE4_PVALUES_REPARTITIONS))),
    )
    max_records = max(
        1,
        int(os.getenv("FASE4_MAX_RECORDS_PER_FILE", str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE))),
    )
    fallback_max_records = max(
        1,
        int(
            os.getenv(
                "FASE4_MAX_RECORDS_PER_FILE_FALLBACK",
                str(DEFAULT_FASE4_MAX_RECORDS_PER_FILE_FALLBACK),
            )
        ),
    )

    spark_tmp_dir = pvalues_output_file.parent / f"{pvalues_output_file.stem}_spark_parts"
    spark = _build_spark_session_for_fase4(app_name="Fase4ValidacaoDistributed")
    spark.sparkContext.setLogLevel("WARN")

    adf_done_set = set((int(jid), str(ticker)) for jid, ticker in adf_done_list)
    adf_done_bc = spark.sparkContext.broadcast(adf_done_set)

    try:
        # matrix_schema: ordem exata do CSV (Janela_ID, Window_Start, Window_End, Ticker1, Ticker2, Correlacao)
        matrix_schema = StructType(
            [
                StructField("Janela_ID", IntegerType(), True),
                StructField("Window_Start", StringType(), True),
                StructField("Window_End", StringType(), True),
                StructField("Ticker1", StringType(), True),
                StructField("Ticker2", StringType(), True),
                StructField("Correlacao", DoubleType(), True),
            ]
        )

        matrix_rdd = (
            spark.read.schema(matrix_schema)
            .option("header", True)
            .csv(str(matrix_file))
            .rdd
            .map(
                lambda r: (
                    _safe_int(r["Janela_ID"]),
                    (
                        str(r["Ticker1"] or "").strip(),
                        str(r["Ticker2"] or "").strip(),
                        _safe_float(r["Correlacao"]),
                        str(r["Window_Start"] or ""),
                        str(r["Window_End"] or ""),
                    ),
                )
            )
            .filter(lambda kv: kv[0] is not None)
            .partitionBy(shuffle_parts)
        )
        matrix_by_janela = matrix_rdd.groupByKey(numPartitions=shuffle_parts).mapValues(list)

        # logret: CSV tem 13 colunas — lê sem schema e seleciona apenas as 3 necessárias
        from pyspark.sql.functions import col
        logret_rdd = (
            spark.read
            .option("header", True)
            .option("inferSchema", "false")
            .csv(str(logret_file))
            .select(
                col("Janela_ID").cast(IntegerType()).alias("Janela_ID"),
                col("Ticker").cast(StringType()).alias("Ticker"),
                col("Retorno_Log").cast(DoubleType()).alias("Retorno_Log"),
            )
            .rdd
            .map(
                lambda r: (
                    _safe_int(r["Janela_ID"]),
                    (str(r["Ticker"] or "").strip(), _safe_float(r["Retorno_Log"])),
                )
            )
            .filter(
                lambda kv: kv[0] is not None
                and kv[1][0] != ""
                and kv[1][1] is not None
                and not np.isnan(float(kv[1][1]))
            )
            .map(lambda kv: (kv[0], (kv[1][0], float(kv[1][1]))))
            .partitionBy(shuffle_parts)
        )
        logret_by_janela = logret_rdd.combineByKey(
            _create_logret_combiner,
            _merge_logret_value,
            _merge_logret_combiners,
            numPartitions=shuffle_parts,
        )

        joined = matrix_by_janela.leftOuterJoin(logret_by_janela, numPartitions=shuffle_parts)
        storage_level_name = os.getenv("FASE4_STORAGE_LEVEL", "DISK_ONLY").strip().upper()
        storage_level = StorageLevel.DISK_ONLY if storage_level_name == "DISK_ONLY" else StorageLevel.MEMORY_AND_DISK

        processed = (
            joined.map(
                lambda kv: _compute_window_outputs(
                    janela_id=int(kv[0]),
                    janela_rows=list(kv[1][0]),
                    logret_map=kv[1][1] if kv[1][1] is not None else {},
                    adf_done=adf_done_bc.value,
                )
            )
            .filter(lambda pair: bool(pair[0]))
            .persist(storage_level)
        )

        print(f"[INFO] Fase 4 Spark: storage_level={storage_level_name}.")
        print("[INFO] Fase 4 Spark: processando janelas (progresso de summaries)...")

        summary_rows: List[Dict[str, Any]] = []
        summary_iter = processed.map(lambda pair: pair[0]).toLocalIterator()
        if tqdm is not None:
            for summary in tqdm(summary_iter, total=expected_windows, desc="Fase4 janelas", unit="janela"):
                summary_rows.append(summary)
        else:
            count = 0
            for summary in summary_iter:
                summary_rows.append(summary)
                count += 1
                if count % 5 == 0:
                    print(f"[INFO] Fase 4 Spark: {count} janelas processadas...")

        pvalues_schema = StructType(
            [
                StructField("Janela_ID", IntegerType(), True),
                StructField("Window_Start", StringType(), True),
                StructField("Window_End", StringType(), True),
                StructField("Ticker1", StringType(), True),
                StructField("Ticker2", StringType(), True),
                StructField("Correlacao", DoubleType(), True),
                StructField("pvalue", DoubleType(), True),
            ]
        )
        pvalues_rdd = processed.flatMap(lambda pair: pair[1]).map(
            lambda r: (
                _safe_int(r.get("Janela_ID")),
                str(r.get("Window_Start", "")),
                str(r.get("Window_End", "")),
                str(r.get("Ticker1", "")),
                str(r.get("Ticker2", "")),
                _safe_float(r.get("Correlacao")),
                _safe_float(r.get("pvalue")),
            )
        )
        pvalues_df = spark.createDataFrame(pvalues_rdd, schema=pvalues_schema)

        print("[INFO] Fase 4 Spark: gravando p-values reparticionados...")

        write_attempts = [max_records]
        if fallback_max_records not in write_attempts:
            write_attempts.append(fallback_max_records)

        last_write_error = None
        wrote_ok = False
        for mr in write_attempts:
            try:
                if spark_tmp_dir.exists():
                    shutil.rmtree(spark_tmp_dir, ignore_errors=True)
                (
                    pvalues_df.repartition(repartitions, "Janela_ID")
                    .write
                    .mode("overwrite")
                    .option("header", True)
                    .option("maxRecordsPerFile", str(mr))
                    .csv(str(spark_tmp_dir))
                )
                print(
                    f"[INFO] Fase 4 Spark: p-values gravados em parts ({repartitions} partições, max {mr} linhas/arquivo)."
                )
                wrote_ok = True
                break
            except Exception as e_write:
                last_write_error = e_write
                print(f"[WARN] Escrita Spark de p-values falhou com maxRecordsPerFile={mr}: {e_write}")

        if not wrote_ok:
            raise RuntimeError(
                f"Falha na escrita Spark para p-values após tentativas {write_attempts}. Ultimo erro: {last_write_error}"
            )

        part_files = sorted(spark_tmp_dir.glob("part-*.csv"))
        if part_files:
            _merge_chunks_to_csv(part_files, pvalues_output_file)
        else:
            _write_csv_rows(pvalues_output_file, [], PVALUES_COLUMNS)

        processed.unpersist()
        return summary_rows

    finally:
        try:
            adf_done_bc.unpersist()
        except Exception:
            pass
        spark.stop()
        if spark_tmp_dir.exists():
            shutil.rmtree(spark_tmp_dir, ignore_errors=True)


def fase4_validacao(force: bool = False):
    print(f"\n{'#'*80}\nFASE 4: VALIDAÇÃO ESTATÍSTICA\n{'#'*80}")

    _root = Path(__file__).resolve().parent.parent
    done_file = _root / "pipeline/.fase4_done"
    if done_file.exists() and not force:
        print("[INFO] Fase 4 já concluída. Pulando execução.")
        return

    if _maybe_dispatch_fase4_runtime(_root, force=force):
        return

    _apply_fase4_runtime_defaults()

    # Diagnostico: mostrar configurações Spark/memória
    runtime_settings = _get_fase4_runtime_settings()

    print("\n" + "="*70)
    print("[INFO] Fase 4: Configurações Spark (distribuído, sem pandas no pipeline)")
    print("="*70)
    print(f"  Spark Master     : {runtime_settings['master']}")
    print(f"  Driver Memory    : {runtime_settings['driver_memory']} (recomendado: 6-8GB)")
    print(f"  Executor Memory  : {runtime_settings['executor_memory']} (recomendado: 5-7GB)")
    print(f"  Driver Overhead  : {runtime_settings['driver_memory_overhead']} (additional out-of-heap)")
    print(f"  Max Result Size  : {runtime_settings['driver_max_result_size']} (coleta/serialize do driver)")
    print(f"  Shuffle Partitions: {runtime_settings['shuffle_partitions']} (distribuição de carga)")
    print(f"  PValues Repartitions: {runtime_settings['pvalues_repartitions']} (parte de saída)")
    print(f"  Max Records/File : {runtime_settings['max_records_per_file']} (evita arquivos muito grandes)")
    print(f"  RAPIDS Habilitado: {runtime_settings['rapids_enabled']}")
    print("="*70 + "\n")

    # Usa caminho absoluto relativo ao diretório raiz do projeto
    _projeto_root = Path(__file__).resolve().parent.parent
    img_dir = _projeto_root / "figures/validation"
    img_dir.mkdir(parents=True, exist_ok=True)

    matrix_file = _projeto_root / "data/processed/correlacoes_matrizes_long.csv"
    if not matrix_file.exists():
        print(f"[ERRO] Arquivo não encontrado: {matrix_file}")
        print(f"[INFO] CWD atual: {os.getcwd()}")
        return

    janela_ids = _scan_unique_int_values(matrix_file, "Janela_ID")
    if not janela_ids:
        print(f"[ERRO] Nenhuma Janela_ID encontrada em {matrix_file}")
        return

    logret_file = _projeto_root / "data/processed/dados_consolidados.csv"
    _scan_ticker_stats(logret_file)

    adf_csv = img_dir / "adf_results.csv"
    adf_existing_rows = _read_csv_rows(adf_csv) if adf_csv.exists() else []
    adf_done = set()
    for r in adf_existing_rows:
        jid = _safe_int(r.get("Janela_ID"))
        ticker = str(r.get("Ticker", "")).strip()
        if jid is not None and ticker:
            adf_done.add((jid, ticker))

    correlation_dir = _projeto_root / "data/correlation"
    correlation_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fase 4: iniciando processamento distribuído em Spark (sem pandas).")
    try:
        distributed_results = _run_fase4_distributed(
            matrix_file=matrix_file,
            logret_file=logret_file,
            adf_done_list=list(adf_done),
            pvalues_output_file=correlation_dir / "pvalues_long.csv",
            expected_windows=len(janela_ids),
        )
    except Exception as exc:
        if sys.platform.startswith("linux") and os.getenv("ENABLE_RAPIDS", "0") == "1":
            print(f"[WARN] Falha no modo GPU+CPU da fase4: {exc}")
            print("[INFO] Reexecutando fase4 em CPU após falha do caminho acelerado.")
            _apply_emergency_cpu_profile()
            distributed_results = _run_fase4_distributed(
                matrix_file=matrix_file,
                logret_file=logret_file,
                adf_done_list=list(adf_done),
                pvalues_output_file=correlation_dir / "pvalues_long.csv",
                expected_windows=len(janela_ids),
            )
        elif sys.platform.startswith("linux"):
            print(f"[WARN] Falha na fase4 em modo CPU: {exc}")
            print("[INFO] Reexecutando fase4 com perfil de emergência anti-OOM...")
            _apply_emergency_cpu_profile()
            distributed_results = _run_fase4_distributed(
                matrix_file=matrix_file,
                logret_file=logret_file,
                adf_done_list=list(adf_done),
                pvalues_output_file=correlation_dir / "pvalues_long.csv",
                expected_windows=len(janela_ids),
            )
        else:
            raise

    results: List[Dict[str, Any]] = []
    crossmethod_summary: List[Dict[str, Any]] = []
    ks_rows: List[Dict[str, Any]] = []
    mp_rows: List[Dict[str, Any]] = []
    adf_rows: List[Dict[str, Any]] = []
    for summary in distributed_results:
        if not summary:
            continue
        results.append(
            {
                "Janela_ID": summary.get("Janela_ID"),
                "label": summary.get("label", ""),
                "ward_best_k": summary.get("ward_best_k", np.nan),
                "ward_silhouette": summary.get("ward_silhouette", np.nan),
                "ward_modularity": summary.get("ward_modularity", np.nan),
            }
        )
        crossmethod_summary.extend(summary.get("cross_rows", []))
        adf_rows.extend(summary.get("adf_rows", []))
        ks_row = summary.get("ks_row")
        if ks_row is not None:
            ks_rows.append(ks_row)
        mp_row = summary.get("mp_row")
        if mp_row is not None:
            mp_rows.append(mp_row)

    if not results:
        return

    order_map = {jid: idx for idx, jid in enumerate(janela_ids)}
    results.sort(key=lambda r: order_map.get(r.get("Janela_ID"), 0))

    labels = [r["label"] for r in results]
    silhouettes = [r["ward_silhouette"] for r in results]
    best_ks = [r["ward_best_k"] for r in results]
    modulares = [r["ward_modularity"] for r in results]
    aris = [np.nan] * max(0, (len(labels) - 1))

    # Lazy import para evitar carregar matplotlib no import do módulo
    from src.visualization.plots import plot_metric_evolution

    plot_metric_evolution(labels, silhouettes, "Silhouette score", img_dir / "silhouette_evolucao.png", "Silhouette ao longo das janelas")
    plot_metric_evolution(labels, best_ks, "Best K", img_dir / "bestk_evolucao.png", "Best K ao longo das janelas")
    plot_metric_evolution(labels, modulares, "Modularidade", img_dir / "modularidade_evolucao.png", "Modularidade ao longo das janelas")
    plot_metric_evolution(labels[1:], aris, "ARI", img_dir / "ari_evolucao.png", "Estabilidade dos clusters (ARI)")

    _write_csv_rows(img_dir / "crossmethod_summary.csv", crossmethod_summary, CROSS_COLUMNS)
    _write_csv_rows(img_dir / "ks_results.csv", ks_rows, KS_COLUMNS)
    _write_csv_rows(img_dir / "mp_results.csv", mp_rows, MP_COLUMNS)

    # Gráfico duplo KS + Marchenko-Pastur
    if ks_rows and mp_rows:
        from src.visualization.plots import plot_ks_mp_dual
        ks_ordered  = sorted(ks_rows, key=lambda r: order_map.get(r.get("Janela_ID"), 0))
        mp_ordered  = sorted(mp_rows,  key=lambda r: order_map.get(r.get("Janela_ID"), 0))
        ks_labels   = [r["Window_Start"] for r in ks_ordered]
        ks_stats    = [r["ks_stat"]  for r in ks_ordered]
        mp_signal   = [r["var_signal_frac"] for r in mp_ordered]
        mp_mkt      = [r["market_mode_frac"] for r in mp_ordered]
        plot_ks_mp_dual(
            labels=ks_labels,
            ks_stats=ks_stats,
            mp_signal_frac=mp_signal,
            mp_market_frac=mp_mkt,
            output_path=img_dir / "ks_mp_evolucao.png",
        )

    adf_new_rows = adf_rows
    adf_merged = _dedupe_keep_last(adf_existing_rows + adf_new_rows, ("Janela_ID", "Ticker"))
    adf_cols = _infer_fieldnames(adf_merged, ADF_BASE_COLUMNS)
    _write_csv_rows(adf_csv, adf_merged, adf_cols)

    all_pngs = list(img_dir.glob("*.png"))
    zip_path = img_dir / "fase4_figuras.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in all_pngs:
            zipf.write(f, arcname=f.name)

    ks_stat_mean = _mean_numeric(row.get("ks_stat") for row in ks_rows)
    ks_pvalue_mean = _mean_numeric(row.get("ks_pvalue") for row in ks_rows)

    stationary_flags = []
    for row in adf_merged:
        b = _to_bool(row.get("adf_stationary"))
        if b is not None:
            stationary_flags.append(1.0 if b else 0.0)
    pct_stationary = (float(np.mean(stationary_flags)) * 100.0) if stationary_flags else np.nan

    adf_stat_mean = _mean_numeric(row.get("adf_stat") for row in adf_merged)
    adf_pvalue_mean = _mean_numeric(row.get("adf_pvalue") for row in adf_merged)

    ward_rows = [r for r in crossmethod_summary if str(r.get("method", "")).strip().lower() == "ward"]
    ward_best_k_mean = _mean_numeric(r.get("best_k") for r in ward_rows)
    ward_silhouette_mean = _mean_numeric(r.get("silhouette") for r in ward_rows)
    ward_modularity_mean = _mean_numeric(r.get("modularity") for r in ward_rows)

    summary_row = {
        "KS_stat_mean": ks_stat_mean,
        "KS_pvalue_mean": ks_pvalue_mean,
        "ADF_pct_stationary": pct_stationary,
        "ADF_stat_mean": adf_stat_mean,
        "ADF_pvalue_mean": adf_pvalue_mean,
        "Ward_best_k_mean": ward_best_k_mean,
        "Ward_silhouette_mean": ward_silhouette_mean,
        "Ward_modularity_mean": ward_modularity_mean,
    }

    summary_csv = img_dir / "validation_summary.csv"
    summary_columns = list(summary_row.keys())
    _write_csv_rows(summary_csv, [summary_row], summary_columns)

    paper_dir = Path("paper")
    paper_dir.mkdir(parents=True, exist_ok=True)

    summary_tex = paper_dir / "validation_summary_table.tex"
    _write_simple_latex_table(
        summary_tex,
        "Summary of statistical validation diagnostics over all rolling windows.",
        "tab:validation-summary",
        summary_columns,
        [summary_row],
        float_decimals=4,
    )

    cluster_sample_tex = paper_dir / "clusterization_sample_table.tex"
    if crossmethod_summary:
        cols = ["Janela_ID", "Window_Start", "Window_End", "method", "best_k", "silhouette", "modularity"]
        cluster_sample_rows = [{c: r.get(c, "") for c in cols} for r in crossmethod_summary[:5]]
        _write_simple_latex_table(
            cluster_sample_tex,
            "Sample of clustering diagnostics by method and rolling window.",
            "tab:cluster-sample",
            cols,
            cluster_sample_rows,
            float_decimals=6,
        )

    done_file.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa a Fase 4 de validação estatística.")
    parser.add_argument("--force", action="store_true", help="Ignora a sentinela pipeline/.fase4_done e reexecuta a fase 4.")
    args = parser.parse_args()
    fase4_validacao(force=args.force)
