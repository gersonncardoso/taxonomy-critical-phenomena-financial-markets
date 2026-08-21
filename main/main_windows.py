import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import getpass
import socket
import datetime
import logging
import traceback
import subprocess
import glob
from pyspark import SparkConf
from pyspark.sql import SparkSession
from tqdm import tqdm
from src.utils import (
    log_env_info, setup_logging, debug_spark_env, configurar_ambiente_windows, configurar_spark_windows, executar_pipeline
)

def main_windows():
    so = sys.platform.lower()
    usuario = getpass.getuser()
    cwd = os.getcwd()
    python_path = sys.executable
    hostname = socket.gethostname()
    start_time = datetime.datetime.now()
    log4j_path, log4j_uri = configurar_ambiente_windows()
    log_env_info(so, usuario, cwd, python_path, hostname, start_time)
    setup_logging(start_time, so, usuario, hostname, cwd, python_path)
    debug_spark_env()
    conf = configurar_spark_windows(log4j_uri)
    print('[DEBUG] conf.getAll():', conf.getAll())
    logging.info('[DEBUG] conf.getAll(): %s', conf.getAll())
    print('[TESTE] Antes de criar SparkSession...')
    logging.info('[TESTE] Antes de criar SparkSession...')
    try:
        spark = (
            SparkSession.builder
            .config(conf=conf)
            .config("spark.driver.extraJavaOptions", "-Dlog4j2.configurationFile=file:main/log4j2.properties")
            .config("spark.executor.extraJavaOptions", "-Dlog4j2.configurationFile=file:main/log4j2.properties")
            .getOrCreate()
        )
        print('[TESTE] SparkSession criada com sucesso!')
        logging.info('SparkSession criada. master: %s', spark.sparkContext.master)
        logging.info('[TESTE] SparkSession criada com sucesso!')
        print('[TESTE] Iniciando verificações de ambiente...')
        logging.info('[TESTE] Iniciando verificações de ambiente...')
        print('[TESTE] Verificação JAVA_HOME:', os.environ.get('JAVA_HOME'))    
        print('[TESTE] Verificação HADOOP_HOME:', os.environ.get('HADOOP_HOME'))
        print('[TESTE] Verificação PATH contém hadoop/bin:', 'C:\\hadoop\\bin' in os.environ['PATH'])
        try:
            java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True)
            print('[TESTE] Versão do Java:')
            print(java_version)
            logging.info('[TESTE] Versão do Java: %s', java_version)
        except Exception as e:
            print('[TESTE] Não foi possível obter a versão do Java:', e)        
            logging.error('[TESTE] Não foi possível obter a versão do Java: %s', e)
        print('[TESTE] Iniciando funções utilitárias...')
        logging.info('[TESTE] Iniciando funções utilitárias...')
        executar_pipeline(spark, is_gpu=False)
    except Exception as e:
        print('[TESTE] ERRO ao criar SparkSession!')
        logging.error('Erro ao criar SparkSession:')
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main_windows()
