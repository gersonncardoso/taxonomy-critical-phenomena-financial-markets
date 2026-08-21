"""
B3 Data Downloader - Versão Incremental v2.0
=============================================
Baixa dados históricos de ações da B3 com download incremental.

MELHORIAS v2.0:
- Download incremental (verifica o que já existe)
- Modo FORCE (apaga e re-baixa tudo)
- Log de downloads em JSON
- Verificação de qualidade de dados
- Salva apenas formato LONG
- Detecção automática de períodos faltantes
- Remove duplicatas automaticamente
- Mantém histórico e adiciona novos dados

Autor: Gerson Nassor Cardoso
Instituição: UNIFESP
Data: 2026-02-18
Versão: 2.0

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import requests
import pandas as pd
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import setup_logger

logger = setup_logger('b3_downloader')


class B3Downloader:
    """
    Classe para download incremental de dados históricos da B3
    
    Características v2.0:
    - Download incremental (não re-baixa dados existentes)
    - Modo FORCE (re-baixa tudo)
    - Salva em formato LONG
    - Mantém log de downloads
    - Identifica períodos faltantes
    - Remove duplicatas automaticamente
    """
    
    BASE_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist"
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Inicializa o downloader
        
        Parameters:
        -----------
        data_dir : str
            Diretório onde os arquivos serão salvos
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo principal (LONG format)
        self.output_file = self.data_dir / 'b3_dados_completos.csv'
        self.main_file = self.output_file  # Alias para compatibilidade
        
        # Log de downloads
        self.log_file = self.data_dir / 'download_log.json'
        self.log = self._load_log()
        
        logger.info("B3Downloader v2.0 inicializado")
        logger.info(f"Diretório: {self.data_dir}")
        logger.info(f"Arquivo principal: {self.output_file.name}")
    
    def _load_log(self) -> Dict:
        """Carrega log de downloads"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'last_update': None,
            'downloads': [],
            'total_records': 0,
            'date_range': {'start': None, 'end': None}
        }
    
    def _save_log(self):
        """Salva log de downloads"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.log, indent=2, fp=f, ensure_ascii=False)
            logger.debug("Log salvo com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao salvar log: {e}")
    
    def _parse_cotahist_line(self, line: str) -> Optional[Dict]:
        """
        Faz o parsing de uma linha do arquivo COTAHIST
        
        Layout: http://www.b3.com.br/data/files/33/67/B9/50/D84057102C784E47AC094EA8/SeriesHistoricas_Layout.pdf
        """
        try:
            return {
                'Date': datetime.strptime(line[2:10], '%Y%m%d').date(),
                'Ticker': line[12:24].strip(),
                'Tipo_Mercado': line[24:27].strip(),
                'Nome_Empresa': line[27:39].strip(),
                'Preco_Abertura': float(line[56:69]) / 100,
                'Preco_Maximo': float(line[69:82]) / 100,
                'Preco_Minimo': float(line[82:95]) / 100,
                'Preco_Medio': float(line[95:108]) / 100,
                'Preco_Fechamento': float(line[108:121]) / 100,
                'Volume': float(line[170:188]) / 100,
                'Quantidade_Negocios': int(line[147:152])
            }
        except Exception:
            return None
    
    def _download_and_extract(self, url: str, timeout: int = 60) -> Optional[str]:
        """Baixa e extrai arquivo ZIP da B3"""
        try:
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    txt_file = z.namelist()[0]
                    with z.open(txt_file) as f:
                        content = f.read().decode('latin-1')
                return content
            else:
                return None
        except Exception as e:
            logger.warning(f"Erro ao baixar {url}: {e}")
            return None
    
    def _parse_content(self, content: str) -> Optional[pd.DataFrame]:
        """Converte conteúdo COTAHIST em DataFrame"""
        lines = content.strip().split('\n')
        data_lines = [line for line in lines if line.startswith('01')]
        
        records = []
        for line in data_lines:
            record = self._parse_cotahist_line(line)
            if record:
                records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            # Filtra apenas mercado à vista (010)
            df = df[df['Tipo_Mercado'] == '010'].copy()
            df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
            return df
        
        return None
    
    def get_existing_data_range(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Obtém range de datas já baixadas
        
        Returns:
            Tuple (data_inicio, data_fim) ou (None, None)
        """
        if not self.output_file.exists():
            return None, None
        
        try:
            # Lê apenas primeira e última linha para performance
            df_sample = pd.read_csv(self.output_file, usecols=['Date'], nrows=1)
            df_last = pd.read_csv(self.output_file, usecols=['Date']).tail(1)
            
            if not df_sample.empty and not df_last.empty:
                return str(df_sample['Date'].iloc[0]), str(df_last['Date'].iloc[0])
        except:
            pass
        
        return None, None
    
    def get_existing_dates(self) -> List[str]:
        """
        Obtém lista de todas as datas já baixadas
        
        Returns:
            Lista de datas únicas (formato YYYY-MM-DD)
        """
        if not self.output_file.exists():
            return []
        
        try:
            df = pd.read_csv(self.output_file, usecols=['Date'])
            dates = pd.to_datetime(df['Date']).dt.date.unique()
            return sorted([str(d) for d in dates])
        except:
            return []
    
    def identify_missing_dates(self, 
                              data_inicio: str, 
                              data_fim: str) -> List[str]:
        """
        Identifica datas faltando no período
        
        Args:
            data_inicio: Data inicial (YYYY-MM-DD)
            data_fim: Data final (YYYY-MM-DD)
        
        Returns:
            Lista de datas faltando (apenas dias úteis)
        """
        # Gerar todas as datas úteis do período
        inicio = pd.to_datetime(data_inicio)
        fim = pd.to_datetime(data_fim)
        todas_datas = pd.bdate_range(start=inicio, end=fim, freq='B')
        todas_datas_str = set([d.strftime('%Y-%m-%d') for d in todas_datas])
        
        # Datas já existentes
        existentes = set(self.get_existing_dates())
        
        # Datas faltando
        faltando = sorted(list(todas_datas_str - existentes))
        
        return faltando
    
    def download_mes(self, ano: int, mes: int) -> Optional[pd.DataFrame]:
        """
        Baixa dados de um mês específico
        
        Args:
            ano: Ano (ex: 2023)
            mes: Mês (1-12)
        
        Returns:
            DataFrame ou None
        """
        url = f"{self.BASE_URL}/COTAHIST_M{mes:02d}{ano}.ZIP"
        
        logger.info(f"Baixando {mes:02d}/{ano}")
        
        content = self._download_and_extract(url)
        
        if content:
            df = self._parse_content(content)
            if df is not None:
                logger.info(f"{mes:02d}/{ano}: {len(df):,} registros")
                return df
        
        logger.warning(f"{mes:02d}/{ano}: Falhou")
        return None
    
    def download_periodo_completo(self, 
                                  data_inicio: str, 
                                  data_fim: str) -> Optional[pd.DataFrame]:
        """
        Baixa todos os dados de um período (usado internamente)
        
        Args:
            data_inicio: Data inicial (YYYY-MM-DD)
            data_fim: Data final (YYYY-MM-DD)
        
        Returns:
            DataFrame com todos os dados ou None
        """
        logger.info(f"Baixando período: {data_inicio} a {data_fim}")
        
        # Determinar meses a baixar
        dt_inicio = pd.to_datetime(data_inicio)
        dt_fim = pd.to_datetime(data_fim)
        
        meses = []
        current = dt_inicio.replace(day=1)
        while current <= dt_fim:
            meses.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        # Download
        dfs = []
        
        for ano, mes in meses:
            df = self.download_mes(ano, mes)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            logger.error("Nenhum dado foi baixado")
            return None
        
        # Combinar dados
        df_final = pd.concat(dfs, ignore_index=True)
        
        # Filtrar período exato
        df_final = df_final[
            (df_final['Date'] >= dt_inicio.date()) & 
            (df_final['Date'] <= dt_fim.date())
        ]
        
        # Remover duplicatas
        df_final = df_final.drop_duplicates(subset=['Date', 'Ticker'], keep='first')
        df_final = df_final.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        logger.info(f"Download completo: {len(df_final):,} registros")
        
        return df_final
    
    def download_periodo_incremental(self, 
                                    data_inicio: str, 
                                    data_fim: str,
                                    force: bool = False) -> Dict[str, Any]:
        """
        Download incremental - NÃO re-baixa dados que já existem
        MANTÉM histórico e remove duplicatas
        
        Args:
            data_inicio: Data inicial (YYYY-MM-DD)
            data_fim: Data final (YYYY-MM-DD)
            force: Se True, apaga tudo e re-baixa (CUIDADO!)
        
        Returns:
            Dict com estatísticas do download
        """
        logger.info("="*80)
        logger.info("DOWNLOAD INCREMENTAL DA B3")
        logger.info("="*80)
        
        # Se force=True, apagar arquivo existente
        if force and self.output_file.exists():
            logger.warning(f"MODO FORCE: Apagando arquivo existente: {self.output_file}")
            print(f"   🗑️  Removendo: {self.output_file}")
            self.output_file.unlink()
            print(f"   ✅ Arquivo removido")
        
        # Verificar se já existe arquivo
        df_existente = None
        ultima_data_existente = None
        primeira_data_existente = None
        
        if self.output_file.exists() and not force:
            try:
                logger.info(f"Arquivo existente encontrado: {self.output_file}")
                print(f"   📂 Carregando dados existentes...")
                df_existente = pd.read_csv(self.output_file)
                
                # CORREÇÃO: Garantir que Date é datetime antes de min/max
                if 'Date' in df_existente.columns:
                    df_existente['Date'] = pd.to_datetime(df_existente['Date']).dt.date
                    ultima_data_existente = df_existente['Date'].max()
                    primeira_data_existente = df_existente['Date'].min()
                    logger.info(f"Período existente: {primeira_data_existente} a {ultima_data_existente}")
                    print(f"   ✅ Período existente: {primeira_data_existente} a {ultima_data_existente}")
                
                logger.info(f"Dados existentes: {len(df_existente):,} registros")
                print(f"   ✅ Registros existentes: {len(df_existente):,}")
                
            except Exception as e:
                logger.warning(f"Erro ao ler arquivo existente: {e}")
                print(f"   ⚠️  Erro ao ler arquivo existente: {e}")
                df_existente = None
        
        # Definir período a baixar
        data_inicio_download = data_inicio
        
        if df_existente is not None and ultima_data_existente is not None:
            # Baixar apenas dados APÓS a última data existente
            from datetime import timedelta
            
            # Converter string para date
            if isinstance(data_inicio, str):
                data_inicio_obj = datetime.strptime(data_inicio, '%Y-%m-%d').date()
            else:
                data_inicio_obj = data_inicio
            
            # Próximo dia após última data
            proxima_data = ultima_data_existente + timedelta(days=1)
            
            # Se próxima_data > data_fim, nada a baixar
            if isinstance(data_fim, str):
                data_fim_obj = datetime.strptime(data_fim, '%Y-%m-%d').date()
            else:
                data_fim_obj = data_fim
            
            if proxima_data > data_fim_obj:
                logger.info("Dados já atualizados até a data solicitada!")
                print(f"\n   ✅ Dados já atualizados até {data_fim_obj}")
                print(f"   ℹ️  Nada a baixar")
                
                return {
                    'status': 'complete',
                    'novos_registros': 0,
                    'total_registros': len(df_existente),
                    'start': str(primeira_data_existente),
                    'end': str(ultima_data_existente),
                    'ativos': df_existente['Ticker'].nunique() if 'Ticker' in df_existente.columns else 0,
                    'downloaded': 0,
                    'failed': 0
                }
            
            data_inicio_download = str(proxima_data)
            logger.info(f"Download incremental: {data_inicio_download} a {data_fim}")
            print(f"\n   📥 Baixando dados de {data_inicio_download} até {data_fim}")
        else:
            logger.info(f"Download completo: {data_inicio} a {data_fim}")
            print(f"\n   📥 Baixando dados de {data_inicio} até {data_fim}")
        
        # Baixar novos dados
        logger.info("Baixando novos dados da B3...")
        print(f"   ⏳ Baixando dados da B3...")
        
        df_novos = self.download_periodo_completo(data_inicio_download, data_fim)
        
        if df_novos is None or df_novos.empty:
            logger.warning("Nenhum dado novo baixado")
            print(f"   ⚠️  Nenhum dado novo disponível")
            
            if df_existente is not None:
                return {
                    'status': 'complete',
                    'novos_registros': 0,
                    'total_registros': len(df_existente),
                    'start': str(primeira_data_existente),
                    'end': str(ultima_data_existente),
                    'ativos': df_existente['Ticker'].nunique() if 'Ticker' in df_existente.columns else 0,
                    'downloaded': 0,
                    'failed': 0
                }
            else:
                return {
                    'status': 'failed',
                    'novos_registros': 0,
                    'total_registros': 0,
                    'start': None,
                    'end': None,
                    'ativos': 0,
                    'downloaded': 0,
                    'failed': 1
                }
        
        # CORREÇÃO: Garantir tipo consistente em df_novos
        df_novos['Date'] = pd.to_datetime(df_novos['Date']).dt.date
        
        print(f"   ✅ Novos dados baixados: {len(df_novos):,} registros")
        
        # Combinar com dados existentes
        if df_existente is not None:
            logger.info(f"Combinando {len(df_existente):,} registros existentes com {len(df_novos):,} novos")
            print(f"   🔄 Combinando dados existentes com novos...")
            
            # CONCATENAR
            df_final = pd.concat([df_existente, df_novos], ignore_index=True)
            
            # REMOVER DUPLICATAS (manter o mais recente)
            antes = len(df_final)
            df_final = df_final.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
            depois = len(df_final)
            
            if antes != depois:
                logger.info(f"Removidas {antes - depois:,} linhas duplicadas")
                print(f"   🗑️  Removidas {antes - depois:,} linhas duplicadas")
            
            novos_registros = len(df_novos)
        else:
            df_final = df_novos
            novos_registros = len(df_novos)
        
        # CORREÇÃO: Garantir tipo consistente antes de min/max
        df_final['Date'] = pd.to_datetime(df_final['Date']).dt.date
        
        # Ordenar por data e ticker
        df_final = df_final.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        # Salvar
        logger.info(f"Salvando dados consolidados: {self.output_file}")
        print(f"   💾 Salvando dados consolidados...")
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(self.output_file, index=False)
        
        print(f"   ✅ Arquivo salvo: {self.output_file}")
        
        # Atualizar log
        self.log['last_update'] = datetime.now().isoformat()
        self.log['total_records'] = len(df_final)
        self.log['date_range'] = {
            'start': str(df_final['Date'].min()),
            'end': str(df_final['Date'].max())
        }
        self.log['downloads'].append({
            'timestamp': datetime.now().isoformat(),
            'period': f"{data_inicio} até {data_fim}",
            'new_records': novos_registros,
            'total_after': len(df_final),
            'force': force
        })
        self._save_log()
        
        stats = {
            'status': 'success',
            'novos_registros': novos_registros,
            'total_registros': len(df_final),
            'start': str(df_final['Date'].min()),
            'end': str(df_final['Date'].max()),
            'ativos': df_final['Ticker'].nunique() if 'Ticker' in df_final.columns else 0,
            'downloaded': 1,
            'failed': 0
        }
        
        logger.info("="*80)
        logger.info("DOWNLOAD CONCLUÍDO")
        logger.info("="*80)
        logger.info(f"Novos registros: {stats['novos_registros']:,}")
        logger.info(f"Total de registros: {stats['total_registros']:,}")
        logger.info(f"Período: {stats['start']} a {stats['end']}")
        logger.info(f"Ativos únicos: {stats['ativos']}")
        logger.info("="*80)
        
        print(f"\n   📊 Resumo:")
        print(f"   • Novos registros: {stats['novos_registros']:,}")
        print(f"   • Total de registros: {stats['total_registros']:,}")
        print(f"   • Período: {stats['start']} a {stats['end']}")
        print(f"   • Ativos únicos: {stats['ativos']}")
        
        return stats
    
    def filtrar_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Filtra dados para incluir apenas tickers especificados
        
        Args:
            tickers: Lista de tickers
        
        Returns:
            DataFrame filtrado
        """
        if not self.output_file.exists():
            logger.warning("Nenhum dado baixado ainda")
            return pd.DataFrame()
        
        logger.info(f"Filtrando {len(tickers)} tickers")
        
        df = pd.read_csv(self.output_file)
        df_filtrado = df[df['Ticker'].isin(tickers)].copy()
        
        tickers_encontrados = df_filtrado['Ticker'].unique()
        tickers_faltando = set(tickers) - set(tickers_encontrados)
        
        logger.info(f"Encontrados: {len(tickers_encontrados)}/{len(tickers)}")
        
        if tickers_faltando:
            logger.warning(f"Não encontrados: {len(tickers_faltando)}")
        
        return df_filtrado
    
    def get_info(self) -> Dict:
        """
        Obtém informações sobre os dados baixados
        
        Returns:
            Dict com estatísticas
        """
        if not self.output_file.exists():
            return {
                'exists': False,
                'total_records': 0
            }
        
        df = pd.read_csv(self.output_file, usecols=['Date', 'Ticker'])
        
        return {
            'exists': True,
            'file': str(self.output_file),
            'size_mb': self.output_file.stat().st_size / 1024 / 1024,
            'total_records': len(df),
            'tickers': df['Ticker'].nunique(),
            'dates': df['Date'].nunique(),
            'date_range': {
                'start': str(df['Date'].min()),
                'end': str(df['Date'].max())
            },
            'last_update': self.log.get('last_update')
        }
    
    def print_info(self):
        """Imprime informações formatadas"""
        info = self.get_info()
        
        print("\n" + "="*80)
        print("📊 INFORMAÇÕES DOS DADOS")
        print("="*80)
        
        if not info['exists']:
            print("\n⚠️  Nenhum dado baixado ainda")
            print("\n💡 Execute: downloader.download_periodo_incremental('2014-01-01', '2024-12-31')")
        else:
            print(f"\n📄 Arquivo: {info['file']}")
            print(f"💾 Tamanho: {info['size_mb']:.2f} MB")
            print(f"\n📊 Dados:")
            print(f"   Registros: {info['total_records']:,}")
            print(f"   Tickers: {info['tickers']}")
            print(f"   Dias: {info['dates']}")
            print(f"   Período: {info['date_range']['start']} até {info['date_range']['end']}")
            
            if info['last_update']:
                dt = datetime.fromisoformat(info['last_update'])
                print(f"\n🕐 Última atualização: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("="*80)


# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ============================================================================

def download_rapido(data_inicio: str, 
                   data_fim: str,
                   tickers: Optional[List[str]] = None,
                   force: bool = False) -> pd.DataFrame:
    """
    Função simplificada para download rápido
    
    Args:
        data_inicio: Data inicial (YYYY-MM-DD)
        data_fim: Data final (YYYY-MM-DD)
        tickers: Lista de tickers (opcional)
        force: Forçar re-download
    
    Returns:
        DataFrame com dados
    
    Exemplo:
        >>> df = download_rapido('2024-01-01', '2024-12-31', 
        ...                      tickers=['PETR4', 'VALE3'])
    """
    downloader = B3Downloader()
    
    # Download incremental
    downloader.download_periodo_incremental(data_inicio, data_fim, force=force)
    
    # Filtrar tickers se especificado
    if tickers:
        df = downloader.filtrar_tickers(tickers)
    else:
        df = pd.read_csv(downloader.output_file)
    
    downloader.print_info()
    
    return df


if __name__ == '__main__':
    # Teste
    print("🧪 Modo de teste\n")
    
    downloader = B3Downloader()
    downloader.print_info()
    
    # Teste com dezembro/2023
    downloader.download_periodo_incremental('2023-12-01', '2023-12-31')
    downloader.print_info()