"""
Verificação e Instalação Automática de Dependências
====================================================

Verifica e instala automaticamente apenas pacotes que realmente faltam.

Autor: Gerson Nassor Cardoso
Instituição: UNIFESP
Data: 2026-02-18
"""

import subprocess
import sys
import importlib
import platform
from pathlib import Path
from typing import List, Dict, Tuple


# Mapeamento de nomes de pacotes pip → nomes de importação
MAPEAMENTO_PACOTES = {
    'python-louvain': 'community',
    'scikit-learn': 'sklearn',
    'beautifulsoup4': 'bs4',
    'python-dateutil': 'dateutil',
}


def extrair_nome_pacote(spec: str) -> str:
    """
    Extrai nome do pacote de uma especifica��ão
    
    Args:
        spec: String tipo "numpy>=1.24.0"
    
    Returns:
        Nome do pacote: "numpy"
    """
    for sep in ['>=', '==', '<=', '>', '<', '~=', '[']:
        if sep in spec:
            return spec.split(sep)[0].strip()
    return spec.strip()


def obter_nome_importacao(pacote_pip: str) -> str:
    """
    Obtém nome de importação a partir do nome pip
    
    Args:
        pacote_pip: Nome no pip (ex: "python-louvain")
    
    Returns:
        Nome para importar (ex: "community")
    """
    if pacote_pip in MAPEAMENTO_PACOTES:
        return MAPEAMENTO_PACOTES[pacote_pip]
    
    return pacote_pip.replace('-', '_')


def verificar_pacote_instalado(pacote: str) -> bool:
    """
    Verifica se um pacote está instalado
    
    Args:
        pacote: Nome do pacote
    
    Returns:
        True se instalado
    """
    nome_import = obter_nome_importacao(pacote)
    
    try:
        importlib.import_module(nome_import)
        return True
    except ImportError:
        return False


def instalar_pacote(spec: str, quiet: bool = True) -> bool:
    """
    Instala um pacote via pip (SEM forçar upgrade)
    
    Args:
        spec: Especificação do pacote
        quiet: Se True, suprime output
    
    Returns:
        True se instalação bem-sucedida
    """
    # NÃO usar --upgrade para não reinstalar o que já está OK
    cmd = [sys.executable, "-m", "pip", "install", spec]
    
    if quiet:
        cmd.append('--quiet')
    
    try:
        subprocess.check_call(
            cmd,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None
        )
        return True
    except subprocess.CalledProcessError:
        return False


def ler_requirements() -> List[str]:
    """
    Lê requirements.txt
    
    Returns:
        Lista de especificações de pacotes
    """
    requirements_file = Path('requirements.txt')
    
    if not requirements_file.exists():
        return []
    
    pacotes = []
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pacotes.append(line)
    
    return pacotes


def verificar_todas_dependencias() -> Tuple[List[str], List[str], List[str]]:
    """
    Verifica todas as dependências Python
    
    Returns:
        Tupla (instalados, faltando_specs, faltando_nomes)
    """
    pacotes_specs = ler_requirements()
    
    instalados = []
    faltando_specs = []
    faltando_nomes = []
    
    for spec in pacotes_specs:
        nome = extrair_nome_pacote(spec)
        
        if verificar_pacote_instalado(nome):
            instalados.append(nome)
        else:
            faltando_specs.append(spec)
            faltando_nomes.append(nome)
    
    return instalados, faltando_specs, faltando_nomes


def instalar_dependencias_faltando(
    faltando_specs: List[str],
    faltando_nomes: List[str],
    auto: bool = True
) -> Dict[str, bool]:
    """
    Instala APENAS pacotes que realmente faltam
    
    Args:
        faltando_specs: Lista de especificações
        faltando_nomes: Lista de nomes
        auto: Se True, instala automaticamente
    
    Returns:
        Dict {nome_pacote: sucesso}
    """
    if not faltando_specs:
        return {}
    
    print(f"\n📦 Pacotes Python a instalar ({len(faltando_specs)}):")
    for nome, spec in zip(faltando_nomes, faltando_specs):
        print(f"   - {spec}")
    
    if not auto:
        print(f"\n💡 Instalar automaticamente? (s/N): ", end='')
        resposta = input().strip().lower()
        
        if resposta != 's':
            print(f"⏭️  Instalação cancelada pelo usuário")
            return {nome: False for nome in faltando_nomes}
    
    print(f"\n🔄 Instalando pacotes Python...")
    
    resultados = {}
    
    for nome, spec in zip(faltando_nomes, faltando_specs):
        print(f"   📦 {nome:.<30} ", end='', flush=True)
        
        sucesso = instalar_pacote(spec, quiet=True)
        resultados[nome] = sucesso
        
        if sucesso:
            print(f"✅")
        else:
            print(f"❌")
    
    return resultados


def verificar_e_instalar_dependencias(auto: bool = True, verbose: bool = True) -> bool:
    """
    Verifica e instala APENAS dependências faltando
    
    Args:
        auto: Se True, instala automaticamente
        verbose: Se True, mostra detalhes
    
    Returns:
        True se todas as dependências estão OK
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 VERIFICANDO DEPENDÊNCIAS PYTHON")
        print(f"{'='*80}")
    
    # Verificar quais estão instaladas
    instalados, faltando_specs, faltando_nomes = verificar_todas_dependencias()
    
    if verbose:
        print(f"\n📊 Status:")
        print(f"   ✅ Já instalados: {len(instalados)}")
        print(f"   ⏳ Faltando:      {len(faltando_nomes)}")
    
    # Se tudo OK
    if not faltando_specs:
        if verbose:
            print(f"\n✅ Todas as dependências Python estão instaladas!")
            print(f"{'='*80}")
        return True
    
    # Instalar faltando
    resultados = instalar_dependencias_faltando(
        faltando_specs,
        faltando_nomes,
        auto=auto
    )
    
    # Verificar resultados
    sucesso_total = all(resultados.values())
    falhas = [nome for nome, ok in resultados.items() if not ok]
    
    if verbose:
        print(f"\n{'='*80}")
        if sucesso_total:
            print(f"✅ TODAS AS DEPENDÊNCIAS PYTHON INSTALADAS COM SUCESSO!")
        else:
            print(f"⚠️  ALGUMAS DEPENDÊNCIAS FALHARAM:")
            for nome in falhas:
                print(f"   ❌ {nome}")
            print(f"\n💡 Execute manualmente:")
            print(f"   pip install {' '.join(falhas)}")
        print(f"{'='*80}")
    
    return sucesso_total


def verificar_pdflatex() -> Dict[str, any]:
    """
    Verifica se pdflatex está disponível
    
    Returns:
        Dict com status e informações
    """
    info = {
        'disponivel': False,
        'versao': None,
        'comando_instalacao': None
    }
    
    try:
        result = subprocess.run(
            ['pdflatex', '--version'],
            check=True,
            capture_output=True,
            text=True
        )
        
        info['disponivel'] = True
        info['versao'] = result.stdout.split('\n')[0]
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        info['disponivel'] = False
        
        sistema = platform.system()
        if sistema == 'Windows':
            info['comando_instalacao'] = 'choco install miktex'
        elif sistema == 'Darwin':
            info['comando_instalacao'] = 'brew install basictex'
        elif sistema == 'Linux':
            info['comando_instalacao'] = 'sudo apt install texlive-full'
    
    return info


def verificar_dependencias_opcionais(verbose: bool = True) -> Dict[str, Dict]:
    """
    Verifica dependências opcionais (LaTeX)
    
    Args:
        verbose: Se True, mostra detalhes
    
    Returns:
        Dict com status
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 VERIFICANDO DEPENDÊNCIAS OPCIONAIS")
        print(f"{'='*80}")
    
    status = {}
    latex_info = verificar_pdflatex()
    status['latex'] = latex_info
    
    if verbose:
        print(f"\n📝 LaTeX (para compilar paper):")
        if latex_info['disponivel']:
            print(f"   ✅ DISPONÍVEL")
            print(f"   📍 {latex_info['versao']}")
        else:
            print(f"   ⚠️  NÃO DISPONÍVEL")
            print(f"   📝 FASE 7 (compilar paper) será pulada")
            if latex_info['comando_instalacao']:
                print(f"\n   💡 Para instalar:")
                print(f"      {latex_info['comando_instalacao']}")
        
        print(f"{'='*80}")
    
    return status


def verificar_todas_dependencias_completo(auto: bool = True, verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Verifica e instala dependências Python + verifica opcionais
    
    Args:
        auto: Se True, instala Python automaticamente
        verbose: Se True, mostra detalhes
    
    Returns:
        Tupla (python_ok, opcionais_status)
    """
    python_ok = verificar_e_instalar_dependencias(auto=auto, verbose=verbose)
    opcionais_status = verificar_dependencias_opcionais(verbose=verbose)
    
    return python_ok, opcionais_status


if __name__ == "__main__":
    print("="*80)
    print("TESTE COMPLETO DE DEPENDÊNCIAS")
    print("="*80)
    
    python_ok, opcionais = verificar_todas_dependencias_completo(
        auto=True,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("RESUMO")
    print(f"{'='*80}")
    print(f"   Python:  {'✅ OK' if python_ok else '❌ FALHOU'}")
    print(f"   LaTeX:   {'✅ OK' if opcionais['latex']['disponivel'] else '⚠️  Não disponível'}")
    print(f"{'='*80}")
    
    sys.exit(0 if python_ok else 1)