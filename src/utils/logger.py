import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str = None):
    """Configura um logger com output para arquivo e console.
    
    Args:
        name: Nome do logger
        log_file: Caminho para o arquivo de log (opcional)
    
    Returns:
        Logger configurado
    """
    # Criar o logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formato do log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 