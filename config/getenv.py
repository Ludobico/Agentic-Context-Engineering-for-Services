import os
from typing import Union
import shutil
import configparser
from dotenv import load_dotenv

class GetEnv:
    def __init__(self):
        self.curdir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.abspath(os.path.join(self.curdir, 'config.ini'))
        self.example_config_path = os.path.abspath(os.path.join(self.curdir, 'config-example.ini'))

        load_dotenv(os.path.join(self.curdir, '..', '.env'))
        
        if not os.path.exists(self.config_path):
            if os.path.exists(self.example_config_path):
                shutil.copy(self.example_config_path, self.config_path)
            else:
                raise FileNotFoundError(
                    "Neither `config.ini` nor `config-example.ini` were found"
                )
        
        self.props = configparser.ConfigParser()
        self.DEFAULT_SECTION = "DEFAULT"
        self.EMBEDDING_SECTION = "EMBEDDING"
        self.LLM_SECTION = "LLM"
        self.PLAYBOOK_SECTION = "PLAYBOOK"
        self.DATABASE_SECTION = "DATABASE"
        self.MEMORY_SECTION = "MEMORY"
        self.EVAL_SECTION = "EVAL"
        self.MONITORING_SECTION = "MONITORING"
        self.props.read(self.config_path, encoding='utf-8')

    def _ensure_dir(self, path : Union[str, os.PathLike]):
        if not os.path.exists(path):
            os.makedirs(path)

    @property
    def get_openai_api_key(self):
        openai_gpt_api_key = self.props[self.LLM_SECTION]['OPENAI_API_KEY']
        return openai_gpt_api_key
    
    @property
    def get_claude_api_key(self):
        claude_api_key = self.props[self.LLM_SECTION]['CLAUDE_API_KEY']
        return claude_api_key

    @property
    def get_gemini_api_key(self):
        gemini_api_key = self.props[self.LLM_SECTION]['GEMINI_API_KEY']
        return gemini_api_key

    @property
    def get_openai_model(self):
        return self.props[self.LLM_SECTION]['OPENAI_MODEL']

    @property
    def get_claude_model(self):
        return self.props[self.LLM_SECTION]['CLAUDE_MODEL']

    @property
    def get_gemini_model(self):
        return self.props[self.LLM_SECTION]['GEMINI_MODEL']
    
    @property
    def get_huggingface_token(self):
        huggingface_token = self.props.get(self.EMBEDDING_SECTION, 'HUGGINGFACE_ACCESS_TOKEN', fallback='')
        return huggingface_token.strip() or None
    
    @property
    def get_language_code(self):
        language_code = self.props[self.DEFAULT_SECTION]['LANGUAGE']
        return language_code
    
    @property
    def get_playbook_config(self):
        playbook_config = self.props[self.PLAYBOOK_SECTION]
        return playbook_config
    
    @property
    def get_memory_config(self):
        memory_config = self.props[self.MEMORY_SECTION]
        return memory_config
    
    @property
    def get_redis_port(self):
        return os.getenv("REDIS_PORT", 6379)
    
    @property
    def get_redis_host(self):
        return os.getenv("REDIS_HOST", 'localhost')
    
    @property
    def get_backend_config(self):
        return int(os.getenv("BACKEND_PORT", 8000))
    
    @property
    def get_eval_config(self):
        if self.EVAL_SECTION not in self.props:
            return None
        eval_config = self.props[self.EVAL_SECTION]
        return eval_config
    
    @property
    def get_database_config(self):
        database_config = self.props[self.DATABASE_SECTION]
        return database_config
    
    @property
    def get_embedding_gpu(self):
        return self.props.getboolean(self.EMBEDDING_SECTION, 'USE_GPU')
    
    @property
    def get_db_dir(self):
        db_dir = os.path.join(self.curdir, '..', self.get_database_config['SQLITE_DB_DIR'])
        self._ensure_dir(db_dir)
        return db_dir
    
    @property
    def get_db_name(self):
        sqlite_db_name = self.get_database_config['SQLITE_DB_NAME']
        return sqlite_db_name
    
    @property
    def get_db_path(self):
        sqlite_db_name = self.get_database_config['SQLITE_DB_NAME']
        sqlite_db_path = os.path.join(self.get_db_dir, sqlite_db_name)
        return sqlite_db_path
    
    @property
    def get_vector_store_dir(self):
        vector_store_dir = os.path.join(self.curdir, '..', self.get_database_config['VECTOR_STORE_DIR'])
        self._ensure_dir(vector_store_dir)
        return vector_store_dir
    
    @property
    def get_vector_store_name(self):
        vector_store_name = self.get_database_config['VECTOR_STORE_NAME']
        return vector_store_name
    
    @property
    def get_vector_store_path(self):
        vector_store_name = self.get_vector_store_name
        vector_store_path = os.path.join(self.get_vector_store_dir, vector_store_name)
        return vector_store_path
    
    @property
    def get_log_dir(self):
        log_dir = os.path.join(self.curdir, '..', 'logs')
        self._ensure_dir(log_dir)
        return log_dir
    
    @property
    def get_eval_dir(self):
        eval_dir = os.path.join(self.curdir, '..', 'evaluation')
        self._ensure_dir(eval_dir)
        return eval_dir
    @property
    def get_figures_dir(self):
        figures_dir = os.path.join(self.get_eval_dir, 'figures')
        self._ensure_dir(figures_dir)
        return figures_dir
    
    @property
    def get_monitoring_enabled(self) -> bool:
        if self.MONITORING_SECTION not in self.props:
            return False
        return self.props.getboolean(self.MONITORING_SECTION, 'MONITOR', fallback=False)
    
    @property
    def get_langsmith_api_key(self):
        if self.MONITORING_SECTION not in self.props:
            return ""
        return self.props.get(self.MONITORING_SECTION, 'LANG_SMITH_API_KEY', fallback='').strip()
    
    @property
    def get_langsmith_project_name(self) -> str:
        if self.MONITORING_SECTION not in self.props:
            return ""
        return self.props.get(self.MONITORING_SECTION, 'LANG_SMITH_PROJECT_NAME', fallback='').strip()
