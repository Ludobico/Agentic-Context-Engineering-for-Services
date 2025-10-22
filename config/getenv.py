import os
from typing import Union
import configparser

class GetEnv:
    def __init__(self):
        self.curdir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.abspath(os.path.join(self.curdir, 'config.ini'))
        
        if not os.path.exists(self.config_path):
            self.config_path = os.path.abspath(os.path.join(self.curdir, 'config-example.ini'))
            if not os.path.exists(self.config_path):
                raise FileNotFoundError("Neither `config.ini` nor `config-example.ini` were found in the config directory.")
        
        self.props = configparser.ConfigParser()
        self.DEFAULT_SECTION = "DEFAULT"
        self.PLAYBOOK_SECTION = "PLAYBOOK"
        self.DATABASE_SECTION = "DATABASE"
        self.CELERY_SECTION = "CELERY"
        self.props.read(self.config_path, encoding='utf-8')

    def _ensure_dir(self, path : Union[str, os.PathLike]):
        if not os.path.exists(path):
            os.makedirs(path)

    @property
    def get_openai_api_key(self):
        openai_gpt_api_key = self.props[self.DEFAULT_SECTION]['OPENAI_GPT_API_KEY']
        return openai_gpt_api_key
    
    @property
    def get_huggingface_token(self):
        huggingface_token = self.props[self.DEFAULT_SECTION]['HUGGINGFACE_ACCESS_TOKEN']
        return huggingface_token
    
    @property
    def get_language_code(self):
        language_code = self.props[self.DEFAULT_SECTION]['LANGUAGE']
        return language_code
    
    @property
    def get_playbook_config(self):
        playbook_config = self.props[self.PLAYBOOK_SECTION]
        return playbook_config
    
    @property
    def get_database_config(self):
        database_config = self.props[self.DATABASE_SECTION]
        return database_config
    
    @property
    def get_celery_config(self):
        celery_config = self.props[self.CELERY_SECTION]
        return celery_config
    
    @property
    def get_celery_broker_url(self):
        redis_host = self.get_celery_config['REDIS_HOST']
        redis_port = self.get_celery_config['REDIS_PORT']
        celery_redis_db = self.get_celery_config['CELERY_REDIS_DB']

        celery_broker_url = f"redis://{redis_host}:{redis_port}/{celery_redis_db}"
        return celery_broker_url
    
    @property
    def get_celery_result_backend(self):
        redis_host = self.get_celery_config['REDIS_HOST']
        redis_port = self.get_celery_config['REDIS_PORT']
        celery_redis_db = self.get_celery_config['CELERY_REDIS_DB']

        celery_result_backend = f"redis://{redis_host}:{redis_port}/{celery_redis_db}"
        return celery_result_backend
    
    @property
    def get_db_dir(self):
        db_dir = os.path.join(self.curdir, '..', self.get_database_config['SQLITE_DB_DIR'])
        self._ensure_dir(db_dir)
        return db_dir
    
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
    def get_vector_store_path(self):
        vector_store_name = self.get_database_config['VECTOR_STORE_NAME']
        vector_store_path = os.path.join(self.get_vector_store_dir, vector_store_name)
        return vector_store_path
    
    
