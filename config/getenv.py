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
    
    