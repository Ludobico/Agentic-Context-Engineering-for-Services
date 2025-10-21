import os, sys
root_project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import inspect
from typing import Callable, Any, Type, List, Dict
from utils.highlight import highlight_print
import logging
from datetime import datetime
import pytz

def function_help(func_or_class : Callable[[Any], Any]):
    """
    Get help information and code variables for a function or class
    
    Args:
        func_or_class (Callable[[Any], Any]): Function or class to inspect
        
    Returns:
        None: Prints help information and code variables
    """
    help(func_or_class)
    highlight_print(func_or_class.__code__.co_varnames)

def function_inspect(func_or_class : Callable[[Any], Any]):
    """
    Inspect and analyze the parameter signature of a callable object (function or class).
    
    Args:
        func_or_class (Callable[[Any], Any]): Target function or class to inspect
        
    Returns:
        None: Prints parameter information including name and parameter kind 
        (e.g. POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD)
    """

    parameters = inspect.signature(func_or_class).parameters
    for name, param in parameters.items():
        print(f"Parameter : {name}, kind : {param.kind}")

def class_methods_instpect(cls : Type) -> List[str]:
    """
    Inspect and retrieve all method definitions of a given class object.
    
    Args:
        cls (Type): Class object to inspect
        
    Returns:
        List[str]: List of method names defined in the class
        
    Raises:
        TypeError: If the provided argument is not a class object
        
    Notes:
        Uses inspect.getmembers() with predicate inspect.isfunction to filter class methods.
        Prints method name and defining module for each method found.
    """
    
    methods = []

    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        methods.append(name)
        print(f"Method : {name}, Defined in : {member.__module__}")

def pick_kwargs(source : dict, keys : List[str]) -> dict:
    """
    Pick specific keys from source dictionary and return new dictionary with only those keys.

    Args:
        source (dict): Source dictionary to pick keys from
        keys (List[str]): List of keys to pick from source dictionary
    
    Returns:
        dict: New dictionary containing only the specified keys and their values from source

    Example:
        ```python
        def example_function(**kwargs):
            extract_keys_for_func_a = ["param_a", "param_b", "param_c"]
            extract_keys_for_func_b = ["param_x"]

            # kwargs에서 필요한 파라미터만 선택
            func_a_params = pick_kwargs(kwargs, extract_keys_for_func_a)
            result = func_a(**func_a_params)

            func_b_params = pick_kwargs(kwargs, extract_keys_for_func_b)
            func_b(data=result, **func_b_params)
        ```
    """
    return {key: source[key] for key in keys if key in source}

class LazyFileHandler(logging.FileHandler):
    """최초 로그가 기록될 때만 파일을 생성하는 핸들러"""
    def __init__(self, filename, mode="a", encoding=None, delay=True):
        # delay=True 로 지정해야 즉시 파일을 열지 않음
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)

class Logger:
    """
    A simple logger class for both console and file logging.

    This class provides a straightforward way to log messages to the console
    and optionally to a file. It handles logger initialization, formatting,
    and handler management.

    Args:
        name (str, optional): The name of the logger. Defaults to __name__.
        level (int, optional): The logging level. The logger will handle messages
            with this level and above. Defaults to logging.DEBUG.
            Available levels:
            - logging.CRITICAL (50): For critical errors (highest severity).
            - logging.ERROR (40): For serious errors.
            - logging.WARNING (30): For warnings or unexpected events.
            - logging.INFO (20): For general informational messages.
            - logging.DEBUG (10): For detailed debugging information (lowest severity).
        save_to_file (bool, optional): If True, logs will be saved to a file.
            Defaults to False.
        log_dir (str, optional): The directory where log files will be stored.
            Only used if save_to_file is True. Defaults to "logs".
        log_file (str, optional): The name of the log file.
            Only used if save_to_file is True. Defaults to "app.log".

    Usage:
        # 1. Basic console logging
        logger = Logger(__name__)
        logger.info("This is an info message.")
        logger.debug("This is a debug message.")

        # 2. Logging to a file
        file_logger = Logger('file_logger', save_to_file=True, log_dir='my_logs', log_file='my_app.log')
        file_logger.warning("This message will be saved in my_logs/my_app.log")
    """
    def __init__(self,
                 name=__name__,
                 level=logging.DEBUG,
                 save_to_file=False,
                 log_dir = "logs",
                 log_file="app.log",
                 console_output = True
                 ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False # 중복 로그 방지

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s") # 날짜 + 로그레벨 + 메시지
        formatter.converter = self._kst_time

        if not self.logger.handlers:
            # 콘솔 핸들러
            if console_output:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            # 파일 저장 핸들러 (lazy 생성)
            if save_to_file:
                save_path = os.path.join(root_project, log_dir)
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, log_file)

                file_handler = LazyFileHandler(file_path, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        self.save_to_file = save_to_file

    def _kst_time(*args):
        return datetime.now(pytz.timezone("Asia/Seoul")).timetuple()

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def is_saving_to_file(self):
        return self.save_to_file