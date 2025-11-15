import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

class RobotCustomLogger:

    
    _instance: Optional['RobotCustomLogger'] = None
    _default_filename = "custom_logger.log"
    _icons = {
        'info': 'ℹ️',
        'error': '🚨',
        'success': '✅',
        'debug': '🐛',
        'warning': '⚠️',
        'separator': '--------------------------------',
        'brain': '🧠',
        'start': '🚀'
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger("CustomLogger")
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False
            self._log_path: Optional[str] = None
            self._handler: Optional[logging.FileHandler] = None
            self._initialized = True

    def ensure_handler(self):
        if not self._handler:
            self._log_path = self._resolve_log_path()
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            
            self._handler = logging.FileHandler(self._log_path)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._handler.setFormatter(formatter)
            self.logger.addHandler(self._handler)

    def _resolve_log_path(self) -> str:
        try:
            from robot.libraries.BuiltIn import BuiltIn
            output_dir = BuiltIn().get_variable_value("${OUTPUT_DIR}")
            if output_dir:
                return os.path.join(output_dir, self._default_filename)
        except Exception:
            pass

        candidates = [
            os.environ.get("CI_LOG_DIR"),
            os.path.join(os.getcwd(), "logs"),
            tempfile.gettempdir()
        ]

        for base_dir in candidates:
            if not base_dir:
                continue
                
            try:
                test_path = os.path.join(base_dir, self._default_filename)
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                with open(test_path, 'a'):
                    pass
                return test_path
            except Exception as e:
                continue

        return os.path.join(tempfile.gettempdir(), self._default_filename)

    def info(self, message: str, robot_log: bool = False):
        self._log('info', message, robot_log)

    def error(self, message: str, robot_log: bool = False):
        self._log('error', message, robot_log)

    def success(self, message: str, robot_log: bool = False):
        self._log('success', message, robot_log)

    def debug(self, message: str, robot_log: bool = False):
        self._log('debug', message, robot_log)

    def warning(self, message: str, robot_log: bool = False):
        self._log('warning', message, robot_log)

    def _log(self, level: str, message: str, robot_log: bool):
        self.ensure_handler()
        icon = self._icons.get(level, '')
        full_message = f"{icon} {message}" if icon else message

        # Map 'success' to 'info' for standard Python logger
        log_level = level if level != 'success' else 'info'
        getattr(self.logger, log_level)(full_message)

        if robot_log:
            self._robot_console_log(level, message)

    def _robot_console_log(self, level: str, message: str):
        try:
            from robot.api import logger
            robot_level = level if level != 'success' else 'info'
            getattr(logger, robot_level)(message)
        except ImportError:
            pass

    @property
    def log_path(self) -> str:
        if not self._log_path:
            self.ensure_handler()
        return str(self._log_path)
    