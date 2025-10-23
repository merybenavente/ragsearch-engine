"""Logging configuration for the vector database application"""

import structlog
import logging
import sys
from typing import Any, Dict
from enum import Enum


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    json_format: bool = False,
    include_request_id: bool = True
) -> None:
    """
    Configure structured logging for the application
    
    Args:
        level: Logging level
        json_format: Whether to use JSON format for logs
        include_request_id: Whether to include request IDs in logs
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.value),
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.value)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class"""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """
    Log a function call with parameters
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger("function_calls")
    logger.info(
        "Function called",
        function=func_name,
        parameters=kwargs
    )


def log_performance(operation: str, duration_ms: float, **context: Any) -> None:
    """
    Log performance metrics
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **context: Additional context
    """
    logger = get_logger("performance")
    logger.info(
        "Performance metric",
        operation=operation,
        duration_ms=duration_ms,
        **context
    )


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an error with context
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    logger = get_logger("errors")
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {}
    )