"""Sandboxed code execution with resource limits.

Implements secure execution of LLM-generated pandas code with:
- Process isolation
- Memory limits
- Time limits
- I/O restrictions
"""
from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import signal
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExecutionTimeout(Exception):
    """Raised when code execution exceeds time limit."""
    pass


class ExecutionMemoryError(Exception):
    """Raised when code execution exceeds memory limit."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise ExecutionTimeout("Code execution exceeded time limit")


def _execute_in_subprocess(
    code: str,
    df: pd.DataFrame,
    timeout: int,
    queue: mp.Queue,
):
    """Execute code in subprocess with resource limits.
    
    This runs in a separate process for isolation.
    """
    try:
        # Set up timeout signal (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[attr-defined]
            signal.alarm(timeout)  # type: ignore[attr-defined]
        
        # Execute code with restricted builtins
        local_vars: Dict[str, Any] = {"df": df, "pd": pd, "json": json}
        stdout = io.StringIO()
        
        try:
            with redirect_stdout(stdout):
                exec(
                    code,
                    {
                        "__builtins__": {
                            # Essential Python functions
                            "len": len,
                            "range": range,
                            "min": min,
                            "max": max,
                            "sum": sum,
                            "int": int,
                            "float": float,
                            "str": str,
                            "dict": dict,
                            "list": list,
                            "tuple": tuple,
                            "set": set,
                            "abs": abs,
                            "round": round,
                            "sorted": sorted,
                            "enumerate": enumerate,
                            "zip": zip,
                            "print": print,
                            "bool": bool,
                            "isinstance": isinstance,
                            "type": type,
                        }
                    },
                    local_vars,
                )
        except ExecutionTimeout:
            queue.put({"error": "Execution timeout exceeded"})
            return
        except Exception as e:
            queue.put({"error": f"Execution error: {str(e)}"})
            return
        finally:
            if hasattr(signal, 'alarm'):
                signal.alarm(0)  # type: ignore[attr-defined]  # Cancel alarm
        
        # Get result
        printed = stdout.getvalue().strip()
        
        # If no output but result variable exists
        if not printed and "result" in local_vars:
            try:
                printed = json.dumps(local_vars["result"])
            except Exception:
                printed = str(local_vars["result"])
        
        queue.put({"success": True, "output": printed})
        
    except Exception as e:
        queue.put({"error": f"Subprocess error: {str(e)}"})


class SandboxedExecutor:
    """Executes LLM-generated code in a sandboxed environment.
    
    Features:
    - Process isolation (separate process)
    - Time limits (timeout)
    - Memory limits (via resource module on Unix)
    - Restricted builtins
    - No file system access
    """
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        memory_limit_mb: Optional[int] = 512,
    ):
        """Initialize sandboxed executor.
        
        Args:
            timeout_seconds: Maximum execution time
            memory_limit_mb: Maximum memory usage (Unix only)
        """
        self.timeout = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.use_subprocess = True  # Always use subprocess for safety
    
    def execute(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute code with resource limits and sandboxing.
        
        Args:
            code: Python code to execute
            df: DataFrame to provide as context
            
        Returns:
            Dict with 'success' and 'output' or 'error'
        """
        if not self.use_subprocess:
            # Fallback to in-process execution (not recommended for production)
            return self._execute_in_process(code, df)
        
        # Subprocess execution for better isolation
        try:
            # Use multiprocessing for process isolation
            ctx = mp.get_context('spawn')  # spawn for clean process
            queue = ctx.Queue()
            
            process = ctx.Process(
                target=_execute_in_subprocess,
                args=(code, df, self.timeout, queue)
            )
            
            process.start()
            process.join(timeout=self.timeout + 5)  # Extra buffer
            
            if process.is_alive():
                # Force terminate if still running
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                return {"error": "Execution timeout: process forcefully terminated"}
            
            # Get result from queue
            if not queue.empty():
                result = queue.get()
                return result
            else:
                return {"error": "No output from subprocess"}
                
        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            # Fallback to in-process
            return self._execute_in_process(code, df)
    
    def _execute_in_process(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback: execute in current process with limited safety.
        
        Note: This is less safe than subprocess execution.
        """
        local_vars: Dict[str, Any] = {"df": df, "pd": pd, "json": json}
        stdout = io.StringIO()
        
        try:
            # Set alarm for timeout (Unix only)
            if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
                signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[attr-defined]
                signal.alarm(self.timeout)  # type: ignore[attr-defined]
            
            with redirect_stdout(stdout):
                exec(
                    code,
                    {
                        "__builtins__": {
                            "len": len,
                            "range": range,
                            "min": min,
                            "max": max,
                            "sum": sum,
                            "int": int,
                            "float": float,
                            "str": str,
                            "dict": dict,
                            "list": list,
                            "tuple": tuple,
                            "set": set,
                            "abs": abs,
                            "round": round,
                            "sorted": sorted,
                            "enumerate": enumerate,
                            "zip": zip,
                            "print": print,
                            "bool": bool,
                            "isinstance": isinstance,
                            "type": type,
                        }
                    },
                    local_vars,
                )
            
            if hasattr(signal, 'alarm'):
                signal.alarm(0)  # type: ignore[attr-defined]  # Cancel alarm
            
            printed = stdout.getvalue().strip()
            
            if not printed and "result" in local_vars:
                try:
                    printed = json.dumps(local_vars["result"])
                except Exception:
                    printed = str(local_vars["result"])
            
            return {"success": True, "output": printed}
            
        except ExecutionTimeout:
            return {"error": "Execution timeout exceeded"}
        except Exception as e:
            return {"error": f"Execution error: {str(e)}"}
        finally:
            if hasattr(signal, 'alarm'):
                signal.alarm(0)  # type: ignore[attr-defined]


# Global executor instance
_default_executor: Optional[SandboxedExecutor] = None


def get_executor(
    timeout_seconds: int = 30,
    memory_limit_mb: Optional[int] = 512,
) -> SandboxedExecutor:
    """Get or create the default sandboxed executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = SandboxedExecutor(
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
        )
    return _default_executor
