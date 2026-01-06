"""Compatibility shim so notebooks can import `from model import ...`.
This simply re-exports symbols from `src.model` which is the actual implementation.
"""
try:
    from src.model import CrossModalStoryModelTF  # type: ignore

    __all__ = ["CrossModalStoryModelTF"]
except Exception as e:
    raise ImportError("Could not import `CrossModalStoryModelTF` from `src.model`. Ensure `src/model.py` exists and is importable.") from e
