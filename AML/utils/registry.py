from typing import Optional


class Registry:
    def __init__(self, lib: str):
        """_summary_

        Args:
            lib (str): which lib? Model | Metric | Loss
        """
        self._lib = lib
        self._registry = {}

    def register(self, name: Optional[str] = None):
        def decorator(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise KeyError(
                    f"{self._lib.capitalize()} '{key}' is already registered."
                )
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, name):
        cls = self._registry.get(name)
        if cls is None:
            raise KeyError(
                f"{self._lib.capitalize()} '{name}' not found in the registry."
            )
        return cls

    def list_keys(self):
        return list(self._registry.keys())
