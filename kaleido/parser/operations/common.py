from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

import kaleido
from kaleido.parser.errors import UnknownPrimitiveOps
import warnings

__all__ = [
    'registers',
    'import_all_modules_for_register',
]


class OpRegister:
    """primitive tensor operations."""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def _norm_key(self, key: str) -> str:
        # key is normalized into lower cases.
        return key.lower()

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__

        key = self._norm_key(key)
        if key in self._dict:
            warnings.warning(
                "Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            return decorator(None, param)
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        key = self._norm_key(key)
        try:
            return self._dict[key]
        except KeyError as e:
            raise UnknownPrimitiveOps(
                f"unknown primitive operation: {str(e)}.")

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


class registers():
    """All module registers."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    tensor_primitives = OpRegister('tensor_primitives')
    access = OpRegister('access_pattern')


ALL_OPS = [
    ("kaleido.parser.operations", "tensor_primitives"),
    ("kaleido.parser.operations", "access_patterns"),
]


def import_all_modules_for_register():
    errors = []
    for base_dir, module in ALL_OPS:
        try:
            full_name = base_dir + "." + module
            importlib.import_module(full_name)
        except ImportError as error:
            errors.append((name, error))

    for name, err in errors:
        warnings.warn("Fail to import module {}: {}.".format(name, err))
