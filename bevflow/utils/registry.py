"""按名字构建的最小注册表。"""


class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, key):
        def deco(fn_or_cls):
            assert key not in self._items, f"{self.name}: duplicate key {key}"
            self._items[key] = fn_or_cls
            return fn_or_cls
        return deco

    def get(self, key):
        assert key in self._items, f"{self.name}: unknown key {key}, have {sorted(self._items)}"
        return self._items[key]

    def keys(self):
        return sorted(self._items)
