import sys

class Mock:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    @classmethod
    def __getattr__(cls, name):
        return Mock()

# List all modules that should be mocked (fake imports)
MOCK_MODULES = ["mpi4py"]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)