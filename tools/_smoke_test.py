import sys
import os
print(f"python={sys.version}")
print(f"executable={sys.executable}")
print(f"cwd={os.getcwd()}")

try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}")
except ImportError as e:
    print(f"TORCH_IMPORT_ERROR: {e}")

try:
    import numpy as np
    print(f"numpy={np.__version__}")
except ImportError as e:
    print(f"NUMPY_IMPORT_ERROR: {e}")

print("SMOKE_DONE")
