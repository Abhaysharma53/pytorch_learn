import numpy as np
import torch
#print(np.__version__)
print(torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. CPU will be used.")