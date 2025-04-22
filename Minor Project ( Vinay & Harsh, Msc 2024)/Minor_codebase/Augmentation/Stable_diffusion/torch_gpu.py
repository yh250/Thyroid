import torch
if torch.cuda.is_available():
    print('true')
else: print('false')