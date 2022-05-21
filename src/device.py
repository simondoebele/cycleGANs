import torch


is_cuda_available = torch.cuda.is_available()
print("Is CUDA available? {}".format(is_cuda_available))
if is_cuda_available:
    print("Current device: {}".format(torch.cuda.get_device_name(0)))
else:
    print('Running on CPU')
print()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'