import torch

if torch.cuda.is_available():
  num_cuda_devices = torch.cuda.device_count()

  print(f"Найдено {num_cuda_devices} CUDA-устройств:")

  # Вывод информации о каждом CUDA-устройстве
  for i in range(num_cuda_devices):
    device_name = torch.cuda.get_device_name(i)
    print(f"Устройство {i}: {device_name}")
else:
  print("GPU is not available!")


