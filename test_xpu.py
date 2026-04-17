"""
test_xpu.py - 验证 Intel XPU (GPU) 是否可用
"""
import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch version: {torch.__version__}")
print(f"IPEX version: {ipex.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")

if torch.xpu.is_available():
    print(f"XPU device count: {torch.xpu.device_count()}")
    print(f"XPU device name: {torch.xpu.get_device_name(0)}")
    # 简单张量运算测试
    a = torch.ones(3, 3).to("xpu")
    b = torch.ones(3, 3).to("xpu")
    c = a + b
    print(f"Tensor on XPU: {c}")
    print("✅ Intel GPU 测试通过")
else:
    print("⚠️ XPU 不可用，将回退到 CPU")
