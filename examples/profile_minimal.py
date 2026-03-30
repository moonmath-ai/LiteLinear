import torch
import lite_linear._cuda as _cuda_ext


def main():
    device = torch.device("cuda")

    # All real configs from bench_ffn.py: (name, M, K, N)
    configs = [
        ("w1_small", 1400, 4096, 16384),
        ("w1_large", 5600, 4096, 16384),
        ("w2_small", 1400, 16384, 4096),
        ("w2_large", 5600, 16384, 4096),
    ]
    R = 32

    # Pre-allocate all tensors
    tensors = []
    for name, M, K, N in configs:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        A = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.1
        B = torch.randn(R, K, device=device, dtype=torch.bfloat16) * 0.1
        Q_fp8 = (torch.randn(N, K, device=device) * 0.1).to(torch.float8_e4m3fn)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.01
        tensors.append((name, M, K, N, x, A, B, Q_fp8, bias))

    # Warmup all configs first (4 launches to skip)
    print("Warmup phase...")
    for name, M, K, N, x, A, B, Q_fp8, bias in tensors:
        _cuda_ext.fused_forward(x, Q_fp8, A, B, bias, 1.0)
    torch.cuda.synchronize()
    print("Warmup done.")

    # Profile runs (4 launches to profile)
    print("\nProfile phase...")
    for name, M, K, N, x, A, B, Q_fp8, bias in tensors:
        print(f"  {name}: M={M}, K={K}, N={N}")
        _cuda_ext.fused_forward(x, Q_fp8, A, B, bias, 1.0)
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
