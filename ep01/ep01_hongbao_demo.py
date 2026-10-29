#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨年红包 · 你的第一个 GPU 程序（视频 EP01 配套代码）

视频里做的事，全部在这一个文件里：
  1048583 个账户，每个账户余额 +5 元；加完还是负数的账户按 0 记。
  用 GPU 上万个工人同时动手，十秒内发完全部红包。

怎么跑（二选一）：

  A. Mac 直跑（推荐，无需 GPU）：
     pip install tilelang torch
     python ep01_hongbao_demo.py            # 自动用 Metal 后端真编译真运行

  B. 有 NVIDIA GPU 的机器：
     把下面 device = "mps" 改成 device = "cuda" 再跑。

视频对应关系：
  切段分组   -> T.Kernel(T.ceildiv(N, block), threads=block)
  编号公式   -> idx = bx * block + i          （组号 × 256 + 组内编号）
  守卫       -> if idx < N:                   （编号小于总数才动手）
  真正的活   -> C[idx] = T.max(A[idx]+B[idx], T.float32(0))
  三道保险   -> 文件末尾的 self_test()
"""

import torch
import tilelang
import tilelang.language as T

device = "mps"  # Mac 用 "mps"（Metal），NVIDIA 显卡改成 "cuda"

_cache = {}  # 缓存字典：钥匙 = 账户总数 N（同规模只翻译一次）


def hongbao(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """给 A 里每个账户加 B 里对应的钱，负数按 0 记，返回新账本 C。"""
    N = int(A.shape[0])
    if N not in _cache:
        block = 256  # 每组 256 个工人

        @T.prim_func
        def kernel(
            A: T.Tensor((N,), T.float32),  # 账本（原余额）
            B: T.Tensor((N,), T.float32),  # 红包（每人 5 元）
            C: T.Tensor((N,), T.float32),  # 新账本（发完后）
        ):
            with T.Kernel(T.ceildiv(N, block), threads=block) as bx:  # 切 4097 组
                for i in T.Parallel(block):        # 分工表：一人认领一个 i
                    idx = bx * block + i           # 编号公式：组号×256+组内编号
                    if idx < N:                    # 守卫：编号超出总数就别动手
                        C[idx] = T.max(A[idx] + B[idx], T.float32(0))  # 加完跟 0 比，谁大取谁

        _cache[N] = tilelang.compile(kernel)

    C = torch.empty(N, dtype=torch.float32, device=A.device)  # 全新账本，绝不改原账本
    _cache[N](A, B, C)
    return C


def self_test():
    """三道保险：对样例 / 对大账本 / 原账本没动过。全过打印 ALL PASS。"""
    # 保险一：借 4 个测试账户，手工算好标准答案
    A = torch.tensor([-3.0, 2.0, 0.0, 4.0], device=device)
    B = torch.tensor([5.0, 5.0, 5.0, 5.0], device=device)
    A0, B0 = A.clone(), B.clone()
    C = hongbao(A, B)
    assert torch.equal(C.cpu(), torch.tensor([2.0, 7.0, 5.0, 9.0]).cpu()), C
    # 保险三（顺路查）：原账本一个字都没变
    assert torch.equal(A, A0) and torch.equal(B, B0), "原账本被改了！"

    # 保险二：一百万个账户，跟现成软件对答案（专测最后那一段）
    N = 1_048_583  # = 256 × 4096 + 7，故意不凑整
    gen = torch.Generator().manual_seed(20260101)
    A = (torch.randn(N, generator=gen) * 10).to(device)
    B = torch.full((N,), 5.0, device=device)
    A0 = A.clone()
    C = hongbao(A, B)
    expect = torch.clamp(A0.cpu() + B.cpu(), min=0.0)
    assert torch.allclose(C.cpu(), expect), "大账本对不上！"
    assert torch.equal(A, A0), "原账本被改了！"

    print("ALL PASS")


if __name__ == "__main__":
    self_test()
