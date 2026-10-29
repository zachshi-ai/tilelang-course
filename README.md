# TileLang 图解课 · 扎克的自留地

用**原创场景 + 动画**把 [TileLang](https://github.com/tile-ai/tilelang) 的每个知识点讲透。
每个知识点一个视频；每个视频配一个**单文件可跑代码**，就在本仓库对应目录里。

## 视频对应代码

| 集数 | 视频 | 代码 |
|---|---|---|
| EP01 | 十秒，发完一百万个红包（GPU 运行模型：切段分组 / 编号 / 守卫） | [`ep01/ep01_hongbao_demo.py`](ep01/ep01_hongbao_demo.py) |
| EP02 | 座位表压成一条线（flatten 与整除取余） | 制作中 |

## 怎么跑（EP01 为例）

```bash
pip install tilelang torch
python ep01_hongbao_demo.py        # Mac 自动走 Metal；NVIDIA 显卡把 device 改成 "cuda"
```

跑完打印 `ALL PASS`（自带三道保险自测：对样例 / 对大账本 / 原账本没动过）。

## 视频在哪

抖音 / B 站搜：**扎克的自留地**
