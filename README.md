# audio-multiband-split

## 题面

母带处理加分频压缩后，分频点附近出现凹陷、重组合还有梳状感：分频器用 Butterworth 级联代替 Linkwitz-Riley，分频点处出现约 -6 dB 凹陷；分频点不校验递增，频段重叠；各段相加不做全通补偿，重组合出现梳状滤波；阶数允许奇数，LR 公式失效；逐段增益直接相乘，未按 dB 换算；增益变化不平滑，切换时爆音；分频点超出奈奎斯特不报错，滤波器发散；重组合不补偿延迟，高频提前；批量共用滤波器状态，轨道之间串扰；sample_rate=0 不校验，触发除零。multiband/crossover.py 是核心，tests/test_crossover.py 15个红。要修的是：实现正确的 Linkwitz-Riley 分频、严格配置校验、全通与延迟补偿、dB 增益换算与平滑包络、批量状态隔离以及除零防护，并保持各频段求和可无损还原原信号。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始失败环境必须恰好输出 `15 failed`，且不能出现 collection error、import error 或 syntax error。修复完成后，同一命令应输出 `15 passed`。测试文件 `tests/test_crossover.py` 是验收契约，不可修改；也不得通过改动测试、跳过测试或联网安装依赖来改变结果。实现只允许使用 Python 标准库（`math`、`cmath`）和 pytest，必须离线、无随机、不 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 文件与约束

- `multiband/crossover.py`：8 个公开函数，当前全部抛出 `NotImplementedError`，docstring 描述完整契约。
- `tests/test_crossover.py`：15 个测试，覆盖 10 个缺陷以及 5 个组合场景，全部调用上述桩。
- `conftest.py`：把仓库根目录加入 `sys.path`，保证测试可导入 `multiband`。
- `.gitignore`：忽略 Python/pytest 缓存。
- 音频统一使用 `list[float]`；分频点单位为 Hz；所有时域重建与增益比较的绝对容差为 `1e-6`。
