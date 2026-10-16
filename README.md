# Physical Intelligence：六关节机械臂 VLA 开发栈

面向 Hiwonder 六关节机械臂的硬件隔离型视觉-语言-动作（VLA）软件栈。项目当前重点是建立可重复的接口、任务、数据与策略评估流程：默认运行在确定性的模拟机械臂上，可通过 RPC 接口连接独立服务；LeRobot 插件和 ACT / SmolVLA 适配器提供数据及模型集成入口。

> **当前验证边界：** mock、RPC 合约、LeRobot 插件 / 数据集，以及 ACT 和 SmolVLA 的有限模型冒烟流程已在项目记录中验证。真实机械臂、相机、标定、物理安全范围、真实数据训练和 sim-to-real 均未由这些验证证明。模型冒烟结果不代表任务能力，也不能直接下发到机械臂。

## 项目能力

- 统一的六关节观测、动作、任务和健康状态数据契约。
- `MockBackend` 确定性模拟环境，以及遵循 HTTP RPC 合约的 `RpcBackend` / 本地 mock 服务。
- 所有动作经过安全控制器：检查关节包络、单步变化、观测新鲜度和命令看门狗；急停和故障路径有测试覆盖。
- `red_to_blue` 示例任务，使用合成 RGB 图像和 mock 策略验证闭环软件流程。
- LeRobot 0.6.1 机器人插件、LeRobotDataset v3 mock 录制，以及 ACT / SmolVLA 可选集成。
- 报告、硬件接入清单和配置样例，用于记录验证证据及后续门槛。

## 架构

```text
CLI / Task loop
      │
Policy adapter ── Observation / Action contracts
      │                         │
      └──────── SafetyController┘
                    │
            RobotBackend interface
             ┌──────┴──────┐
         MockBackend   RpcBackend ── HTTP RPC service
```

应用包不依赖 `Arm_Lib`。硬件驱动应隔离在后端服务中；本仓库的 RPC 客户端不会绕过安全控制器。`configs/real.example.yaml` 是未来部署参数示例，不是已完成的硬件驱动配置。当前命令行主要通过参数选择后端，不能假定所有 YAML 字段会自动加载。

## 环境与安装

核心开发需要 Python 3.12。GPU 和模型依赖是可选项；安装 PyTorch 时，请先根据操作系统、驱动和目标设备，从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择命令。依赖清单不固定 CUDA wheel 地址。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 可选：先安装与你的驱动匹配的 PyTorch，再安装模型集成
python -m pip install <PyTorch 官方选择器给出的命令>
python -m pip install -e ".[smolvla]"
python -m pip install -e .\lerobot_robot_hiwonder_arm
```

LeRobot 插件固定使用 `lerobot==0.6.1`，其环境要求 Python 3.12。CPU-only 或不需要模型时，可跳过 PyTorch 和 `[smolvla]` 安装。`uv.lock` 用于依赖解析复现；如安装 / 更新依赖后，请运行 `uv lock --check` 检查锁文件与项目元数据一致。

## 快速开始：全程模拟

```powershell
# 确定性模拟任务（无需硬件、模型或网络）
robot-vla run --backend mock --policy mock --task red_to_blue

# 查看一次模拟观测
robot-vla inspect --backend mock --once

# 录制 LeRobotDataset v3 模拟数据（需要可选 LeRobot 依赖）
robot-vla record-mock --episodes 3 --output datasets/mock/red_to_blue

# 在终端 A 启动本地 RPC mock 服务
robot-vla mock-server

# 在终端 B 通过 RPC 合约执行同一模拟任务
robot-vla run --backend rpc --policy mock --task red_to_blue \
  --base-url http://127.0.0.1:8765
```

mock 轨迹和数据集是软件集成样本，数据会标记为 `simulation_mock`，并明确禁止作为真实策略训练数据。命令产物默认写入 `logs/` 或 `datasets/`，这些运行数据不会提交到仓库。

## 模型集成与评估边界

模型命令显式 opt-in，可能下载 checkpoint 并需要 GPU：

```powershell
robot-vla model-smoke-test --policy smolvla --device cuda
robot-vla model-smoke-test --policy act --device cuda
robot-vla benchmark --policy smolvla --device cuda --episodes 3
robot-vla benchmark --policy act --checkpoint PATH_TO_TRAINED_ACT --device cuda --episodes 3
```

SmolVLA 默认加载公开的 `lerobot/smolvla_base` checkpoint。ACT 冒烟可在合成数据上执行一次训练步骤并检查输出形状；ACT benchmark 必须指定训练好的 checkpoint。`benchmark` 在固定 mock 场景中运行，输出成功率、步数、安全拒绝数和推理延迟；它不测量真实机械臂表现。`run` 目前只支持闭环 mock 策略；学习策略的 rollout 尚未接入该命令。

## 配置文件说明

`configs/` 中的 YAML 是经审阅的参数样例，供开发和后续配置加载器使用。当前 CLI 的主要参数仍来自命令行和代码默认值，修改 YAML 不会自动改变 CLI 行为。

| 文件 | 用途 | 重要边界 |
|---|---|---|
| `mock.yaml` | mock 运行参数样例 | 只用于模拟 |
| `real.example.yaml` | RPC 部署参数模板 | 无真实地址、标定文件或已验证机械范围；默认拒绝真实动作 |
| `joints.mock.yaml` | 六关节软件包络示例 | `driver_envelope_only`，不是机械限位或零位标定 |
| `tasks.yaml` | `red_to_blue` 文本和别名样例 | 示例任务定义，不代表实机视觉验证 |
| `act.yaml` | ACT 输入形状和评估设置 | checkpoint 为空；mock 数据不得用于最终真实策略训练 |
| `smolvla.yaml` | SmolVLA checkpoint 和输入形状样例 | 单相机合成输入不能证明实机相机布局兼容 |

提交前请勿把私有 checkpoint、真实轨迹、摄像头图像、标定记录、访问令牌或个人设备地址放进版本库。请使用本地私有配置保存部署值。

## 安全与硬件边界

- 默认开发、测试和示例均使用 `MockBackend`；本项目文档中的硬件连接清单是未来步骤，不是已执行记录。
- 真实 RPC 动作需要显式 real mode、存在的标定文件和 `physical_calibration_verified: true`。这些字段必须对应真实测量和人工复核，不能为了让命令运行而填写。
- mock 关节范围是软件驱动包络，不代表机械安全边界、零位或可用行程。
- 接入硬件前，按 `docs/hardware-checklist.md` 从网络和只读健康检查逐步推进到单关节小幅动作；反馈不稳定时停止动作升级。
- 未完成物理标定和监督测试前，不要把任何模型输出送往真实机械臂。

## 验证与项目状态

详细的历史验收范围见 [`docs/acceptance-status.md`](docs/acceptance-status.md)，平台依赖快照见 [`docs/environment-report.md`](docs/environment-report.md)。该记录报告了 25 项常规测试通过、1 项可选 ACT 模型冒烟通过，以及 mock 任务 45 步成功；这是记录中的环境结果，不保证其他机器当前状态相同。SmolVLA / ACT 检查只覆盖模型加载和张量接口，不覆盖任务成功能力。

本机重新验证的常规开发命令：

```powershell
ruff check .
pytest
uv lock --check
python -m pip check
```

pytest 默认排除需要 checkpoint / GPU 的 `model` 标记测试。运行模型测试前请阅读对应测试和依赖说明，并显式指定 `-m model`。

## 目录结构

```text
src/robot_vla/                  核心契约、后端、安全、任务、CLI、策略适配
lerobot_robot_hiwonder_arm/     LeRobot 自定义机器人插件
configs/                        参数与任务 YAML 样例
tests/                          单元、合约和集成测试
docs/                           验收证据、环境快照和硬件接入清单
```

## 许可

本项目按仓库根目录的 MIT License 发布。LeRobot、PyTorch、模型 checkpoint 和其他依赖仍分别适用其各自许可条款。
