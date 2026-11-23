# MAS-PromptBeach-Pro 评测改进与使用方法

MAS-PromptBeach-Pro 基于 Juyang Bai 和 Laixi Shi 的 [MAS-PromptBench](https://github.com/juyangbai/MAS-PromptBench)。[项目 README](../README.md#changes-in-mas-promptbeach-pro) 列出了新增文件及其用途。

本次把此前 WSL 终端中的临时修补移进了项目。任务流程改造以本轮使用的中央式 LangGraph 入口以及 API-Bank、ToolHop 公共路径为主；其他框架也会复用公共配置和数据加载，但并未逐一运行完整评测。主要入口仍是 `python -m ...`，无需再粘贴覆盖函数的长脚本。没有更改模型生成的答案、代码或补丁来提高分数。

## 公共模块现在分别在哪里

| 文件 | 负责什么 | 为什么放在公共层 |
| --- | --- | --- |
| `core/settings.py`、`core/profiles.py` | 读取环境变量、保存模型配置、保留已验证配置 | 换任务时复用配置，换模型时使用另一份配置 |
| `core/providers.py`、`core/llm.py` | 根据接入点生成参数，在最终请求中适配 token 上限字段 | LangChain 可能重命名字段，仅修改初始变量无法保证发出的字段正确 |
| `core/data.py` | 优先本地数据、HF 缓存，再下载 | 多个模型共用数据缓存，避免把相对文件名当成当前目录中的文件 |
| `core/preflight.py` | 检查 NumPy、PyArrow、datasets 的导入与数据转换 | 在调用模型前发现环境兼容问题，并提供明确的安装版本 |
| `core/checkpoints.py`、`core/batch.py` | 保存生成产物、运行配置、逐题记录和失败分类，支持续跑 | 评分可以重做，已经生成的回答不必重新收费生成 |
| `core/runtime.py` | 工具调用历史规范化、错误 ToolMessage、结束原因记录 | GPQA、BFCL 等 LangGraph 任务遵守同一套工具消息协议 |

“环境检查”现在有了明确实现，即 `core/preflight.py`，不是一个独立安装的软件。

## 1. 保存和复用模型配置

正常批处理会按“模型名 + 接入点”自动记住最近使用的配置，文件名以 `auto_` 开头。用 `python -m core.profiles list` 可以查看，下次可将该名字赋给 `MAS_MODEL_PROFILE`。自动记住不代表验证通过，之前确认过的 `verified_settings` 会单独保留。

正常生成成功后，还会单独保存 `successful_settings`（包括新接入的兼容模型），不要求答案正确。请求异常或用户之后尝试的失败配置不会覆盖上次成功配置；评分失败仍可保留已经成功生成时的参数。程序不会为此发送额外请求、自动尝试参数或自动重试接入。

下次明确设置相同的 `MODEL_ID` 和 `VLLM_BASE_URL` 时，未显式设置的模型参数会优先读取这份成功配置；当前环境变量和明确选定的配置文件仍优先。换模型或服务地址不会复用另一组连接的成功配置。也可以显式选择 `MAS_MODEL_PROFILE=auto_名称@successful`。保存位置仍为下面的模型配置目录，API Key 不会写入。

这里的“成功”只表示正常生成完成，不代表输出上限、思考模式或工具调用能力已经经过专门验证；`@verified` 仍由用户在 smoke 检查后手动确认。

在已经 export 好千问配置的 WSL 窗口中运行：

```bash
python -m core.profiles save qwen35_27b
```

文件默认保存在 `~/.config/mas-promptbench/models/qwen35_27b.json`。只保存模型、地址、采样、思考模式、输出上限等配置，**不保存 API Key**。

新窗口激活环境并进入项目后，选择这份配置：

```bash
export MAS_MODEL_PROFILE=qwen35_27b
```

API Key 仍需在当前窗口设置。显式 export 的同名变量优先于配置文件；换模型时不要保留上一模型的 `MODEL_ID`、`VLLM_BASE_URL` 等覆盖项。可用下列命令查看保存内容：

```bash
python -m core.profiles show qwen35_27b
```

完成自己的 smoke 测试后，可手动标记当前保存版本已经验证。这个命令本身不会调用 API，也不会自动判定 smoke 是否通过：

```bash
python -m core.profiles verify qwen35_27b
```

以后再次 `save` 会保存试用配置，同时保留上一次 `verified_settings`。要固定使用已验证版本：

```bash
export MAS_MODEL_PROFILE=qwen35_27b@verified
```

下次 GPT、DeepSeek 各自保存一个名字。接入点通常可自动识别；代理服务可显式设置 `MODEL_PROVIDER=qwen|deepseek|openai|vllm|compatible`。不认识的远程接入点按 `compatible` 处理，不自动附加其他厂商的思考参数。

千问发送 `max_tokens`、`enable_thinking=false`；DeepSeek 使用 `thinking: {"type":"disabled"}`；OpenAI 默认使用 `max_completion_tokens`，不带千问或 vLLM 的额外参数。**GPT 不同型号的参数支持不同，仍需单独 smoke**。可通过 `MODEL_OMIT_PARAMS=temperature,top_p,seed` 排除不支持的字段，或用 `MODEL_EXTRA_BODY` 指定该接入点支持的 JSON 参数。`ENABLE_THINKING=false` 不会被翻译成一个通用的 GPT 思考关闭参数。

## 2. 环境与数据

只检查环境，不调用模型：

```bash
python -m core.preflight
```

报告使用英文状态 `ok` / `dependency_incompatible`，列出已安装版本和错误。当前验证的 Python 3.11 组合是 NumPy 1.26.4、PyArrow 20.0.0、datasets 3.6.0；报告提供安装命令，不会在导入或评测时自动改动环境。

数据默认使用 `~/.cache/mas-promptbench/data`，可用 `MAS_DATA_DIR` 指定其他位置。加载顺序是：已准备的本地副本 → 已有 HF 本地缓存 → 缺失时下载。缓存版本不明确时不随意挑选；不完整或版本不符会报错。API-Bank 继续使用原来的本地 upstream 数据目录。

APPS 从固定提交 `21e74ddf8de1a21436da12e3e653065c5213e9d1` 的 `test.jsonl` 读取，兼容原始 `id` 字段，随后保存成本地 Arrow 数据。不会要求在项目根目录放一个 `train.jsonl`。

也支持直接复用上次已经准备的 50 题文件，无需重新下载完整数据集：

```bash
export APPS_LOCAL_DATA="results/baseline_centralized4_qwen35_27b_nothinking/apps_20261116_214127_LvxX19/eval_data.json"
```

或将同格式文件放到 `$MAS_DATA_DIR/apps/eval_data.json`（未设置 MAS_DATA_DIR 时使用默认缓存根目录）。这种文件是已选题目的本地子集，沿用每题至多 20 个测试的协议；请求超出缓存的题目或测试上限会明确报错，避免把 50 题误报为 100 题。

LCB 流式读取本地/HF 缓存中的原始 JSONL，避免复制数 GB 的隐藏测试。默认固定提交 `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`，使用 release v3 的三个文件；可用 `LCB_RELEASE`、`LCB_REVISION` 显式改变数据范围。**比较不同模型时使用同一份题目 ID 清单和数据版本**，不能把不同 release 默认当成同一评测。

只允许读本地、缺失时立即报错：

```bash
export MAS_OFFLINE=1
```

这只约束数据和 SWE 仓库/镜像准备。模型 API 仍走网络；HotpotQA 的维基百科检索也仍需联网。取消限制用 `unset MAS_OFFLINE`。

## 3. 先保存产物，再评分与续跑

例如新建一轮 MATH smoke（此命令会调用你配置的模型）：

```bash
python -m topologies.centralized.langgraph.math.langgraph_math --batch --limit 3 --out-dir results/qwen_math_smoke_new
```

同一轮中断后，使用**原命令和原参数**，追加：

```bash
--resume
```

同时需要重试已经记录的执行故障时，追加：

```bash
--resume --retry-execution-errors
```

`--resume` 跳过已有题目，继续未完成题；评分中断的题优先读取已有回答、代码或补丁。普通答错不会因为含有 `error` 字段而被重跑。改变模型、团队规模入口、采样参数、题目范围或评分选项时应使用新目录；smoke 3 题和正式 100 题也应分开。SWE 的 `--workers` 可在 1 和 2 之间调整，不影响已有题目的复用。

输出 JSONL 旁新增 `<文件名>.state/`：

- `run_config.json`：本轮不含 Key 的配置与题目内容指纹；续跑前核对。
- `artifacts/`：完整生成结果、对话、token 用量和生成耗时。摘要中的 raw 可能仍是短片段，这里保留完整内容。
- `records/`：每题最新完整记录，采用临时文件加原子替换，防止写一半。
- `attempts/`：各次完成记录，保留执行故障被补跑前的诊断。

JSONL 是便于统计的导出视图；即使重建视图时中断，也能从逐题记录恢复。直接重复同一 CLI 输出目录会提示使用 `--resume` 或新目录，避免覆盖第一次结果。旧实验目录没有这些运行指纹，不能直接作为新机制的续跑目录；本次没有改写旧实验结果。

## 4. 九类任务分别调整了什么

| 任务 | 改进 | 原因 |
| --- | --- | --- |
| MATH | 最终请求字段适配；记录 `answer_not_found`、`output_truncated`、`turn_limit_reached` 等英文原因；配置复用与续跑 | 只依据实际结束原因判断截断，不把答错推断为 API 故障 |
| HotpotQA | 固定 HTTPS、User-Agent、请求超时；返回英文状态和 HTTP 码，记录检索状态 | 403、超时、无结果、非 JSON 响应和模型请求错误可以分开定位 |
| GPQA | 环境预检；工具参数错误返回匹配调用 ID 的 ToolMessage；逐题续跑 | 提前发现依赖问题，保证后续请求消息成对 |
| API-Bank | `grading_failure` 与 `execution_error` 分离；API 名称、调用格式、参数签名和结果不符是正常评分失败 | 避免通过反复补跑答错题抬高成绩 |
| BFCL | 共用工具历史处理；未知/未执行工具返回错误 ToolMessage；先保存生成结果再评分 | 修复框架历史校验，保留真实错误而非伪造工具成功 |
| ToolHop | 保持原有任务逻辑，继承公共配置、数据缓存和逐题检查点 | 按讨论范围，不额外改工具执行规则 |
| LCB | 从 metadata 读取函数名；真实数据按每行一个 JSON 参数解析；补充 typing 名称；大参数走标准输入 | 避免列表拆错、类型注解缺失和命令行参数长度限制导致误判 |
| APPS | 固定来源、本地优先；保存完整程序后评分；英文失败分类 | 避免相对路径误解析，判题中断时复用生成程序 |
| SWE-bench | 本地仓库/镜像优先；独立干净 checkout；官方仓库专用测试脚本与解析器；参考补丁验证缓存；先存补丁/token；最多两个并行任务 | 复用准备工作，避免旧补丁污染和通用 pytest 误评分 |

ToolMessage 会进入后续模型请求，让模型看到“工具参数无效”或“工具未执行”。它能影响模型接下来的生成，但**不保证模型会改对**；其确定作用是把调用与返回配对，满足接口协议。正常等待执行的调用不会在工具执行前被补成失败。

## 5. SWE 本地复用与并行

只准备仓库和镜像、最多两个并行下载，不调用模型：

```bash
python -m core.swe_prepare --limit 2 --workers 2
```

也可用 `--only 实例ID ...` 准备正式清单中的题目。Git 对象缓存默认在 `~/.cache/mas-promptbench/repos`；镜像沿用 `~/containers/swe`。会优先寻找 `~/swe_work_*` 中以前准备的实例仓库，也支持用 `SWE_LOCAL_REPO_ROOTS` 指定旧仓库根目录。

正式中央式 LangGraph SWE 入口支持 `--workers 1` / `--workers 2`，默认一个；团队规模变体也复用该入口。两个任务使用独立进程、工作目录和逐题输出，汇总文件由父进程写，避免互相覆盖。并行评测需要 WSL/Linux，并会同时占用两份运行资源。

每次从缓存仓库重新检出题目要求的 `base_commit`，不带入上次模型的修改。参考补丁的通过检查按实例、镜像文件、官方 harness 版本和生成脚本缓存，相关内容变化后重新验证。

模型补丁导致 `%APPDATA%` 格式化冲突或测试导入/收集失败时，保留补丁与用量，交由官方评分记录为未解决；不会自动修改模型补丁。如果容器没启动、参考补丁未通过、没有测试区间标记或日志无法解析，则按执行故障停止，保留生成产物供恢复。

## 验证范围

新增测试覆盖实际 SDK 请求体（HTTP mock）、配置不保存 Key、已验证配置保留、评分中断后复用产物、只重试执行故障、工具消息编号与 token 元数据、大输入及 typing、APPS 原始字段、本地缓存、SWE 官方未解决判定、参考检查失效规则和双进程隔离。验证不启动收费模型评测。
