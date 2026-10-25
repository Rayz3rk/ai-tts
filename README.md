# 声场 · 本地音频工作台

在现有 Voice-Pro 旁新增一个可独立运行的音频工作台，支持预设音色、F5 本地微调、RVC 目标音色训练与翻唱，以及 DiffSinger 歌词/MIDI 演唱和目标音色串联。原 Voice-Pro 仍在 `voice-pro-4.0.0/`。

## 启动

Windows PowerShell，项目目录下执行：

```powershell
# 首次安装：Python 3.12、PyTorch 2.8 CUDA 12.8 和音频依赖
.\scripts\setup.ps1

# CPU 环境可改为：.\scripts\setup.ps1 -Device cpu
.\start-studio.ps1
```

浏览器打开 <http://127.0.0.1:7861>。在「模型与任务」里显式下载所需模型，查看下载日志；模型完整后即可生成音频。无需启动 Voice-Pro，也无需下载它的 ASR、翻译和分离模型。安装过程需要联网；模型下载完成后，推理和训练读取本地文件。

Python 包默认使用清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`。GPU 专用 PyTorch 轮子使用官方 CUDA 索引，spaCy 的英文模型使用官方发布文件，TTS 权重使用对应模型仓库。

环境已有依赖时也可直接运行：

```powershell
.venv\Scripts\python.exe -m audio_studio serve --port 7861 --device auto
```

## PostgreSQL 与配置

项目配置文件是根目录的 `.env`，与本文件同级；以项目目录为工作目录启动时会自动读取。日常运行使用其中的 `AUDIO_STUDIO_DATABASE_URL`。仓库仅提供空白配置模板，不包含任何数据库账号密码或本机运行数据。已有本地配置时，不要复制模板覆盖 `.env`。

首次在另一台机器安装时，先启动 PostgreSQL 15 或更新版本，再执行：

```powershell
# 仅在 .env 尚不存在时执行，随后用编辑器填写管理员用户名和密码
Copy-Item .env.example .env
.venv\Scripts\python.exe -m audio_studio db-init
.\start-studio.ps1
```

初始字段为 `AUDIO_STUDIO_PG_ADMIN_HOST`、`AUDIO_STUDIO_PG_ADMIN_PORT`、`AUDIO_STUDIO_PG_ADMIN_USER`、`AUDIO_STUDIO_PG_ADMIN_PASSWORD`。`db-init` 新建独立数据库和普通 LOGIN 角色，随机生成应用密码，写回应用连接配置并清空管理员密码。角色没有超级用户、建库或建角色权限。若默认名称已被其他对象占用，会使用带随机后缀的新名称；已有应用连接可用时重复执行不会重建数据库。不要把 `.env` 或连接密码提交、截图或贴到日志中。

表结构位于 `audio_studio` schema，按版本迁移，分别管理工作区、模型、预设、任务、音频素材、数据集和训练记录。首次接入保留 `studio_data/catalogue.pre-postgresql.json` 备份，导入原有模型、预设、任务以及可读的历史 WAV 元数据；重复启动不会覆盖已经修改的素材名称、标签和收藏。`studio_data/workspace.json` 保存工作区身份，移动目录或恢复备份时须一并保留。

配置数据库后，PostgreSQL 是元数据来源；连接失败会阻止启动或返回服务错误，不会自动切换成旧 JSON 数据。未配置连接时可使用原有 JSON 模式进行离线开发，但音频素材管理需要 PostgreSQL。外部环境变量优先于 `.env`，`GET /api/status` 中的 `storage.backend` 可确认实际使用的后端。

## 预设与音频生成

内置 8 个预设，覆盖旁白、日常对话、新闻、有声书、语音助手，以及美式和英式英文。用途预设由音色和语速构成，不承诺模型没有提供的情绪控制。

| 引擎 | 用途 | 输入 | 本地输出 |
| --- | --- | --- | --- |
| Kokoro 82M | 轻量中英文固定音色 | 文本、音色、语速 | 24kHz PCM WAV |
| F5-TTS v1 | 参考音色合成、自有录音微调 | 文本、1–12 秒参考 WAV、参考文本 | 24kHz PCM WAV |

「声音预设」可以保存自定义音色、用途、语速、采样步数和随机种子。F5 需要上传清晰参考音频，并填写与录音完全对应的转写；填写转写后不需要另行下载 ASR 模型。内置预设可复制，不可覆盖或删除。支持的 Kokoro 音色为普通话 `zf_xiaobei/zf_xiaoni/zf_xiaoxiao/zf_xiaoyi/zm_yunjian/zm_yunxi/zm_yunxia/zm_yunyang`，美式英语 `af_heart/am_adam`，英式英语 `bf_emma/bm_george`。

长文本分段生成后拼接完整音频。生成成功后可试听和下载 WAV；缺失权重、无效输入或推理错误会显示失败原因，不返回占位音频。

「音频素材」集中管理生成音频、参考录音和导入的验收样例，支持按名称、文本、文件名和标签搜索，修改名称与标签、收藏、分页、试听和下载。数据库保存文件路径、时长、采样率及关联模型、预设、任务等元数据；WAV 文件仍保存在磁盘。默认 Web 生成音频位于 `studio_data/audio/`，上传参考音频位于 `studio_data/references/`，历史验收样例位于 `outputs/`。CLI `synthesize` 使用 `--output` 指定位置；配置 PostgreSQL 后，Web 与 CLI 的生成音频都会立即登记，CLI `validate` 也会登记数据集检查结果。素材不存在时页面会标明文件缺失。

当前工作台实现的是朗读、参考音色朗读和 F5 语音微调。目标音色演唱、歌声转换及歌词加旋律合成的区别与后续实施顺序见 [歌声功能路线](docs/singing-roadmap.md)。

## 本地训练

F5 语音训练见下文。目标音色翻唱与歌词演唱使用独立的歌声流程：

### 目标音色翻唱

侧栏「06 目标音色翻唱」支持 RVC v2 带 F0 模型：可导入本机 `.pth` 和可选 `.index`，也可用同一人的干净 WAV 目录训练新音色。首次下载 ContentVec、RMVPE 与 RVC v2/40k G/D 预训练权重，约 775MB；训练无需 CSV 或逐句转写。录音检查、分段准备、真实对抗训练、检查点、检索索引和导出模型均在本机完成，成功后自动注册音色。

选择目标音色，上传演唱干声或填写本机音频路径，设置升降调和索引混合比例后生成。可另选独立伴奏和音量，输出混音与转换后干声，均可试听、下载并自动收录。「翻唱」入口要求已分离的人声，不自动从整首歌曲中分离伴奏。上传音频限 0.5–600 秒、150MB，解码后规范为 WAV。

导入的 RVC v2/F0 模型支持 32/40/48kHz；本地训练产物为 40kHz。当前不接受 v1 或无 F0 的模型。RVC 需要已训练的音色模型，短参考录音不会直接成为免训练的歌声音色。请用更多、覆盖目标音域的真实干净录音评估个人音色相似度、长音与泛化；内置验收模型只用合成样本验证链路。

### 歌词与旋律演唱

侧栏「07 歌词与旋律演唱」使用作者发布的绮萱 v2.7.0 DiffSinger ONNX 模型和匹配的 OpenVPI 神经声码器，下载压缩包约 474MB。支持普通话汉字或拼音：导入 MIDI，明确选择单声部旋律轨道，再把歌词填入音符；也可直接编辑表格中的歌词、音高、起点与时长。MIDI 速度变化会转换为实际秒数，重叠和弦不会被静默当成单声部旋律。一字多音的后续音符使用 `-`，休止音高使用 `rest`。

检查乐谱后选择原音色生成 44.1kHz 人声，或选择已就绪的 RVC 目标音色，自动执行「歌词/乐谱 → DiffSinger → RVC」。串联时同时保留声库原音色与转换结果，记录乐谱、模型、任务、参数及来源署名。最长 10 分钟，按乐句生成；当前 DiffSinger 使用 CPU，RVC 跟随工作台设备设置。

绮萱模型、词典与原始使用条款完整保存在模型目录中。输出与素材记录附有“AI 合成 · 绮萱 / DiffSinger”署名；公开发布时按作者条款注明来源。该声库的声学输出及其转换版本不用于音色训练；本项目 RVC 训练验收使用 Kokoro 合成样本，未使用绮萱输出训练。声码器权重按 CC BY-NC-SA 4.0 发布，完整条件见随附 NOTICE 和[官方声库条款](https://github.com/yqzhishen/qixuan-diffsinger/blob/main/terms_of_use/Terms_of_Use.zh-CN.md)。

以下为 F5 语音训练流程。

1. 在模型页下载 F5 基础模型，包含配套词表和 Vocos 声码器。
2. 准备逐句 WAV 和 CSV。现有 `outputs/mandarin_tts_recording_script.csv` 含 260 条录音稿，可以直接使用，但必须自行完成实际录音。
3. 在「本地训练」填写 CSV 和录音目录的绝对路径，点击检查数据。
4. 设置训练名称、轮数和学习率，提交训练。12GB GPU 建议从 batch_size=1、梯度累积=4 开始；实际显存消耗与录音长度有关。
5. 在任务页查看真实 loss、训练日志和状态。训练成功会导出 EMA 权重并自动注册模型，在声音预设中选择该模型并配置参考音频即可生成声音。

数据格式一：现有录音稿格式，`id` 按原样保留前导零，与 `001.wav` 等对应。

```csv
id,text,style
001,今天的天气很好。,neutral
002,请把音量调小一点。,command
```

数据格式二：显式音频路径，可使用录音目录下的相对路径或本地绝对路径。

```csv
audio_path,text
001.wav,今天的天气很好。
002.wav,请把音量调小一点。
```

此例的音频目录应直接指向包含 `001.wav` 和 `002.wav` 的文件夹。

CSV 支持 UTF-8/BOM 和逗号、竖线、制表符或分号分隔。检查器逐个检查文件是否存在、转写是否为空、重复录音、有效 WAV、0.3–30 秒时长、非有限数值和静音，并报告削波与重采样提示。准备阶段将录音复制、转为单声道 24kHz，保留原始文件；生成 F5 的 Arrow 数据、时长 JSON 和保持 token 索引不变的词表。词表之外的字符会阻止训练。

完整基础模型训练使用 F5 CFM/DiT、预训练权重、AdamW、梯度累积、激活检查点、CUDA BF16/FP16（按硬件能力选择）、EMA 和学习率调度。数据加载器单进程运行，兼容 Windows。训练将检查有限 loss 和梯度；每 100 次更新及训练完成时保存检查点。每个训练目录包含：

```text
dataset/                  # 可独立保存的数据与词表
metrics.jsonl             # 每次更新的 loss 和学习率
checkpoints/model_last.pt # 训练状态检查点（保留作研究记录）
model.safetensors         # 可直接用于 F5 推理的 EMA 权重
training.json             # 参数与训练结果
```

当前界面每次启动独立微调，可选择已注册模型继续微调权重；不提供优化器状态的断点恢复按钮。训练 loss 和一次冒烟测试不能代表音色质量，应使用未参与训练的文本试听评估。

## API 与命令行

API 文档在 <http://127.0.0.1:7861/docs>。主要入口：

| API | 功能 |
| --- | --- |
| `GET /api/status` | 设备、依赖、模型文件就绪状态、存储后端 |
| `GET/POST /api/presets` | 列出 / 保存预设 |
| `POST /api/references` | 上传参考 WAV |
| `POST /api/models/{id}/download` | 按需下载模型 |
| `POST /api/models/register` | 注册本地 F5 权重、词表和声码器 |
| `POST /api/synthesize` | 提交音频生成任务 |
| `POST /api/datasets/validate` | 检查本地数据 |
| `POST /api/train` | 提交真实微调任务 |
| `GET /api/jobs/{id}` | 状态、结果、日志 |
| `POST /api/jobs/{id}/cancel` | 取消排队或运行中的任务 |
| `GET /api/audio-assets` | 音频素材搜索与分页（query、limit、offset） |
| `PATCH /api/audio-assets/{id}` | 修改名称、标签、收藏 |
| `GET /api/audio-assets/{id}/content` | 试听或下载已登记的音频文件 |
| `GET /api/covers/status` / `POST /api/covers/assets` | 翻唱组件和音色状态 / 显式下载 |
| `POST /api/covers/models` | 检查并导入 RVC v2 目标音色 |
| `POST /api/covers/audio` / `POST /api/covers/convert` | 输入干声/伴奏 / 转换和可选混音 |
| `POST /api/covers/datasets/validate` / `POST /api/covers/train` | 无转写音色数据检查 / 本地训练 |
| `GET /api/singing/status` / `POST /api/singing/assets` | 歌声模型状态 / 显式下载 |
| `POST /api/singing/midi` / `POST /api/singing/score/validate` | MIDI 轨道与歌词导入 / 乐谱校验 |
| `POST /api/singing/generate` | 歌词/音符演唱，可串联 RVC 目标音色 |

下载、生成、训练通过串行队列进入独立进程，避免抢占显存。取消运行中的任务会终止其工作进程；任务结束后模型随进程释放。接入 PostgreSQL 后任务状态和请求写入数据库，本地保留任务恢复记录和工作进程日志；服务中断后未完成的任务标为失败，需重新提交。请以单服务进程运行，勿在同一数据目录启动多个服务或并发运行 CLI 训练。

```powershell
.venv\Scripts\python.exe -m audio_studio status
.venv\Scripts\python.exe -m audio_studio download kokoro
.venv\Scripts\python.exe -m audio_studio download f5
.venv\Scripts\python.exe -m audio_studio validate outputs\mandarin_tts_recording_script.csv recordings
.venv\Scripts\python.exe -m audio_studio synthesize --preset zh-narration --text '你好，这是本地语音模型。' --output outputs\hello.wav
```

HTTP 示例（PowerShell）：

```powershell
$request = @{ preset_id = 'zh-narration'; text = '你好，这是本地生成的声音。' } | ConvertTo-Json
$job = Invoke-RestMethod http://127.0.0.1:7861/api/synthesize -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($request))
Invoke-RestMethod "http://127.0.0.1:7861/api/jobs/$($job.id)"
```

## 部署与数据目录

默认只监听 `127.0.0.1`，默认数据目录为当前项目的 `studio_data/`。`AUDIO_STUDIO_DATA` 可指定绝对目录；包含模型、参考音频、生成音频、训练产物、工作区身份和任务日志。

备份时停止工作台，使用 `pg_dump` 对实际应用数据库导出，再复制完整 `studio_data/`、需要保留的 `outputs/` 和外部训练录音；安全保存 `.env` 的应用连接凭据。只备份数据库不会包含音频和模型文件，只备份旧 `catalogue.json` 也不能恢复接入 PostgreSQL 后的修改。例如，默认数据库名称未更改时：

```powershell
# 密码通过 PostgreSQL 工具的交互提示提供，不写在命令行里
pg_dump -h 127.0.0.1 -p 5432 -U audio_studio_app -W -d audio_studio -Fc -f audio_studio.dump
```

恢复时用 PostgreSQL 工具将备份恢复到专用数据库，恢复上述磁盘目录并配置新的应用连接。保留 `workspace.json` 才能继续使用原工作区记录；目录内路径会随数据目录位置调整，目录外文件需保持可访问或重新登记。

本机部署使用 `start-studio.ps1`，自动选择当前 PyTorch 可用的设备；也可指定 `-Device cpu` 或 `-Device cuda`。本次验收环境为 CPU 版 PyTorch。需要局域网访问时先设置 `AUDIO_STUDIO_TOKEN`，再指定 `serve --host 0.0.0.0`；浏览器页面右上角可填写令牌，API 使用 `Authorization: Bearer <token>`。页面默认同源访问。远程使用可在前面配置 HTTPS 反向代理。

提供 CPU Docker 部署配置：

```powershell
$env:AUDIO_STUDIO_TOKEN = '请替换为你自己的随机长令牌'
# 先在本地 .env 配置 AUDIO_STUDIO_DOCKER_DATABASE_URL，指向容器可访问的 PostgreSQL
docker compose up --build -d
```

Compose 使用独立的 `AUDIO_STUDIO_DOCKER_DATABASE_URL`，不直接套用本机运行的连接。宿主机数据库可使用 `host.docker.internal` 作为主机名；容器内的 `127.0.0.1` 指向容器自身。连接可以采用 `host=host.docker.internal port=5432 dbname=audio_studio user=audio_studio_app password=...` 的 libpq 格式，按 `.env` 规则妥善引用密码。PostgreSQL 必须已允许来自容器网络的连接；当前配置不会自动修改其监听地址或认证规则。管理员建库凭据不会传入容器。

容器默认仍只映射本机端口，挂载 `studio_data/` 到 `/data`。切换容器前停止本机工作台，同一工作区不能同时在两个路径运行。容器内训练路径必须使用容器内的路径；额外数据目录需要自行映射。Docker 配置使用 CPU 版 PyTorch，不配置 NVIDIA 容器运行时。Windows GPU 使用本机脚本。

数据目录中的模型和参考音频路径会在目录移动或跨系统挂载时自动调整；数据目录以外的文件需要另行挂载或重新登记。Docker 配置已通过解析，本机 Docker 引擎未运行，尚未进行镜像构建验收。

## 测试和验收

```powershell
.venv\Scripts\python.exe -m pytest -q
```

测试包括预设持久化、接口校验、文件上传、访问控制、任务失败恢复、训练数据检查、词表保持、真实小型 DiT 训练与 EMA 导出重载，以及 Voice-Pro 的多段输出和错误传播回归。真实小型训练使用测试信号验证梯度和文件链路，不将其当作可用语音模型。依赖不完整时，模型训练相关测试会标记跳过。

PostgreSQL 集成测试另需设置 `AUDIO_STUDIO_TEST_DATABASE_URL`，使用独立测试集群，避免写入日常数据库。测试会创建独立工作区；建库权限测试仅针对本地测试端口 `55432`。未提供测试连接时相关测试会跳过。仓储与应用测试覆盖实际事务、历史迁移、重启持久化、文件路径调整、素材搜索编辑和引用关系。

系统 Python 已有 Playwright 和 Edge 时，可运行独立浏览器验收；脚本会使用应用的 `.venv` 启动临时服务和数据目录，检查预设操作、数据错误及手机布局，不下载模型：

```powershell
python scripts/browser_smoke.py --screenshots outputs
```

所有内置预设的真实语音输出，以及包含完整 F5 微调和再次生成音频的 API 验收：

```powershell
.venv\Scripts\python.exe scripts\verify_presets.py
# 先下载两种模型并启动本地服务，再执行；会执行一次完整基础模型的梯度更新
.venv\Scripts\python.exe scripts\verify_local_pipeline.py --train
```

后者使用明确标记的合成样本检查训练与部署链路，不将验收产物当作用户的个人音色模型。

测试与验证范围见 [验证说明](docs/validation.md)。本机详细验收记录和生成文件保留在本地，不随源码发布。

三种歌声入口的使用方法、批量填词、测试干声与独立自动创作环境见 [歌声工作方式](docs/song-workflows.md)。

## 模型与原项目

- [F5-TTS 官方源码与训练接口](https://github.com/SWivid/F5-TTS)；[官方模型卡](https://huggingface.co/SWivid/F5-TTS)。
- [Kokoro 官方推理库](https://github.com/hexgrad/kokoro)；[官方模型卡及许可](https://huggingface.co/hexgrad/Kokoro-82M)。
- 显式下载记录上游仓库 revision，并保留模型卡。模型的使用范围由对应模型卡中的许可确定。
- Voice-Pro 的修复：Kokoro 保存所有生成片段；F5 合成失败时停止后续文件处理。其余完整应用依赖及功能独立于本工作台。
