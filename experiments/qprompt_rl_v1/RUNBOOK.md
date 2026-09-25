# remote-home 最小迁移操作说明

这里提供可运行的清单/核验/无覆盖提升工具，以及需要绑定路径后执行的rsync模板。**没有连接任何服务器，也没有迁移用户数据。** 工具以可信单用户目录为前提；不能代替服务器隔离、数据许可或HDF5内部字段检查。

## 1. 必须先确定的私有参数

执行方先只读查看已配置SSH alias与项目runtime配置；没有的信息再向用户索取。

- OLD_SSH、NEW_SSH：旧/新服务器现有SSH alias。
- CODE_ROOT：已提交且clean的最小代码工作树；旧repo大依赖先完成最小port。
- DATA_ROOT：canonical HDF5 `images/`、`labels/`的根。若原始manifest/split在另一根，配置`metadata_root`；工具把两源根映射到目标同一`data/`，不修改冻结元数据。
- ASSETS_ROOT：逐文件确认的资料/权重目录；不是整个cache。
- REMOTE_HOME_ABS：新服务器remote-home的真实绝对路径；经realpath和挂载/用户/quota核对。

不把私钥、密码、token填进配置。不要关闭SSH host-key校验。切勿用`rsync old:/... new:/...`：rsync命令需在其中一端执行。

## 2. 在旧服务器准备精确清单

把`migration_config.example.json`复制为**私有**配置，填入实际根目录和commit。code_files逐个列出code/tests/config/license；不接受目录或glob。asset_files逐个列出`path/kind/reason/sha256`，预训练权重最多一个；若先在新服务器下载，则不列入旧端迁移，并由新端另外保存取得回执。

M1：RIM/Drishti L+val；M2：累积包括M1，再加对应U、REFUGE L+val。工具始终排除test payload、REFUGE U、任何U标签和旧checkpoint。

```bash
umask 077
python3 minimal_manifest.py build \
  --config /CONFIRMED_PRIVATE/migration_config.json \
  --out /CONFIRMED_PRIVATE/manifest_M1
```

out目录必须不存在，以免覆盖旧回执。工具会核对冻结manifest/split hash、数据角色/患者一致性、每个必要payload hash和code commit/clean状态；只对元数据和文件字节操作，不加载图像或label数组。

产物：`MIGRATION_MANIFEST.private.json`、`MANIFEST.sha256`、`code.files0`、`data.files0`、`assets.files0`。NUL分隔列表避免按空格拆路径。它们包含private路径/文件名，不上传公开repo。

先审阅记录数、文件数、总字节。工具不自动下载缺失资产，不自动复制代码，不执行网络传输。

## 3. rsync模板：在旧服务器执行

以下变量是占位符，不能直接运行。REMOTE_HOME_ABS必须从新机器查明。这里的shell模板为简化安全引用，仅接受无空格/引号的绝对目标路径；其他合法路径由执行方用Python/shlex参数化，不临时去掉校验。

```bash
set -euo pipefail
umask 077
NEW_SSH='REPLACE_WITH_EXISTING_SSH_ALIAS'
REMOTE_HOME_ABS='/REPLACE_WITH_CONFIRMED_REMOTE_HOME'
MANIFEST_DIR='/REPLACE_WITH_GENERATED_MANIFEST_DIRECTORY'
CODE_ROOT='/REPLACE_WITH_BOUND_CODE_ROOT'
DATA_ROOT='/REPLACE_WITH_BOUND_DATA_ROOT'
METADATA_ROOT='/REPLACE_WITH_BOUND_METADATA_ROOT'
ASSETS_ROOT='/REPLACE_WITH_BOUND_ASSETS_ROOT'
TOOL='/REPLACE_WITH/minimal_manifest.py'

[[ "$NEW_SSH" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
[[ "$NEW_SSH" != *REPLACE* && "$REMOTE_HOME_ABS" != *REPLACE* ]] || exit 2
[[ "$REMOTE_HOME_ABS" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 2
[[ "$REMOTE_HOME_ABS" != *'/../'* && "$REMOTE_HOME_ABS" != */.. ]] || exit 2
DIGEST="$(cat "$MANIFEST_DIR/MANIFEST.sha256")"
[[ "$DIGEST" =~ ^[0-9a-f]{64}$ ]] || exit 2
PROJECT_ROOT="$REMOTE_HOME_ABS/sslcl_qprompt_rl_v1"
INCOMING="$PROJECT_ROOT/.incoming/$DIGEST"
SSH_RSH='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes'

# 只读preflight；先审阅输出、quota和host identity。
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$NEW_SSH" \
  "test -d '$REMOTE_HOME_ABS' && test -w '$REMOTE_HOME_ABS' && realpath '$REMOTE_HOME_ABS' && df -Pk '$REMOTE_HOME_ABS' && df -Pi '$REMOTE_HOME_ABS'"
```

目录/连接被确认后才创建本研究目录。若PROJECT_ROOT已存在，先确认归属、本研究标识及回执；不往无关目录塞文件。未确认的symlink必须停止。

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$NEW_SSH" \
  "umask 077; mkdir -p '$INCOMING/code' '$INCOMING/data' '$INCOMING/assets' '$INCOMING/receipts'"

# 先dry-run。ROOT变量必须与manifest中roots逐字一致；脚本不得自己换根。
for name in code data assets; do
  case "$name" in code) src="$CODE_ROOT";; data) src="$DATA_ROOT";; assets) src="$ASSETS_ROOT";; esac
  rsync -rtn --relative --from0 --protect-args \
    --files-from="$MANIFEST_DIR/$name.files0" \
    --itemize-changes --stats -e "$SSH_RSH" \
    "$src/" "$NEW_SSH:$INCOMING/$name/"
done
rsync -rtn --relative --from0 --protect-args \
  --files-from="$MANIFEST_DIR/metadata.files0" \
  --itemize-changes --stats -e "$SSH_RSH" \
  "$METADATA_ROOT/" "$NEW_SSH:$INCOMING/data/"
```

核对dry-run只含allowlist，记录预估字节。再执行真实复制：

```bash
for name in code data assets; do
  case "$name" in code) src="$CODE_ROOT";; data) src="$DATA_ROOT";; assets) src="$ASSETS_ROOT";; esac
  rsync -rt --relative --from0 --protect-args --checksum \
    --files-from="$MANIFEST_DIR/$name.files0" \
    --partial-dir=.rsync-partial --chmod=D700,F600 \
    --itemize-changes --stats -e "$SSH_RSH" \
    "$src/" "$NEW_SSH:$INCOMING/$name/"
done
rsync -rt --relative --from0 --protect-args --checksum \
  --files-from="$MANIFEST_DIR/metadata.files0" \
  --partial-dir=.rsync-partial --chmod=D700,F600 \
  --itemize-changes --stats -e "$SSH_RSH" \
  "$METADATA_ROOT/" "$NEW_SSH:$INCOMING/data/"
rsync -t --protect-args --chmod=F600 -e "$SSH_RSH" \
  "$MANIFEST_DIR/MIGRATION_MANIFEST.private.json" "$TOOL" \
  "$NEW_SSH:$INCOMING/receipts/"
```

没有`--delete`、`--inplace`、`--remove-source-files`。磁盘需要同时容纳incoming与最终文件提升的暂时副本，先算峰值，不仅算最终大小。中断后在相同digest路径续传；不要重命名失败清单来伪装新成功。

## 4. 目标逐文件核验和提升

目标机需要Python≥3.10。工具本身也应与source hash比对（SSH传输后`sha256sum`），记录在回执。运行：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$NEW_SSH" \
  "python3 '$INCOMING/receipts/minimal_manifest.py' verify \
     --manifest '$INCOMING/receipts/MIGRATION_MANIFEST.private.json' \
     --expected-digest '$DIGEST' --destination '$INCOMING'"

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$NEW_SSH" \
  "python3 '$INCOMING/receipts/minimal_manifest.py' promote \
     --manifest '$INCOMING/receipts/MIGRATION_MANIFEST.private.json' \
     --expected-digest '$DIGEST' --staging '$INCOMING' --destination '$PROJECT_ROOT'"
```

verify默认对code/data/assets检查多余文件。promote先全量检查冲突，再将已核验临时文件无覆盖地提交；已存在且hash相同则复用，不同则拒绝。promote是每文件事务，不是整个目录的一次原子事务；中断后可幂等续做，保留失败记录。提升后只验证清单内文件，不删除原有runs/reports。

工具保留incoming和source，不自动清理。完成后在PROJECT_ROOT/receipts另存带phase/digest的结果；incoming的清理仅在确认最终副本与回执完整后由用户另行许可。

## 5. 环境落在remote-home内

下面只示意新环境变量；实际使用前把PROJECT_ROOT设为已验证路径：

```bash
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export TMPDIR="$PROJECT_ROOT/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
```

环境在新机器重建，不rsync旧conda/venv。DINOv2源码/权重都需pin，构造模型时只使用本地可信路径；训练入口必须拒绝隐式联网下载。GPU型号、driver、Torch/CUDA runtime、精度和determinism记录进新fingerprint，不复制旧GPU编号或NAS wrapper。

## 6. 工具的边界

本工具测试过生成fixture下的M1/M2选择、hash、U标签元数据拒绝、路径/目录/secret拒绝、symlink拒绝、dirty git拒绝、冲突不覆盖和幂等提升。未测试真实SSH、真实服务器quota、HDF5内部dataset键、DINOv2下载、GPU或训练。

在共享主机上，同一UID仍可能访问已搬的其他域；runtime role检查不是OS级隔离。需要更强保障时由执行方使用独立账号/容器只挂载当前域和独立evaluator。不要把“文件在同一个data目录”描述成物理不可访问。
