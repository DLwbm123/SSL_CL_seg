# 代码差异、证据与待审阅风险

## 差异

只新增 `native_key_alignment_v0_1/` 与对应 docs 目录，不修改原 native parent、B2 train_stage、数据提供器、F5代码/结果/批准文件。

- alignment.py：真实层/V绑定、稀疏patch gather、独立随机流、D/8 类别条件SWD。
- trainer.py：B2 loss-only适配；继承共同更新引擎；一次额外clean-U，固定步梯度诊断，生产禁用。
- protocol.py：四份真实前缀元数据绑定、16节点D1、完整options与门槛、只读范围校验。
- tests.py：固定7组原生CPU/生成张量测试、物理调用账本和最多2次尝试。
- __main__.py / __init__.py：仅代码准备入口。

CODE_MANIFEST.json 绑定本checkout所有 experiments Python依赖，TEST_REPORT绑定同一代码树与D1 plan。MODEL_TENSOR_BINDING.json 和测试报告记录实际原生结构/梯度证据。CODE_DIFF.txt给出新增文件列表和与指定基线的边界。

## 实测覆盖

1. patch与unfold逐值相等，中心/类别坐标与边界invalid mask；真实[16,16,3,3]、V[144,8]。
2. 有效SWD非零，参考/V无梯度，无支持返回graph-connected零。
3. 上游B存在非零辅助梯度，B=0时A为零符合预期，目标层A/B不接收输入侧辅助梯度。
4. lambda=0与原B2两步的模型/EMA/optimizer/scheduler/prototypes/support/读取计数/CPU及Python RNG完全相等，包含no-op buffer校验。
5. C0曝光与C1/C2/C3同批次监督/KL一致；三个坐标采样位置一致、维度缩放正确；C2不改保护V；更新后父硬投影残差原检查通过。
6. 拒绝旧研究执行能力、非生成provider、非固定正系数；拒绝跨seed/order/phase或改写旧方法身份的前缀。
7. 两个纯生成固定batch各3步的损失下降sanity，非收敛声明；全部optimizer与附加VJP/前向成本单列。

## 待审阅和未来工程风险

- 新loss方法复用了B2原分支运算，但属于loss适配而非修改旧类；后续上游B2若变化须重核lambda=0，不能仅靠继承认为等价。
- 原生CPU资格尺寸32×32，PAS 0/-1和前景分类核只用于生成资格；正式options仍3200/2100步、PAS .6/.7。未执行原生CUDA、真实384输入smoke或实际前缀重新验收。
- 生产调度/新协议 checkpoint-resume 与真实prefix tensor gate尚未授权接入。当前不能直接启动D1，不能借旧F5批准绕过本轮外部审阅。
- mask采用3×3全patch几何有效、PAS仅center；同网格标签直接映射需外部审阅确认与实际输出几何一致，不能改用最终分类头裁剪坐标。
- D/8只补偿单位方向的直接维数尺度，不保证梯度范数匹配。随机键在一次入口QR中固定，不筛选；其QR成本应在未来阶段入口成本中记录。
- 诊断额外VJP和clean-U增加计算量；本轮CPU operation_counts和alignment counters分别记录，不相加重复记账。正式前向、评价、显存峰值和optimizer调用必须继承原成本控制器并另记新增诊断。
- 有些batch没有有效类别时辅助项为零是有效科学观测，不能伪造支持或放松PAS。当前域对齐不能保证历史特征不漂移。
- 不将此提交/CPU PASS写成APPROVED，不复用旧审批，不自动启动任何真实实验。
