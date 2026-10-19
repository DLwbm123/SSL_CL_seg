# 离散语义、标注钳制模型：零更新数学审计

依据：原Gaussian-logit密度是联合语义梯度冲突的主项；受约束更新虽安全却未增强正确结构，标签条件chart仍保留该连续logit因素且600步后几乎无差异。因此先审计不同概率模型，不再盲延长旧校准。

## 生成模型与变分族

对图i，c_i~N(0,I)，Z_ij|c_i,Theta_G~Categorical(softmax([0,G_Theta(c_i)]_j))。m_i~N(0,.5²)，x_i|Z_i,Theta_F为原正规化SAR Student-t场，中心F_Theta(onehot Z_i)、scale .2、D=I−.9B；I_i|x_i,m_i为原scale .1 Student-t观测。所有生成权重仍独立Gaussian q(Theta)，保留原初始p0；图像模型没有实数ell变量。有标签时Y_i=Z_i为无噪声观测（新假设），未标注时Z潜在。条件于c的像素类别独立；空间相关通过全局c解码与x的SAR引入，不再另设连续logit的SAR密度。

L族q(x|I)q(m|I)q(c|Y)，qx/qm沿原物理/外观网络，qc输入onehot(Y)的三前景通道。L下界=E_q[logp(I|x,m)+logp(x|Y,Theta_F)+logp(Y|c,Theta_G)+logp(m)+logp(c)−logqx−logqm−logqc]。这不是删项消融：原ell及qell不属于新模型，原p(Y|ell)被离散生成p(Z|c)取代；p(Y|c)只计一次，不再另加标签CE。模型不同，ELBO绝对数值不可与旧模型比较。

未来U可取q_phi(Z|I)q(x|I)q(m|I)q(c|Z)，q_phi为确定性图像logit均值的逐像素Categorical。下界在离散Z上期望并含H(q_phi)。离散score-function梯度可严格构造，但方差/有效推断尚待验证。本阶段只用两像素16状态的合成模型检查枚举ELBO≤精确logevidence、差值=KL以及枚举/score-function梯度一致，不访问或运行任何真实U/目标。

有标签生成下界不依赖图像识别器q_phi；其学习需单独最小化KL(delta_Y||q_phi)=−logq_phi(Y|I)，是识别拟合目标，不伪称同一个L生成ELBO或添加任意权重的联合损失。后续如何连接无标签训练/持续推断尚未实现，不能因L钳制语义正确就声称SSL/CL成立。禁止调用旧连续模型的corrected预测冒充新模型推断。

## 此次实际范围和成本

唯一原G100权重仅作新变分初始化，不把旧模型后验当跨模型递推prior；p0仍原始prior，全部tensor不变。原两L四翻转batch8、128²，8既有生成draw。8次真实L bound；首draw分别对完整L目标、生成类别项、shape、image求梯度4次，另识别CE梯度1次；合成枚举与score-function各1反向，共7诊断反向，0optimizer/0参数应用/0局部优化。检查归一化、各项计数、qZ在L上缺席、40语义识别参数对L bound梯度全None，且单独识别CE梯度有效；张量/原G104成本账本不变。

不访问heldout图像/标签、真实U/目标或隐藏test，不引入新训练seed、教师、回放或路由。通过只表示新L边界实现和最小概率恒等式成立；不承诺优于旧模型、不立即训练，后继另立固定协议。所有生成/识别局限公开报告，源码/匿名结果交付；原数据/权重/日志私有数据盘，20GB硬限18GB软阈值，写前挂载容量探针与四根空间，中性create-only入口日志，UTC03:30截止。

工程修复：v1在单独识别CE默认mean归约处触发CUDA确定性算子限制；已完成6反向/1bound/0optimizer。仅改为项目已用的reduction=none后显式mean，数学目标不变，不关闭确定性。原失败现场保留，v2新目录完成审计，失败6反向/1bound另计，不清零；v2若完成则累计13反向/9bound。
