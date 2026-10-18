# decisions_priormotion（C3 决策清单）

等你批量裁决（格式：D1: A / D2: 按论文 / …）。已按建议预跑的部分标注现状。等待期间已推进：spec/audit 文档、DSpG 参考实现移植与等价性对拍（1e-6）、参考实现训练双种子在跑。

## 已预决（证据充分，除非你否决）

**D0 评测口径**：论文 Table 2 = 1.0s 单点值。
 现象：均值口径与包络对不上。
 证据：官方权重 slow@1.0s 实测 0.247 vs 论文 MotionNet 行 0.2467，几乎一致。
 结论：【已确认，2026-10-17 实测】对照用 1.0s 单点值，均值口径并存记录。

**D1 双 VAE 结构**：论文 Fig 2 的 Dynamic/Static 双 VAE vs 代码的 present/future 条件 VAE。
 现象：代码无动/静分解；参数量 11.5M 与代码版吻合。
 证据：论文 p.3 Fig 2、p.5 式8；代码 priormotion.py:279-283、vae_model.py:562-730。
 选项：A 以代码为准（条件 VAE，teacher-forcing） B 按论文实现双 VAE（动/静分解）。
 建议：A——代码是唯一跑出论文数字的候选，且 A 的训练已在进行（参考实现双种子，2026-10-18 起）。

**D2 训练配方**：论文 §4.1 bs4/45ep vs 旧脚本 bs8/60ep（wt128 线）。
 证据：论文 p.6；run_priormotionstpn.sh。
 建议：论文口径 bs4/45ep——与 R1B 预算一致且论文数字出自该口径（等效实现为 bs2+累积2，显存限制，已记录）。
 现状：【参考实现已按此口径在跑】。

**D3 sc 权重**：代码 2.0 vs 上游 MotionNet 15.0（论文未给数值）。
 证据：run_priormotionstpn.sh；上游 train_multi_seq.py:78。
 建议：2.0——旧代码实际值，且与 R1 报告"sc=15 在 bs8 下压垮 disp"的发现方向一致。
 现状：【参考实现已用 2.0】。

## 待你裁决

**D4 RVpE 复现深度**（影响工期 2-3 天）：
 现象：论文主结果 0.8324 需要 RVpE（RpE+VpE+PCA）；旧代码只有积木（prior_extractor.py 的 GridInsTransformer/MaskedCrossAttention/位置编码），RpE 栅格分支无实现，主模型未接线。
 证据：论文 p.4 Fig 3、式2-7；prior_extractor.py:6-605；priormotion.py:12（import 未用）。
 选项：A 两步走（先 DSpG 对标 Table 7(e)=0.8943——已在跑；再按论文接线 RVpE 对标 0.8324） B 一步到位直接全配置 C 只复现 DSpG。
 建议：A。

**D5 RVpE 的 RpE 栅格分支**（若 D4 选 A/C 含 RVpE）：
 现象：RpE 的 TSA/SSA 注意力与 ρ 融合无任何代码；VpE 积木齐全。
 证据：论文 p.4 式2-4；全仓 grep 无 TSA/SSA。
 选项：A 严格按论文实现 TSA/SSA/ρ B 简化为卷积+单注意力（论文式2-4 的精神，结构从简）。
 建议：A——这是复现，简化会引入与论文数字的不可控偏差。

**D6 动/静掩码来源**：
 现象：论文说 "off-the-shelf perception model"（现成感知模型）；代码用 GT 的 cls/motion 图（scene_gt）直接充当。
 证据：论文 p.5 §3.2.3；代码 train_VAE.py:508-523。
 选项：A 用 GT（=代码行为；训推不对称与 KL 项成立的前提） B 接一个预训练感知模型。
 建议：A——代码即论文实验的实际行为；"off-the-shelf model"很可能就是指 GT 标注管线本身。

**D7 B′⊕Z_0 融合**：
 现象：式11 未定义 ⊕；代码 = latent repeat(16×16→256×256) + x 残加；conv_fuser 已写但被注释。
 证据：论文 p.5 式11；代码 priormotion.py:331-332。
 选项：A 按代码（repeat+残加） B 启用 conv_fuser。
 建议：A 复现阶段按代码；conv_fuser 已列入小范围探索清单（用户已批准的探索项）。

**D8 扩展口径移植时机**：
 现象：论文 Table 4/5（泛化 GI/长距离/稳定性）需要 eval_general 的扩展口径。
 证据：eval_general.py:316, 624-700, 43-83。
 选项：A 主对照（Table 2/3）完成后移植 B 与主对照同步做。
 建议：A——先保 Table 2/3 主线，扩展口径随后。
