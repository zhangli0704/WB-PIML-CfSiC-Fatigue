# 可学习水氧倍率的假设与验证

主模型：E(vacuum)=0，E(air/oxygen)=1，E(water/oxygen)=10**beta，beta>=0。
只增加一个物理位置参数；原残差候选池、种子、删失似然及组隔离验证规模保留。
每个内层训练子集重新估计beta，外层测试寿命不用于参数、范围或候选选择。
真空0为理想化边界；空气1固定尺度；水氧方向约束是适用条件下的物理先验，不是全条件实验证明。
主拟合无有限beta上界，无新增beta正则项。原有物理参数边界、ridge和环境惩罚仍适用。
计算域对照10/100仅检验一个/两个数量级限制是否改变估计，不是物理上限或待选超参数。
初值中的1/1.5/5只用于同一目标的数值多起点，不是允许权重的离散集合。

数据限制：水氧12条记录仅1300℃、3Hz、R=0.1、2D编织，来自两个试验组，无匹配空气数据。
参数可能与温度、K和试验来源混杂。可拟合不等于可辨识，更不等于测得独立环境反应速率。
若训练无水氧或无空气参考，beta固定0且标记未辨识；不能利用留出环境寿命拟合。
水氧LOEO中的参考倍率1只是未辨识情况下的公开假设，不能称为学习了未知气氛效果。

对照：selected_pipeline为训练内估计；water_1和water_1.5为固定权重条件对照。
所有对照保留修订模型选定残差配置，但重新拟合其训练物理参数和OOF尺度。
原0.5/1.25/1.75/2/3情景仍保留，0.5是允许方向反转的压力测试，不是主模型先验。
no_environment_damage关闭显式环境物理项；所有残差均不使用环境输入，因此该消融移除显式环境作用。environment_penalty_0仅取消惩罚。
温度/应力/频率驱动消融、真空小非零情景及有限权重×频率交互全部保留。
水氧系数不由环境样本比例或不同环境原始平均寿命直接计算。

water_identifiability_profiles.csv同时报告训练集原惩罚目标剖面和保留干扰参数边界的无惩罚似然剖面。
不从剖面挑更高外测精度，不自动给95%区间；若范围内曲线平坦或自由拟合存在局部极值，明确输出。
water_air_counterfactual_order.csv检查物理主干和完整模型；环境共享残差使最终预测保持主干的环境顺序，发现逆转即报错。
这些环境切换是模型情景检查，不是实测空气/水氧配对验证。

本次是审稿和已查看结果之后的模型修订，不称为预注册。敏感性区间是固定OOF预测的试验组bootstrap，
不包含模型重选不确定性，也未作多重比较校正。本次结构修订用于满足既定环境方向先验，不根据外层误差改选水氧权重。

机制资料只支持环境作用及条件依赖，不支持通用1.5倍率：
Liu C.-D. et al. Fatigue Behavior of 2D C/SiC Composites in Water-vapor Containing Environment at 1300℃.
Journal of Inorganic Materials, 2008, 23(4), 729. https://doi.org/10.3724/SP.J.1077.2008.00729
C/SiC Life Prediction for Propulsion Applications. NASA NTRS 20040033966.
https://ntrs.nasa.gov/citations/20040033966
参数诊断方法参考：Raue A. et al. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood.
Bioinformatics 2009;25(15):1923–1929. https://doi.org/10.1093/bioinformatics/btp358
本程序没有直接使用该文的渐近置信阈值，因本模型含约束、惩罚和少量异质试验组。

后续先冻结预测再记录实验结果；空气试验不能标定水氧倍率。倍率需同批材料同工况气氛对照及独立验证组。
替代生存分布仍假设条件独立删失；runout权重分析不证明该假设成立。停止规则仍需文献证据。
预测区间尚非工程设计寿命界限。
