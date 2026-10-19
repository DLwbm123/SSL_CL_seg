# Code自编码依赖零更新报告

固定GT模板匹配为负的一个重要混杂已确认：把模板ell改为GT时继续固定由原ell推断的c，会让ell与c不匹配。为每个反事实模板重编码c后，GT空间与类别匹配在全部四格翻正。但原J初态也已为正，因此这不证明新训练带来了可迁移语义。

|局部坐标|生成后验|模板|code处理|空间差|类别差|
|---|---|---|---|---:|---:|
|initial|initial|GT|fixed|-0.0724884|-0.0037137|
|initial|terminal|GT|fixed|-0.0699071|-0.0034926|
|initial|initial|GT|reencoded|0.0408340|0.0025050|
|initial|terminal|GT|reencoded|0.0410158|0.0023952|
|initial|initial|GT|swapped|-0.0735605|-0.0034561|
|initial|terminal|GT|swapped|-0.0716708|-0.0032200|
|initial|initial|own|fixed|3.8723283|7.1573105|
|initial|terminal|own|fixed|3.8850102|7.1425295|
|initial|initial|own|reencoded|6.1801786|7.3168540|
|initial|terminal|own|reencoded|6.1865945|7.2849817|
|initial|initial|own|swapped|2.0493560|5.6294913|
|initial|terminal|own|swapped|2.0617752|5.6214256|
|terminal|initial|GT|fixed|-0.0676697|-0.0036118|
|terminal|terminal|GT|fixed|-0.0649612|-0.0033729|
|terminal|initial|GT|reencoded|0.0417927|0.0035460|
|terminal|terminal|GT|reencoded|0.0419705|0.0034583|
|terminal|initial|GT|swapped|-0.0706915|-0.0032208|
|terminal|terminal|GT|swapped|-0.0687230|-0.0029489|
|terminal|initial|own|fixed|3.7663221|6.4993868|
|terminal|terminal|own|fixed|3.7843709|6.4950275|
|terminal|initial|own|reencoded|5.9177608|6.7065349|
|terminal|terminal|own|reencoded|5.9310975|6.6882215|
|terminal|initial|own|swapped|1.9962740|5.1018605|
|terminal|terminal|own|swapped|2.0104980|5.0999103|

GT重编码原J初态空间/类别差为+0.04083/+0.00251，终态坐标与后验为+0.04197/+0.00346。自身ell重编码后也为正；交换两图code降低自身模板差，但没有消除正值。说明信号不完全来自固定code，也不能据此说明其为新学到的类语义。

后验置换的增益仍不一致：同一局部坐标下，终态qTheta略提高空间差，但类别差略降低。不能因为读出从负转正就宣布生成参数学习通过；之前负值应解释为特定固定坐标下不匹配，不能推断所有生成能力失效。

1152次批量生成评分，0optimizer、0反向；所有数值有限，fixed格与上一读出1e-5内一致，shape项对code处理不变，退出0。原两训练图及8共同draw，不含标签似然，没有拟合code。全部聚合见[CODE_CLOSURE.json](results/CODE_CLOSURE.json)。

下一步用checkpoint保存的原始先验分布替换qTheta，保持相同局部推断器和全部读出。这能区分训练后生成后验信息与原有随机架构、SAR及推断器对固定生成映射的适配。依然零更新；若先验同样有匹配，正分数不足以支持后验获得语义。
