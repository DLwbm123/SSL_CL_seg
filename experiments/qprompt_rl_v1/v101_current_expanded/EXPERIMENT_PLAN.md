# V101 current expanded-policy returns

V100 rejects argmax as a solution. Refresh Q_train returns using an equal ensemble of all four V99 expanded CE/RL actors. Preserve the same8trainingcontexts,2streams,ENTRY100and12nativeactions. Each state100 full-action return follows this current ensemble at200. The state200 training rows now use the current ensemble reached prefix. Reuse its already measured selected continuation once; collect the other11secondactions.

16jobs x3500nativeupdates and232trainingimageevaluations yield384returns. Add8nativequalification updates; fourmatched CE/RL fits at1024updates each. CE/RL initialize fromownseedV99CE withfixedhistorical rewardscale andidentical labels/budgets. No hyperparameter orseedsearch.

Beforecollection qualify CPUcachedstates against104V100probabilityvectors and8historicalchoices. Afterfit seal312newprobabilityvectors and36policy choices beforegridlookup. OriginalT1sample isprimary; exactconditionalexpectation andargmax aresecondary andbothretained. No newdevelopmentqueryorstudentreplica. Total56008native4096actor3712trainingimageeval,20jobs.
