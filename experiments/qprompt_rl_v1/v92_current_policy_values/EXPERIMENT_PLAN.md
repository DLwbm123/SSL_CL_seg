# V92 current-policy value collection

V91 improves the KL reference but still trains on V86-behavior continuation values and states. This stage changes the collecting policy to the frozen meanV90denseCE policy, holding originalcontexts, two streams, actionspace, complete-action branchbudget, actorinitialization, objective and1024fitsteps fixed. The stale-policy RL andCE counterparts are reused fromV91; freshCE andfreshRL share allnewinformation andfitbudget.

Collection costs41600native updates and2752existingQtrain imageevaluations; qualification8, newdevelopment3200, actor4096. All104endpoints sealed before192newdevimageevaluations. No newseeds, images, roles, domains or checkpoints selectedbyscore. Exactcriteria andbudgets inPREREGISTRATION.json; allpriorSTOPs preserved.
