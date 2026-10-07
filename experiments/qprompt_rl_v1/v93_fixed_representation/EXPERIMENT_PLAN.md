# V93 preserve the CE actor representation

V92 RL increases training-table reward but does not beat matchedCE in deployment. First audit existing cross-stream action values and savedpolicy distributions without optimizerupdates or queryaccess. Then execute one fixed intervention: freeze the actor hidden representation and optimize only its nine-action head, with a matched head-onlyCE control and both preregisteredseeds. Full-network counterparts are reused fromV92. This tests a representation-generalization hypothesis, not a confirmedcausalexplanation.

Sameactorinitialization,rewardtable,KLreference,objectives and1024steps perfit. Architecture unchanged;trainableparameters297 versus1097. Budget4096actor+3208nativeupdates,0newtrainingqueries;120endpointssealedbefore192devimageevaluations. No adaptivechoices afterdiagnosticreadout. Exactdetails andfailurecriterion inPREREGISTRATION.json.
