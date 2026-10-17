# Siemens L-only versus existing SSL, seed163

|Arm|Train Dice (%)|Validation Dice (%)|Train-val gap (pp)|
|---|---:|---:|---:|
|A|90.1844|72.2917|17.8926|
|B|90.3468|65.8176|24.5292|

Existing SSL minus L-only validation: -6.4742pp.
A is lambda_U=0 with the original labeled objective, initialization, sampling and schedule.
The B-loader device-type error occurred after A inference; recovery reused A rows and ran only B train inference.
43,600 scientific steps; 43,601 cumulative optimizer calls including the earlier failed call; recovery added zero.
No independent test, U ground truth or checkpoint selection. Historical B differs in GPU/backend.
Classes, ED/ES and paired-patient summaries are in RESULTS.json. Publication verification is separate.
