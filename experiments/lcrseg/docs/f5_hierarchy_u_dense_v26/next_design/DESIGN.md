# Single next candidate: retain V21 contrasts, remove live child-parent attraction

V26 complete/public161f8f7704fac3d0e093890b899ecac93d60d174 precedes this design. Final67.3651%,F5PASS/B2FAIL;164/O2 oldDrishticup39.2931% and allformal16960densecallspositive/3supportedleaves percall. V23relation,V24pooledorientation,V25Ldense,V26Udense do not satisfy dualgate. OriginalV21 hierarchy itself remainsB2FAIL. These results motivate a minimal component ablation, not an attribution of damage to a specific loss.

## Existing source and fixed analytic comparison

`f5_hierarchy_proto_v21/hierarchy.py:hierarchy_terms` normalizes features and estimates equal-present-image leaf/parentprototypes. Coarse pixel contrast uses detached supported parenttargets; sibling contrast uses detached supported leaf targets; LIVE childprototype to detached parentprototype distance hinge is a separate additive term. It is a feature attraction objective, not a spatial containment constraint. The earlier V21 symmetric-sibling analysis established an angular tradeoff; V22 changed siblingcontrast to tangent while retaining alignment and failed. We now review removing alignment alone, keeping full-feature siblingcontrast.

The fixed standard-library comparison uses unit sibling vectors cos(a)u±sin(a)v, parentprototypeu, background−u, equal class/image votes. At fixedangles.4/.8rad,temperature.1,margin.25:

| Angle | Coarse CE | Sibling CE | Alignment | V21 .1sum | .1(coarse+parts) |
|---|---:|---:|---:|---:|---:|
|.4|6.0279e-9|.04704974|.14733866|.01943884|.00470497|
|.8|4.4510e-7|.00003390|.52883668|.05288710|.00000343|

Greater sibling separation greatly lowers sibling CE, slightly increases the already-small coarse CE, and lowers their sum, yet raises the alignment-augmented total. This is a finite objective tradeoff, not proof that constraints are impossible or that patientfeatures follow these vectors. check_alignment.py asserts this onefixedcomparison;0optimizer/model/autograd/patient/GPUcalls,formal_advancementFALSE. AnalyticPASS is not native qualification. Local invocation exposed the repository-relative method script path in temporaryCPUargv; it ended with0model/optimizer calls. Preserve this deviation and use neutralstdin/env for future invocations; do not rerun merely for naming.

## Exact proposed minimal change

Return to positiveV21/V13. `L_L=originalV13_L+.1*(coarse+parts)`; remove the livechild-to-detachedparent distance hinge entirely from the optimized objective. Originaltemp.1/genericparentmap/featureunitnormalization/equal-imageprototypes/classbalancedparentandsiblingCE/GTnearest/LCTXmaskcollect/3x3seam/ignore andfirstLupdate schedule stay. OriginalV13 U/PAS.7,.5/mainKL/entryclassreference/rawaggregateprojection/SWDR-only/sharedentryEMA/AdamLR/hardinputs/source-own-prefix unchanged. No V22tangent orV23–V26 entryfeature/relation additions, no newshape/organlogic/teacher/head/bank/history/deployment parameters. Hierarchy persists in parent-level contrast plus sibling contrast within configured parent groups; deleting attraction does not establish spatial ancestry or anatomy guarantees.

No extra teacher/student/model forward/formalVJP or optimizer steps. ExpectedV21basefullstudent89040/teacher50888/referenceVJP16960/entryanchor16960/classreduce50880/Rdiag64. This is a fixed removal ablation, not a margin/weight sweep or retry of a previous configuration. Contrast targets and GT support remain detached; gradients tocurrentL A/B/R staylive. Implementation should expose the actual absence of alignment in the semantic loss contract/newsource digest and reject checkpoint mismatch, while preserving original model/deployment/source lifecycle. Unsupportedparent/leaf and ignore/zero decisions remain existingkernel behavior.

Removing attraction can also reduce useful regularization, broaden siblingdirections excessively, or change optimization balance; the coarsecontrast may already be sufficient or remain too restrictive. This design cannot infer clinical morphology/independentretention from its analytical case. No newpatientdata/Ulabel/oldimage permission, no assumption that currentU coversoldanatomy. No newrun or qualification registered by this document.

## Next finite implementation and qualification

Nextheartbeat should implement one isolated contrast-only hierarchy path on the existing singleengine; avoid modifying historical frozen source/checkpoints. Register newcreate-onlyNASroot/exactsourceconfig/data/formula/checkpointsemantics/originaldualF5B2gates and fullV13V21V22V23V24V25V26/V15V17V19V20V10/B0positive-negative controls before optimization. Qualification intended32CPU/10generatedCUDA/2realL,max96/32/4,includesdeliberatefailure; reusedplanned nativefixture/exactunforcedactive2/3 andonefixedgradientcheck only. Check coarse/parts equality toV21 for samefeatures/masks, excludedalignment contribution/gradients, allgenericmap/2D3D/zeroignore/detachedtargets/finiteFD32boundedfit/invalidcases; archiveddisabledV13 fullnative parity/originalUrefSWDR-only/teacherEMA/forcedrawguard/resumeactualsettingsrefusal/deploycapturefailure/currentLzeroU. Activecoarse/part support andnewtermA/B/R gradients mustpass. No generatedfixture/probe/patientsearch.

Only completequalificationPASS/EXIT0/formaladvance permits independentlyregisteredfourtraj/eightstage21200/source0/20uniformreadout;first163/O1engineeringseal thenother3/allsealsbeforemetrics. Beforepossible nextmethodformal573073+B15300=578373;new21200cap594273/599573 would be separate science budget, no historicalreleasedbudgettransfer;qualbeforemain1216CPU317CUDA64L/withB11248/327/66, historical673rollback/5NA/failures preserved. Noautomaticretry/retune/newseed/data/RL/otherorgan. Fullpublicproxyverified completion required before laterround; originaldualF5/B2PASS plusdelivery remains only practical stopping criterion, not clinicalsuccess.
