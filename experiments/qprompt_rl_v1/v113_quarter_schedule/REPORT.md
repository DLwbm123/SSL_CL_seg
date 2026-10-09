# V113 running: fixed quarter-rate diagnostic

V111 andV112 completed and were publicly delivered before launch. Every one of the22V111 methods had negative mean weightednewgain and negative FINALnew−ENTRY100, satisfying the preregistered admission condition. V111 originalRL gate remains negative.

The newrun tests exactly one change: all remaining nativeoptimizer effective learning rates multiplied by.25. OFF/GLOBAL/TIME use the same frozen decisions and source/ENTRY100/contexts/streams as their V1111x controls. All othernative training settings remain fixed. No actorrefit, rategrid, seedselection or earlystop.

Started onGPU4–7. Nativequalification passed8updates with pairedfullsnapshot/RNG replay,16probes and zeroUreads forOFFupdates. Fourworkers each passed200nativeupdates, completing their firstOFFtrajectory; at the startupcheck the rootledger had879deployment+8qualification attempts and matching successes, with0failures. Main and allfourchild process identities and neutralargv were verified withps;GPUdisplay showed neutralPython processnames, about2196MiB each. NASmount/write-readprobe andGPUadmission passed. The oldfitdirectory is a read-only compatibility link toV111, excluded from newcost.

Frozen total:36trajectories×200updates+8qualification=7208nativeupdates,0actorupdates/solves,304stableprobes,432Q_devimageevaluations. All72MID/FINALsnapshots must seal before readout. Actual per-invocation base/effective learning rates are recorded; schedulerbookkeeping remains original as documented inPREREG.

**Status:RUNNING, not results.** The existing hourlyautomation will audit, publish and analyze the complete paired and absolute outcomes. A relative improvement with continued negative absolute adaptation is only partial mitigation; this rate diagnostic cannot establish RL or campaign success.

Private artifacts and logs are onNAS under the runid inSTARTUP_RECEIPT. Only source/protocol/anonymous operational receipts and reports are public. No hiddenUlabels, sealedtest or newpatients/domains are used.
