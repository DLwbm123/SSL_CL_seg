# V106 stable-state sequential deployment

V105 target-matched CE explained almost all previous stable-state RL-minus-CE stream improvement. Train full-table policies from existing training-only states/returns and compare WARM, original CE, matched-target DISTILL and RL during actual sequential native execution. Keep the12-action dictionary and all existing optimization settings.

Primary: fixed stochastic policies, seeds601/602, four development contexts and three newly frozen student streams3/4/5. Secondary: fixed argmax. Include uniform, global, ridge and1NN controls. All240 native trajectories and480 snapshots are sealed before query scoring. Shared development image/source roles mean this remains same-source evidence. No new Q_train collection or grid-based outcome selection.

The complete preregistration specifies48,008native updates,4,097actor updates,one data ridge solve and2,880development image evaluations. No GPUwallclock cap. Candidate positive results require paired and practical gains over the stronger controls, followed by separately frozen fresh-stream confirmation.
