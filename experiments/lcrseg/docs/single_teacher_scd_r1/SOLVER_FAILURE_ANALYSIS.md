# One targeted numerical revision

The old solver already divides a by max(abs(a)). Its hard cap is 2^60 in scaled eta. The published witness has a finite optimum outside that cap; increasing normalization is not a repair.

Let d=a-min(a), v=a.p-min(a), Q0=sum(q[d=0]), delta=min positive d. The exponential-tilt moment is bounded by sum(q*d)/Q0 * exp(-eta*delta). The log-domain upper bound in the authorized protocol therefore certifies an endpoint with margin factor two, without changing the feasible set.

The implementation preserves the exact stored FP64 p and q. It computes v algebraically as sum(d*p)+min(a)*(sum(p)-1), with sum(p)-1 evaluated around the largest probability to retain small terms. This avoids subtraction of nearby b and min(a); it does not replace p_y or renormalize p. Differences from the old rounded dot-product subtraction are explicitly measured, and the independent 100-digit reference interprets the frozen p and normalized a as inputs to the mathematical dot product.

The solution uses 64 fixed bisections in log1p(scaled eta), stable log-weighted moment comparisons and the feasible endpoint. Positive v never becomes a tolerance-based boundary. Exact v=0 uses q conditioned on argmin(a), the KKT boundary solution. Negative v or failed certificates are explicit errors. Ordinary unchanged/nonconflicting q keeps its original bytes.

The mathematical upper endpoint, actual log-moment feasibility and normalized residual <=1e-9 are all checked. Probabilities may underflow to zero; continuous 0 log 0 is retained. Infinite boundary eta is represented by a branch count, and finite eta is aggregated in log space. No infinite JSON numbers or clipped diagnostic eta are emitted.

Only this candidate implementation is qualified. After a frozen E qualification or formal numerical failure, E is stopped without a further repair. Simple ablations continue independently.
