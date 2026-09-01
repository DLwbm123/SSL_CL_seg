# PRES-DSR-SF V0.2 call-graph erratum

The frozen router grid has three seeds and two stages, hence six ridge-router units. Each unit records five lambda rows and four temperature rows. The ridge-local evidence is therefore `6 × (5 + 4) = 54` rows: 30 lambda rows and 24 temperature rows.

Clean M1 temperature selection independently records `3 × 2 × 4 = 24` rows. The final combined CV evidence remains `24 + 30 + 24 = 78` rows.

The V0.2 blocker arose because `fit_ridge_routers` held only the 54 ridge-local rows but asserted the global combined total of 78. It also wrote `pres_dsr_cv.csv` before the 24 M1 rows were added, while final counters later treated M1 plus ridge as 78. The local and global evidence scopes were inconsistent.

V0.2.1 must check 54 only inside the ridge function, preserve all 24 M1 rows, and write separate M1 and ridge CSV files plus the sorted 78-row disjoint union. Changing the global 78-row requirement to 54 is prohibited.

The other frozen output counts remain 915 router scores, 117 confusion rows, 27 cross-expert rows, 120 soft-fusion rows, 90 bootstrap rows, and nine memory-cost rows. The full output total remains 1,356.
