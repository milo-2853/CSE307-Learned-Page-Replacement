# 3–5-minute walkthrough notes

Use these as explanations to learn and adapt, rather than a claim of independent authorship.

**0:00–0:40 — Question and workload.** Explain that we compare four replacement policies on the same sequence. The first half revisits eight hot pages, with occasional sequential background requests. The second half draws uniformly from 32 pages. Show a test trace and identify step 2000 as the shift.

**0:40–1:30 — Classical policies.** Open simulate() in experiment.py. FIFO uses insertion time; hits do not reset this time. LRU uses last reference time, updated on every access. Optimal uses the farthest next reference, so it is a benchmark rather than a deployable prediction system. Each missing page causes one simulated fault; eviction happens only when the cache is full.

**1:30–2:20 — Learned policy.** Explain the three history features. Training labels come from Optimal on separate traces. A small tree scores resident candidates; the largest eviction score wins. The model receives no future test references. Its tree remains fixed, while the recent history updates. Open decision_tree.txt to show readable rules.

**2:20–3:20 — Measurements.** Show hit_ratio.png and aggregate.csv. At 12 frames, LRU falls from 80.25% to 38.04%. Learned falls from 72.66% to 36.99%. Do not say the learned model always wins. The chart's error bars show sample SD over five seeds. Fractional faults in the table are seed averages.

**3:20–4:20 — Explanation and verification.** Uniform random access means any causal cache of 12 distinct pages has expected hit probability 12/32 = 37.5%. That matches the observed post-shift causal results. More frames help here. Optimal can do better because it has future information. Run `python -m unittest -v`; show the reference counts, FIFO anomaly and exhaustive oracle check. If a full rerun takes too long during class, show the completed CSVs and start `python experiment.py` separately.

**4:20–5:00 — Limits and disclosure.** State that these are simulator misses, not measured Linux faults. No claim is made about VMware timing or kernel behavior. The learned policy may suffer from oracle-training states differing from its own states; this is a hypothesis, not an experimentally proven cause. State accurately which parts were AI-assisted and which you reran/revised.

## Questions to prepare for

- Why is Optimal not a real-time policy? It needs future references.
- What differs between FIFO and LRU? Admission age versus most recent access.
- Why retain state at the shift? To study one continuously changing workload.
- Does the classifier retrain? No; only historical feature values update.
- Why separate training and evaluation seeds? To prevent training on evaluated traces.
- Why isn't classification accuracy reported? Track 1 focuses on replacement hit ratios/fault counts; labels on oracle-generated cache states would not by themselves establish policy performance.
- Why does LRU degrade most? Its higher initial hit ratio gives it more locality benefit to lose.
- Why can Learned fail to beat LRU? Historical features cannot fully reconstruct the oracle and training/cache states differ.
- Did a frame increase produce Belady's anomaly in these results? No claim is made; the anomaly is demonstrated only by the dedicated FIFO correctness trace.
- What would change in a real OS? Measured costs would include disk I/O, dirty pages, scheduling and policy overhead, none measured here.
