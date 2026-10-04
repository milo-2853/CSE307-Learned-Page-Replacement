"""Track 1: reproducible page replacement simulation; no kernel/VM measurements."""
import argparse, csv, json, platform, sys
from collections import Counter, deque
from pathlib import Path
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POLICIES = ('FIFO', 'LRU', 'Optimal', 'Learned')
FEATURES = ['recency', 'frequency_64', 'residence_age']

def workload(seed, n=4000):
    """Same shifted trace supplied to every policy and frame capacity."""
    rng = np.random.default_rng(seed)
    half = n // 2
    first = []
    scan = 0
    for _ in range(half):
        if rng.random() < .85:
            first.append(int(rng.integers(0, 8)))
        else:
            first.append(8 + scan % 24)
            scan += 1
    return first + rng.integers(0, 32, n-half).tolist()

def simulate(trace, frames, policy, model=None, collect=False):
    if frames < 1 or policy not in POLICIES:
        raise ValueError('positive frames and a supported policy required')
    if policy == 'Learned' and model is None:
        raise ValueError('Learned requires a fitted classifier')
    # Future indices are consulted ONLY by Optimal, including training oracle.
    future = {}
    if policy == 'Optimal':
        for t, p in enumerate(trace):
            future.setdefault(p, deque()).append(t)
    cache, last, inserted = set(), {}, {}
    window, counts = deque(), Counter()
    rows, samples = [], []
    for t, page in enumerate(trace):
        if policy == 'Optimal':
            future[page].popleft()
        hit, victim = page in cache, None
        if not hit and len(cache) == frames:
            candidates = sorted(cache)
            features = [[t-last[p], counts[p], t-inserted[p]] for p in candidates]
            if policy == 'FIFO':
                victim = min(candidates, key=lambda p: (inserted[p], p))
            elif policy == 'LRU':
                victim = min(candidates, key=lambda p: (last[p], p))
            elif policy == 'Optimal':
                distances = [future[p][0] if future[p] else float('inf') for p in candidates]
                # All tied farthest-future candidates are valid positive labels.
                furthest = max(distances)
                victim = candidates[distances.index(furthest)]
                if collect:
                    samples.extend((f, int(d == furthest)) for f, d in zip(features, distances))
            else:
                positive = list(model.classes_).index(1)
                scores = model.predict_proba(features)[:, positive]
                # Highest eviction probability; tied scores use LRU, then page id.
                victim = max(range(len(candidates)), key=lambda i: (scores[i], features[i][0], -candidates[i]))
                victim = candidates[victim]
            cache.remove(victim)
            del inserted[victim]
        if not hit:
            cache.add(page)
            inserted[page] = t
        last[page] = t
        rows.append({'step': t, 'page': page, 'phase': 'before' if t < len(trace)//2 else 'after',
                     'hit': int(hit), 'fault': int(not hit), 'victim': '' if victim is None else victim})
        window.append(page)
        counts[page] += 1
        if len(window) > 64:
            counts[window.popleft()] -= 1
    return rows, samples

def save_csv(path, rows):
    if not rows:
        raise ValueError('cannot write empty data')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main(out='results'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = []
    for seed in range(8):
        trace = workload(seed, 2000)
        save_csv(out/f'train_trace_{seed}.csv', [{'step':i,'page':p} for i,p in enumerate(trace)])
        for frames in (8,12,16):
            _, samples = simulate(trace, frames, 'Optimal', collect=True)
            train.extend({'seed':seed,'frames':frames, **dict(zip(FEATURES,f)), 'label':y} for f,y in samples)
    save_csv(out/'training_samples.csv', train)
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight='balanced', random_state=0)
    model.fit([[r[k] for k in FEATURES] for r in train], [r['label'] for r in train])
    (out/'decision_tree.txt').write_text(export_text(model, feature_names=FEATURES))
    summaries, all_steps = [], []
    for seed in range(20,25):
        trace = workload(seed)
        save_csv(out/f'test_trace_{seed}.csv', [{'step':i,'page':p} for i,p in enumerate(trace)])
        for frames in (8,12,16):
            for policy in POLICIES:
                rows, _ = simulate(trace, frames, policy, model)
                all_steps.extend({'seed':seed,'frames':frames,'policy':policy,**r} for r in rows)
                for phase in ('before','after'):
                    part = [r for r in rows if r['phase']==phase]
                    hits = sum(r['hit'] for r in part)
                    summaries.append({'seed':seed,'frames':frames,'policy':policy,'phase':phase,
                                      'accesses':len(part),'hits':hits,'faults':len(part)-hits,'hit_ratio':hits/len(part)})
    save_csv(out/'raw_steps.csv', all_steps)
    save_csv(out/'summary.csv', summaries)
    aggregate = []
    for frames in (8,12,16):
        for policy in POLICIES:
            for phase in ('before','after'):
                part = [r for r in summaries if r['frames']==frames and r['policy']==policy and r['phase']==phase]
                h = np.array([r['hit_ratio'] for r in part])
                f = np.array([r['faults'] for r in part])
                aggregate.append({'frames':frames,'policy':policy,'phase':phase,'mean_hit_ratio':float(h.mean()),
                                  'sd_hit_ratio':float(h.std(ddof=1)),'mean_faults':float(f.mean()),'sd_faults':float(f.std(ddof=1))})
    save_csv(out/'aggregate.csv', aggregate)
    fig, axes = plt.subplots(1,3,figsize=(10.5,3.1),sharey=True)
    colors = ['#4675a9','#459d8a']
    for ax, frames in zip(axes,(8,12,16)):
        for j,phase in enumerate(('before','after')):
            part=[r for r in aggregate if r['frames']==frames and r['phase']==phase]
            ax.bar(np.arange(4)+(j-.5)*.36,[r['mean_hit_ratio'] for r in part],.36,
                   yerr=[r['sd_hit_ratio'] for r in part],capsize=3,label=phase.capitalize(),color=colors[j])
        ax.set_xticks(range(4),POLICIES,rotation=20)
        ax.set_title(f'{frames} frames'); ax.set_ylim(0,1); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Hit ratio (mean ± sample SD)'); axes[-1].legend()
    fig.tight_layout(); fig.savefig(out/'hit_ratio.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2,2.7))
    for policy in POLICIES:
        rows=[r for r in all_steps if r['seed']==20 and r['frames']==12 and r['policy']==policy]
        ratios=[np.mean([r['hit'] for r in rows[start:start+200]]) for start in range(0,4000,200)]
        ax.plot(np.arange(100,4000,200),ratios,marker='.',label=policy)
    ax.axvline(2000,color='black',linestyle='--',label='Workload shift')
    ax.set(xlabel='Access index (200-access windows)',ylabel='Hit ratio',ylim=(0,1))
    ax.legend(ncol=5,fontsize=8,loc='lower left'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'timeline.png',dpi=180); plt.close(fig)
    metadata={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,
              'matplotlib':matplotlib.__version__,'training_seeds':list(range(8)),'test_seeds':list(range(20,25)),
              'training_rows':len(train),'tree_depth':int(model.get_depth()),'tree_leaves':int(model.get_n_leaves()),
              'execution':'Python user-space simulation; not Ubuntu/VMware resource measurements'}
    (out/'environment.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='results')
    main(parser.parse_args().out)
