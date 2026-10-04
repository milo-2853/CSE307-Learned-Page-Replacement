"""Independent correctness checks for graded classical implementations."""
import unittest
from functools import lru_cache
from itertools import product
from experiment import simulate

def faults(trace, frames, policy):
    return sum(r['fault'] for r in simulate(trace,frames,policy)[0])

def brute_opt(trace, frames):
    @lru_cache(None)
    def solve(i, cache):
        if i == len(trace): return 0
        p=trace[i]; current=set(cache)
        if p in current: return solve(i+1,cache)
        if len(current)<frames: return 1+solve(i+1,tuple(sorted(current|{p})))
        return 1+min(solve(i+1,tuple(sorted((current-{v})|{p}))) for v in current)
    return solve(0,())

class Tests(unittest.TestCase):
    def test_reference(self):
        trace=[7,0,1,2,0,3,0,4,2,3,0,3,2]
        self.assertEqual([faults(trace,3,p) for p in ('FIFO','LRU','Optimal')],[10,9,7])
    def test_fifo_belady_anomaly(self):
        trace=[1,2,3,4,1,2,5,1,2,3,4,5]
        self.assertEqual([faults(trace,k,'FIFO') for k in (3,4)],[9,10])
    def test_optimal_against_exhaustive_search(self):
        for trace in product(range(3),repeat=6):
            self.assertEqual(faults(trace,2,'Optimal'),brute_opt(trace,2))
    def test_learned_prefix_has_no_future_leakage(self):
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier(random_state=0).fit([[1,2,1],[9,0,9]],[0,1])
        prefix=[1,2,3,1,4,2,1]
        a=simulate(prefix+[8,9,8],2,'Learned',model)[0]
        b=simulate(prefix+[1,2,1],2,'Learned',model)[0]
        self.assertEqual(a[:len(prefix)],b[:len(prefix)])

    def test_accounting_and_bounds(self):
        for policy in ('FIFO','LRU','Optimal'):
            rows,_=simulate([1,1,2,2,1],2,policy)
            self.assertEqual(sum(r['fault'] for r in rows),2)
            self.assertTrue(all(r['hit']+r['fault']==1 for r in rows))

if __name__=='__main__': unittest.main()
