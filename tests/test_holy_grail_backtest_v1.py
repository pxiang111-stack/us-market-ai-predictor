import unittest
import pandas as pd
from holy_grail_backtest_v1 import simulate, stats

class TestHolyGrail(unittest.TestCase):
    def bars(self, rows):
        return pd.DataFrame(rows,index=pd.date_range("2024-01-01",periods=len(rows),freq="B"),
                            columns=["Open","High","Low","Close"])
    def setup(self,side="LONG",trigger=101.01,stop=98.99,target=105):
        return pd.DataFrame([{"setup_date":pd.Timestamp("2024-01-01"),"setup_index":0,
                              "side":side,"trigger":trigger,"stop":stop,"target":target}])
    def test_stop_first_when_stop_and_target_same_bar(self):
        b=self.bars([[100,102,98,100],[100,106,98,102]])
        t=simulate(b,self.setup(),cost_side_bps=0,slip_side_bps=0)
        self.assertEqual(t.iloc[0].exit_reason,"stop")
        self.assertAlmostEqual(t.iloc[0].exit,98.99)
    def test_order_expires_after_next_bar(self):
        b=self.bars([[100,100.5,99,100],[100,100.9,99.5,100],[100,103,99,102]])
        self.assertTrue(simulate(b,self.setup(),cost_side_bps=0,slip_side_bps=0).empty)
    def test_short_stop_first(self):
        b=self.bars([[100,102,98,100],[100,103,94,98]])
        t=simulate(b,self.setup("SHORT",98.99,101.01,95),cost_side_bps=0,slip_side_bps=0)
        self.assertEqual(t.iloc[0].exit_reason,"stop")
        self.assertAlmostEqual(t.iloc[0].exit,101.01)
    def test_empty_stats(self):
        self.assertEqual(stats(pd.DataFrame())["trades"],0)

if __name__=="__main__": unittest.main()
