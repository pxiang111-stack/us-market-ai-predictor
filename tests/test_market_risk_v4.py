import unittest
import numpy as np
import pandas as pd
from market_risk_v4 import make_features, labeled, market_stress

class V4Tests(unittest.TestCase):
    def setUp(self):
        idx=pd.bdate_range("2020-01-01",periods=600); rng=np.random.default_rng(5)
        self.p={"^NDX":pd.Series(1000*np.exp(np.cumsum(rng.normal(.0005,.01,600))),index=idx),
          "SOXL":pd.Series(20*np.exp(np.cumsum(rng.normal(.0008,.03,600))),index=idx),
          "^VIX":pd.Series(20+np.abs(rng.normal(0,4,600)),index=idx),
          "HYG":pd.Series(80*np.exp(np.cumsum(rng.normal(0,.003,600))),index=idx),
          "LQD":pd.Series(100*np.exp(np.cumsum(rng.normal(0,.002,600))),index=idx)}
    def test_features_and_no_embedded_targets(self):
        x=make_features(self.p,"^NDX")
        self.assertIn("vix_level",x); self.assertIn("credit_mom20",x)
        self.assertNotIn("up",x); self.assertNotIn("forward_return",x)
    def test_forward_labels_tail_nan(self):
        x=make_features(self.p,"SOXL"); y=labeled(x,self.p["SOXL"],"5d")
        self.assertTrue(y.up.tail(5).isna().all())
        self.assertTrue(y.forward_return.tail(5).isna().all())
    def test_stress_score_bounded(self):
        rows,s=market_stress(self.p)
        self.assertGreaterEqual(s["risk_score"],0); self.assertLessEqual(s["risk_score"],100)
        self.assertGreaterEqual(s["coverage_items"],4)

if __name__=="__main__": unittest.main()
