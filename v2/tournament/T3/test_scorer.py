import unittest, scorer as S
class T(unittest.TestCase):
    def test_parse(self):
        P = S.parse_verdict
        for o, e in [("human", "human"), ("AI", "ai"), ('"Human".', "human"), ("**ai**", "ai"), (" ai\n\nbecause...", "ai"), ("Human-written", "human"),
                     ("AI-generated", "ai"), ("<think>hmm it is ai</think>human", "human"), ("<think>\nx\n</think>\n\nAI", "ai"),
                     ("<|channel|>analysis<|message|>maybe human<|end|><|start|>assistant<|channel|>final<|message|>ai<|return|>", "ai"),
                     ("", None), (None, None), ("I cannot tell", None), ("human or ai", None), ("The text is human", None), ("unclear", None), ("Humans", None)]:
            self.assertEqual(P(o), e, o)
    def test_correct(self):
        self.assertTrue(S.correct("ai", "ai")); self.assertFalse(S.correct("ai", "human")); self.assertFalse(S.correct("maybe", "human"))
    def test_summary(self):
        rows = [dict(out="ai", gold="ai", kind="S1_ai", source="S1"), dict(out="ai", gold="human", kind="S1_human", source="S1"), dict(out="??", gold="ai", kind="S3_ai", source="S3"), dict(out="human", gold="human", kind="S3_human", source="S3")]
        r = S.summarize(rows); self.assertAlmostEqual(r["acc"], .5); self.assertAlmostEqual(r["format_fail"], .25); self.assertEqual(r["per_kind"]["S3_ai"], 0.0); self.assertEqual(r["per_source"]["S1"], .5)
if __name__ == "__main__": unittest.main()
