import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ART = os.path.dirname(HERE)
sys.path[:0] = [ART, HERE, os.path.join(ART, "..", "harness", "v3")]
