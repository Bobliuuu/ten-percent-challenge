"""L10 answer command wrapper -> ship/answer.py (see REPORT.md)"""
import runpy, os, sys
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ship", "answer.py")
runpy.run_path(sys.argv[0], run_name="__main__")
