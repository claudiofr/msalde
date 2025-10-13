import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import aigct  # noqa: F401 E402

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(current_dir, '..'))