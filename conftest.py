import sys
import os
import subproces

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'src'))

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-e', root, '--quiet'],
    check=False
)
