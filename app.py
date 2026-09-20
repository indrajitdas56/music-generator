import subprocess
import sys

try:
    import torch
    import audiocraft
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio", "audiocraft"])
    import torch
    import audiocraft
    
          
