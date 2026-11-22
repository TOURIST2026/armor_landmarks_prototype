"""Run the same regression tests without installing pytest."""
import inspect
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_pipeline


if __name__ == '__main__':
    started=time.perf_counter()
    count=0
    for name,function in inspect.getmembers(test_pipeline,inspect.isfunction):
        if not name.startswith('test_'):
            continue
        with tempfile.TemporaryDirectory() as directory:
            if 'tmp_path' in inspect.signature(function).parameters:
                function(Path(directory))
            else:
                function()
        count+=1
        print(f'PASS {name}',flush=True)
    print(f'{count} tests passed in {time.perf_counter()-started:.2f}s')
