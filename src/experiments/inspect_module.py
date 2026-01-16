import sys
from pathlib import Path
import inspect

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from time_series_loader import TimeSeriesDataModule

print(f"TimeSeriesDataModule file: {inspect.getfile(TimeSeriesDataModule)}")
sig = inspect.signature(TimeSeriesDataModule.__init__)
print(f"TimeSeriesDataModule.__init__ signature: {sig}")
