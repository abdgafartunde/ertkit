"""
Reading a sounding from a CSV file
==================================

Field data are usually kept in a spreadsheet. Save it as CSV with a header row
naming the columns ``ab2``, ``mn2`` and either ``rhoa`` (apparent resistivity,
ohm meters) or ``resistance`` (transfer resistance, ohms).
"""

# %%
# A small file like those exported from a field sheet. Lines starting with
# ``#`` are comments; extra columns are ignored.

import tempfile
from pathlib import Path

import ertkit
from ertkit import plot, ves

text = """# Example sounding (synthetic values)
ab2,mn2,rhoa,remarks
1.5,0.3,140.9,
2.2,0.3,121.5,
3.2,0.3,95.1,
4.6,0.3,70.8,
6.8,0.3,54.0,
10,0.3,45.3,
10,2.0,46.9,overlap
15,2.0,46.0,
22,2.0,52.1,
32,2.0,66.0,
46,2.0,86.3,
68,2.0,118.8,
100,2.0,164.4,
"""
path = Path(tempfile.mkdtemp()) / "sounding.csv"
path.write_text(text, encoding="utf-8")

# %%
# Read it with a 3% error model. Files from some instruments are not UTF-8;
# pass, for example, ``encoding="latin-1"`` for those.

data = ves.read_csv(path, error=ertkit.ErrorModel(relative=0.03))
print(
    f"{data.n_data} readings, AB/2 from {data.survey.ab2.min()} to {data.survey.ab2.max()} m"
)

# %%
# A smooth inversion is a good first look before choosing a number of layers:

smooth = ves.invert_smooth(data)
print(f"depth of investigation: {smooth.doi:.1f} m")
plot.sounding(data, smooth, labels=["smooth model"])

# %%
plot.layered_model(smooth, labels=["smooth model"], log_depth=True)
