import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

path = Path("fcd.xml")
tree = ET.parse(path)
root = tree.getroot()

filas = []
for timestep in root.iter("timestep"):
    t = float(timestep.attrib["time"])
    for v in timestep.iter("vehicle"):
        attrs = v.attrib.copy()
        attrs["time"] = t
        filas.append(attrs)

df = pd.DataFrame(filas)

for col in ["x", "y", "speed"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col])

print(df.head())
df.to_csv("fcd_data.csv", index=False)
print("Guardado fcd_data.csv")
