# TSLP Algeciras

Optimization of traffic sensor placement (cameras / loops / counters) in the Port of Algeciras, combining:

* Microscopic simulation with **SUMO**
* Preprocessing and construction of a time-expanded network
* A **MILP** to place sensors optimally under a budget and observability/robustness constraints
* Result visualization tools (POIs in SUMO, images, experiment notebook)

---

# 1. Project structure

```text
algeciras_milp_v2/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── simulation/
│   │   │   ├── algeciras.net.xml
│   │   │   ├── *.sumocfg
│   │   │   ├── *.rou.xml
│   │   │   └── edgeData_*.xml / tripinfo_*.xml
│   │   └── .
│   │
│   ├── processed/
│   │   ├── edgeData_scen1.csv
│   │   ├── edgeData_scen2.csv
│   │   ├── tripinfo_scen1.csv
│   │   ├── tripinfo_scen2.csv
│   │   ├── milp_inputs.pkl
│   │   └── selected/
│   │       ├── selected_edges_B20.txt
│   │       ├── sensors_B20.add.xml
│   │       └── .
│   │
│   └── Results/
│       ├── 10_MILP.png
│       ├── 20_MILP.png
│       └── .
│
├── src/
│   ├── io_utils.py
│   ├── preprocessing.py
│   ├── milp_model.py
│   ├── milp_solve.py
│   ├── evaluation.py
│   ├── topology_utils.py
│   ├── poi_utils.py
│   └── SUMO/
│       ├── leer_edgedata.py
│       ├── leer_tripinfo.py
│       └── leer_detector.py
│
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_milp.py
│   └── make_sensor_pois.py
│
└── notebooks/
    └── 01_experimentos.ipynb
```

---

# 2. Installation

## 2.1 Create Conda environment

```bash
conda create -n algeciras_milp python=3.10
conda activate algeciras_milp
pip install -r requirements.txt
```

## 2.2 Install SUMO

Download from:
[https://sumo.dlr.de/releases/](https://sumo.dlr.de/releases/)

Set `SUMO_HOME` (Windows):

```powershell
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

---

# 3. Full workflow

## **Step 1 — Run SUMO and generate data**

The project needs two outputs:

* `edgeData` (flow per edge and per time interval)
* `tripinfo` (detailed vehicle information)

### 3.1 Open the port network in SUMO-GUI

1. Launch **SUMO-GUI**
2. `File → Open Simulation`
3. Select a `.sumocfg` file in:

```text
data/raw/simulation/
```

### 3.2 Generate random routes (optional)

```bash
python %SUMO_HOME%/tools/randomTrips.py ^
  -n data/raw/simulation/algeciras.net.xml ^
  -r data/raw/simulation/routes_random.rou.xml ^
  -b 0 -e 3600 ^
  -p 1.0 --seed 42
```

### 3.3 Enable edgeData/tripinfo outputs

In the `.sumocfg` file add:

```xml
<additional-files value="edgeData.xml,tripinfo.xml"/>
```

Run the simulation from SUMO-GUI or from the console:

```bash
sumo -c data/raw/simulation/escenario1.sumocfg
```

---

## **Step 2 — Convert XML → CSV**

Run:

```bash
python src/SUMO/leer_edgedata.py data/raw/edgeData_port_scen1.xml scen1 data/processed/edgeData_scen1.csv
python src/SUMO/leer_tripinfo.py  data/raw/tripinfo_scen1.xml     scen1 data/processed/tripinfo_scen1.csv
```

Repeat for all scenarios.

---

## **Step 3 — Build inputs for the MILP**

Run:

```bash
python -m scripts.run_preprocessing
```

This produces:

```text
data/processed/milp_inputs.pkl
```

which contains:

* time–expanded network
* pseudomeasurements
* O/D per interval
* operational big-M
* identified critical arcs
* etc.

---

## **Step 4 — Run the MILP**

```bash
python -m scripts.run_milp
```

The script:

* Loads `milp_inputs.pkl`

* Builds the (paper-style) MILP

* Applies:

  * sensor budget
  * critical cycles and cuts constraints
  * weights of type 1/max(|flow|, ε)

* Solves with CBC

* Saves:

  * `selected_edges_BXX.txt`
  * coverage metrics
  * printed summary of results

---

## **Step 5 — Visualize sensors in SUMO (POIs)**

Generate POIs:

```bash
python -m scripts.make_sensor_pois ^
  data/processed/selected/selected_edges_B20.txt ^
  data/raw/simulation/algeciras.net.xml ^
  data/processed/selected/sensors_B20.add.xml
```

### Load POIs in SUMO-GUI

1. Open the `.sumocfg`
2. Menu: **Simulation → Edit Configuration**
3. **Input** tab
4. In `additional-files`, add:

```text
data/processed/selected/sensors_B20.add.xml
```

### Hide red nodes (cleaner view)

In SUMO-GUI:

1. **View → Visualization Settings**
2. **Junctions** tab
3. Disable **Show Junctions**

---

# 4. Interactive notebook

In `notebooks/01_experimentos.ipynb`:

* Load `milp_inputs.pkl`
* Run the MILP with different B values
* Automatically generate POIs
* Visualize metrics
* Simple dashboard with images like:

```text
data/Results/20_MILP.png
```

Ideal for preparing figures for the thesis.

---

# 5. Useful SUMO commands

## Open the network in Netedit

```bash
netedit -n data/raw/simulation/algeciras.net.xml
```

## Run a simulation from the console

```bash
sumo-gui -c data/raw/simulation/escenario1.sumocfg
```

## Export a clean screenshot

In SUMO-GUI:

* **View → Save Screenshot**
* Recommendation: maximize the window → better resolution

---

# 6. Summary of key commands

```bash
conda activate algeciras_milp
python -m scripts.run_preprocessing
python -m scripts.run_milp
python -m scripts.make_sensor_pois data/.../selected_edges.txt data/.../net.xml data/.../sensors.add.xml
```

---

# 7. Contact / Authorship

Project developed for traffic optimization and simulation in the **Port of Algeciras**, integrating:

* mathematical optimization
* graph theory
* SUMO simulation
* data analysis
* robust observability methods

---



