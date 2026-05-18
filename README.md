# ⚡ EV Charging Station Optimizer
**CUDA-Accelerated PSO | Bangalore BBMP Wards**  
BMSCE | Dept. of AI & ML | Guide: Prof. Swaroop B.M.

---

## Project Structure

```
ev_cuda_pso/
├── backend/
│   ├── __init__.py
│   ├── core.py          ← All PSO logic, fitness, CUDA kernel, MOPSO
│   └── main.py          ← FastAPI server + REST endpoints
├── frontend/
│   └── index.html       ← Full dashboard UI (Leaflet map + Chart.js)
├── notebooks/           ← Drop .ipynb files here for reference
├── models/              ← Save trained LSTM weights here
├── data/                ← External datasets (ward shapefiles, etc.)
├── .vscode/
│   ├── launch.json      ← Run / Debug configs
│   └── settings.json
├── requirements.txt
├── run.py               ← Entry point
└── README.md
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python run.py
# → Open http://localhost:8000
```

Or press **F5** in VS Code with the "Run EV Optimizer Dashboard" config selected.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/wards` | All 50 BBMP ward centroids + initial demand |
| POST | `/api/run` | Run CPU + CUDA PSO, return full results |
| POST | `/api/mopso` | Run MOPSO, return Pareto front |

### POST `/api/run` — Request Body

```json
{
  "n_particles": 50,
  "k_stations": 6,
  "max_iter": 80,
  "demand_scale": 1.0,
  "seed": 42
}
```

---

## Core Algorithms (backend/core.py)

| Function | Description |
|---|---|
| `haversine_km` | Geographic distance between coordinates |
| `compute_fitness` | Multi-component EV fitness: coverage + distance + grid overload |
| `decode_particle` | Map PSO particle → station coordinates |
| `generate_demand` | Synthetic hourly EV demand via Gaussian peaks |
| `run_cpu_pso` | CPU PSO with linear inertia decay (w: 0.9→0.4) |
| `run_cuda_pso` | GPU PSO with CUDA kernel (falls back to CPU if no GPU) |
| `run_mopso` | Multi-objective PSO with Pareto non-domination sorting |
| `build_station_design` | Compute fast/slow charger mix per optimal station |

---

## Fitness Function

```
f = α·(1 − coverage) + β·mean_dist_norm + γ·overload

α = 0.5  (coverage weight)
β = 0.3  (distance weight)  
γ = 0.2  (grid overload weight)

Coverage radius: 3 km
Grid cap: 150 kW/station
```

---

## Notes on CUDA

- `run_cuda_pso` automatically detects GPU availability via `numba.cuda`
- If no CUDA device is found, it **silently falls back** to CPU PSO
- GPU kernel handles parallel velocity + position updates; fitness evaluation runs on CPU
- To test GPU path: ensure `numba` + CUDA toolkit are installed and a compatible GPU is present
