import time
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core import (
    sites, ward_names, pop_density, n_sites,
    generate_demand, run_cpu_pso, run_cuda_pso, run_mopso,
    build_station_design, compute_fitness, decode_particle,
    COVERAGE_RADIUS_KM, GRID_CAP_KW,
)

app = FastAPI(title="EV Charging Station Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")


@app.get("/api/wards")
def get_wards():
    demand = generate_demand(scale=1.0)
    return {
        "wards": [
            {
                "name": ward_names[i],
                "lat": float(sites[i, 0]),
                "lon": float(sites[i, 1]),
                "pop_density": float(pop_density[i]),
                "demand": float(demand[i]),
            }
            for i in range(n_sites)
        ]
    }


class RunRequest(BaseModel):
    n_particles: int = 50
    k_stations: int = 6
    max_iter: int = 80
    demand_scale: float = 1.0
    seed: int = 42


@app.post("/api/run")
def run_optimization(req: RunRequest):
    np.random.seed(req.seed)
    demand = generate_demand(scale=req.demand_scale, seed=req.seed)

    t0 = time.time()
    _, fit_cpu, hist_cpu, stations_cpu = run_cpu_pso(
        sites, demand,
        n_particles=req.n_particles,
        k_stations=req.k_stations,
        max_iter=req.max_iter,
    )
    t_cpu = time.time() - t0

    t0 = time.time()
    _, fit_cuda, hist_cuda, stations_cuda = run_cuda_pso(
        sites, demand,
        n_particles=req.n_particles,
        k_stations=req.k_stations,
        max_iter=req.max_iter,
    )
    t_cuda = time.time() - t0

    rand_scores = []
    for _ in range(20):
        p = np.random.uniform(0, 1, req.k_stations)
        sc = decode_particle(p, sites, req.k_stations)
        rand_scores.append(float(compute_fitness(sc, sites, demand)))
    fit_random = float(np.mean(rand_scores))

    speedup = t_cpu / t_cuda if t_cuda > 0 else 1.0
    improvement = ((fit_cpu - fit_cuda) / abs(fit_cpu)) * 100 if fit_cpu != 0 else 0.0

    design = build_station_design(stations_cuda, demand, ward_names)

    wards_payload = [
        {
            "name": ward_names[i],
            "lat": float(sites[i, 0]),
            "lon": float(sites[i, 1]),
            "demand": float(demand[i]),
            "pop_density": float(pop_density[i]),
        }
        for i in range(n_sites)
    ]

    return {
        "metrics": {
            "t_cpu": round(t_cpu, 4),
            "t_cuda": round(t_cuda, 4),
            "fit_cpu": round(fit_cpu, 6),
            "fit_cuda": round(fit_cuda, 6),
            "fit_random": round(fit_random, 6),
            "speedup": round(speedup, 2),
            "improvement": round(improvement, 2),
        },
        "history": {
            "cpu": [round(float(v), 6) for v in hist_cpu],
            "cuda": [round(float(v), 6) for v in hist_cuda],
        },
        "stations": {
            "cpu": [{"lat": float(s[0]), "lon": float(s[1])} for s in stations_cpu],
            "cuda": [{"lat": float(s[0]), "lon": float(s[1])} for s in stations_cuda],
        },
        "design": design,
        "wards": wards_payload,
    }


@app.post("/api/mopso")
def run_mopso_endpoint(req: RunRequest):
    np.random.seed(req.seed)
    demand = generate_demand(scale=req.demand_scale, seed=req.seed)
    archive_pos, archive_obj = run_mopso(
        sites, demand,
        n_particles=min(req.n_particles, 60),
        k_stations=req.k_stations,
        max_iter=min(req.max_iter, 60),
    )
    return {
        "pareto": [
            {"cost": round(float(o[0]), 4), "coverage_loss": round(float(o[1]), 4)}
            for o in archive_obj
        ]
    }
