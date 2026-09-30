```python
from contextlib import asynccontextmanager
import asyncio
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine.graph_model import create_logistics_network
from .engine.simulator import LogisticsSimulator
from .engine.threat_intelligence import ThreatIntelligencePredictor
from .engine.baseline import BaselineRouter
from .engine.weather_integration import APIWeatherProvider
from .engine.multimodal_network import (
    create_multimodal_network,
    load_canonical_hubs,
)
from .engine.route_recommender import RouteRecommender
from .engine.scenario_manager import ScenarioManager
from .engine.supplier_scorer import SupplierScorer


# ============================================================================
# Configuration
# ============================================================================

DEMO_MODE = (
    os.getenv("DEMO_MODE", "false").strip().lower() == "true"
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]


# ============================================================================
# Engine initialization
# ============================================================================

# Legacy simulator network.
#
# This is retained because LogisticsSimulator and BaselineRouter still depend
# on the legacy network. The Supplychainer routing engine uses the canonical
# multimodal graph below.
network = create_logistics_network()

simulator = LogisticsSimulator(
    network,
    weather_provider=APIWeatherProvider(),
)

predictor = ThreatIntelligencePredictor()

baseline = BaselineRouter(network)

# Canonical global routing topology.
multimodal_net = create_multimodal_network()

scenario_mgr = ScenarioManager()

recommender = RouteRecommender(
    multimodal_net,
    predictor,
    simulator,
    scenario_mgr,
    demo_mode=DEMO_MODE,
)

canonical_hubs = load_canonical_hubs()

supplier_scorer = SupplierScorer(
    os.path.join(
        os.path.dirname(__file__),
        "data",
        "suppliers.json",
    )
)


# ============================================================================
# Request models
# ============================================================================

class RecommendRequest(BaseModel):
    source: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)

    cargo_type: str = "general"
    priority: str = "normal"
    budget_sensitivity: str = "medium"

    transport_preference: str = "any"
    routing_policy: str = "STRICT"

    scenario: Optional[str] = None
    overrides: Optional[dict] = None


class SourcingRequest(BaseModel):
    category: str = "Electronics"

    current_inventory: int = Field(
        default=1000,
        ge=0,
    )

    safety_stock: int = Field(
        default=1500,
        ge=0,
    )

    demand_forecast: int = Field(
        default=800,
        ge=0,
    )

    scenario: Optional[str] = None


# ============================================================================
# Application lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(
        "Supplychainer Engine Active: "
        "Canonical Global Registry Loaded."
    )

    # Warm the ML/NLP pipeline in the background.
    #
    # DEMO_MODE should not pretend that the models are warm. The recommender
    # itself may use deterministic fallback behavior while warming.
    asyncio.create_task(
        asyncio.to_thread(
            recommender.run_background_warmup
        )
    )

    yield

    print("Supplychainer Engine Shutdown.")


app = FastAPI(
    title="Smart Supply Chain API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Scenarios
# ============================================================================

@app.get("/api/scenarios")
def get_scenarios():
    """
    Return all available deterministic disruption scenarios.
    """

    return scenario_mgr.get_all_scenarios()


# ============================================================================
# Canonical hubs
# ============================================================================

@app.get("/api/hubs")
def get_hubs():
    """
    Return the canonical hub registry.
    """

    return canonical_hubs


@app.get("/api/hubs/search")
def search_hubs(
    q: str = Query(..., min_length=1),
):
    """
    Search canonical hubs by:
      - display name
      - aliases
      - country
      - internal ID
    """

    query = q.strip().lower()

    results = []

    for hub in canonical_hubs:
        display_name = str(
            hub.get("display_name", "")
        ).lower()

        country = str(
            hub.get("country", "")
        ).lower()

        hub_id = str(
            hub.get("id", "")
        ).lower()

        aliases = [
            str(alias).lower()
            for alias in hub.get("aliases", [])
        ]

        if (
            query in display_name
            or query in country
            or query in hub_id
            or any(query in alias for alias in aliases)
        ):
            results.append(hub)

    return results


# ============================================================================
# Network
# ============================================================================

@app.get("/api/network")
def get_network():
    """
    Return the canonical multimodal graph.

    Direction is preserved because the underlying graph is directed.
    """

    nodes = []

    for node_id, data in multimodal_net.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "physical_id": data.get("physical_id"),
                "display_name": data.get("display_name"),
                "type": data.get("type"),
                "mode": data.get("mode"),
                "country": data.get("country"),
            }
        )

    edges = []

    for source, target, data in multimodal_net.edges(
        data=True
    ):
        edges.append(
            {
                "source": source,
                "target": target,
                "baseline_time": data.get(
                    "baseline_time"
                ),
                "distance": data.get("distance"),
                "transport_mode": data.get(
                    "transport_mode"
                ),
                "type": data.get("type"),
                "cost": data.get("cost"),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
    }


# ============================================================================
# Status
# ============================================================================

@app.get("/api/status")
def get_status():
    """
    Return actual engine state.

    Avoid claiming that the ML model is trained merely because the API
    endpoint exists.
    """

    predictor_ready = getattr(
        predictor,
        "model_loaded",
        None,
    )

    if predictor_ready is None:
        predictor_ready = getattr(
            predictor,
            "is_ready",
            False,
        )

    return {
        "ml_trained": bool(predictor_ready),
        "active_trips": len(
            simulator.active_trips
        ),
        "tick": simulator.time_tick,
        "is_supplychainer": True,
        "geo_scope": "Global (Canonical)",
        "hub_count": len(canonical_hubs),
        "routing_engine_ready": recommender.is_warmed_up,
        "warmup_failed": recommender.warmup_failed,
        "demo_mode": DEMO_MODE,
    }


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):
    await websocket.accept()

    try:
        while True:
            if recommender.warmup_failed:
                status_msg = "WARM-UP FAILED"

            elif not recommender.is_warmed_up:
                status_msg = "WARMING RISK ENGINE"

            else:
                status_msg = "FULLY OPERATIONAL"

            state = {
                "tick": simulator.time_tick,
                "ml_trained": bool(
                    getattr(
                        predictor,
                        "model_loaded",
                        False,
                    )
                ),
                "engine_status": status_msg,
                "hub_registry": "Synchronized",
                "warmup_failed": recommender.warmup_failed,
            }

            await websocket.send_text(
                json.dumps(state)
            )

            await asyncio.sleep(2.0)

    except Exception as exc:
        print(
            f"WebSocket closed: {exc}"
        )


# ============================================================================
# Cities / location registry
# ============================================================================

@app.get("/api/cities")
def get_cities():
    """
    Return the canonical city-to-hub mapping.

    Kept for frontend compatibility.
    """

    path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "canonical_locations.json",
    )

    if not os.path.exists(path):
        return {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load canonical locations: {exc}",
        )


# ============================================================================
# Route recommendation
# ============================================================================

@app.post("/api/recommend")
def recommend_routes(
    req: RecommendRequest,
):
    """
    Generate scenario-aware multimodal route recommendations.
    """

    result = recommender.recommend(
        source=req.source.strip(),
        destination=req.destination.strip(),
        cargo_type=req.cargo_type.strip().lower(),
        priority=req.priority.strip().lower(),
        transport_preference=req.transport_preference.strip().lower(),
        routing_policy=req.routing_policy.strip().upper(),
        scenario=(
            req.scenario.strip().upper()
            if req.scenario
            else None
        ),
        overrides=req.overrides,
    )

    return result


# ============================================================================
# Supplier sourcing
# ============================================================================

@app.post("/api/suppliers")
def get_suppliers(
    req: SourcingRequest,
):
    """
    Rank suppliers against the requested category and active scenario.
    """

    active_disruptions = {}

    if req.scenario:
        scenario_id = (
            req.scenario.strip().upper()
        )

        validation = scenario_mgr.validate_scenario(
            scenario_id,
            graph=multimodal_net,
        )

        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=validation["errors"],
            )

        scenario_mgr.activate_scenario(
            scenario_id
        )

        active_disruptions = (
            scenario_mgr.get_active_disruptions()
        )

    ranked_suppliers = (
        supplier_scorer.get_ranked_suppliers(
            req.category,
            active_disruptions,
        )
    )

    advice = (
        supplier_scorer.get_procurement_advice(
            req.current_inventory,
            req.safety_stock,
            req.demand_forecast,
        )
    )

    return {
        "suppliers": ranked_suppliers,
        "advice": advice,
        "active_disruptions": active_disruptions,
    }
```
