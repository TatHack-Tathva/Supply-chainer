```python
import networkx as nx
import time
from typing import Dict, Any, Optional

from .multimodal_network import MODE_PROFILES, create_multimodal_network
from .threat_intelligence import (
    ThreatIntelligencePredictor,
    ContrastiveNLPEngine,
    CARFFilter,
)
from .news_ingestion import DynamicNewsIngestor
from .node_resolver import NodeResolver


class RouteRecommender:
    """
    Supplychainer Unified Multimodal Optimization Engine.

    Routing pipeline:
        Network
          -> constraints
          -> scenario closures
          -> news intelligence
          -> NLP scoring
          -> CARF filtering
          -> p85 delay prediction
          -> Dijkstra optimization
          -> audit trace
    """

    def __init__(
        self,
        network,
        predictor,
        simulator,
        scenario_mgr,
        demo_mode=False,
    ):
        self.network = network
        self.predictor = predictor
        self.simulator = simulator
        self.scenario_mgr = scenario_mgr
        self.demo_mode = demo_mode

        self.is_warmed_up = False
        self.warmup_failed = False

        self.nlp = ContrastiveNLPEngine(lazy_load=True)
        self.carf = CARFFilter()
        self.news_ingestor = DynamicNewsIngestor()
        self.resolver = NodeResolver()

        print("[STARTUP] Initializing Split-Node Global Topology...")
        self.unified_graph = create_multimodal_network()
        print("[STARTUP] Unified Engine Ready.")

        if self.demo_mode:
            self.is_warmed_up = True

    # ================================================================
    # WARMUP
    # ================================================================

    def run_background_warmup(self):
        """Warm ML/NLP components and populate baseline edge intelligence."""

        if self.is_warmed_up:
            return

        print("[WARMUP] Calibrating global threat floor...")

        try:
            self.predictor.warmup()
            self.nlp.warmup()

            for u, v, edge in self.unified_graph.edges(data=True):
                mode = edge.get("transport_mode", "road")

                # Transfers already have an explicit network risk.
                if mode == "transfer":
                    edge["base_threat"] = float(
                        edge.get("risk", 0.03)
                    )
                    edge["base_news"] = "Standard transfer conditions."
                    continue

                news = self._get_fallback_news(mode)

                score = self.nlp.get_semantic_score(news)
                threat = self.carf.apply_filter(
                    score,
                    news,
                    mode,
                )

                edge["base_threat"] = max(
                    0.0,
                    min(1.0, float(threat)),
                )
                edge["base_news"] = news

            self.is_warmed_up = True
            print("[WARMUP] Unified Calibration Complete.")

        except Exception as exc:
            self.warmup_failed = True
            print(f"[WARMUP] Error during warmup: {exc}")

    # ================================================================
    # NEWS
    # ================================================================

    def _get_fallback_news(self, mode: str) -> str:
        fallback = getattr(
            self.news_ingestor,
            "fallback_news",
            {},
        )

        if isinstance(fallback, dict):
            return fallback.get(
                mode,
                "Normal operating conditions.",
            )

        return "Normal operating conditions."

    def _get_live_news(
        self,
        source: str,
        destination: str,
        mode: str,
    ):
        """
        Obtain live news when the installed news ingestor supports it.

        Falls back safely to the existing mode-specific fallback news.
        """

        getter = getattr(
            self.news_ingestor,
            "get_latest_news",
            None,
        )

        if callable(getter):
            attempts = [
                {
                    "origin": source,
                    "destination": destination,
                    "transport_mode": mode,
                },
                {"mode": mode},
                {},
            ]

            for kwargs in attempts:
                try:
                    result = getter(**kwargs)
                    text = self._normalize_news_result(result)

                    if text:
                        return text, "LIVE"

                except TypeError:
                    # Try the next compatible signature.
                    continue

                except Exception as exc:
                    print(f"[NEWS] Live ingestion failed: {exc}")
                    break

        return self._get_fallback_news(mode), "FALLBACK"

    @staticmethod
    def _normalize_news_result(result) -> Optional[str]:
        """Convert common news result formats into plain text."""

        if result is None:
            return None

        if isinstance(result, str):
            return result.strip() or None

        if isinstance(result, list):
            pieces = []

            for item in result:
                if isinstance(item, str):
                    pieces.append(item)

                elif isinstance(item, dict):
                    for key in (
                        "title",
                        "headline",
                        "summary",
                        "description",
                    ):
                        if item.get(key):
                            pieces.append(str(item[key]))
                            break

            return " ".join(pieces).strip() or None

        if isinstance(result, dict):
            pieces = []

            for key in (
                "title",
                "headline",
                "summary",
                "description",
                "text",
            ):
                if result.get(key):
                    pieces.append(str(result[key]))

            return " ".join(pieces).strip() or None

        return None

    # ================================================================
    # EDGE INTELLIGENCE
    # ================================================================

    def _calculate_edge_intelligence(
        self,
        edge: Dict[str, Any],
        source: str,
        destination: str,
    ) -> Dict[str, Any]:
        """
        Calculate NLP/CARF/ML intelligence for one transport edge.

        Transfer edges retain their explicit network risk and do not
        receive a transport delay prediction.
        """

        mode = edge.get(
            "transport_mode",
            "road",
        )

        if mode == "transfer" or edge.get("type") == "transfer":
            return {
                "threat": max(
                    0.0,
                    min(
                        1.0,
                        float(edge.get("risk", 0.03)),
                    ),
                ),
                "nlp_score": 0.0,
                "ml_p85_delay": 0.0,
                "news": "Standard transfer conditions.",
                "intel_source": "NETWORK",
                "ml_reason": "Transfer friction only.",
            }

        news, intel_source = self._get_live_news(
            source,
            destination,
            mode,
        )

        # NLP
        try:
            nlp_score = float(
                self.nlp.get_semantic_score(news)
            )
        except Exception as exc:
            print(f"[NLP] Error: {exc}")
            nlp_score = 0.0

        nlp_score = max(
            0.0,
            min(1.0, nlp_score),
        )

        # CARF
        try:
            threat = float(
                self.carf.apply_filter(
                    nlp_score,
                    news,
                    mode,
                )
            )
        except Exception as exc:
            print(f"[CARF] Error: {exc}")
            threat = nlp_score

        threat = max(
            0.0,
            min(1.0, threat),
        )

        # ML p85 delay
        p85_delay = 0.0
        ml_reason = "No ML delay available."

        origin_node = edge.get(
            "origin_node",
            edge.get("from", source),
        )

        destination_node = edge.get(
            "destination_node",
            edge.get("to", destination),
        )

        try:
            prediction = (
                self.predictor.predict_worst_case_delay(
                    origin_node,
                    destination_node,
                    mode,
                    "Disrupted" if threat > 0 else "Clear",
                )
            )

            if isinstance(prediction, dict):
                p85_delay = float(
                    prediction.get(
                        "final_delay_presented",
                        prediction.get(
                            "p85_delay",
                            0.0,
                        ),
                    )
                )

                ml_reason = str(
                    prediction.get(
                        "reason",
                        prediction.get(
                            "explanation",
                            "ML p85 delay prediction.",
                        ),
                    )
                )
            else:
                p85_delay = float(prediction)

            p85_delay = max(
                0.0,
                p85_delay,
            )

        except TypeError:
            # Compatibility with predictors exposing fewer arguments.
            try:
                prediction = (
                    self.predictor.predict_worst_case_delay(
                        origin_node,
                        destination_node,
                        mode,
                    )
                )

                if isinstance(prediction, dict):
                    p85_delay = float(
                        prediction.get(
                            "final_delay_presented",
                            prediction.get(
                                "p85_delay",
                                0.0,
                            ),
                        )
                    )
                else:
                    p85_delay = float(prediction)

                p85_delay = max(
                    0.0,
                    p85_delay,
                )

            except Exception as exc:
                ml_reason = (
                    f"ML prediction unavailable: {exc}"
                )

        except Exception as exc:
            ml_reason = (
                f"ML prediction unavailable: {exc}"
            )

        return {
            "threat": threat,
            "nlp_score": nlp_score,
            "ml_p85_delay": p85_delay,
            "news": news,
            "intel_source": intel_source,
            "ml_reason": ml_reason,
        }

    # ================================================================
    # SCENARIO HANDLING
    # ================================================================

    @staticmethod
    def _scenario_blocks_node(
        node_data: Dict[str, Any],
        scenario_name: str,
    ) -> bool:
        """
        Identify physical chokepoints closed by a scenario.

        canonical_hubs.json explicitly defines:
            CHOKE-SUEZ
            type = choke_point
            mode = sea
        """

        if not scenario_name:
            return False

        scenario = str(
            scenario_name
        ).upper()

        physical_id = str(
            node_data.get(
                "physical_id",
                "",
            )
        ).upper()

        display_name = str(
            node_data.get(
                "display_name",
                "",
            )
        ).upper()

        node_type = str(
            node_data.get(
                "type",
                "",
            )
        ).lower()

        # ------------------------------------------------------------
        # SUEZ
        # ------------------------------------------------------------

        if scenario in {
            "SUEZ_BLOCK",
            "SUEZ_BLOCKAGE",
        }:
            return (
                physical_id == "CHOKE-SUEZ"
                or (
                    node_type == "choke_point"
                    and "SUEZ" in (
                        physical_id
                        + " "
                        + display_name
                    )
                )
            )

        # ------------------------------------------------------------
        # PANAMA
        # ------------------------------------------------------------

        if scenario in {
            "PANAMA_BLOCK",
            "PANAMA_BLOCKAGE",
        }:
            return (
                "PANAMA" in physical_id
                or "PANAMA" in display_name
                or physical_id in {
                    "PORT-BALBOA",
                    "PORT-COLON",
                }
            )

        return False

    def _apply_scenario_constraints(
        self,
        graph,
        active_scenario,
    ):
        """Remove virtual nodes representing closed physical chokepoints."""

        if not active_scenario:
            return

        scenario_name = str(
            active_scenario.get(
                "name",
                "",
            )
        )

        nodes_to_remove = []

        for node, data in graph.nodes(data=True):
            if self._scenario_blocks_node(
                data,
                scenario_name,
            ):
                nodes_to_remove.append(node)

        if nodes_to_remove:
            print(
                f"[SCENARIO] {scenario_name}: "
                f"blocking {len(nodes_to_remove)} virtual node(s)"
            )

            graph.remove_nodes_from(
                nodes_to_remove
            )

    # ================================================================
    # ROUTING CONSTRAINTS
    # ================================================================

    @staticmethod
    def _mode_supports_cargo(
        mode: str,
        cargo_type: str,
    ) -> bool:
        if mode == "transfer":
            return True

        profile = MODE_PROFILES.get(
            mode,
            {},
        )

        restrictions = profile.get(
            "cargo_restrictions",
            [],
        )

        return cargo_type not in restrictions

    def _apply_edge_constraints(
        self,
        graph,
        transport_preference: str,
        routing_policy: str,
        cargo_type: str,
    ):
        """
        Apply hard routing constraints.

        STRICT:
            selected transport mode is mandatory.
            unsupported cargo/mode combinations are removed.

        PREFERRED:
            transport preference remains available as a soft preference.
            Cargo restrictions are still enforced.
        """

        policy = str(
            routing_policy or "STRICT"
        ).upper()

        preference = str(
            transport_preference or "any"
        ).lower()

        edges_to_remove = []

        for u, v, data in graph.edges(data=True):
            mode = str(
                data.get(
                    "transport_mode",
                    "road",
                )
            ).lower()

            # Cargo restrictions are always real constraints.
            if not self._mode_supports_cargo(
                mode,
                cargo_type,
            ):
                edges_to_remove.append((u, v))
                continue

            # Transport preference is hard only under STRICT.
            if (
                policy == "STRICT"
                and preference != "any"
            ):
                allowed_modes = {
                    preference,
                    "transfer",
                }

                # Road is needed to reach/connect many hubs even when
                # the requested primary mode is air/sea/rail.
                allowed_modes.add("road")

                if mode not in allowed_modes:
                    edges_to_remove.append((u, v))

        graph.remove_edges_from(
            edges_to_remove
        )

    # ================================================================
    # EDGE WEIGHTS
    # ================================================================

    @staticmethod
    def _edge_weight(
        u,
        v,
        edge,
        persona: str,
        priority: str,
        transport_preference: str,
        routing_policy: str,
    ) -> float:
        """
        Calculate optimization weight.

        Lower weight = more desirable route.
        """

        baseline_time = float(
            edge.get(
                "baseline_time",
                0.0,
            )
        )

        ml_delay = float(
            edge.get(
                "ml_p85_delay",
                0.0,
            )
        )

        scenario_delay = float(
            edge.get(
                "scenario_delay",
                0.0,
            )
        )

        cost = float(
            edge.get(
                "cost",
                0.0,
            )
        )

        threat = float(
            edge.get(
                "effective_threat",
                0.05,
            )
        )

        effective_time = (
            baseline_time
            + ml_delay
            + scenario_delay
        )

        priority = str(
            priority or "normal"
        ).lower()

        # Higher urgency means time matters more.
        priority_factor = {
            "urgent": 0.70,
            "high": 0.85,
            "normal": 1.00,
            "low": 1.20,
        }.get(
            priority,
            1.00,
        )

        if persona == "FASTEST":
            weight = (
                effective_time
                * priority_factor
            )

        elif persona == "SAFEST":
            risk_penalty = (
                1.0
                + threat * 12.0
            )

            # Urgent shipments still care about delay while
            # safety remains a major factor.
            weight = (
                effective_time
                * risk_penalty
                * (0.85 + priority_factor * 0.15)
            )

        else:
            # BALANCED
            time_component = (
                effective_time * 0.45
            )

            cost_component = (
                cost / 150.0
            ) * 0.30

            risk_component = (
                threat * 40.0
            ) * 0.25

            weight = (
                time_component
                + cost_component
                + risk_component
            )

        # PREFERRED is a soft mode preference rather than a hard filter.
        if (
            str(routing_policy or "STRICT").upper()
            == "PREFERRED"
            and str(transport_preference or "any").lower()
            != "any"
        ):
            if edge.get("transport_mode") == transport_preference:
                weight *= 0.90

        return max(
            weight,
            0.000001,
        )

    # ================================================================
    # MAIN ROUTER
    # ================================================================

    def recommend(
        self,
        source: str,
        destination: str,
        transport_preference: str = "any",
        routing_policy: str = "STRICT",
        cargo_type: str = "general",
        priority: str = "normal",
        scenario: str = None,
        overrides: dict = None,
    ) -> dict:

        t0 = time.perf_counter()

        overrides = overrides or {}

        avoid_hubs = overrides.get(
            "avoid_chokepoints",
            [],
        )

        cost_ceiling = float(
            overrides.get(
                "cost_ceiling",
                999999,
            )
        )

        # max_delay is expressed in HOURS.
        max_delay = float(
            overrides.get(
                "max_delay",
                999999,
            )
        )

        # ------------------------------------------------------------
        # Resolve source/destination
        # ------------------------------------------------------------

        res_s = (
            self.resolver.resolve_node_to_entry_point(
                source
            )
        )

        res_d = (
            self.resolver.resolve_node_to_entry_point(
                destination
            )
        )

        if "error" in res_s:
            return {
                "error": res_s["error"]
            }

        if "error" in res_d:
            return {
                "error": res_d["error"]
            }

        s_vnode = res_s["id"]
        d_vnode = res_d["id"]

        # ------------------------------------------------------------
        # Scenario
        # ------------------------------------------------------------

        active_scenario = (
            self.scenario_mgr.activate_scenario(
                scenario
            )
        )

        disruptions = (
            self.scenario_mgr.get_active_disruptions()
        )

        scenario_name = (
            active_scenario.get("name")
            if active_scenario
            else None
        )

        # ------------------------------------------------------------
        # Candidate generation
        # ------------------------------------------------------------

        candidates = []

        intelligence_cache = {}

        for persona in (
            "FASTEST",
            "SAFEST",
            "BALANCED",
        ):

            try:
                G = self.unified_graph.copy()

                # ----------------------------------------------------
                # Hub avoidance
                # ----------------------------------------------------

                if avoid_hubs:
                    nodes_to_remove = [
                        node
                        for node, data
                        in G.nodes(data=True)
                        if data.get("physical_id")
                        in avoid_hubs
                    ]

                    G.remove_nodes_from(
                        nodes_to_remove
                    )

                # ----------------------------------------------------
                # Scenario closure
                # ----------------------------------------------------

                self._apply_scenario_constraints(
                    G,
                    active_scenario,
                )

                # ----------------------------------------------------
                # Cargo + transport constraints
                # ----------------------------------------------------

                self._apply_edge_constraints(
                    G,
                    transport_preference,
                    routing_policy,
                    cargo_type,
                )

                # ----------------------------------------------------
                # Intelligence
                # ----------------------------------------------------

                for u, v, edge in G.edges(data=True):

                    cache_key = (
                        u,
                        v,
                    )

                    if cache_key not in intelligence_cache:
                        intelligence_cache[
                            cache_key
                        ] = (
                            self._calculate_edge_intelligence(
                                edge,
                                source,
                                destination,
                            )
                        )

                    intel = intelligence_cache[
                        cache_key
                    ]

                    edge["nlp_score"] = intel[
                        "nlp_score"
                    ]

                    edge["base_threat"] = intel[
                        "threat"
                    ]

                    edge["ml_p85_delay"] = intel[
                        "ml_p85_delay"
                    ]

                    edge["base_news"] = intel[
                        "news"
                    ]

                    edge["intel_source"] = intel[
                        "intel_source"
                    ]

                    edge["ml_reason"] = intel[
                        "ml_reason"
                    ]

                    # ------------------------------------------------
                    # Scenario disruption
                    # ------------------------------------------------

                    destination_data = G.nodes[v]

                    physical_id = destination_data.get(
                        "physical_id"
                    )

                    scenario_delay = 0.0
                    scenario_threat = 0.0

                    if physical_id in disruptions:
                        disruption = disruptions[
                            physical_id
                        ]

                        scenario_delay = float(
                            disruption.get(
                                "delay",
                                0.0,
                            )
                        )

                        scenario_threat = float(
                            disruption.get(
                                "threat",
                                0.0,
                            )
                        )

                        edge["scenario_reason"] = (
                            disruption.get(
                                "reason",
                                "Scenario disruption.",
                            )
                        )

                    edge["scenario_delay"] = (
                        scenario_delay
                    )

                    edge["effective_threat"] = max(
                        float(
                            edge.get(
                                "base_threat",
                                0.0,
                            )
                        ),
                        scenario_threat,
                    )

                    edge["effective_threat"] = max(
                        0.0,
                        min(
                            1.0,
                            edge["effective_threat"],
                        ),
                    )

                # ----------------------------------------------------
                # Dijkstra
                # ----------------------------------------------------

                def weight_func(u, v, edge):
                    return self._edge_weight(
                        u,
                        v,
                        edge,
                        persona,
                        priority,
                        transport_preference,
                        routing_policy,
                    )

                path = nx.dijkstra_path(
                    G,
                    s_vnode,
                    d_vnode,
                    weight=weight_func,
                )

                # ----------------------------------------------------
                # Build route details
                # ----------------------------------------------------

                legs = []

                baseline_eta = 0.0
                ml_p85_total = 0.0
                scenario_delay_total = 0.0
                transfer_eta = 0.0
                total_eta = 0.0

                total_cost = 0.0
                transit_cost = 0.0
                transfer_cost = 0.0

                max_threat = 0.0
                baseline_risk = 0.0
                scenario_risk = 0.0

                live_count = 0
                fallback_count = 0

                for i in range(
                    len(path) - 1
                ):

                    u = path[i]
                    v = path[i + 1]

                    edge = G[u][v]

                    from_data = G.nodes[u]
                    to_data = G.nodes[v]

                    mode = edge.get(
                        "transport_mode",
                        "road",
                    )

                    edge_baseline = float(
                        edge.get(
                            "baseline_time",
                            0.0,
                        )
                    )

                    edge_ml_delay = float(
                        edge.get(
                            "ml_p85_delay",
                            0.0,
                        )
                    )

                    edge_scenario_delay = float(
                        edge.get(
                            "scenario_delay",
                            0.0,
                        )
                    )

                    edge_eta = (
                        edge_baseline
                        + edge_ml_delay
                        + edge_scenario_delay
                    )

                    edge_cost = float(
                        edge.get(
                            "cost",
                            0.0,
                        )
                    )

                    edge_threat = float(
                        edge.get(
                            "effective_threat",
                            0.05,
                        )
                    )

                    intel_source = edge.get(
                        "intel_source",
                        "NETWORK",
                    )

                    if intel_source == "LIVE":
                        live_count += 1
                    elif intel_source == "FALLBACK":
                        fallback_count += 1

                    baseline_eta += edge_baseline
                    ml_p85_total += edge_ml_delay
                    scenario_delay_total += (
                        edge_scenario_delay
                    )

                    total_eta += edge_eta
                    total_cost += edge_cost

                    max_threat = max(
                        max_threat,
                        edge_threat,
                    )

                    if edge.get("type") == "transfer":
                        transfer_eta += edge_eta
                        transfer_cost += edge_cost

                    else:
                        transit_cost += edge_cost

                        baseline_risk = max(
                            baseline_risk,
                            float(
                                edge.get(
                                    "base_threat",
                                    0.0,
                                )
                            ),
                        )

                    if edge_scenario_delay > 0:
                        scenario_risk = max(
                            scenario_risk,
                            edge_threat,
                        )

                    reason = edge.get(
                        "base_news",
                        "Standard conditions.",
                    )

                    if edge_scenario_delay > 0:
                        reason = edge.get(
                            "scenario_reason",
                            reason,
                        )

                    legs.append(
                        {
                            "from": from_data.get(
                                "physical_id",
                                u,
                            ),
                            "to": to_data.get(
                                "physical_id",
                                v,
                            ),
                            "to_name": to_data.get(
                                "display_name",
                                to_data.get(
                                    "physical_id",
                                    v,
                                ),
                            ),
                            "mode": str(
                                mode
                            ).upper(),
                            "type": edge.get(
                                "type",
                                "transit",
                            ),
                            "distance": round(
                                float(
                                    edge.get(
                                        "distance",
                                        0.0,
                                    )
                                ),
                                1,
                            ),
                            "baseline_eta": round(
                                edge_baseline,
                                2,
                            ),
                            "ml_p85_delay": round(
                                edge_ml_delay,
                                2,
                            ),
                            "scenario_delay": round(
                                edge_scenario_delay,
                                2,
                            ),
                            "eta": round(
                                edge_eta,
                                2,
                            ),
                            "cost": round(
                                edge_cost,
                                2,
                            ),
                            "threat": round(
                                edge_threat,
                                3,
                            ),
                            "nlp_score": round(
                                float(
                                    edge.get(
                                        "nlp_score",
                                        0.0,
                                    )
                                ),
                                3,
                            ),
                            "reason": reason,
                            "intel_source": intel_source,
                            "ml_reason": edge.get(
                                "ml_reason",
                                "",
                            ),
                        }
                    )

                # ----------------------------------------------------
                # Output constraints
                # ----------------------------------------------------

                if total_cost > cost_ceiling:
                    continue

                if total_eta > max_delay:
                    continue

                # ----------------------------------------------------
                # Audit trace
                # ----------------------------------------------------

                trace = {
                    "eta": {
                        "baseline": round(
                            baseline_eta,
                            2,
                        ),
                        "ml_p85": round(
                            ml_p85_total,
                            2,
                        ),
                        "transfer": round(
                            transfer_eta,
                            2,
                        ),
                        "scenario": round(
                            scenario_delay_total,
                            2,
                        ),
                        "total": round(
                            total_eta,
                            2,
                        ),
                    },
                    "cost": {
                        "transit": round(
                            transit_cost,
                            2,
                        ),
                        "transfer": round(
                            transfer_cost,
                            2,
                        ),
                        "scenario": 0.0,
                        "total": round(
                            total_cost,
                            2,
                        ),
                    },
                    "risk": {
                        "baseline": round(
                            baseline_risk,
                            3,
                        ),
                        "scenario": round(
                            scenario_risk,
                            3,
                        ),
                        "max": round(
                            max_threat,
                            3,
                        ),
                    },
                    "intelligence": {
                        "live_edges": live_count,
                        "fallback_edges": fallback_count,
                    },
                }

                candidates.append(
                    {
                        "persona": persona,
                        "primary_mode": "MULTIMODAL",
                        "legs": legs,
                        "adjusted_eta": round(
                            total_eta,
                            2,
                        ),
                        "total_cost": round(
                            total_cost,
                            2,
                        ),
                        "threat_level": round(
                            max_threat,
                            3,
                        ),
                        "audit_trace": trace,
                        "explanation": (
                            self._generate_forensic_explanation(
                                persona,
                                trace,
                            )
                        ),
                        "override_applied": bool(
                            avoid_hubs
                            or cost_ceiling
                            < 999999
                            or max_delay
                            < 999999
                        ),
                    }
                )

            except nx.NodeNotFound:
                continue

            except nx.NetworkXNoPath:
                continue

            except Exception as exc:
                print(
                    f"[ROUTING ERROR] "
                    f"{persona}: {exc}"
                )

        if not candidates:
            return {
                "error": (
                    "No valid multimodal route "
                    "under current strategic constraints."
                )
            }

        # ------------------------------------------------------------
        # Deduplicate identical physical routes
        # ------------------------------------------------------------

        final = []
        seen = set()

        for candidate in sorted(
            candidates,
            key=lambda x: x["adjusted_eta"],
        ):

            path_signature = tuple(
                (
                    leg["from"],
                    leg["to"],
                    leg["mode"],
                )
                for leg in candidate["legs"]
            )

            if path_signature in seen:
                continue

            seen.add(path_signature)
            final.append(candidate)

        return {
            "origin": source,
            "destination": destination,
            "active_scenario": scenario_name,
            "request_meta": {
                "transport_preference": transport_preference,
                "routing_policy": routing_policy,
                "cargo_type": cargo_type,
                "priority": priority,
            },
            "recommendations": final[:3],
            "processing_ms": round(
                (
                    time.perf_counter()
                    - t0
                ) * 1000,
                2,
            ),
        }

    # ================================================================
    # EXPLANATIONS
    # ================================================================

    @staticmethod
    def _generate_forensic_explanation(
        persona,
        trace,
    ):
        """Generate explanations exclusively from computed values."""

        total_eta = trace["eta"]["total"]
        baseline_eta = trace["eta"]["baseline"]
        ml_delay = trace["eta"]["ml_p85"]
        scenario_delay = trace["eta"]["scenario"]

        total_cost = trace["cost"]["total"]
        threat = trace["risk"]["max"]
        transfer_eta = trace["eta"]["transfer"]

        if persona == "FASTEST":
            return (
                f"Time-optimized route. "
                f"Calculated ETA: {total_eta:.1f}h. "
                f"Baseline transit: {baseline_eta:.1f}h; "
                f"p85 ML delay: {ml_delay:.1f}h; "
                f"scenario delay: {scenario_delay:.1f}h."
            )

        if persona == "SAFEST":
            return (
                f"Risk-optimized route. "
                f"Maximum modeled threat: {threat:.2f}. "
                f"Calculated ETA: {total_eta:.1f}h. "
                f"Transfer time: {transfer_eta:.1f}h."
            )

        return (
            f"Balanced route. "
            f"Calculated ETA: {total_eta:.1f}h. "
            f"Total transport cost: {total_cost:.2f}. "
            f"Maximum modeled threat: {threat:.2f}."
        )
```
