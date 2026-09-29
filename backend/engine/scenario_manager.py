
from copy import deepcopy
from typing import Dict, List, Any, Optional


class ScenarioManager:
    """
    Supplychainer Scenario Trigger Engine.

    Provides deterministic disruption definitions for demonstrations,
    testing, and scenario-aware route optimization.

    A scenario can:
      - increase threat
      - add delay
      - hard-block affected physical nodes
      - identify the transport mode affected

    The manager itself does not modify the routing graph. It only
    provides validated scenario state to the routing engine.
    """

    SCENARIOS = {
        "SUEZ_BLOCK": {
            "name": "Suez Canal Blockage",
            "description": (
                "Critical maritime corridor obstructed by vessel grounding."
            ),
            "affected_nodes": ["CHOKE-SUEZ"],
            "threat_level": 1.0,
            "delay_hours": 240,
            "reason": (
                "Vessel grounding in Canal Narrows. "
                "Canal authority estimates 10-day salvage window."
            ),
            "mode": "sea",
            "blocked": True,
        },

        "RED_SEA_CONFLICT": {
            "name": "Red Sea Escalation",
            "description": (
                "Increased regional instability affecting Bab el-Mandeb."
            ),
            "affected_nodes": ["CHOKE-BABEL"],
            "threat_level": 0.85,
            "delay_hours": 72,
            "reason": (
                "Regional conflict escalation. "
                "Vessels rerouting via Cape of Good Hope for risk mitigation."
            ),
            "mode": "sea",
            "blocked": False,
        },

        "LA_PORT_STRIKE": {
            "name": "LA Port Strike",
            "description": (
                "Labor dispute causing terminal shutdowns in Los Angeles."
            ),
            "affected_nodes": [
                "PORT-LOSANGELES",
                "PORT-LONGBEACH",
            ],
            "threat_level": 0.9,
            "delay_hours": 120,
            "reason": (
                "Terminal labor strike. Picket lines at all major berths. "
                "Throughput at 0%."
            ),
            "mode": "sea",
            "blocked": True,
        },

        "CHENNAI_FLOOD": {
            "name": "Chennai Monsoon Flooding",
            "description": (
                "Extreme weather disrupting South India logistics."
            ),
            "affected_nodes": [
                "PORT-CHENNAI",
                "HUB-CHENNAI",
            ],
            "threat_level": 0.75,
            "delay_hours": 48,
            "reason": (
                "Severe urban flooding. Inland road access to Port and "
                "Logistics Park is underwater."
            ),
            "mode": "road",
            "blocked": False,
        },

        "DUBAI_AIR_CONGESTION": {
            "name": "Dubai Hub Surge",
            "description": (
                "Massive cargo backlog at DXB/DWC."
            ),
            "affected_nodes": ["AIR-DUBAI"],
            "threat_level": 0.65,
            "delay_hours": 24,
            "reason": (
                "Regional cargo surge exceeding ground handling capacity. "
                "48h clearance backlog."
            ),
            "mode": "air",
            "blocked": False,
        },

        "HORMUZ_CLOSURE": {
            "name": "Hormuz Strait Escalation",
            "description": (
                "Strategic maritime choke point tension."
            ),
            "affected_nodes": ["CHOKE-HORMUZ"],
            "threat_level": 1.0,
            "delay_hours": 168,
            "reason": (
                "Strategic naval activity. "
                "Vessels holding position at Jebel Ali / Colombo."
            ),
            "mode": "sea",
            "blocked": True,
        },
    }

    VALID_MODES = {"road", "rail", "air", "sea"}

    def __init__(self):
        self.active_scenario_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Scenario activation
    # ------------------------------------------------------------------

    def activate_scenario(
        self,
        scenario_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Activate one scenario.

        Invalid or empty scenario IDs clear the active scenario.
        A deep copy is returned so callers cannot mutate SCENARIOS.
        """

        if scenario_id is None or scenario_id == "":
            self.active_scenario_id = None
            return None

        scenario_id = str(scenario_id).strip().upper()

        if scenario_id not in self.SCENARIOS:
            self.active_scenario_id = None
            return None

        self.active_scenario_id = scenario_id

        return deepcopy(self.SCENARIOS[scenario_id])

    # ------------------------------------------------------------------
    # Active scenario
    # ------------------------------------------------------------------

    def get_active_scenario(self) -> Optional[Dict[str, Any]]:
        """
        Return a safe copy of the currently active scenario.
        """

        if self.active_scenario_id is None:
            return None

        scenario = self.SCENARIOS.get(self.active_scenario_id)

        if scenario is None:
            return None

        return deepcopy(scenario)

    # ------------------------------------------------------------------
    # Disruptions
    # ------------------------------------------------------------------

    def get_active_disruptions(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert the active scenario into a physical-node disruption map.

        Example:

            {
                "CHOKE-SUEZ": {
                    "delay": 240,
                    "threat": 1.0,
                    "reason": "...",
                    "source": "SCENARIO_OVERRIDE",
                    "mode": "sea",
                    "blocked": True
                }
            }

        The routing engine can then decide how to apply these
        disruptions to virtual graph nodes.
        """

        scenario = self.get_active_scenario()

        if scenario is None:
            return {}

        disruptions: Dict[str, Dict[str, Any]] = {}

        delay = max(0.0, float(scenario.get("delay_hours", 0.0)))
        threat = min(
            1.0,
            max(0.0, float(scenario.get("threat_level", 0.0))),
        )

        mode = scenario.get("mode")

        for node in scenario.get("affected_nodes", []):
            disruptions[node] = {
                "delay": delay,
                "threat": threat,
                "reason": scenario.get(
                    "reason",
                    "Scenario disruption active.",
                ),
                "source": "SCENARIO_OVERRIDE",
                "mode": mode,
                "blocked": bool(scenario.get("blocked", False)),
            }

        return disruptions

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_scenario(
        self,
        scenario_id: str,
        graph=None,
    ) -> Dict[str, Any]:
        """
        Validate a scenario definition.

        If a graph is supplied, affected physical IDs are also checked
        against the graph's physical_id node attributes.
        """

        scenario_id = str(scenario_id).strip().upper()

        if scenario_id not in self.SCENARIOS:
            return {
                "valid": False,
                "errors": [f"Unknown scenario: {scenario_id}"],
            }

        scenario = self.SCENARIOS[scenario_id]
        errors = []

        # Validate mode.
        mode = scenario.get("mode")
        if mode not in self.VALID_MODES:
            errors.append(
                f"Invalid transport mode '{mode}' "
                f"for scenario {scenario_id}."
            )

        # Validate threat.
        try:
            threat = float(scenario.get("threat_level"))
            if not 0.0 <= threat <= 1.0:
                errors.append(
                    f"Threat level must be between 0 and 1: {threat}"
                )
        except (TypeError, ValueError):
            errors.append("Threat level is not numeric.")

        # Validate delay.
        try:
            delay = float(scenario.get("delay_hours"))
            if delay < 0:
                errors.append(
                    f"Delay cannot be negative: {delay}"
                )
        except (TypeError, ValueError):
            errors.append("Delay is not numeric.")

        # Validate affected nodes.
        affected_nodes = scenario.get("affected_nodes")

        if not isinstance(affected_nodes, list) or not affected_nodes:
            errors.append("Scenario has no affected nodes.")

        # Optional graph validation.
        if graph is not None:
            physical_ids = {
                data.get("physical_id")
                for _, data in graph.nodes(data=True)
            }

            for node in affected_nodes or []:
                if node not in physical_ids:
                    errors.append(
                        f"Affected physical node '{node}' "
                        f"does not exist in the routing graph."
                    )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Scenario listing
    # ------------------------------------------------------------------

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """
        Return all scenarios in a frontend-friendly format.
        """

        scenarios = []

        for scenario_id, scenario in self.SCENARIOS.items():
            item = deepcopy(scenario)
            item["id"] = scenario_id
            scenarios.append(item)

        return scenarios
```
