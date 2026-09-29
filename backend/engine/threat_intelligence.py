```python
import os
import re
import json
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch


# ============================================================
# Production Artifacts
# ============================================================

MODEL_PATH = "./Execution/risk_model.pkl"
ENCODER_PATH = "./Execution/label_encoders.pkl"
NLP_ANCHORS_PATH = "./Execution/nlp_anchors.pt"
CALIBRATION_PATH = "./Execution/calibration_profiles.json"


# ============================================================
# Deterministic operational priors
# Used when the ML model is unavailable or inference fails.
# ============================================================

MODE_PRIORS = {
    "road": 2.5,
    "sea": 48.0,
    "air": 12.0,
    "rail": 18.0,
}


# ============================================================
# Threat Intelligence / Quantile ML Predictor
# ============================================================

class ThreatIntelligencePredictor:
    """
    Supplychainer Quantile ML Decision Brain.

    Pipeline:
        Route conditions
            ↓
        Encoded ML features
            ↓
        p85 delay prediction
            ↓
        Historical calibration bounds
            ↓
        Defensible final delay

    The predictor is deliberately defensive:
    - Missing artifacts do not crash the application.
    - Unknown categorical values fall back safely.
    - Invalid model predictions are rejected.
    - Calibration values are clamped to sane ranges.
    """

    def __init__(self, lazy_load: bool = False):
        self.is_trained = False
        self.model = None
        self.encoders = {}
        self.profiles = {}

        self.hub_map = {
            "Seattle": "Seattle Port",
            "Portland": "Portland Terminal",
            "San Francisco": "San Francisco Port",
            "Los Angeles": "Los Angeles Port",
            "Salt Lake City": "Salt Lake City Hub",
            "Denver": "Denver Terminal",
            "Phoenix": "Phoenix Logistics",
            "Dallas": "Dallas Corridor",
            "Houston": "Houston Port",
            "Chicago": "Chicago Rail Hub",
            "St. Louis": "St. Louis Hub",
            "Atlanta": "Atlanta Air Hub",
            "Miami": "Miami Port",
            "New York": "New York Port",
            "Boston": "Boston Terminal",
            "Mumbai": "Mumbai Port",
            "Kochi": "Kochi Port",
            "Delhi": "Delhi Air Cargo",
            "Chennai": "Chennai Port",
        }

        if not lazy_load:
            self.warmup()

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    def warmup(self):
        """Load production artifacts once."""
        if self.is_trained:
            return

        print("[PREDICTOR] Starting warmup...")

        if not os.path.exists(MODEL_PATH):
            print(
                "[PREDICTOR] Production model missing. "
                "Using deterministic fallback mode."
            )
            return

        if not os.path.exists(ENCODER_PATH):
            print(
                "[PREDICTOR] Label encoders missing. "
                "Using deterministic fallback mode."
            )
            return

        try:
            self.model = joblib.load(MODEL_PATH)
            self.encoders = joblib.load(ENCODER_PATH)

            if not isinstance(self.encoders, dict):
                raise TypeError("Label encoder artifact is not a dictionary.")

            self.is_trained = True

            # Calibration profiles are optional.
            if os.path.exists(CALIBRATION_PATH):
                try:
                    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                        loaded_profiles = json.load(f)

                    if isinstance(loaded_profiles, dict):
                        self.profiles = loaded_profiles
                    else:
                        print(
                            "[PREDICTOR] Calibration file has invalid format. "
                            "Using defensive defaults."
                        )
                        self.profiles = {}

                except Exception as exc:
                    print(
                        f"[PREDICTOR] Calibration load failed: {exc}. "
                        "Using defensive defaults."
                    )
                    self.profiles = {}

            print(
                f"[PREDICTOR] Loaded production model. "
                f"Calibration profiles: {len(self.profiles)}"
            )

        except Exception as exc:
            print(
                f"[PREDICTOR] Production artifact loading failed: {exc}. "
                "Using deterministic fallback mode."
            )

            self.model = None
            self.encoders = {}
            self.is_trained = False

    # --------------------------------------------------------
    # Feature encoding
    # --------------------------------------------------------

    def _encode_feature(self, value: str, key: str) -> int:
        """
        Encode a categorical feature.

        Unknown values fall back to the first known class instead of
        crashing the entire routing request.
        """

        if key not in self.encoders:
            raise KeyError(f"Encoder '{key}' is missing.")

        encoder = self.encoders[key]
        classes = list(encoder.classes_)

        if not classes:
            raise ValueError(f"Encoder '{key}' contains no classes.")

        value = str(value)

        # Resolve city names into the hub/node names used during training.
        if key in {"Origin_Node", "Destination_Node"}:
            resolved = self.hub_map.get(value, value)
        else:
            resolved = value

        # Exact match.
        if resolved in classes:
            return int(encoder.transform([resolved])[0])

        # Case-insensitive match.
        lowered = {
            str(cls).strip().lower(): cls
            for cls in classes
        }

        match = lowered.get(resolved.strip().lower())

        if match is not None:
            return int(encoder.transform([match])[0])

        # Safe deterministic fallback.
        fallback_class = classes[0]

        print(
            f"[PREDICTOR] Unknown {key}='{value}'. "
            f"Using encoder fallback='{fallback_class}'."
        )

        return int(encoder.transform([fallback_class])[0])

    # --------------------------------------------------------
    # Deterministic fallback
    # --------------------------------------------------------

    def _fallback_prediction(
        self,
        transport_mode: str,
        reason: str
    ) -> Dict[str, Any]:
        """Return a safe deterministic delay when ML is unavailable."""

        mode = str(transport_mode).lower().strip()
        delay = MODE_PRIORS.get(mode, 12.0)

        return {
            "raw_model_prediction": delay,
            "calibrated_delay": delay,
            "baseline_systemic_friction": delay,
            "final_delay_presented": delay,
            "calibration_reason": reason,
            "p_quantile": 0.85,
            "is_defensible": True,
        }

    # --------------------------------------------------------
    # p85 prediction
    # --------------------------------------------------------

    def predict_worst_case_delay(
        self,
        origin: str,
        destination: str,
        transport_mode: str,
        leg_type: str = "Global_Freight",
        condition_flag: str = "Clear",
        nlp_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Stage 4:

            Input features
                ↓
            Raw p85 prediction
                ↓
            Historical floor/cap calibration
                ↓
            Final defensible delay

        The returned `final_delay_presented` is the value that should
        be consumed by the route recommender.
        """

        mode = str(transport_mode).lower().strip()

        # ----------------------------------------------------
        # Defensive input normalization
        # ----------------------------------------------------

        try:
            nlp_score = float(nlp_score)
        except (TypeError, ValueError):
            nlp_score = 0.0

        nlp_score = float(np.clip(nlp_score, 0.0, 1.0))

        # ----------------------------------------------------
        # ML unavailable → deterministic operational prior
        # ----------------------------------------------------

        if not self.is_trained or self.model is None:
            return self._fallback_prediction(
                mode,
                "Deterministic Operational Prior (ML unavailable)"
            )

        try:
            # ------------------------------------------------
            # Encode model features
            # ------------------------------------------------

            feat_origin = self._encode_feature(
                origin,
                "Origin_Node"
            )

            feat_dest = self._encode_feature(
                destination,
                "Destination_Node"
            )

            feat_mode = self._encode_feature(
                transport_mode,
                "Transport_Mode"
            )

            feat_leg = self._encode_feature(
                leg_type,
                "Leg_Type"
            )

            feat_cond = self._encode_feature(
                condition_flag,
                "Condition_Flag"
            )

            X_input = pd.DataFrame(
                [{
                    "Leg_Type": feat_leg,
                    "Origin_Node": feat_origin,
                    "Destination_Node": feat_dest,
                    "Transport_Mode": feat_mode,
                    "Condition_Flag": feat_cond,
                    "NLP_Severity_Score": nlp_score,
                }]
            )

            # ------------------------------------------------
            # Raw p85 prediction
            # ------------------------------------------------

            raw_prediction = float(self.model.predict(X_input)[0])

            if not np.isfinite(raw_prediction):
                raise ValueError(
                    f"Model returned invalid prediction: {raw_prediction}"
                )

            # Delay cannot be negative.
            raw_prediction = max(0.0, raw_prediction)

            # ------------------------------------------------
            # Calibration profile
            # ------------------------------------------------

            profile = self.profiles.get(
                mode,
                {
                    "floor": 0.0,
                    "cap": 240.0,
                }
            )

            try:
                floor = float(profile.get("floor", 0.0))
                cap = float(profile.get("cap", 240.0))
            except (TypeError, ValueError):
                floor = 0.0
                cap = 240.0

            # Defensive calibration bounds.
            floor = max(0.0, floor)
            cap = max(floor, cap)

            # ------------------------------------------------
            # Apply p95 cap
            # ------------------------------------------------

            calibrated_delay = min(
                max(0.0, raw_prediction),
                cap
            )

            # ------------------------------------------------
            # Restore systemic p5 friction
            # ------------------------------------------------

            final_delay = max(
                calibrated_delay,
                floor
            )

            # ------------------------------------------------
            # Explainability
            # ------------------------------------------------

            if final_delay == floor and calibrated_delay < floor:
                reason = (
                    "Baseline Operational Friction "
                    f"(Historical p5: {floor}h)"
                )

            elif calibrated_delay < raw_prediction:
                reason = (
                    "Operational Cap Applied "
                    f"(Historical p95 Bound: {cap}h)"
                )

            elif raw_prediction > floor:
                reason = "Quantile Disruption Prediction (p85 Risk)"

            else:
                reason = "Optimal Flow"

            return {
                "raw_model_prediction": round(raw_prediction, 2),
                "calibrated_delay": round(calibrated_delay, 2),
                "baseline_systemic_friction": round(floor, 2),
                "final_delay_presented": round(final_delay, 2),
                "calibration_reason": reason,
                "p_quantile": 0.85,
                "is_defensible": True,
            }

        except Exception as exc:
            print(
                f"[PREDICTOR] Calibration inference error: {exc}. "
                "Falling back to deterministic operational prior."
            )

            return self._fallback_prediction(
                mode,
                f"ML inference fallback: {type(exc).__name__}"
            )


# ============================================================
# Contrastive NLP Engine
# ============================================================

class ContrastiveNLPEngine:
    """
    Stage 2: Contrastive semantic threat detection.

    Score calculation:

        disaster_similarity
                 -
        safe_similarity
                 ↓
              margin
                 ↓
          noise threshold
                 ↓
          calibration multiplier
                 ↓
          [0, 1] threat score
    """

    def __init__(self, lazy_load: bool = False):
        self._ready = False

        # Small semantic margins are treated as noise.
        self.noise_floor = 0.04

        # Converts cosine-similarity margin into operational risk.
        self.calibration_multiplier = 0.35

        self.model = None
        self.util = None
        self.disaster_matrix = None
        self.safe_matrix = None

        if not lazy_load:
            self.warmup()

    def warmup(self):
        """Load the sentence-transformer and anchor matrices."""

        if self._ready:
            return

        print("[NLP ENGINE] Starting warmup...")

        try:
            from sentence_transformers import SentenceTransformer, util

            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            self.util = util

            if not os.path.exists(NLP_ANCHORS_PATH):
                print(
                    "[NLP ENGINE] Anchor matrix missing. "
                    "NLP engine disabled."
                )
                self._ready = False
                return

            anchors = torch.load(
                NLP_ANCHORS_PATH,
                map_location="cpu"
            )

            if (
                not isinstance(anchors, dict)
                or "disaster_matrix" not in anchors
                or "safe_matrix" not in anchors
            ):
                raise ValueError(
                    "NLP anchor artifact has an invalid format."
                )

            self.disaster_matrix = anchors["disaster_matrix"]
            self.safe_matrix = anchors["safe_matrix"]

            self._ready = True

            print(
                "[NLP ENGINE] Historical disaster/safe "
                "anchor matrices loaded."
            )

        except Exception as exc:
            print(
                f"[NLP ENGINE] Warmup failed: {exc}"
            )

            self._ready = False
            self.model = None
            self.util = None

    def get_semantic_score(self, news_text: str) -> float:
        """
        Convert news text into a normalized semantic threat score.

        Important:
        A positive disaster-vs-safe margin is a threat.
        A negative/weak margin is not.
        """

        if not self._ready:
            return 0.0

        if not news_text or len(news_text.strip()) < 5:
            return 0.0

        try:
            # Split long articles so one huge article does not
            # create an uncontrolled embedding input.
            chunks = [
                news_text[i:i + 256]
                for i in range(0, len(news_text), 256)
            ]

            chunk_embeddings = self.model.encode(
                chunks,
                convert_to_tensor=True
            )

            disaster_scores = self.util.cos_sim(
                chunk_embeddings,
                self.disaster_matrix
            )

            safe_scores = self.util.cos_sim(
                chunk_embeddings,
                self.safe_matrix
            )

            disaster_max = float(
                np.max(
                    disaster_scores.detach().cpu().numpy()
                )
            )

            safe_max = float(
                np.max(
                    safe_scores.detach().cpu().numpy()
                )
            )

            # Positive = more disaster-like than safe-like.
            margin = disaster_max - safe_max

            # Weak/negative evidence is noise.
            if margin <= self.noise_floor:
                return 0.0

            # Convert semantic margin into operational score.
            score = margin * self.calibration_multiplier

            # Never allow NLP to generate an invalid risk.
            return float(
                np.clip(score, 0.0, 1.0)
            )

        except Exception as exc:
            print(
                f"[NLP ENGINE] Semantic scoring failed: {exc}"
            )
            return 0.0


# ============================================================
# CARF — Context-Aware Relevance Filter
# ============================================================

class CARFFilter:
    """
    Stage 3: Context-Aware Relevance Filter.

    CARF answers:

        "Does this news event actually belong to the
         transport mode affected by this route?"

    Behaviour:

    1. Relevant target-mode evidence
       → preserve threat.

    2. Strong evidence for another transport mode and
       no target-mode evidence
       → suppress threat.

    3. Generic disruption with no clear modal evidence
       → preserve threat because the event may be
          multimodal.

    This avoids both:
        - false positives from unrelated transport news
        - false negatives from generic disruptions.
    """

    def __init__(self):
        # Strongly mode-specific keywords.
        #
        # Avoid ambiguous words such as "terminal" and "station"
        # as standalone evidence because they can refer to multiple
        # transport modes.

        self.relevance_map = {
            "air": [
                "airport",
                "airline",
                "flight",
                "airspace",
                "aviation",
                "aircraft",
                "runway",
                "cargo flight",
                "freight flight",
                "air cargo",
                "cargo plane",
            ],

            "sea": [
                "port",
                "seaport",
                "vessel",
                "ship",
                "canal",
                "ocean",
                "maritime",
                "dock",
                "harbor",
                "harbour",
                "container ship",
                "cargo ship",
                "shipping",
                "freight ship",
                "tanker",
                "container vessel",
                "suez canal",
                "panama canal",
            ],

            "rail": [
                "rail",
                "railway",
                "railroad",
                "rail line",
                "rail network",
                "locomotive",
                "train",
                "freight rail",
                "cargo train",
                "rail freight",
                "derailment",
                "rail track",
            ],

            "road": [
                "highway",
                "truck",
                "traffic",
                "bridge",
                "road",
                "roadway",
                "motorway",
                "freight truck",
                "cargo truck",
                "road freight",
                "truck route",
                "road closure",
            ],
        }

        # These are stronger signals that an article is primarily
        # about a particular mode.
        self.strong_mode_keywords = {
            "air": {
                "airport",
                "airline",
                "flight",
                "airspace",
                "aviation",
                "aircraft",
                "runway",
                "air cargo",
                "cargo plane",
            },

            "sea": {
                "port",
                "seaport",
                "vessel",
                "ship",
                "canal",
                "maritime",
                "container ship",
                "cargo ship",
                "tanker",
                "container vessel",
                "suez canal",
                "panama canal",
            },

            "rail": {
                "railway",
                "railroad",
                "locomotive",
                "train",
                "freight rail",
                "cargo train",
                "derailment",
                "rail track",
            },

            "road": {
                "highway",
                "truck",
                "motorway",
                "freight truck",
                "cargo truck",
                "road closure",
                "truck route",
            },
        }

        # Generic words that can describe disruptions across
        # multiple transport modes.
        self.generic_disruption_keywords = {
            "disruption",
            "delay",
            "delays",
            "closure",
            "closed",
            "blocked",
            "blockade",
            "strike",
            "strikes",
            "protest",
            "protests",
            "flood",
            "flooding",
            "earthquake",
            "storm",
            "hurricane",
            "cyclone",
            "fire",
            "explosion",
            "sanctions",
            "war",
            "conflict",
            "shortage",
            "supply chain",
            "supply-chain",
            "logistics",
        }

    # --------------------------------------------------------
    # Text normalization
    # --------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and casing."""

        text = str(text or "").lower()

        # Normalize Unicode-ish punctuation into spaces.
        text = re.sub(r"[^a-z0-9\s-]", " ", text)

        # Collapse repeated whitespace.
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize normalized text."""

        return set(
            re.findall(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                text.lower()
            )
        )

    # --------------------------------------------------------
    # Keyword matching
    # --------------------------------------------------------

    def _contains_keyword(
        self,
        text: str,
        tokens: set[str],
        keyword: str
    ) -> bool:
        """
        Match both single-word and multi-word keywords.

        Multi-word keywords are matched against normalized text.
        Single-word keywords are matched against tokens.
        """

        keyword = keyword.lower().strip()

        if not keyword:
            return False

        if " " in keyword:
            return keyword in text

        return keyword in tokens

    def _has_mode_evidence(
        self,
        text: str,
        tokens: set[str],
        mode: str
    ) -> bool:
        """Return whether the article contains target-mode evidence."""

        keywords = self.relevance_map.get(mode, [])

        return any(
            self._contains_keyword(
                text,
                tokens,
                keyword
            )
            for keyword in keywords
        )

    def _has_strong_mode_evidence(
        self,
        text: str,
        tokens: set[str],
        mode: str
    ) -> bool:
        """Return whether the article strongly identifies a mode."""

        keywords = self.strong_mode_keywords.get(mode, set())

        return any(
            self._contains_keyword(
                text,
                tokens,
                keyword
            )
            for keyword in keywords
        )

    def _has_generic_disruption_evidence(
        self,
        text: str,
        tokens: set[str]
    ) -> bool:
        """
        Detect disruptions that may affect multiple transport modes.
        """

        return any(
            self._contains_keyword(
                text,
                tokens,
                keyword
            )
            for keyword in self.generic_disruption_keywords
        )

    # --------------------------------------------------------
    # Main CARF filter
    # --------------------------------------------------------

    def apply_filter(
        self,
        semantic_score: float,
        news_context: str,
        transport_mode: str
    ) -> float:
        """
        Apply context-aware modal filtering.

        Rules:

        Target mode present:
            preserve semantic threat.

        Another mode strongly present and target mode absent:
            suppress threat.

        Generic disruption with no specific mode:
            preserve semantic threat.

        No useful evidence:
            conservatively preserve semantic signal rather than
            creating a false negative.
        """

        try:
            semantic_score = float(semantic_score)
        except (TypeError, ValueError):
            return 0.0

        semantic_score = float(
            np.clip(semantic_score, 0.0, 1.0)
        )

        if semantic_score <= 0.0:
            return 0.0

        mode = str(
            transport_mode or ""
        ).lower().strip()

        # Unknown mode → no modal filtering.
        if mode not in self.relevance_map:
            return semantic_score

        text = self._normalize_text(news_context)

        if not text:
            return 0.0

        tokens = self._tokenize(text)

        # ----------------------------------------------------
        # Rule 1: target-mode evidence
        # ----------------------------------------------------

        if self._has_mode_evidence(
            text,
            tokens,
            mode
        ):
            return semantic_score

        # ----------------------------------------------------
        # Rule 2: clearly another mode
        # ----------------------------------------------------

        for other_mode in self.relevance_map:

            if other_mode == mode:
                continue

            if self._has_strong_mode_evidence(
                text,
                tokens,
                other_mode
            ):
                # The article explicitly describes another
                # transport domain and provides no evidence
                # for the requested mode.
                return 0.0

        # ----------------------------------------------------
        # Rule 3: generic disruption
        # ----------------------------------------------------

        if self._has_generic_disruption_evidence(
            text,
            tokens
        ):
            return semantic_score

        # ----------------------------------------------------
        # Rule 4: ambiguous article
        # ----------------------------------------------------
        #
        # Don't manufacture a zero merely because CARF cannot
        # prove modal relevance. The NLP engine already provides
        # the semantic signal; CARF's job is primarily to remove
        # clearly irrelevant cross-modal events.

        return semantic_score

    # --------------------------------------------------------
    # Threat aggregation
    # --------------------------------------------------------

    def max_pool_threats(
        self,
        scores: List[float]
    ) -> float:
        """
        Return the strongest valid threat signal.

        Invalid / NaN values are ignored.
        """

        if not scores:
            return 0.0

        valid_scores = []

        for score in scores:
            try:
                value = float(score)

                if np.isfinite(value):
                    valid_scores.append(
                        float(np.clip(value, 0.0, 1.0))
                    )

            except (TypeError, ValueError):
                continue

        if not valid_scores:
            return 0.0

        return float(
            np.clip(
                np.max(valid_scores),
                0.0,
                1.0
            )
        )
```
