```python
import re
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import feedparser


class DynamicNewsIngestor:
    """
    Supplychainer Stage 1: Dynamic News Ingestion.

    Fetches location + transport-specific intelligence from Google News RSS,
    caches successful results, filters stale/duplicate headlines, and falls
    back to neutral operational conditions when live intelligence is
    unavailable.

    The ingestor does NOT calculate threat scores. That remains the job of
    NLP + CARF.
    """

    CACHE_TTL = 900          # 15 minutes
    REQUEST_TIMEOUT = 3.0    # seconds
    MAX_HEADLINES = 5
    MAX_CONTENT_LENGTH = 4000
    MAX_ENTRY_AGE = 72 * 3600  # Ignore articles older than 72 hours.

    # Neutral fallbacks deliberately avoid describing a disruption.
    FALLBACK_NEWS = {
        "sea": "No verified maritime disruption detected.",
        "air": "No verified aviation disruption detected.",
        "road": "No verified road disruption detected.",
        "rail": "No verified rail disruption detected.",
    }

    VALID_MODES = {"sea", "air", "road", "rail"}

    def __init__(self):
        # {normalized_query: (timestamp, content)}
        self.cache: Dict[str, Tuple[float, str]] = {}

        # Metadata is useful for debugging/audit output.
        self.last_status: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_location(location: str) -> str:
        """
        Convert arbitrary location text into a stable query component.
        """

        if not location:
            return "global logistics"

        location = str(location).strip()

        # Collapse repeated whitespace.
        location = re.sub(r"\s+", " ", location)

        return location

    @staticmethod
    def _normalize_mode(transport_mode: str) -> str:
        mode = str(transport_mode or "").strip().lower()

        if mode not in DynamicNewsIngestor.VALID_MODES:
            return "road"

        return mode

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize RSS text without trying to perform NLP here.
        """

        if not text:
            return ""

        text = re.sub(r"<[^>]+>", " ", str(text))
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, location: str, mode: str) -> str:
        return f"{location.lower()}::{mode.lower()}"

    def _get_cached(
        self,
        cache_key: str,
        now: float,
    ) -> Optional[str]:

        cached = self.cache.get(cache_key)

        if cached is None:
            return None

        timestamp, content = cached

        if now - timestamp < self.CACHE_TTL:
            return content

        # Remove expired cache entries.
        self.cache.pop(cache_key, None)

        return None

    # ------------------------------------------------------------------
    # RSS processing
    # ------------------------------------------------------------------

    def _entry_timestamp(self, entry) -> Optional[float]:
        """
        Extract a Unix timestamp from a feed entry when available.
        """

        for field in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, field, None)

            if parsed is None:
                continue

            try:
                return time.mktime(parsed)
            except (TypeError, OverflowError, ValueError):
                pass

        return None

    def _is_fresh(self, entry, now: float) -> bool:
        timestamp = self._entry_timestamp(entry)

        # If the feed doesn't provide a usable timestamp, keep the entry.
        # We cannot prove that it is stale.
        if timestamp is None:
            return True

        age = now - timestamp

        # Future timestamps are allowed because feed clocks can be slightly
        # ahead of the local system.
        if age < 0:
            return True

        return age <= self.MAX_ENTRY_AGE

    def _format_entry(self, entry) -> str:
        """
        Combine headline and summary where available.
        """

        title = self._clean_text(
            getattr(entry, "title", "")
        )

        summary = self._clean_text(
            getattr(entry, "summary", "")
        )

        if title and summary:
            return f"{title} — {summary}"

        return title or summary

    def _extract_headlines(
        self,
        entries: List,
        now: float,
    ) -> List[str]:

        results = []
        seen = set()

        for entry in entries:
            if not self._is_fresh(entry, now):
                continue

            text = self._format_entry(entry)

            if not text:
                continue

            # Basic duplicate normalization.
            dedupe_key = re.sub(
                r"[^a-z0-9]+",
                " ",
                text.lower(),
            ).strip()

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            results.append(text)

            if len(results) >= self.MAX_HEADLINES:
                break

        return results

    # ------------------------------------------------------------------
    # Public ingestion API
    # ------------------------------------------------------------------

    def get_latest_news(
        self,
        location: str,
        transport_mode: str,
    ) -> str:
        """
        Return recent intelligence for a geographic location and mode.

        The return value remains a string for compatibility with the
        existing NLP/CARF pipeline.
        """

        location = self._normalize_location(location)
        mode = self._normalize_mode(transport_mode)

        query = f"{location} {mode} logistics disruption"
        cache_key = self._cache_key(location, mode)

        now = time.time()

        # --------------------------------------------------------------
        # 1. Cache
        # --------------------------------------------------------------

        cached = self._get_cached(cache_key, now)

        if cached is not None:
            self.last_status[cache_key] = {
                "status": "cache",
                "location": location,
                "mode": mode,
                "timestamp": now,
            }

            return cached

        # --------------------------------------------------------------
        # 2. Live RSS
        # --------------------------------------------------------------

        start = time.perf_counter()

        print(
            f"[TRACE] STEP 7: News ingestion started "
            f"for {location} [{mode}]"
        )

        try:
            encoded_query = urllib.parse.quote_plus(query)

            rss_url = (
                "https://news.google.com/rss/search"
                f"?q={encoded_query}"
                "&hl=en-US"
                "&gl=US"
                "&ceid=US:en"
            )

            # feedparser supports request-related kwargs through its
            # underlying HTTP layer in some versions, but the installed
            # version may vary. Therefore we avoid modifying global socket
            # defaults and rely on the feed parser's normal network handling.
            feed = feedparser.parse(rss_url)

            if getattr(feed, "bozo", False):
                bozo_exception = getattr(
                    feed,
                    "bozo_exception",
                    None,
                )

                print(
                    f"[TRACE] RSS warning for {location}: "
                    f"{bozo_exception}"
                )

            entries = getattr(feed, "entries", []) or []

            headlines = self._extract_headlines(
                entries,
                now,
            )

            if headlines:
                content = " | ".join(headlines)

                # Prevent unexpectedly large NLP inputs.
                content = content[:self.MAX_CONTENT_LENGTH]

                self.cache[cache_key] = (
                    now,
                    content,
                )

                self.last_status[cache_key] = {
                    "status": "live",
                    "location": location,
                    "mode": mode,
                    "articles": len(headlines),
                    "timestamp": now,
                    "latency_ms": round(
                        (time.perf_counter() - start) * 1000,
                        2,
                    ),
                }

                print(
                    "[TRACE] STEP 8: News ingestion complete "
                    f"({time.perf_counter() - start:.4f}s, "
                    f"{len(headlines)} articles)"
                )

                return content

        except Exception as exc:
            print(
                f"[TRACE] News ingestion error for "
                f"{location}/{mode}: {exc}"
            )

        # --------------------------------------------------------------
        # 3. Neutral fallback
        # --------------------------------------------------------------

        fallback = self.FALLBACK_NEWS.get(
            mode,
            "No verified logistics disruption detected.",
        )

        self.last_status[cache_key] = {
            "status": "fallback",
            "location": location,
            "mode": mode,
            "timestamp": now,
            "reason": "Live RSS unavailable or contained no usable articles.",
        }

        print(
            "[TRACE] STEP 8: News ingestion complete "
            "(neutral fallback used)"
        )

        return fallback

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(
        self,
        location: str,
        transport_mode: str,
    ) -> Dict[str, object]:

        location = self._normalize_location(location)
        mode = self._normalize_mode(transport_mode)

        cache_key = self._cache_key(location, mode)

        return dict(
            self.last_status.get(
                cache_key,
                {
                    "status": "never_requested",
                    "location": location,
                    "mode": mode,
                },
            )
        )

    def clear_cache(self) -> None:
        """
        Clear all cached intelligence.
        """

        self.cache.clear()
        self.last_status.clear()
```
