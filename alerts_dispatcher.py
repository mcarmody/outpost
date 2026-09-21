"""Outpost Webhook Alert Dispatcher & Rate-Limiter.

Formats and dispatches rich Discord/Slack webhooks when high-value target species
are detected, with per-species/stream rate-limiting to prevent alert storms.
"""

import time
from typing import Any, Dict, List, Optional
import requests

# Color codes for Discord embed sidebar
SPECIES_COLORS = {
    "marine": 0x06B6D4,      # Cyan
    "avian": 0x10B981,       # Emerald
    "mammal": 0xF59E0B,      # Amber
    "predator": 0xEF4444,    # Rose
}

SPECIES_CATEGORY = {
    "garibaldi": "marine",
    "giant_sea_bass": "marine",
    "kelp_bass": "marine",
    "bald_eagle": "avian",
    "blue_jay": "avian",
    "osprey": "avian",
    "brown_bear": "mammal",
    "salmon": "marine",
    "wolf": "predator",
}


class AlertDispatcher:
    def __init__(self, default_cooldown_seconds: float = 300.0, webhook_url: Optional[str] = None):
        self.default_cooldown = default_cooldown_seconds
        self.webhook_url = webhook_url
        self.last_dispatched: Dict[str, float] = {}  # key: f"{stream_id}:{species}" -> timestamp
        self.dispatch_history: List[Dict[str, Any]] = []

    def _get_cooldown_key(self, stream_id: str, species: str) -> str:
        return f"{stream_id.lower()}:{species.lower()}"

    def is_rate_limited(self, stream_id: str, species: str, cooldown_override: Optional[float] = None) -> bool:
        """Checks if a notification for this stream and species is currently in cooldown."""
        key = self._get_cooldown_key(stream_id, species)
        last_time = self.last_dispatched.get(key, 0.0)
        cooldown = cooldown_override if cooldown_override is not None else self.default_cooldown
        return (time.time() - last_time) < cooldown

    def format_discord_payload(self, event: Dict[str, Any], stream_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Formats a rich Discord webhook embed payload."""
        species = event.get("species", "Unknown").replace("_", " ").title()
        stream_id = event.get("stream_id", "unknown_stream")
        stream_name = stream_info.get("name", stream_id) if stream_info else stream_id
        provider = stream_info.get("provider", "Explore.org") if stream_info else "Explore.org"
        conf_pct = round(event.get("confidence", 0.0) * 100, 1)

        raw_species = event.get("species", "").lower()
        category = SPECIES_CATEGORY.get(raw_species, "mammal")
        color = SPECIES_COLORS.get(category, 0x6366F1)

        embed = {
            "title": f"🚨 Wildlife Sighting: {species}",
            "description": f"Target detection on **{stream_name}** ({provider})",
            "color": color,
            "fields": [
                {"name": "Species", "value": species, "inline": True},
                {"name": "Confidence", "value": f"{conf_pct}%", "inline": True},
                {"name": "Stream ID", "value": f"`{stream_id}`", "inline": True},
            ],
            "timestamp": event.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            "footer": {"text": "Project Outpost · RTX 4090 Vision Core"},
        }

        if event.get("snapshot_url"):
            embed["image"] = {"url": event["snapshot_url"]}

        if event.get("bbox"):
            bbox_str = f"[{', '.join(map(str, event['bbox']))}]"
            embed["fields"].append({"name": "Bounding Box (XYXY)", "value": f"`{bbox_str}`", "inline": False})

        return {
            "username": "Outpost Wildlife Beacon",
            "avatar_url": "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=128",
            "embeds": [embed],
        }

    def dispatch(
        self,
        event: Dict[str, Any],
        stream_info: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Dispatches webhook if not rate-limited."""
        stream_id = event.get("stream_id", "default")
        species = event.get("species", "default")

        if not force and self.is_rate_limited(stream_id, species):
            return {
                "status": "rate_limited",
                "message": f"Alert for {species} on {stream_id} suppressed by cooldown.",
                "cooldown_remaining_sec": round(
                    self.default_cooldown - (time.time() - self.last_dispatched[self._get_cooldown_key(stream_id, species)]),
                    1,
                ),
            }

        payload = self.format_discord_payload(event, stream_info)
        sent = False

        if self.webhook_url and not self.webhook_url.startswith("mock://"):
            try:
                resp = requests.post(self.webhook_url, json=payload, timeout=3.0)
                sent = resp.status_code in [200, 204]
            except Exception as e:
                print(f"[!] Warning: Webhook dispatch error: {e}")
        else:
            # Mock / dry-run delivery
            sent = True

        now = time.time()
        self.last_dispatched[self._get_cooldown_key(stream_id, species)] = now
        record = {
            "dispatched_at": now,
            "stream_id": stream_id,
            "species": species,
            "confidence": event.get("confidence"),
            "delivered": sent,
            "payload": payload,
        }
        self.dispatch_history.append(record)

        return {
            "status": "dispatched",
            "species": species,
            "stream_id": stream_id,
            "delivered": sent,
        }
