"""Ecological Biodiversity & Species Abundance Analytics Engine for Project Outpost.

Calculates real-time quantitative biodiversity metrics and ecological health indices
across monitored marine reef, avian feeder, and subarctic riparian wildlife streams:
- Species Richness (S) and Total Abundance (N)
- Shannon-Wiener Diversity Index (H')
- Simpson's Diversity Index (D), Index of Diversity (1-D), and Reciprocal Index (1/D)
- Pielou's Evenness Index (J')
- Margalef's Species Richness Index (D_mg)
- Berger-Parker Dominance Index (d)
- Trophic and Ecological Guild Distribution (Marine, Avian, Predator, Herbivore)
- Stream-by-Stream Biome Diversity Comparison and Ranking
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Species to ecological guild mapping
SPECIES_GUILDS: Dict[str, str] = {
    # Marine Kelp Forest & Reef
    "garibaldi": "marine_reef",
    "giant_sea_bass": "marine_apex",
    "kelp_bass": "marine_reef",
    "leopard_shark": "marine_apex",
    "bat_ray": "marine_benthic",
    "california_spiny_lobster": "marine_benthic",
    "california_sheephead": "marine_reef",
    "harbor_seal": "marine_mammal",
    "sea_otter": "marine_mammal",
    "salmon": "anadromous_prey",
    "sockeye_salmon": "anadromous_prey",
    
    # Avian Feeder & Canopy
    "bald_eagle": "avian_raptor",
    "osprey": "avian_raptor",
    "peregrine_falcon": "avian_raptor",
    "red_tailed_hawk": "avian_raptor",
    "blue_jay": "avian_forager",
    "northern_cardinal": "avian_forager",
    "black_capped_chickadee": "avian_forager",
    "american_goldfinch": "avian_forager",
    "mourning_dove": "avian_forager",
    "downy_woodpecker": "avian_forager",
    "pileated_woodpecker": "avian_forager",
    
    # Terrestrial Subarctic & Forest
    "brown_bear": "terrestrial_apex",
    "grizzly_bear": "terrestrial_apex",
    "grey_wolf": "terrestrial_apex",
    "wolf": "terrestrial_apex",
    "coyote": "terrestrial_mesopredator",
    "red_fox": "terrestrial_mesopredator",
    "river_otter": "riparian_predator",
    "moose": "terrestrial_herbivore",
    "elk": "terrestrial_herbivore",
    "black_bear": "terrestrial_omnivore",
}

GUILD_CATEGORIES: Dict[str, str] = {
    "marine_reef": "Marine Ecosystem",
    "marine_apex": "Marine Ecosystem",
    "marine_benthic": "Marine Ecosystem",
    "marine_mammal": "Marine Ecosystem",
    "anadromous_prey": "Marine Ecosystem",
    "avian_raptor": "Avian Biome",
    "avian_forager": "Avian Biome",
    "terrestrial_apex": "Terrestrial Carnivore",
    "terrestrial_mesopredator": "Terrestrial Carnivore",
    "riparian_predator": "Terrestrial Carnivore",
    "terrestrial_herbivore": "Terrestrial Herbivore",
    "terrestrial_omnivore": "Terrestrial Herbivore",
}


def normalize_species_name(species: str) -> str:
    """Normalizes species strings into snake_case lookup keys."""
    return species.strip().lower().replace(" ", "_").replace("-", "_")


def classify_guild(species: str) -> Tuple[str, str]:
    """Returns (specific_guild, broad_category) for a species."""
    norm = normalize_species_name(species)
    guild = SPECIES_GUILDS.get(norm, "general_wildlife")
    category = GUILD_CATEGORIES.get(guild, "General Wildlife")
    return guild, category


def calculate_biodiversity_metrics(species_counts: Dict[str, int]) -> Dict[str, Any]:
    """Calculates comprehensive quantitative ecological biodiversity metrics.
    
    Parameters:
        species_counts: Mapping of species_name -> total_sightings_count
    
    Returns:
        Dict containing Shannon H', Simpson's D, Pielou's J', Margalef's D_mg,
        species richness, dominance metrics, and ecological health status.
    """
    total_abundance = sum(species_counts.values())
    species_richness = len(species_counts)

    if total_abundance == 0 or species_richness == 0:
        return {
            "species_richness_s": 0,
            "total_abundance_n": 0,
            "shannon_diversity_index_h": 0.0,
            "simpson_diversity_index_d": 0.0,
            "simpson_index_of_diversity": 0.0,
            "simpson_reciprocal_index": 0.0,
            "pielou_evenness_j": 0.0,
            "margalef_richness_index": 0.0,
            "berger_parker_dominance": 0.0,
            "dominant_species": None,
            "ecological_health": "insufficient_data",
            "guild_distribution": {},
            "broad_category_distribution": {},
            "species_breakdown": [],
        }

    # Proportions p_i = n_i / N
    proportions: Dict[str, float] = {}
    shannon_h = 0.0
    simpson_d = 0.0
    dominant_species = None
    max_count = -1

    species_breakdown = []

    for sp, count in species_counts.items():
        if count > max_count:
            max_count = count
            dominant_species = sp

        pi = count / total_abundance
        proportions[sp] = pi

        if pi > 0.0:
            shannon_h -= pi * math.log(pi)
            simpson_d += pi * pi

        guild, category = classify_guild(sp)
        species_breakdown.append({
            "species": sp,
            "count": count,
            "relative_abundance_pct": round(pi * 100.0, 2),
            "guild": guild,
            "category": category,
        })

    # Sort breakdown by count descending
    species_breakdown.sort(key=lambda x: x["count"], reverse=True)

    # Derived ecological indices
    simpson_1_minus_d = 1.0 - simpson_d
    simpson_reciprocal = round(1.0 / simpson_d, 2) if simpson_d > 0 else 0.0

    # Pielou's Evenness J' = H' / ln(S)
    if species_richness > 1:
        pielou_j = shannon_h / math.log(species_richness)
    elif species_richness == 1:
        pielou_j = 1.0
    else:
        pielou_j = 0.0

    # Margalef's Richness D_mg = (S - 1) / ln(N)
    if total_abundance > 1:
        margalef_d = (species_richness - 1) / math.log(total_abundance)
    else:
        margalef_d = 0.0

    # Berger-Parker Dominance d = N_max / N
    berger_parker = max_count / total_abundance if total_abundance > 0 else 0.0

    # Ecological Health Classification
    if total_abundance < 5:
        health_status = "insufficient_data"
    elif species_richness == 1:
        health_status = "low_diversity_monoculture"
    elif shannon_h >= 2.0 and pielou_j >= 0.70:
        health_status = "exceptional_biodiversity"
    elif shannon_h >= 1.4:
        health_status = "healthy_biodiverse"
    elif shannon_h >= 0.8:
        health_status = "moderate_diversity"
    else:
        health_status = "low_diversity_monoculture"

    # Guild & Category distribution
    guild_dist: Dict[str, int] = {}
    broad_dist: Dict[str, int] = {}
    for item in species_breakdown:
        g = item["guild"]
        c = item["category"]
        guild_dist[g] = guild_dist.get(g, 0) + item["count"]
        broad_dist[c] = broad_dist.get(c, 0) + item["count"]

    return {
        "species_richness_s": species_richness,
        "total_abundance_n": total_abundance,
        "shannon_diversity_index_h": round(shannon_h, 3),
        "simpson_diversity_index_d": round(simpson_d, 3),
        "simpson_index_of_diversity": round(simpson_1_minus_d, 3),
        "simpson_reciprocal_index": simpson_reciprocal,
        "pielou_evenness_j": round(pielou_j, 3),
        "margalef_richness_index": round(margalef_d, 3),
        "berger_parker_dominance": round(berger_parker, 3),
        "dominant_species": dominant_species,
        "ecological_health": health_status,
        "guild_distribution": guild_dist,
        "broad_category_distribution": broad_dist,
        "species_breakdown": species_breakdown,
    }


class BiodiversityEngine:
    """Orchestrates continuous ecological biodiversity telemetry across Outpost streams."""

    def __init__(self, db_module: Any):
        self.db = db_module

    def get_overall_biodiversity(
        self,
        stream_id: Optional[str] = None,
        db_path: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Calculates global or stream-scoped biodiversity telemetry from SQLite storage."""
        stats = self.db.get_species_stats_summary(stream_id=stream_id, db_path=db_path)
        species_counts = {row["species"]: row["count"] for row in stats}
        metrics = calculate_biodiversity_metrics(species_counts)
        metrics["stream_id"] = stream_id or "all_streams"
        metrics["timestamp"] = time.time()
        return metrics

    def get_comparative_stream_ranking(
        self,
        stream_ids: Optional[List[str]] = None,
        db_path: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Compares and ranks registered wildlife streams by biodiversity health."""
        target_streams = stream_ids or ["anacapa_kelp_01", "cornell_feeder_01", "katmai_brooks_01"]
        results = []

        for sid in target_streams:
            stream_metrics = self.get_overall_biodiversity(stream_id=sid, db_path=db_path)
            results.append({
                "stream_id": sid,
                "species_richness": stream_metrics["species_richness_s"],
                "total_abundance": stream_metrics["total_abundance_n"],
                "shannon_h": stream_metrics["shannon_diversity_index_h"],
                "simpson_diversity": stream_metrics["simpson_index_of_diversity"],
                "evenness_j": stream_metrics["pielou_evenness_j"],
                "dominant_species": stream_metrics["dominant_species"],
                "ecological_health": stream_metrics["ecological_health"],
                "dominant_guild": next(iter(stream_metrics.get("broad_category_distribution", {})), "Unknown"),
            })

        # Rank streams by Shannon Diversity Index (H') descending
        results.sort(key=lambda x: x["shannon_h"], reverse=True)

        return {
            "timestamp": time.time(),
            "ranked_streams": results,
            "highest_biodiversity_stream": results[0]["stream_id"] if results else None,
            "total_streams_audited": len(results),
        }
