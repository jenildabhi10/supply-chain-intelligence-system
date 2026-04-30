"""
Port importance index — trade-exposure weight for each monitored port.

The impact_index captures how much disruption at a given port matters to
the overall supply chain. A disruption at LA/LB (40% of US container imports)
is categorically more impactful than one at a smaller gateway port.

Weights are derived from publicly available 2023 U.S. port TEU volumes
(Bureau of Transportation Statistics / PIERS data). Normalised to [0, 1]
so they can be used directly as a model input alongside p_disruption and
anomaly_score.

If a port_id is not in the table, a default of 0.05 is returned
(smallest monitored port level) rather than crashing.
"""

from __future__ import annotations

# Approximate 2023 annual TEU throughput (millions) for each monitored port.
# Source: BTS/PIERS public summaries + port authority annual reports.
_RAW_TEU_M: dict[str, float] = {
    "la_lb":      19.3,   # combined San Pedro Bay complex
    "ny_nj":       8.9,
    "savannah":    5.8,
    "seattle":     3.7,   # combined Seattle + Tacoma
    "houston":     3.2,
    "charleston":  2.8,
    "norfolk":     2.9,
    "oakland":     2.5,
    "miami":       1.2,
    "baltimore":   0.9,
}

_MAX_TEU = max(_RAW_TEU_M.values())

# Normalised [0, 1] impact index — la_lb = 1.0 by construction
IMPACT_INDEX: dict[str, float] = {
    port_id: round(teu / _MAX_TEU, 4)
    for port_id, teu in _RAW_TEU_M.items()
}

_DEFAULT_IMPACT = 0.05


def get_impact_index(port_id: str) -> float:
    """Return the normalised trade-exposure weight for a port."""
    return IMPACT_INDEX.get(port_id, _DEFAULT_IMPACT)
