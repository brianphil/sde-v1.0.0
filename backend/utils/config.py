"""Configuration utilities.

Provides:
- `load_model_config()` — loads `model_config.yaml` (YAML preferred, JSON fallback)
- `get_business_parameters(state)` — read business tuning parameters from `SystemState.learning`

This module avoids hardcoding defaults in code and centralizes model parameters.
"""

from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def _default_config_path() -> Path:
    # Prefer backend/config/model_config.yaml within the project
    root = Path(__file__).resolve().parents[2]
    preferred = root / "backend" / "config" / "model_config.yaml"
    if preferred.exists():
        return preferred

    # Fallback: repository root model_config.yaml (legacy)
    fallback = root / "model_config.yaml"
    return fallback


def load_model_config(path: str = None) -> dict:
    """Load model configuration from YAML (preferred) or JSON.

    Returns empty dict if no config found.
    """
    cfg_path = Path(path) if path else _default_config_path()
    if not cfg_path.exists():
        logger.debug(f"Model config not found at {cfg_path}")
        return {}

    text = cfg_path.read_text(encoding="utf-8")
    if YAML_AVAILABLE and cfg_path.suffix in (".yaml", ".yml"):
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            logger.exception("Failed to parse YAML model config")
            return {}

    # Fallback: try JSON
    try:
        return json.loads(text) or {}
    except Exception:
        logger.exception("Failed to parse model config as JSON")
        return {}


def get_business_parameters(state) -> dict:
    """Read business-tunable parameters from system state.

    This intentionally reads from `state.learning` when present. In future,
    this function can be replaced with a database-backed lookup for live tuning.
    """
    if state is None:
        return {}

    try:
        learning = getattr(state, "learning", None)
        if not learning:
            return {}

        # Expose pfa_rules and default_time_windows (if present)
        business = {}
        if getattr(learning, "pfa_rules", None):
            business["pfa_rules"] = learning.pfa_rules

        # Some experiments may store time windows in LearningState
        if getattr(learning, "default_time_windows", None):
            business["time_windows"] = learning.default_time_windows

        return business
    except Exception:
        logger.exception("Failed to read business parameters from state")
        return {}


def apply_vehicle_configurations(
    state, model_config: dict = None, business_overrides: dict = None
) -> None:
    """Apply configuration defaults and business overrides to vehicles in `state.fleet`.

    Precedence (highest -> lowest):
      1) existing attributes on `Vehicle` instance (do not overwrite)
      2) business_overrides per `vehicle_id` (dict)
      3) per-vehicle-type defaults from `model_config['cfa']['vehicle_type_defaults']`
      4) nothing (leave None)

    This mutates the `Vehicle` objects in-place to set economic attributes used
    by CFA (e.g., `fuel_cost_per_km`, `driver_cost_per_hour`).
    """
    if state is None or not hasattr(state, "fleet"):
        return

    cfg = model_config or load_model_config()
    cfa_cfg = cfg.get("cfa", {}) if isinstance(cfg, dict) else {}
    type_defaults = cfa_cfg.get("vehicle_type_defaults", {}) or {}

    # Business overrides may come from argument or from state.learning
    overrides = business_overrides or {}
    try:
        learning = getattr(state, "learning", None)
        if learning and getattr(learning, "vehicle_overrides", None):
            overrides = {**overrides, **learning.vehicle_overrides}
    except Exception:
        logger.debug("No vehicle overrides found in learning state")

    for vid, vehicle in getattr(state, "fleet", {}).items():
        try:
            vtype = getattr(vehicle, "vehicle_type", None)

            # Fuel cost per km
            if getattr(vehicle, "fuel_cost_per_km", None) is None:
                # 1) business override for specific vehicle id
                v_override = (
                    overrides.get(vid, {}) if isinstance(overrides, dict) else {}
                )
                if v_override and v_override.get("fuel_cost_per_km") is not None:
                    vehicle.fuel_cost_per_km = float(v_override.get("fuel_cost_per_km"))
                else:
                    # 2) vehicle-type defaults from model config
                    td = (
                        type_defaults.get(vtype, {})
                        if isinstance(type_defaults, dict)
                        else {}
                    )
                    if td and td.get("fuel_cost_per_km") is not None:
                        vehicle.fuel_cost_per_km = float(td.get("fuel_cost_per_km"))

            # Driver cost per hour
            if getattr(vehicle, "driver_cost_per_hour", None) is None:
                v_override = (
                    overrides.get(vid, {}) if isinstance(overrides, dict) else {}
                )
                if v_override and v_override.get("driver_cost_per_hour") is not None:
                    vehicle.driver_cost_per_hour = float(
                        v_override.get("driver_cost_per_hour")
                    )
                else:
                    td = (
                        type_defaults.get(vtype, {})
                        if isinstance(type_defaults, dict)
                        else {}
                    )
                    if td and td.get("driver_cost_per_hour") is not None:
                        vehicle.driver_cost_per_hour = float(
                            td.get("driver_cost_per_hour")
                        )

        except Exception:
            logger.exception(f"Failed to apply vehicle config to {vid}")


# Configuration management
