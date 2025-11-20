# Senga SDE Framework - AI Agent Instructions

## Executive Context

**Senga Sequential Decision Engine (SDE)** is an event-driven, learning-based route optimization system for mesh logistics (Nairobi → Nakuru/Eldoret/Kitale). The system implements Warren Powell's Sequential Decision Analytics framework using all four policy classes (PFA, CFA, VFA, DLA) and their hybrids to make continuous decisions and learn from operational feedback.

**Core Problem Solved:** Senga operates a warehouseless mesh network requiring intelligent consolidation of orders into optimal routes. The system must:

- Dynamically pool and route orders as they arrive
- Optimize multi-destination pickup and delivery sequences
- Learn from operational outcomes to improve predictions
- Balance immediate costs against long-term network efficiency

**Success Criteria:**

- ✅ Functional: Engine makes automated routing decisions for incoming orders
- ✅ Learning: System improves cost/time predictions from operational feedback
- ✅ Operational: Generates executable route plans for operations team
- ✅ Scalable: Handles order arrival rate and fleet size without degradation

## Architecture Overview

**Senga** is a Spatial Decision Engine that combines real-time route optimization with reinforcement learning. It processes delivery orders, optimizes routes via Powell algorithms, and learns from feedback to improve future decisions.

### Core Components

#### Backend Architecture (FastAPI)

- **API Layer** (`backend/api/`): RESTful routes for decisions, orders; WebSocket for real-time updates
- **Core Models** (`backend/core/models/`): Domain entities (Order, Vehicle, Route, Customer, State)
- **Powell Engine** (`backend/core/powell/`): Algorithm suite (PFA, VFA, CFA, DLA, hybrids) for optimization
- **Learning System** (`backend/core/learning/`): TD-learning feedback loop, parameter updates, pattern mining

# Copilot / AI Agent Instructions — Senga SDE

Brief: This file gives concise, actionable guidance so an AI coding agent can be productive quickly. Focus on the files and workflows below when changing decision logic, learning, or config.

Core places to read first:

- `backend/core/powell/engine.py` — policy selection and how policies are wired
- `backend/core/powell/{pfa.py,cfa.py,vfa.py,dla.py,hybrids.py}` — actual policy implementations
- `backend/services/event_orchestrator.py` — decision → commit → learning flow, where pre/post states are captured
- `backend/services/learning_coordinator.py` — feedback processing, experience buffering, when training is triggered
- `backend/services/state_manager.py` — immutable state updates; use `set_current_state` and `apply_event`
- `backend/config/model_config.yaml` & `backend/utils/config.py` — runtime hyperparams and per-vehicle defaults
- `demo.py` — runnable integration example used in CI/manual validation

Quick run / test commands (Windows PowerShell):

```
pip install -r requirements.txt
python .\demo.py            # quick end-to-end smoke test (uses demo data)
uvicorn backend.api.main:app --reload   # run API
pytest -q                      # run tests
```

Conventions & patterns you must follow:

- State is treated immutably. Do not mutate `SystemState` in-place; use StateManager helpers (`apply_event`, `set_current_state`).
- Config-driven: prefer values from `backend/config/model_config.yaml` via `backend/utils/config.py` rather than hardcoding constants.
- Policy interfaces: policies return `PolicyDecision`/`HybridDecision`. `engine.commit_decision` converts those to route objects for the StateManager to persist.
- Learning lifecycle: capture pre-decision features in `EventOrchestrator` (per-route DecisionContext), store pending experiences in VFA, then `LearningCoordinator.process_outcome` completes experiences and calls `vfa.train_from_buffer` when thresholds met.

Code-change checklist (minimal safe steps for algorithm edits):

1. Update policy logic in the appropriate `backend/core/powell/*.py` file.
2. Update config defaults in `backend/config/model_config.yaml` if introducing tunables.
3. If behavior affects commit/route shape, adjust `EventOrchestrator` so pending experiences capture the necessary per-route DecisionContext.
4. Add or update small unit tests under `backend/core/...` and run `pytest`.
5. Run `python .\demo.py` to validate end-to-end flow and learning telemetry.

Important gotchas for agents (do these checks automatically):

- Do not introduce hard-coded, region-specific rules (e.g., Eastleigh/Majid text). Use `special_handling_tags` or customer attributes.
- When adding PyTorch training, make sure to: (a) guard imports with availability checks, (b) handle missing next-state by treating it as terminal for bootstrapping, and (c) log training loss to `LearningCoordinator.vfa_telemetry`.
- When changing CFA cost params, keep parameter bounds and safe clip logic to avoid divergence.
- Persist model artifacts via `PowellEngine.get_learned_state()` and restore with `restore_learned_state()` during engine startup.

Useful examples to copy/paste:

- Extracting per-route DecisionContext in `EventOrchestrator`:
  - see `backend/services/event_orchestrator.py` lines around pending experience capture
- Adding experience and training in `LearningCoordinator.process_outcome`:
  - call `engine.vfa.complete_pending_experience(route_id, reward)` or `engine.vfa.add_experience(...)`, then `engine.vfa.train_from_buffer(...)`

If unsure, prioritize: (1) state immutability, (2) config-first parameters, (3) safe training guards.

Ask for clarifications: tell me which policy you will change, the expected input features, and how you will validate (unit test + demo run). I will patch the files and run the demo/tests on your behalf.
