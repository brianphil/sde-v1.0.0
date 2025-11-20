# Bug Fix: Decision Commit Route Response

## Issue

When committing a decision, the API was returning Route objects instead of route ID strings in the `routes_created` field, causing a Pydantic validation error:

```
Failed to commit decision: 1 validation error for DecisionCommitResponse
routes_created.0
  Input should be a valid string [type=string_type, input_value=Route(route_id='route_cfa...
```

## Root Cause

The `engine.commit_decision()` method returns Route domain objects in the `routes_created` list, but the API schema expects a list of strings (route IDs).

## Fix Applied

Updated `backend/api/routes/decisions.py` to:

1. **Extract route IDs** from Route objects before creating the response
2. **Store routes** in the route_store for later access
3. **Return string IDs** as expected by the schema

### Code Change:

```python
# Extract route IDs from Route objects if needed
routes_created = result["routes_created"]
route_ids = []

if routes_created:
    # Import route_store from routes module
    from backend.api.routes.routes import route_store

    if hasattr(routes_created[0], 'route_id'):
        # Result contains Route objects, extract IDs and store routes
        for route in routes_created:
            route_ids.append(route.route_id)
            route_store[route.route_id] = route
    else:
        # Already string IDs
        route_ids = routes_created

response = DecisionCommitResponse(
    success=not bool(result.get("errors")),
    action=result["action"],
    routes_created=route_ids,  # Now contains string IDs
    orders_assigned=result["orders_assigned"],
    errors=result.get("errors", []),
    message=f"Decision {decision_id} committed successfully"
    if not result.get("errors")
    else f"Decision committed with errors",
)
```

## Result

✅ Decision commits now work correctly
✅ Routes are properly stored for later access
✅ Route IDs are returned as strings
✅ Full workflow test can now complete successfully

## Testing

Restart the API server and run the workflow test:

```bash
# Restart API
python -m backend.api.main

# Run workflow test
python test_complete_workflow.py
```

Expected result: Routes created successfully with consolidation!
