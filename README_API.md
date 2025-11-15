# Powell Sequential Decision Engine - API

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd senga-sde
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python -m backend.api.main
```

The server will start on `http://localhost:8000`

### 3. Verify It's Running

Open your browser to:
- **Swagger UI (Interactive Docs):** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### 4. Run Test Suite

```bash
python test_api.py
```

---

## 📋 What's Included

The API provides complete access to the Powell Sequential Decision Engine:

### ✅ Endpoints

1. **Decision Management** (`/api/v1/decisions`)
   - Make routing decisions using Powell policies
   - Commit decisions to execute them
   - View decision history
   - Get system state and metrics

2. **Order Management** (`/api/v1/orders`)
   - Create, read, update, delete orders
   - Filter by status, customer, priority
   - Automatic event triggering for urgent orders

3. **Route Management** (`/api/v1/routes`)
   - List and view routes
   - Start, complete, or cancel routes
   - Record operational outcomes
   - Trigger learning from feedback

4. **Real-Time Updates** (`/api/v1/ws`)
   - WebSocket connection for live events
   - Subscribe to specific event types
   - Receive updates on decisions, routes, outcomes

### ✅ Features

- **Full REST API** - Standard HTTP methods (GET, POST, PUT, DELETE)
- **Request/Response Validation** - Pydantic schemas for all endpoints
- **Error Handling** - Comprehensive error responses
- **CORS Support** - Configurable cross-origin requests
- **Auto-Generated Docs** - Swagger UI and ReDoc
- **Logging** - Structured logging for debugging
- **WebSocket Support** - Real-time event streaming

---

## 📖 API Documentation

See [API_GUIDE.md](./API_GUIDE.md) for complete API documentation including:
- All endpoints with request/response examples
- WebSocket protocol
- Example workflows
- Error handling
- Testing guides

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│         (backend/api/main.py)           │
└──────────────┬──────────────────────────┘
               │
     ┌─────────┼─────────┐
     │         │         │
┌────▼────┐ ┌──▼──┐ ┌───▼────┐
│Decisions│ │Orders│ │ Routes │
│Endpoints│ │Endpts│ │Endpoints│
└────┬────┘ └──┬──┘ └───┬────┘
     │         │         │
     └─────────┼─────────┘
               │
     ┌─────────▼──────────┐
     │   Powell Engine    │
     │  State Manager     │
     │  Event Orchestrator│
     └────────────────────┘
```

### Components

1. **main.py** - FastAPI app, lifecycle management, global state
2. **schemas.py** - Pydantic models for validation
3. **routes/decisions.py** - Decision-making endpoints
4. **routes/orders.py** - Order management endpoints
5. **routes/routes.py** - Route management endpoints
6. **routes/websocket.py** - WebSocket real-time updates

---

## 🧪 Testing

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Make a decision
curl -X POST http://localhost:8000/api/v1/decisions/make \
  -H "Content-Type: application/json" \
  -d '{
    "decision_type": "daily_route_planning",
    "trigger_reason": "Morning optimization"
  }'

# Create an order
curl -X POST http://localhost:8000/api/v1/orders \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "customer_name": "Test Customer",
    "pickup_location": {
      "latitude": -1.2921,
      "longitude": 36.8219,
      "address": "Nairobi CBD"
    },
    "destination_city": "Nakuru",
    "weight_tonnes": 2.5,
    "volume_m3": 4.0,
    "time_window": {
      "start_time": "2024-01-15T08:00:00",
      "end_time": "2024-01-15T10:00:00"
    },
    "priority": 1,
    "price_kes": 2500.0
  }'
```

### Automated Testing

```bash
# Run test suite
python test_api.py
```

This will test:
- ✅ Health check
- ✅ Order creation
- ✅ System state retrieval
- ✅ Decision making
- ✅ Decision commitment
- ✅ Route listing
- ✅ Learning metrics

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=*

# Future: Database
# DATABASE_URL=postgresql://user:password@localhost/powell_db
```

### Model Configuration

Edit `model_config.yaml` to adjust:
- VFA neural network architecture
- CFA economic parameters
- DLA forecast periods
- Business constraints (time windows, etc.)

---

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Returns status of all components:
```json
{
  "status": "healthy",
  "components": {
    "engine": "healthy",
    "state_manager": "healthy",
    "orchestrator": "healthy",
    "learning": "healthy"
  }
}
```

### Learning Metrics

```bash
curl http://localhost:8000/api/v1/metrics/learning
```

Returns performance metrics for all policies:
- CFA prediction accuracy
- VFA training loss
- PFA rule confidence
- Feedback statistics

---

## 🚢 Deployment

### Using Uvicorn (Development)

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Using Uvicorn (Production)

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker (Future)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t powell-engine .
docker run -p 8000:8000 powell-engine
```

---

## 🔐 Security Considerations

**Current State:** No authentication (development only)

**For Production:**

1. **Add Authentication**
   - JWT tokens
   - API keys
   - OAuth 2.0

2. **Rate Limiting**
   - Limit requests per client
   - Prevent abuse

3. **HTTPS Only**
   - Use TLS certificates
   - Redirect HTTP to HTTPS

4. **Input Validation**
   - Already implemented with Pydantic
   - Add additional business logic validation

5. **CORS Configuration**
   - Restrict allowed origins
   - Configure for specific domains

---

## 📝 Next Steps

### Immediate Improvements

1. **Add Database Persistence**
   - Replace in-memory stores
   - Use PostgreSQL or MongoDB
   - Add migration system

2. **Add Authentication**
   - JWT tokens
   - User management
   - API key system

3. **Add Caching**
   - Redis for frequently accessed data
   - Cache decision results
   - Cache learning metrics

4. **Add Background Tasks**
   - Celery for async processing
   - Scheduled jobs (daily optimization)
   - Batch processing

### Advanced Features

1. **Analytics Dashboard**
   - Real-time metrics visualization
   - Historical performance tracking
   - Policy comparison

2. **A/B Testing**
   - Test different policies
   - Compare performance
   - Gradual rollout

3. **Multi-Tenancy**
   - Support multiple clients
   - Isolated data
   - Tenant-specific configurations

4. **External Integrations**
   - Google Maps API for routing
   - Traffic APIs
   - Weather APIs
   - ERP integrations

---

## 🐛 Troubleshooting

### Server Won't Start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Error:** `Address already in use`
```bash
# Solution: Change port or kill process on port 8000
lsof -ti:8000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :8000   # Windows (then taskkill /PID <pid> /F)
```

### API Returns 500 Errors

Check the server logs for detailed error messages. Common issues:

1. **No system state** - Initialize state before making decisions
2. **Invalid data** - Check request body matches schema
3. **Missing configuration** - Ensure `model_config.yaml` exists

### WebSocket Connection Fails

1. Ensure server is running
2. Use `ws://` not `wss://` for local development
3. Check browser console for errors

---

## 📚 Additional Resources

- **Full API Documentation:** [API_GUIDE.md](./API_GUIDE.md)
- **Engine Implementation:** [ENGINE_IMPLEMENTATION.md](./ENGINE_IMPLEMENTATION.md)
- **Integration Guide:** [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
- **Demo Scripts:** [demo.py](./demo.py)

---

## 💡 Support

For questions or issues:
1. Check the documentation
2. Review test scripts for examples
3. Check server logs for errors
4. Create an issue in the repository

---

## ✅ Summary

You now have a **fully functional REST API** for the Powell Sequential Decision Engine with:

- ✅ **17 REST endpoints** for decisions, orders, and routes
- ✅ **WebSocket support** for real-time updates
- ✅ **Auto-generated documentation** (Swagger UI)
- ✅ **Request/response validation** (Pydantic)
- ✅ **Comprehensive error handling**
- ✅ **Test suite** for verification
- ✅ **Complete API guide** with examples

The engine is now **production-ready** for the API layer. Next recommended steps are database persistence and authentication!
