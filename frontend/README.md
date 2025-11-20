# Powell SDE Frontend

React + TypeScript frontend for the Powell Sequential Decision Engine.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **TailwindCSS** - Styling
- **React Router** - Navigation
- **React Query** - API state management
- **Lucide React** - Icons
- **Recharts** - Data visualization

## Features

### Pages

1. **Dashboard** - Real-time overview with key metrics
   - System status (orders, vehicles, routes)
   - Learning component telemetry (VFA, CFA, PFA)
   - Quick metrics visualization

2. **Orders** - Order management
   - List all orders
   - Create new orders
   - View order details

3. **Fleet** - Vehicle management
   - View all vehicles
   - Vehicle status monitoring
   - Capacity tracking

4. **Routes** - Route visualization
   - Active routes
   - Completed routes
   - Route details and metrics

5. **Decisions** - Decision making interface
   - Make new decisions with Powell Engine
   - View decision history
   - Decision confidence and expected value

6. **Learning Metrics** - Telemetry dashboard
   - VFA: Training progress, experience replay, learning rate
   - CFA: Cost parameters, convergence status, accuracy
   - PFA: Pattern rules, confidence, exploration rate

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The frontend runs on `http://localhost:3000` by default.

The backend API is proxied from `http://localhost:8000`:
- `/api` routes to backend API endpoints
- `/ws` routes to WebSocket connections

## API Integration

The frontend integrates with the FastAPI backend through:

- **REST API**: All CRUD operations for orders, vehicles, routes, etc.
- **WebSocket** (coming soon): Real-time updates for active routes and decisions
- **React Query**: Automatic caching and refetching

## Project Structure

```
src/
├── components/     # Reusable UI components
│   └── Layout.tsx # Main layout with sidebar navigation
├── pages/         # Page components
│   ├── Dashboard.tsx
│   ├── Orders.tsx
│   ├── Fleet.tsx
│   ├── RoutesPage.tsx
│   ├── Decisions.tsx
│   └── LearningMetrics.tsx
├── services/      # API service layer
│   └── api.ts     # Axios-based API client
├── types/         # TypeScript type definitions
│   └── index.ts   # Domain models, API responses
├── App.tsx        # Main app with routing
└── main.tsx       # Entry point
```

## Environment Variables

Create a `.env` file if needed:

```
VITE_API_BASE_URL=http://localhost:8000
```

## Production Deployment

1. Build the production bundle:
   ```bash
   npm run build
   ```

2. The `dist/` directory contains the optimized production build

3. Serve with any static file server or integrate with FastAPI

## Integration with Backend

The frontend expects the backend to be running at `http://localhost:8000` with the following endpoints:

- `GET /health` - Health check
- `GET /api/v1/orders` - List orders
- `POST /api/v1/orders` - Create order
- `GET /api/v1/vehicles` - List vehicles
- `GET /api/v1/routes` - List routes
- `POST /api/v1/decisions/make` - Make decision
- `GET /api/v1/metrics` - Get learning metrics

## License

Part of the Powell SDE project.
