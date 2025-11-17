"""FastAPI application for Powell Sequential Decision Engine.

This API provides endpoints for:
- Making routing decisions
- Managing orders, vehicles, and routes
- Recording operational outcomes
- Real-time updates via WebSocket
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import traceback

from backend.api.schemas import ErrorResponse, HealthResponse
from backend.api.routes import decisions, orders, routes, websocket, vehicles, customers
from backend.api import auth  # Authentication routes
from backend.db.database import init_database, close_database, check_database_health
from backend.core.powell.engine import PowellEngine
from backend.services.state_manager import StateManager
from backend.services.event_orchestrator import EventOrchestrator
from backend.services.learning_coordinator import LearningCoordinator
from backend.core.models.domain import Vehicle, VehicleStatus, Location, Customer
from backend.core.models.state import SystemState, EnvironmentState, LearningState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global application state
class AppState:
    """Centralized application state."""

    def __init__(self):
        self.engine: PowellEngine = None
        self.state_manager: StateManager = None
        self.orchestrator: EventOrchestrator = None
        self.learning_coordinator: LearningCoordinator = None

    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Powell Engine components...")

        try:
            # Initialize core components
            self.engine = PowellEngine()
            self.state_manager = StateManager()
            self.orchestrator = EventOrchestrator(self.engine, self.state_manager)
            self.learning_coordinator = LearningCoordinator(engine=self.engine)

            # Register learning handler with orchestrator
            self.orchestrator.register_learning_handler(
                self.learning_coordinator.process_outcome
            )

            # Initialize system state with sample fleet and customers
            self._initialize_system_state()

            logger.info("✅ All components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def _initialize_system_state(self):
        """Initialize system state with sample vehicles and customers."""
        logger.info("Initializing system state with sample data...")

        now = datetime.now()
        morning_time = now.replace(hour=8, minute=30)

        # Create sample fleet
        fleet = {
            "VEH_001": Vehicle(
                vehicle_id="VEH_001",
                vehicle_type="5T",
                capacity_weight_tonnes=5.0,
                capacity_volume_m3=8.0,
                current_location=Location(
                    latitude=-1.2921,
                    longitude=36.8219,
                    address="Nairobi Depot",
                    zone="Depot"
                ),
                available_at=morning_time,
                status=VehicleStatus.AVAILABLE,
                driver_id="DRIVER_001",
                fuel_cost_per_km=9.0,
                driver_cost_per_hour=500.0,
            ),
            "VEH_002": Vehicle(
                vehicle_id="VEH_002",
                vehicle_type="5T",
                capacity_weight_tonnes=5.0,
                capacity_volume_m3=8.0,
                current_location=Location(
                    latitude=-1.2921,
                    longitude=36.8219,
                    address="Nairobi Depot",
                    zone="Depot"
                ),
                available_at=morning_time,
                status=VehicleStatus.AVAILABLE,
                driver_id="DRIVER_002",
                fuel_cost_per_km=9.0,
                driver_cost_per_hour=500.0,
            ),
            "VEH_003": Vehicle(
                vehicle_id="VEH_003",
                vehicle_type="10T",
                capacity_weight_tonnes=10.0,
                capacity_volume_m3=15.0,
                current_location=Location(
                    latitude=-1.2921,
                    longitude=36.8219,
                    address="Nairobi Depot",
                    zone="Depot"
                ),
                available_at=morning_time,
                status=VehicleStatus.AVAILABLE,
                driver_id="DRIVER_003",
                fuel_cost_per_km=12.0,
                driver_cost_per_hour=600.0,
            ),
        }

        # Create sample customers
        customers = {
            "CUST_MAJID": Customer(
                customer_id="CUST_MAJID",
                name="Majid Retailers",
                locations=[
                    Location(
                        latitude=-1.2921,
                        longitude=36.8219,
                        address="Eastleigh Store",
                        zone="Eastleigh"
                    )
                ],
                delivery_blocked_times=[
                    {"day": "Wednesday", "time_start": "14:00", "time_end": "15:30"}
                ],
                priority_level=1,
                fresh_food_customer=True,
            ),
            "CUST_ABC": Customer(
                customer_id="CUST_ABC",
                name="ABC Corporation",
                locations=[
                    Location(
                        latitude=-1.2800,
                        longitude=36.8300,
                        address="Nairobi CBD Office",
                        zone="CBD"
                    )
                ],
                priority_level=0,
            ),
        }

        # Create environment
        env = EnvironmentState(
            current_time=morning_time,
            traffic_conditions={
                "CBD": 0.5,
                "Eastleigh": 0.3,
                "Nakuru": 0.2,
                "Eldoret": 0.15,
                "Kitale": 0.1,
            },
            weather="clear",
        )

        # Create and set system state
        state = SystemState(
            pending_orders={},
            fleet=fleet,
            customers=customers,
            environment=env,
            active_routes={},
            completed_routes={},
            learning=LearningState(),
        )

        self.state_manager.set_current_state(state)

        logger.info(f"✅ System state initialized:")
        logger.info(f"   - {len(fleet)} vehicles available")
        logger.info(f"   - {len(customers)} customers configured")
        logger.info(f"   - Environment: {env.weather}, {len(env.traffic_conditions)} traffic zones")

    def shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down Powell Engine components...")
        # Add any cleanup logic here if needed
        logger.info("✅ Shutdown complete")


# Create global app state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("🚀 Starting Powell Sequential Decision Engine API")

    # Initialize database
    logger.info("Initializing database...")
    try:
        await init_database()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

    # Initialize app components
    app_state.initialize()

    yield

    # Shutdown
    app_state.shutdown()

    # Close database
    logger.info("Closing database connections...")
    await close_database()
    logger.info("👋 Powell Engine API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Powell Sequential Decision Engine API",
    description="API for making sequential routing decisions using the Powell framework",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
        ).model_dump(),
    )


# Health check endpoint
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        components={
            "engine": "initialized" if app_state.engine else "not_initialized",
            "state_manager": "initialized" if app_state.state_manager else "not_initialized",
            "orchestrator": "initialized" if app_state.orchestrator else "not_initialized",
            "learning": "initialized" if app_state.learning_coordinator else "not_initialized",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check."""
    try:
        # Check all components
        components_status = {
            "engine": "healthy" if app_state.engine else "unhealthy",
            "state_manager": "healthy" if app_state.state_manager else "unhealthy",
            "orchestrator": "healthy" if app_state.orchestrator else "unhealthy",
            "learning": "healthy" if app_state.learning_coordinator else "unhealthy",
        }

        # Check database health
        db_health = await check_database_health()
        components_status["database"] = "healthy" if db_health else "unhealthy"

        overall_status = "healthy" if all(
            s == "healthy" for s in components_status.values()
        ) else "degraded"

        return HealthResponse(
            status=overall_status,
            components=components_status,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Include routers
app.include_router(auth.router)  # Auth router has its own prefix
app.include_router(decisions.router, prefix="/api/v1", tags=["Decisions"])
app.include_router(orders.router, prefix="/api/v1", tags=["Orders"])
app.include_router(customers.router, prefix="/api/v1", tags=["Customers"])
app.include_router(routes.router, prefix="/api/v1", tags=["Routes"])
app.include_router(vehicles.router, prefix="/api/v1", tags=["Vehicles"])
app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
