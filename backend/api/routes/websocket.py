"""WebSocket endpoint for real-time updates from Powell Sequential Decision Engine."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Set
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: dict = {}  # connection_id -> set of event types

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[connection_id] = set()
        logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, websocket: WebSocket, connection_id: str):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")

    def subscribe(self, connection_id: str, event_types: Set[str]):
        """Subscribe a connection to specific event types."""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].update(event_types)
            logger.info(
                f"Connection {connection_id} subscribed to: {event_types}"
            )

    def unsubscribe(self, connection_id: str, event_types: Set[str]):
        """Unsubscribe a connection from specific event types."""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].difference_update(event_types)
            logger.info(
                f"Connection {connection_id} unsubscribed from: {event_types}"
            )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")

    async def broadcast(self, message: dict, event_type: str = None):
        """Broadcast a message to all subscribed connections."""
        disconnected = []

        for websocket in self.active_connections:
            # Find connection_id for this websocket
            connection_id = None
            for cid, ws in enumerate(self.active_connections):
                if ws == websocket:
                    connection_id = f"conn_{cid}"
                    break

            # Check if connection is subscribed to this event type
            if event_type and connection_id:
                if connection_id not in self.subscriptions:
                    continue
                if event_type not in self.subscriptions[connection_id]:
                    continue

            try:
                await websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected connections
        for websocket in disconnected:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates.

    Clients can subscribe to different event types:
    - decision_made: New decision created
    - decision_committed: Decision committed and executed
    - order_created: New order received
    - route_created: New route created
    - route_started: Route started
    - route_completed: Route completed
    - outcome_recorded: Operational outcome recorded
    - learning_updated: Model parameters updated

    Message format:
    {
        "type": "event_type",
        "data": {...},
        "timestamp": "2024-01-15T10:30:00"
    }

    Client can send subscription requests:
    {
        "action": "subscribe",
        "events": ["decision_made", "route_created"]
    }
    """
    connection_id = f"conn_{id(websocket)}"

    await manager.connect(websocket, connection_id)

    try:
        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "connected",
                "connection_id": connection_id,
                "message": "Connected to Powell Engine WebSocket",
                "timestamp": datetime.now().isoformat(),
                "available_events": [
                    "decision_made",
                    "decision_committed",
                    "order_created",
                    "route_created",
                    "route_started",
                    "route_completed",
                    "outcome_recorded",
                    "learning_updated",
                ],
            },
            websocket,
        )

        # Default subscription: subscribe to all events
        manager.subscribe(
            connection_id,
            {
                "decision_made",
                "decision_committed",
                "order_created",
                "route_created",
                "route_started",
                "route_completed",
                "outcome_recorded",
                "learning_updated",
            },
        )

        # Listen for messages from client
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()

                action = data.get("action")

                if action == "subscribe":
                    # Subscribe to events
                    events = set(data.get("events", []))
                    manager.subscribe(connection_id, events)

                    await manager.send_personal_message(
                        {
                            "type": "subscription_confirmed",
                            "subscribed_events": list(events),
                            "timestamp": datetime.now().isoformat(),
                        },
                        websocket,
                    )

                elif action == "unsubscribe":
                    # Unsubscribe from events
                    events = set(data.get("events", []))
                    manager.unsubscribe(connection_id, events)

                    await manager.send_personal_message(
                        {
                            "type": "unsubscription_confirmed",
                            "unsubscribed_events": list(events),
                            "timestamp": datetime.now().isoformat(),
                        },
                        websocket,
                    )

                elif action == "ping":
                    # Respond to ping with pong
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat(),
                        },
                        websocket,
                    )

                elif action == "get_subscriptions":
                    # Return current subscriptions
                    subscribed_events = list(
                        manager.subscriptions.get(connection_id, set())
                    )
                    await manager.send_personal_message(
                        {
                            "type": "subscriptions",
                            "events": subscribed_events,
                            "timestamp": datetime.now().isoformat(),
                        },
                        websocket,
                    )

                else:
                    # Unknown action
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "message": f"Unknown action: {action}",
                            "timestamp": datetime.now().isoformat(),
                        },
                        websocket,
                    )

            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat(),
                    },
                    websocket,
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, connection_id)


# Helper functions for broadcasting events
async def broadcast_decision_made(decision_id: str, decision_data: dict):
    """Broadcast decision_made event to all subscribed clients."""
    await manager.broadcast(
        {
            "type": "decision_made",
            "data": {
                "decision_id": decision_id,
                "policy": decision_data.get("policy"),
                "action": decision_data.get("action"),
                "confidence": decision_data.get("confidence"),
                "routes_count": len(decision_data.get("routes", [])),
            },
            "timestamp": datetime.now().isoformat(),
        },
        event_type="decision_made",
    )


async def broadcast_order_created(order_id: str, customer_id: str, priority: int):
    """Broadcast order_created event to all subscribed clients."""
    await manager.broadcast(
        {
            "type": "order_created",
            "data": {
                "order_id": order_id,
                "customer_id": customer_id,
                "priority": priority,
            },
            "timestamp": datetime.now().isoformat(),
        },
        event_type="order_created",
    )


async def broadcast_route_created(route_id: str, vehicle_id: str, order_count: int):
    """Broadcast route_created event to all subscribed clients."""
    await manager.broadcast(
        {
            "type": "route_created",
            "data": {
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "order_count": order_count,
            },
            "timestamp": datetime.now().isoformat(),
        },
        event_type="route_created",
    )


async def broadcast_route_status_change(route_id: str, status: str):
    """Broadcast route status change to all subscribed clients."""
    event_type = f"route_{status}"
    await manager.broadcast(
        {
            "type": event_type,
            "data": {
                "route_id": route_id,
                "status": status,
            },
            "timestamp": datetime.now().isoformat(),
        },
        event_type=event_type,
    )


async def broadcast_outcome_recorded(outcome_id: str, route_id: str):
    """Broadcast outcome_recorded event to all subscribed clients."""
    await manager.broadcast(
        {
            "type": "outcome_recorded",
            "data": {
                "outcome_id": outcome_id,
                "route_id": route_id,
            },
            "timestamp": datetime.now().isoformat(),
        },
        event_type="outcome_recorded",
    )


async def broadcast_learning_updated(policy: str, metrics: dict):
    """Broadcast learning_updated event to all subscribed clients."""
    await manager.broadcast(
        {
            "type": "learning_updated",
            "data": {
                "policy": policy,
                "metrics": metrics,
            },
            "timestamp": datetime.now().isoformat(),
        },
        event_type="learning_updated",
    )
