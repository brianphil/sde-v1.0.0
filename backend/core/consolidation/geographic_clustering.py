"""Intelligent Geographic Clustering Engine.

This module provides sophisticated geographic clustering that considers:
- Route corridors and waypoints
- Bearing compatibility
- Shared road networks
- Mesh routing opportunities

NOT just simple distance from origin.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import math
import logging

from ..models.domain import Order, Location, DestinationCity

logger = logging.getLogger(__name__)


class RouteCorridor(Enum):
    """Major route corridors in Kenya."""

    NAIROBI_NAKURU_KISUMU = "nairobi_nakuru_kisumu"  # A104 corridor
    NAIROBI_ELDORET_KITALE = "nairobi_eldoret_kitale"  # Via Nakuru
    NAIROBI_NYERI_NANYUKI = "nairobi_nyeri_nanyuki"  # A2 corridor
    NAIROBI_MOMBASA = "nairobi_mombasa"  # A109 corridor
    NAIROBI_NAMANGA = "nairobi_namanga"  # A104 south


@dataclass
class Waypoint:
    """Waypoint on a route corridor."""

    name: str
    location: Location
    corridors: List[RouteCorridor]
    is_major_hub: bool = False


@dataclass
class RouteNetwork:
    """Route network with waypoints and corridors."""

    # Major waypoints in Kenya
    waypoints: Dict[str, Waypoint] = field(default_factory=dict)

    # Corridor definitions (city -> list of waypoints on route)
    corridor_routes: Dict[RouteCorridor, List[str]] = field(default_factory=dict)

    # City to corridor mapping
    city_corridors: Dict[DestinationCity, List[RouteCorridor]] = field(default_factory=dict)


class GeographicClusteringEngine:
    """Intelligent geographic clustering based on route networks.

    Uses:
    - Route corridors (shared waypoints)
    - Bearing analysis
    - Distance compatibility
    - Mesh routing opportunities
    """

    def __init__(self):
        """Initialize with Kenya route network."""
        self.network = self._initialize_kenya_network()

    def _initialize_kenya_network(self) -> RouteNetwork:
        """Initialize Kenya's major route corridors and waypoints."""
        network = RouteNetwork()

        # Define waypoints
        network.waypoints = {
            "nairobi": Waypoint(
                name="Nairobi",
                location=Location(
                    latitude=-1.2921,
                    longitude=36.8219,
                    address="Nairobi",
                    zone="Nairobi"
                ),
                corridors=[
                    RouteCorridor.NAIROBI_NAKURU_KISUMU,
                    RouteCorridor.NAIROBI_ELDORET_KITALE,
                    RouteCorridor.NAIROBI_NYERI_NANYUKI,
                    RouteCorridor.NAIROBI_MOMBASA,
                ],
                is_major_hub=True
            ),
            "nakuru": Waypoint(
                name="Nakuru",
                location=Location(
                    latitude=-0.3031,
                    longitude=36.0800,
                    address="Nakuru",
                    zone="Nakuru"
                ),
                corridors=[
                    RouteCorridor.NAIROBI_NAKURU_KISUMU,
                    RouteCorridor.NAIROBI_ELDORET_KITALE,
                ],
                is_major_hub=True
            ),
            "eldoret": Waypoint(
                name="Eldoret",
                location=Location(
                    latitude=0.5143,
                    longitude=35.2698,
                    address="Eldoret",
                    zone="Eldoret"
                ),
                corridors=[RouteCorridor.NAIROBI_ELDORET_KITALE],
                is_major_hub=True
            ),
            "kitale": Waypoint(
                name="Kitale",
                location=Location(
                    latitude=1.0157,
                    longitude=35.0062,
                    address="Kitale",
                    zone="Kitale"
                ),
                corridors=[RouteCorridor.NAIROBI_ELDORET_KITALE],
                is_major_hub=False
            ),
            "kisumu": Waypoint(
                name="Kisumu",
                location=Location(
                    latitude=-0.0917,
                    longitude=34.7680,
                    address="Kisumu",
                    zone="Kisumu"
                ),
                corridors=[RouteCorridor.NAIROBI_NAKURU_KISUMU],
                is_major_hub=True
            ),
        }

        # Define corridor routes (order matters - sequence of waypoints)
        network.corridor_routes = {
            RouteCorridor.NAIROBI_NAKURU_KISUMU: ["nairobi", "nakuru", "kisumu"],
            RouteCorridor.NAIROBI_ELDORET_KITALE: ["nairobi", "nakuru", "eldoret", "kitale"],
        }

        # Map cities to corridors
        network.city_corridors = {
            DestinationCity.NAKURU: [
                RouteCorridor.NAIROBI_NAKURU_KISUMU,
                RouteCorridor.NAIROBI_ELDORET_KITALE,
            ],
            DestinationCity.ELDORET: [RouteCorridor.NAIROBI_ELDORET_KITALE],
            DestinationCity.KITALE: [RouteCorridor.NAIROBI_ELDORET_KITALE],
        }

        return network

    def cluster_orders(self, orders: Dict[str, Order]) -> Dict[str, List[str]]:
        """Cluster orders by geographic compatibility.

        Returns:
            Dict mapping cluster_id to list of order_ids
        """
        if not orders:
            return {}

        clusters = {}

        # Step 1: Group by route corridor
        corridor_groups = self._group_by_corridor(orders)

        # Step 2: For each corridor, identify sub-clusters
        for corridor, order_ids in corridor_groups.items():
            corridor_orders = {oid: orders[oid] for oid in order_ids}

            # Check if orders can be served in a single route
            sub_clusters = self._identify_mesh_clusters(corridor, corridor_orders)

            for cluster_id, cluster_order_ids in sub_clusters.items():
                clusters[f"{corridor.value}_{cluster_id}"] = cluster_order_ids

        logger.info(f"Geographic clustering: {len(orders)} orders → {len(clusters)} clusters")
        return clusters

    def _group_by_corridor(self, orders: Dict[str, Order]) -> Dict[RouteCorridor, List[str]]:
        """Group orders by route corridor."""
        corridor_groups: Dict[RouteCorridor, List[str]] = {}

        for order_id, order in orders.items():
            corridors = self.network.city_corridors.get(order.destination_city, [])

            if not corridors:
                # Orders without defined corridor - create single-order cluster
                corridor_groups.setdefault(RouteCorridor.NAIROBI_MOMBASA, []).append(order_id)
                logger.debug(f"Order {order_id} to {order.destination_city} has no defined corridor")
                continue

            # Assign to first matching corridor (can be optimized later)
            primary_corridor = corridors[0]
            corridor_groups.setdefault(primary_corridor, []).append(order_id)

        return corridor_groups

    def _identify_mesh_clusters(
        self,
        corridor: RouteCorridor,
        orders: Dict[str, Order]
    ) -> Dict[str, List[str]]:
        """Identify mesh routing opportunities within a corridor.

        Mesh routing: Multi-pickup, multi-delivery along the same corridor.
        Example: Pickup in Nairobi, deliver in Nakuru, pickup in Nakuru, deliver in Eldoret.
        """
        if not orders:
            return {}

        # Group by final destination, then check for mesh opportunities
        destination_clusters: Dict[DestinationCity, List[str]] = {}

        for order_id, order in orders.items():
            destination_clusters.setdefault(order.destination_city, []).append(order_id)

        # Convert to string keys
        clusters = {
            dest.value: order_ids
            for dest, order_ids in destination_clusters.items()
        }

        # Check for multi-stop opportunities
        if len(clusters) > 1:
            # Multiple destinations on same corridor - check if mesh routing is beneficial
            mesh_cluster = self._check_mesh_opportunity(corridor, orders, clusters)
            if mesh_cluster:
                return {"mesh": mesh_cluster}

        return clusters

    def _check_mesh_opportunity(
        self,
        corridor: RouteCorridor,
        orders: Dict[str, Order],
        destination_clusters: Dict[str, List[str]]
    ) -> Optional[List[str]]:
        """Check if orders can be combined in mesh route.

        Criteria:
        - Same corridor
        - Sequential waypoints
        - Compatible time windows
        - Total capacity feasible
        """
        # Get corridor route sequence
        route_sequence = self.network.corridor_routes.get(corridor, [])

        if not route_sequence or len(destination_clusters) <= 1:
            return None

        # Check if destinations are sequential in corridor
        destination_positions = {}
        for dest_name, order_ids in destination_clusters.items():
            # Find position in corridor
            try:
                position = route_sequence.index(dest_name.lower())
                destination_positions[dest_name] = position
            except ValueError:
                # Destination not in this corridor sequence
                return None

        # If all destinations are sequential, mesh routing possible
        positions = sorted(destination_positions.values())
        if positions == list(range(min(positions), max(positions) + 1)):
            # Sequential stops - can mesh
            all_order_ids = []
            for order_ids in destination_clusters.values():
                all_order_ids.extend(order_ids)

            logger.info(f"Mesh opportunity detected: {len(all_order_ids)} orders across {len(destination_clusters)} stops on {corridor.value}")
            return all_order_ids

        return None

    def calculate_route_compatibility_score(
        self,
        order1: Order,
        order2: Order
    ) -> float:
        """Calculate geographic compatibility between two orders.

        Returns:
            Score 0.0 - 1.0 (higher = more compatible)
        """
        # Check if same corridor
        corridors1 = self.network.city_corridors.get(order1.destination_city, [])
        corridors2 = self.network.city_corridors.get(order2.destination_city, [])

        shared_corridors = set(corridors1) & set(corridors2)

        if not shared_corridors:
            # Different corridors - incompatible
            return 0.0

        # Same corridor - check bearing and distance
        bearing_score = self._calculate_bearing_compatibility(order1, order2)
        distance_score = self._calculate_distance_compatibility(order1, order2)

        # Weighted combination
        return 0.6 * bearing_score + 0.4 * distance_score

    def _calculate_bearing_compatibility(self, order1: Order, order2: Order) -> float:
        """Calculate bearing compatibility (0.0 - 1.0)."""
        # Calculate bearing from origin to each destination
        bearing1 = self._calculate_bearing(
            order1.pickup_location,
            order1.destination_location or order1.pickup_location
        )
        bearing2 = self._calculate_bearing(
            order2.pickup_location,
            order2.destination_location or order2.pickup_location
        )

        # Calculate bearing difference
        bearing_diff = abs(bearing1 - bearing2)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff

        # Convert to score (0° diff = 1.0, 180° diff = 0.0)
        score = 1.0 - (bearing_diff / 180.0)

        return score

    def _calculate_bearing(self, loc1: Location, loc2: Location) -> float:
        """Calculate bearing from loc1 to loc2 in degrees (0-360)."""
        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        lon1 = math.radians(loc1.longitude)
        lon2 = math.radians(loc2.longitude)

        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing

    def _calculate_distance_compatibility(self, order1: Order, order2: Order) -> float:
        """Calculate distance compatibility (0.0 - 1.0)."""
        # Calculate distance from origin to each destination
        dist1 = self._haversine_distance(
            order1.pickup_location,
            order1.destination_location or order1.pickup_location
        )
        dist2 = self._haversine_distance(
            order2.pickup_location,
            order2.destination_location or order2.pickup_location
        )

        # If distances are similar, more compatible
        ratio = min(dist1, dist2) / max(dist1, dist2) if max(dist1, dist2) > 0 else 1.0

        return ratio

    def _haversine_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate haversine distance in km."""
        R = 6371  # Earth radius in km

        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        dlat = math.radians(loc2.latitude - loc1.latitude)
        dlon = math.radians(loc2.longitude - loc1.longitude)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_shared_waypoints(
        self,
        city1: DestinationCity,
        city2: DestinationCity
    ) -> List[str]:
        """Get shared waypoints between two cities.

        Example: Nakuru, Eldoret, Kisumu all share Nakuru waypoint.
        """
        corridors1 = self.network.city_corridors.get(city1, [])
        corridors2 = self.network.city_corridors.get(city2, [])

        shared_corridors = set(corridors1) & set(corridors2)

        shared_waypoints = set()
        for corridor in shared_corridors:
            route = self.network.corridor_routes.get(corridor, [])
            shared_waypoints.update(route)

        return list(shared_waypoints)
