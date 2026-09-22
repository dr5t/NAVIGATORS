/**
 * Navigators IDR - Map Matching (JavaScript Edge Port)
 * Snaps estimated positions to the road network to bound drift.
 */

class RoadSegment {
    constructor(id, start, end, name = "", speed_limit = 50.0, one_way = false) {
        this.id = id;
        this.start = start; // [East, North]
        this.end = end;     // [East, North]
        this.name = name;
        this.speed_limit = speed_limit;
        this.one_way = one_way;

        let dx = this.end[0] - this.start[0];
        let dy = this.end[1] - this.start[1];
        this.length = Math.sqrt(dx * dx + dy * dy);
        this.heading = 0.0;
        this.direction = [0, 0];

        if (this.length > 0) {
            this.heading = Math.atan2(dx, dy); // East, North
            this.direction = [dx / this.length, dy / this.length];
        }
    }
}

class RoadNetwork {
    constructor() {
        this.segments = [];
        this.spatialHash = new Map();
        this.gridSize = 200.0; // 200m grid cells
    }

    async loadOSMNetwork(url) {
        try {
            const response = await fetch(url);
            const data = await response.json();

            if (data.roads) {
                for (let road of data.roads) {
                    this.addRoad(road.points, road.id, road.name, road.speed_limit, road.one_way);
                }
            }
            this.buildSpatialHash();
            console.log(`[Map Matcher] Loaded ${this.segments.length} road segments.`);
            return true;
        } catch (e) {
            console.error("[Map Matcher] Failed to load OSM Network:", e);
            return false;
        }
    }

    _hash(x, y) {
        let gx = Math.floor(x / this.gridSize);
        let gy = Math.floor(y / this.gridSize);
        return `${gx},${gy}`;
    }

    buildSpatialHash() {
        this.spatialHash.clear();
        for (let seg of this.segments) {
            // Index the entire segment so long roads remain matchable at their midpoint.
            const minX = Math.floor(Math.min(seg.start[0], seg.end[0]) / this.gridSize);
            const maxX = Math.floor(Math.max(seg.start[0], seg.end[0]) / this.gridSize);
            const minY = Math.floor(Math.min(seg.start[1], seg.end[1]) / this.gridSize);
            const maxY = Math.floor(Math.max(seg.start[1], seg.end[1]) / this.gridSize);
            for (let x = minX; x <= maxX; x++) {
                for (let y = minY; y <= maxY; y++) {
                    const key = `${x},${y}`;
                    if (!this.spatialHash.has(key)) this.spatialHash.set(key, new Set());
                    this.spatialHash.get(key).add(seg);
                }
            }
        }
    }

    getCandidateSegments(position, searchRadius) {
        if (this.spatialHash.size === 0) return this.segments;

        let candidates = new Set();
        let r = searchRadius + 100.0; // Margin

        let minX = position[0] - r;
        let maxX = position[0] + r;
        let minY = position[1] - r;
        let maxY = position[1] + r;

        for (let x = minX; x <= maxX + this.gridSize; x += this.gridSize) {
            for (let y = minY; y <= maxY + this.gridSize; y += this.gridSize) {
                let h = this._hash(x, y);
                if (this.spatialHash.has(h)) {
                    for (let seg of this.spatialHash.get(h)) {
                        candidates.add(seg);
                    }
                }
            }
        }
        return Array.from(candidates);
    }

    addSegment(segment) {
        this.segments.push(segment);
    }

    addRoad(points, road_id = "", name = "", speed_limit = 50.0, one_way = false) {
        for (let i = 0; i < points.length - 1; i++) {
            let seg = new RoadSegment(
                `${road_id}_${i}`,
                points[i],
                points[i + 1],
                name,
                speed_limit,
                one_way
            );
            this.addSegment(seg);
        }
    }

    generateGridNetwork(center = [0, 0], grid_size = 100.0, num_blocks = 5) {
        let half = num_blocks * grid_size / 2;

        // East-West
        for (let i = 0; i <= num_blocks; i++) {
            let y = center[1] - half + i * grid_size;
            this.addRoad([
                [center[0] - half, y],
                [center[0] + half, y]
            ], `ew_${i}`, `East-West Road ${i}`);
        }

        // North-South
        for (let i = 0; i <= num_blocks; i++) {
            let x = center[0] - half + i * grid_size;
            this.addRoad([
                [x, center[1] - half],
                [x, center[1] + half]
            ], `ns_${i}`, `North-South Road ${i}`);
        }
    }

    nearestPointOnSegment(point, segment) {
        let vx = point[0] - segment.start[0];
        let vy = point[1] - segment.start[1];

        let ux = segment.end[0] - segment.start[0];
        let uy = segment.end[1] - segment.start[1];

        let length_sq = ux * ux + uy * uy;
        if (length_sq < 1e-10) {
            return { nearest: [...segment.start], dist: Math.sqrt(vx * vx + vy * vy) };
        }

        let dot = vx * ux + vy * uy;
        let t = Math.max(0.0, Math.min(1.0, dot / length_sq));

        let nx = segment.start[0] + t * ux;
        let ny = segment.start[1] + t * uy;

        let dx = point[0] - nx;
        let dy = point[1] - ny;
        let dist = Math.sqrt(dx * dx + dy * dy);

        return { nearest: [nx, ny], dist: dist };
    }
}

class GeometricMapMatcher {
    constructor(road_network, search_radius = 50.0) {
        this.roads = road_network;
        this.search_radius = search_radius;
    }

    match(position, heading = null) {
        let best_point = [...position];
        let best_distance = Infinity;
        let best_segment = null;

        let candidates = this.roads.getCandidateSegments(position, this.search_radius);

        for (let seg of candidates) {
            let res = this.roads.nearestPointOnSegment(position, seg);
            let nearest = res.nearest;
            let dist = res.dist;

            if (dist < best_distance && dist <= this.search_radius) {
                if (heading !== null) {
                    let heading_diff = Math.abs(heading - seg.heading);
                    heading_diff = Math.min(heading_diff, 2 * Math.PI - heading_diff);

                    if (!seg.one_way) {
                        heading_diff = Math.min(heading_diff, Math.abs(heading_diff - Math.PI));
                    }

                    if (heading_diff > Math.PI / 3) {
                        dist *= (1 + heading_diff);
                    }
                }

                if (dist < best_distance) {
                    best_distance = dist;
                    best_point = nearest;
                    best_segment = seg;
                }
            }
        }

        let confidence = Math.max(0.0, 1.0 - best_distance / this.search_radius);

        return {
            snapped_position: best_point,
            matched_segment: best_segment,
            distance_to_road: best_distance,
            confidence: confidence,
            heading_correction: best_segment ? best_segment.heading : null
        };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RoadSegment, RoadNetwork, GeometricMapMatcher };
} else {
    window.RoadNetwork = RoadNetwork;
    window.GeometricMapMatcher = GeometricMapMatcher;
}
