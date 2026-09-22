









class NavigatorsRouter {
    constructor() {
        this.osrmBaseUrl = 'https://router.project-osrm.org/route/v1/driving';
    }

    









    async route(startLat, startLon, endLat, endLon, localMap = null, forceOffline = false) {
        if (!Number.isFinite(startLat) || !Number.isFinite(startLon) ||
            !Number.isFinite(endLat) || !Number.isFinite(endLon)) {
            return { success: false, error: 'Invalid start or destination coordinates.' };
        }

        
        if (!forceOffline && typeof navigator !== 'undefined' && navigator.onLine) {
            try {
                const onlineResult = await this.routeOnline(startLat, startLon, endLat, endLon);
                if (onlineResult.success) {
                    return onlineResult;
                }
                console.warn('[NavigatorsRouter] Online routing failed, falling back to local A* graph:', onlineResult.error);
            } catch (err) {
                console.warn('[NavigatorsRouter] Online route request error:', err.message);
            }
        }

        
        if (localMap && localMap.data && Array.isArray(localMap.data.roads) && localMap.data.roads.length > 0) {
            return this.routeOffline(startLat, startLon, endLat, endLon, localMap);
        }

        return {
            success: false,
            error: (typeof navigator !== 'undefined' && navigator.onLine)
                ? 'Routing service unavailable and local offline road database is not loaded.' 
                : 'Offline mode active and no downloaded street network covers this area.'
        };
    }

    


    async routeOnline(startLat, startLon, endLat, endLon) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 6000);

        try {
            const url = `${this.osrmBaseUrl}/${startLon},${startLat};${endLon},${endLat}?overview=full&geometries=geojson&steps=true`;
            const resp = await fetch(url, { signal: controller.signal });
            if (!resp.ok) {
                return { success: false, error: `OSRM server error (${resp.status})` };
            }
            const data = await resp.json();
            if (data.code !== 'Ok' || !data.routes || !data.routes.length) {
                return { success: false, error: data.message || 'No route found between coordinates.' };
            }

            const route = data.routes[0];
            const coordinates = route.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
            
            const steps = [];
            if (route.legs && route.legs[0] && route.legs[0].steps) {
                for (const step of route.legs[0].steps) {
                    const mType = step.maneuver?.type || 'turn';
                    const mMod = step.maneuver?.modifier || '';
                    const rName = step.name || 'road';
                    let instruction = step.maneuver?.instruction;
                    if (!instruction) {
                        if (mType === 'depart') instruction = 'Head towards ' + rName;
                        else if (mType === 'arrive') instruction = 'Arrive at destination';
                        else instruction = `${mType} ${mMod} onto ${rName}`.trim();
                    }
                    steps.push({
                        instruction,
                        distance_m: step.distance,
                        duration_s: step.duration,
                        location: [step.maneuver.location[1], step.maneuver.location[0]]
                    });
                }
            }

            return {
                success: true,
                source: 'osrm_online',
                distance_m: route.distance,
                duration_s: route.duration,
                coordinates,
                steps
            };
        } catch (e) {
            return { success: false, error: e.name === 'AbortError' ? 'Route request timed out.' : e.message };
        } finally {
            clearTimeout(timeout);
        }
    }

    


    routeOffline(startLat, startLon, endLat, endLon, localMap) {
        const roads = localMap.data.roads;
        const origin = localMap.data.origin;
        const metersPerDegree = localMap.metersPerDegree;
        const lonScale = localMap.lonScale;

        
        const startENU = [(startLon - origin.lon) * lonScale, (startLat - origin.lat) * metersPerDegree];
        const endENU = [(endLon - origin.lon) * lonScale, (endLat - origin.lat) * metersPerDegree];

        
        const snapDist = 18.0; 
        const nodes = []; 

        function getOrCreateNode(x, y) {
            for (let i = 0; i < nodes.length; i++) {
                const n = nodes[i];
                const dx = n.x - x, dy = n.y - y;
                if (dx * dx + dy * dy <= snapDist * snapDist) {
                    return n.id;
                }
            }
            const id = nodes.length;
            nodes.push({ id, x, y, neighbors: [] });
            return id;
        }

        for (const road of roads) {
            if (!road.points || road.points.length < 2) continue;
            let prevNodeId = getOrCreateNode(road.points[0][0], road.points[0][1]);
            for (let i = 1; i < road.points.length; i++) {
                const currNodeId = getOrCreateNode(road.points[i][0], road.points[i][1]);
                if (currNodeId !== prevNodeId) {
                    const dx = nodes[currNodeId].x - nodes[prevNodeId].x;
                    const dy = nodes[currNodeId].y - nodes[prevNodeId].y;
                    const dist = Math.hypot(dx, dy);

                    nodes[prevNodeId].neighbors.push({ id: currNodeId, dist, roadName: road.name });
                    nodes[currNodeId].neighbors.push({ id: prevNodeId, dist, roadName: road.name });
                }
                prevNodeId = currNodeId;
            }
        }

        if (nodes.length < 2) {
            return { success: false, error: 'Insufficient road network vertices to compute offline route.' };
        }

        
        let startNode = -1, endNode = -1;
        let minStartDist = Infinity, minEndDist = Infinity;

        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            const dStart = Math.hypot(n.x - startENU[0], n.y - startENU[1]);
            const dEnd = Math.hypot(n.x - endENU[0], n.y - endENU[1]);
            if (dStart < minStartDist) { minStartDist = dStart; startNode = i; }
            if (dEnd < minEndDist) { minEndDist = dEnd; endNode = i; }
        }

        if (minStartDist > 800 || minEndDist > 800) {
            return { success: false, error: 'Start or destination is beyond the coverage of the downloaded road network.' };
        }

        if (startNode === endNode) {
            return {
                success: true,
                source: 'local_astar',
                distance_m: minStartDist + minEndDist,
                duration_s: Math.round((minStartDist + minEndDist) / 8.33), 
                coordinates: [[startLat, startLon], [endLat, endLon]],
                steps: [{ instruction: 'Arrive at destination', distance_m: minStartDist + minEndDist, duration_s: 10, location: [endLat, endLon] }]
            };
        }

        
        const openSet = new Set([startNode]);
        const cameFrom = new Map();
        const cameFromEdge = new Map();

        const gScore = new Float64Array(nodes.length).fill(Infinity);
        gScore[startNode] = 0;

        const fScore = new Float64Array(nodes.length).fill(Infinity);
        fScore[startNode] = Math.hypot(nodes[startNode].x - nodes[endNode].x, nodes[startNode].y - nodes[endNode].y);

        let iterations = 0;
        const maxIterations = 20000;

        while (openSet.size > 0 && iterations++ < maxIterations) {
            
            let current = -1;
            let lowestF = Infinity;
            for (const nodeId of openSet) {
                if (fScore[nodeId] < lowestF) {
                    lowestF = fScore[nodeId];
                    current = nodeId;
                }
            }

            if (current === endNode) {
                
                const pathNodeIds = [current];
                while (cameFrom.has(current)) {
                    current = cameFrom.get(current);
                    pathNodeIds.unshift(current);
                }

                
                const coords = [[startLat, startLon]];
                let totalDist = minStartDist;

                for (let i = 0; i < pathNodeIds.length; i++) {
                    const n = nodes[pathNodeIds[i]];
                    const lat = origin.lat + n.y / metersPerDegree;
                    const lon = origin.lon + n.x / lonScale;
                    coords.push([lat, lon]);
                }
                coords.push([endLat, endLon]);
                totalDist += gScore[endNode] + minEndDist;

                
                const steps = [{ instruction: 'Start along local road network', distance_m: minStartDist, duration_s: Math.round(minStartDist / 8.33), location: coords[0] }];
                for (let i = 1; i < pathNodeIds.length - 1; i++) {
                    const edge = cameFromEdge.get(pathNodeIds[i]);
                    if (edge && edge.roadName) {
                        steps.push({
                            instruction: `Continue on ${edge.roadName}`,
                            distance_m: edge.dist,
                            duration_s: Math.round(edge.dist / 8.33),
                            location: coords[i]
                        });
                    }
                }
                steps.push({ instruction: 'Arrive at destination', distance_m: minEndDist, duration_s: Math.round(minEndDist / 8.33), location: [endLat, endLon] });

                return {
                    success: true,
                    source: 'local_astar',
                    distance_m: Math.round(totalDist),
                    duration_s: Math.round(totalDist / 8.33),
                    coordinates: coords,
                    steps
                };
            }

            openSet.delete(current);

            const currNode = nodes[current];
            for (const neighbor of currNode.neighbors) {
                const neighborId = neighbor.id;
                const tentativeG = gScore[current] + neighbor.dist;

                if (tentativeG < gScore[neighborId]) {
                    cameFrom.set(neighborId, current);
                    cameFromEdge.set(neighborId, neighbor);
                    gScore[neighborId] = tentativeG;
                    const h = Math.hypot(nodes[neighborId].x - nodes[endNode].x, nodes[neighborId].y - nodes[endNode].y);
                    fScore[neighborId] = tentativeG + h;
                    openSet.add(neighborId);
                }
            }
        }

        return {
            success: false,
            error: 'No traversable road connection found between points on the local road network.'
        };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = NavigatorsRouter;
} else if (typeof window !== 'undefined') {
    window.NavigatorsRouter = NavigatorsRouter;
}
