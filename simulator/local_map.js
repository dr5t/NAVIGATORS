/**
 * Navigators IDR - Local Offline OpenStreetMap & POI Database
 * Manages downloaded OSM street vector networks and community points of interest.
 * Handles equirectangular coordinate transforms, spatial bounds, and map layer rendering.
 */

class LocalMap {
    static async load() {
        const saved = typeof caches !== 'undefined' ? await (await caches.open('navigators-map-data-v1')).match('./data/road_network.json') : null;
        if (!saved) return null;
        const response = await saved.json();
        return new LocalMap(response);
    }

    static async download(lat, lon) {
        if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 84 || Math.abs(lon) > 179) {
            throw new Error('Choose an area away from the poles and date line.');
        }
        const scale = 6371000 * Math.PI / 180;
        const dlat = 1000 / scale, dlon = dlat / Math.cos(lat * Math.PI / 180);
        const bounds = [lat - dlat, lon - dlon, lat + dlat, lon + dlon];
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 35000);

        try {
            // Query both street ways and amenity / POI nodes
            const query = `[out:json][timeout:25];(way["highway"]["area"!="yes"](${bounds.join(',')});node["amenity"~"fuel|hospital|pharmacy|atm|parking|restaurant|cafe"](${bounds.join(',')});node["charging_station"](${bounds.join(',')}););out geom;`;
            const response = await fetch('https://overpass-api.de/api/interpreter', {
                method: 'POST',
                body: new URLSearchParams({ data: query }),
                signal: controller.signal,
            });

            if (!response.ok) throw new Error(`Map download failed (${response.status}). Try again later.`);
            const result = await response.json();
            if (result.remark) throw new Error('Map service returned an incomplete area. Try again later.');

            const roads = (result.elements || []).filter(el => el.type === 'way' && el.geometry?.length >= 2).map(way => ({
                id: String(way.id),
                name: way.tags?.name || way.tags?.highway || 'Unnamed road',
                points: way.geometry.map(point => [(point.lon - lon) * scale * Math.cos(lat * Math.PI / 180), (point.lat - lat) * scale]),
            }));

            const categoryMap = {
                fuel: 'fuel', gas_station: 'fuel',
                hospital: 'hospital', clinic: 'hospital',
                pharmacy: 'pharmacy',
                atm: 'atm', bank: 'atm',
                parking: 'parking',
                restaurant: 'restaurant', cafe: 'restaurant', fast_food: 'restaurant',
                charging_station: 'charging_station'
            };

            const pois = (result.elements || []).filter(el => el.type === 'node' && (el.tags?.amenity || el.tags?.charging_station)).map(node => {
                const raw = node.tags?.amenity || (node.tags?.charging_station ? 'charging_station' : 'landmark');
                const cat = categoryMap[raw] || 'landmark';
                return {
                    id: String(node.id),
                    name: node.tags?.name || (cat.charAt(0).toUpperCase() + cat.slice(1).replace('_', ' ')),
                    category: cat,
                    lat: node.lat,
                    lon: node.lon,
                    source: 'osm'
                };
            });

            return new LocalMap({
                origin: { lat, lon },
                bounds,
                downloaded_at: new Date().toISOString(),
                roads,
                pois
            });
        } finally {
            clearTimeout(timeout);
        }
    }

    constructor(data) {
        if (!Number.isFinite(data.origin?.lat) || !Number.isFinite(data.origin?.lon) ||
            !Array.isArray(data.roads) || !data.roads.length) {
            throw new Error('Invalid or empty local OSM database.');
        }
        this.data = data;
        this.data.pois = Array.isArray(data.pois) ? data.pois : [];

        // Synchronize approved community contributions into the local POI database
        this.syncApprovedCommunityPlaces();

        // Equirectangular projection constants
        this.metersPerDegree = 6371000 * Math.PI / 180;
        this.lonScale = this.metersPerDegree * Math.cos(data.origin.lat * Math.PI / 180);
        this.bounds = [Infinity, Infinity, -Infinity, -Infinity];

        for (const road of data.roads) {
            if (!Array.isArray(road.points) || road.points.length < 2) continue;
            for (const point of road.points) {
                if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) {
                    continue;
                }
                this.bounds[0] = Math.min(this.bounds[0], point[0]);
                this.bounds[1] = Math.min(this.bounds[1], point[1]);
                this.bounds[2] = Math.max(this.bounds[2], point[0]);
                this.bounds[3] = Math.max(this.bounds[3], point[1]);
            }
        }
    }

    syncApprovedCommunityPlaces() {
        if (typeof localStorage === 'undefined') return;
        try {
            const raw = localStorage.getItem('navigators_place_submissions');
            if (!raw) return;
            const submissions = JSON.parse(raw);
            const approved = submissions.filter(s => s.state === 'Approved' && s.location);

            for (const item of approved) {
                const parts = item.location.split(',').map(s => parseFloat(s.trim()));
                if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
                    // Check if already in pois
                    if (!this.data.pois.some(p => p.id === item.id)) {
                        this.data.pois.push({
                            id: item.id,
                            name: item.name,
                            category: item.category || 'landmark',
                            lat: parts[0],
                            lon: parts[1],
                            hours: item.hours,
                            phone: item.phone,
                            source: 'community'
                        });
                    }
                }
            }
        } catch (e) {
            console.warn('[LocalMap] Community sync error:', e);
        }
    }

    getPois(category = null, refLat = null, refLon = null) {
        let list = this.data.pois || [];
        if (category && category !== 'all') {
            list = list.filter(p => p.category === category);
        }
        if (Number.isFinite(refLat) && Number.isFinite(refLon)) {
            const scale = 6371000 * Math.PI / 180;
            const cosLat = Math.cos(refLat * Math.PI / 180);
            list = list.map(p => {
                const dx = (p.lon - refLon) * scale * cosLat;
                const dy = (p.lat - refLat) * scale;
                const dist = Math.hypot(dx, dy);
                return { ...p, distance_m: Math.round(dist) };
            }).sort((a, b) => a.distance_m - b.distance_m);
        }
        return list;
    }

    toLatLon([east, north]) {
        return [
            this.data.origin.lat + north / this.metersPerDegree,
            this.data.origin.lon + east / this.lonScale
        ];
    }

    toENU(lat, lon) {
        return [
            (lon - this.data.origin.lon) * this.lonScale,
            (lat - this.data.origin.lat) * this.metersPerDegree
        ];
    }

    contains(lat, lon) {
        if (this.data.bounds) {
            const [south, west, north, east] = this.data.bounds;
            return lat >= south && lat <= north && lon >= west && lon <= east;
        }
        const [e, n] = this.toENU(lat, lon);
        return e >= this.bounds[0] && e <= this.bounds[2] && n >= this.bounds[1] && n <= this.bounds[3];
    }

    centerView(map) {
        if (map && this.data.origin) {
            map.setView([this.data.origin.lat, this.data.origin.lon], 15);
        }
    }

    draw(map) {
        if (!this.data || !this.data.roads || !map) return;

        // 1. Render road vector geometries
        const renderer = L.canvas({ padding: 0.5 });
        const roadLayers = this.data.roads.map(road => {
            const line = L.polyline(road.points.map(p => this.toLatLon(p)), {
                renderer,
                color: '#4c7b59',
                weight: 2.2,
                opacity: 0.75,
            });
            const label = document.createElement('span');
            label.textContent = road.name || 'Unnamed road';
            line.bindTooltip(label, { sticky: true });
            return line;
        });

        if (this.layer) map.removeLayer(this.layer);
        this.layer = L.featureGroup(roadLayers).addTo(map).bringToBack();

        // 2. Render POI markers
        if (this.poiLayer) map.removeLayer(this.poiLayer);
        const poiMarkers = (this.data.pois || []).map(poi => {
            const iconHtml = `<div class="poi-pin-inner poi-${poi.category}"></div>`;
            const icon = L.divIcon({
                className: 'poi-map-pin',
                html: iconHtml,
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });
            const marker = L.marker([poi.lat, poi.lon], { icon });
            const catLabel = poi.category.charAt(0).toUpperCase() + poi.category.slice(1).replace('_', ' ');
            const popupHtml = `
                <div style="font-family: var(--font-main); font-size: 12px; color: var(--text-primary); line-height: 1.4;">
                    <strong style="font-size: 13px; display: block; color: var(--text-primary);">${poi.name}</strong>
                    <div style="font-size: 11px; color: var(--accent-cyan); margin: 2px 0 6px;">${catLabel} ${poi.source === 'community' ? '· Community Verified' : ''}</div>
                    ${poi.hours ? `<div style="font-size: 10px; color: var(--text-muted); margin-bottom: 6px;">${poi.hours}</div>` : ''}
                    <button class="button primary" style="width: 100%; padding: 4px 8px; font-size: 11px;" onclick="window.setDestination(${poi.lat}, ${poi.lon}, '${poi.name.replace(/'/g, "\\'")}')">
                        Route Here
                    </button>
                </div>
            `;
            marker.bindPopup(popupHtml);
            return marker;
        });

        this.poiLayer = L.featureGroup(poiMarkers).addTo(map);
        map.attributionControl.addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> · ODbL');
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = LocalMap;
} else if (typeof window !== 'undefined') {
    window.LocalMap = LocalMap;
}
