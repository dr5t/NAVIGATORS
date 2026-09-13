/** The downloaded OSM road database is shared by the basemap and EKF matcher. */
class LocalMap {
    static async load() {
        const saved = typeof caches !== 'undefined' ? await (await caches.open('navigators-map-data-v1')).match('./data/road_network.json') : null;
        if (!saved) return null;
        const response = await saved.json();
        return new LocalMap(response);
    }

    static async download(lat, lon) {
        if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 84 || Math.abs(lon) > 179) throw new Error('Choose an area away from the poles and date line.');
        const scale = 6371000 * Math.PI / 180;
        const dlat = 1000 / scale, dlon = dlat / Math.cos(lat * Math.PI / 180);
        const bounds = [lat - dlat, lon - dlon, lat + dlat, lon + dlon];
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 35000);
        try {
            const response = await fetch('https://overpass-api.de/api/interpreter', {
                method: 'POST', body: new URLSearchParams({ data: `[out:json][timeout:25];way["highway"]["area"!="yes"](${bounds.join(',')});out geom;` }), signal: controller.signal,
            });
            if (!response.ok) throw new Error(`Map download failed (${response.status}). Try again later.`);
            const result = await response.json();
            if (result.remark) throw new Error('Map service returned an incomplete area. Try again later.');
            const roads = (result.elements || []).filter(way => way.type === 'way' && way.geometry?.length >= 2).map(way => ({
                id: String(way.id), name: way.tags?.name || way.tags?.highway || 'Unnamed road',
                points: way.geometry.map(point => [(point.lon - lon) * scale * Math.cos(lat * Math.PI / 180), (point.lat - lat) * scale]),
            }));
            return new LocalMap({ origin: { lat, lon }, bounds, downloaded_at: new Date().toISOString(), roads });
        } finally { clearTimeout(timeout); }
    }

    constructor(data) {
        if (!Number.isFinite(data.origin?.lat) || !Number.isFinite(data.origin?.lon) ||
            !Array.isArray(data.roads) || !data.roads.length) {
            throw new Error('Invalid or empty local OSM database.');
        }
        this.data = data;
        // Must match the downloader's equirectangular projection.
        this.metersPerDegree = 6371000 * Math.PI / 180;
        this.lonScale = this.metersPerDegree * Math.cos(data.origin.lat * Math.PI / 180);
        this.bounds = [Infinity, Infinity, -Infinity, -Infinity];
        for (const road of data.roads) {
            if (!Array.isArray(road.points) || road.points.length < 2) throw new Error('Invalid OSM road.');
            for (const point of road.points) {
                if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) {
                    throw new Error('Invalid OSM coordinate.');
                }
                this.bounds[0] = Math.min(this.bounds[0], point[0]);
                this.bounds[1] = Math.min(this.bounds[1], point[1]);
                this.bounds[2] = Math.max(this.bounds[2], point[0]);
                this.bounds[3] = Math.max(this.bounds[3], point[1]);
            }
        }
    }

    toLatLon([east, north]) {
        return [this.data.origin.lat + north / this.metersPerDegree,
            this.data.origin.lon + east / this.lonScale];
    }

    toENU(lat, lon) {
        return [(lon - this.data.origin.lon) * this.lonScale,
            (lat - this.data.origin.lat) * this.metersPerDegree];
    }

    contains(lat, lon) {
        if (this.data.bounds) {
            const [south, west, north, east] = this.data.bounds;
            return lat >= south && lat <= north && lon >= west && lon <= east;
        }
        const [e, n] = this.toENU(lat, lon);
        return e >= this.bounds[0] && e <= this.bounds[2] && n >= this.bounds[1] && n <= this.bounds[3];
    }

    draw(map) {
        if (!this.data || !this.data.roads) return;

        const renderer = L.canvas({ padding: 0.5 });
        const layers = this.data.roads.map(road => {
            const line = L.polyline(road.points.map(p => this.toLatLon(p)), {
                renderer, color: '#8d9c83', weight: 2.4, opacity: 0.78,
            });
            const label = document.createElement('span');
            label.textContent = road.name || 'Unnamed road';
            line.bindTooltip(label, { sticky: true });
            return line;
        });
        this.layer = L.featureGroup(layers).addTo(map).bringToBack();
        map.attributionControl.addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> · ODbL · Local road map');
    }
}

if (typeof module !== 'undefined') module.exports = LocalMap;
