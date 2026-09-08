/** The downloaded OSM road database is shared by the basemap and EKF matcher. */
class LocalMap {
    static async load() {
        const response = await fetch('./data/road_network.json');
        if (!response.ok) throw new Error('Local OSM database is missing. Download the demo area first.');
        return new LocalMap(await response.json());
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
        const [e, n] = this.toENU(lat, lon);
        return e >= this.bounds[0] && e <= this.bounds[2] && n >= this.bounds[1] && n <= this.bounds[3];
    }

    draw(map) {
        const renderer = L.canvas({ padding: 0.5 });
        const layers = this.data.roads.map(road => {
            const line = L.polyline(road.points.map(p => this.toLatLon(p)), {
                renderer, color: '#536780', weight: 3, opacity: 0.9,
            });
            const label = document.createElement('span');
            label.textContent = road.name || 'Unnamed road';
            line.bindTooltip(label, { sticky: true });
            return line;
        });
        L.featureGroup(layers).addTo(map).bringToBack();
        map.attributionControl.addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> · ODbL · Local road map');
        map.fitBounds([this.toLatLon(this.bounds.slice(0, 2)), this.toLatLon(this.bounds.slice(2))]);
    }
}

if (typeof module !== 'undefined') module.exports = LocalMap;
