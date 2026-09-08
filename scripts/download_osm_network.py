"""
Navigators IDR — OpenStreetMap Downloader
Downloads road network data from OpenStreetMap via Overpass API for local map matching.
"""

import json
import urllib.request
import urllib.parse
import os
import math

def latlon_to_enu(lat, lon, lat_ref, lon_ref):
    """Simple equirectangular projection to ENU in meters."""
    R = 6371000.0  # Earth radius in meters
    dLat = math.radians(lat - lat_ref)
    dLon = math.radians(lon - lon_ref)
    
    x = R * dLon * math.cos(math.radians(lat_ref))
    y = R * dLat
    return [x, y]

def download_osm_network(lat: float, lon: float, radius: float = 1000.0, output_file: str = "data/road_network.json"):
    """
    Download road network from OSM and convert to ENU relative to lat/lon.
    """
    print(f"Downloading OSM road network around {lat}, {lon} (Radius: {radius}m)...")
    
    # Calculate bounding box roughly
    # 1 deg lat = 111km
    dlat = radius / 111000.0
    dlon = radius / (111000.0 * math.cos(math.radians(lat)))
    
    bbox = f"{lat - dlat},{lon - dlon},{lat + dlat},{lon + dlon}"
    
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]["highway"!="pedestrian"]["highway"!="footway"]["highway"!="path"]["highway"!="steps"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    
    url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query.strip())
    
    req = urllib.request.Request(url, headers={'User-Agent': 'NavigatorsIDR/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error downloading OSM data: {e}")
        return False
        
    print(f"Processing {len(data.get('elements', []))} OSM elements...")
    
    nodes = {}
    roads = []
    
    for element in data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = {
                'lat': element['lat'],
                'lon': element['lon'],
                'enu': latlon_to_enu(element['lat'], element['lon'], lat, lon)
            }
            
    for element in data['elements']:
        if element['type'] == 'way':
            if 'nodes' in element and len(element['nodes']) > 1:
                way_nodes = []
                for n_id in element['nodes']:
                    if n_id in nodes:
                        way_nodes.append(nodes[n_id]['enu'])
                
                if len(way_nodes) > 1:
                    tags = element.get('tags', {})
                    name = tags.get('name', 'Unknown Road')
                    oneway = tags.get('oneway', 'no') == 'yes'
                    maxspeed = tags.get('maxspeed', '50')
                    try:
                        speed_limit = float(maxspeed.replace(' mph', '').replace(' km/h', '').strip())
                        if 'mph' in maxspeed:
                            speed_limit *= 1.60934
                    except:
                        speed_limit = 50.0
                        
                    roads.append({
                        'id': str(element['id']),
                        'name': name,
                        'speed_limit': speed_limit,
                        'one_way': oneway,
                        'points': way_nodes
                    })
                    
    print(f"Successfully extracted {len(roads)} drivable roads.")
    
    if not roads:
        print("No roads found; existing database was preserved.")
        return False

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    network_data = {
        'source': 'OpenStreetMap',
        'license': 'ODbL-1.0',
        'projection': 'equirectangular',
        'earth_radius_m': 6371000.0,
        'bounds': [lat - dlat, lon - dlon, lat + dlat, lon + dlon],
        'origin': {'lat': lat, 'lon': lon},
        'roads': roads
    }
    
    with open(output_file, 'w') as f:
        json.dump(network_data, f)
        
    print(f"Saved road network to {output_file}")
    
    # Also save to simulator directory
    sim_file = "simulator/data/road_network.json"
    os.makedirs(os.path.dirname(sim_file), exist_ok=True)
    with open(sim_file, 'w') as f:
        json.dump(network_data, f)
    print(f"Saved road network to {sim_file}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=13.0326)
    parser.add_argument("--lon", type=float, default=77.5582)
    parser.add_argument("--radius", type=float, default=2000.0, help="Download radius in meters")
    parser.add_argument("--output", default="data/road_network.json")
    args = parser.parse_args()
    if not (-85 < args.lat < 85 and -180 <= args.lon <= 180 and args.radius > 0):
        parser.error("Use latitude between -85 and 85, longitude between -180 and 180, and a positive radius")
    raise SystemExit(0 if download_osm_network(args.lat, args.lon, args.radius, args.output) else 1)
