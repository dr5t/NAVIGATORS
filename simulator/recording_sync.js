
class RecordingSync {
    constructor() {
        this.token = sessionStorage.getItem('pc-pairing-token') || '';
        this.trips = [];
        this.busy = false;
        setInterval(() => this.flush(), 5000);
        window.addEventListener('online', () => this.flush());
    }

    status(message, isError = false) { 
        const el = document.getElementById('pcSyncStatus');
        if (el) {
            el.textContent = message;
            el.style.color = isError ? 'red' : '';
        }
    }

    async pair(token) {
        const response = await fetch('/recordings/status', { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
        if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('Open the app from the PC HTTPS server and check its pairing code.');
        const report = await response.json();
        if (report.status !== 'connected') throw new Error('This address is not the PC recording server.');
        this.token = token;
        sessionStorage.setItem('pc-pairing-token', token);
        this.status(`PC connected · ${report.completed_trips} completed trips. New recordings will sync.`);
        await this.flush();
    }

    start(data, metadata) {
        if (!this.token) return null;
        const trip = { id: crypto.randomUUID(), data, metadata, cursor: 0, sequence: 0, ended: false };
        this.trips.push(trip);
        return trip;
    }

    finish(trip) {
        if (trip) trip.ended = true;
        return this.flush();
    }

    async flush() {
        if (this.busy || !this.token) return;
        this.busy = true;
        try {
            const cache = await caches.open('navigators-recording-outbox-v1');
            for (const trip of [...this.trips]) {
                while (trip.cursor < trip.data.timestamps.length || trip.ended) {
                    const end = Math.min(trip.cursor + 500, trip.data.timestamps.length);
                    const final = trip.ended && end === trip.data.timestamps.length;
                    const data = Object.fromEntries(Object.entries(trip.data).map(([key, values]) => [key, values.slice(trip.cursor, end)]));
                    const url = `/recordings/${trip.id}/${String(trip.sequence).padStart(4, '0')}`;
                    await cache.put(url, new Response(JSON.stringify({ metadata: trip.metadata, data, final }), { headers: { 'Content-Type': 'application/json' } }));
                    trip.cursor = end;
                    trip.sequence++;
                    if (final) { this.trips.splice(this.trips.indexOf(trip), 1); break; }
                }
            }
            const pending = await cache.keys();
            if (!pending.length) return;
            this.status(`${pending.length} batches saved on phone · sending to PC…`);
            for (const request of pending.sort((a, b) => a.url.localeCompare(b.url))) {
                const payload = await (await cache.match(request)).text();
                const response = await fetch(request.url, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.token}` }, body: payload, signal: AbortSignal.timeout(10000) });
                if (!response.ok) throw new Error(response.status === 401 ? 'Pair with the PC again.' : `PC rejected a batch (${response.status}); local copy retained.`);
                const receipt = await response.json();
                if (receipt.stored !== Number(new URL(request.url).pathname.split('/').at(-1))) throw new Error('PC acknowledgement was invalid.');
                await cache.delete(request);
            }
            this.status('All queued batches received by PC · data/phone_recordings');
        } catch (error) {
            this.status(`Sync paused: ${error.message} Reconnect and retry; keep the recording page open or stop and save a backup.`, true);
        } finally { this.busy = false; }
    }
}

window.recordingSync = new RecordingSync();
