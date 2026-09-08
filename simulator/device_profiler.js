/** Timings are collected on the browser/device actually running this code. */
class DeviceProfiler {
    constructor() {
        this.reset();
        this.startupMs = null;
        this.modelBytes = null;
    }
    reset() {
        this.samples = { tcn: [], ekf: [], map_matching: [], total_loop: [] };
        this.loops = 0;
    }
    record(component, milliseconds) {
        if (!Number.isFinite(milliseconds)) return;
        const values = this.samples[component];
        values.push(milliseconds);
        if (values.length > 600) values.shift();
    }
    report() {
        const timings = {};
        for (const [key, samples] of Object.entries(this.samples)) {
            const sorted = [...samples].sort((a, b) => a - b);
            timings[key] = { count: sorted.length,
                mean_ms: sorted.length ? sorted.reduce((a, b) => a + b, 0) / sorted.length : null,
                p50_ms: sorted.length ? sorted[Math.floor((sorted.length - 1) * 0.5)] : null,
                p95_ms: sorted.length ? sorted[Math.ceil((sorted.length - 1) * 0.95)] : null,
                max_ms: sorted.length ? sorted.at(-1) : null };
        }
        return { label: 'Measured in this browser; identify the physical device before citing results',
            recorded_at: new Date().toISOString(), user_agent: navigator.userAgent,
            platform: navigator.platform, hardware_concurrency: navigator.hardwareConcurrency,
            startup_ms: this.startupMs, model_size_bytes: this.modelBytes,
            page_load_ms: performance.getEntriesByType?.('navigation')[0]?.loadEventEnd || null,
            ram_bytes: null, ram_note: 'Browser does not expose reliable process RAM. Measure with the target OS profiler.',
            approximate_js_heap_bytes: performance.memory?.usedJSHeapSize ?? null,
            heap_note: 'Optional Chromium estimate; excludes some WASM/browser allocations and is not process RAM.',
            timing_scope: 'Latest 600 actual calls per component; first inference included. Total includes UI and fusion, excludes sensor event callback.',
            total_navigation_steps: this.loops, timings };
    }
}
if (typeof module !== 'undefined') module.exports = DeviceProfiler;
