
class CausalIMUFilter {
    constructor(removeGravity = true) {
        this.history = [];
        this.filtered = null;
        this.removeGravity = removeGravity;
    }
    step(sample, dt) {
        this.history.push([...sample]);
        if (this.history.length > 5) this.history.shift();
        const median = sample.map((_, j) => {
            const column = this.history.map(row => row[j]).sort((a, b) => a - b);
            const middle = Math.floor(column.length / 2);
            return column.length % 2 ? column[middle] : (column[middle - 1] + column[middle]) / 2;
        });
        const gain = 1 - Math.exp(-2 * Math.PI * 20 * dt);
        this.filtered = this.filtered === null ? median : this.filtered.map((v, j) => v + gain * (median[j] - v));
        const filtered = [...this.filtered];
        if (this.removeGravity) {
            filtered[2] -= 9.81;
        }
        return filtered;
    }
}
if (typeof module !== 'undefined') module.exports = CausalIMUFilter;
