/**
 * Lightweight Matrix Math Library for JavaScript EKF
 * Avoids the need for heavy external dependencies.
 */

class Matrix {
    static zeros(rows, cols) {
        let m = [];
        for (let i = 0; i < rows; i++) {
            m.push(new Array(cols).fill(0));
        }
        return m;
    }

    static eye(n) {
        let m = Matrix.zeros(n, n);
        for (let i = 0; i < n; i++) m[i][i] = 1;
        return m;
    }

    static add(A, B) {
        let r = A.length, c = A[0].length;
        let m = Matrix.zeros(r, c);
        for (let i = 0; i < r; i++)
            for (let j = 0; j < c; j++)
                m[i][j] = A[i][j] + B[i][j];
        return m;
    }

    static sub(A, B) {
        let r = A.length, c = A[0].length;
        let m = Matrix.zeros(r, c);
        for (let i = 0; i < r; i++)
            for (let j = 0; j < c; j++)
                m[i][j] = A[i][j] - B[i][j];
        return m;
    }

    static scale(A, s) {
        let r = A.length, c = A[0].length;
        let m = Matrix.zeros(r, c);
        for (let i = 0; i < r; i++)
            for (let j = 0; j < c; j++)
                m[i][j] = A[i][j] * s;
        return m;
    }

    static mul(A, B) {
        let rA = A.length, cA = A[0].length, cB = B[0].length;
        let m = Matrix.zeros(rA, cB);
        for (let i = 0; i < rA; i++) {
            for (let j = 0; j < cB; j++) {
                let sum = 0;
                for (let k = 0; k < cA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                m[i][j] = sum;
            }
        }
        return m;
    }

    static transpose(A) {
        let r = A.length, c = A[0].length;
        let m = Matrix.zeros(c, r);
        for (let i = 0; i < r; i++)
            for (let j = 0; j < c; j++)
                m[j][i] = A[i][j];
        return m;
    }

    static inv2x2(A) {
        let det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        if (Math.abs(det) < 1e-9) throw new Error("Matrix is singular");
        let invDet = 1.0 / det;
        return [
            [ A[1][1] * invDet, -A[0][1] * invDet],
            [-A[1][0] * invDet,  A[0][0] * invDet]
        ];
    }

    static inv3x3(A) {
        let a = A[0][0], b = A[0][1], c = A[0][2];
        let d = A[1][0], e = A[1][1], f = A[1][2];
        let g = A[2][0], h = A[2][1], i = A[2][2];

        let det = a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g);
        if (Math.abs(det) < 1e-9) throw new Error("Matrix is singular");
        let invDet = 1.0 / det;

        return [
            [(e*i - f*h)*invDet, (c*h - b*i)*invDet, (b*f - c*e)*invDet],
            [(f*g - d*i)*invDet, (a*i - c*g)*invDet, (c*d - a*f)*invDet],
            [(d*h - e*g)*invDet, (b*g - a*h)*invDet, (a*e - b*d)*invDet]
        ];
    }
}

// Export for ES modules or global window
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Matrix;
} else {
    window.Matrix = Matrix;
}
