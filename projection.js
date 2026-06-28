// ─── Projection Mapping & Fullscreen ─────────────────────────────────
// 4-corner perspective warp for ceiling projector alignment.
// Calibration saved to localStorage, toggled with P key.

(function () {
  const STORAGE_KEY = 'volumetric-led-projection';
  const HANDLE_RADIUS = 14;

  let calibrating = false;
  let uiHidden = false;
  let cursorTimeout = null;
  let activeCorner = null;

  // Default corners: [topLeft, topRight, bottomRight, bottomLeft] as fractions 0–1
  let corners = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 }
  ];

  loadCalibration();

  const overlay = document.getElementById('projection-overlay');
  const svg = document.getElementById('projection-svg');
  const status = document.getElementById('projection-status');

  function loadCalibration() {
    try {
      let saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        let parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length === 4) {
          corners = parsed;
        }
      }
    } catch (e) { /* ignore */ }
  }

  function saveCalibration() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(corners));
  }

  function isDefaultCorners() {
    return corners[0].x === 0 && corners[0].y === 0 &&
           corners[1].x === 1 && corners[1].y === 0 &&
           corners[2].x === 1 && corners[2].y === 1 &&
           corners[3].x === 0 && corners[3].y === 1;
  }

  // Compute CSS matrix3d from 4-corner mapping
  // Maps unit square corners to the specified screen positions
  function computeMatrix3d(w, h, pts) {
    // Source corners (original canvas: full viewport)
    let s = [
      [0, 0], [w, 0], [w, h], [0, h]
    ];
    // Destination corners (warped)
    let d = [
      [pts[0].x * w, pts[0].y * h],
      [pts[1].x * w, pts[1].y * h],
      [pts[2].x * w, pts[2].y * h],
      [pts[3].x * w, pts[3].y * h]
    ];

    // Solve the 3x3 projective transform from src → dst
    // then convert to CSS matrix3d (4x4 homogeneous)
    let m = getProjectiveTransform(s, d);
    if (!m) return null;

    // CSS matrix3d is column-major
    return `matrix3d(${m[0]},${m[3]},0,${m[6]},${m[1]},${m[4]},0,${m[7]},0,0,1,0,${m[2]},${m[5]},0,${m[8]})`;
  }

  // Solve 8-parameter projective transform (3x3 matrix with h33=1)
  // from 4 src points to 4 dst points
  function getProjectiveTransform(src, dst) {
    // Build 8x8 system: for each point pair (sx,sy) → (dx,dy):
    //   dx = (h11*sx + h12*sy + h13) / (h31*sx + h32*sy + 1)
    //   dy = (h21*sx + h22*sy + h23) / (h31*sx + h32*sy + 1)
    // Rearranged:
    //   h11*sx + h12*sy + h13 - h31*sx*dx - h32*sy*dx = dx
    //   h21*sx + h22*sy + h23 - h31*sx*dy - h32*sy*dy = dy
    let A = [];
    let b = [];
    for (let i = 0; i < 4; i++) {
      let sx = src[i][0], sy = src[i][1];
      let dx = dst[i][0], dy = dst[i][1];
      A.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
      b.push(dx);
      A.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
      b.push(dy);
    }
    let h = solveLinear8(A, b);
    if (!h) return null;
    // Return 3x3 matrix [h11,h12,h13, h21,h22,h23, h31,h32, 1]
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  // Gaussian elimination for 8x8 system
  function solveLinear8(A, b) {
    let n = 8;
    let M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
      }
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      if (Math.abs(M[col][col]) < 1e-10) return null;
      for (let row = col + 1; row < n; row++) {
        let f = M[row][col] / M[col][col];
        for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
      }
    }
    let x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = M[i][n];
      for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
      x[i] /= M[i][i];
    }
    return x;
  }

  function applyTransform() {
    let canvas = document.querySelector('canvas');
    if (!canvas) return;
    if (isDefaultCorners()) {
      canvas.style.transform = '';
      canvas.style.transformOrigin = '';
      return;
    }
    let w = window.innerWidth;
    let h = window.innerHeight;
    let css = computeMatrix3d(w, h, corners);
    if (css) {
      canvas.style.transformOrigin = '0 0';
      canvas.style.transform = css;
    }
  }

  function renderHandles() {
    let w = window.innerWidth;
    let h = window.innerHeight;
    let labels = ['TL', 'TR', 'BR', 'BL'];
    let html = '';

    // Draw quad outline
    html += `<polygon points="${corners.map(c => `${c.x * w},${c.y * h}`).join(' ')}"
      fill="none" stroke="rgba(0,170,255,0.3)" stroke-width="1" stroke-dasharray="6,4"/>`;

    // Draw crosshair at center
    let cx = corners.reduce((s, c) => s + c.x * w, 0) / 4;
    let cy = corners.reduce((s, c) => s + c.y * h, 0) / 4;
    html += `<line x1="${cx - 15}" y1="${cy}" x2="${cx + 15}" y2="${cy}" stroke="rgba(0,170,255,0.2)" stroke-width="1"/>`;
    html += `<line x1="${cx}" y1="${cy - 15}" x2="${cx}" y2="${cy + 15}" stroke="rgba(0,170,255,0.2)" stroke-width="1"/>`;

    // Draw corner handles
    for (let i = 0; i < 4; i++) {
      let px = corners[i].x * w;
      let py = corners[i].y * h;
      html += `<circle class="corner-handle" data-idx="${i}"
        cx="${px}" cy="${py}" r="${HANDLE_RADIUS}"
        fill="rgba(0,170,255,0.15)" stroke="#0af" stroke-width="2"/>`;
      html += `<text x="${px}" y="${py + 4}" text-anchor="middle"
        fill="#0af" font-family="Courier New" font-size="10"
        pointer-events="none">${labels[i]}</text>`;
    }
    svg.innerHTML = html;
  }

  function startCalibration() {
    calibrating = true;
    overlay.classList.add('active');
    status.classList.add('active');
    renderHandles();
    applyTransform();
  }

  function stopCalibration() {
    calibrating = false;
    overlay.classList.remove('active');
    status.classList.remove('active');
    saveCalibration();
    applyTransform();
  }

  function resetCalibration() {
    corners = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 }
    ];
    saveCalibration();
    applyTransform();
    if (calibrating) renderHandles();
  }

  // Mouse handling for corner dragging
  svg.addEventListener('mousedown', function (e) {
    let handle = e.target.closest('.corner-handle');
    if (!handle) return;
    activeCorner = parseInt(handle.dataset.idx);
    e.preventDefault();
  });

  window.addEventListener('mousemove', function (e) {
    if (activeCorner === null) return;
    let w = window.innerWidth;
    let h = window.innerHeight;
    corners[activeCorner].x = Math.max(0, Math.min(1, e.clientX / w));
    corners[activeCorner].y = Math.max(0, Math.min(1, e.clientY / h));
    renderHandles();
    applyTransform();
  });

  window.addEventListener('mouseup', function () {
    if (activeCorner !== null) {
      activeCorner = null;
      saveCalibration();
    }
  });

  // Touch handling for corner dragging
  svg.addEventListener('touchstart', function (e) {
    let handle = e.target.closest('.corner-handle');
    if (!handle) return;
    activeCorner = parseInt(handle.dataset.idx);
    e.preventDefault();
  }, { passive: false });

  window.addEventListener('touchmove', function (e) {
    if (activeCorner === null) return;
    let t = e.touches[0];
    let w = window.innerWidth;
    let h = window.innerHeight;
    corners[activeCorner].x = Math.max(0, Math.min(1, t.clientX / w));
    corners[activeCorner].y = Math.max(0, Math.min(1, t.clientY / h));
    renderHandles();
    applyTransform();
    e.preventDefault();
  }, { passive: false });

  window.addEventListener('touchend', function () {
    if (activeCorner !== null) {
      activeCorner = null;
      saveCalibration();
    }
  });

  // Fullscreen
  window.toggleFullscreen = function () {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => {
        hideUI();
        startCursorHideTimer();
      }).catch(() => {});
    } else {
      document.exitFullscreen();
    }
  };

  document.addEventListener('fullscreenchange', function () {
    if (!document.fullscreenElement) {
      showUI();
      document.body.classList.remove('hide-cursor');
      clearTimeout(cursorTimeout);
    }
  });

  // UI visibility
  function hideUI() {
    uiHidden = true;
    document.getElementById('info').classList.add('ui-hidden');
    document.getElementById('controls').classList.add('ui-hidden');
  }

  function showUI() {
    uiHidden = false;
    document.getElementById('info').classList.remove('ui-hidden');
    document.getElementById('controls').classList.remove('ui-hidden');
  }

  function toggleUI() {
    if (uiHidden) showUI(); else hideUI();
  }

  // Auto-hide cursor in fullscreen after inactivity
  function startCursorHideTimer() {
    document.body.classList.remove('hide-cursor');
    clearTimeout(cursorTimeout);
    cursorTimeout = setTimeout(() => {
      if (document.fullscreenElement) {
        document.body.classList.add('hide-cursor');
      }
    }, 3000);
  }

  document.addEventListener('mousemove', function () {
    if (document.fullscreenElement) {
      document.body.classList.remove('hide-cursor');
      startCursorHideTimer();
    }
  });

  // Projection mapping toggle
  window.toggleProjectionMapping = function () {
    if (calibrating) {
      stopCalibration();
    } else {
      startCalibration();
    }
  };

  // Keyboard shortcuts
  document.addEventListener('keydown', function (e) {
    if (e.key === 'f' || e.key === 'F') {
      if (!calibrating) window.toggleFullscreen();
    }
    if (e.key === 'p' || e.key === 'P') {
      window.toggleProjectionMapping();
    }
    if (e.key === 'r' || e.key === 'R') {
      if (calibrating) resetCalibration();
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      toggleUI();
    }
  });

  // Reapply transform on window resize
  window.addEventListener('resize', function () {
    applyTransform();
    if (calibrating) renderHandles();
  });

  // Apply saved calibration on load
  if (!isDefaultCorners()) {
    requestAnimationFrame(() => {
      requestAnimationFrame(applyTransform);
    });
  }
})();
