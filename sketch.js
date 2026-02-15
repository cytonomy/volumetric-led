// ─── Volumetric LED Display Simulator ────────────────────────────────
// Simulates outputting audio-reactive data to a voxel cloud display
// with realistic LED point rendering (bloom, individual dots, glow).

// ─── Config ──────────────────────────────────────────────────────────
let GRID = 16;             // voxels per axis (16^3 = 4096 LEDs)
let SPACING = 22;          // distance between LEDs in pixels
const ROTATE_SPEED = 0.003;

// ─── State ───────────────────────────────────────────────────────────
let voxels;                // Float32Array[GRID^3 * 4] → r,g,b,brightness
let fft, mic;
let audioReady = false;
let paused = false;
let patternMode = 1;       // 1-9
let colorScheme = 0;       // cycles through palettes
let camRotX = -0.4;
let camRotY = 0.5;
let camZoom = 1.0;
let dragging = false;
let lastMX, lastMY;
let spectrum = [];
let bass, mid, treble, energy;
let beatThreshold = 0;
let beatDecay = 0;
let beatHit = false;
let impulse = 0;           // fast-decaying spike for reactive patterns
let frameCount_ = 0;
let glContext;             // WebGL context reference
window.audioSensitivity = 1.0; // slider-controlled sensitivity multiplier

// ─── Lightning state — L-system continuous growth ────────────────────
// Growth tips are live agents that advance ~1 voxel per frame, forking
// frequently into dense dendritic trees. Hubs spawn dynamically with
// music. The whole system is always growing, branching, and fading.

let L_MAX_TIPS = 1200;         // hard cap — scales with grid size
const L_MAX_HUBS = 6;          // max dynamic hubs — fewer for cleaner structure
const L_BASE_HUBS = 2;         // starting hub count
const L_HUB_SPAWN_CHANCE = 0.6;// beat-triggered hub spawn probability
const L_BRANCH_BASE = 0.12;    // base branch chance per step
const L_BRANCH_DEPTH_MULT = 1.12; // branch prob multiplier per depth level
const L_MAX_DEPTH = 18;        // max branching depth for fine wisps
const L_TIP_SPEED_BASE = 0.08; // base voxels-per-frame advance (very slow crawl)
const L_TIP_SPEED_AUDIO = 0.5; // added speed from audio energy
const L_TIP_LIFE_BASE = 120;   // base tip lifetime in frames
const L_TIP_LIFE_VARIANCE = 80;// random lifetime variance
const L_POWER_DECAY = 0.965;   // power multiplied per step
const L_CHILD_POWER = 0.75;    // child power relative to parent at fork
const L_SPREAD_BASE = 0.5;     // base angular spread for direction jitter
const L_SPREAD_BASS = 0.8;     // additional spread from bass

let lightningDecay;            // Float32Array[GRID^3] — per-voxel structure (dendrites/axons)
let neuralSignal;              // Float32Array[GRID^3] — per-voxel signal brightness (action potentials)
let lightningHubs = [];        // somas (cell bodies) [{x,y,z,energy,age}, ...]
let lightningTips = [];        // live growth tips (dendrite/axon growth)
let signalPulses = [];         // active signal pulses traveling along paths
let lightningRng;              // seeded PRNG state
let lightningSpawnAccum = 0;   // smoothed fractional spawn accumulator
let lightningSpawnRate = 0;    // smoothed target spawn rate
let signalSpawnAccum = 0;      // accumulator for signal pulse spawning

// ─── Tree state (mode 6) ─────────────────────────────────────────────
let treeDecay;                 // Float32Array[GRID^3] — tree voxel brightness
let treeType;                  // Float32Array[GRID^3] — voxel type: 0=empty, 1=trunk, 2=branch, 3=leaf
let treeTrunks = [];           // persistent trunk positions [{x,z,height,age}, ...]
let treeTips = [];             // growing branch tips
let treeRng = 54321;
let treeSpawnAccum = 0;
let treeInitialized = false;

function tRand() {
  treeRng ^= treeRng << 13;
  treeRng ^= treeRng >> 17;
  treeRng ^= treeRng << 5;
  return ((treeRng >>> 0) % 10000) / 10000;
}

function initTrees() {
  treeDecay = new Float32Array(GRID * GRID * GRID);
  treeType = new Float32Array(GRID * GRID * GRID);
  treeTrunks = [];
  treeTips = [];
  treeRng = 54321;
  // Plant 3-6 trees on the floor (y = GRID-1 is bottom)
  let numTrees = 3 + floor(tRand() * 4);
  for (let i = 0; i < numTrees; i++) {
    treeTrunks.push({
      x: 2 + tRand() * (GRID - 4),
      z: 2 + tRand() * (GRID - 4),
      height: 0,      // current trunk height (grows upward)
      maxHeight: floor(GRID * 0.4 + tRand() * GRID * 0.3),
      age: 0,
      grown: false     // trunk fully grown?
    });
  }
  treeInitialized = true;
}

function updateTrees() {
  if (!treeInitialized || !treeDecay) initTrees();
  let GGt = GRID * GRID;

  // Type-aware decay: trunks barely fade; branches and leaves fade FASTER when quiet
  // This creates the "fade to just trunks" effect when music dies down
  let quietFactor = 1 - impulse; // 1 when silent, 0 when loud
  for (let i = 0; i < treeDecay.length; i++) {
    if (treeDecay[i] < 0.003) { treeDecay[i] = 0; treeType[i] = 0; continue; }
    let tp = treeType[i];
    if (tp >= 1 && tp < 1.5) {
      // Trunk — extremely persistent, almost no decay regardless of music
      treeDecay[i] *= 0.9995;
    } else if (tp >= 2 && tp < 2.5) {
      // Branch — ephemeral, decays MUCH faster when quiet
      // loud: 0.975, quiet: 0.92 (rapid fade)
      treeDecay[i] *= 0.975 - quietFactor * 0.055;
    } else if (tp >= 3) {
      // Leaf — fastest decay, nearly instant fade when quiet
      // loud: 0.965, quiet: 0.88 (very rapid fade)
      treeDecay[i] *= 0.965 - quietFactor * 0.085;
    } else {
      treeDecay[i] *= 0.98 - quietFactor * 0.04;
    }
  }

  // Grow trunks upward — very persistent, thick, clearly visible
  for (let t = 0; t < treeTrunks.length; t++) {
    let trunk = treeTrunks[t];
    trunk.age++;
    let tx = (trunk.x + 0.5) | 0;
    let tz = (trunk.z + 0.5) | 0;

    // Grow trunk upward (y goes from GRID-1 at floor to 0 at top)
    // Grows every 2 frames, or every frame on beats
    if (!trunk.grown && (trunk.age % 2 === 0 || beatHit)) {
      trunk.height = min(trunk.height + 1, trunk.maxHeight);
      if (trunk.height >= trunk.maxHeight) trunk.grown = true;
    }

    // Light trunk voxels — very bright, thick at base, tapers up
    for (let h = 0; h < trunk.height; h++) {
      let ty = GRID - 1 - h;
      if (ty < 0 || ty >= GRID) continue;
      // Strong base power — trunks are the anchor
      let basePower = 0.85 + energy * 0.1;
      let heightRatio = h / trunk.maxHeight;
      let taper = 1 - heightRatio * 0.4;
      // Thickness: full 3x3 at base, axis-cross in upper half, single at very top
      let thickLimit = heightRatio < 0.3 ? 2 : (heightRatio < 0.7 ? 1 : 0);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dz2 = -1; dz2 <= 1; dz2++) {
          let taxicab = abs(dx) + abs(dz2);
          if (taxicab > thickLimit) continue;
          let vx = tx + dx, vz = tz + dz2;
          if (vx < 0 || vx >= GRID || vz < 0 || vz >= GRID) continue;
          let vi = vz * GGt + ty * GRID + vx;
          let val = basePower * taper / (1 + (dx*dx + dz2*dz2) * 1.2);
          // Trunks override — always stay lit
          if (val > treeDecay[vi]) {
            treeDecay[vi] = val;
            treeType[vi] = 1; // mark as trunk
          }
        }
      }
    }

    // Spawn branch tips — start branching once trunk is 20% grown (earlier for more prolific growth)
    // Gated by impulse — no new branches when music is quiet
    if (trunk.height > trunk.maxHeight * 0.2 && impulse > 0.05) {
      let branchChance = 0.06 + energy * 0.18 + impulse * 0.15 + (beatHit ? bass * 0.5 : 0);
      // More branches once fully grown
      if (trunk.grown) branchChance *= 2.0;
      if (tRand() < branchChance && treeTips.length < 600) {
        let startH = floor(trunk.height * (0.3 + tRand() * 0.7));
        let sy = GRID - 1 - startH;
        let angle = tRand() * TWO_PI;
        let upBias = 0.3 + tRand() * 0.4;
        treeTips.push({
          x: trunk.x, y: sy, z: trunk.z,
          dx: cos(angle) * (1 - upBias), dy: -upBias, dz: sin(angle) * (1 - upBias),
          power: 0.35 + energy * 0.2 + impulse * 0.15 + (beatHit ? 0.2 : 0),
          depth: 0, life: 25 + floor(tRand() * 35),
          age: 0, isLeaf: false
        });
      }
    }
  }

  // Advance tree branch tips
  for (let i = treeTips.length - 1; i >= 0; i--) {
    let tip = treeTips[i];
    tip.age++;
    if (tip.age > tip.life || tip.power < 0.01) {
      treeTips[i] = treeTips[treeTips.length - 1];
      treeTips.length--;
      continue;
    }

    // Advance
    let jitter = 0.2 + tip.depth * 0.05;
    tip.dx += (tRand() - 0.5) * jitter;
    tip.dy += (tRand() - 0.5) * jitter * 0.5 - 0.02;
    tip.dz += (tRand() - 0.5) * jitter;
    let dl = sqrt(tip.dx*tip.dx + tip.dy*tip.dy + tip.dz*tip.dz) + 0.001;
    tip.dx /= dl; tip.dy /= dl; tip.dz /= dl;

    tip.x += tip.dx * 0.7;
    tip.y += tip.dy * 0.7;
    tip.z += tip.dz * 0.7;

    // Light voxel — branches thin, leaves are single bright points
    let ix = (tip.x + 0.5) | 0, iy = (tip.y + 0.5) | 0, iz = (tip.z + 0.5) | 0;
    if (ix >= 0 && ix < GRID && iy >= 0 && iy < GRID && iz >= 0 && iz < GRID) {
      let vi = iz * GGt + iy * GRID + ix;
      if (tip.isLeaf) {
        // Leaves: single voxel, distinctly bright pop
        let val = treeDecay[vi] + tip.power * 1.5;
        treeDecay[vi] = val < 1 ? val : 1;
        treeType[vi] = 3; // mark as leaf
      } else {
        // Branches: single voxel, moderate brightness
        let val = treeDecay[vi] + tip.power * 0.4;
        treeDecay[vi] = val < 1 ? val : 1;
        if (treeType[vi] < 1.5) treeType[vi] = 2; // mark as branch (don't overwrite trunk)
      }
    }

    // Branches fade faster than before — ephemeral
    tip.power *= tip.isLeaf ? 0.94 : 0.96;

    // Branch or spawn leaves — very generous forking for fuller canopy
    if (tip.depth < 8 && treeTips.length < 600) {
      let forkP = 0.15 + tip.depth * 0.05 + treble * 0.15 + impulse * 0.1;
      if (tRand() < forkP) {
        let a2 = tRand() * TWO_PI;
        let childUp = -0.2 - tRand() * 0.3;
        let spread = 0.6 + tip.depth * 0.1;
        let isLeaf = tip.depth >= 2; // leaves start earlier for more canopy
        treeTips.push({
          x: tip.x, y: tip.y, z: tip.z,
          dx: cos(a2) * spread, dy: childUp, dz: sin(a2) * spread,
          power: tip.power * (isLeaf ? 0.8 : 0.6),
          depth: tip.depth + 1,
          life: tip.life * (isLeaf ? 0.6 + tRand() * 0.3 : 0.4 + tRand() * 0.3),
          age: 0,
          isLeaf: isLeaf
        });
      }
    }
  }

  // Spawn new trees on beat — very generous
  if (beatHit && tRand() < 0.35 && treeTrunks.length < 12) {
    treeTrunks.push({
      x: 2 + tRand() * (GRID - 4),
      z: 2 + tRand() * (GRID - 4),
      height: 0,
      maxHeight: floor(GRID * 0.4 + tRand() * GRID * 0.3),
      age: 0, grown: false
    });
  }
}

// ─── Spline Web state (mode 7) ──────────────────────────────────────
let splineDecay;               // Float32Array[GRID^3] — per-voxel spline brightness
let splinePaths = [];          // [{points: [{x,y,z},...], age, life, brightness}, ...]
let splineInitialized = false;
let splineRng = 77777;

function sRand() {
  splineRng ^= splineRng << 13;
  splineRng ^= splineRng >> 17;
  splineRng ^= splineRng << 5;
  return ((splineRng >>> 0) % 10000) / 10000;
}

function initSplines() {
  splineDecay = new Float32Array(GRID * GRID * GRID);
  splinePaths = [];
  splineRng = 77777;
  // Pre-generate a tangled web of 3D spline curves
  for (let i = 0; i < 20; i++) {
    generateSplinePath();
  }
  splineInitialized = true;
}

function generateSplinePath() {
  // Create a smooth 3D spline as a series of points using random walk with inertia
  let pts = [];
  let x = 1 + sRand() * (GRID - 2);
  let y = 1 + sRand() * (GRID - 2);
  let z = 1 + sRand() * (GRID - 2);
  let dx = (sRand() - 0.5) * 0.8;
  let dy = (sRand() - 0.5) * 0.8;
  let dz = (sRand() - 0.5) * 0.8;
  let steps = 30 + floor(sRand() * 50);

  for (let s = 0; s < steps; s++) {
    pts.push({x: x, y: y, z: z});
    // Smooth direction changes — organic curves
    dx += (sRand() - 0.5) * 0.25;
    dy += (sRand() - 0.5) * 0.25;
    dz += (sRand() - 0.5) * 0.25;
    // Dampen to prevent sharp turns
    let dLen = Math.sqrt(dx*dx + dy*dy + dz*dz) + 0.001;
    let maxSpeed = 0.6;
    if (dLen > maxSpeed) { dx *= maxSpeed/dLen; dy *= maxSpeed/dLen; dz *= maxSpeed/dLen; }
    // Attract toward center to keep splines entangled
    let cx = GRID/2, cy = GRID/2, cz = GRID/2;
    dx += (cx - x) * 0.008;
    dy += (cy - y) * 0.008;
    dz += (cz - z) * 0.008;
    // Advance
    x += dx; y += dy; z += dz;
    // Soft bounce off walls
    if (x < 0.5) { x = 0.5; dx = Math.abs(dx) * 0.5; }
    if (x > GRID - 0.5) { x = GRID - 0.5; dx = -Math.abs(dx) * 0.5; }
    if (y < 0.5) { y = 0.5; dy = Math.abs(dy) * 0.5; }
    if (y > GRID - 0.5) { y = GRID - 0.5; dy = -Math.abs(dy) * 0.5; }
    if (z < 0.5) { z = 0.5; dz = Math.abs(dz) * 0.5; }
    if (z > GRID - 0.5) { z = GRID - 0.5; dz = -Math.abs(dz) * 0.5; }
  }

  splinePaths.push({
    points: pts,
    phase: sRand() * TWO_PI,        // unique phase offset for wave traveling
    speed: 0.5 + sRand() * 1.5,     // wave travel speed variation
    hue: sRand(),                     // unique color offset
    brightness: 0                     // current brightness (music-driven)
  });
}

function updateSplines() {
  if (!splineInitialized || !splineDecay) initSplines();
  let t = frameCount_ * 0.015;
  let GGs = GRID * GRID;

  // Decay existing glow — fade when music dies
  let splineDecayRate = 0.92 - impulse * 0.08; // fast fade when quiet, slow when loud
  for (let i = 0; i < splineDecay.length; i++) {
    splineDecay[i] *= splineDecayRate;
    if (splineDecay[i] < 0.01) splineDecay[i] = 0;
  }

  // Occasionally generate new spline paths on beats for variety
  if (beatHit && sRand() < 0.15 && splinePaths.length < 35) {
    generateSplinePath();
  }
  // Remove very old paths occasionally to cycle
  if (splinePaths.length > 25 && sRand() < 0.01) {
    splinePaths.shift();
  }

  // Light up splines with traveling waves driven by audio
  for (let p = 0; p < splinePaths.length; p++) {
    let path = splinePaths[p];
    let pts = path.points;

    // Each spline has a traveling wave of brightness
    // Wave position and width driven by audio
    let wavePos = (t * path.speed + path.phase) % pts.length;
    let waveWidth = 3 + impulse * 8 + energy * 4; // narrow when quiet, wide when loud
    let peakBright = impulse * 1.5 + energy * 0.5; // dim when quiet, bright when loud

    // Beat makes a bright flash propagate
    if (beatHit) peakBright += bass * 1.0;

    for (let s = 0; s < pts.length; s++) {
      let pt = pts[s];
      let ix = (pt.x + 0.5) | 0, iy = (pt.y + 0.5) | 0, iz = (pt.z + 0.5) | 0;
      if (ix < 0 || ix >= GRID || iy < 0 || iy >= GRID || iz < 0 || iz >= GRID) continue;

      // Distance along the spline from wave center (circular wrapping)
      let dist = Math.abs(s - wavePos);
      if (dist > pts.length / 2) dist = pts.length - dist;

      // Gaussian-like falloff from wave center
      let waveBright = peakBright * Math.exp(-dist * dist / (waveWidth * waveWidth + 0.1));

      // Also add a subtle base structure visibility when there's any energy
      waveBright += energy * 0.05;

      if (waveBright > 0.01) {
        let vi = iz * GGs + iy * GRID + ix;
        let v = splineDecay[vi] + waveBright;
        splineDecay[vi] = v < 1 ? v : 1;
      }
    }
  }
}

// ─── Fish state (mode 9) ────────────────────────────────────────────
let fishDecay;                 // Float32Array[GRID^3] — per-voxel fish brightness
let fishList = [];             // [{x,y,z, dx,dy,dz, phase, size, age}, ...]
let fishInitialized = false;
let fishRng = 99999;

function fRand() {
  fishRng ^= fishRng << 13;
  fishRng ^= fishRng >> 17;
  fishRng ^= fishRng << 5;
  return ((fishRng >>> 0) % 10000) / 10000;
}

function initFish() {
  fishDecay = new Float32Array(GRID * GRID * GRID);
  fishList = [];
  fishRng = 99999;
  // Spawn initial school of fish
  let numFish = 6 + floor(fRand() * 5);
  for (let i = 0; i < numFish; i++) {
    spawnFish();
  }
  fishInitialized = true;
}

function spawnFish() {
  let cx = GRID / 2, cy = GRID / 2, cz = GRID / 2;
  let angle = fRand() * TWO_PI;
  let elev = (fRand() - 0.5) * PI * 0.6;
  fishList.push({
    x: cx + (fRand() - 0.5) * GRID * 0.5,
    y: cy + (fRand() - 0.5) * GRID * 0.5,
    z: cz + (fRand() - 0.5) * GRID * 0.5,
    dx: Math.cos(angle) * Math.cos(elev),
    dy: Math.sin(elev) * 0.3,
    dz: Math.sin(angle) * Math.cos(elev),
    phase: fRand() * TWO_PI,      // unique tail wag phase
    size: 0.7 + fRand() * 0.6,    // size variation
    tailWag: 0,                     // current tail wag amount
    speed: 0.03 + fRand() * 0.04   // base movement speed
  });
}

function drawFishVoxel(fx, fy, fz, brightness, GGf) {
  let ix = (fx + 0.5) | 0, iy = (fy + 0.5) | 0, iz = (fz + 0.5) | 0;
  if (ix < 0 || ix >= GRID || iy < 0 || iy >= GRID || iz < 0 || iz >= GRID) return;
  let vi = iz * GGf + iy * GRID + ix;
  let v = fishDecay[vi] + brightness;
  fishDecay[vi] = v < 1 ? v : 1;
}

function updateFish() {
  if (!fishInitialized || !fishDecay) initFish();
  let t = frameCount_ * 0.015;
  let GGf = GRID * GRID;
  let cx = GRID / 2, cy = GRID / 2, cz = GRID / 2;

  // Decay — fish trails fade
  let fishFade = 0.88 - impulse * 0.06;
  for (let i = 0; i < fishDecay.length; i++) {
    fishDecay[i] *= fishFade;
    if (fishDecay[i] < 0.01) fishDecay[i] = 0;
  }

  // Spawn more fish on beats
  if (beatHit && fRand() < 0.2 && fishList.length < 20) {
    spawnFish();
  }

  // Update each fish
  for (let f = 0; f < fishList.length; f++) {
    let fish = fishList[f];

    // Audio-reactive speed — faster with music
    let moveSpeed = fish.speed * (0.3 + impulse * 2.0 + energy * 0.8);

    // Tail wag frequency driven by music
    fish.tailWag = Math.sin(t * (3 + impulse * 8) + fish.phase) * (0.3 + impulse * 0.7);

    // Schooling behavior — steer toward center of mass and align with neighbors
    let avgDx = 0, avgDy = 0, avgDz = 0;
    let sepX = 0, sepY = 0, sepZ = 0;
    for (let o = 0; o < fishList.length; o++) {
      if (o === f) continue;
      let other = fishList[o];
      let ddx = other.x - fish.x, ddy = other.y - fish.y, ddz = other.z - fish.z;
      let dist2 = ddx*ddx + ddy*ddy + ddz*ddz + 0.01;
      // Alignment
      avgDx += other.dx; avgDy += other.dy; avgDz += other.dz;
      // Separation — push away from close neighbors
      if (dist2 < 9) {
        sepX -= ddx / dist2;
        sepY -= ddy / dist2;
        sepZ -= ddz / dist2;
      }
    }
    let n = fishList.length - 1;
    if (n > 0) { avgDx /= n; avgDy /= n; avgDz /= n; }

    // Steer toward center to keep school together
    let toCenterX = (cx - fish.x) * 0.003;
    let toCenterY = (cy - fish.y) * 0.003;
    let toCenterZ = (cz - fish.z) * 0.003;

    // Swirl around Y axis — creates the swirling motion
    let swirlStrength = 0.02 + impulse * 0.06;
    let swirlX = -fish.dz * swirlStrength;
    let swirlZ = fish.dx * swirlStrength;

    // Combine forces
    fish.dx += avgDx * 0.01 + sepX * 0.05 + toCenterX + swirlX;
    fish.dy += avgDy * 0.01 + sepY * 0.05 + toCenterY;
    fish.dz += avgDz * 0.01 + sepZ * 0.05 + toCenterZ + swirlZ;

    // Keep vertical movement gentle
    fish.dy *= 0.95;

    // Normalize direction
    let dLen = Math.sqrt(fish.dx*fish.dx + fish.dy*fish.dy + fish.dz*fish.dz) + 0.001;
    fish.dx /= dLen; fish.dy /= dLen; fish.dz /= dLen;

    // Move
    fish.x += fish.dx * moveSpeed;
    fish.y += fish.dy * moveSpeed;
    fish.z += fish.dz * moveSpeed;

    // Soft bounds
    if (fish.x < 1) { fish.x = 1; fish.dx = Math.abs(fish.dx); }
    if (fish.x > GRID-1) { fish.x = GRID-1; fish.dx = -Math.abs(fish.dx); }
    if (fish.y < 1) { fish.y = 1; fish.dy = Math.abs(fish.dy); }
    if (fish.y > GRID-1) { fish.y = GRID-1; fish.dy = -Math.abs(fish.dy); }
    if (fish.z < 1) { fish.z = 1; fish.dz = Math.abs(fish.dz); }
    if (fish.z > GRID-1) { fish.z = GRID-1; fish.dz = -Math.abs(fish.dz); }

    // Draw the fish shape — elongated diamond/torpedo with tail fin
    // Fish body along its direction vector
    let sz = fish.size * (GRID / 16); // scale with grid size
    let bodyLen = sz * 2.5;
    let bodyWidth = sz * 0.6;
    let fishBright = 0.3 + impulse * 0.7 + energy * 0.3;
    if (beatHit) fishBright += bass * 0.5;

    // Build a local coordinate frame: forward (dx,dy,dz), right, up
    let fwdX = fish.dx, fwdY = fish.dy, fwdZ = fish.dz;
    // Right vector (cross forward with world up)
    let rightX = fwdZ, rightY = 0, rightZ = -fwdX;
    let rLen = Math.sqrt(rightX*rightX + rightZ*rightZ) + 0.001;
    rightX /= rLen; rightZ /= rLen;
    // Up vector (cross right with forward)
    let upX = fwdY * rightZ - fwdZ * rightY;
    let upY = fwdZ * rightX - fwdX * rightZ;
    let upZ = fwdX * rightY - fwdY * rightX;

    // Draw body voxels along the fish shape
    let segments = floor(bodyLen * 1.5) + 3;
    for (let s = 0; s <= segments; s++) {
      let frac = s / segments; // 0 = nose, 1 = tail
      let along = (frac - 0.35) * bodyLen; // offset so body center is at fish position

      // Body cross-section radius: tapers at nose and tail, widest at 30%
      let profileFrac = frac;
      let radius;
      if (profileFrac < 0.3) {
        radius = bodyWidth * (profileFrac / 0.3); // nose taper
      } else if (profileFrac < 0.6) {
        radius = bodyWidth; // widest section
      } else {
        radius = bodyWidth * (1 - (profileFrac - 0.6) / 0.4) * 0.7; // tail taper
      }

      // Tail fin flare
      if (frac > 0.8) {
        let tailFrac = (frac - 0.8) / 0.2;
        let wagOffset = fish.tailWag * tailFrac * sz * 0.8;
        // Tail wags side to side
        let tx = fish.x + fwdX * along + rightX * wagOffset;
        let ty = fish.y + fwdY * along + rightY * wagOffset;
        let tz = fish.z + fwdZ * along + rightZ * wagOffset;
        // Tail fin is wider vertically
        let tailHeight = sz * 0.5 * tailFrac;
        drawFishVoxel(tx, ty, tz, fishBright * (0.6 + tailFrac * 0.3), GGf);
        drawFishVoxel(tx, ty + tailHeight, tz, fishBright * 0.4, GGf);
        drawFishVoxel(tx, ty - tailHeight, tz, fishBright * 0.4, GGf);
      }

      // Body cross section
      let bx = fish.x + fwdX * along;
      let by = fish.y + fwdY * along;
      let bz = fish.z + fwdZ * along;

      // Body brightness: brightest at center, dimmer at edges
      let segBright = fishBright * (0.5 + 0.5 * (1 - Math.abs(frac - 0.35) * 2));
      drawFishVoxel(bx, by, bz, segBright, GGf); // center

      if (radius > 0.3) {
        // Side voxels
        drawFishVoxel(bx + rightX * radius, by, bz + rightZ * radius, segBright * 0.6, GGf);
        drawFishVoxel(bx - rightX * radius, by, bz - rightZ * radius, segBright * 0.6, GGf);
        // Top/bottom voxels
        drawFishVoxel(bx + upX * radius * 0.5, by + upY * radius * 0.5, bz + upZ * radius * 0.5, segBright * 0.5, GGf);
        drawFishVoxel(bx - upX * radius * 0.5, by - upY * radius * 0.5, bz - upZ * radius * 0.5, segBright * 0.5, GGf);
      }
    }

    // Dorsal fin (top of fish, around 30-50% body)
    for (let s = 0; s < 3; s++) {
      let frac = 0.25 + s * 0.08;
      let along = (frac - 0.35) * bodyLen;
      let finHeight = sz * 0.4 * (1 - s * 0.2);
      let fx2 = fish.x + fwdX * along + upX * finHeight;
      let fy2 = fish.y + fwdY * along + upY * finHeight - finHeight * 0.5;
      let fz2 = fish.z + fwdZ * along + upZ * finHeight;
      drawFishVoxel(fx2, fy2, fz2, fishBright * 0.35, GGf);
    }
  }
}

// Color palettes: each is a function(t, brightness) → [r,g,b]
const palettes = [
  // Cyan / magenta
  (t, b) => {
    let r = sin(t * 2.0) * 0.5 + 0.5;
    let g = sin(t * 3.0 + 1.0) * 0.3 + 0.2;
    let bl = sin(t * 1.5 + 2.5) * 0.5 + 0.5;
    return [r * b * 255, g * b * 255, bl * b * 255];
  },
  // Fire
  (t, b) => {
    return [b * 255, b * t * 180, b * t * t * 60];
  },
  // Ocean
  (t, b) => {
    return [b * t * 40, b * 140, b * 255];
  },
  // Rainbow
  (t, b) => {
    let h = (t * 360) % 360;
    let c = hslToRgb(h, 0.9, b * 0.5);
    return c;
  }
];

function hslToRgb(h, s, l) {
  h /= 360;
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    let q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    let p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [r * 255, g * 255, b * 255];
}

// ─── p5 Setup ────────────────────────────────────────────────────────
function setup() {
  createCanvas(windowWidth, windowHeight, WEBGL);
  pixelDensity(1);

  voxels = new Float32Array(GRID * GRID * GRID * 4);
  fft = new p5.FFT(0.85, 256);

  // Grab GL context from p5's renderer
  glContext = drawingContext;

  spectrum = new Array(256).fill(0);
  bass = mid = treble = energy = 0;

  // Neural/Lightning init
  lightningDecay = new Float32Array(GRID * GRID * GRID);
  neuralSignal = new Float32Array(GRID * GRID * GRID);
  lightningRng = 12345;
  initLightningHubs();
}

// ─── Grid resize ──────────────────────────────────────────────────────
// Called from HTML slider — reallocates buffers and resets lightning state
window.resizeGrid = function(newGrid) {
  newGrid = constrain(floor(newGrid), 4, 32);
  if (newGrid === GRID) return;
  GRID = newGrid;
  // Auto-adjust spacing so the total cube size stays roughly constant
  SPACING = floor(350 / GRID);
  // Scale tip cap with grid volume — fewer tips on large grids for perf
  L_MAX_TIPS = min(2000, max(400, floor(GRID * GRID * GRID * 0.04)));
  // Reallocate buffers
  voxels = new Float32Array(GRID * GRID * GRID * 4);
  lightningDecay = new Float32Array(GRID * GRID * GRID);
  neuralSignal = new Float32Array(GRID * GRID * GRID);
  // Reset lightning — tips and hubs reference old grid coords
  lightningTips = [];
  signalPulses = [];
  treeInitialized = false;
  splineInitialized = false;
  fishInitialized = false;
  lightningSpawnAccum = 0;
  lightningSpawnRate = 0;
  lightningRng = 12345;
  initLightningHubs();
};

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

// ─── Audio ───────────────────────────────────────────────────────────
function startAudio() {
  if (audioReady) return;
  userStartAudio().then(() => {
    mic = new p5.AudioIn();
    mic.start();
    fft.setInput(mic);
    audioReady = true;
    document.querySelector('#controls button').textContent = 'Audio Active';
    document.querySelector('#controls button').style.borderColor = '#0f0';
    document.querySelector('#controls button').style.color = '#0f0';
  });
}

// ─── Analyze audio ───────────────────────────────────────────────────
function analyzeAudio() {
  if (!audioReady) {
    // Synthetic demo data — punchier beats, more dynamic
    let t = frameCount_ * 0.02;
    // Kick drum pulse — sharp transient every ~1.5s
    let kickPhase = (t * 2.8) % TWO_PI;
    let kick = kickPhase < 0.4 ? Math.exp(-kickPhase * 8) : 0;
    // Snare/mid hit offset from kick
    let snarePhase = ((t * 2.8) + PI) % TWO_PI;
    let snare = snarePhase < 0.3 ? Math.exp(-snarePhase * 10) * 0.7 : 0;
    for (let i = 0; i < 256; i++) {
      let freq = i / 256;
      // Base texture
      spectrum[i] = (sin(t * (1 + freq * 3)) * 0.5 + 0.5) *
                    (1 - freq * 0.6) *
                    (sin(t * 0.7 + freq * 10) * 0.3 + 0.7) * 140;
      // Strong kick in bass bins
      if (i < 25) {
        spectrum[i] += kick * 220;
      }
      // Snare in mid bins
      if (i > 30 && i < 90) {
        spectrum[i] += snare * 160;
      }
      // Treble sparkle on hits
      if (i > 100) {
        spectrum[i] += (kick * 0.3 + snare * 0.5) * 100 * (1 - freq * 0.5);
      }
    }
  } else {
    spectrum = fft.analyze();
  }

  bass = 0; mid = 0; treble = 0;
  for (let i = 0; i < 256; i++) {
    let v = spectrum[i] / 255;
    if (i < 20) bass += v;
    else if (i < 80) mid += v;
    else treble += v;
  }
  bass /= 20;
  mid /= 60;
  treble /= 176;

  // Apply sensitivity multiplier from slider
  let sens = window.audioSensitivity;
  bass = min(1, bass * sens);
  mid = min(1, mid * sens);
  treble = min(1, treble * sens);
  energy = (bass * 3 + mid * 2 + treble) / 6;

  beatHit = false;
  // Bass beat detection — primary trigger
  if (bass > beatThreshold + 0.12 && beatDecay <= 0) {
    beatHit = true;
    beatDecay = 6;
  }
  // Mid-range hit detection — secondary trigger for more frequent small bursts
  if (mid > 0.4 && beatDecay <= 0 && !beatHit) {
    beatHit = true;
    beatDecay = 10;
  }
  beatThreshold = lerp(beatThreshold, bass, 0.06);
  if (beatDecay > 0) beatDecay--;

  // Smoothed impulse — rises fast toward audio energy, decays gently
  // Gives responsive but non-jumpy gating for modes 1/3/4
  let targetImpulse = energy * energy * 2.0;
  if (beatHit) targetImpulse = min(1, 0.5 + bass * 0.6);
  // Fast rise, moderate fall — stays lit a bit after transients
  if (targetImpulse > impulse) {
    impulse = lerp(impulse, targetImpulse, 0.5);  // fast attack
  } else {
    impulse = lerp(impulse, targetImpulse, 0.1);  // smooth decay
  }
}

// ─── Lightning system — L-system continuous growth ──────────────────
// Instead of pre-computed bolts, the system uses live "growth tips"
// that advance ~1 voxel per frame, fork using L-system rules driven
// by audio, and organically fade/die. Hubs spawn dynamically.

function lRand() {
  lightningRng ^= lightningRng << 13;
  lightningRng ^= lightningRng >> 17;
  lightningRng ^= lightningRng << 5;
  return ((lightningRng >>> 0) % 10000) / 10000;
}

function initLightningHubs() {
  lightningHubs = [];
  for (let i = 0; i < L_BASE_HUBS; i++) {
    lightningHubs.push({
      x: 2 + lRand() * (GRID - 4),
      y: 2 + lRand() * (GRID - 4),
      z: 2 + lRand() * (GRID - 4),
      energy: 0.5 + lRand() * 0.5,
      age: 0
    });
  }
}

function driftLightningHubs(t) {
  for (let i = 0; i < lightningHubs.length; i++) {
    let h = lightningHubs[i];
    h.x = constrain(h.x + sin(t * 0.3 + i * 2.1) * 0.05, 1, GRID - 2);
    h.y = constrain(h.y + cos(t * 0.25 + i * 1.7) * 0.05, 1, GRID - 2);
    h.z = constrain(h.z + sin(t * 0.35 + i * 3.3) * 0.05, 1, GRID - 2);
    h.age++;
  }
}

// Create a new growth tip from a hub
function spawnTipFromHub(hub, power) {
  // Random direction biased away from grid center for far-reaching growth
  let cx = GRID / 2, cy = GRID / 2, cz = GRID / 2;
  let awayX = hub.x - cx, awayY = hub.y - cy, awayZ = hub.z - cz;
  let awayLen = sqrt(awayX * awayX + awayY * awayY + awayZ * awayZ) + 0.001;
  // Mix: 40% away from center + 60% random for variety
  let dx = (awayX / awayLen) * 0.4 + (lRand() - 0.5) * 1.2;
  let dy = (awayY / awayLen) * 0.4 + (lRand() - 0.5) * 1.2;
  let dz = (awayZ / awayLen) * 0.4 + (lRand() - 0.5) * 1.2;
  let dLen = sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
  dx /= dLen; dy /= dLen; dz /= dLen;

  return {
    x: hub.x, y: hub.y, z: hub.z,
    dx: dx, dy: dy, dz: dz,
    power: power,
    depth: 0,
    life: L_TIP_LIFE_BASE + lRand() * L_TIP_LIFE_VARIANCE,
    age: 0,
    hubIdx: lightningHubs.indexOf(hub),
    stepAccum: 0  // fractional step accumulator
  };
}

// Light voxels at continuous position — single voxel per step for clean paths
// Power scales with depth: trunks bright, tips dimmer
function lightVoxel(px, py, pz, power, depth) {
  let ix = (px + 0.5) | 0, iy = (py + 0.5) | 0, iz = (pz + 0.5) | 0;
  if (ix < 0 || ix >= GRID || iy < 0 || iy >= GRID || iz < 0 || iz >= GRID) return;
  let vi = iz * GRID * GRID + iy * GRID + ix;
  // All depths: single voxel only — keeps paths clean and distinct
  let pwr = depth > 4 ? power * 0.7 : power;
  let v = lightningDecay[vi] + pwr;
  lightningDecay[vi] = v < 1 ? v : 1;
}

// Fork a tip into a child branch
function forkTip(parent) {
  // Perpendicular-ish deviation from parent direction
  // Create orthogonal basis from parent dir
  let px = parent.dx, py = parent.dy, pz = parent.dz;
  // Find an arbitrary perpendicular
  let ax, ay, az;
  if (abs(px) < 0.7) { ax = 1; ay = 0; az = 0; }
  else { ax = 0; ay = 1; az = 0; }
  // Cross product: parent × arbitrary = perp1
  let p1x = py * az - pz * ay;
  let p1y = pz * ax - px * az;
  let p1z = px * ay - py * ax;
  let p1Len = sqrt(p1x * p1x + p1y * p1y + p1z * p1z) + 0.001;
  p1x /= p1Len; p1y /= p1Len; p1z /= p1Len;
  // Cross product: parent × perp1 = perp2
  let p2x = py * p1z - pz * p1y;
  let p2y = pz * p1x - px * p1z;
  let p2z = px * p1y - py * p1x;

  // Angular spread increases with audio and depth
  let spread = L_SPREAD_BASE + bass * L_SPREAD_BASS + parent.depth * 0.04;
  let angle = (lRand() - 0.5) * spread * 2;
  let tilt = (lRand() - 0.5) * spread * 2;

  // New direction: parent + angular offsets
  let fwdWeight = 0.5 + lRand() * 0.3; // keep some forward momentum
  let ndx = px * fwdWeight + p1x * sin(angle) + p2x * sin(tilt);
  let ndy = py * fwdWeight + p1y * sin(angle) + p2y * sin(tilt);
  let ndz = pz * fwdWeight + p1z * sin(angle) + p2z * sin(tilt);
  let nLen = sqrt(ndx * ndx + ndy * ndy + ndz * ndz) + 0.001;
  ndx /= nLen; ndy /= nLen; ndz /= nLen;

  let childPower = parent.power * L_CHILD_POWER;
  let childLife = parent.life * (0.5 + lRand() * 0.4);

  return {
    x: parent.x, y: parent.y, z: parent.z,
    dx: ndx, dy: ndy, dz: ndz,
    power: childPower,
    depth: parent.depth + 1,
    life: childLife,
    age: 0,
    hubIdx: parent.hubIdx,
    stepAccum: 0
  };
}

function updateLightning() {
  let t = frameCount_ * 0.015;
  driftLightningHubs(t);

  // ── Decay existing glow ──────────────────────────────────────────
  // Very slow fade — tendrils persist long, high cutoff for clean distinct paths
  let decayRate = 0.997 - energy * 0.008;
  for (let i = 0; i < lightningDecay.length; i++) {
    lightningDecay[i] *= decayRate;
    if (lightningDecay[i] < 0.06) lightningDecay[i] = 0; // high cutoff — only real paths survive
  }

  // ── Hub glow — single voxel per soma, no neighbor bleed ──────────
  let hubGlow = 0.12 + impulse * 0.25;
  if (beatHit) hubGlow = min(1, 0.4 + bass * 0.4);
  for (let h = 0; h < lightningHubs.length; h++) {
    let hub = lightningHubs[h];
    let hx = round(hub.x), hy = round(hub.y), hz = round(hub.z);
    if (hx >= 0 && hx < GRID && hy >= 0 && hy < GRID && hz >= 0 && hz < GRID) {
      let vi = hz * GRID * GRID + hy * GRID + hx;
      lightningDecay[vi] = max(lightningDecay[vi], hubGlow);
    }
  }

  // ── Advance all growth tips ──────────────────────────────────────
  // Speed is heavily music-dependent: near-zero when quiet, picks up with energy
  let speed = L_TIP_SPEED_BASE + impulse * L_TIP_SPEED_AUDIO + (beatHit ? bass * 0.4 : 0);
  let newTips = [];

  // Iterate tips — use swap-and-pop instead of splice for O(1) removal
  let tipLen = lightningTips.length;
  for (let i = tipLen - 1; i >= 0; i--) {
    let tip = lightningTips[i];
    tip.age++;

    // Kill dead tips (swap with last, pop)
    if (tip.age > tip.life || tip.power < 0.015) {
      lightningTips[i] = lightningTips[--tipLen];
      lightningTips.length = tipLen;
      continue;
    }

    // Kill tips that wander too far outside grid
    if (tip.x < -1 || tip.x > GRID || tip.y < -1 || tip.y > GRID || tip.z < -1 || tip.z > GRID) {
      lightningTips[i] = lightningTips[--tipLen];
      lightningTips.length = tipLen;
      continue;
    }

    // Accumulate fractional steps
    tip.stepAccum += speed;

    // Take integer steps this frame
    while (tip.stepAccum >= 1.0 && tip.age <= tip.life && tip.power > 0.015) {
      tip.stepAccum -= 1.0;

      // Organic direction jitter — more wobble at deeper depths
      let jitter = 0.15 + tip.depth * 0.02 + energy * 0.1;
      tip.dx += (lRand() - 0.5) * jitter;
      tip.dy += (lRand() - 0.5) * jitter;
      tip.dz += (lRand() - 0.5) * jitter;
      // Re-normalize direction
      let dLen = sqrt(tip.dx * tip.dx + tip.dy * tip.dy + tip.dz * tip.dz) + 0.001;
      tip.dx /= dLen; tip.dy /= dLen; tip.dz /= dLen;

      // Advance position
      tip.x += tip.dx;
      tip.y += tip.dy;
      tip.z += tip.dz;

      // Light current voxel — leading edge is brightest, depth controls thickness
      let ageFrac = tip.age / tip.life;
      let edgePower = tip.power * (1.2 + bass * 0.8 + (beatHit ? 0.6 : 0));
      edgePower *= (1 - ageFrac * 0.3); // fade as tip ages
      lightVoxel(tip.x, tip.y, tip.z, edgePower, tip.depth);

      // Power decay per step
      tip.power *= L_POWER_DECAY;

      // ── L-system branching ─────────────────────────────────────
      if (tip.depth < L_MAX_DEPTH && lightningTips.length + newTips.length < L_MAX_TIPS) {
        // Branch probability: increases with depth (more fractal branching at tips)
        // Audio: treble = intricate branching, mid = fork frequency, bass = spread
        let branchP = L_BRANCH_BASE * pow(L_BRANCH_DEPTH_MULT, tip.depth);
        branchP *= (1 + treble * 1.2 + mid * 0.6);
        // Slightly suppress branching in first few steps for clean trunks
        if (tip.age < 3) branchP *= 0.3;
        // Beat boost
        if (beatHit) branchP *= 1.8;

        // Can fork 1-3 times per step at high probability
        let forks = 0;
        if (lRand() < branchP) forks++;
        if (lRand() < branchP * 0.35) forks++;
        if (lRand() < branchP * 0.1 && tip.depth < L_MAX_DEPTH - 3) forks++;

        for (let f = 0; f < forks; f++) {
          if (lightningTips.length + newTips.length < L_MAX_TIPS) {
            newTips.push(forkTip(tip));
          }
        }
      }
    }
  }

  // Add newly forked tips
  for (let t2 = 0; t2 < newTips.length; t2++) {
    lightningTips.push(newTips[t2]);
  }

  // ── Dynamic hub management ────────────────────────────────────────
  // Spawn new hubs on beats
  if (beatHit && lightningHubs.length < L_MAX_HUBS && lRand() < L_HUB_SPAWN_CHANCE) {
    lightningHubs.push({
      x: 1 + lRand() * (GRID - 2),
      y: 1 + lRand() * (GRID - 2),
      z: 1 + lRand() * (GRID - 2),
      energy: 0.6 + bass * 0.4,
      age: 0
    });
  }

  // Remove old hubs that have been quiet (keep at least L_BASE_HUBS)
  if (lightningHubs.length > L_BASE_HUBS) {
    for (let h = lightningHubs.length - 1; h >= L_BASE_HUBS; h--) {
      if (lightningHubs[h].age > 600 + lRand() * 400) {
        lightningHubs.splice(h, 1);
      }
    }
  }

  // Relocate a hub on beat for variety
  if (beatHit && lRand() < 0.3) {
    let ri = floor(lRand() * lightningHubs.length);
    lightningHubs[ri].x = 1 + lRand() * (GRID - 2);
    lightningHubs[ri].y = 1 + lRand() * (GRID - 2);
    lightningHubs[ri].z = 1 + lRand() * (GRID - 2);
    lightningHubs[ri].age = 0;
  }

  // ── Signal pulses (action potentials) ─────────────────────────────
  // Bright packets that travel outward from somas along existing structure.
  // They follow voxels with high lightningDecay values (the dendrite paths).
  let sigDecay = 0.82 - energy * 0.05;
  for (let i = 0; i < neuralSignal.length; i++) {
    neuralSignal[i] *= sigDecay;
    if (neuralSignal[i] < 0.005) neuralSignal[i] = 0;
  }

  let GGsig = GRID * GRID;
  // Advance signal pulses — each pulse moves along structure, speed tied to music
  // Pulses use a step accumulator just like tips — only move when music drives them
  let pulseSpeed = 0.2 + impulse * 0.8 + (beatHit ? bass * 0.5 : 0);
  for (let si = signalPulses.length - 1; si >= 0; si--) {
    let pulse = signalPulses[si];
    pulse.age++;
    if (pulse.age > pulse.life || pulse.power < 0.02) {
      signalPulses[si] = signalPulses[signalPulses.length - 1];
      signalPulses.length--;
      continue;
    }

    // Accumulate fractional steps — music controls pulse travel speed
    pulse.stepAccum = (pulse.stepAccum || 0) + pulseSpeed;
    if (pulse.stepAccum < 1.0) continue; // don't move yet
    pulse.stepAccum -= 1.0;

    // Light current voxel brightly — single voxel only for clean signal paths
    let pix = (pulse.x + 0.5) | 0, piy = (pulse.y + 0.5) | 0, piz = (pulse.z + 0.5) | 0;
    if (pix >= 0 && pix < GRID && piy >= 0 && piy < GRID && piz >= 0 && piz < GRID) {
      let vi = piz * GGsig + piy * GRID + pix;
      let sv = neuralSignal[vi] + pulse.power;
      neuralSignal[vi] = sv < 1 ? sv : 1;
    }

    // Move pulse — follow highest-decay neighbor (path following)
    let bestVal = -1, bestDx = 0, bestDy = 0, bestDz = 0;
    let tries = 0;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          let cx = pix + dx, cy = piy + dy, cz = piz + dz;
          if (cx < 0 || cx >= GRID || cy < 0 || cy >= GRID || cz < 0 || cz >= GRID) continue;
          let cvi = cz * GGsig + cy * GRID + cx;
          // Prefer structure (lightningDecay) and avoid backtracking
          let val = lightningDecay[cvi] * 0.8 - neuralSignal[cvi] * 2.0;
          // Forward bias — prefer continuing in current direction
          val += (dx * pulse.dx + dy * pulse.dy + dz * pulse.dz) * 0.3;
          val += lRand() * 0.15; // randomness for natural feel
          if (val > bestVal) { bestVal = val; bestDx = dx; bestDy = dy; bestDz = dz; tries++; }
        }
      }
    }

    if (tries > 0 && bestVal > -0.5) {
      pulse.x += bestDx;
      pulse.y += bestDy;
      pulse.z += bestDz;
      // Update direction for forward-bias
      let dl = sqrt(bestDx*bestDx + bestDy*bestDy + bestDz*bestDz) + 0.001;
      pulse.dx = bestDx/dl; pulse.dy = bestDy/dl; pulse.dz = bestDz/dl;
    } else {
      // No good path — die
      pulse.power = 0;
    }
    pulse.power *= 0.96;
  }

  // Spawn signal pulses from somas — strongly music reactive
  // Only fire signals when there's real energy — quiet = no signals
  let sigTarget = impulse * 3.0 + (beatHit ? bass * 6 : 0);
  signalSpawnAccum += sigTarget * 0.25;
  let sigToSpawn = floor(signalSpawnAccum);
  signalSpawnAccum -= sigToSpawn;
  for (let s = 0; s < sigToSpawn && signalPulses.length < 200; s++) {
    let hi = floor(lRand() * lightningHubs.length);
    let hub = lightningHubs[hi];
    let dx = lRand() - 0.5, dy = lRand() - 0.5, dz = lRand() - 0.5;
    let dl = sqrt(dx*dx+dy*dy+dz*dz) + 0.001;
    signalPulses.push({
      x: hub.x, y: hub.y, z: hub.z,
      dx: dx/dl, dy: dy/dl, dz: dz/dl,
      power: 0.4 + bass * 0.6 + (beatHit ? 0.4 : 0),
      life: 60 + floor(lRand() * 60),
      age: 0,
      stepAccum: 0
    });
  }

  // ── Smoothed tip spawning ──────────────────────────────────────────
  // Target spawn rate smoothly tracks audio. Beats push it up, it
  // decays smoothly between beats — no sudden jumps.
  let targetRate = 0.05 + energy * 0.8 + bass * 0.5 + impulse * 1.5;
  // Multiply by number of hubs for more activity with more hubs
  targetRate *= lightningHubs.length * 0.4;
  // Smooth toward target — fast rise, slower fall for organic feel
  let smoothUp = 0.3, smoothDown = 0.08;
  if (targetRate > lightningSpawnRate) {
    lightningSpawnRate = lerp(lightningSpawnRate, targetRate, smoothUp);
  } else {
    lightningSpawnRate = lerp(lightningSpawnRate, targetRate, smoothDown);
  }

  // Accumulate and spawn fractional tips across frames
  lightningSpawnAccum += lightningSpawnRate;
  let toSpawn = floor(lightningSpawnAccum);
  lightningSpawnAccum -= toSpawn;

  for (let s = 0; s < toSpawn; s++) {
    if (lightningTips.length >= L_MAX_TIPS) break;
    // Pick a random hub
    let hi = floor(lRand() * lightningHubs.length);
    let hub = lightningHubs[hi];
    // Power scales smoothly with audio
    let power = 0.3 + energy * 0.3 + impulse * 0.4;
    lightningTips.push(spawnTipFromHub(hub, power));
  }
}

// ─── Pattern generators ──────────────────────────────────────────────
function generatePattern() {
  let t = frameCount_ * 0.015;
  let half = GRID / 2;
  let invHalf = 1 / half;
  let palette = palettes[colorScheme % palettes.length];
  let GG = GRID * GRID;

  // Pre-update simulations
  if (patternMode === 5) updateLightning();
  if (patternMode === 6) updateTrees();
  if (patternMode === 7) updateSplines();
  if (patternMode === 9) updateFish();

  for (let z = 0; z < GRID; z++) {
    let nz = (z - half + 0.5) * invHalf;
    let nzz = nz * nz;
    for (let y = 0; y < GRID; y++) {
      let ny = (y - half + 0.5) * invHalf;
      let nyz = ny * ny + nzz;
      for (let x = 0; x < GRID; x++) {
        let idx = (z * GG + y * GRID + x) * 4;
        let nx = (x - half + 0.5) * invHalf;
        let r = sqrt(nx*nx + nyz);
        let brightness = 0;
        let colorT = 0;

        switch (patternMode) {
          case 1: { // Radial pulse — tight shell, controlled size
            // Shell radius is small & proportional to impulse — max ~0.6 radius
            let shellR = impulse * 0.6;
            let shell = Math.exp(-abs(r - shellR) * 18) * impulse;
            let freqIdx = floor(((ny + 1) / 2) * 128);
            let freqVal = spectrum[constrain(freqIdx, 0, 255)] / 255;
            // Tight impulse gating — fades fast
            brightness = shell * 1.2 + freqVal * impulse * 0.5;
            // Beat flash — small concentrated burst
            if (beatHit) brightness += Math.exp(-r * 5) * 0.8 * bass;
            // Tiny ambient ripple
            brightness += sin(r * 10 - t * 6) * 0.015 * energy;
            colorT = r * 0.5 + ny * 0.3 + t * 0.1;
            break;
          }
          case 2: { // Frequency columns — grow upward from floor, smoothed
            let binX = floor(((nx + 1) / 2) * 15);
            let binZ = floor(((nz + 1) / 2) * 15);
            let bin = (binX + binZ * 4) % 256;
            let freqVal = spectrum[bin] / 255;
            // Smooth amplification — less aggressive to reduce jumpiness
            let ampFreq = freqVal * (0.8 + energy * 1.8 + impulse * 1.0);
            ampFreq = ampFreq > 1 ? 1 : ampFreq;
            // Column grows upward from bottom (ny=1 is bottom, ny=-1 is top)
            let colHeight = ampFreq * 1.8 - 0.8;
            // ny goes from -1 (top) to 1 (bottom); columns grow up from bottom
            if (ny > -colHeight) {
              let distFromTop = ny + colHeight; // distance below column top
              // Softer gradient — less aggressive falloff
              brightness = (0.3 + ampFreq * 0.6) * Math.exp(-distFromTop * 0.7);
              // Gentle floor glow
              brightness = max(brightness, ampFreq * energy * 0.25);
              // Smooth impulse modulation — not as spiky
              brightness *= (0.5 + impulse * 0.8);
            }
            colorT = ampFreq * 0.8 + ny * 0.1;
            // Gentler beat flash
            if (beatHit && ny > -colHeight) brightness += 0.2 * bass;
            break;
          }
          case 3: { // Plasma / organic — impulse-gated, fades hard
            let p = sin(nx * 3 + t * 2) * cos(nz * 3 - t) +
                    sin(ny * 4 + t * 1.5 + bass * 6) +
                    cos(r * 5 - t * 3 + mid * 5);
            p = p / 3 * 0.5 + 0.5;
            // Gated by impulse — snaps on, fades fast
            brightness = p * impulse * 2.5;
            if (treble > 0.2) {
              let sparkle = sin(nx * 50 + t * 20) * sin(ny * 50 - t * 15) * sin(nz * 50 + t * 10);
              if (sparkle > 0.6) brightness += treble * impulse * 1.5;
            }
            colorT = p + t * 0.05;
            if (beatHit) brightness += (1 - r) * 0.8 * bass;
            break;
          }
          case 4: { // Helix — radius grows with music, shrinks when quiet
            let angle = atan2(nx, nz);
            let helixSpeed = t * (2 + impulse * 6);
            let helixTwist = 6 + bass * 4;
            let helixA = sin(angle * 2 + ny * helixTwist + helixSpeed);
            let helixB = sin(angle * 2 + ny * helixTwist + helixSpeed + PI);
            let rXZ = sqrt(nx*nx + nz*nz);
            // Column radius grows with impulse — wide when loud, tiny when quiet
            let helixR = 0.05 + impulse * 0.55; // 0.05 at rest → 0.6 at max
            let tubeWidth = 0.04 + impulse * 0.08; // tube thickness also scales
            // Distance from each helix strand
            let strandRA = helixR + helixA * tubeWidth;
            let strandRB = helixR + helixB * tubeWidth;
            let dA = abs(rXZ - strandRA);
            let dB = abs(rXZ - strandRB);
            let shellA = Math.exp(-dA * (18 + impulse * 10));
            let shellB = Math.exp(-dB * (18 + impulse * 10));
            // Impulse gates everything — dark when quiet
            brightness = (shellA + shellB) * impulse * 2.0;
            colorT = (angle / TWO_PI + 0.5) + ny * 0.2;
            if (beatHit) brightness += Math.exp(-r * 3) * 0.5 * bass;
            break;
          }
          case 5: { // Neural network — somas, dendrites, action potentials
            let vi = z * GG + y * GRID + x;
            let structure = lightningDecay[vi]; // persistent dendrite paths
            let signal = neuralSignal[vi];       // bright traveling signals
            // Only show voxels on real paths — very strict for clean spline-like lines
            if (structure < 0.1 && signal < 0.04) { voxels[idx+3] = 0; continue; }
            // Structure brightness breathes with music — dim when quiet, glows with energy
            let structBright = structure * (0.2 + impulse * 0.6 + energy * 0.35);
            let combined = structBright + signal;
            if (combined < 0.03) { voxels[idx+3] = 0; continue; }
            brightness = combined;
            // Nearest soma for color mapping
            let minD2 = 9999;
            for (let h = 0; h < lightningHubs.length; h++) {
              let dhx = x - lightningHubs[h].x;
              let dhy = y - lightningHubs[h].y;
              let dhz = z - lightningHubs[h].z;
              let d2 = dhx*dhx + dhy*dhy + dhz*dhz;
              if (d2 < minD2) minD2 = d2;
            }
            let hubProx = 1 / (1 + minD2 * 0.02);
            // Structure is amber-ish (colorT ~0.1-0.3), signals push toward blue/white (higher colorT)
            colorT = hubProx * 0.3 + signal * 0.4 + t * 0.01;
            break;
          }
          case 6: { // Growing trees — persistent trunks, ephemeral branches, bright leaves
            let vi = z * GG + y * GRID + x;
            let rawVal = treeDecay ? treeDecay[vi] : 0;
            if (rawVal < 0.005) { voxels[idx+3] = 0; continue; }
            let tp = treeType ? treeType[vi] : 0;
            if (tp >= 1 && tp < 1.5) {
              // Trunk — bright, warm, dominant — always visible anchor
              // Slight music breathing but stays visible even when quiet
              brightness = rawVal * (0.9 + impulse * 0.3);
              colorT = 0.08 + (1 - y / GRID) * 0.12; // warm brown range
            } else if (tp >= 3) {
              // Leaf — distinctly bright pop, music-boosted
              brightness = rawVal * (1.0 + impulse * 0.8 + energy * 0.3);
              colorT = 0.35 + rawVal * 0.1 + t * 0.01; // green/bright range
            } else {
              // Branch — thinner, dimmer, music-reactive
              brightness = rawVal * (0.4 + impulse * 0.5);
              colorT = 0.15 + (1 - y / GRID) * 0.15; // between trunk and leaf
            }
            if (beatHit) brightness += rawVal * bass * 0.3;
            break;
          }
          case 7: { // Spline Web — tangled curves that light up with music
            let vi = z * GG + y * GRID + x;
            let rawVal = splineDecay ? splineDecay[vi] : 0;
            if (rawVal < 0.02) { voxels[idx+3] = 0; continue; }
            // Spline brightness breathes with music
            brightness = rawVal * (0.3 + impulse * 1.2 + energy * 0.5);
            if (beatHit) brightness += rawVal * bass * 0.5;
            // Color varies with position and time for depth
            colorT = (x + y + z) / (GRID * 3) + rawVal * 0.2 + t * 0.02;
            break;
          }
          case 8: { // Pulsing torus / ring — geometric, reactive
            let rXZ = sqrt(nx * nx + nz * nz);
            let angle = atan2(nx, nz);
            // Torus major radius pulses with bass
            let majorR = 0.5 + bass * 0.15;
            // Distance from torus tube center
            let tubeR = sqrt((rXZ - majorR) * (rXZ - majorR) + ny * ny);
            // Tube radius pulses with impulse
            let minorR = 0.12 + impulse * 0.15;
            let torusDist = abs(tubeR - minorR);
            let torusBright = Math.exp(-torusDist * 25) * impulse * 2.0;
            // Ripples around the ring driven by frequency
            let freqIdx = floor(((angle / TWO_PI + 0.5) % 1) * 128);
            let fv = spectrum[constrain(freqIdx, 0, 255)] / 255;
            torusBright += fv * impulse * 0.4 * Math.exp(-torusDist * 15);
            // Second ring perpendicular (XY plane)
            let rXY = sqrt(nx * nx + ny * ny);
            let tubeR2 = sqrt((rXY - majorR) * (rXY - majorR) + nz * nz);
            let torus2 = Math.exp(-abs(tubeR2 - minorR) * 25) * impulse * 1.2;
            brightness = max(torusBright, torus2);
            colorT = (angle / TWO_PI + 0.5) + tubeR * 0.3;
            if (beatHit) brightness += Math.exp(-tubeR * 8) * 0.5 * bass;
            break;
          }
          case 9: { // Fish school — swirling 3D fish models
            let vi = z * GG + y * GRID + x;
            let rawVal = fishDecay ? fishDecay[vi] : 0;
            if (rawVal < 0.02) { voxels[idx+3] = 0; continue; }
            // Fish brightness pulses with music
            brightness = rawVal * (0.4 + impulse * 0.8 + energy * 0.3);
            if (beatHit) brightness += rawVal * bass * 0.4;
            // Color: aquatic palette — mix of blues and silvers with position variation
            let fishHue = rawVal * 0.3 + (y / GRID) * 0.15 + t * 0.01;
            colorT = fishHue;
            break;
          }
        }

        // Clamp and gamma — avoid constrain/pow overhead
        if (brightness < 0) brightness = 0;
        else if (brightness > 1) brightness = 1;
        brightness = brightness * (0.2 + brightness * 0.8); // cheap approx of pow(b, 0.8)

        let ct = colorT < 0 ? -colorT : colorT;
        ct = ct - ((ct | 0)); // fast fmod 1
        let col = palette(ct, brightness);

        voxels[idx]     = col[0];
        voxels[idx + 1] = col[1];
        voxels[idx + 2] = col[2];
        voxels[idx + 3] = brightness;
      }
    }
  }
}

// ─── Rendering ───────────────────────────────────────────────────────
function draw() {
  if (!paused) frameCount_++;

  background(5, 5, 8);
  analyzeAudio();
  if (!paused) generatePattern();

  // Camera positioning using p5's camera()
  let totalSize = GRID * SPACING;
  let camDist = totalSize * 1.3 / camZoom;
  let rotY = camRotY + (paused ? 0 : frameCount_ * ROTATE_SPEED);

  // Auto-rotate
  if (!dragging && !paused) {
    camRotY += ROTATE_SPEED * 0.3;
  }

  let camX = camDist * sin(rotY) * cos(camRotX);
  let camY = camDist * sin(camRotX);
  let camZ = camDist * cos(rotY) * cos(camRotX);
  camera(camX, camY, camZ, 0, 0, 0, 0, 1, 0);

  // Disable depth test for additive blending to work properly
  let gl = glContext;
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive blending

  let half = GRID / 2;
  let totalVoxels = GRID * GRID * GRID;
  // Adaptive thresholds — skip more aggressively on large grids
  let bloomThresh = totalVoxels > 8000 ? 0.15 : 0.1;
  let coreThresh = totalVoxels > 8000 ? 0.04 : 0.02;
  // Skip bloom pass entirely on very large grids for performance
  let skipBloom = totalVoxels > 20000;
  // Skip hot-white center on large grids
  let skipHotCenter = totalVoxels > 10000;

  // Single merged render pass — bloom halo + core dot + hot center per voxel
  noFill();
  for (let i = 0; i < totalVoxels; i++) {
    let idx = i * 4;
    let b = voxels[idx + 3];
    if (b < coreThresh) continue;

    let r = voxels[idx];
    let g = voxels[idx + 1];
    let bl = voxels[idx + 2];

    // Compute xyz from flat index
    let z = (i / (GRID * GRID)) | 0;
    let rem = i - z * GRID * GRID;
    let y = (rem / GRID) | 0;
    let x = rem - y * GRID;
    let px = (x - half + 0.5) * SPACING;
    let py = (y - half + 0.5) * SPACING;
    let pz = (z - half + 0.5) * SPACING;

    // Bloom halo (skip on large grids or dim voxels)
    if (!skipBloom && b >= bloomThresh) {
      strokeWeight(b * 18 + 4);
      stroke(r, g, bl, b * 35);
      point(px, py, pz);
    }

    // Core bright dot
    let coreSize = 2 + b * 4;
    let alpha = 60 + b * 195;
    strokeWeight(coreSize);
    stroke(r, g, bl, alpha);
    point(px, py, pz);

    // Hot white center for bright LEDs (skip on large grids)
    if (!skipHotCenter && b > 0.5) {
      let wb = (b - 0.5) * 2;
      strokeWeight(coreSize * 0.4);
      stroke(
        r + (255 - r) * wb * 0.7,
        g + (255 - g) * wb * 0.7,
        bl + (255 - bl) * wb * 0.7,
        b * 180
      );
      point(px, py, pz);
    }
  }

  // Restore blend mode and depth test
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.enable(gl.DEPTH_TEST);

  // Draw faint enclosure wireframe
  push();
  noFill();
  stroke(25, 25, 30);
  strokeWeight(0.8);
  box(totalSize, totalSize, totalSize);
  pop();

  // ─── 2D HUD overlay ───────────────────────────────────────────────
  // Switch to screen-space 2D
  push();
  resetMatrix();
  let halfW = width / 2;
  let halfH = height / 2;
  ortho(-halfW, halfW, -halfH, halfH, -1, 1);

  noStroke();
  gl.disable(gl.DEPTH_TEST);

  // Spectrum bars at bottom right
  let barW = 2;
  let barMaxH = 60;
  let ox = halfW - 256 * barW - 20;
  let oy = halfH - 20;

  for (let i = 0; i < 256; i++) {
    let v = spectrum[i] / 255;
    let h = v * barMaxH;
    let palette = palettes[colorScheme % palettes.length];
    let c = palette(i / 256, v);
    fill(c[0], c[1], c[2], 140);
    rect(ox + i * barW, oy - h, barW - 1, h);
  }

  // Labels
  fill(80);
  textSize(10);
  textFont('monospace');
  text('FFT SPECTRUM', ox, oy + 14);
  text('BASS: ' + nf(bass, 1, 2), ox, oy + 26);
  text('MID: ' + nf(mid, 1, 2), ox + 90, oy + 26);
  text('TRE: ' + nf(treble, 1, 2), ox + 170, oy + 26);

  // Mode / color labels — prominent, top-left
  let modeNames = ['', 'RADIAL PULSE', 'FREQ COLUMNS', 'PLASMA', 'HELIX', 'NEURONS', 'FOREST', 'SPLINE WEB', 'TORUS', 'FISH'];
  let palNames = ['CYAN/MAG', 'FIRE', 'OCEAN', 'RAINBOW'];
  let lx = -halfW + 12;

  // Large mode label
  fill(120);
  textSize(16);
  text(patternMode + ' \u2022 ' + modeNames[patternMode], lx, -halfH + 24);

  // Color scheme label
  fill(80);
  textSize(12);
  text('COLOR: ' + palNames[colorScheme % palNames.length], lx, -halfH + 44);

  // Small info at bottom left
  let ly = halfH - 16;
  fill(45);
  textSize(10);
  text('FPS: ' + floor(frameRate()), lx, ly);
  text('VOXELS: ' + GRID + '\u00B3 = ' + (GRID*GRID*GRID), lx, ly - 14);

  // Audio status (top right)
  if (audioReady) {
    fill(0, 200, 0);
    text('\u25CF MIC LIVE', halfW - 100, -halfH + 20);
  } else {
    fill(90);
    text('DEMO MODE', halfW - 100, -halfH + 20);
  }

  gl.enable(gl.DEPTH_TEST);
  pop();
}

// ─── Interaction ─────────────────────────────────────────────────────
function mousePressed() {
  if (mouseButton === LEFT) {
    dragging = true;
    lastMX = mouseX;
    lastMY = mouseY;
  }
}

function mouseDragged() {
  if (dragging) {
    let dx = mouseX - lastMX;
    let dy = mouseY - lastMY;
    camRotY -= dx * 0.005;
    camRotX -= dy * 0.005;
    camRotX = constrain(camRotX, -HALF_PI * 0.9, HALF_PI * 0.9);
    lastMX = mouseX;
    lastMY = mouseY;
  }
}

function mouseReleased() {
  dragging = false;
}

function mouseWheel(event) {
  camZoom *= event.delta > 0 ? 0.93 : 1.07;
  camZoom = constrain(camZoom, 0.3, 3.0);
  return false;
}

function keyPressed() {
  if (key === '1') patternMode = 1;
  if (key === '2') patternMode = 2;
  if (key === '3') patternMode = 3;
  if (key === '4') patternMode = 4;
  if (key === '5') patternMode = 5;
  if (key === '6') patternMode = 6;
  if (key === '7') patternMode = 7;
  if (key === '8') patternMode = 8;
  if (key === '9') patternMode = 9;
  if (key === ' ') { paused = !paused; return false; }
  if (key === 'c' || key === 'C') colorScheme = (colorScheme + 1) % palettes.length;
}
