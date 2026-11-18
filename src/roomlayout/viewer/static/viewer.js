// Room layout viewer: draws scene.json (written by roomlayout.viewer) with three.js.
// Our frame: meters, Z-up, yaw CCW, asset front = local +Y.  three.js: Y-up.
// Mapping: (x, y, z)_ours -> (x, z, -y)_three; yaw -> rotation.y; local front +Y -> -Z.
// GLBs from roomlayout.assets face glTF +Z, so they get an extra 180-degree turn (loadModel).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const $ = (id) => document.getElementById(id);
const v3 = (x, y, z = 0) => new THREE.Vector3(x, z, -y);
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

const state = {
  scene: null, layouts: [], current: 0, poses: {}, objects: {}, walls: [],
  selected: null, locked: new Set(), served: false, edited: false, tweens: [],
};
window.__viewer = state;

// ------------------------------------------------------------------ renderer
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.85;
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.appendChild(renderer.domElement);

const scene3 = new THREE.Scene();
const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
scene3.background = new THREE.Color(dark ? 0x1d1b19 : 0xf4f1ec);
const pmrem = new THREE.PMREMGenerator(renderer);
scene3.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;

const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
state.camera = camera; // exposed for browser tests
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;

scene3.add(new THREE.HemisphereLight(0xfff6ea, 0x8a7f72, 0.35));
const sun = new THREE.DirectionalLight(0xfff2df, 2.4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
sun.shadow.radius = 4;
scene3.add(sun, sun.target);

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

// ------------------------------------------------------------------ materials
function woodTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 1024;
  const g = c.getContext("2d");
  const rows = 8, rnd = mulberry(7);
  for (let r = 0; r < rows; r++) {
    let x = -rnd() * 600;
    while (x < 1024) {
      const len = 380 + rnd() * 420, tone = 150 + rnd() * 34;
      g.fillStyle = `rgb(${tone + 18}, ${tone - 22}, ${tone - 70})`;
      g.fillRect(x, r * 128, len, 128);
      g.globalAlpha = 0.08;
      for (let k = 0; k < 14; k++) {
        g.fillStyle = rnd() > 0.5 ? "#5a3b1c" : "#fff4e0";
        g.fillRect(x, r * 128 + rnd() * 128, len, 1 + rnd() * 3);
      }
      g.globalAlpha = 1;
      g.fillStyle = "rgba(60,40,20,0.35)";
      g.fillRect(x, r * 128, 2, 128);
      x += len;
    }
    g.fillStyle = "rgba(60,40,20,0.30)";
    g.fillRect(0, r * 128, 1024, 2);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(1 / 2, 1 / 2); // one texture = 2 m x 2 m (8 planks of 0.25 m)
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}
function mulberry(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const COLOR_WORDS = {
  white: 0xeeeae3, gray: 0x8d8f91, grey: 0x8d8f91, black: 0x2c2b2a, blue: 0x46607e,
  green: 0x6d8a6a, beige: 0xcdbb9e, brown: 0x7a5a3f, oak: 0xc29b6c, ash: 0xd9c4a0,
  walnut: 0x6b4a33, pine: 0xd7b98a, birch: 0xe0cfaa, red: 0x9c3f36, yellow: 0xd9b44a,
};
const CATEGORY_COLOR = {
  sofa: 0x8a8f96, chair: 0xb08a63, table: 0xc49a6c, bed: 0xe9e4dc, storage: 0xd8cfc0,
  television: 0x222222, refrigerator: 0xdedede, oven: 0x3a3a3a, stove: 0x3a3a3a,
  fireplace: 0x9d8f80, sink: 0xe8e8e8, toilet: 0xf2f2f2, bathtub: 0xf2f2f2, dishwasher: 0xd6d6d6,
};
function baseColor(a) {
  const words = (a.label || "").toLowerCase();
  for (const [w, c] of Object.entries(COLOR_WORDS)) if (words.includes(w)) return c;
  return CATEGORY_COLOR[a.category] ?? 0xbdb3a6;
}
const fabric = (c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.95, metalness: 0 });
const solid = (c, r = 0.55) => new THREE.MeshStandardMaterial({ color: c, roughness: r, metalness: 0 });

// ------------------------------------------------------------------ furniture proxies
// Built in three.js local space: x = width, y = up, front = -z, origin = bottom centre.
function mesh(geo, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  return m;
}
const rbox = (w, h, d, r = 0.03) => new RoundedBoxGeometry(Math.max(w, 0.01), Math.max(h, 0.01),
  Math.max(d, 0.01), 3, Math.min(r, w / 2.2, h / 2.2, d / 2.2));
const box = (w, h, d) => new THREE.BoxGeometry(Math.max(w, 0.005), Math.max(h, 0.005), Math.max(d, 0.005));

function legs(g, w, d, h, mat, s = 0.045, inset = 0.04) {
  for (const sx of [-1, 1]) for (const sz of [-1, 1])
    g.add(mesh(box(s, h, s), mat, sx * (w / 2 - inset - s / 2), h / 2, sz * (d / 2 - inset - s / 2)));
}

function sofa(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group(), m = fabric(c);
  const seatH = Math.min(0.45, h * 0.55), backT = Math.min(0.22, d * 0.25), armW = Math.min(0.16, w * 0.08);
  g.add(mesh(rbox(w, seatH - 0.08, d, 0.04), m, 0, (seatH - 0.08) / 2 + 0.08, 0));
  g.add(mesh(rbox(w, h - 0.08, backT, 0.05), m, 0, (h - 0.08) / 2 + 0.08, d / 2 - backT / 2));
  for (const s of [-1, 1]) g.add(mesh(rbox(armW, h * 0.75 - 0.08, d, 0.04), m, s * (w / 2 - armW / 2), (h * 0.75 - 0.08) / 2 + 0.08, 0));
  const n = Math.max(1, Math.round((w - 2 * armW) / 0.75)), cw = (w - 2 * armW) / n;
  for (let i = 0; i < n; i++) g.add(mesh(rbox(cw - 0.02, 0.12, d - backT - 0.02, 0.05), m,
    -w / 2 + armW + cw * (i + 0.5), seatH + 0.04, -backT / 2));
  legs(g, w, d, 0.08, solid(0x3b2f26), 0.04, 0.06);
  return g;
}

function sectional(a, c) {
  // Footprint is an L in our local frame; recover seat depth and chaise side from it.
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group(), m = fabric(c);
  const ys = [...new Set(a.footprint.map((p) => +p[1].toFixed(4)))].sort((p, q) => p - q);
  const ySeat = ys.find((y) => y > -d / 2 + 1e-3 && y < d / 2 - 1e-3) ?? d / 2;
  const seatDepth = ySeat + d / 2;
  const front = a.footprint.filter((p) => Math.abs(p[1] - d / 2) < 1e-3).map((p) => p[0]);
  const cx0 = Math.min(...front), cx1 = Math.max(...front);
  const shape = new THREE.Shape(a.footprint.map(([x, y]) => new THREE.Vector2(x, y)));
  const seatH = Math.min(0.42, h * 0.5);
  const base = new THREE.ExtrudeGeometry(shape, { depth: seatH - 0.08, bevelEnabled: true,
    bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 2 });
  base.rotateX(-Math.PI / 2);
  g.add(mesh(base, m, 0, 0.08, 0));
  const backT = Math.min(0.22, seatDepth * 0.25);
  g.add(mesh(rbox(w, h - 0.08, backT, 0.05), m, 0, (h - 0.08) / 2 + 0.08, d / 2 - backT / 2));
  // arm on the end without the chaise
  const armW = 0.15, chaiseLeft = cx0 < 0 && cx1 < w / 2 - 1e-3; // chaise reaches the front at -x (ours)
  const armX = chaiseLeft ? w / 2 - armW / 2 : -w / 2 + armW / 2;
  g.add(mesh(rbox(armW, h * 0.72 - 0.08, seatDepth, 0.04), m, armX, (h * 0.72 - 0.08) / 2 + 0.08, d / 2 - seatDepth / 2));
  // chaise cushion + seat cushions
  const cw = cx1 - cx0;
  g.add(mesh(rbox(cw - 0.04, 0.12, d - backT - 0.04, 0.05), m, (cx0 + cx1) / 2, seatH + 0.04, -backT / 2 + 0.0));
  const restX0 = chaiseLeft ? cx1 : -w / 2 + armW, restX1 = chaiseLeft ? w / 2 - armW : cx0;
  const n = Math.max(1, Math.round((restX1 - restX0) / 0.75)), sw = (restX1 - restX0) / n;
  for (let i = 0; i < n; i++) g.add(mesh(rbox(sw - 0.02, 0.12, seatDepth - backT - 0.02, 0.05), m,
    restX0 + sw * (i + 0.5), seatH + 0.04, d / 2 - backT - (seatDepth - backT) / 2));
  return g;
}

function bed(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group();
  const frame = solid(c, 0.5), linen = fabric(0xf3efe8), frameH = 0.3;
  g.add(mesh(rbox(w, frameH - 0.05, d, 0.02), frame, 0, (frameH - 0.05) / 2 + 0.05, 0));
  g.add(mesh(rbox(w - 0.08, 0.22, d - 0.12, 0.06), linen, 0, frameH + 0.11, -0.03));
  g.add(mesh(rbox(w, h, 0.06, 0.02), frame, 0, h / 2, d / 2 - 0.03));
  g.add(mesh(rbox(w - 0.1, 0.06, d * 0.55, 0.03), fabric(0xb9c3c9), 0, frameH + 0.24, -d * 0.2));
  for (const s of [-1, 1]) g.add(mesh(rbox(w * 0.38, 0.12, 0.4, 0.06), linen, s * w * 0.22, frameH + 0.28, d / 2 - 0.32));
  legs(g, w, d, 0.05, frame, 0.05, 0.03);
  return g;
}

function table(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group(), m = solid(c, 0.5);
  const top = 0.035;
  g.add(mesh(rbox(w, top, d, 0.01), m, 0, h - top / 2, 0));
  legs(g, w, d, h - top, m, 0.045, 0.05);
  return g;
}

function chair(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group(), m = solid(c, 0.5);
  const seat = Math.min(0.46, h * 0.57);
  g.add(mesh(rbox(w, 0.04, d * 0.92, 0.01), m, 0, seat - 0.02, -d * 0.04));
  legs(g, w, d * 0.92, seat - 0.04, m, 0.035, 0.03);
  g.add(mesh(rbox(w, h - seat, 0.03, 0.01), m, 0, seat + (h - seat) / 2, d / 2 - 0.05));
  return g;
}

function shelf(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group(), m = solid(c, 0.6), t = 0.018;
  g.add(mesh(box(t, h, d), m, -w / 2 + t / 2, h / 2, 0), mesh(box(t, h, d), m, w / 2 - t / 2, h / 2, 0));
  g.add(mesh(box(w, t, d), m, 0, h - t / 2, 0), mesh(box(w, 0.06, d), m, 0, 0.03, 0));
  g.add(mesh(box(w, h, 0.006), solid(new THREE.Color(c).multiplyScalar(0.92)), 0, h / 2, d / 2 - 0.003));
  const n = Math.max(1, Math.floor(h / 0.36));
  for (let i = 1; i < n; i++) g.add(mesh(box(w - 2 * t, t, d - 0.01), m, 0, (h / n) * i, -0.005));
  return g;
}

function cabinet(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group();
  g.add(mesh(rbox(w, h, d, 0.01), solid(c, 0.6), 0, h / 2, 0));
  g.add(mesh(box(0.004, h * 0.9, 0.004), solid(0x777777), 0, h / 2, -d / 2 - 0.002));
  return g;
}

function tv(a) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group();
  const panelH = Math.min(h, w * 0.6);
  g.add(mesh(rbox(w, panelH, Math.min(0.06, d), 0.01), new THREE.MeshStandardMaterial({
    color: 0x111111, roughness: 0.25, metalness: 0.2 }), 0, h - panelH / 2, 0));
  if (a.mount !== "wall" && h - panelH > 0.05) g.add(mesh(box(w * 0.3, h - panelH, Math.min(0.25, d)), solid(0x333333), 0, (h - panelH) / 2, 0));
  return g;
}

function generic(a, c) {
  const { width: w, depth: d, height: h } = a.dims, g = new THREE.Group();
  g.add(mesh(rbox(w, h, d, 0.02), solid(c, 0.7), 0, h / 2, 0));
  return g;
}

function buildProxy(a) {
  const c = baseColor(a), sub = a.subtype || "";
  if (a.category === "sofa") return a.l_shaped ? sectional(a, c) : sofa(a, c);
  if (a.category === "bed") return bed(a, c);
  if (a.category === "chair") return chair(a, c);
  if (a.category === "table") return table(a, c);
  if (a.category === "television") return tv(a);
  if (a.category === "storage") return ["bookshelf", "shelf"].includes(sub) || /book|shel/.test(a.label) ? shelf(a, c) : cabinet(a, c);
  return generic(a, c);
}

// ------------------------------------------------------------------ room
function buildRoom(room) {
  const group = new THREE.Group();
  const shape = new THREE.Shape(room.floor.map(([x, y]) => new THREE.Vector2(x, y)));
  const floorGeo = new THREE.ShapeGeometry(shape);
  floorGeo.rotateX(-Math.PI / 2);
  const floor = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({ map: woodTexture(), roughness: 0.62 }));
  floor.receiveShadow = true;
  floor.name = "floor";
  group.add(floor);

  const wallColor = dark ? 0xd9d2c7 : 0xf1ece4;
  const byWall = Object.groupBy ? Object.groupBy(room.openings, (o) => o.wall_id)
    : room.openings.reduce((acc, o) => ((acc[o.wall_id] ||= []).push(o), acc), {});
  for (const w of room.walls) {
    const [sx, sy] = w.start, [ex, ey] = w.end, L = Math.hypot(ex - sx, ey - sy);
    if (L < 1e-3) continue;
    const dx = (ex - sx) / L, dy = (ey - sy) / L, t = w.thickness || 0.1, H = w.height;
    // basis: U along the wall, V up, W outward (right of the direction; interior is left)
    const U = v3(dx, dy), V = new THREE.Vector3(0, 1, 0), W = v3(dy, -dx);
    const basis = new THREE.Matrix4().makeBasis(U, V, W).setPosition(v3(sx, sy));
    const mat = new THREE.MeshStandardMaterial({ color: wallColor, roughness: 0.92 });
    const wg = new THREE.Group();
    wg.applyMatrix4(basis);
    const piece = (u0, u1, z0, z1) => {
      if (u1 - u0 < 1e-3 || z1 - z0 < 1e-3) return;
      wg.add(mesh(box(u1 - u0, z1 - z0, t), mat, (u0 + u1) / 2, (z0 + z1) / 2, t / 2));
    };
    const ops = (byWall[w.id] || []).slice().sort((p, q) => p.offset - q.offset);
    let u = 0;
    for (const o of ops) {
      const u0 = Math.max(0, o.offset - o.width / 2), u1 = Math.min(L, o.offset + o.width / 2);
      piece(u, u0, 0, H);
      const bottom = o.type === "window" ? o.sill : 0, top = Math.min(H, bottom + o.height);
      piece(u0, u1, 0, bottom);
      piece(u0, u1, top, H);
      if (o.type === "window") windowPane(wg, u0, u1, bottom, top, t);
      if (o.type === "door") doorFrame(wg, group, o, u0, u1, top, t, w);
      u = Math.max(u, u1);
    }
    piece(u, L, 0, H);
    // cap: a dark line on top so the wall section reads in the dollhouse view
    wg.add(mesh(box(L, 0.01, t), solid(0x5a524a, 0.8), L / 2, H + 0.005, t / 2));
    group.add(wg);
    state.walls.push({ group: wg, mat, normal: W, center: v3(sx + dx * L / 2, sy + dy * L / 2, H / 2) });
  }
  for (const ob of room.obstacles) {
    const s = new THREE.Shape(ob.footprint.map(([x, y]) => new THREE.Vector2(x, y)));
    const geo = new THREE.ExtrudeGeometry(s, { depth: ob.height, bevelEnabled: false });
    geo.rotateX(-Math.PI / 2);
    const m = new THREE.Mesh(geo, solid(0xbfb6aa, 0.85));
    m.castShadow = m.receiveShadow = true;
    group.add(m);
  }
  return group;
}

function windowPane(wg, u0, u1, z0, z1, t) {
  const glass = new THREE.MeshPhysicalMaterial({ color: 0xbfd8e6, roughness: 0.05, transmission: 0.6,
    transparent: true, opacity: 0.35, metalness: 0 });
  const pane = mesh(new THREE.PlaneGeometry(u1 - u0, z1 - z0), glass, (u0 + u1) / 2, (z0 + z1) / 2, t / 2);
  pane.castShadow = false;
  wg.add(pane);
  const fm = solid(0xfafafa, 0.4), f = 0.04;
  wg.add(mesh(box(u1 - u0, f, t * 0.6), fm, (u0 + u1) / 2, z0 + f / 2, t / 2));
  wg.add(mesh(box(u1 - u0, f, t * 0.6), fm, (u0 + u1) / 2, z1 - f / 2, t / 2));
  wg.add(mesh(box(f, z1 - z0, t * 0.6), fm, u0 + f / 2, (z0 + z1) / 2, t / 2));
  wg.add(mesh(box(f, z1 - z0, t * 0.6), fm, u1 - f / 2, (z0 + z1) / 2, t / 2));
}

function doorFrame(wg, room, o, u0, u1, top, t, w) {
  const fm = solid(0xf7f4ee, 0.5), f = 0.05;
  wg.add(mesh(box(f, top, t + 0.02), fm, u0 - f / 2, top / 2, t / 2));
  wg.add(mesh(box(f, top, t + 0.02), fm, u1 + f / 2, top / 2, t / 2));
  wg.add(mesh(box(u1 - u0 + 2 * f, f, t + 0.02), fm, (u0 + u1) / 2, top + f / 2, t / 2));
  if (!o.swing) return;
  // swing arc on the floor, inside the room (interior = -W in wall space)
  const hingeAtStart = o.swing.hinge !== "end", r = u1 - u0;
  const into = o.swing.into_room !== false ? -1 : 1;
  const arc = new THREE.Mesh(new THREE.RingGeometry(r - 0.012, r, 32, 1, 0, Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0x9a8f83, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
  arc.rotation.x = -Math.PI / 2 * -into; // ring lies in the floor plane
  arc.position.set(hingeAtStart ? u0 : u1, 0.003, 0);
  arc.scale.x = hingeAtStart ? 1 : -1;
  arc.rotation.x = into < 0 ? Math.PI / 2 : -Math.PI / 2;
  wg.add(arc);
}

// ------------------------------------------------------------------ assets
const loader = new GLTFLoader();
function addAsset(a) {
  const holder = new THREE.Group();
  holder.userData.assetId = a.id;
  const proxy = buildProxy(a);
  proxy.position.y = a.elevation || 0;
  holder.add(proxy);
  const outline = footprintOutline(a);
  holder.add(outline);
  scene3.add(holder);
  state.objects[a.id] = { asset: a, holder, proxy, outline, model: null };
  if (a.mesh) loadModel(a);
}

function footprintOutline(a) {
  const pts = a.footprint.map(([x, y]) => new THREE.Vector3(x, 0.004, -y));
  pts.push(pts[0].clone());
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x2f6f62, transparent: true, opacity: 0 }));
  line.renderOrder = 2;
  return line;
}

function loadModel(a) {
  // Progressive: the proxy shows at once; the generated model fades in when it arrives.
  loader.load(a.mesh, (gltf) => {
    const o = state.objects[a.id], model = gltf.scene;
    model.position.y = a.elevation || 0;
    // roomlayout GLBs are glTF with front = +Z: ours (x, y, z) -> glTF (-x, z, y)
    // (assets/frames.py OURS_TO_GLTF). The proxies here use ours -> (x, z, -y), front -Z.
    // The two differ by 180 degrees about Y.
    model.rotation.y = Math.PI;
    model.traverse((m) => { if (m.isMesh) { m.castShadow = m.receiveShadow = true;
      m.material = m.material.clone(); m.material.transparent = true; m.material.opacity = 0; } });
    o.holder.add(model);
    o.model = model;
    fade(model, 0, 1, 600, () => model.traverse((m) => { if (m.isMesh) m.material.transparent = false; }));
    fadeProxyOut(o.proxy, 600);
  }, undefined, () => {});
}
function fade(obj, from, to, ms, done) {
  const t0 = performance.now();
  state.tweens.push((now) => {
    const k = Math.min(1, (now - t0) / ms);
    obj.traverse((m) => { if (m.isMesh) m.material.opacity = from + (to - from) * ease(k); });
    if (k >= 1) { done && done(); return false; }
    return true;
  });
}
function fadeProxyOut(proxy, ms) {
  proxy.traverse((m) => { if (m.isMesh) { m.material = m.material.clone(); m.material.transparent = true; } });
  fade(proxy, 1, 0, ms, () => (proxy.visible = false));
}

function setPose(id, p) {
  const o = state.objects[id];
  if (!o) return;
  o.holder.position.copy(v3(p.x, p.y));
  o.holder.rotation.y = p.yaw;
  state.poses[id] = { ...p };
}

function animateTo(placements, ms = 750) {
  const from = Object.fromEntries(Object.entries(state.poses).map(([k, v]) => [k, { ...v }]));
  const t0 = performance.now();
  state.tweens.push((now) => {
    const k = ease(Math.min(1, (now - t0) / ms));
    for (const [id, to] of Object.entries(placements)) {
      const f = from[id] || to;
      setPose(id, { x: f.x + (to.x - f.x) * k, y: f.y + (to.y - f.y) * k, yaw: f.yaw + wrap(to.yaw - f.yaw) * k });
    }
    return k < 1;
  });
}

// ------------------------------------------------------------------ panel
function renderVerdict(s) {
  const el = $("verdict"), fit = s.fit, feasible = s.layouts.filter((l) => l.feasible).length;
  const fits = fit ? fit.fits : feasible > 0;
  el.className = "card " + (fits ? "fits" : "nofit");
  let html = `<div class="big">${fits ? "It fits" : "It doesn't fit"}</div>`;
  if (fit) html += `<div class="muted">Furniture ${fit.furniture_m2.toFixed(1)} m² on ${fit.free_floor_m2.toFixed(1)} m² of free floor</div>`;
  if (fits) html += `<div class="muted">${feasible} layout${feasible === 1 ? "" : "s"} that keep every rule</div>`;
  if (!fits && fit && fit.advice.length) html += `<ul>${fit.advice.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>`;
  el.innerHTML = html;
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function renderTabs() {
  const nav = $("tabs");
  nav.innerHTML = "";
  state.layouts.forEach((l, i) => {
    const b = document.createElement("button");
    b.textContent = l.id === "edited" ? "Your edit" : `Layout ${i + 1}`;
    b.className = i === state.current ? "on" : "";
    b.title = l.feasible ? "keeps every hard rule" : "breaks a hard rule";
    b.onclick = () => showLayout(i);
    nav.appendChild(b);
  });
}

function renderLines(layout) {
  const ul = $("lines");
  ul.innerHTML = "";
  const icon = { why: "✓", bent: "!", broken: "✕" };
  for (const ln of layout?.lines || []) {
    const li = document.createElement("li");
    li.className = `${ln.kind} ${ln.origin === "user" || ln.origin === "vlm" ? "user" : ""}`;
    li.innerHTML = `<span class="icon">${icon[ln.kind]}</span><span>${esc(ln.text)}</span>`;
    li.onmouseenter = () => highlight(ln.assets, true);
    li.onmouseleave = () => highlight(ln.assets, false);
    ul.appendChild(li);
  }
  if (!ul.children.length) ul.innerHTML = `<li class="muted"><span></span><span>No layout to explain.</span></li>`;
}

function highlight(ids, on) {
  for (const id of ids || []) {
    const o = state.objects[id];
    if (!o) continue;
    o.outline.material.opacity = on || state.selected === id || state.locked.has(id) ? 1 : 0;
    for (const part of [o.proxy, o.model]) part?.traverse((m) => {
      if (!m.isMesh || !m.material.emissive) return;
      if (!m.userData.mat0) { m.material = m.material.clone(); m.userData.mat0 = true; }
      m.material.emissive.setHex(on ? 0x2f6f62 : 0x000000);
      m.material.emissiveIntensity = on ? 0.35 : 0;
    });
  }
}

function showLayout(i) {
  state.current = i;
  const l = state.layouts[i];
  if (l) animateTo(l.placements);
  renderTabs();
  renderLines(l);
}

// ------------------------------------------------------------------ editing
const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2();
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let drag = null;

function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const movable = Object.values(state.objects).filter((o) => o.asset.movable).map((o) => o.holder);
  const hit = raycaster.intersectObjects(movable, true).find((h) => h.object.isMesh);
  if (!hit) return null;
  let n = hit.object;
  while (n && !n.userData.assetId) n = n.parent;
  return n?.userData.assetId ?? null;
}
function floorPoint() {
  const p = new THREE.Vector3();
  return raycaster.ray.intersectPlane(floorPlane, p) ? p : null;
}

function select(id) {
  const prev = state.selected;
  state.selected = id;
  if (prev) highlight([prev], false);
  if (id) highlight([id], true);
  const a = id && state.objects[id].asset;
  $("selection").textContent = a ? `Selected: ${a.label || a.id}${state.locked.has(id) ? " (locked)" : ""}` : "Nothing selected.";
  for (const b of ["rot-l", "rot-r", "lock"]) $(b).disabled = !id;
  $("lock").textContent = id && state.locked.has(id) ? "Unlock" : "Lock";
}

renderer.domElement.addEventListener("pointerdown", (ev) => {
  const id = pick(ev);
  if (!id) return;
  select(id);
  const p = floorPoint();
  if (!p) return;
  const pos = state.poses[id];
  drag = { id, dx: pos.x - p.x, dy: pos.y - -p.z, moved: false };
  controls.enabled = false;
  renderer.domElement.setPointerCapture(ev.pointerId);
});
renderer.domElement.addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const p = floorPoint();
  if (!p) return;
  const pose = state.poses[drag.id];
  setPose(drag.id, { x: p.x + drag.dx, y: -p.z + drag.dy, yaw: pose.yaw });
  drag.moved = true;
});
renderer.domElement.addEventListener("pointerup", () => {
  if (!drag) return;
  const moved = drag.moved;
  drag = null;
  controls.enabled = true;
  if (moved) edited();
});

function rotate(deg) {
  const id = state.selected;
  if (!id) return;
  const p = state.poses[id];
  setPose(id, { ...p, yaw: wrap(p.yaw + (deg * Math.PI) / 180) });
  edited();
}
function toggleLock() {
  const id = state.selected;
  if (!id) return;
  state.locked.has(id) ? state.locked.delete(id) : state.locked.add(id);
  select(id);
  $("resolve").disabled = !state.served || state.locked.size === 0;
}
$("rot-l").onclick = () => rotate(15);
$("rot-r").onclick = () => rotate(-15);
$("lock").onclick = toggleLock;
window.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === "q" || ev.key === "Q") rotate(15);
  if (ev.key === "e" || ev.key === "E") rotate(-15);
  if (ev.key === "l" || ev.key === "L") toggleLock();
  if (ev.key === "Escape") select(null);
});

let evalTimer = null;
function edited() {
  if (!state.served) { $("edit-status").textContent = "Static page: serve it to check edits (python -m roomlayout.viewer serve DIR)."; return; }
  clearTimeout(evalTimer);
  evalTimer = setTimeout(async () => {
    $("edit-status").textContent = "Checking…";
    const r = await post("api/evaluate", { placements: state.poses });
    if (!r) return;
    const i = state.layouts.findIndex((l) => l.id === "edited");
    if (i >= 0) state.layouts[i] = r; else state.layouts.push(r);
    state.current = state.layouts.findIndex((l) => l.id === "edited");
    renderTabs();
    renderLines(r);
    $("edit-status").textContent = r.feasible ? "Your edit keeps every rule." : "Your edit breaks a rule (see above).";
  }, 150);
}

$("resolve").onclick = async () => {
  const locks = [...state.locked].map((id) => ({ id, ...state.poses[id] }));
  $("edit-status").textContent = "Re-arranging around the locked pieces…";
  $("resolve").disabled = true;
  const r = await post("api/solve", { locks });
  $("resolve").disabled = false;
  if (!r) return;
  if (!r.layouts.length) { $("edit-status").textContent = "Nothing fits around those locks. " + (r.notes[1] || ""); return; }
  state.layouts = r.layouts;
  $("edit-status").textContent = `${r.layouts.filter((l) => l.feasible).length} new layouts around your locked pieces.`;
  showLayout(0);
};

async function post(url, body) {
  try {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || res.statusText);
    return j;
  } catch (e) {
    $("edit-status").textContent = e instanceof TypeError ? `Couldn't reach the solver: ${e.message}` : `Can't do that: ${e.message}.`;
    return null;
  }
}

// ------------------------------------------------------------------ views
function frame(top = false) {
  const pts = state.scene.room.floor;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const target = v3(cx, cy, 0.4);
  controls.target.copy(target);
  if (top) camera.position.copy(v3(cx, cy - 0.001, span * 1.9));
  else camera.position.copy(v3(cx + span * 0.55, cy - span * 1.05, span * 1.05));
  camera.lookAt(target);
  sun.position.copy(v3(cx - span * 0.6, cy - span * 0.4, span * 1.6 + 3));
  sun.target.position.copy(v3(cx, cy));
  const sc = sun.shadow.camera;
  sc.left = sc.bottom = -span; sc.right = sc.top = span; sc.near = 0.5; sc.far = span * 4 + 10;
  sc.updateProjectionMatrix();
  $("view-3d").classList.toggle("on", !top);
  $("view-top").classList.toggle("on", top);
}
$("view-3d").onclick = () => frame(false);
$("view-top").onclick = () => frame(true);

const CUTAWAY = 0.12; // near walls drop to 12% of their height (dollhouse view)
function updateWalls() {
  // Dollhouse: walls between the camera and the room are cut down to a short stub, so
  // the furniture stays visible while the room outline (and door gaps) still read.
  const top = controls.getPolarAngle() < 0.25;
  for (const w of state.walls) {
    const toCam = camera.position.clone().sub(w.center).setY(0);
    const near = w.normal.dot(toCam) > 0 && !top;
    const target = near ? CUTAWAY : 1;
    w.group.scale.y += (target - w.group.scale.y) * 0.25;
  }
}

// ------------------------------------------------------------------ main
async function main() {
  const s = await (await fetch("scene.json")).json();
  state.scene = s;
  document.title = `${s.title} · Room Layout Viewer`;
  $("title").textContent = s.title;
  if (s.request) { $("request").hidden = false; $("request").textContent = `“${s.request}”`; }
  scene3.add(buildRoom(s.room));
  for (const a of s.assets) addAsset(a);
  for (const [id, p] of Object.entries(s.pinned || {})) setPose(id, p);
  state.layouts = s.layouts;
  const first = s.layouts[0];
  if (first) for (const [id, p] of Object.entries(first.placements)) setPose(id, p);
  // Pieces with no pose (nothing fits): line them up outside the room so they can be seen.
  const xs = s.room.floor.map((p) => p[0]), ys = s.room.floor.map((p) => p[1]);
  let x0 = Math.min(...xs);
  for (const a of s.assets) if (!state.poses[a.id]) {
    setPose(a.id, { x: x0 + a.dims.width / 2, y: Math.min(...ys) - a.dims.depth / 2 - 0.6, yaw: 0 });
    x0 += a.dims.width + 0.3;
  }
  renderVerdict(s);
  renderTabs();
  renderLines(first);
  resize();
  frame(false);
  try {
    const ping = await fetch("api/ping");
    state.served = ping.ok;
  } catch { state.served = false; }
  if (!state.served) $("edit-help").textContent = "Static page: serve it to edit (python -m roomlayout.viewer serve DIR). You can still compare layouts.";
  renderer.setAnimationLoop((now) => {
    state.tweens = state.tweens.filter((f) => f(now));
    controls.update();
    updateWalls();
    renderer.render(scene3, camera);
  });
  setTimeout(() => (window.__viewerReady = true), 900);
}
main().catch((e) => { $("title").textContent = "Couldn't load scene.json"; console.error(e); });
