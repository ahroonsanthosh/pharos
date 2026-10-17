"use strict";

// ---------------------------------------------------------------- state

const state = {
  feeds: [],
  selected: null, // feed id
  records: [],
  map: null, // { info, img }
  view: { zoom: 1, cx: 0, cy: 0 }, // map-preview pixel at the canvas centre
  camera: null, // { feedId, stream, ws, timer }
};

const $ = (sel) => document.querySelector(sel);
const form = $("#feed-form");
const canvas = $("#map");
const ctx = canvas.getContext("2d");

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

const postJson = (path, body) =>
  api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

// ---------------------------------------------------------------- feeds list

async function refreshFeeds() {
  try {
    state.feeds = await api("/api/feeds");
  } catch {
    return;
  }
  const list = $("#feed-list");
  list.replaceChildren(...state.feeds.map((f) => {
    const li = document.createElement("li");
    li.className = f.id === state.selected ? "selected" : "";
    const status = f.last?.status && f.state === "running" ? f.last.status : f.state;
    li.innerHTML = `<span class="name"></span><span class="state">${status}</span>`;
    li.querySelector(".name").textContent = f.name;
    li.onclick = () => selectFeed(f.id);
    return li;
  }));
  $("#no-feeds").hidden = state.feeds.length > 0;
  if (state.selected && !state.feeds.some((f) => f.id === state.selected)) selectFeed(null);
}

function selectFeed(id) {
  state.selected = id;
  state.records = [];
  state.map = null;
  $("#delete-feed").hidden = !id;
  $("#preview").removeAttribute("src");
  refreshFeeds();
  pollRecords();
}

async function pollRecords() {
  const id = state.selected;
  if (!id) return renderDetails(null);
  let data;
  try {
    data = await api(`/api/feeds/${id}/records?since=${state.records.length}`);
  } catch {
    return;
  }
  if (id !== state.selected) return;
  state.records.push(...data.records);
  if (data.map && state.map?.info.name !== data.map) await loadMap(data.map);
  if (data.records.length) $("#preview").src = `/api/feeds/${id}/preview.jpg?n=${state.records.length}`;
  renderDetails(data);
  draw();
}

async function loadMap(name) {
  const info = (await api("/api/maps")).find((m) => m.name === name);
  if (!info) return;
  const img = new Image();
  img.src = `/api/maps/${encodeURIComponent(name)}/preview.jpg`;
  await img.decode();
  state.map = { info, img };
  state.view = { zoom: Math.min(canvas.clientWidth / img.width, canvas.clientHeight / img.height), cx: img.width / 2, cy: img.height / 2 };
}

// ---------------------------------------------------------------- details panel

const fmt = (v, digits = 1, unit = "") => (v === undefined || v === null ? "–" : `${Number(v).toFixed(digits)}${unit}`);

function renderDetails(feed) {
  const status = $("#status");
  const last = state.records.at(-1);
  $("#preview-empty").hidden = !!last;
  if (!feed) {
    status.className = "status lost";
    status.textContent = "no feed";
    $("#readout").replaceChildren();
    $("#manual-controls").hidden = true;
    return;
  }
  if (feed.state === "error" || feed.error) {
    status.className = "status error";
    status.textContent = feed.error || feed.state;
  } else if (feed.state !== "running") {
    status.className = `status ${last?.status ?? "lost"}`;
    status.textContent = feed.state === "starting" ? "starting…" : `${feed.state} · ${last?.status ?? "no frames"}`;
  } else {
    status.className = `status ${last?.status ?? "lost"}`;
    status.textContent = last?.status ?? "waiting for frames";
  }
  const rows = [
    ["Latitude", last?.lat !== undefined ? last.lat.toFixed(6) : "–"],
    ["Longitude", last?.lon !== undefined ? last.lon.toFixed(6) : "–"],
    ["Heading", fmt(last?.heading_deg, 1, "°")],
    ["Uncertainty", last?.sigma_m !== undefined ? `±${fmt(last.sigma_m, 1, " m")}` : "–"],
    ["Match score", fmt(last?.score, 2)],
    ["Height (reported)", fmt(last?.altitude_m, 1, " m")],
    ["Camera tilt", fmt(last?.tilt_deg, 1, "°")],
    ["Error vs GNSS", fmt(last?.error_m, 1, " m")],
    ["Frames", String(state.records.length)],
    ["Processing", fmt(last?.latency_ms, 0, " ms")],
  ];
  $("#readout").replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = k;
    dd.textContent = v;
    return [dt, dd];
  }));
  const manual = feed.spec.telemetry === "manual" && !feed.spec.source.startsWith("demo:");
  const controls = $("#manual-controls");
  if (manual && controls.hidden) {
    $("#manual-alt").value = feed.spec.altitude_m;
    $("#manual-tilt").value = feed.spec.tilt_deg;
  }
  controls.hidden = !manual;
}

$("#manual-apply").onclick = () =>
  api(`/api/feeds/${state.selected}/telemetry`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ altitude_m: +$("#manual-alt").value, tilt_deg: +$("#manual-tilt").value }),
  });

$("#delete-feed").onclick = async () => {
  const id = state.selected;
  if (state.camera?.feedId === id) stopCamera();
  await api(`/api/feeds/${id}`, { method: "DELETE" });
  selectFeed(null);
};

// ---------------------------------------------------------------- map canvas

function toPreview(x, y) {
  const { info, img } = state.map;
  const s = img.width / info.width;
  const u = (x - info.extent[0]) / info.gsd;
  const v = (info.extent[3] - y) / info.gsd;
  return [(u + 0.5) * s - 0.5, (v + 0.5) * s - 0.5];
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("#map-empty").hidden = !!state.map;
  if (!state.map) return;
  const { img, info } = state.map;
  const located = state.records.filter((r) => r.x !== undefined);
  const last = located.at(-1);
  if ($("#follow").checked && last) [state.view.cx, state.view.cy] = toPreview(last.x, last.y);

  const zoom = state.view.zoom * dpr;
  const { cx, cy } = state.view;
  const screen = ([px, py]) => [(px - cx) * zoom + canvas.width / 2, (py - cy) * zoom + canvas.height / 2];
  const [ox, oy] = screen([0, 0]);
  ctx.drawImage(img, ox, oy, img.width * zoom, img.height * zoom);

  const gnss = state.records.filter((r) => r.gnss);
  if (gnss.length > 1) {
    ctx.strokeStyle = "#4cd07d";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    gnss.forEach((r, i) => {
      const [sx, sy] = screen(toPreview(r.gnss.x, r.gnss.y));
      i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
    });
    ctx.stroke();
  }
  for (const r of located) {
    const [sx, sy] = screen(toPreview(r.x, r.y));
    ctx.fillStyle = r.status === "tracking" ? "#3b8cff" : "#ffa53b";
    ctx.beginPath();
    ctx.arc(sx, sy, 2.5 * dpr, 0, 2 * Math.PI);
    ctx.fill();
  }
  if (last) {
    const [sx, sy] = screen(toPreview(last.x, last.y));
    const s = img.width / info.width;
    const radius = Math.max((last.sigma_m / info.gsd) * s * zoom, 6 * dpr);
    const h = (last.heading_deg * Math.PI) / 180;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, 2 * Math.PI);
    ctx.moveTo(sx, sy);
    ctx.lineTo(sx + Math.sin(h) * 22 * dpr, sy - Math.cos(h) * 22 * dpr);
    ctx.stroke();
  }
}

let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
  $("#follow").checked = false;
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  state.view.cx -= (e.clientX - drag.x) / state.view.zoom;
  state.view.cy -= (e.clientY - drag.y) / state.view.zoom;
  drag = { x: e.clientX, y: e.clientY };
  draw();
});
canvas.addEventListener("pointerup", () => (drag = null));
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  state.view.zoom *= Math.exp(-e.deltaY * 0.0015);
  draw();
}, { passive: false });
window.addEventListener("resize", draw);
$("#follow").onchange = draw;

// ---------------------------------------------------------------- new-feed form

function showFormSections() {
  const kind = form.kind.value;
  for (const el of form.querySelectorAll("[data-kind]")) el.hidden = el.dataset.kind !== kind;
  for (const el of form.querySelectorAll("[data-not-kind]")) el.hidden = el.dataset.notKind === kind;
  for (const el of form.querySelectorAll("[data-telemetry]")) el.hidden = el.dataset.telemetry !== form.telemetry.value;
  if (kind === "browser") listCameras();
  if (kind === "push") showPushUrls();
}

async function fillFormOptions() {
  const [scenarios, maps] = await Promise.all([api("/api/scenarios"), api("/api/maps")]);
  form.scenario.replaceChildren(...scenarios.map((s) => new Option(s.replaceAll("_", " "), s, s === "nominal", s === "nominal")));
  form.map.replaceChildren(...maps.map((m) => new Option(`${m.name} (${m.size_km[0]} × ${m.size_km[1]} km)`, m.name)));
}

async function showPushUrls() {
  const { lan_ips } = await api("/api/info");
  const hosts = lan_ips.length ? lan_ips : [location.hostname];
  const path = form.push_path.value || "drone";
  const urls = hosts.flatMap((h) => [
    `rtmp://${h}:1935/${path}  (DJI Fly / Pilot "Live streaming", OBS)`,
    `srt://${h}:8890?streamid=publish:${path}`,
    `rtsp://${h}:8554/${path}`,
  ]);
  if (!lan_ips.length) urls.push(`From another device, replace ${location.hostname} with this computer's Wi-Fi IP address.`);
  $("#push-urls").replaceChildren(...urls.map((u) => Object.assign(document.createElement("li"), { textContent: u })));
}

async function listCameras() {
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    probe.getTracks().forEach((t) => t.stop());
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    form.camera.replaceChildren(...devices.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
  } catch (err) {
    form.camera.replaceChildren(new Option(`camera unavailable: ${err.message}`, ""));
  }
}

async function upload(file) {
  const body = new FormData();
  body.append("file", file);
  return (await api("/api/uploads", { method: "POST", body })).path;
}

$("#upload-map-btn").onclick = async () => {
  const file = form.map_file.files[0];
  const name = form.map_name.value || file?.name.replace(/\.[^.]+$/, "");
  if (!file) return showError("Choose a GeoTIFF first.");
  const body = new FormData();
  body.append("files", file);
  try {
    await api(`/api/maps?name=${encodeURIComponent(name)}`, { method: "POST", body });
    await fillFormOptions();
    form.map.value = name;
    showError("");
  } catch (err) {
    showError(err.message);
  }
};

function showError(message) {
  const el = $("#form-error");
  el.textContent = message;
  el.hidden = !message;
}

async function buildSpec() {
  const f = form;
  const kind = f.kind.value;
  if (kind === "demo") return { source: `demo:${f.scenario.value}`, name: f.name.value || `demo: ${f.scenario.value}` };
  const spec = {
    name: f.name.value,
    map: f.map.value,
    hfov_deg: +f.hfov_deg.value,
    rate_hz: +f.rate_hz.value,
    telemetry: f.telemetry.value,
    altitude_m: +f.altitude_m.value,
    tilt_deg: +f.tilt_deg.value,
    mount: f.mount.value,
    video_delay_s: +f.video_delay_s.value,
  };
  if (!spec.map) throw new Error("Upload or choose a map first.");
  if (spec.telemetry === "mavlink") spec.telemetry = `mavlink:udpin:0.0.0.0:${f.mavlink_port.value}`;
  if (kind === "push") spec.source = `push:${f.push_path.value}`;
  if (kind === "url") spec.source = f.url.value.trim();
  if (kind === "browser") spec.source = "browser";
  if (kind === "device") spec.source = String(f.device.value);
  if (kind === "file") {
    if (!f.video.files[0]) throw new Error("Choose a video file.");
    spec.source = await upload(f.video.files[0]);
    if (f.srt.files[0]) spec.telemetry = `srt:${await upload(f.srt.files[0])}`;
  }
  if (spec.telemetry === "srt") throw new Error("Choose the .SRT file for DJI telemetry.");
  if (!spec.source) throw new Error("Enter a source.");
  spec.name ||= spec.source.split("/").pop();
  return spec;
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const submit = form.querySelector("button[type=submit]");
  submit.disabled = true;
  try {
    const feed = await postJson("/api/feeds", await buildSpec());
    if (feed.spec.source === "browser") await startCamera(feed.id, form.camera.value, feed.spec.rate_hz);
    form.hidden = true;
    showError("");
    selectFeed(feed.id);
  } catch (err) {
    showError(err.message);
  } finally {
    submit.disabled = false;
  }
};

$("#new-feed-btn").onclick = () => {
  form.hidden = false;
  fillFormOptions();
  showFormSections();
};
$("#cancel-btn").onclick = () => (form.hidden = true);
form.kind.onchange = showFormSections;
form.telemetry.onchange = showFormSections;
form.push_path.oninput = showPushUrls;

// ---------------------------------------------------------------- browser camera

async function startCamera(feedId, deviceId, rateHz) {
  stopCamera();
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 } },
  });
  const video = $("#camera-video");
  video.srcObject = stream;
  await video.play();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/feeds/${feedId}/frames`);
  const grab = document.createElement("canvas");
  const timer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
    grab.width = video.videoWidth;
    grab.height = video.videoHeight;
    grab.getContext("2d").drawImage(video, 0, 0);
    grab.toBlob((blob) => blob && ws.send(blob), "image/jpeg", 0.85);
  }, 1000 / rateHz);
  state.camera = { feedId, stream, ws, timer };
}

function stopCamera() {
  if (!state.camera) return;
  clearInterval(state.camera.timer);
  state.camera.ws.close();
  state.camera.stream.getTracks().forEach((t) => t.stop());
  state.camera = null;
}

// ---------------------------------------------------------------- start

refreshFeeds();
setInterval(refreshFeeds, 2000);
setInterval(pollRecords, 500);
draw();
