// Golden-ratio drawing on a face crop, shared by the picking and scores pages.
// Points are fractions (0-1) of the square face crop, so an SVG with viewBox
// 0 0 1 1 laid over an object-fit:contain image lines up with the face.
(() => {
  const PHI = 1.618;
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const NAMES = {
    id: ["Panjang ÷ lebar wajah", "Lebar mulut ÷ lebar hidung", "Hidung–dagu ÷ bibir–dagu", "Rentang mata ÷ lebar mulut", "Pupil–dagu ÷ pupil–dahi"],
    en: ["Face length ÷ width", "Mouth width ÷ nose width", "Nose–chin ÷ lips–chin", "Eye span ÷ mouth width", "Pupils–chin ÷ pupils–forehead"],
  };
  const COLORS = ["#f5c542", "#4fc3f7", "#81c784", "#ff8a65", "#ce93d8"];

  // The five proportions the golden-ratio score uses, each with the two
  // segments it compares (drawn solid and dashed).
  function ratios(P, lang = "id") {
    const st = mid(P.lip_top, P.lip_bottom), pu = mid(P.pupil_r, P.pupil_l);
    const defs = [
      [[P.top, P.chin], [P.cheek_r, P.cheek_l]],
      [[P.mouth_r, P.mouth_l], [P.ala_r, P.ala_l]],
      [[P.nose_tip, P.chin], [st, P.chin]],
      [[P.eye_r_out, P.eye_l_out], [P.mouth_r, P.mouth_l]],
      [[pu, P.chin], [pu, P.top]],
    ];
    return defs.map(([a, b], i) => ({ name: NAMES[lang][i], color: COLORS[i], a, b, value: d(...a) / d(...b) }));
  }
  // 0-100: how close a ratio is to 1.618 (100 = exactly, 0 = 50% or more off).
  const closeness = (v) => Math.max(0, Math.round(100 * (1 - Math.abs(Math.log(v / PHI)) / Math.log(1.5))));

  // Overlay SVG. `only` = index of one ratio to draw alone (others faded).
  function svg(P, lang = "id", only = null) {
    const rs = ratios(P, lang);
    // Nudge the second segment sideways so overlapping segments stay visible.
    const seg = ([p, q], color, dash, off) => {
      const dx = q[0] - p[0], dy = q[1] - p[1], n = Math.hypot(dx, dy) || 1;
      const ox = (-dy / n) * off, oy = (dx / n) * off;
      return `<line x1="${p[0] + ox}" y1="${p[1] + oy}" x2="${q[0] + ox}" y2="${q[1] + oy}" stroke="${color}"
        stroke-width="0.006" stroke-linecap="round" ${dash ? 'stroke-dasharray="0.012 0.01"' : ""}/>`;
    };
    const lines = rs.map((r, i) => `<g opacity="${only == null || only === i ? 1 : 0.12}">
      ${seg(r.a, r.color, false, 0)}${seg(r.b, r.color, true, 0.008 * (i + 1))}</g>`).join("");
    const dots = Object.values(P).map(([x, y]) => `<circle cx="${x}" cy="${y}" r="0.006" fill="#fff" stroke="#000" stroke-width="0.002"/>`).join("");
    return `<svg class="phi" viewBox="0 0 1 1" preserveAspectRatio="xMidYMid meet" aria-hidden="true">${lines}${dots}</svg>`;
  }
  window.Phi = { PHI, ratios, closeness, svg };
})();
