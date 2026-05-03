/** 供 Vite 与浏览器共用：去掉首尾空白及成对引号，避免 .env 写成 KEY="…" 时把引号带进 Bearer */

export function normalizeMinimaxEnvValue(s: string | undefined): string {
  if (s == null || typeof s !== 'string') return '';
  let t = s.trim();
  if (
    t.length >= 2 &&
    ((t[0] === '"' && t[t.length - 1] === '"') || (t[0] === "'" && t[t.length - 1] === "'"))
  ) {
    t = t.slice(1, -1).trim();
  }
  return t;
}
