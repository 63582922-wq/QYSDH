#!/usr/bin/env node
/**
 * Render / 生产：用 Node 读 **运行时** `process.env`，写入 `index.html` 内的 `window.__RUNTIME_APP_ENV__`，
 * 避免仅 `vite build` 阶段拿不到密钥导致前端一直是空字符串。
 */
import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const dist = path.join(root, 'dist');

for (const name of ['.env', '.env.local', '.env.production', '.env.production.local']) {
  dotenv.config({ path: path.join(root, name) });
}

const RUNTIME_KEYS = [
  'GEMINI_API_KEY',
  'GEMINI_LIVE_SPEECH_VOICE_NAME',
  'MINIMAX_API_KEY',
  'MINIMAX_API_BASE',
  'MINIMAX_GROUP_ID',
  'MINIMAX_VOICE_ID',
  'MINIMAX_TTS_MODEL',
  'MINIMAX_TTS_SPEED',
  'MINIMAX_CHAT_API_KEY',
  'MINIMAX_CHAT_API_BASE',
  'MINIMAX_CHAT_MODEL',
  'MINIMAX_CHAT_VISION_MODEL',
  'MINIMAX_CHAT_SEPARATE_PROXY',
];

function buildRuntimePayload() {
  const o = {};
  for (const k of RUNTIME_KEYS) {
    o[k] = process.env[k] != null ? String(process.env[k]) : '';
  }
  return o;
}

const runtimePayload = buildRuntimePayload();
const gem = Boolean(String(runtimePayload.GEMINI_API_KEY || '').trim());
const mini =
  Boolean(String(runtimePayload.MINIMAX_API_KEY || '').trim()) ||
  Boolean(String(runtimePayload.MINIMAX_CHAT_API_KEY || '').trim());
console.info(
  `[serve-render-dist] GEMINI_API_KEY=${gem ? 'present' : 'MISSING'} MiniMax(voice/chat)=${mini ? 'present' : 'MISSING'}`,
);

const indexPath = path.join(dist, 'index.html');
if (!fs.existsSync(indexPath)) {
  console.error('[serve-render-dist] Missing dist/index.html — run `npm run build` first.');
  process.exit(1);
}

function injectRuntime(html) {
  const inject = `<script>window.__RUNTIME_APP_ENV__=${JSON.stringify(runtimePayload)}<\/script>`;
  if (html.includes('__RUNTIME_APP_ENV__')) return html;
  return html.replace('</head>', `${inject}</head>`);
}

const indexHtml = injectRuntime(fs.readFileSync(indexPath, 'utf8'));

const app = express();
const port = Number(process.env.PORT) || 4173;

app.use(express.static(dist, { index: false }));
app.use((_req, res) => {
  res.status(200).type('html').send(indexHtml);
});

app.listen(port, '0.0.0.0', () => {
  console.info(`[serve-render-dist] listening on 0.0.0.0:${port}`);
});
