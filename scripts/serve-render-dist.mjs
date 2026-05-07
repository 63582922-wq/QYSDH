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

/** 启动时合并（不设 override，避免仓库里空占位符盖掉宿主已注入的 process.env） */
for (const name of ['.env', '.env.local', '.env.production', '.env.production.local']) {
  dotenv.config({ path: path.join(root, name) });
}

/**
 * 每次下发 HTML 前再读本地覆盖文件，便于本机改 `.env.local` 里的 GEMINI_* 后**无需重启 Node**。
 * 仅用 `override: true` 重载「*.local」，不动 `.env`，减轻误用空值盖掉 Render 环境变量的风险。
 */
function reloadLocalEnvOverrides() {
  for (const name of ['.env.local', '.env.production.local']) {
    const fp = path.join(root, name);
    try {
      if (!fs.existsSync(fp) || !fs.statSync(fp).isFile()) continue;
    } catch {
      continue;
    }
    dotenv.config({ path: fp, override: true });
  }
}

const RUNTIME_KEYS = [
  'GEMINI_API_KEY',
  'GEMINI_LIVE_SPEECH_VOICE_NAME',
  'MINIMAX_API_KEY',
  'MINIMAX_API_BASE',
  'MINIMAX_GROUP_ID',
  'MINIMAX_VOICE_ID',
  'MINIMAX_TTS_MODEL',
  'MINIMAX_VOICE_CLONE_MODEL',
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

function logRuntimeEnvPresence() {
  const p = buildRuntimePayload();
  const gem = Boolean(String(p.GEMINI_API_KEY || '').trim());
  const mini =
    Boolean(String(p.MINIMAX_API_KEY || '').trim()) ||
    Boolean(String(p.MINIMAX_CHAT_API_KEY || '').trim());
  console.info(
    `[serve-render-dist] GEMINI_API_KEY=${gem ? 'present' : 'MISSING'} MiniMax(voice/chat)=${mini ? 'present' : 'MISSING'}`,
  );
}

const indexPath = path.join(dist, 'index.html');
if (!fs.existsSync(indexPath)) {
  console.error('[serve-render-dist] Missing dist/index.html — run `npm run build` first.');
  process.exit(1);
}

function injectRuntime(html, runtimePayload) {
  const inject = `<script>window.__RUNTIME_APP_ENV__=${JSON.stringify(runtimePayload)}<\/script>`;
  if (html.includes('__RUNTIME_APP_ENV__')) return html;
  /** 必须早于 Vite 打在 head 里的 `type="module"` 执行，否则首包已用空 Key 初始化 SDK */
  if (html.includes('<head>')) {
    return html.replace('<head>', `<head>${inject}`);
  }
  return html.replace('</head>', `${inject}</head>`);
}

function renderSpaIndexHtml() {
  reloadLocalEnvOverrides();
  return injectRuntime(fs.readFileSync(indexPath, 'utf8'), buildRuntimePayload());
}

logRuntimeEnvPresence();

const app = express();
const port = Number(process.env.PORT) || 4173;

app.use(express.static(dist, { index: false }));

/**
 * 部署或本地 rebuild 后，Vite 会换新 hashed 文件名；若浏览器仍用缓存的旧 index.html，
 * 会请求已不存在的 /assets/index-OLDHASH.js。若此处再回退成 SPA 的 index.html（text/html），
 * 浏览器会报「Expected a JavaScript module… MIME type text/html」。
 */
app.use((req, res, next) => {
  const p = req.path || '';
  const looksLikeBundledAsset =
    p.startsWith('/assets/') || /\.(js|mjs|css|map|wasm)$/i.test(path.extname(p));
  if (looksLikeBundledAsset) {
    res
      .status(404)
      .type('text/plain; charset=utf-8')
      .send(
        '资源不存在（多半是缓存了旧版页面）。请对本页强制刷新（Cmd+Shift+R）或清空本站缓存后再打开；部署后应重新打开标签页以加载新的 index.html。',
      );
    return;
  }
  next();
});

app.use((_req, res) => {
  res.setHeader('Cache-Control', 'no-cache');
  res.status(200).type('html').send(renderSpaIndexHtml());
});

app.listen(port, '0.0.0.0', () => {
  console.info(`[serve-render-dist] listening on 0.0.0.0:${port}`);
});
