import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import dotenv from 'dotenv';
import fs from 'node:fs';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';
import {normalizeMinimaxEnvValue} from './src/minimaxEnv';

function readEnvFileParsed(fp: string): Record<string, string> {
  let raw = fs.readFileSync(fp, 'utf8');
  if (raw.charCodeAt(0) === 0xfeff) raw = raw.slice(1);
  return dotenv.parse(raw);
}

/**
 * 合并 MINIMAX_*：先读 `.env` / `.env.<mode>` / `.env.<mode>.local`，再 **始终单独合并 `.env.local`**。
 * 这样本机你在 `.env.local` 里写的 `MINIMAX_API_KEY` / `MINIMAX_API_BASE` 一定压过其它文件，避免「明明改了 local 仍像旧 Key」。
 * 若某键在所有文件里都不存在，再回退到 `loadEnv`（便于 CI 用环境变量注入）。
 */
function minimaxFromEnvFiles(mode: string, envDir: string): Partial<Record<string, string>> {
  const merged: Partial<Record<string, string>> = {};
  const baseNames = ['.env', `.env.${mode}`, `.env.${mode}.local`];
  for (const name of baseNames) {
    const fp = path.join(envDir, name);
    try {
      if (!fs.existsSync(fp) || !fs.statSync(fp).isFile()) continue;
    } catch {
      continue;
    }
    const parsed = readEnvFileParsed(fp);
    for (const [k, v] of Object.entries(parsed)) {
      if (k.startsWith('MINIMAX_') && v !== undefined) merged[k] = String(v);
    }
  }
  const localPath = path.join(envDir, '.env.local');
  try {
    if (fs.existsSync(localPath) && fs.statSync(localPath).isFile()) {
      const parsed = readEnvFileParsed(localPath);
      for (const [k, v] of Object.entries(parsed)) {
        if (k.startsWith('MINIMAX_') && v !== undefined) merged[k] = String(v);
      }
    }
  } catch {
    /* ignore */
  }
  return merged;
}

/**
 * 合并策略：非空优先。
 * 避免仓库里 `.env` / `.env.production` 写了空的 `MINIMAX_API_KEY=` 时，盖掉 Render 等在构建期注入的 process.env。
 */
function pickMinimax(
  fileVars: Partial<Record<string, string>>,
  env: Record<string, string>,
  key: string,
): string | undefined {
  const fromFile = fileVars[key];
  const fromEnv = env[key];
  const fileNorm = fromFile !== undefined ? normalizeMinimaxEnvValue(fromFile) : '';
  const envNorm = fromEnv !== undefined ? normalizeMinimaxEnvValue(fromEnv) : '';
  if (fileNorm !== '') return fromFile;
  if (envNorm !== '') return fromEnv;
  return fromFile !== undefined ? fromFile : fromEnv;
}

export default defineConfig(({mode}) => {
  const root = path.resolve(__dirname);
  // root / envDir 与 config 同目录：从仓库父目录执行 `npm run dev` 时仍能读到子项目里的 .env.local
  const env = loadEnv(mode, root, '');
  const mf = minimaxFromEnvFiles(mode, root);
  /** 与官方文档一致：国内常用 https://api.minimaxi.com，国际常用 https://api.minimax.io；开发代理目标读自 MINIMAX_API_BASE */
  const minimaxProxyTarget =
    normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_API_BASE')).replace(/\/$/, '') ||
    'https://api.minimax.io';
  const minimaxChatBaseSan = normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_CHAT_API_BASE'));
  const minimaxChatProxyTarget =
    minimaxChatBaseSan.replace(/\/$/, '') || minimaxProxyTarget;
  const separateChatProxy =
    minimaxChatBaseSan !== '' &&
    minimaxChatProxyTarget.replace(/\/$/, '') !== minimaxProxyTarget.replace(/\/$/, '');
  const minimaxKeySan = normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_API_KEY'));
  const minimaxChatKeySan = normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_CHAT_API_KEY'));
  /** 生产构建在 Render 等 CI 上跑时，在 Build Logs 里可核对密钥是否被注入（不打印密钥内容） */
  if (mode === 'production') {
    console.info(
      `[vite-build] MiniMax: MINIMAX_API_KEY=${minimaxKeySan ? 'present' : 'MISSING'}, MINIMAX_CHAT_API_KEY=${minimaxChatKeySan ? 'present' : 'MISSING'}`,
    );
  }
  if (mode === 'development') {
    const keyHint = !minimaxKeySan
      ? '（未配置）'
      : minimaxKeySan.startsWith('sk-cp-')
        ? 'sk-cp-…（Token Plan，语音 HTTP 会失败）'
        : minimaxKeySan.startsWith('sk-api-')
          ? 'sk-api-…（开放平台形态，可试语音）'
          : `${minimaxKeySan.slice(0, 8)}…`;
    console.info(`[vite] MiniMax 语音代理 → ${minimaxProxyTarget}`);
    console.info(`[vite] MINIMAX_API_KEY 形态: ${keyHint}`);
    if (separateChatProxy) {
      console.info(`[vite] MiniMax 对话代理 → ${minimaxChatProxyTarget}（路径 /minimax-chat-api）`);
    } else {
      console.info('[vite] MiniMax 对话与语音共用 /minimax-api');
    }
    if (minimaxChatKeySan) {
      console.info(`[vite] 已配置 MINIMAX_CHAT_API_KEY（克隆模式图文对话）`);
    }
  }
  return {
    root,
    envDir: root,
    plugins: [react(), tailwindcss()],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_LIVE_SPEECH_VOICE_NAME': JSON.stringify(env.GEMINI_LIVE_SPEECH_VOICE_NAME || ''),
      'process.env.MINIMAX_API_KEY': JSON.stringify(minimaxKeySan),
      'process.env.MINIMAX_API_BASE': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_API_BASE')),
      ),
      'process.env.MINIMAX_GROUP_ID': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_GROUP_ID')),
      ),
      'process.env.MINIMAX_VOICE_ID': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_VOICE_ID')),
      ),
      'process.env.MINIMAX_TTS_MODEL': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_TTS_MODEL')),
      ),
      'process.env.MINIMAX_TTS_SPEED': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_TTS_SPEED')),
      ),
      'process.env.MINIMAX_CHAT_API_KEY': JSON.stringify(minimaxChatKeySan),
      'process.env.MINIMAX_CHAT_API_BASE': JSON.stringify(minimaxChatBaseSan),
      'process.env.MINIMAX_CHAT_MODEL': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_CHAT_MODEL')),
      ),
      'process.env.MINIMAX_CHAT_VISION_MODEL': JSON.stringify(
        normalizeMinimaxEnvValue(pickMinimax(mf, env, 'MINIMAX_CHAT_VISION_MODEL')),
      ),
      'process.env.MINIMAX_CHAT_SEPARATE_PROXY': JSON.stringify(separateChatProxy ? '1' : ''),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      host: true,
      port: 5173,
      strictPort: false,
      open: true,
      proxy: {
        // 开发环境绕过浏览器对 api.minimax.io 的 CORS；生产环境可自建同源代理并设置 MINIMAX_API_BASE
        '/minimax-api': {
          target: minimaxProxyTarget,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/minimax-api/, ''),
        },
        ...(separateChatProxy
          ? {
              '/minimax-chat-api': {
                target: minimaxChatProxyTarget,
                changeOrigin: true,
                rewrite: (p) => p.replace(/^\/minimax-chat-api/, ''),
              },
            }
          : {}),
      },
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify — file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
    /** `vite preview`（Render 等用 start command 跑 preview）会校验 Host，否则报 Blocked request */
    preview: {
      host: true,
      strictPort: false,
      allowedHosts: true,
    },
  };
});
