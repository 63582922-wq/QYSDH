/// <reference types="vite/client" />

import { normalizeMinimaxEnvValue } from './minimaxEnv';

declare global {
  interface Window {
    /**
     * 生产环境由 `scripts/serve-render-dist.mjs` 在 HTML 中注入；
     * 优先于 Vite `define` 在 `vite build` 时写入的常量（解决 Render 仅运行期有 Key、构建期为空的问题）。
     */
    __RUNTIME_APP_ENV__?: Record<string, string | undefined>;
  }
}

/** 原始 trim；不套 minimax 引号规则 */
export function runtimeEnvTrimmed(key: string): string {
  if (typeof window === 'undefined') return '';
  const v = window.__RUNTIME_APP_ENV__?.[key];
  if (v == null) return '';
  return String(v).trim();
}

export function runtimeMinimaxTrimmed(key: string): string {
  return normalizeMinimaxEnvValue(runtimeEnvTrimmed(key));
}

export function getGeminiApiKey(): string {
  const r = runtimeEnvTrimmed('GEMINI_API_KEY');
  if (r) return r;
  return typeof process !== 'undefined' && process.env.GEMINI_API_KEY
    ? String(process.env.GEMINI_API_KEY).trim()
    : '';
}

export function getGeminiLiveSpeechVoiceNameFromRuntimeOrDefine(): string {
  const r = runtimeEnvTrimmed('GEMINI_LIVE_SPEECH_VOICE_NAME');
  if (r) return r;
  if (typeof process !== 'undefined' && process.env.GEMINI_LIVE_SPEECH_VOICE_NAME != null) {
    const t = String(process.env.GEMINI_LIVE_SPEECH_VOICE_NAME).trim();
    if (t.length > 0) return t;
  }
  return '';
}
