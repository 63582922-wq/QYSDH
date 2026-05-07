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
  /**
   * 生产由 `serve-render-dist` 注入 `__RUNTIME_APP_ENV__` 时，**禁止**再回退到 Vite `define` 写死的
   * `process.env.GEMINI_API_KEY`（那是 `npm run build` 当刻的旧值），否则本机改 `.env.local` 仍像「没换新密钥」。
   * 值经 `normalizeMinimaxEnvValue`：去掉 `.env` 里成对引号与首尾空白，避免 Google 报 API_KEY_INVALID。
   */
  if (typeof window !== 'undefined' && window.__RUNTIME_APP_ENV__ != null) {
    const v = window.__RUNTIME_APP_ENV__['GEMINI_API_KEY'];
    return normalizeMinimaxEnvValue(v != null ? String(v) : undefined);
  }
  return normalizeMinimaxEnvValue(
    typeof process !== 'undefined' && process.env.GEMINI_API_KEY != null
      ? String(process.env.GEMINI_API_KEY)
      : undefined,
  );
}

export function getGeminiLiveSpeechVoiceNameFromRuntimeOrDefine(): string {
  if (typeof window !== 'undefined' && window.__RUNTIME_APP_ENV__ != null) {
    const v = window.__RUNTIME_APP_ENV__['GEMINI_LIVE_SPEECH_VOICE_NAME'];
    return normalizeMinimaxEnvValue(v != null ? String(v) : undefined);
  }
  const t = normalizeMinimaxEnvValue(
    typeof process !== 'undefined' && process.env.GEMINI_LIVE_SPEECH_VOICE_NAME != null
      ? String(process.env.GEMINI_LIVE_SPEECH_VOICE_NAME)
      : undefined,
  );
  return t.length > 0 ? t : '';
}
