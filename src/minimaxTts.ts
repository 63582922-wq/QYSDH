/// <reference types="vite/client" />

/** MiniMax 文本转语音（T2A HTTP），与官方 OpenAPI 一致：https://platform.minimax.io/docs/api-reference/speech-t2a-http */

import { normalizeMinimaxEnvValue } from './minimaxEnv';

export { normalizeMinimaxEnvValue };

export function getMinimaxApiKey(): string {
  return normalizeMinimaxEnvValue(
    typeof process !== 'undefined' && typeof process.env.MINIMAX_API_KEY === 'string'
      ? process.env.MINIMAX_API_KEY
      : undefined,
  );
}

/**
 * 开发态弱提示：sk-cp- 多为 Token Plan，语音 HTTP 常不兼容。
 * 不再 throw，避免与本机 `.env` 与注入结果不一致时，上传前就被同一句话拦住。
 */
export function warnMinimaxTokenPlanKeyInDev(key: string): void {
  const k = key.trim();
  if (!k || !k.startsWith('sk-cp-')) return;
  if (import.meta.env.DEV) {
    console.warn(
      '[MiniMax] 当前 Key 前缀为 sk-cp-（多为 Token Plan）。语音上传/复刻/T2A 通常需开放平台密钥，并与 MINIMAX_API_BASE 地域一致；若失败以 Network 响应为准。',
    );
  }
}

export interface MinimaxT2aResponse {
  data?: {
    audio?: string;
    status?: number;
  } | null;
  base_resp?: {
    status_code?: number;
    status_msg?: string;
  };
}

export function resolveMinimaxApiRoot(): string {
  const fromEnv =
    typeof process !== 'undefined' && typeof process.env.MINIMAX_API_BASE === 'string'
      ? normalizeMinimaxEnvValue(process.env.MINIMAX_API_BASE)
      : '';
  /** 开发环境始终走同源 `/minimax-api`，由 Vite 代理到 `MINIMAX_API_BASE`（未配则 .io），避免直连国内/国际域名触发 CORS */
  if (import.meta.env.DEV) return '/minimax-api';
  if (fromEnv) return fromEnv.replace(/\/$/, '');
  return 'https://api.minimax.io';
}

/** 部分控制台（含企业/套餐）除 API Key 外提供 GroupId，需作为 URL 查询参数 */
export function withMinimaxGroupQuery(url: string): string {
  const gid =
    typeof process !== 'undefined' && typeof process.env.MINIMAX_GROUP_ID === 'string'
      ? normalizeMinimaxEnvValue(process.env.MINIMAX_GROUP_ID)
      : '';
  if (!gid) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}GroupId=${encodeURIComponent(gid)}`;
}

export function stripTextForTts(text: string): string {
  let s = typeof text === 'string' ? text : String(text ?? '');
  s = s.replace(/\*\*([^*]+)\*\*/g, '$1');
  s = s.replace(/\*([^*]+)\*/g, '$1');
  s = s.replace(/`{1,3}[^`]*`{1,3}/g, ' ');
  s = s.replace(/\u0000/g, '');
  return s.replace(/\s+/g, ' ').trim();
}

function hexToUint8Array(hex: string): Uint8Array {
  const clean = hex.replace(/\s/g, '');
  if (clean.length % 2 !== 0) throw new Error('Invalid hex audio length');
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < clean.length; i += 2) {
    out[i / 2] = parseInt(clean.slice(i, i + 2), 16);
  }
  return out;
}

export interface SynthesizeMinimaxOptions {
  text: string;
  signal?: AbortSignal;
  /** 优先于环境变量 `MINIMAX_VOICE_ID`（例如音色复刻得到的 `voice_id`） */
  voiceId?: string | null;
}

export async function synthesizeMinimaxTtsToMp3Blob(options: SynthesizeMinimaxOptions): Promise<Blob> {
  const apiKey = getMinimaxApiKey();
  if (!apiKey) throw new Error('MINIMAX_API_KEY is not configured');
  warnMinimaxTokenPlanKeyInDev(apiKey);

  const model =
    (typeof process !== 'undefined' && process.env.MINIMAX_TTS_MODEL) || 'speech-2.8-hd';
  const envVoice =
    typeof process !== 'undefined' && process.env.MINIMAX_VOICE_ID
      ? process.env.MINIMAX_VOICE_ID.trim()
      : '';
  // 默认：官方系统音色「磁性男声」；中文为主可在 .env 设 Chinese (Mandarin)_Male_Announcer
  const voiceId =
    (options.voiceId && String(options.voiceId).trim()) ||
    envVoice ||
    'English_magnetic_voiced_man';
  const speedRaw =
    typeof process !== 'undefined' && process.env.MINIMAX_TTS_SPEED != null
      ? Number(process.env.MINIMAX_TTS_SPEED)
      : 1;
  const speed = Number.isFinite(speedRaw) && speedRaw > 0 ? Math.min(2, speedRaw) : 1;

  const root = resolveMinimaxApiRoot();
  const url = withMinimaxGroupQuery(`${root}/v1/t2a_v2`);

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      text: options.text,
      stream: false,
      language_boost: 'auto',
      voice_setting: {
        voice_id: voiceId,
        speed,
        vol: 1,
        pitch: 0,
      },
      audio_setting: {
        sample_rate: 32000,
        bitrate: 128000,
        format: 'mp3',
        channel: 1,
      },
    }),
    signal: options.signal,
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`MiniMax HTTP ${res.status}: ${t.slice(0, 240)}`);
  }

  const json = (await res.json()) as MinimaxT2aResponse;
  const code = json.base_resp?.status_code;
  if (code !== 0 && code !== undefined) {
    throw new Error(json.base_resp?.status_msg || `MiniMax status_code=${code}`);
  }
  const hex = json.data?.audio;
  if (!hex || typeof hex !== 'string') {
    throw new Error('MiniMax response missing data.audio');
  }
  const bytes = hexToUint8Array(hex);
  return new Blob([bytes], { type: 'audio/mpeg' });
}
