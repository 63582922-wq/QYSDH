/// <reference types="vite/client" />

/**
 * MiniMax 快速复刻 → TTS 全流程（与本仓库实现一致）：
 *
 * 1. `POST /v1/files/upload`（purpose=voice_clone）→ 得到临时 **`file_id`**（上传素材，非最终音色名）。
 * 2. 本应用在本地生成符合官方规则的 **`voice_id`**（`newClonedVoiceId()`，如 `sub_` + 随机串），
 *    再 `POST /v1/voice_clone`，请求体带上 `file_id` + 该 **`voice_id`** + 预览文案等 → 在云端**注册**这把克隆音色。
 *    官方响应里**不会**再返回一个新的 voice_id，只会可选返回 **`demo_audio`** 预览 URL；之后 T2A 用的就是你在第 2 步传入的同一个 **`voice_id`**。
 * 3. 把该 **`voice_id`** 写入 `localStorage`，`ChatOverlay` 里 `synthesizeMinimaxTtsToMp3Blob({ voiceId })` 走 `POST /v1/t2a_v2`，与 `MINIMAX_VOICE_ID` 二选一优先用复刻 id。
 *
 * @see https://platform.minimax.io/docs/guides/speech-voice-clone
 * @see https://platform.minimax.io/docs/api-reference/voice-cloning-clone
 */

import {
  getMinimaxApiKey,
  resolveMinimaxApiRoot,
  warnMinimaxTokenPlanKeyInDev,
  withMinimaxGroupQuery,
} from './minimaxTts';

/** 供 App / ChatOverlay 判断「是否配置了可用 Key」（含 trim、去引号） */
export { getMinimaxApiKey } from './minimaxTts';

/** 与官方说明一致：https://platform.minimax.io/docs/guides/speech-voice-clone */

export const MINIMAX_CLONE_MIN_DURATION_SEC = 10;
export const MINIMAX_CLONE_MAX_DURATION_SEC = 5 * 60;
export const MINIMAX_CLONE_MAX_FILE_BYTES = 20 * 1024 * 1024;

/** 与 ChatOverlay / App 设置共用，供自动朗读 MiniMax TTS 读取 */
export const MINIMAX_CLONED_VOICE_STORAGE_KEY = 'subconscious_minimax_cloned_voice_id';

export function readStoredMinimaxClonedVoiceId(): string | null {
  try {
    return localStorage.getItem(MINIMAX_CLONED_VOICE_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function writeStoredMinimaxClonedVoiceId(id: string | null): void {
  try {
    const t = id?.trim();
    if (t) localStorage.setItem(MINIMAX_CLONED_VOICE_STORAGE_KEY, t);
    else localStorage.removeItem(MINIMAX_CLONED_VOICE_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export function validateCloneAudioFile(file: File, lang: 'zh' | 'en'): string | null {
  if (file.size > MINIMAX_CLONE_MAX_FILE_BYTES) {
    return lang === 'zh' ? '文件超过 20MB。' : 'File exceeds 20MB.';
  }
  const byName = /\.(mp3|m4a|wav|aac)$/i.test(file.name);
  /** m4a 在部分浏览器/系统上为 audio/mp4、audio/quicktime、audio/aac 等，需放宽否则误拒 */
  const byType =
    /audio\/(mpeg|wav|wave|x-wav|mp4|x-m4a|m4a|aac|quicktime|3gpp|3gpp2)/i.test(file.type) ||
    /video\/(mp4|quicktime)/i.test(file.type) ||
    file.type === '';
  if (!byName && !byType) {
    return lang === 'zh' ? '请使用 mp3、m4a 或 wav。' : 'Use mp3, m4a, or wav.';
  }
  return null;
}

export function getAudioFileDurationSec(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const audio = document.createElement('audio');
    audio.preload = 'metadata';
    const cleanup = () => {
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* ignore */
      }
    };
    audio.onloadedmetadata = () => {
      const d = audio.duration;
      cleanup();
      if (!Number.isFinite(d) || d <= 0) {
        reject(new Error('invalid_duration'));
        return;
      }
      resolve(d);
    };
    audio.onerror = () => {
      cleanup();
      reject(new Error('audio_load_error'));
    };
    audio.src = url;
  });
}

export async function uploadMinimaxVoiceCloneSource(file: File, signal?: AbortSignal): Promise<string> {
  const apiKey = getMinimaxApiKey();
  if (!apiKey) throw new Error('MINIMAX_API_KEY is not configured');
  warnMinimaxTokenPlanKeyInDev(apiKey);

  const root = resolveMinimaxApiRoot();
  const fd = new FormData();
  fd.append('purpose', 'voice_clone');
  fd.append('file', file, file.name);

  const res = await fetch(withMinimaxGroupQuery(`${root}/v1/files/upload`), {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: fd,
    signal,
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`upload ${res.status}: ${t.slice(0, 240)}`);
  }

  const json = (await res.json()) as {
    /** OpenAPI 为 int64；部分环境可能序列化为 string */
    file?: { file_id?: string | number };
    base_resp?: { status_code?: number; status_msg?: string };
  };
  const code = json.base_resp?.status_code;
  if (code !== undefined && code !== 0) {
    throw new Error(json.base_resp?.status_msg || `upload status_code=${code}`);
  }
  const id = json.file?.file_id;
  if (id === undefined || id === null || id === '') {
    throw new Error('upload response missing file.file_id');
  }
  return id;
}

export async function requestMinimaxVoiceClone(params: {
  /** 上传接口返回的 file_id（数值或字符串原样传给 JSON） */
  sourceFileId: string | number;
  voiceId: string;
  previewText?: string;
  model?: string;
  signal?: AbortSignal;
}): Promise<void> {
  const apiKey = getMinimaxApiKey();
  if (!apiKey) throw new Error('MINIMAX_API_KEY is not configured');
  warnMinimaxTokenPlanKeyInDev(apiKey);

  const root = resolveMinimaxApiRoot();
  const model =
    params.model ||
    (typeof process !== 'undefined' && process.env.MINIMAX_TTS_MODEL?.trim()) ||
    'speech-2.8-hd';
  const text =
    params.previewText ||
    '你好，这是用你的声音复刻的效果。Hello, this is a preview of your cloned voice.';

  const res = await fetch(withMinimaxGroupQuery(`${root}/v1/voice_clone`), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_id: params.sourceFileId,
      voice_id: params.voiceId,
      text,
      model,
    }),
    signal: params.signal,
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`voice_clone ${res.status}: ${t.slice(0, 240)}`);
  }

  const json = (await res.json()) as {
    /** 预览音频 URL；与最终可用的 voice_id 无关 */
    demo_audio?: string;
    base_resp?: { status_code?: number; status_msg?: string };
  };
  const code = json.base_resp?.status_code;
  if (code !== undefined && code !== 0) {
    throw new Error(json.base_resp?.status_msg || `voice_clone status_code=${code}`);
  }
}

/**
 * 生成将传给 `voice_clone` 的自定义 voice_id（注册成功后 T2A 也用同一字符串）。
 * 官方约束：长度 [8,256]、以英文字母开头、仅字母数字与 `-` `_`、不得以 `-` `_` 结尾、勿与已有 id 重复。
 */
export function newClonedVoiceId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `sub_${crypto.randomUUID().replace(/-/g, '')}`;
  }
  return `sub_${Date.now()}_${Math.random().toString(16).slice(2, 14)}`;
}
