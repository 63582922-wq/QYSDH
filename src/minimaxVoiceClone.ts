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

import { runtimeMinimaxTrimmed } from './appClientEnv';
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

/** 多条复刻音色档案（可选昵称），当前朗读使用的 id 仍由上面 key 指向「选中」项 */
export const MINIMAX_VOICE_PROFILES_KEY = 'subconscious_minimax_voice_profiles';

/** 与 localStorage 同步；同一标签页内刷新可回填（部分 WebView / 分区存储异常时略稳一点） */
const MINIMAX_VOICE_SESSION_BACKUP_ID = 'subconscious_minimax_voice_id_sess_bak';
const MINIMAX_VOICE_SESSION_BACKUP_PROFILES = 'subconscious_minimax_voice_profiles_sess_bak';

function mirrorMinimaxVoiceToSession(): void {
  try {
    const id = localStorage.getItem(MINIMAX_CLONED_VOICE_STORAGE_KEY);
    const prof = localStorage.getItem(MINIMAX_VOICE_PROFILES_KEY);
    if (id != null && String(id).trim() !== '') {
      sessionStorage.setItem(MINIMAX_VOICE_SESSION_BACKUP_ID, String(id).trim());
    } else {
      sessionStorage.removeItem(MINIMAX_VOICE_SESSION_BACKUP_ID);
    }
    if (prof != null && String(prof).trim() !== '') {
      sessionStorage.setItem(MINIMAX_VOICE_SESSION_BACKUP_PROFILES, prof);
    } else {
      sessionStorage.removeItem(MINIMAX_VOICE_SESSION_BACKUP_PROFILES);
    }
  } catch {
    /* ignore */
  }
}

/**
 * 页面加载时调用：若 localStorage 被清空但同标签页 session 仍有镜像，则写回 localStorage。
 * @returns 是否发生了恢复（便于 UI 刷新状态）
 */
export function ensureMinimaxVoicePersistenceLoaded(): boolean {
  try {
    const hasId = !!localStorage.getItem(MINIMAX_CLONED_VOICE_STORAGE_KEY)?.trim();
    const hasProf = !!localStorage.getItem(MINIMAX_VOICE_PROFILES_KEY)?.trim();
    if (hasId || hasProf) {
      mirrorMinimaxVoiceToSession();
      return false;
    }
    const bid = sessionStorage.getItem(MINIMAX_VOICE_SESSION_BACKUP_ID)?.trim();
    const bprof = sessionStorage.getItem(MINIMAX_VOICE_SESSION_BACKUP_PROFILES);
    if (!bid && (!bprof || String(bprof).trim() === '')) return false;
    if (bid) localStorage.setItem(MINIMAX_CLONED_VOICE_STORAGE_KEY, bid);
    if (bprof && String(bprof).trim() !== '') {
      localStorage.setItem(MINIMAX_VOICE_PROFILES_KEY, bprof);
    }
    mirrorMinimaxVoiceToSession();
    return true;
  } catch {
    return false;
  }
}

export type MinimaxVoiceProfile = {
  voiceId: string;
  /** 用户备注，默认可为空 */
  label?: string;
  createdAt: number;
};

function parseProfiles(raw: string | null): MinimaxVoiceProfile[] {
  if (!raw) return [];
  try {
    const p = JSON.parse(raw);
    if (!Array.isArray(p)) return [];
    return p
      .filter((x: unknown) => x && typeof (x as MinimaxVoiceProfile).voiceId === 'string')
      .map((x: MinimaxVoiceProfile) => ({
        voiceId: String(x.voiceId).trim(),
        label: typeof x.label === 'string' ? x.label.trim() || undefined : undefined,
        createdAt: typeof x.createdAt === 'number' ? x.createdAt : Date.now(),
      }))
      .filter((x) => x.voiceId.length > 0);
  } catch {
    return [];
  }
}

/** 已保存的复刻音色列表（最新在前）；会自动把旧版「仅单 id」迁移进列表 */
export function readVoiceProfiles(): MinimaxVoiceProfile[] {
  let list = parseProfiles(
    (() => {
      try {
        return localStorage.getItem(MINIMAX_VOICE_PROFILES_KEY);
      } catch {
        return null;
      }
    })(),
  );
  try {
    const legacy = localStorage.getItem(MINIMAX_CLONED_VOICE_STORAGE_KEY)?.trim();
    if (legacy && !list.some((p) => p.voiceId === legacy)) {
      list = [{ voiceId: legacy, createdAt: Date.now() }, ...list];
    }
  } catch {
    /* ignore */
  }
  const seen = new Set<string>();
  return list.filter((p) => {
    if (seen.has(p.voiceId)) return false;
    seen.add(p.voiceId);
    return true;
  });
}

export function writeVoiceProfiles(profiles: MinimaxVoiceProfile[]): void {
  try {
    localStorage.setItem(MINIMAX_VOICE_PROFILES_KEY, JSON.stringify(profiles));
    mirrorMinimaxVoiceToSession();
  } catch {
    /* ignore */
  }
}

/** 复刻成功或手动导入：写入列表并设为当前朗读音色 */
export function upsertVoiceProfile(voiceId: string, label?: string): void {
  const id = voiceId.trim();
  if (!id) return;
  const rest = readVoiceProfiles().filter((p) => p.voiceId !== id);
  const next: MinimaxVoiceProfile[] = [
    { voiceId: id, label: label?.trim() || undefined, createdAt: Date.now() },
    ...rest,
  ];
  writeVoiceProfiles(next);
  writeStoredMinimaxClonedVoiceId(id);
}

/** 从列表移除；若删掉的是当前选中，则自动选中列表第一条或清空（退回 env 默认音色） */
export function removeVoiceProfile(voiceId: string): void {
  const id = voiceId.trim();
  if (!id) return;
  const next = readVoiceProfiles().filter((p) => p.voiceId !== id);
  writeVoiceProfiles(next);
  const active = readStoredMinimaxClonedVoiceId()?.trim();
  if (active === id) {
    writeStoredMinimaxClonedVoiceId(next[0]?.voiceId ?? null);
  }
}

export function setActiveVoiceProfileId(voiceId: string | null): void {
  writeStoredMinimaxClonedVoiceId(voiceId?.trim() || null);
}

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
    mirrorMinimaxVoiceToSession();
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
  return String(id);
}

/** voice_clone 文档要求 file_id 为 int64；纯数字字符串在 JSON 中应发 number，避免 invalid params */
function normalizeVoiceCloneFileIdForJsonBody(id: string | number): string | number {
  if (typeof id === 'number' && Number.isFinite(id) && Number.isInteger(id)) {
    return id;
  }
  const s = String(id).trim();
  if (/^\d+$/.test(s)) {
    const n = Number(s);
    if (Number.isSafeInteger(n)) return n;
  }
  return id;
}

export async function requestMinimaxVoiceClone(params: {
  /** 上传接口返回的 file_id（在请求体会规范为 number，若超出安全整数则保持原样） */
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
  /** 与 T2A 默认解耦：voice_clone 常用 speech-02-hd；勿沿用 speech-2.8-hd 以免接口报 invalid params */
  const model =
    params.model ||
    runtimeMinimaxTrimmed('MINIMAX_VOICE_CLONE_MODEL') ||
    (typeof process !== 'undefined' && process.env.MINIMAX_VOICE_CLONE_MODEL) ||
    'speech-02-hd';
  const text =
    params.previewText ||
    '你好，这是用你的声音复刻的效果。Hello, this is a preview of your cloned voice.';

  const file_id = normalizeVoiceCloneFileIdForJsonBody(params.sourceFileId);

  const res = await fetch(withMinimaxGroupQuery(`${root}/v1/voice_clone`), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_id,
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
