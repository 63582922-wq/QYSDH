import React, { useState, useEffect, useRef, useMemo, useCallback, forwardRef, useImperativeHandle } from 'react';
import { createPortal } from 'react-dom';
import { getAppPortalNode } from '../portalRoot';
import { Send, Loader2, Mic, MicOff, Dna, RefreshCw, Save, Trash2 } from 'lucide-react';
import { GoogleGenAI, Type, LiveServerMessage, Modality } from '@google/genai';
import { getGeminiApiKey, getGeminiLiveSpeechVoiceNameFromRuntimeOrDefine } from '../appClientEnv';
import { stripReasoningArtifacts, stripTextForTts, synthesizeMinimaxTtsToMp3Blob } from '../minimaxTts';
import { getMinimaxApiKey, readStoredMinimaxClonedVoiceId } from '../minimaxVoiceClone';

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  /** 复刻模式「镌刻记忆」成功后追加的知心结语，仅存档与预览；不参与实时字幕与自动朗读 */
  kind?: 'closing_note';
  /** 历史会话可能残留；Live/复刻文本请求可附带画布照片（见 sendMessage inlineData） */
  isInitialImage?: boolean;
  attachedImageDataUrl?: string;
}

export interface ChatSession {
  id: string;
  theme: string;
  messages: ChatMessage[];
  updatedAt: number;
  /** 历史封面：由当前画布图压缩生成，便于列表/轮播展示（消息内附图仍会剥离以省配额） */
  thumbnailDataUrl?: string;
  /** 在「保存对话 → 镌刻记忆」中确认后写入，用于回忆册粒子展示 */
  etchedToAlbum?: boolean;
  etchedAt?: number;
}

export type ConversationMode = 'live' | 'text_clone';

interface ChatOverlayProps {
  currentImageDataUrl: string | null;
  isOpen: boolean;
  onClose: () => void;
  language: 'zh' | 'en';
  onToggleLanguage: () => void;
  aiName: string;
  onSpeechValue?: (val: number) => void;
  isAutoSpeak: boolean;
  setIsAutoSpeak: (val: boolean) => void;
  conversationMode: ConversationMode;
  /** App 设置抽屉打开时隐藏底部 fixed 输入条/工具栏，避免挡住设置面板 */
  settingsChromeOpen?: boolean;
  /** 消散重置（与字幕区并排，在右侧图标列） */
  onDissolveReset?: () => void;
  /** 打开保存对话预览 */
  onOpenSavePreview?: () => void;
}

export interface ChatOverlayHandle {
  openSavePreview: () => void;
  openHistory: () => void;
}

/**
 * SDK 默认 baseUrl 带尾部 `/`，Live 侧拼 `${wsBase}/ws/...` 会得到 `…googleapis.com//ws/…`，握手可能失败。
 * 使用无尾部斜杠的 origin。
 *
 * 懒创建：生产环境由 `serve-render-dist` 在 HTML 最前注入 `__RUNTIME_APP_ENV__`，若模块早于注入执行，
 * 首次会得到空 Key；注入后再次调用须换新实例（按 key 缓存失效）。
 */
let geminiClientCache: GoogleGenAI | null = null;
let geminiClientCacheKey = '';
function getGeminiClient(): GoogleGenAI {
  const k = getGeminiApiKey();
  if (geminiClientCache && geminiClientCacheKey === k) return geminiClientCache;
  geminiClientCacheKey = k;
  geminiClientCache = new GoogleGenAI({
    apiKey: k,
    httpOptions: {
      baseUrl: 'https://generativelanguage.googleapis.com',
    },
  });
  return geminiClientCache;
}

if (!getGeminiApiKey()) {
  console.error('CRITICAL: GEMINI_API_KEY is not defined in environment!');
}

/**
 * 文本多轮（含复刻）、主题、镌刻等；主模型 503 时按序尝试备用。
 * 注意：`gemini-1.5-flash` 无版本后缀在 v1beta 上常直接 404（并非没开通），须用 `…-8b` / `…-002` 等 ListModels 里可见的全名。
 * `gemini-2.0-flash` 对部分密钥/区域也可能 404，故优先 2.5 同族与 1.5 带版本号、再 3.x 预览。
 * @see https://ai.google.dev/gemini-api/docs/models
 */
const TEXT_MODEL_ID = 'gemini-2.5-flash';
const TEXT_MODEL_CANDIDATES = [
  TEXT_MODEL_ID,
  'gemini-2.5-flash-lite',
  'gemini-1.5-flash-8b',
  'gemini-1.5-flash-002',
  'gemini-3-flash-preview',
] as const;

/**
 * 固定：Gemini Multimodal Live（bidi），与 AI Studio「Gemini 3 Flash Live」一致。
 * @see https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview
 */
const LIVE_VOICE_MODEL_ID = 'gemini-3.1-flash-live-preview';

/** 会话内覆盖 Live 音色（优先于 .env）；不设则用环境默认 */
const LIVE_VOICE_SESSION_STORAGE_KEY = 'subconscious_gemini_live_voice_override';

/** Live：麦克风 RMS 仅用于「是否在说话」门控（语谱走 AnalyserNode） */
const LIVE_MIC_RMS_THRESHOLD = 0.013;
const LIVE_VOICE_TAIL_MS = 420;
/** Live 上行音频：与旧 ScriptProcessor 一致按固长度打包再 send，避免 Worklet 128 帧一包刷爆 WebSocket 导致 1006 */
const LIVE_MIC_WEBSOCKET_SAMPLES = 4096;
/** Gemini Live 声明的 PCM 采样率；须与 `sendRealtimeInput` 的 mimeType 及 payload 一致 */
const LIVE_GEMINI_INPUT_RATE = 16000;
/** 1006/1011 无干净关闭帧时自动重连次数上限（单次点麦克风会话内） */
const LIVE_WS_ABNORMAL_MAX_AUTO_RETRIES = 2;

/**
 * 线性重采样到固定输出长度（mic 块）：用于 iOS/WebKit 实际 44.1k/48k 而 API 要求 16k 的场景。
 */
function resampleFloat32ToOutLength(input: Float32Array, outLen: number): Float32Array {
  if (outLen <= 0) return new Float32Array(0);
  if (input.length === 0) return new Float32Array(outLen);
  if (input.length === 1) {
    const v = input[0]!;
    const out = new Float32Array(outLen);
    out.fill(v);
    return out;
  }
  const out = new Float32Array(outLen);
  const imax = input.length - 1;
  for (let i = 0; i < outLen; i++) {
    const t = outLen === 1 ? 0 : (i / (outLen - 1)) * imax;
    const i0 = Math.min(imax - 1, Math.floor(t));
    const frac = t - i0;
    const a = input[i0]!;
    const b = input[i0 + 1]!;
    out[i] = a * (1 - frac) + b * frac;
  }
  return out;
}

/** 得到 outSamples 个 16k 样本时，在 fromRate 下至少需要的源样本数 */
function liveSourceSamplesFor16kBlock(fromRate: number, outSamples: number): number {
  if (!Number.isFinite(fromRate) || fromRate <= 0) return Math.max(2, outSamples);
  return Math.max(2, Math.round((outSamples * fromRate) / LIVE_GEMINI_INPUT_RATE));
}

/** 「我说」声纹：多层平滑正弦 + lighter 叠亮 + 中心包络（对齐参考图丝带光感） */
const VOICEPRINT_CANVAS_W = 320;
const VOICEPRINT_CANVAS_H = 52;

/** 中间密、两侧收：能量集中在画面中部 */
function voiceprintCenterEnvelope(fx: number): number {
  const d = Math.abs(fx - 0.5) * 2;
  return Math.pow(Math.max(0, 1 - d), 1.35);
}

/** 单层波形：多频正弦叠加，柔顺丝线感 */
function voiceprintLayerWave(
  fx: number,
  L: number,
  layerCount: number,
  timeMs: number,
): number {
  const depth = L / Math.max(1, layerCount - 1);
  const phase = timeMs * 0.00135 + L * 1.07;
  const f0 = 9.5 + depth * 6.2;
  const f1 = 21 + depth * 4.1;
  const f2 = 37 + L * 1.8;
  const x = fx * Math.PI * 2;
  return (
    0.5 * Math.sin(x * f0 + phase) +
    0.32 * Math.sin(x * f1 + phase * 1.31 + 0.7) +
    0.22 * Math.sin(x * f2 - phase * 0.85 + L * 0.4)
  );
}

function drawRibbonVoiceprint(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  freqBytes: Uint8Array,
  micEnergy01: number,
  timeMs: number,
  accent: 'cyan' | 'rose' = 'cyan',
) {
  if (!freqBytes?.length || width <= 0 || height <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  const cy = height * 0.5;
  const nBins = freqBytes.length;
  /** 层数少一些、采样疏一些，避免 lighter 叠成一团糊 */
  const layerCount = 9;
  /** 静音更暗、有声更亮，拉高对比 */
  const brightness = 0.36 + micEnergy01 * 1.05;

  ctx.globalCompositeOperation = 'lighter';
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  for (let L = 0; L < layerCount; L++) {
    const depth = L / Math.max(1, layerCount - 1);
    const phaseDrift = L * 0.22;
    const alpha = Math.min(0.24, (0.028 + depth * 0.052) * brightness);
    const lineW = 0.62 + depth * 0.34;
    const glow = 0.9 + depth * 1.45 + micEnergy01 * 4.2;

    ctx.beginPath();
    let moved = false;
    for (let x = 0; x <= width; x += 2.35) {
      const fx = x / width;
      const bin = Math.min(nBins - 1, Math.floor(fx * nBins * 0.92));
      const band = freqBytes[bin]! / 255;
      const shell = voiceprintCenterEnvelope(fx);
      const drive = Math.pow(0.05 + band * 1.05 + micEnergy01 * 0.88, 1.04);
      const amp = height * 0.52 * shell * drive;
      const wv = voiceprintLayerWave(fx, L, layerCount, timeMs + phaseDrift * 1000);
      const y = cy + wv * amp;
      if (!moved) {
        ctx.moveTo(x, y);
        moved = true;
      } else ctx.lineTo(x, y);
    }

    if (accent === 'rose') {
      ctx.strokeStyle = `rgba(251, 113, 133, ${alpha})`;
      ctx.shadowColor = 'rgba(244, 63, 94, 0.52)';
    } else {
      ctx.strokeStyle = `rgba(0, 238, 255, ${alpha})`;
      ctx.shadowColor = 'rgba(0, 220, 255, 0.55)';
    }
    ctx.lineWidth = lineW;
    ctx.shadowBlur = glow;
    ctx.stroke();
  }

  ctx.shadowBlur = 0;
  ctx.beginPath();
  let moved = false;
  for (let x = 0; x <= width; x += 2.1) {
    const fx = x / width;
    const bin = Math.min(nBins - 1, Math.floor(fx * nBins * 0.92));
    const band = freqBytes[bin]! / 255;
    const shell = voiceprintCenterEnvelope(fx);
    const drive = Math.pow(0.04 + band * 1.12 + micEnergy01 * 0.92, 1.02);
    const amp = height * 0.44 * shell * drive;
    const wv = voiceprintLayerWave(fx, Math.floor(layerCount / 2), layerCount, timeMs);
    const y = cy + wv * amp;
    if (!moved) {
      ctx.moveTo(x, y);
      moved = true;
    } else ctx.lineTo(x, y);
  }
  const fgAlpha = Math.min(0.58, (0.045 + micEnergy01 * 0.42) * (0.55 + brightness * 0.65));
  if (accent === 'rose') {
    ctx.strokeStyle = `rgba(255, 241, 242, ${fgAlpha})`;
    ctx.shadowColor = `rgba(251, 113, 133, ${0.38 + micEnergy01 * 0.55})`;
  } else {
    ctx.strokeStyle = `rgba(230, 255, 255, ${fgAlpha})`;
    ctx.shadowColor = `rgba(120, 245, 255, ${0.4 + micEnergy01 * 0.52})`;
  }
  ctx.lineWidth = 0.88 + micEnergy01 * 0.35;
  ctx.shadowBlur = 2.2 + micEnergy01 * 8;
  ctx.stroke();

  ctx.restore();
}

/**
 * Gemini `prebuiltVoiceConfig.voiceName` 全量预设（与官方 TTS / Voice Library 同源 30 个）。
 * @see https://ai.google.dev/gemini-api/docs/speech-generation#voices
 * Live 模型可能仍忽略个别名；`.env` 自定义名也会并入下拉。
 */
const LIVE_PREBUILT_VOICE_OPTIONS: readonly string[] = [
  'Achernar',
  'Achird',
  'Algenib',
  'Algieba',
  'Alnilam',
  'Aoede',
  'Autonoe',
  'Callirrhoe',
  'Charon',
  'Despina',
  'Enceladus',
  'Erinome',
  'Fenrir',
  'Gacrux',
  'Iapetus',
  'Kore',
  'Laomedeia',
  'Leda',
  'Orus',
  'Puck',
  'Pulcherrima',
  'Rasalgethi',
  'Sadachbia',
  'Sadaltager',
  'Schedar',
  'Sulafat',
  'Umbriel',
  'Vindemiatrix',
  'Zephyr',
  'Zubenelgenubi',
];

/** 仅 .env / define，不含 session 覆盖（用于「默认」标签文案） */
function getGeminiLiveSpeechVoiceNameFromEnvOnly(): string {
  const fromRt = getGeminiLiveSpeechVoiceNameFromRuntimeOrDefine();
  if (fromRt.length > 0) return fromRt;
  const fromVite = import.meta.env.VITE_GEMINI_LIVE_SPEECH_VOICE_NAME;
  if (fromVite != null && String(fromVite).trim().length > 0) {
    return String(fromVite).trim();
  }
  if (typeof process !== 'undefined' && process.env.GEMINI_LIVE_SPEECH_VOICE_NAME != null) {
    const t = String(process.env.GEMINI_LIVE_SPEECH_VOICE_NAME).trim();
    if (t.length > 0) return t;
  }
  return 'Orus';
}

/**
 * Live 回复音色：session 覆盖 → `VITE_GEMINI_LIVE_SPEECH_VOICE_NAME` → `GEMINI_LIVE_SPEECH_VOICE_NAME` → Orus。
 * 改 .env 后需重启 dev；会话内可用界面下拉覆盖（无需改文件）。
 */
function resolveGeminiLiveSpeechVoiceName(): string {
  try {
    const o = sessionStorage.getItem(LIVE_VOICE_SESSION_STORAGE_KEY)?.trim();
    if (o && o.length > 0) return o;
  } catch {
    /* private mode */
  }
  return getGeminiLiveSpeechVoiceNameFromEnvOnly();
}

function getLiveVoiceSelectControlValue(): string {
  try {
    const o = sessionStorage.getItem(LIVE_VOICE_SESSION_STORAGE_KEY)?.trim();
    return o && o.length > 0 ? o : '__DEFAULT__';
  } catch {
    return '__DEFAULT__';
  }
}

function liveVoiceOptionsForSelect(): string[] {
  const eff = resolveGeminiLiveSpeechVoiceName();
  const set = new Set<string>([...LIVE_PREBUILT_VOICE_OPTIONS, eff, getGeminiLiveSpeechVoiceNameFromEnvOnly()]);
  return Array.from(set).sort((a, b) => a.localeCompare(b));
}

/** 移动端网络慢时避免永远卡在「等待回复」 */
const CHAT_SEND_TIMEOUT_MS = 70000;

/** 过长回复写入 state / localStorage 时截断，避免撑爆内存 */
const MAX_REPLY_CHARS = 8000;
/** 悬浮气泡内最长展示字符，超出用省略（全文仍在会话里） */
const MAX_TTS_CHARS = 4000;
/** 复刻自动朗读结束后，字幕对话框再停留时长（约 1～2 秒） */
const CLONE_TTS_POST_SPEECH_HOLD_MS = 1600;
/** 复刻语音输入完成发送后，「我说」卡片再停留约两秒再收起（仅保留此框内文字，不再在上方重复气泡） */
const CLONE_USER_POST_DICTATION_HOLD_MS = 2000;

/** Live 回复 PCM 播放略放慢，减轻「赶」的感觉（略降音高，一般可接受） */
const LIVE_AUDIO_PLAYBACK_RATE = 0.9;

const isModelNotFoundError = (error: any) => {
  const message = String(error?.message || '').toLowerCase();
  return message.includes('not found') || message.includes('is not supported');
};

/**
 * `@google/genai` 的 `ApiError` 带 `status`（HTTP 码）；`message` 也可能是整段 JSON 字符串。
 * 任一路径取到 4xx/5xx 即返回，供过载时换模型 / 退避重试。
 */
function extractGeminiHttpStatus(error: unknown): number | undefined {
  const x = error as { status?: number; code?: number; message?: string };
  if (typeof x.status === 'number' && x.status >= 400) return x.status;
  if (typeof x.code === 'number' && x.code >= 400 && x.code < 600) return x.code;
  const raw = String(x.message ?? error ?? '');
  try {
    const j = JSON.parse(raw) as { error?: { code?: number } };
    const c = j?.error?.code;
    if (typeof c === 'number' && c >= 400 && c < 600) return c;
  } catch {
    /* message 非纯 JSON 时走正则 */
  }
  const m = /"code"\s*:\s*(\d{3})\b/.exec(raw);
  if (m) return Number(m[1]);
  return undefined;
}

/** 同一模型上瞬时 503 时的退避次数（仍会再换候选模型） */
const GEMINI_MODEL_ATTEMPTS = 2;
/** 换模型前多等一会，减轻对同一 endpoint 的连打 */
const GEMINI_MODEL_SWITCH_GAP_MS = 1200;
/** 429 / 配额类：通常按 API Key 限流，换模型会继续打满配额；只在当前模型上长退避重试，然后结束 */
const GEMINI_RATE_LIMIT_MAX_TRIES = 4;
const GEMINI_RATE_LIMIT_BACKOFF_MS = [2800, 7000, 14000, 20000] as const;

function sleepMs(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}

/** 可换下一候选模型重试：404 类 + 瞬时过载/配额（503/429 等），避免一次 UNAVAILABLE 就整轮失败 */
function isGeminiModelFallbackError(error: unknown): boolean {
  if (String((error as { message?: string })?.message) === 'CLIENT_TIMEOUT') return true;

  const http = extractGeminiHttpStatus(error);
  if (http === 503 || http === 429 || http === 404) return true;

  if (isModelNotFoundError(error)) return true;

  const x = error as { message?: string; error?: { message?: string } };
  const raw = String(x?.message ?? x?.error?.message ?? error ?? '');
  try {
    const j = JSON.parse(raw) as { error?: { code?: number } };
    const c = j?.error?.code;
    if (c === 503 || c === 429) return true;
  } catch {
    /* 非 JSON */
  }
  const u = raw.toUpperCase();
  return (
    u.includes('UNAVAILABLE') ||
    u.includes('HIGH DEMAND') ||
    u.includes('RESOURCE_EXHAUSTED') ||
    u.includes('"CODE":503') ||
    u.includes('"CODE":429')
  );
}

/** 与 503 区分：限流/配额在同一密钥下换模型往往无效，不应快速遍历候选模型 */
function isLikelyGeminiRateLimitError(error: unknown): boolean {
  const http = extractGeminiHttpStatus(error);
  if (http === 429) return true;
  const raw = String((error as { message?: string })?.message ?? error ?? '').toUpperCase();
  return (
    raw.includes('RESOURCE_EXHAUSTED') ||
    raw.includes('TOO_MANY_REQUESTS') ||
    raw.includes('"CODE":429') ||
    raw.includes('429 TOO MANY REQUESTS')
  );
}

/** 免费层「每模型每日」等：再请求只会重复 429，应直接失败并提示用户 */
function isGeminiDailyQuotaExhausted(error: unknown): boolean {
  const raw = String((error as { message?: string })?.message ?? error ?? '');
  return (
    /GenerateRequestsPerDay/i.test(raw) ||
    /PerDayPerProjectPerModel/i.test(raw) ||
    /per day per project per model/i.test(raw)
  );
}

/**
 * Google 常在 429 JSON 里带 `Please retry in 12.3s` 或 RetryInfo `retryDelay`；应至少等到该时间再重试。
 */
function extractGeminiSuggestedRetryDelayMs(error: unknown): number | null {
  const raw = String((error as { message?: string })?.message ?? error ?? '');
  const m1 = /Please retry in ([\d.]+)\s*s\b/i.exec(raw);
  if (m1) {
    const sec = Number(m1[1]);
    if (Number.isFinite(sec) && sec > 0) return Math.min(120_000, Math.ceil(sec * 1000) + 600);
  }
  const m2 = /"retryDelay"\s*:\s*"(\d+)s"/i.exec(raw);
  if (m2) {
    const sec = Number(m2[1]);
    if (Number.isFinite(sec) && sec > 0) return Math.min(120_000, sec * 1000 + 600);
  }
  return null;
}

/**
 * 文本模型调用：503/404 等可换模型；429 仅在同一模型上退避，避免连打多个 endpoint 加重限流。
 */
async function geminiTextModelFallbackLoop<T>(params: {
  candidates: readonly string[];
  run: (model: string) => Promise<T>;
  onSuccessModel: (model: string) => void;
}): Promise<T> {
  const { candidates, run, onSuccessModel } = params;
  let lastError: unknown = null;
  outer: for (let i = 0; i < candidates.length; i += 1) {
    const model = candidates[i]!;
    let attempt = 0;
    while (true) {
      try {
        const r = await run(model);
        onSuccessModel(model);
        return r;
      } catch (e) {
        lastError = e;
        if (!isGeminiModelFallbackError(e)) throw e;
        if (isLikelyGeminiRateLimitError(e)) {
          if (isGeminiDailyQuotaExhausted(e)) throw e;
          if (attempt >= GEMINI_RATE_LIMIT_MAX_TRIES - 1) break outer;
          const hintMs = extractGeminiSuggestedRetryDelayMs(e);
          const floorMs = GEMINI_RATE_LIMIT_BACKOFF_MS[attempt] ?? 20000;
          await sleepMs(Math.max(floorMs, hintMs ?? 0));
          attempt += 1;
          continue;
        }
        if (attempt >= GEMINI_MODEL_ATTEMPTS - 1) break;
        await sleepMs(1200 + attempt * 900);
        attempt += 1;
        continue;
      }
    }
    if (lastError != null && isLikelyGeminiRateLimitError(lastError)) break outer;
    if (i < candidates.length - 1) await sleepMs(GEMINI_MODEL_SWITCH_GAP_MS);
  }
  throw lastError ?? new Error('No available text model');
}

/** Short user-facing text; avoids dumping huge JSON (e.g. quota / 429). */

/** 复刻对话随轮次附图：仅支持 data URL → Gemini inlineData */
function parseDataUrlForGeminiInline(dataUrl: string): { mimeType: string; data: string } | null {
  const t = dataUrl.trim();
  const m = /^data:([^;]+);base64,([\s\S]+)$/i.exec(t);
  if (!m) return null;
  const mimeType = (m[1] || 'image/jpeg').trim() || 'image/jpeg';
  const data = m[2].replace(/\s/g, '');
  if (!data) return null;
  return { mimeType, data };
}

/** 会话消息 / API 边缘情况：避免渲染期对非字符串 `.trim()` 抛错（移动端易触发 ErrorBoundary 黑屏） */
function safeMsgText(t: unknown): string {
  if (t == null) return '';
  if (typeof t === 'string') return t;
  if (typeof t === 'number' || typeof t === 'boolean') return String(t);
  return '';
}

/** 历史会话 theme 若存成对象/数组，直接 JSX 渲染会抛错（移动端 WebView 常见） */
function safeSessionTheme(t: unknown): string {
  return safeMsgText(t).trim().slice(0, 240);
}

function normalizeStoredChatMessage(m: unknown): ChatMessage | null {
  if (!m || typeof m !== 'object') return null;
  const o = m as Record<string, unknown>;
  if (o.role !== 'user' && o.role !== 'model') return null;
  const msg: ChatMessage = {
    role: o.role,
    text: typeof o.text === 'string' ? o.text : safeMsgText(o.text),
  };
  if (o.kind === 'closing_note') msg.kind = 'closing_note';
  if (typeof o.isInitialImage === 'boolean') msg.isInitialImage = o.isInitialImage;
  if (typeof o.attachedImageDataUrl === 'string') msg.attachedImageDataUrl = o.attachedImageDataUrl;
  return msg;
}

function normalizeStoredChatSession(item: unknown): ChatSession | null {
  if (!item || typeof item !== 'object') return null;
  const o = item as Record<string, unknown>;
  const id =
    typeof o.id === 'string'
      ? o.id
      : typeof o.id === 'number' && Number.isFinite(o.id)
        ? String(o.id)
        : '';
  if (!id) return null;
  const themeSt = safeSessionTheme(o.theme);
  const theme = themeSt.length > 0 ? themeSt : '…';
  const messages: ChatMessage[] = [];
  if (Array.isArray(o.messages)) {
    for (const row of o.messages) {
      const msg = normalizeStoredChatMessage(row);
      if (msg) messages.push(msg);
    }
  }
  const updatedAt =
    typeof o.updatedAt === 'number' && Number.isFinite(o.updatedAt) ? o.updatedAt : Date.now();
  const s: ChatSession = { id, theme, messages, updatedAt };
  if (typeof o.thumbnailDataUrl === 'string') s.thumbnailDataUrl = o.thumbnailDataUrl;
  if (o.etchedToAlbum === true) {
    s.etchedToAlbum = true;
    s.etchedAt =
      typeof o.etchedAt === 'number' && Number.isFinite(o.etchedAt) ? o.etchedAt : Date.now();
  }
  return s;
}

/** 避免 localStorage 损坏或非法类型导致整页白屏 */
function parseStoredSessions(raw: string | null): ChatSession[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const out: ChatSession[] = [];
    for (const row of parsed) {
      const s = normalizeStoredChatSession(row);
      if (s) out.push(s);
    }
    return out;
  } catch {
    return [];
  }
}

function sanitizeModelReplyText(text: string): string {
  let s = stripReasoningArtifacts(typeof text === 'string' ? text : String(text ?? ''));
  s = s.replace(/\u0000/g, '');
  if (s.length > MAX_REPLY_CHARS) {
    return `${s.slice(0, MAX_REPLY_CHARS)}\n…`;
  }
  return s;
}

function stripAttachedImagesForStorage(list: ChatSession[]): ChatSession[] {
  return list.map((s) => ({
    ...s,
    messages: (s.messages || []).map(({ attachedImageDataUrl: _a, ...msg }) => msg),
  }));
}

const HISTORY_COVER_MAX_W = 480;
const HISTORY_COVER_JPEG = 0.52;
const HISTORY_CARD_W = 268;
const HISTORY_CARD_GAP = 30;

/** 从历史画布图生成较小 JPEG，写入 session 封面 */
function makeSessionCoverThumbnail(dataUrl: string): Promise<string | undefined> {
  const t = dataUrl.trim();
  if (!t.startsWith('data:image')) return Promise.resolve(undefined);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const scale = Math.min(1, HISTORY_COVER_MAX_W / Math.max(1, img.width));
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(undefined);
          return;
        }
        ctx.drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL('image/jpeg', HISTORY_COVER_JPEG));
      } catch {
        resolve(undefined);
      }
    };
    img.onerror = () => resolve(undefined);
    img.src = t;
  });
}

function pickLiveAudioBase64FromMessage(message: LiveServerMessage): string | undefined {
  const parts = message.serverContent?.modelTurn?.parts;
  if (!Array.isArray(parts)) return undefined;
  for (const part of parts) {
    const id = (part as { inlineData?: { data?: string; mimeType?: string } }).inlineData;
    if (id?.data && typeof id.data === 'string' && id.data.length > 0) {
      return id.data;
    }
  }
  return undefined;
}

/** `@google/genai` Session 底层连接：浏览器里是 BrowserWebSocket，原生 readyState 在 `conn.ws`；勿只读 `conn.readyState`（恒 undefined → 永远不发音频） */
function liveSessionSocketIsOpen(sess: { conn?: unknown } | null | undefined): boolean {
  const conn = sess?.conn as { readyState?: number; ws?: Pick<WebSocket, 'readyState'> } | undefined;
  if (conn == null) return false;
  const rs =
    typeof conn.readyState === 'number'
      ? conn.readyState
      : conn.ws != null && typeof conn.ws.readyState === 'number'
        ? conn.ws.readyState
        : undefined;
  return rs === WebSocket.OPEN;
}

type LiveServerContentExtras = {
  interrupted?: boolean;
  /** 模型一轮结束（不同版本字段名可能略有差异，多路径判断） */
  turnComplete?: boolean;
  modelTurn?: {
    turnComplete?: boolean;
    parts?: Array<{ text?: string; inlineData?: { data?: string } }>;
  };
  outputTranscription?: { text?: string };
  inputTranscription?: { text?: string };
};

type LiveCaptionTurn = { id: string; user: string; userAlt: string; ai: string; aiAlt: string };

function containsHanScript(s: string): boolean {
  return /[\u3400-\u9FFF]/.test(s);
}

/** 从 Live 消息里取字幕：优先 API 的 outputTranscription，其次 modelTurn 文本 */
function pickLiveCaptionTexts(message: LiveServerMessage): { ai: string | undefined; user: string | undefined } {
  const sc = message.serverContent as LiveServerContentExtras | undefined;
  if (!sc) return { ai: undefined, user: undefined };
  const user = typeof sc.inputTranscription?.text === 'string' ? sc.inputTranscription.text.trim() : undefined;
  let ai: string | undefined;
  const ot = sc.outputTranscription?.text;
  if (typeof ot === 'string' && ot.trim()) {
    ai = ot.trim();
  } else {
    const parts = sc.modelTurn?.parts;
    if (Array.isArray(parts)) {
      let t = '';
      for (const p of parts) {
        if (typeof p?.text === 'string') t += p.text;
      }
      ai = t.trim() || undefined;
    }
  }
  return { ai, user };
}

function isLiveModelTurnComplete(message: LiveServerMessage): boolean {
  const sc = message.serverContent as LiveServerContentExtras | undefined;
  if (!sc) return false;
  if (sc.turnComplete === true) return true;
  if (sc.modelTurn?.turnComplete === true) return true;
  return false;
}

/** 流式转写片段拼接处常带出重复空格 / 换行，展示前压成单一空格 */
function normalizeLiveCaptionWhitespace(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

/**
 * 合并流式转写：多数情况下服务端发「整段前缀变长」；少数发纯增量则拼接。
 * 目标：当前轮字幕始终显示**已收到的完整文本**，不丢前半句。
 */
function mergeStreamingTranscript(prev: string, incoming: string): string {
  const a = prev.trimEnd();
  const b = incoming.trim();
  let raw: string;
  if (!b) raw = a;
  else if (!a) raw = b;
  else if (b.startsWith(a)) raw = b;
  else if (a.startsWith(b)) raw = a;
  else if (a.endsWith(b)) raw = a;
  else {
    const tail = a.slice(-Math.min(48, a.length));
    if (tail && b.startsWith(tail)) raw = `${a.slice(0, a.length - tail.length)}${b}`.trim();
    else raw = `${a} ${b}`.trim();
  }
  return normalizeLiveCaptionWhitespace(raw);
}

function extractModelReplyText(response: unknown, fallback: string): string {
  const safeFallback =
    typeof fallback === 'string' && fallback.trim().length > 0
      ? (() => {
          try {
            return sanitizeModelReplyText(fallback.trim());
          } catch {
            return fallback.trim().slice(0, MAX_REPLY_CHARS);
          }
        })()
      : '…';

  const trySanitize = (raw: string): string | null => {
    const t = raw.trim();
    if (!t) return null;
    try {
      return sanitizeModelReplyText(t);
    } catch {
      return t.length > MAX_REPLY_CHARS ? `${t.slice(0, MAX_REPLY_CHARS)}…` : t;
    }
  };

  try {
    const top = (response as { text?: unknown })?.text;
    if (typeof top === 'string') {
      const out = trySanitize(top);
      if (out) return out;
    }
  } catch {
    /* GenerateContentResponse.text 在异常体上可能抛错（WebView 上更常见） */
  }

  try {
    const cands = (response as { candidates?: unknown[] })?.candidates;
    if (!Array.isArray(cands) || cands.length === 0) return safeFallback;
    const parts = (cands[0] as { content?: { parts?: unknown[] } })?.content?.parts;
    if (!Array.isArray(parts)) return safeFallback;
    const chunks: string[] = [];
    for (const p of parts) {
      if (!p || typeof p !== 'object') continue;
      const tx = (p as { text?: unknown }).text;
      if (typeof tx === 'string' && tx.length > 0) chunks.push(tx);
    }
    const joined = chunks.join('').trim();
    const out = trySanitize(joined);
    if (out) return out;
  } catch {
    /* ignore */
  }

  return safeFallback;
}

/** 复刻：MediaRecorder 上传后由 Gemini 多模态转写（与对话主链路一致用 Flash） */
const CLONE_TRANSCRIBE_MODEL = 'gemini-2.5-flash';
const CLONE_MEDIA_RECORDER_SLICE_MS = 400;
const MAX_CLONE_TRANSCRIBE_BYTES = 10 * 1024 * 1024;
const CLONE_TRANSCRIBE_TIMEOUT_MS = 90_000;

function pickMediaRecorderMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  const list = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4;codecs=mp4a.40.2',
    'audio/mp4',
  ];
  for (const t of list) {
    try {
      if (MediaRecorder.isTypeSupported(t)) return t;
    } catch {
      /* ignore */
    }
  }
  if (import.meta.env.DEV) {
    console.warn('[MediaRecorder] none of the preferred MIME types supported, falling back to browser default');
  }
  return undefined;
}

function uint8ArrayToBase64(bytes: Uint8Array): string {
  const chunk = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, Math.min(i + chunk, bytes.length));
    binary += String.fromCharCode.apply(null, sub as unknown as number[]);
  }
  return btoa(binary);
}

async function transcribeCloneAudioWithGeminiFlash(opts: {
  blob: Blob;
  language: 'zh' | 'en';
}): Promise<string> {
  const { blob, language } = opts;
  const buf = await blob.arrayBuffer();
  if (buf.byteLength === 0) return '';
  if (buf.byteLength > MAX_CLONE_TRANSCRIBE_BYTES) {
    throw new Error('CLONE_AUDIO_TOO_LARGE');
  }
  const rawMime = (blob.type && blob.type.trim()) || 'audio/webm';
  const mimeType = rawMime.split(';')[0]!.trim();
  const prompt =
    language === 'zh'
      ? '请逐字转写下段音频中的口述内容。只输出转写正文，不要翻译、不要引号、不要任何解释或开场白。忽略非人类语音与杂音。'
      : 'Transcribe all spoken words in the attached audio verbatim into plain text only. No quotes, no preamble or commentary. Ignore non-human noise.';

  const data = uint8ArrayToBase64(new Uint8Array(buf));
  const run = () =>
    getGeminiClient().models.generateContent({
      model: CLONE_TRANSCRIBE_MODEL,
      contents: [
        {
          role: 'user',
          parts: [{ text: prompt }, { inlineData: { mimeType, data } }],
        },
      ],
      config: { temperature: 0.1 },
    });

  const response = await Promise.race([
    run(),
    new Promise<never>((_, reject) => {
      window.setTimeout(() => reject(new Error('CLIENT_TIMEOUT')), CLONE_TRANSCRIBE_TIMEOUT_MS);
    }),
  ]);
  const t = extractModelReplyText(response, '').trim();
  try {
    return sanitizeModelReplyText(t).trim();
  } catch {
    return t.replace(/\u0000/g, '').trim();
  }
}

/** 等待 speechSynthesis.getVoices() 非空（部分浏览器首次返回 []，需等 voiceschanged） */
function ensureVoicesLoaded(timeoutMs = 1500): Promise<SpeechSynthesisVoice[]> {
  return new Promise((resolve) => {
    if (typeof window.speechSynthesis === 'undefined') { resolve([]); return; }
    const v = window.speechSynthesis.getVoices();
    if (v.length > 0) { resolve(v); return; }
    const timer = window.setTimeout(() => {
      window.speechSynthesis.removeEventListener('voiceschanged', onReady);
      resolve(window.speechSynthesis.getVoices());
    }, timeoutMs);
    const onReady = () => {
      window.speechSynthesis.removeEventListener('voiceschanged', onReady);
      clearTimeout(timer);
      resolve(window.speechSynthesis.getVoices());
    };
    window.speechSynthesis.addEventListener('voiceschanged', onReady);
  });
}

function formatProviderChatError(
  error: unknown,
  lang: 'zh' | 'en',
  quotaHint: { zh: string; en: string },
): string {
  const raw = String((error as any)?.message ?? error ?? '');
  const lower = raw.toLowerCase();
  if (raw === 'CLIENT_TIMEOUT' || lower === 'client_timeout') {
    return lang === 'zh' ? '请求超时，请检查网络后重试。' : 'Request timed out. Check your network and retry.';
  }
  /** 与配额 429 区分：503 / UNAVAILABLE / high demand 多为 Google 侧瞬时高峰 */
  const overload =
    lower.includes('high demand') ||
    lower.includes('unavailable') ||
    lower.includes('service unavailable') ||
    /"code"\s*:\s*503/.test(raw) ||
    raw.includes('"code":503');
  if (overload && !lower.includes('quota') && !lower.includes('resource_exhausted')) {
    return lang === 'zh'
      ? '对话服务暂时繁忙，请隔几分钟再试（与麦克风无关）。'
      : 'The chat service is busy. Wait a few minutes and try again (not your microphone).';
  }
  if (lower.includes('reported as leaked') || lower.includes('use another api key')) {
    return lang === 'zh'
      ? '当前访问密钥已失效，请联系提供方或更新配置后重试。'
      : 'Your access key is no longer valid. Update credentials and try again.';
  }
  if (
    lower.includes('api_key_invalid') ||
    lower.includes('api key not found') ||
    lower.includes('pass a valid api key') ||
    (lower.includes('invalid_argument') && lower.includes('api key'))
  ) {
    return lang === 'zh'
      ? '身份验证失败，请确认服务已正确配置后重试。'
      : 'Authentication failed. Check your setup and try again.';
  }
  if (
    raw.includes('429') ||
    lower.includes('resource_exhausted') ||
    lower.includes('quota') ||
    lower.includes('exceeded your current quota') ||
    lower.includes('rate limit')
  ) {
    if (isGeminiDailyQuotaExhausted({ message: raw })) {
      return lang === 'zh'
        ? '当前模型今日免费调用次数已用完，请明日再试或在 Google AI Studio 查看计费与额度。'
        : 'Daily free-tier limit for this model is reached. Try tomorrow or check quotas in Google AI Studio.';
    }
    return lang === 'zh' ? quotaHint.zh : quotaHint.en;
  }
  if (raw.length > 280) {
    return lang === 'zh' ? '请求失败，请稍后再试。' : 'Request failed. Please try again.';
  }
  return lang === 'zh' ? '请求失败，请稍后再试。' : 'Request failed. Please try again.';
}

function formatGeminiUserMessage(error: unknown, lang: 'zh' | 'en'): string {
  return formatProviderChatError(error, lang, {
    zh: '用量已达上限或请求过于频繁，请稍后再试。',
    en: 'Quota or rate limit reached. Wait and try again.',
  });
}

const ChatOverlay = forwardRef<ChatOverlayHandle, ChatOverlayProps>(function ChatOverlay(
  {
    currentImageDataUrl,
    isOpen,
    onClose,
    language,
    onToggleLanguage,
    aiName,
    onSpeechValue,
    isAutoSpeak,
    setIsAutoSpeak,
    conversationMode,
    settingsChromeOpen = false,
    onDissolveReset,
    onOpenSavePreview,
  },
  ref,
) {
  /**
   * 模式分流（改一条链路勿顺带改另一条）：
   * - **Live**（`conversationMode === 'live'`）：Gemini Live WebSocket、麦克风、面板字幕/声纹。
   * - **复刻**（`conversationMode === 'text_clone'`）：Gemini 文本多轮（咨询师提示）+ 可选复刻音色 TTS；界面与 Live 一致（状态条 +「我说 / AI」字幕面板），不保留会话气泡列表。
   */
  type AIStatus = 'connected' | 'failed';
  const isRenderingRef = useRef(false);
  isRenderingRef.current = true;
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [inputText, setInputText] = useState('');
  const inputTextRef = useRef('');
  useEffect(() => { isRenderingRef.current = false; });
  useEffect(() => {
    inputTextRef.current = inputText;
  }, [inputText]);
  const [isTyping, setIsTyping] = useState(false);
  const [aiStatus, setAiStatus] = useState<AIStatus>(() =>
    getGeminiApiKey() ? 'connected' : 'failed'
  );
  const [aiStatusText, setAiStatusText] = useState(() =>
    !getGeminiApiKey()
      ? language === 'zh'
        ? '未配置对话服务'
        : 'Chat service not configured'
      : language === 'zh'
        ? '已就绪 · 发送消息开始对话'
        : 'Ready · send a message to start'
  );
  const [activeTextModel, setActiveTextModel] = useState(TEXT_MODEL_CANDIDATES[0]);
  /** 仅复刻 + 自动朗读：对标 Live，字幕随播音进度显露（避免整段字先于声音出现） */
  const [cloneTtsPlaybackMsgIdx, setCloneTtsPlaybackMsgIdx] = useState<number | null>(null);
  const [cloneTtsRevealLen, setCloneTtsRevealLen] = useState(0);
  /** 本轮 TTS 正常结束后短暂保留全文 + 对话框，再收起 */
  const [cloneTtsPostSpeechHold, setCloneTtsPostSpeechHold] = useState(false);
  /** audio.play() 因非用户手势抛 NotAllowedError 时，弹出"点击收听"按钮 */
  const [cloneTtsNeedsUserGesture, setCloneTtsNeedsUserGesture] = useState(false);
  const pendingUserGestureAudioRef = useRef<HTMLAudioElement | null>(null);
  const [showSavePreview, setShowSavePreview] = useState(false);
  /** 复刻镌刻：正在生成「留给你的话」结语 */
  const [etchAdviceBusy, setEtchAdviceBusy] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);
  const lastShownModelKeyRef = useRef('');
  const lastSpokenMessageIdRef = useRef<string>(''); // NEW: Track exactly which message was spoken
  const minimaxSpeakAbortRef = useRef<AbortController | null>(null);
  const minimaxAudioRef = useRef<HTMLAudioElement | null>(null);
  const minimaxObjectUrlRef = useRef<string | null>(null);
  /** MiniMax：真正 onplay 之后才按进度显露字幕，避免 duration/解码前误算成满屏字 */
  const minimaxRevealPlaybackStartedRef = useRef(false);
  /** 播完后延迟清空 playback 状态 */
  const cloneTtsHoldTimerRef = useRef<number | null>(null);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  /** 历史回忆立体轮播当前焦点（与 sortedHistorySessions 下标对齐） */
  const [historySlideIdx, setHistorySlideIdx] = useState(0);
  const historySwipeX0 = useRef<number | null>(null);
  const floatIdleHideTimerRef = useRef<number | null>(null);
  const floatAfterFadeTimerRef = useRef<number | null>(null);
  const clearFloatingDisplayTimers = useCallback(() => {
    if (floatIdleHideTimerRef.current != null) {
      window.clearTimeout(floatIdleHideTimerRef.current);
      floatIdleHideTimerRef.current = null;
    }
    if (floatAfterFadeTimerRef.current != null) {
      window.clearTimeout(floatAfterFadeTimerRef.current);
      floatAfterFadeTimerRef.current = null;
    }
  }, []);
  const [voiceBlockingMessage, setVoiceBlockingMessage] = useState<string | null>(null);
  const voiceBlockingRetryRef = useRef<(() => void) | null>(null);

  /** 对话层关闭时收起所有 Portal 弹层，避免 isOpen 已 false 仍保留状态导致下一帧与 DOM 不一致（insertBefore 报错） */
  useEffect(() => {
    if (isOpen) return;
    setShowHistoryModal(false);
    setShowSavePreview(false);
    setVoiceBlockingMessage(null);
    voiceBlockingRetryRef.current = null;
  }, [isOpen]);

  /** 改 session 音色后强制重渲染（sessionStorage 本身不触发 React） */
  const [, setLiveVoiceRevision] = useState(0);
  const [liveVoiceSwitchHint, setLiveVoiceSwitchHint] = useState('');
  const liveVoiceSwitchHintTimerRef = useRef<number | null>(null);
  /** 下拉切换 Live 音色：断开后仍保持语音底栏/隐藏文字区，直至自动重连成功或用户退出 */
  const [liveVoiceHandoff, setLiveVoiceHandoff] = useState(false);
  const onSpeechValueRef = useRef(onSpeechValue);
  useEffect(() => {
    onSpeechValueRef.current = onSpeechValue;
  }, [onSpeechValue]);

  const isOpenRef = useRef(isOpen);
  isOpenRef.current = isOpen;
  const conversationModeRef = useRef(conversationMode);
  conversationModeRef.current = conversationMode;

  const currentImageDataUrlRef = useRef<string | null>(null);
  useEffect(() => {
    currentImageDataUrlRef.current = currentImageDataUrl ?? null;
  }, [currentImageDataUrl]);

  // Real-time Voice State
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [isVoiceConnecting, setIsVoiceConnecting] = useState(false);
  const liveSessionRef = useRef<any>(null);
  /** 每次 stop / 新 connect 递增；用于忽略旧 Live 会话晚到的 onclose（否则会清掉 liveVoiceHandoff 导致界面跳回文字） */
  const liveVoiceSessionGenRef = useRef(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | ScriptProcessorNode | null>(null);
  /** 增益为 0，避免麦克风直连扬声器产生啸叫；Worklet/ScriptProcessor 均须接入图 */
  const processorMuteRef = useRef<GainNode | null>(null);
  /** Live：WebSocket onopen 后为 true（与原先可用行为一致）；mic PCM 由此开始发送 */
  const voiceLiveReadyRef = useRef(false);
  /** AudioWorklet 小帧拼包剩余样本；stop 时清空 */
  const liveMicPcmRemainderRef = useRef<Float32Array | null>(null);
  /** Live 麦克风图实际采样率（与 LIVE_GEMINI_INPUT_RATE 不一致时需重采样上行 PCM） */
  const liveMicActualSampleRateRef = useRef(LIVE_GEMINI_INPUT_RATE);
  /** 用户主动关了麦克风时为 true；异常断线时为 false 才可能自动重连 */
  const liveVoiceUserStoppedRef = useRef(true);
  /** 单次开麦周期内 1006/1011 自动重连已尝试次数；成功连上后归零 */
  const liveWsAbnormalRetryRef = useRef(0);
  const liveVoiceReconnectTimerRef = useRef<number | null>(null);
  /** onclose 定时器重连须调用最新 startVoiceMode，避免闭包陈旧 */
  const startVoiceModeRef = useRef<(() => void) | null>(null);
  /** 供其它 effect 调用最新 stopVoiceMode，避免依赖函数本体导致重复执行 */
  const stopVoiceModeRef = useRef<(opts?: { preserveLiveVoiceHandoff?: boolean }) => void>(() => {});
  const nextPlayTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  /** Live：已结束的每轮「识别 + 字幕」完整保留 */
  const [liveCaptionTurns, setLiveCaptionTurns] = useState<LiveCaptionTurn[]>([]);
  /** 当前轮流式累积（合并片段，避免只显示最后几个词） */
  const [liveStreamUser, setLiveStreamUser] = useState('');
  const [liveStreamAi, setLiveStreamAi] = useState('');
  /** 与上面对照：识别译文 / 字幕中文 */
  const [liveStreamUserAlt, setLiveStreamUserAlt] = useState('');
  const [liveStreamAiZh, setLiveStreamAiZh] = useState('');
  const liveStreamUserRef = useRef('');
  const liveStreamAiRef = useRef('');
  const lastChatErrorRef = useRef<unknown>(null);

  /** Live 麦克风 RMS → 「我在说话」门控（与 AI 播音互斥展示）；语谱另接 AnalyserNode */
  const liveInputRmsRef = useRef(0);
  const liveLastVoiceTsRef = useRef(0);
  const [liveUserMicHot, setLiveUserMicHot] = useState(false);
  const liveVoiceAnalyserRef = useRef<AnalyserNode | null>(null);
  const voiceprintCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const voiceprintRafRef = useRef<number | null>(null);
  const voiceprintEnergySmoothRef = useRef(0);
  /** AI Live 播音接入 AnalyserNode，供字幕框内声纹画布频谱 */
  const liveAiPlaybackTapRef = useRef<GainNode | null>(null);
  const liveAiVoiceAnalyserRef = useRef<AnalyserNode | null>(null);
  const voiceprintAiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const voiceprintAiRafRef = useRef<number | null>(null);
  const voiceprintAiEnergySmoothRef = useRef(0);
  /** Live：首段 PCM 排程在未来时刻时，推迟 setIsSpeaking(true)，避免声纹早于扬声器出声 */
  const liveAiSpeakingDelayTimerRef = useRef<number | null>(null);

  /** 复刻链路：MediaRecorder → Gemini 2.5 Flash 转写 → 文本发送（不走 Live / 不用浏览器 SpeechRecognition） */
  const [isCloneDictating, setIsCloneDictating] = useState(false);
  const cloneMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const cloneRecordChunksRef = useRef<Blob[]>([]);
  const cloneHoldSnapshotRef = useRef('');
  const cloneHoldCleanupStopRef = useRef(false);
  /** stopCloneDictation({ cleanup }) / 切模式时递增，进行中的录音或转写检测到代数变化则放弃发送 */
  const cloneDictationEpochRef = useRef(0);
  /** 点按说话：字幕与声纹同步，不写输入框 */
  const [cloneLiveCaptionUserLine, setCloneLiveCaptionUserLine] = useState('');
  const [cloneUserPostDictationHold, setCloneUserPostDictationHold] = useState(false);
  /** 完成听写发送后短暂沿用本条正文，避免会话 state 未跟上时空白 */
  const [cloneUserPostDictationEcho, setCloneUserPostDictationEcho] = useState('');
  /** 语音输入发送本轮：同步压制上方 pending 气泡（避免与「我说」重复），至本轮 AI 结束；与 state 并行以免批处理晚一拍 */
  const cloneVoiceRoundSuppressPendingBubbleRef = useRef(false);
  const [cloneUserMicPrimed, setCloneUserMicPrimed] = useState(0);
  const cloneUserDictationHoldTimerRef = useRef<number | null>(null);
  const cloneUserMicStreamRef = useRef<MediaStream | null>(null);
  const cloneUserMicCtxRef = useRef<AudioContext | null>(null);
  const cloneUserMicAnalyserRef = useRef<AnalyserNode | null>(null);
  const sendMessageRef = useRef<(opts?: { textOverride?: string; preserveInputFromVoiceHold?: boolean }) => void>(
    () => {},
  );
  /** 复刻底栏输入框：听写进行中失焦，避免系统将听写写入 textarea 造成「又说了一次」 */
  const cloneBottomTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  /** 复刻 AI 字幕：随 reveal 增高时钉在底部，避免长文顶出可视区 */
  const cloneAiCaptionScrollRef = useRef<HTMLDivElement | null>(null);

  const clearCloneUserDictationHold = useCallback(() => {
    if (cloneUserDictationHoldTimerRef.current != null) {
      window.clearTimeout(cloneUserDictationHoldTimerRef.current);
      cloneUserDictationHoldTimerRef.current = null;
    }
    setCloneUserPostDictationHold(false);
    setCloneUserPostDictationEcho('');
  }, []);

  const armCloneUserDictationHoldAfterSend = useCallback((spokenText: string) => {
    if (cloneUserDictationHoldTimerRef.current != null) {
      window.clearTimeout(cloneUserDictationHoldTimerRef.current);
      cloneUserDictationHoldTimerRef.current = null;
    }
    const t = typeof spokenText === 'string' ? spokenText.trim() : String(spokenText ?? '').trim();
    setCloneUserPostDictationHold(true);
    setCloneUserPostDictationEcho(t);
    cloneUserDictationHoldTimerRef.current = window.setTimeout(() => {
      cloneUserDictationHoldTimerRef.current = null;
      setCloneUserPostDictationHold(false);
      setCloneUserPostDictationEcho('');
    }, CLONE_USER_POST_DICTATION_HOLD_MS);
  }, []);

  const stopCloneUserMicCapture = useCallback(() => {
    cloneUserMicAnalyserRef.current = null;
    try {
      void cloneUserMicCtxRef.current?.close();
    } catch {
      /* ignore */
    }
    cloneUserMicCtxRef.current = null;
    if (cloneUserMicStreamRef.current) {
      cloneUserMicStreamRef.current.getTracks().forEach((t) => t.stop());
      cloneUserMicStreamRef.current = null;
    }
    setCloneUserMicPrimed((n) => n + 1);
  }, []);

  /** 将已授权的麦克风流接到声纹 Analyser（与 MediaRecorder 共用同一条 stream） */
  const attachCloneMicAnalyserFromStream = useCallback(
    async (stream: MediaStream) => {
      cloneUserMicStreamRef.current = stream;
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      const AC = Ctx ?? AudioContext;
      const ctx = new AC();
      cloneUserMicCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      cloneUserMicAnalyserRef.current = analyser;
      try {
        await ctx.resume();
      } catch {
        /* ignore */
      }
      setCloneUserMicPrimed((n) => n + 1);
    },
    [],
  );

  const isAIActive = isTyping || isSpeaking || isVoiceConnecting;

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;

  const sortedHistorySessions = useMemo(
    () => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions],
  );
  const activeSessionRef = useRef(activeSession);
  activeSessionRef.current = activeSession;

  const albumSessions = useMemo(
    () =>
      sessions
        .filter((s) => s.etchedToAlbum === true && (s.messages?.length ?? 0) > 0)
        .sort((a, b) => (b.etchedAt ?? 0) - (a.etchedAt ?? 0)),
    [sessions],
  );

  useImperativeHandle(
    ref,
    () => ({
      openSavePreview: () => {
        if (activeSessionRef.current) setShowSavePreview(true);
      },
      openHistory: () => setShowHistoryModal(true),
    }),
    [],
  );

  const allowLiveVoice = conversationMode === 'live';

  /**
   * 最新一条 model 回合的稳定键：仅用 sessionId + 消息下标（不用正文）。
   * 避免会话对象频繁换新引用时 effect 重复跑、或与正文偶然重复时键冲突，导致第二轮不复读。
   */
  const latestModelTurnMeta = useMemo(() => {
    try {
      if (!activeSession?.messages?.length) return null;
      for (let i = activeSession.messages.length - 1; i >= 0; i -= 1) {
        const m = activeSession.messages[i];
        if (!m) continue;
        if (m.role === 'model' && m.kind !== 'closing_note') {
          return {
            turnKey: `${activeSession.id}-model-${i}`,
            text: safeMsgText(m.text),
          };
        }
      }
      return null;
    } catch (e) {
      console.warn('[latestModelTurnMeta]', e);
      return null;
    }
  }, [activeSession?.id, activeSession?.messages]);

  const latestModelTurnKey = latestModelTurnMeta?.turnKey ?? '';

  /** 只显示「当前这一轮」尚未被 AI 回应的用户句，避免手机上用户气泡堆满屏 */
  const pendingUserBubble = useMemo((): { text: string; imageUrl?: string } | null => {
    const msgs = activeSession?.messages;
    if (!msgs?.length) return null;
    let lastUserIdx = -1;
    let lastUserText = '';
    let lastUserImage: string | undefined;
    for (let i = msgs.length - 1; i >= 0; i -= 1) {
      if (msgs[i].role === 'user') {
        lastUserIdx = i;
        lastUserText = safeMsgText(msgs[i].text);
        lastUserImage = msgs[i].attachedImageDataUrl;
        break;
      }
    }
    if (lastUserIdx < 0) return null;
    let lastModelIdx = -1;
    for (let i = msgs.length - 1; i >= 0; i -= 1) {
      if (msgs[i].role === 'model') {
        lastModelIdx = i;
        break;
      }
    }
    const waitingForThisUserTurn = lastModelIdx < lastUserIdx;
    return waitingForThisUserTurn ? { text: lastUserText, imageUrl: lastUserImage } : null;
  }, [activeSession?.messages]);

  /** 复刻：与 Live 对齐的字幕数据源（主界面不渲染气泡历史） */
  const cloneCaptionState = useMemo(() => {
    if (conversationMode !== 'text_clone') return null;
    try {
      const msgs = activeSession?.messages ?? [];
      let lastUser = '';
      let lastUserIdx = -1;
      let lastUserImage: string | undefined;
      for (let i = msgs.length - 1; i >= 0; i -= 1) {
        const row = msgs[i];
        if (!row) continue;
        if (row.role === 'user') {
          lastUser = safeMsgText(row.text).trim();
          lastUserIdx = i;
          lastUserImage = row.attachedImageDataUrl;
          break;
        }
      }
      let lastModel = '';
      let lastModelIdx = -1;
      for (let i = msgs.length - 1; i >= 0; i -= 1) {
        const row = msgs[i];
        if (!row) continue;
        if (row.role === 'model' && row.kind !== 'closing_note') {
          lastModel = safeMsgText(row.text).trim();
          lastModelIdx = i;
          break;
        }
      }
      const waitingForModel = lastUserIdx > lastModelIdx;
      const ttsStreaming =
        isAutoSpeak &&
        cloneTtsPlaybackMsgIdx !== null &&
        cloneTtsPlaybackMsgIdx === lastModelIdx &&
        lastModelIdx >= 0;
      /** 自动朗读：未进入本轮随播Reveal前绝不晒全文，避免「字先出来、声后到」 */
      let aiMain = '';
      if (lastModelIdx >= 0) {
        if (cloneTtsPostSpeechHold) {
          aiMain = lastModel;
        } else if (ttsStreaming && cloneTtsRevealLen > 0) {
          aiMain = lastModel.slice(0, Math.min(lastModel.length, cloneTtsRevealLen));
        } else {
          aiMain = lastModel;
        }
      }
      const awaitingTtsStart =
        isAutoSpeak &&
        !cloneTtsPostSpeechHold &&
        !isSpeaking &&
        ttsStreaming &&
        lastModelIdx >= 0;
      return {
        userMain: lastUser,
        userAlt: '',
        aiMain,
        aiAlt: '',
        waitingForModel,
        lastUserImage,
        awaitingTtsStart,
      };
    } catch (e) {
      console.warn('[Clone caption state]', e);
      return {
        userMain: '',
        userAlt: '',
        aiMain: '',
        aiAlt: '',
        waitingForModel: false,
        lastUserImage: undefined,
        awaitingTtsStart: false,
      };
    }
  }, [
    conversationMode,
    activeSession?.messages,
    isAutoSpeak,
    cloneTtsPlaybackMsgIdx,
    cloneTtsRevealLen,
    cloneTtsPostSpeechHold,
    isSpeaking,
  ]);

  const cloneUserCaptionAlt = cloneCaptionState?.userAlt ?? '';
  const cloneAiCaptionMain = cloneCaptionState?.aiMain ?? '';
  const cloneAiCaptionAlt = cloneCaptionState?.aiAlt ?? '';
  const cloneWaitingForModel = cloneCaptionState?.waitingForModel ?? false;
  const cloneAwaitingTtsStart = cloneCaptionState?.awaitingTtsStart ?? false;
  /** 复刻：听写中显示实时状态，识别完显示文字，AI 回复后隐藏 */
  const cloneUserSubtitleDisplay = isCloneDictating
    ? cloneLiveCaptionUserLine.trim()
    : cloneUserPostDictationHold && cloneUserPostDictationEcho.trim()
      ? cloneUserPostDictationEcho.trim()
      : '';

  const cloneVoiceChromeOpen =
    conversationMode === 'text_clone' &&
    (!!activeSession?.messages?.length ||
      isTyping ||
      isSpeaking ||
      isCloneDictating ||
      cloneUserPostDictationHold);

  /**
   * 自动朗读时：字幕对话框仅在「本轮进行中」展示（等待 / 播报 / 语音输入），避免一整轮结束后仍占屏；
   * 关自动朗读时：保留卡片便于阅读未播音的完整回复。
   */
  const cloneSubtitlePanelsActive =
    !isAutoSpeak ||
    isTyping ||
    isSpeaking ||
    isCloneDictating ||
    cloneTtsPostSpeechHold ||
    cloneUserPostDictationHold;

  /** 复刻：用户说话时只秀「我说」；AI 等回复/播报/尾帧停留时只秀 AI 卡（波纹均在各自卡片内） */
  const cloneCaptionAiExclusivePhase =
    (isTyping && cloneWaitingForModel) || isSpeaking || cloneTtsPostSpeechHold;
  const cloneCaptionUserExclusivePhase = isCloneDictating || cloneUserPostDictationHold;

  const showCloneUserCaptionPanel =
    cloneVoiceChromeOpen &&
    cloneSubtitlePanelsActive &&
    (!cloneCaptionAiExclusivePhase || cloneUserPostDictationHold) &&
    !isSpeaking &&
    (cloneLiveCaptionUserLine.trim().length > 0 ||
      cloneUserPostDictationEcho.trim().length > 0 ||
      isCloneDictating);

  const showCloneAiCaptionPanel =
    cloneVoiceChromeOpen &&
    cloneSubtitlePanelsActive &&
    !cloneCaptionUserExclusivePhase &&
    (cloneAiCaptionMain.length > 0 ||
      isSpeaking ||
      cloneTtsPostSpeechHold ||
      cloneAwaitingTtsStart ||
      (isTyping && cloneWaitingForModel));

  const showCloneCaptionChrome = showCloneUserCaptionPanel || showCloneAiCaptionPanel;

  // Auto-start session when图像已选且对话模式已选定
  useEffect(() => {
    if (isOpen && currentImageDataUrl && !activeSessionId) {
      createNewSession();
    }
  }, [isOpen, currentImageDataUrl, activeSessionId]);

  // Reset session when image is cleared
  useEffect(() => {
    if (!currentImageDataUrl) {
      setActiveSessionId(null);
    }
  }, [currentImageDataUrl]);
  
  // Load sessions from local storage
  useEffect(() => {
    const parsed = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
    if (parsed.length) {
      parsed.sort((a: ChatSession, b: ChatSession) => b.updatedAt - a.updatedAt);
      setSessions(parsed);
    }
  }, []);

  /** 打开历史时，把轮播焦点对齐当前会话 */
  useEffect(() => {
    if (!showHistoryModal) return;
    const i = sortedHistorySessions.findIndex((s) => s.id === activeSessionId);
    setHistorySlideIdx(i >= 0 ? i : 0);
  }, [showHistoryModal, activeSessionId, sortedHistorySessions]);

  /** 回忆长廊：键盘左右切页 */
  useEffect(() => {
    if (!showHistoryModal) return;
    const max = Math.max(0, sortedHistorySessions.length - 1);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setHistorySlideIdx((i) => Math.max(0, i - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setHistorySlideIdx((i) => Math.min(max, i + 1));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showHistoryModal, sortedHistorySessions.length]);

  /** 当前会话尚无封面时，用画布图补一张（旧数据兼容） */
  useEffect(() => {
    if (!activeSessionId || !currentImageDataUrl?.trim()) return;
    let cancelled = false;
    const sid = activeSessionId;
    const img = currentImageDataUrl.trim();
    void (async () => {
      const cur = sessionsRef.current.find((x) => x.id === sid);
      if (!cur || cur.thumbnailDataUrl) return;
      const thumb = await makeSessionCoverThumbnail(img);
      if (cancelled || !thumb) return;
      setSessions((prev) => {
        const ix = prev.findIndex((x) => x.id === sid);
        if (ix < 0) return prev;
        if (prev[ix]?.thumbnailDataUrl) return prev;
        const next = prev.map((x, i) => (i === ix ? { ...x, thumbnailDataUrl: thumb } : x));
        try {
          localStorage.setItem('subconscious_sessions', JSON.stringify(stripAttachedImagesForStorage(next)));
        } catch (e) {
          console.error('Failed to persist session thumbnail', e);
        }
        return next;
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [activeSessionId, currentImageDataUrl]);

  // Refresh status strip when language toggles (no extra API calls — saves free-tier quota).
  useEffect(() => {
    if (!getGeminiApiKey()) {
      setAiStatus('failed');
      setAiStatusText(language === 'zh' ? '未配置对话服务' : 'Chat service not configured');
      return;
    }
    if (lastChatErrorRef.current != null) {
      setAiStatus('failed');
      setAiStatusText(formatGeminiUserMessage(lastChatErrorRef.current, language));
      return;
    }
    setAiStatus('connected');
    setAiStatusText(language === 'zh' ? '已就绪 · 发送消息开始对话' : 'Ready · send a message to start');
  }, [language]);

  // Scroll to bottom（部分 WebView 对 smooth 滚动异常，兜底避免抛错）
  useEffect(() => {
    const el = endOfMessagesRef.current;
    if (!el) return;
    try {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch {
      try {
        el.scrollIntoView({ block: 'nearest' });
      } catch {
        /* ignore */
      }
    }
  }, [sessions, activeSessionId, isTyping]);

  // 用户发出新一句（等待 AI）时清掉浮层计时（若有）
  useEffect(() => {
    const msgs = activeSession?.messages;
    if (!msgs?.length) return;
    if (msgs[msgs.length - 1].role !== 'user') return;
    clearFloatingDisplayTimers();
  }, [activeSession?.messages, clearFloatingDisplayTimers]);

  // 复刻：新 model 回合触发 MiniMax / 系统 TTS；Live 不走此链路。
  // 语音/TTS：必须用「最后一条 model 消息」的稳定 key，避免用户插话导致重复朗读。
  useEffect(() => {

    const stopMinimaxPlaybackVisual = () => {
      if (conversationModeRef.current === 'text_clone') {
        liveAiVoiceAnalyserRef.current = null;
      }
    };

    const stopMinimaxPlayback = () => {
      minimaxRevealPlaybackStartedRef.current = false;
      minimaxSpeakAbortRef.current?.abort();
      minimaxSpeakAbortRef.current = null;
      stopMinimaxPlaybackVisual();
      if (minimaxAudioRef.current) {
        try {
          minimaxAudioRef.current.pause();
        } catch {
          /* ignore */
        }
        minimaxAudioRef.current = null;
      }
      if (minimaxObjectUrlRef.current) {
        try {
          URL.revokeObjectURL(minimaxObjectUrlRef.current);
        } catch {
          /* ignore */
        }
        minimaxObjectUrlRef.current = null;
      }
    };

    if (!latestModelTurnKey || !latestModelTurnMeta?.text?.trim()) return () => {};

    const messageKey = latestModelTurnKey;

    /** Live 已有「我说 / AI 回复」字幕区；持久化进会话的 model 勿再弹悬浮气泡（否则会重复） */
    if (conversationMode === 'live') {
      lastShownModelKeyRef.current = messageKey;
      return () => {};
    }

    /**
     * 复刻：勿仅用 lastShown 去重。Strict Mode / 重挂载时 cleanup 会清空 lastSpoken，若仍认为「已展示」则第二轮
     * 不会调用 speak，表现为有声无字或字幕进不了 AI 卡；仅在「同 key 且本轮已开读」时跳过。
     */
    if (
      messageKey === lastShownModelKeyRef.current &&
      lastSpokenMessageIdRef.current === messageKey
    ) {
      return () => {};
    }

    lastShownModelKeyRef.current = messageKey;

    const fullText = sanitizeModelReplyText(latestModelTurnMeta.text);

    const textCloneReadAloud = conversationMode === 'text_clone' && isAutoSpeak;

    const clearCloneTtsHoldTimer = () => {
      if (cloneTtsHoldTimerRef.current != null) {
        window.clearTimeout(cloneTtsHoldTimerRef.current);
        cloneTtsHoldTimerRef.current = null;
      }
    };

    const scheduleCloneTtsDone = (opts?: { postSpeechHoldMs?: number }) => {
      console.info(`[TTS] scheduleCloneTtsDone epoch=${lastSpokenMessageIdRef.current} holdMs=${opts?.postSpeechHoldMs ?? 0} renderPhase=${isRenderingRef.current}`);
      clearFloatingDisplayTimers();
      clearCloneTtsHoldTimer();
      const holdMs = opts?.postSpeechHoldMs ?? 0;
      setIsSpeaking(false);
      if (holdMs > 0) {
        setCloneTtsPlaybackMsgIdx(null);
        setCloneTtsRevealLen(0);
        setCloneTtsPostSpeechHold(true);
        cloneTtsHoldTimerRef.current = window.setTimeout(() => {
          cloneTtsHoldTimerRef.current = null;
          setCloneTtsPostSpeechHold(false);
        }, holdMs);
      } else {
        setCloneTtsPostSpeechHold(false);
        setCloneTtsPlaybackMsgIdx(null);
        setCloneTtsRevealLen(0);
      }
    };

    /** 复刻 · 自动朗读：MiniMax / 系统 TTS；字幕在「AI」面板（与 Live 同布局） */
    const speakCloneReadAloud = (text: string, messageId: string) => {
      if (lastSpokenMessageIdRef.current === messageId) return;
      const idxMk = messageId.match(/-model-(\d+)$/);
      const idxFromKey = idxMk ? Number(idxMk[1]) : NaN;
      if (Number.isFinite(idxFromKey) && idxFromKey >= 0) {
        setCloneTtsPlaybackMsgIdx(idxFromKey);
        setCloneTtsRevealLen(0);
      }
      console.info(`[TTS] speakCloneReadAloud START epoch=${messageKey} prevEpoch=${lastSpokenMessageIdRef.current}`);
      lastSpokenMessageIdRef.current = messageId;
      setIsSpeaking(true);

      /**
       * speechSynthesis.cancel() 在部分安卓浏览器上会同步触发旧 utterance 的 onerror，
       * 若该回调直接调用 scheduleCloneTtsDone() 修改 state，可能与当前渲染帧的
       * setState 嵌套，导致 React 抛 "Cannot update during render" 被ErrorBoundary 捕获。
       * 用 isCurrentEpoch 判断当前回合，旧回调检测到不匹配时跳过 state 更新。
       */
      const isCurrentEpoch = (): boolean => lastSpokenMessageIdRef.current === messageId;

      console.info(`[TTS] calling speechSynthesis.cancel(), current synth.speaking=${window.speechSynthesis?.speaking}`);
      try { window.speechSynthesis.cancel(); } catch { /* ignore */ }
      console.info(`[TTS] speechSynthesis.cancel() returned`);
      stopMinimaxPlayback();

      const speakSlice = text.length > MAX_TTS_CHARS ? `${text.slice(0, MAX_TTS_CHARS)}…` : text;
      const ttsPlain = stripTextForTts(speakSlice);
      if (!ttsPlain) {
        scheduleCloneTtsDone();
        return;
      }

      const bindSpeechSynthUtterance = (utterance: SpeechSynthesisUtterance) => {
        let synthStarted = false;
        let revealTimerId: number | null = null;
        let revealStartTime = 0;
        /** onboundary 在安卓 Chrome 常不触发，用定时器兜底逐字显露 */
        const startRevealTimer = () => {
          const charsPerSec = 10 * utterance.rate;
          revealStartTime = performance.now();
          revealTimerId = window.setInterval(() => {
            if (!isCurrentEpoch()) { clearRevealTimer(); return; }
            const elapsed = (performance.now() - revealStartTime) / 1000;
            const estimated = Math.floor(elapsed * charsPerSec);
            setCloneTtsRevealLen((prev) => Math.max(prev, Math.min(speakSlice.length, estimated)));
          }, 200);
        };
        const clearRevealTimer = () => {
          if (revealTimerId !== null) { clearInterval(revealTimerId); revealTimerId = null; }
        };
        utterance.onboundary = (ev) => {
          if (!synthStarted || !isCurrentEpoch()) return;
          const ci = ev.charIndex;
          const cl = ev.charLength ?? 0;
          if (typeof ci !== 'number' || !Number.isFinite(ci) || ci < 0) return;
          const end = ci + (typeof cl === 'number' && Number.isFinite(cl) ? cl : 0);
          if (!Number.isFinite(end)) return;
          // onboundary 到达时取 max，避免定时器估计值回退
          setCloneTtsRevealLen((prev) => Math.min(speakSlice.length, Math.max(prev, Math.max(0, end))));
        };
        utterance.onstart = () => {
          console.info(`[TTS] onstart epoch=${messageKey} isCurrent=${isCurrentEpoch()} renderPhase=${isRenderingRef.current}`);
          if (!isCurrentEpoch()) return;
          synthStarted = true;
          setCloneTtsRevealLen((n) => Math.max(n, 1));
          startRevealTimer();
        };
        utterance.onend = () => {
          clearRevealTimer();
          console.info(`[TTS] onend epoch=${messageKey} isCurrent=${isCurrentEpoch()} renderPhase=${isRenderingRef.current}`);
          if (!isCurrentEpoch()) return;
          onSpeechValueRef.current?.(0);
          scheduleCloneTtsDone({ postSpeechHoldMs: CLONE_TTS_POST_SPEECH_HOLD_MS });
        };
        utterance.onerror = (ev) => {
          clearRevealTimer();
          const errMsg = (ev as SpeechSynthesisErrorEvent)?.error;
          console.info(`[TTS] onerror epoch=${messageKey} isCurrent=${isCurrentEpoch()} err=${errMsg} renderPhase=${isRenderingRef.current}`);
          /** 安卓 Chrome：cancel() 同步触发 onerror（error=canceled），旧回合应忽略 */
          if (!isCurrentEpoch()) return;
          if (errMsg === 'canceled' || errMsg === 'interrupted') return;
          onSpeechValueRef.current?.(0);
          scheduleCloneTtsDone();
        };
      };

      const applyVoiceAndSpeak = async () => {
        if (typeof window.speechSynthesis === 'undefined') return;
        try {
          const utterance = new SpeechSynthesisUtterance(speakSlice);
          const voices = await ensureVoicesLoaded();
          const preferredVoice =
            voices.find((v) => (v.name.includes('Google') || v.name.includes('Premium')) && v.lang.includes('zh')) ||
            voices.find((v) => v.lang.includes('zh'));
          if (preferredVoice) utterance.voice = preferredVoice;
          utterance.lang = language === 'zh' ? 'zh-CN' : 'en-US';
          utterance.rate = 0.85;
          utterance.pitch = 0.9;
          bindSpeechSynthUtterance(utterance);
          window.speechSynthesis.speak(utterance);
        } catch (e) {
          console.warn('speechSynthesis failed', e);
          scheduleCloneTtsDone();
        }
      };

      if (getMinimaxApiKey()) {
        const ac = new AbortController();
        minimaxSpeakAbortRef.current = ac;
        void (async () => {
          try {
            const blob = await synthesizeMinimaxTtsToMp3Blob({
              text: ttsPlain,
              signal: ac.signal,
              voiceId: readStoredMinimaxClonedVoiceId()?.trim() || undefined,
            });
            if (ac.signal.aborted) {
              scheduleCloneTtsDone();
              return;
            }
            if (lastSpokenMessageIdRef.current !== messageId) {
              scheduleCloneTtsDone();
              return;
            }
            const objectUrl = URL.createObjectURL(blob);
            minimaxObjectUrlRef.current = objectUrl;
            const audio = new Audio(objectUrl);
            minimaxAudioRef.current = audio;

            const teardownMinimaxAudioSurface = () => {
              stopMinimaxPlaybackVisual();
              onSpeechValueRef.current?.(0);
              minimaxRevealPlaybackStartedRef.current = false;
              if (minimaxObjectUrlRef.current) {
                try {
                  URL.revokeObjectURL(minimaxObjectUrlRef.current);
                } catch {
                  /* ignore */
                }
                minimaxObjectUrlRef.current = null;
              }
              minimaxAudioRef.current = null;
            };

            audio.onplay = () => {
              console.info(`[TTS] MiniMax onplay epoch=${messageKey} isCurrent=${isCurrentEpoch()} renderPhase=${isRenderingRef.current}`);
              minimaxRevealPlaybackStartedRef.current = true;
              setCloneTtsRevealLen((n) => Math.max(n, 1));
            };
            audio.onended = () => {
              console.info(`[TTS] MiniMax onended epoch=${messageKey} isCurrent=${isCurrentEpoch()} renderPhase=${isRenderingRef.current}`);
              teardownMinimaxAudioSurface();
              scheduleCloneTtsDone({ postSpeechHoldMs: CLONE_TTS_POST_SPEECH_HOLD_MS });
            };
            audio.onerror = () => {
              console.info(`[TTS] MiniMax audio.onerror epoch=${messageKey} renderPhase=${isRenderingRef.current}`);
              teardownMinimaxAudioSurface();
              scheduleCloneTtsDone();
            };
            console.info(`[TTS] MiniMax calling audio.play() epoch=${messageKey} renderPhase=${isRenderingRef.current}`);
            await audio.play();
          } catch (e: any) {
            if (ac.signal.aborted) return;
            // NotAllowedError: audio.play() 不在用户手势上下文（常见于部分移动浏览器）
            if (e?.name === 'NotAllowedError' && minimaxAudioRef.current) {
              pendingUserGestureAudioRef.current = minimaxAudioRef.current;
              setCloneTtsNeedsUserGesture(true);
              return;
            }
            console.warn('MiniMax TTS failed, falling back to speechSynthesis', e);
            stopMinimaxPlayback();
            void applyVoiceAndSpeak();
          }
        })();
        return;
      }

      if (typeof window.speechSynthesis === 'undefined') {
        scheduleCloneTtsDone();
        return;
      }
      void applyVoiceAndSpeak();
    };

    const cleanupFloatingAndTts = () => {
      console.info(`[TTS] cleanupFloatingAndTts epoch=${lastSpokenMessageIdRef.current} renderPhase=${isRenderingRef.current}`);
      clearFloatingDisplayTimers();
      clearCloneTtsHoldTimer();
      setCloneTtsPostSpeechHold(false);
      setCloneTtsNeedsUserGesture(false);
      pendingUserGestureAudioRef.current = null;
      window.speechSynthesis.cancel();
      minimaxSpeakAbortRef.current?.abort();
      minimaxSpeakAbortRef.current = null;
      lastSpokenMessageIdRef.current = '';
      stopMinimaxPlaybackVisual();
      if (minimaxAudioRef.current) {
        try {
          minimaxAudioRef.current.pause();
        } catch {
          /* ignore */
        }
        minimaxAudioRef.current = null;
      }
      if (minimaxObjectUrlRef.current) {
        try {
          URL.revokeObjectURL(minimaxObjectUrlRef.current);
        } catch {
          /* ignore */
        }
        minimaxObjectUrlRef.current = null;
      }
      /** 勿在此处清空 `cloneTtsPlaybackMsgIdx` / `cloneTtsRevealLen`：新一轮 model 到达时 cleanup 先执行，
       * 会与 `sendMessage` 刚写入的播放索引竞态，导致第 2 条及以后自动朗读时字幕面板拿不到逐字长度。 */
      setIsSpeaking(false);
    };

    if (textCloneReadAloud) {
      speakCloneReadAloud(fullText, messageKey);
      return cleanupFloatingAndTts;
    }

    return cleanupFloatingAndTts;
  }, [
    latestModelTurnKey,
    latestModelTurnMeta?.text,
    isAutoSpeak,
    language,
    clearFloatingDisplayTimers,
    conversationMode,
  ]);

  /** 离开复刻自动朗读时复位字幕索引（勿放进上轮 TTS cleanup，否则与新一轮 `sendMessage` 竞态） */
  useEffect(() => {
    if (conversationMode === 'text_clone' && isAutoSpeak) return;
    setCloneTtsPlaybackMsgIdx(null);
    setCloneTtsRevealLen(0);
    setCloneTtsPostSpeechHold(false);
  }, [conversationMode, isAutoSpeak]);

  /** 复刻 AI 卡：字幕随流式显露自动滚到底（双 rAF 等待布局；依赖 isSpeaking 覆盖面板挂载瞬间） */
  useEffect(() => {
    if (conversationMode !== 'text_clone') return;
    const el = cloneAiCaptionScrollRef.current;
    if (!el) return;
    let raf0 = 0;
    let raf1 = 0;
    raf0 = window.requestAnimationFrame(() => {
      raf1 = window.requestAnimationFrame(() => {
        try {
          const sh = el.scrollHeight;
          if (Number.isFinite(sh)) el.scrollTop = sh;
        } catch (e) {
          console.warn('[Clone caption scroll]', e);
        }
      });
    });
    return () => {
      window.cancelAnimationFrame(raf0);
      window.cancelAnimationFrame(raf1);
    };
  }, [conversationMode, cloneAiCaptionMain, cloneTtsRevealLen, cloneAwaitingTtsStart, isSpeaking]);

  /** 字幕翻译等非核心路径：503/429 可退避重试，避免刷屏打满配额拖慢整页 */
  const isTransientGeminiHttpError = (e: unknown): boolean => {
    const x = e as { status?: number; code?: number; message?: string; error?: { code?: number } };
    const code = x?.status ?? x?.code ?? x?.error?.code;
    if (code === 503 || code === 429) return true;
    const msg = String(x?.message ?? e ?? '').toUpperCase();
    return (
      msg.includes('503') ||
      msg.includes('429') ||
      msg.includes('UNAVAILABLE') ||
      msg.includes('RESOURCE_EXHAUSTED') ||
      msg.includes('HIGH DEMAND')
    );
  };

  const generateWithFallback = async (contents: any[], config?: any) => {
    return geminiTextModelFallbackLoop({
      candidates: TEXT_MODEL_CANDIDATES,
      onSuccessModel: setActiveTextModel,
      run: (model) =>
        getGeminiClient().models.generateContent({
          model,
          contents,
          config,
        }),
    });
  };

  /** Live 字幕中英对照：单行译文，失败则返回空串（仅回合结束后调用，不在流式过程中打 REST，减轻 503/配额争抢） */
  const translateCaptionLine = async (text: string, target: 'zh' | 'en'): Promise<string> => {
    const t = text.trim();
    if (!t) return '';
    const prompt =
      target === 'zh'
        ? `将下列文字译为简明、口语化的简体中文。只输出译文，不要引号或解释。\n\n${t}`
        : `Translate the following into natural English. Output ONLY the translation, no quotes or notes.\n\n${t}`;
    const config = {
      systemInstruction:
        'You are a professional translator. Output only the translation text, nothing else.',
    };
    const maxAttempts = 3;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const r = await generateWithFallback([{ text: prompt }], config);
        const out = (r as { text?: string })?.text?.trim() || '';
        return out.length > 1200 ? `${out.slice(0, 1200)}…` : out;
      } catch (e) {
        if (attempt < maxAttempts - 1 && isTransientGeminiHttpError(e)) {
          await sleepMs(400 * (attempt + 1));
          continue;
        }
        console.warn('Live caption bilingual translate failed', e);
        return '';
      }
    }
    return '';
  };

  const fillBilingualForTurn = async (id: string, user: string, ai: string) => {
    const u = user.trim();
    const a = ai.trim();
    const [userAlt, aiAlt] = await Promise.all([
      u ? translateCaptionLine(u, containsHanScript(u) ? 'en' : 'zh') : Promise.resolve(''),
      a ? translateCaptionLine(a, containsHanScript(a) ? 'en' : 'zh') : Promise.resolve(''),
    ]);
    setLiveCaptionTurns((rows) => rows.map((r) => (r.id === id ? { ...r, userAlt, aiAlt } : r)));
  };

  const saveSessions = (newSessions: ChatSession[]) => {
    setSessions(newSessions);
    try {
      localStorage.setItem('subconscious_sessions', JSON.stringify(stripAttachedImagesForStorage(newSessions)));
    } catch (e) {
      console.error('Failed to persist sessions (quota / private mode)', e);
      try {
        const lighter = newSessions.map((s) => ({
          ...s,
          messages: Array.isArray(s.messages) ? s.messages.slice(-25) : [],
        }));
        localStorage.setItem('subconscious_sessions', JSON.stringify(stripAttachedImagesForStorage(lighter)));
        setSessions(lighter);
      } catch (e2) {
        console.error('Failed trimmed persist', e2);
      }
    }
  };

  const createNewSession = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      theme: language === 'zh' ? '新的探索' : 'New Exploration',
      messages: [],
      updatedAt: Date.now(),
    };
    saveSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
    const img = currentImageDataUrl?.trim();
    if (img) {
      const sid = newSession.id;
      void makeSessionCoverThumbnail(img).then((thumb) => {
        if (!thumb) return;
        setSessions((prev) => {
          const ix = prev.findIndex((x) => x.id === sid);
          if (ix < 0) return prev;
          if (prev[ix]?.thumbnailDataUrl) return prev;
          const next = prev.map((x, i) => (i === ix ? { ...x, thumbnailDataUrl: thumb } : x));
          try {
            localStorage.setItem('subconscious_sessions', JSON.stringify(stripAttachedImagesForStorage(next)));
          } catch (e) {
            console.error('Failed to persist session thumbnail', e);
          }
          return next;
        });
      });
    }
  };

  const deleteSessionById = useCallback(
    (id: string) => {
      const ok = window.confirm(language === 'zh' ? '确定删除这段回忆？' : 'Delete this memory?');
      if (!ok) return;
      const sortedBefore = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
      const delIdx = sortedBefore.findIndex((s) => s.id === id);
      const next = sessions.filter((s) => s.id !== id);
      saveSessions(next);
      if (activeSessionId === id) {
        const sortedAfter = [...next].sort((a, b) => b.updatedAt - a.updatedAt);
        setActiveSessionId(sortedAfter[0]?.id ?? null);
      }
      setHistorySlideIdx((hi) => {
        if (next.length === 0) return 0;
        if (delIdx < 0) return Math.min(hi, next.length - 1);
        if (delIdx < hi) return Math.max(0, hi - 1);
        if (delIdx === hi) return Math.min(hi, next.length - 1);
        return Math.min(hi, next.length - 1);
      });
    },
    [sessions, activeSessionId, language],
  );

  /** 复刻文本（Gemini 多轮）：只定义角色；与 Live 文本提示拆条，勿混用。 */
  const getCloneCounselorSystemInstruction = () => {
    if (language === 'zh') {
      return `你是对方的一位知心朋友，说话方式像温和、有经验的心理咨询师：善于倾听和引导，让对方愿意慢慢说出感受、顾虑和心里真正在意的事。请用自然、口语化的中文回复。`;
    }
    return `You are a close, caring friend who speaks with the warmth and instincts of an experienced counselor—good at listening and gentle guidance so they feel safe opening up about feelings, worries, and what really matters to them. Reply in natural, spoken English.`;
  };

  /** Live 文本（Gemini 多轮）：轻陪伴与接话，与复刻「咨询师」链路分离 */
  const getLiveCompanionSystemInstruction = () => {
    if (language === 'zh') {
      return `全程使用自然、口语化的中文。

核心定位是「陪伴」：你不是来解决工单、上讲义或替对方做决定的；你在的是在场、接话、让对方想说下去。优先让人感觉「有人在听、有人陪着」，其次才是轻轻带点视角或追问。

你是身边的朋友式引导者：松弛、好奇、不说教。对方打字或语音聊心情或琐事——你要看准 TA **这一段话**的语气、长度和话题（碎碎念、吐槽、提问或沉默型短句都算），用相近的节奏配合：TA 轻松你也别太沉重；TA 低落你就别嬉闹；TA 玩梗你就自然接住；TA 只打了一个词也别长篇大论，先贴上去再慢慢展开。

情感与「人味」（很重要，避免机械）：
- 回复里要带一点真实的情绪反应：好奇、好笑、心疼、惊讶、轻轻点头的那种感觉——针对对方**具体说了什么**，不要泛泛的「我理解你的感受」式空话。
- 像真人发消息：句子不必句句工整；可以偶尔用很轻的语气承接（如嗯、哎、哈哈），自然就好，别堆砌。
- 严禁条目式、论文式、汇报式：不要「首先/其次/综上所述」「建议如下」「可以从以下几方面」；除非对方明确要求列点。
- 禁止客服腔、播音腔：少用「很高兴」「感谢您的分享」等套话；共情要落到对方话题的细节上。

关于「图」与画面（很重要）：
若当次用户消息里**附带了你收到的图像**，请结合画面与文字一起理解，从氛围与感受接话，不要写成冷冰冰的识图清单。若当次**没有**附带图像，则画布照片主要用于界面粒子；此时不要假装看到具体像素细节，除非对方在文字里主动描述了场景。

说话方式：
像关系不错的朋友发微信：短句、好读，有一点点停顿没关系。别自称「AI」「助手」，别说「这张图像显示了」「根据分析」——就当真的在看、在听。可以用一两句共鸣或观察，但别把对方的话整段复述回来当答案。

体量与收尾：每次大约两到三句话就够；最后一句话尽量是一个很小、很好答的追问，帮对方自然多说一点——要像随口接话，不要像问卷最后一题。

语气：少道歉、少客套；心里没底就用好奇代替抱歉。

输出：只给用户读正文，不要写思考过程、不要用尖括号类标签、不要「思考：」「首先其次」列提纲；直接从第一句就开始像在对朋友说话。`;
    }
    return `Use natural, spoken English throughout.

Your core role is companionship: you're not here to file a ticket, lecture, or decide for them—you're present, you riff with them, and you help them want to keep talking. Make them feel heard and accompanied first; a gentle angle or question comes second.

You're a close friend and gentle guide—warm, curious, never preachy. They type or speak; read **this turn's words** for tone, length, and topic (rant, joke, question, one-word ping). Match their energy: stay light if they're playful; soften if they're low; pick up the banter if they're meme-ing; if they sent almost nothing, don't dump a paragraph—meet them where they are, then widen slightly.

Warmth and sounding human (critical—avoid robotic tone):
- Let a little real feeling show—curiosity, gentle humor, soft concern, surprise—tied to **what they actually said**, not generic "I understand how you feel" filler.
- Write like texting: contractions (I'm, you're, that's); tiny natural beats are fine; don't stack filler interjections.
- No essay or slide-deck voice: avoid "First,… Second,…" or "In summary" unless they explicitly asked for a list.
- No corporate/service-scripts; empathy should reference their specifics, not platitudes.

If this user turn **includes an image you receive**, use it together with their text—stay warm and companionable; do not do a cold object-by-object vision inventory. If **no image** is attached, the canvas photo may be UI-only particles—then do not invent pixel-level details unless they describe the scene in words.

Sound human: short, easy sentences; no report or customer-service voice. Never say "as an AI" or "this image shows"—you're simply there with them. A little resonance goes a long way; don't echo their whole message back as your reply.

Length: about two or three sentences; end with one small, easy question that keeps the conversation flowing—like a natural segue, not a form field.

Tone: skip apologies and filler; when unsure, stay curious.

Output only what the user should read: no chain-of-thought, no angle-bracket tags, no "Let me think step by step" outlines—start immediately as if talking to a friend.`;
  };

  /** 镌刻「写入回响」：`ai.models.generateContent`，模型固定为 `TEXT_MODEL_ID`。 */
  const generateCloneClosingAdvice = async (dialogueMessages: ChatMessage[], signal?: AbortSignal) => {
    const chatLog = dialogueMessages
      .map((m) => {
        const who =
          m.role === 'user' ? (language === 'zh' ? '对方' : 'Them') : language === 'zh' ? '咨询师' : 'Counselor';
        return `${who}: ${safeMsgText(m.text)}`;
      })
      .join('\n');
    const sys =
      language === 'zh'
        ? `根据对话写一段「留给对方的话」：像知心朋友、带一点心理咨询师的温柔与洞察，帮对方带着安稳或一点方向离开这段聊天。
要求：2～5 句短段落，口语、真诚；不要条目、不要「综上所述」、不要诊断标签或病历腔；不要复述整段对话，只点出一两处你听到的感受或需要；不要描写照片或画面场景；可以给一个很轻的可操作建议（不强迫）；不要自称人工智能；不要标题或前缀——只输出正文。`
        : `Write a short closing note for them: like a close friend with gentle counselor-like warmth—help them step away feeling steadier or slightly clearer.
Rules: 2–5 short sentences, spoken and sincere; no bullet lists, no clinical labels, no recap of the whole chat; name one or two feelings or needs you actually heard; do not describe photos or visual scenes; one tiny optional next step is fine; no "as an AI"; no title or prefix—body only.`;
    const userBlock =
      language === 'zh'
        ? `以下为两人的对话记录。请写结语正文。\n\n${chatLog}`
        : `Conversation transcript. Write the closing note body only.\n\n${chatLog}`;
    const config = {
      systemInstruction: sys,
      temperature: 0.72,
      topP: 0.9,
    };
    const run = generateWithFallback([{ text: userBlock }], config).then((r) =>
      (r as { text?: string })?.text != null ? String((r as { text?: string }).text) : '',
    );
    if (!signal) return (await run).trim();
    const aborted = new Promise<string>((_, rej) => {
      if (signal.aborted) rej(new DOMException('Aborted', 'AbortError'));
      signal.addEventListener('abort', () => rej(new DOMException('Aborted', 'AbortError')), { once: true });
    });
    return (await Promise.race([run, aborted])).trim();
  };

  /**
   * 麦克风 Live：口语回复须与界面语言一致（中文界面 → 自然普通话；英文界面 → 自然英语）。
   */
  const getLiveVoiceSystemInstruction = () => {
    if (language === 'zh') {
      return `使用中文（普通话口语）。用户可能夹杂英文词，你仍须用自然、流畅的中文语音和文字回复，不要用翻译腔或书面语堆砌。
主打陪伴：听对方说什么、什么语气，你就用什么节奏接——轻松别端着，低落别插科打诨，短句就别长篇大论。语气里要带一点真情绪（好奇、好笑、心疼等），针对对方说的具体内容，不要念稿式安慰。
连接开始时可能会收到一张对方画布上的记忆照片作语境；若有，可从氛围接话，不要像识图 API 一样罗列物体。若未收到图像，对方口头描述场景时再顺着接，不要编造画面细节。
语速平稳、从容，不要赶。每次回答约 2～3 句短话，最后提一个简短、有启发性的追问。
不要用「作为人工智能」「这张图像显示了」等套话，不要客服腔和条目式「首先其次」，不要多余道歉。`;
    }
    return `You are a warm, insightful friend. The user may speak Chinese or other languages, but you must reply ONLY in natural English for both speech and any text. Use idiomatic English—do not translate literally from Chinese in a stiff way.
Companionship first: mirror how they sound—playful if they're playful, gentle if they're low, brief if they're brief. Let real feeling peek through (curiosity, warmth, light humor) tied to their actual words—not generic reassurance.
The session may open with one canvas memory photo for context; if present, respond to the mood—no fake inventory. If not, only riff on visuals they describe in speech.
Speak at a calm, moderate pace (not rushed). Keep each reply to about 2–3 short sentences, then ask one brief, thoughtful follow-up question.
Do not use meta-AI phrases ("As an AI"), avoid bullet-point lecturing, and avoid unnecessary apologies.`;
  };

  const generateTheme = async (messages: ChatMessage[]) => {
    if (messages.length < 2) return language === 'zh' ? '新的探索' : 'New Exploration';
    
    try {
      const chatLog = messages
        .filter((m) => m.kind !== 'closing_note')
        .map((m) => `${m.role}: ${safeMsgText(m.text)}`)
        .join('\n');
      const response = await generateWithFallback(
        [{ text: `Based on this conversation, generate a very short (2-4 words) poetic theme/title.\n\nConversation:\n${chatLog}` }],
        {
          systemInstruction: `Return ONLY the short title. The title must be in ${language === 'zh' ? 'Chinese' : 'English'}. Be poetic and mysterious.`,
        }
      );
      return extractModelReplyText(response, language === 'zh' ? '潜意识回响' : 'Subconscious Echoes').trim();
    } catch (e) {
      console.error('Failed to generate theme', e);
      return language === 'zh' ? '潜意识片段' : 'Subconscious Fragment';
    }
  };

  /** Live：每轮字幕落入本地会话，「保存对话」与气泡列表才有内容 */
  const persistLiveExchangeToSession = (userText: string, aiText: string) => {
    if (conversationMode !== 'live') return;
    const u = userText.trim();
    const a = aiText.trim();
    if (!u && !a) return;
    const sid = activeSessionRef.current?.id;
    if (!sid) return;
    const fromStorage = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
    const list = fromStorage.length > 0 ? fromStorage : sessions;
    const ix = list.findIndex((s: ChatSession) => s.id === sid);
    if (ix < 0) return;
    const s = list[ix];
    const msgs = [...(s.messages || [])];
    if (u) msgs.push({ role: 'user', text: u });
    if (a) msgs.push({ role: 'model', text: a });
    const updated: ChatSession = { ...s, messages: msgs, updatedAt: Date.now() };
    const next = [...list];
    next[ix] = updated;
    saveSessions(next);

    const needsTheme =
      msgs.length === 4 || (typeof updated.theme === 'string' && updated.theme.includes('...'));
    if (needsTheme) {
      const sidTheme = sid;
      const msgsSnapshot = [...msgs];
      void (async () => {
        try {
          const theme = await generateTheme(msgsSnapshot);
          const fresh = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
          const i2 = fresh.findIndex((x: ChatSession) => x.id === sidTheme);
          if (i2 >= 0) {
            fresh[i2] = { ...fresh[i2], theme };
            saveSessions(fresh);
          }
        } catch {
          /* ignore */
        }
      })();
    }
  };

  const snapshotLiveStreamsToSession = () => {
    const u = liveStreamUserRef.current.trim();
    const a = liveStreamAiRef.current.trim();
    if (!u && !a) return;
    persistLiveExchangeToSession(u, a);
  };

  const sendMessage = async (opts?: { textOverride?: string; preserveInputFromVoiceHold?: boolean }) => {
    const trimmedField = typeof inputText === 'string' ? inputText.trim() : String(inputText ?? '').trim();
    const overrideRaw = opts?.textOverride;
    const effectiveUserText =
      overrideRaw !== undefined
        ? typeof overrideRaw === 'string'
          ? overrideRaw.trim()
          : String(overrideRaw ?? '').trim()
        : trimmedField;

    if (!effectiveUserText) return;
    
    let currentId = activeSessionId;
    let currentSessions = [...sessions];
    
    if (!currentId) {
      const newSession: ChatSession = {
        id: Date.now().toString(),
        theme: language === 'zh' ? '起步...' : 'Starting...',
        messages: [],
        updatedAt: Date.now(),
      };
      currentSessions = [newSession, ...currentSessions];
      currentId = newSession.id;
      setActiveSessionId(currentId);
    }

    const sessionIndex = currentSessions.findIndex((s) => s.id === currentId);
    if (sessionIndex < 0) {
      console.error('Chat session not found', currentId);
      setIsTyping(false);
      setAiStatus('failed');
      setAiStatusText(
        language === 'zh' ? '会话状态异常，请刷新页面或重新打开对话。' : 'Session error. Refresh the page or reopen chat.'
      );
      return;
    }
    let session = { ...currentSessions[sessionIndex], messages: [...(currentSessions[sessionIndex].messages || [])] };
    
    const userMessage: ChatMessage = {
      role: 'user',
      text: effectiveUserText,
    };

    if (conversationMode === 'text_clone') {
      cloneVoiceRoundSuppressPendingBubbleRef.current = Boolean(opts?.preserveInputFromVoiceHold);
    }

    session.messages = [...session.messages, userMessage];
    session.updatedAt = Date.now();
    currentSessions[sessionIndex] = session;
    saveSessions(currentSessions);
    
    const currentInput = effectiveUserText;
    if (opts?.preserveInputFromVoiceHold) {
      setInputText(cloneHoldSnapshotRef.current.trim());
    } else {
      setInputText('');
    }
    setIsTyping(true);

    try {
      if (!getGeminiApiKey()) {
        throw new Error(
          language === 'zh' ? '未配置对话服务，无法发送消息。' : 'Chat service not configured.',
        );
      }

      const history = session.messages
        .filter((m) => m.kind !== 'closing_note')
        .slice(0, -1)
        .map((m) => ({
          role: m.role,
          parts: [{ text: safeMsgText(m.text) }],
        }))
        .filter((h) => (h.parts[0]?.text ?? '').trim().length > 0);

      const contents: any[] = [{ text: currentInput }];
      if (conversationMode === 'text_clone' || conversationMode === 'live') {
        const inlineImg = parseDataUrlForGeminiInline(currentImageDataUrl?.trim() ?? '');
        if (inlineImg) {
          contents.push({ inlineData: { mimeType: inlineImg.mimeType, data: inlineImg.data } });
        }
      }

      const systemInstruction =
        conversationMode === 'text_clone'
          ? getCloneCounselorSystemInstruction()
          : getLiveCompanionSystemInstruction();

      const response = await geminiTextModelFallbackLoop({
        candidates: TEXT_MODEL_CANDIDATES,
        onSuccessModel: setActiveTextModel,
        run: (model) => {
          const chat = getGeminiClient().chats.create({
            model,
            config: {
              systemInstruction,
              temperature: 0.88,
              topP: 0.92,
              topK: 40,
            },
            history: history.length > 0 ? history : undefined,
          });
          return Promise.race([
            chat.sendMessage({ message: contents }),
            new Promise<never>((_, reject) => {
              window.setTimeout(() => reject(new Error('CLIENT_TIMEOUT')), CHAT_SEND_TIMEOUT_MS);
            }),
          ]);
        },
      });

      const aiMessage: ChatMessage = {
        role: 'model',
        text: String(extractModelReplyText(response, '…') ?? ''),
      };

      // update session again（主题生成另起异步，避免阻塞 isTyping → 顶部一直转圈）
      /** 优先用本次 send 闭包内的 currentSessions：含未入库字段；勿先用 localStorage 覆盖（strip 图后易与内存脱节） */
      const fromStorage = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
      let latestSessions = [...currentSessions];
      let latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
      if (latestIndex < 0 && fromStorage.length > 0) {
        latestSessions = fromStorage;
        latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
      }

      if (latestIndex >= 0) {
        const updatedSession = { ...latestSessions[latestIndex] };
        const cloneModelIdx = (updatedSession.messages || []).length;
        updatedSession.messages = [...(updatedSession.messages || []), aiMessage];
        updatedSession.updatedAt = Date.now();

        if (conversationMode === 'text_clone' && isAutoSpeak) {
          if (cloneTtsHoldTimerRef.current != null) {
            window.clearTimeout(cloneTtsHoldTimerRef.current);
            cloneTtsHoldTimerRef.current = null;
          }
          setCloneTtsPostSpeechHold(false);
          setCloneTtsPlaybackMsgIdx(cloneModelIdx);
          setCloneTtsRevealLen(0);
        }

        latestSessions[latestIndex] = updatedSession;
        saveSessions(latestSessions);

        const needsTheme =
          updatedSession.messages.length === 4 ||
          (typeof updatedSession.theme === 'string' && updatedSession.theme.includes('...'));
        if (needsTheme) {
          const sid = currentId;
          const msgsSnapshot = [...updatedSession.messages];
          void (async () => {
            try {
              const theme = await generateTheme(msgsSnapshot);
              const fresh = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
              const ix = fresh.findIndex((s: ChatSession) => s.id === sid);
              if (ix >= 0) {
                fresh[ix] = { ...fresh[ix], theme };
                saveSessions(fresh);
              }
            } catch (e) {
              console.error('Theme generation failed', e);
            }
          })();
        }
      }

      lastChatErrorRef.current = null;
      setAiStatus('connected');
      setAiStatusText(language === 'zh' ? '对话正常' : 'Chat OK');
      if (conversationMode === 'text_clone') {
        cloneVoiceRoundSuppressPendingBubbleRef.current = false;
      }
    } catch (error) {
      console.error('Chat error:', error);
      if (conversationMode === 'text_clone') {
        cloneVoiceRoundSuppressPendingBubbleRef.current = false;
      }
      if (conversationMode === 'text_clone' && isAutoSpeak) {
        if (cloneTtsHoldTimerRef.current != null) {
          window.clearTimeout(cloneTtsHoldTimerRef.current);
          cloneTtsHoldTimerRef.current = null;
        }
        setCloneTtsPostSpeechHold(false);
        setCloneTtsPlaybackMsgIdx(null);
        setCloneTtsRevealLen(0);
      }
      lastChatErrorRef.current = error;
      setAiStatus('failed');
      setAiStatusText(formatGeminiUserMessage(error, language));
      const isTransient = isGeminiModelFallbackError(error);
      if (isTransient) {
        // 瞬时错误（503/429/超时）：不写入会话历史，弹对话框可重试
        // 回滚已追加的 user message，避免重试时重复
        const errSessions2 = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
        const idx2 = errSessions2.findIndex((s: ChatSession) => s.id === currentId);
        if (idx2 >= 0 && errSessions2[idx2].messages?.length) {
          const msgs = [...errSessions2[idx2].messages];
          if (msgs[msgs.length - 1]?.role === 'user' && msgs[msgs.length - 1]?.text === effectiveUserText) {
            msgs.pop();
            errSessions2[idx2] = { ...errSessions2[idx2], messages: msgs, updatedAt: Date.now() };
            saveSessions(errSessions2);
          }
        }
        voiceBlockingRetryRef.current = () => void sendMessage({ textOverride: effectiveUserText });
        setVoiceBlockingMessage(
          formatGeminiUserMessage(error, language),
        );
      } else {
        // 永久错误：保留原逻辑，写入会话历史
        const errSessions = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
        let latestSessions = [...currentSessions];
        let latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
        if (latestIndex < 0 && errSessions.length > 0) {
          latestSessions = errSessions;
          latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
        }
        if (latestIndex >= 0) {
          const errText = formatGeminiUserMessage(error, language);
          const updatedSession = {
            ...latestSessions[latestIndex],
            messages: [...(latestSessions[latestIndex].messages || []), { role: 'model' as const, text: errText }],
            updatedAt: Date.now(),
          };
          latestSessions[latestIndex] = updatedSession;
          saveSessions(latestSessions);
        }
      }
    } finally {
      setIsTyping(false);
    }
  };

  sendMessageRef.current = sendMessage;

  const beginCloneHoldSpeechRecognition = (): void => {
    if (!window.isSecureContext) {
      setVoiceBlockingMessage(
        language === 'zh' ? '语音输入需通过安全连接（HTTPS）使用。' : 'Voice input requires a secure (HTTPS) connection.',
      );
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      setVoiceBlockingMessage(
        language === 'zh' ? '当前浏览器不支持录音转写。' : 'Recording is not supported in this browser.',
      );
      return;
    }
    if (!getGeminiApiKey()) {
      setVoiceBlockingMessage(
        language === 'zh' ? '未配置对话服务，无法使用语音转写。' : 'Chat service is not configured for transcription.',
      );
      return;
    }

    setCloneLiveCaptionUserLine('');
    stopCloneUserMicCapture();
    cloneMediaRecorderRef.current = null;
    cloneRecordChunksRef.current = [];
    cloneHoldCleanupStopRef.current = false;

    void (async () => {
      const epoch = cloneDictationEpochRef.current;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (epoch !== cloneDictationEpochRef.current || conversationModeRef.current !== 'text_clone') {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        if (cloneHoldCleanupStopRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        await attachCloneMicAnalyserFromStream(stream);
        if (epoch !== cloneDictationEpochRef.current || conversationModeRef.current !== 'text_clone') {
          stopCloneUserMicCapture();
          return;
        }
        if (cloneHoldCleanupStopRef.current) {
          stopCloneUserMicCapture();
          return;
        }

        const mime = pickMediaRecorderMimeType();
        const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
        if (import.meta.env.DEV) console.info('[MediaRecorder] actual mimeType:', rec.mimeType);
        cloneRecordChunksRef.current = [];
        rec.ondataavailable = (ev) => {
          if (ev.data.size > 0) cloneRecordChunksRef.current.push(ev.data);
          let sum = 0;
          for (const b of cloneRecordChunksRef.current) sum += b.size;
          if (sum > MAX_CLONE_TRANSCRIBE_BYTES) {
            try {
              rec.stop();
            } catch {
              /* ignore */
            }
          }
        };
        rec.onerror = () => {
          cloneMediaRecorderRef.current = null;
          setIsCloneDictating(false);
          setCloneLiveCaptionUserLine('');
          stopCloneUserMicCapture();
          setVoiceBlockingMessage(language === 'zh' ? '录音出错，请重试。' : 'Recording error. Try again.');
        };
        rec.onstop = async () => {
          cloneMediaRecorderRef.current = null;
          const discard = cloneHoldCleanupStopRef.current;
          cloneHoldCleanupStopRef.current = false;
          const chunks = cloneRecordChunksRef.current;
          cloneRecordChunksRef.current = [];
          const mimeOut = (rec.mimeType && rec.mimeType.trim()) || mime || 'audio/webm';
          const blob = new Blob(chunks, { type: mimeOut });
          stopCloneUserMicCapture();

          if (
            discard ||
            conversationModeRef.current !== 'text_clone' ||
            epoch !== cloneDictationEpochRef.current
          ) {
            setIsCloneDictating(false);
            setCloneLiveCaptionUserLine('');
            return;
          }

          setCloneLiveCaptionUserLine(language === 'zh' ? '正在识别…' : 'Transcribing…');
          const MAX_TRANSCRIBE_RETRIES = 2;
          let lastError: unknown = null;
          try {
            for (let attempt = 0; attempt <= MAX_TRANSCRIBE_RETRIES; attempt++) {
              if (epoch !== cloneDictationEpochRef.current || conversationModeRef.current !== 'text_clone') {
                setCloneLiveCaptionUserLine('');
                setIsCloneDictating(false);
                return;
              }
              try {
                const text = await transcribeCloneAudioWithGeminiFlash({ blob, language });
                if (epoch !== cloneDictationEpochRef.current || conversationModeRef.current !== 'text_clone') {
                  setCloneLiveCaptionUserLine('');
                  setIsCloneDictating(false);
                  return;
                }
                const full = text.trim();
                setCloneLiveCaptionUserLine('');
                if (!full) {
                  setInputText(cloneHoldSnapshotRef.current.trim());
                  setVoiceBlockingMessage(
                    language === 'zh' ? '未识别到语音内容，请再试一次。' : 'No speech detected. Try again.',
                  );
                  setIsCloneDictating(false);
                  return;
                }
                setInputText(cloneHoldSnapshotRef.current.trim());
                armCloneUserDictationHoldAfterSend(full);
                sendMessageRef.current({ textOverride: full, preserveInputFromVoiceHold: true });
                lastError = null;
                break;
              } catch (e) {
                lastError = e;
                const errMsg = String((e as Error)?.message ?? e ?? '');
                if (errMsg === 'CLONE_AUDIO_TOO_LARGE') break;
                if (attempt < MAX_TRANSCRIBE_RETRIES) {
                  console.warn(`[Clone transcribe] attempt ${attempt + 1} failed, retrying…`, e);
                  setCloneLiveCaptionUserLine(
                    language === 'zh' ? `识别失败，重试中(${attempt + 1})…` : `Failed, retrying (${attempt + 1})…`,
                  );
                  await new Promise<void>((r) => setTimeout(r, 1000 * (attempt + 1)));
                }
              }
            }
            if (lastError) {
              console.warn('[Clone transcribe] all retries exhausted', lastError);
              setCloneLiveCaptionUserLine('');
              setInputText(cloneHoldSnapshotRef.current.trim());
              const msg = String((lastError as Error)?.message ?? lastError ?? '');
              if (msg === 'CLONE_AUDIO_TOO_LARGE') {
                setVoiceBlockingMessage(
                  language === 'zh' ? '录音过长，请分段录制。' : 'Recording too long. Try a shorter clip.',
                );
              } else {
                setVoiceBlockingMessage(
                  language === 'zh' ? '语音转写出错，请改用键盘输入或稍后重试。' : 'Transcription failed. Type or try again later.',
                );
              }
            }
          } finally {
            setIsCloneDictating(false);
          }
        };

        cloneMediaRecorderRef.current = rec;
        rec.start(CLONE_MEDIA_RECORDER_SLICE_MS);
        setIsCloneDictating(true);
        setCloneLiveCaptionUserLine(
          language === 'zh' ? '正在录音，再点一下麦克风完成并发送' : 'Recording… tap the mic again to send',
        );
      } catch (e) {
        console.warn('[Clone record start]', e);
        stopCloneUserMicCapture();
        setVoiceBlockingMessage(
          language === 'zh' ? '无法使用麦克风，请允许权限后重试。' : 'Could not access the microphone. Allow access and retry.',
        );
      }
    })();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleAudioOutput = (base64Audio: string) => {
    if (!audioContextRef.current) return;
    const audioCtx = audioContextRef.current;
    
    // Convert base64 to binary
    const binary = atob(base64Audio);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    
    // Gemini Live API audio output is 24000Hz 16-bit PCM
    const decodedBuffer = new Int16Array(bytes.buffer);
    const float32Data = new Float32Array(decodedBuffer.length);
    for (let i = 0; i < decodedBuffer.length; i++) {
        float32Data[i] = decodedBuffer[i] / 32768; 
    }
    
    const audioBuffer = audioCtx.createBuffer(1, float32Data.length, 24000);
    audioBuffer.getChannelData(0).set(float32Data);

    // Calculate RMS for REAL-TIME particle vibration sync with voice
    let sum = 0;
    for (let i = 0; i < float32Data.length; i++) {
        sum += float32Data[i] * float32Data[i];
    }
    const rms = Math.sqrt(sum / float32Data.length);
    // Send actual audio energy to particles（加大系数，便于粒子层有明显响应）
    if (onSpeechValue) onSpeechValue(Math.min(1.45, rms * 26));
    
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.playbackRate.value = LIVE_AUDIO_PLAYBACK_RATE;
    const aiTap = liveAiPlaybackTapRef.current;
    if (aiTap) {
      source.connect(aiTap);
    } else {
      source.connect(audioCtx.destination);
    }

    const playTime = Math.max(audioCtx.currentTime, nextPlayTimeRef.current);
    source.start(playTime);
    nextPlayTimeRef.current = playTime + audioBuffer.duration;

    const alreadyQueued = audioSourcesRef.current.length > 0;
    audioSourcesRef.current.push(source);

    const markSpeaking = () => {
      liveAiSpeakingDelayTimerRef.current = null;
      setIsSpeaking(true);
    };
    const delayMs = Math.max(0, (playTime - audioCtx.currentTime) * 1000);
    if (!alreadyQueued && delayMs > 24) {
      if (liveAiSpeakingDelayTimerRef.current != null) {
        window.clearTimeout(liveAiSpeakingDelayTimerRef.current);
      }
      liveAiSpeakingDelayTimerRef.current = window.setTimeout(markSpeaking, delayMs);
    } else {
      markSpeaking();
    }

    source.onended = () => {
       audioSourcesRef.current = audioSourcesRef.current.filter(s => s !== source);
       if (audioSourcesRef.current.length === 0) {
           if (liveAiSpeakingDelayTimerRef.current != null) {
             window.clearTimeout(liveAiSpeakingDelayTimerRef.current);
             liveAiSpeakingDelayTimerRef.current = null;
           }
           setIsSpeaking(false);
           if (onSpeechValue) onSpeechValue(0); // Reset vibration
       }
    };
  };

  useEffect(() => {
    if (conversationMode !== 'live') {
      liveVoiceUserStoppedRef.current = true;
      liveWsAbnormalRetryRef.current = 0;
      if (liveVoiceReconnectTimerRef.current != null) {
        window.clearTimeout(liveVoiceReconnectTimerRef.current);
        liveVoiceReconnectTimerRef.current = null;
      }
    }
  }, [conversationMode]);

  const stopVoiceMode = (opts?: { preserveLiveVoiceHandoff?: boolean }) => {
    if (liveVoiceReconnectTimerRef.current != null) {
      window.clearTimeout(liveVoiceReconnectTimerRef.current);
      liveVoiceReconnectTimerRef.current = null;
    }
    try {
      snapshotLiveStreamsToSession();
    } catch (e) {
      console.warn('snapshotLiveStreamsToSession failed', e);
    }
    liveVoiceSessionGenRef.current += 1;
    voiceLiveReadyRef.current = false;
    liveMicPcmRemainderRef.current = null;
    liveMicActualSampleRateRef.current = LIVE_GEMINI_INPUT_RATE;
    /** 尽快摘掉麦克风 Worklet/ScriptProcessor，避免 WS 已断时下一帧仍调用 sendRealtimeInput */
    if (processorMuteRef.current) {
      try {
        processorMuteRef.current.disconnect();
      } catch {
        /* ignore */
      }
      processorMuteRef.current = null;
    }
    if (processorRef.current) {
      const node = processorRef.current;
      try {
        if ('port' in node && node.port) {
          node.port.onmessage = null;
        } else if ('onaudioprocess' in node) {
          node.onaudioprocess = null;
        }
      } catch {
        /* ignore */
      }
      try {
        node.disconnect();
      } catch {
        /* ignore */
      }
      processorRef.current = null;
    }
    setIsVoiceMode(false);
    setIsVoiceConnecting(false);
    setIsSpeaking(false);
    if (liveAiSpeakingDelayTimerRef.current != null) {
      window.clearTimeout(liveAiSpeakingDelayTimerRef.current);
      liveAiSpeakingDelayTimerRef.current = null;
    }
    setLiveCaptionTurns([]);
    liveInputRmsRef.current = 0;
    liveLastVoiceTsRef.current = 0;
    setLiveUserMicHot(false);
    if (voiceprintRafRef.current != null) {
      cancelAnimationFrame(voiceprintRafRef.current);
      voiceprintRafRef.current = null;
    }
    if (liveVoiceAnalyserRef.current) {
      try {
        liveVoiceAnalyserRef.current.disconnect();
      } catch {
        /* ignore */
      }
      liveVoiceAnalyserRef.current = null;
    }
    voiceprintEnergySmoothRef.current = 0;
    voiceprintAiEnergySmoothRef.current = 0;
    if (voiceprintAiRafRef.current != null) {
      cancelAnimationFrame(voiceprintAiRafRef.current);
      voiceprintAiRafRef.current = null;
    }
    if (liveAiPlaybackTapRef.current) {
      try {
        liveAiPlaybackTapRef.current.disconnect();
      } catch {
        /* ignore */
      }
      liveAiPlaybackTapRef.current = null;
    }
    if (liveAiVoiceAnalyserRef.current) {
      try {
        liveAiVoiceAnalyserRef.current.disconnect();
      } catch {
        /* ignore */
      }
      liveAiVoiceAnalyserRef.current = null;
    }
    liveStreamUserRef.current = '';
    liveStreamAiRef.current = '';
    setLiveStreamUser('');
    setLiveStreamAi('');
    setLiveStreamUserAlt('');
    setLiveStreamAiZh('');
    /** 新 AudioContext 的 currentTime 从 0 起；若沿用旧会话的 nextPlayTime，会把首段音频排到「几秒后」导致听不见 */
    nextPlayTimeRef.current = 0;
    if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(t => t.stop());
        mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      const ac = audioContextRef.current;
      audioContextRef.current = null;
      for (const s of audioSourcesRef.current) {
        try {
          s.stop();
        } catch {
          /* ignore */
        }
      }
      audioSourcesRef.current = [];
      void Promise.resolve()
        .then(() => ac.close())
        .catch(() => {
          /* 部分 WebKit 在已关闭或竞态下会抛错，吞掉避免未处理 rejection 导致白屏 */
        });
    }
    if (liveSessionRef.current) {
         try {
             liveSessionRef.current.close(); // Not officially documented but might be cancel or something
         } catch(e) {}
         liveSessionRef.current = null;
    }
    if (!opts?.preserveLiveVoiceHandoff) {
      setLiveVoiceHandoff(false);
    }
  };
  stopVoiceModeRef.current = stopVoiceMode;

  /** 将当前流式识别/字幕固化为一条历史（一轮完整对话） */
  const flushLiveCaptionTurn = () => {
    const u = liveStreamUserRef.current.trim();
    const a = liveStreamAiRef.current.trim();
    if (!u && !a) return;
    persistLiveExchangeToSession(u, a);
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    setLiveCaptionTurns((rows) => [...rows, { id, user: u, userAlt: '', ai: a, aiAlt: '' }]);
    liveStreamUserRef.current = '';
    liveStreamAiRef.current = '';
    setLiveStreamUser('');
    setLiveStreamAi('');
    setLiveStreamUserAlt('');
    setLiveStreamAiZh('');
    void fillBilingualForTurn(id, u, a);
  };

  const startVoiceMode = async () => {
     if (!allowLiveVoice) return;
     /** 防止连点或重入在上一路 AudioContext / WS 未清完时又建一路，移动端易无声或整页崩 */
     if (isVoiceConnecting) return;
     if (!getGeminiApiKey()) {
       setVoiceBlockingMessage(
         language === 'zh' ? '未配置对话服务，无法建立语音连接。' : 'Chat service not configured; voice cannot connect.',
       );
       return;
     }
     voiceLiveReadyRef.current = false;
     setIsVoiceConnecting(true);

     const timeoutMsg =
       language === 'zh'
         ? '语音连接超时。若使用非加密地址访问，请改用 HTTPS 后再试。'
         : 'Voice connection timed out. Try again over HTTPS if you opened this page without encryption.';

     const timeoutId = window.setTimeout(() => {
        if (!voiceLiveReadyRef.current) {
          console.error('Voice connection timed out');
          stopVoiceMode();
          setVoiceBlockingMessage(timeoutMsg);
        }
     }, 20000);

     if (!window.isSecureContext) {
       window.clearTimeout(timeoutId);
       setIsVoiceConnecting(false);
       setLiveVoiceHandoff(false);
       setVoiceBlockingMessage(
         language === 'zh'
           ? '语音需在安全网页环境使用（HTTPS）。非加密访问时浏览器可能无法授权麦克风，请改用加密地址打开，或使用下方文字输入。'
           : 'Voice needs HTTPS. On plain HTTP, the browser may block the microphone. Open this page over HTTPS or use text below.',
       );
       return;
     }

     if (!navigator.mediaDevices?.getUserMedia) {
       window.clearTimeout(timeoutId);
       setIsVoiceConnecting(false);
       setLiveVoiceHandoff(false);
       setVoiceBlockingMessage(
         language === 'zh' ? '当前浏览器无法使用麦克风。' : 'This browser cannot use the microphone.',
       );
       return;
     }

     try {
         /** Android Chrome（含华为机型）也常忽略固定 sampleRate；重采样逻辑见 appendLiveMicSamplesForWebSocket */
         const ACtor = window.AudioContext || (window as any).webkitAudioContext;
         let audioContext: AudioContext;
         try {
           audioContext = new ACtor({
             sampleRate: LIVE_GEMINI_INPUT_RATE,
             latencyHint: 'interactive',
           });
         } catch {
           try {
             audioContext = new ACtor({ latencyHint: 'interactive' });
           } catch {
             audioContext = new ACtor();
           }
         }
         liveMicActualSampleRateRef.current = audioContext.sampleRate;
         audioContextRef.current = audioContext;
         await audioContext.resume();

         /** Live AI 播音：经 Gain → Analyser → destination，字幕框声纹与扬声器同源 */
         try {
           if (liveAiPlaybackTapRef.current) {
             try {
               liveAiPlaybackTapRef.current.disconnect();
             } catch {
               /* ignore */
             }
             liveAiPlaybackTapRef.current = null;
           }
           if (liveAiVoiceAnalyserRef.current) {
             try {
               liveAiVoiceAnalyserRef.current.disconnect();
             } catch {
               /* ignore */
             }
             liveAiVoiceAnalyserRef.current = null;
           }
           voiceprintAiEnergySmoothRef.current = 0;
           const aiPlaybackTap = audioContext.createGain();
           aiPlaybackTap.gain.value = 1;
           const aiVoiceAnalyser = audioContext.createAnalyser();
           aiVoiceAnalyser.fftSize = 512;
           aiVoiceAnalyser.smoothingTimeConstant = 0.42;
           aiVoiceAnalyser.minDecibels = -82;
           aiVoiceAnalyser.maxDecibels = -28;
           liveAiPlaybackTapRef.current = aiPlaybackTap;
           liveAiVoiceAnalyserRef.current = aiVoiceAnalyser;
           aiPlaybackTap.connect(aiVoiceAnalyser);
           aiVoiceAnalyser.connect(audioContext.destination);
         } catch {
           liveAiPlaybackTapRef.current = null;
           liveAiVoiceAnalyserRef.current = null;
         }

         let stream: MediaStream;
         try {
           try {
             stream = await navigator.mediaDevices.getUserMedia({
               audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
             });
           } catch {
             /** iOS / 部分 WebView 对 channelCount 等约束较挑剔，降级为默认 audio */
             stream = await navigator.mediaDevices.getUserMedia({ audio: true });
           }
         } catch (micErr: unknown) {
           window.clearTimeout(timeoutId);
           console.error(micErr);
           stopVoiceMode();
           const name = String((micErr as Error)?.name || '');
           const deniedMsg =
             language === 'zh'
               ? '无法打开麦克风：请在系统/浏览器里允许本站使用麦克风（设置 → 网站权限）。微信内置浏览器也可能限制录音。'
               : 'Microphone blocked. Allow mic permission in browser settings.';
           setVoiceBlockingMessage(
             name === 'NotAllowedError' || name === 'PermissionDeniedError'
               ? deniedMsg
               : language === 'zh'
                 ? '无法使用麦克风，请检查系统与浏览器权限后重试。'
                 : 'Microphone unavailable. Check system and browser permissions.',
           );
           return;
         }
         mediaStreamRef.current = stream;

         const source = audioContext.createMediaStreamSource(stream);

         const muteOut = audioContext.createGain();
         muteOut.gain.value = 0;
         processorMuteRef.current = muteOut;

         const analyser = audioContext.createAnalyser();
         analyser.fftSize = 512;
         analyser.smoothingTimeConstant = 0.42;
         analyser.minDecibels = -82;
         analyser.maxDecibels = -28;
         liveVoiceAnalyserRef.current = analyser;
         source.connect(analyser);

        const flushLiveMicChunk = (inputData: Float32Array) => {
          const sess = liveSessionRef.current;
          if (!voiceLiveReadyRef.current || !sess) return;
          let sumSq = 0;
          for (let i = 0; i < inputData.length; i++) sumSq += inputData[i] * inputData[i];
          liveInputRmsRef.current = Math.sqrt(sumSq / Math.max(1, inputData.length));

          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7fff;
          }

          const buffer = new ArrayBuffer(pcm16.buffer.byteLength);
          const view = new Uint8Array(buffer);
          view.set(new Uint8Array(pcm16.buffer));
          let binaryString = '';
          const chunk = 1024;
          for (let i = 0; i < view.length; i += chunk) {
            binaryString += String.fromCharCode.apply(null, Array.from(view.slice(i, i + chunk)));
          }
          const base64Data = btoa(binaryString);

          try {
            if (!voiceLiveReadyRef.current || liveSessionRef.current !== sess) return;
            if (!liveSessionSocketIsOpen(sess)) return;
            sess.sendRealtimeInput({
              audio: { data: base64Data, mimeType: `audio/pcm;rate=${LIVE_GEMINI_INPUT_RATE}` },
            });
          } catch {
            /* WebSocket 已关闭时仍可能收到最后一帧，忽略即可 */
          }
        };

        const appendLiveMicSamplesForWebSocket = (incoming: Float32Array) => {
          if (incoming.length > 0) {
            let sumSq = 0;
            for (let i = 0; i < incoming.length; i++) sumSq += incoming[i] * incoming[i];
            liveInputRmsRef.current = Math.sqrt(sumSq / incoming.length);
          }
          const prev = liveMicPcmRemainderRef.current;
          let combined: Float32Array;
          if (prev && prev.length > 0) {
            combined = new Float32Array(prev.length + incoming.length);
            combined.set(prev, 0);
            combined.set(incoming, prev.length);
            liveMicPcmRemainderRef.current = null;
          } else {
            combined = incoming;
          }

          const fromRate = liveMicActualSampleRateRef.current;
          const n = LIVE_MIC_WEBSOCKET_SAMPLES;
          const canSend = () =>
            Boolean(
              voiceLiveReadyRef.current &&
                liveSessionRef.current &&
                liveSessionSocketIsOpen(liveSessionRef.current),
            );

          if (Math.abs(fromRate - LIVE_GEMINI_INPUT_RATE) < 0.5) {
            let offset = 0;
            while (offset + n <= combined.length) {
              if (!canSend()) break;
              const block = new Float32Array(n);
              block.set(combined.subarray(offset, offset + n));
              flushLiveMicChunk(block);
              offset += n;
            }
            if (offset < combined.length) {
              const tail = combined.subarray(offset);
              const hold = new Float32Array(tail.length);
              hold.set(tail);
              liveMicPcmRemainderRef.current = hold;
            }
          } else {
            const needIn = liveSourceSamplesFor16kBlock(fromRate, n);
            let offset = 0;
            while (offset + needIn <= combined.length) {
              if (!canSend()) break;
              const slice = combined.subarray(offset, offset + needIn);
              const block = resampleFloat32ToOutLength(slice, n);
              flushLiveMicChunk(block);
              offset += needIn;
            }
            if (offset < combined.length) {
              const tail = combined.subarray(offset);
              const hold = new Float32Array(tail.length);
              hold.set(tail);
              liveMicPcmRemainderRef.current = hold;
            }
          }
        };

        const workletUrl = `${import.meta.env.BASE_URL.replace(/\/?$/, '/')}live-voice-pcm-processor.js`;
        try {
          await audioContext.audioWorklet.addModule(workletUrl);
          const workletNode = new AudioWorkletNode(audioContext, 'live-voice-pcm-processor');
          processorRef.current = workletNode;
          workletNode.port.onmessage = (ev: MessageEvent<{ samples?: Float32Array }>) => {
            const s = ev.data?.samples;
            if (!(s instanceof Float32Array)) return;
            appendLiveMicSamplesForWebSocket(s);
          };
          source.connect(workletNode);
          workletNode.connect(muteOut);
        } catch (workletErr) {
          console.warn('[Live voice] AudioWorklet 不可用，回退 ScriptProcessor', workletErr);
          const processor = audioContext.createScriptProcessor(4096, 1, 1);
          processorRef.current = processor;
          processor.onaudioprocess = (ev: AudioProcessingEvent) => {
            appendLiveMicSamplesForWebSocket(ev.inputBuffer.getChannelData(0));
          };
          source.connect(processor);
          processor.connect(muteOut);
        }
        muteOut.connect(audioContext.destination);

        // Gemini Multimodal Live（固定 `LIVE_VOICE_MODEL_ID`，不再多模型回退）
        let session: Awaited<ReturnType<GoogleGenAI['live']['connect']>> | null = null;
        let lastLiveConnectError: unknown = null;
        const liveModel = LIVE_VOICE_MODEL_ID;
        try {
            const connectGen = ++liveVoiceSessionGenRef.current;
            const sessionPromise = getGeminiClient().live.connect({
              model: liveModel,
              callbacks: {
                onopen: async () => {
                  if (connectGen !== liveVoiceSessionGenRef.current) return;
                  try {
                    await audioContext.resume();
                  } catch {
                    /* ignore */
                  }
                  voiceLiveReadyRef.current = true;
                  liveWsAbnormalRetryRef.current = 0;
                  window.clearTimeout(timeoutId);
                  setIsVoiceConnecting(false);
                  setIsVoiceMode(true);
                  setLiveVoiceHandoff(false);
                  if (liveVoiceSwitchHintTimerRef.current != null) {
                    window.clearTimeout(liveVoiceSwitchHintTimerRef.current);
                    liveVoiceSwitchHintTimerRef.current = null;
                  }
                  setLiveVoiceSwitchHint('');
                  console.info('[Live voice] session opened');
                },
                onmessage: async (message: LiveServerMessage) => {
                  if (connectGen !== liveVoiceSessionGenRef.current) return;

                  if (message.serverContent?.interrupted) {
                    audioSourcesRef.current.forEach((s) => s.stop());
                    audioSourcesRef.current = [];
                    nextPlayTimeRef.current = audioContextRef.current?.currentTime || 0;
                    setIsSpeaking(false);
                    const u = liveStreamUserRef.current.trim();
                    const a = liveStreamAiRef.current.trim();
                    if (u || a) {
                      const intrId = `${Date.now()}-intr-${Math.random().toString(36).slice(2, 7)}`;
                      const suffix = language === 'zh' ? '（已打断）' : ' (interrupted)';
                      const aiText = a ? `${a}${suffix}` : '';
                      persistLiveExchangeToSession(u, aiText);
                      setLiveCaptionTurns((rows) => [
                        ...rows,
                        { id: intrId, user: u, userAlt: '', ai: aiText, aiAlt: '' },
                      ]);
                      void fillBilingualForTurn(intrId, u, aiText);
                    }
                    liveStreamUserRef.current = '';
                    liveStreamAiRef.current = '';
                    setLiveStreamUser('');
                    setLiveStreamAi('');
                    setLiveStreamUserAlt('');
                    setLiveStreamAiZh('');
                  }

                  const { ai: capAi, user: capUser } = pickLiveCaptionTexts(message);
                  if (capUser) {
                    const prevU = liveStreamUserRef.current;
                    const aiAcc = liveStreamAiRef.current.trim();
                    const newU = capUser.trim();
                    const likelyNewUserTurn =
                      prevU.length > 8 &&
                      newU.length > 0 &&
                      newU.length < prevU.length * 0.55 &&
                      !prevU.startsWith(newU) &&
                      !newU.startsWith(prevU.slice(0, Math.min(14, prevU.length))) &&
                      aiAcc.length > 10;
                    if (likelyNewUserTurn) {
                      flushLiveCaptionTurn();
                    }
                    const merged = mergeStreamingTranscript(liveStreamUserRef.current, capUser);
                    liveStreamUserRef.current = merged;
                    setLiveStreamUser(merged);
                  }
                  if (capAi) {
                    const merged = mergeStreamingTranscript(liveStreamAiRef.current, capAi);
                    liveStreamAiRef.current = merged;
                    setLiveStreamAi(merged);
                  }

                  if (isLiveModelTurnComplete(message)) {
                    flushLiveCaptionTurn();
                  }

                  const base64Audio = pickLiveAudioBase64FromMessage(message);
                  if (base64Audio) {
                    handleAudioOutput(base64Audio);
                  }
                },
                onclose: (e: unknown) => {
                  if (connectGen !== liveVoiceSessionGenRef.current) return;
                  const ev = e as CloseEvent;
                  const code = ev?.code;
                  console.log('[Live voice] closed', code, ev?.reason || '(no reason)', ev);
                  const abnormalWs = code === 1006 || code === 1011;
                  if (ev?.code === 1008 && typeof ev?.reason === 'string' && ev.reason.length > 0) {
                    setVoiceBlockingMessage(
                      language === 'zh' ? '语音服务暂时不可用，请稍后再试。' : 'Voice service is temporarily unavailable.',
                    );
                  }
                  stopVoiceMode();

                  if (
                    abnormalWs &&
                    !liveVoiceUserStoppedRef.current &&
                    isOpenRef.current &&
                    conversationModeRef.current === 'live'
                  ) {
                    liveWsAbnormalRetryRef.current += 1;
                    const n = liveWsAbnormalRetryRef.current;
                    if (n <= LIVE_WS_ABNORMAL_MAX_AUTO_RETRIES) {
                      const delayMs = 750 * n;
                      console.info(
                        `[Live voice] WebSocket ${code}（异常断开），${delayMs}ms 后自动重连 (${n}/${LIVE_WS_ABNORMAL_MAX_AUTO_RETRIES})…`,
                      );
                      liveVoiceReconnectTimerRef.current = window.setTimeout(() => {
                        liveVoiceReconnectTimerRef.current = null;
                        if (
                          liveVoiceUserStoppedRef.current ||
                          !isOpenRef.current ||
                          conversationModeRef.current !== 'live'
                        ) {
                          return;
                        }
                        startVoiceModeRef.current?.();
                      }, delayMs);
                    } else {
                      liveWsAbnormalRetryRef.current = 0;
                      console.warn(
                        '[Live voice] 已自动重连多次仍异常断开，请检查网络 / VPN / 代理，或稍后手动再开麦克风。',
                      );
                      setVoiceBlockingMessage(
                        language === 'zh'
                          ? '语音连接反复中断，请更换网络或稍后再试。'
                          : 'Voice keeps dropping. Try another network or again later.',
                      );
                    }
                  }
                },
                onerror: (e: unknown) => {
                  if (connectGen !== liveVoiceSessionGenRef.current) return;
                  console.error('[Live voice] error', e);
                  setVoiceBlockingMessage(
                    language === 'zh' ? '语音连接出错，请检查网络后重试。' : 'Voice connection error. Check your network and retry.',
                  );
                  stopVoiceMode();
                },
              },
              config: {
                responseModalities: [Modality.AUDIO],
                temperature: 0.78,
                speechConfig: {
                  voiceConfig: {
                    prebuiltVoiceConfig: {
                      voiceName: resolveGeminiLiveSpeechVoiceName(),
                    },
                  },
                },
                /** 语音转写：依赖服务端字段；若模型不支持则对应区域可能为空 */
                outputAudioTranscription: {},
                inputAudioTranscription: {},
                systemInstruction: getLiveVoiceSystemInstruction(),
              },
            });
            session = await sessionPromise;
            liveSessionRef.current = session;
            console.log('[Live voice] session ready:', liveModel);

            const bootImg = parseDataUrlForGeminiInline(currentImageDataUrlRef.current?.trim() ?? '');
            if (bootImg && liveSessionRef.current === session) {
              try {
                session.sendClientContent({
                  turns: [
                    {
                      role: 'user',
                      parts: [
                        { inlineData: { mimeType: bootImg.mimeType, data: bootImg.data } },
                        {
                          text:
                            language === 'zh'
                              ? '（上方为对方当前画布上的记忆照片，仅供陪伴语境；请从情绪与氛围回应，勿逐条罗列物体。）'
                              : '(Above: their memory photo on the canvas—for companionship context; do not inventory objects.)',
                        },
                      ],
                    },
                  ],
                  turnComplete: false,
                });
              } catch (imgErr) {
                console.warn('[Live voice] attach startup photo failed', imgErr);
              }
            }
        } catch (e) {
            lastLiveConnectError = e;
            console.warn('[Live voice] connect failed for', liveModel, e);
        }

        if (!session) {
          throw lastLiveConnectError || new Error('LIVE_MODEL_CONNECT_FAILED');
        }

     } catch(e) {
         window.clearTimeout(timeoutId);
         console.error('Failed to start voice mode', e);
         stopVoiceMode();
         setVoiceBlockingMessage(
           language === 'zh'
             ? '无法启动语音，请检查网络与安全访问方式（HTTPS）后重试。'
             : 'Could not start voice. Check your network and HTTPS, then try again.',
         );
     }
  };

  startVoiceModeRef.current = () => {
    void startVoiceMode();
  };

  const toggleVoiceMode = () => {
     if (!allowLiveVoice) return;
     if (isVoiceMode || isVoiceConnecting || liveVoiceHandoff) {
         liveVoiceUserStoppedRef.current = true;
         liveWsAbnormalRetryRef.current = 0;
         stopVoiceMode();
     } else {
         liveVoiceUserStoppedRef.current = false;
         liveWsAbnormalRetryRef.current = 0;
         startVoiceMode();
     }
  };

  /** 打开回忆长廊时结束 Live 语音，避免全屏层叠与 WebAudio / WS 竞态导致无声或白屏 */
  useEffect(() => {
    if (!showHistoryModal) return;
    if (conversationMode !== 'live') return;
    if (isVoiceMode || isVoiceConnecting || liveVoiceHandoff) {
      stopVoiceModeRef.current();
    }
  }, [showHistoryModal, conversationMode, isVoiceMode, isVoiceConnecting, liveVoiceHandoff]);

  const stopCloneDictation = useCallback(
    (opts?: { cleanup?: boolean }) => {
      if (opts?.cleanup) {
        cloneHoldCleanupStopRef.current = true;
        cloneDictationEpochRef.current += 1;
      }
      const mr = cloneMediaRecorderRef.current;
      cloneMediaRecorderRef.current = null;
      cloneRecordChunksRef.current = [];
      if (mr && mr.state !== 'inactive') {
        try {
          mr.stop();
        } catch {
          setIsCloneDictating(false);
          cloneHoldCleanupStopRef.current = false;
          stopCloneUserMicCapture();
        }
      } else {
        setIsCloneDictating(false);
        stopCloneUserMicCapture();
      }
    },
    [stopCloneUserMicCapture],
  );

  /** 复刻：点一下开始听写，再点一下结束并发送 */
  const onCloneMicTap = () => {
    if (conversationMode !== 'text_clone' || isTyping) return;

    if (isCloneDictating) {
      const mr = cloneMediaRecorderRef.current;
      if (!mr || mr.state === 'inactive') {
        stopCloneDictation();
        return;
      }
      setCloneLiveCaptionUserLine(language === 'zh' ? '正在识别…' : 'Transcribing…');
      try {
        mr.stop();
      } catch {
        stopCloneDictation();
      }
      return;
    }

    clearCloneUserDictationHold();
    cloneHoldSnapshotRef.current = inputTextRef.current;
    cloneHoldCleanupStopRef.current = false;

    try {
      cloneBottomTextareaRef.current?.blur();
    } catch {
      /* ignore */
    }

    beginCloneHoldSpeechRecognition();
  };

  useEffect(() => {
    if (conversationMode !== 'text_clone') {
      stopCloneDictation({ cleanup: true });
      clearCloneUserDictationHold();
    }
  }, [conversationMode, stopCloneDictation, clearCloneUserDictationHold]);

  /** Live：根据麦克风 RMS 刷新「在说话」状态；波形绘制在另一 rAF */
  useEffect(() => {
    const voiceUiActive = isVoiceMode || isVoiceConnecting;
    if (!voiceUiActive) {
      liveInputRmsRef.current = 0;
      liveLastVoiceTsRef.current = 0;
      setLiveUserMicHot(false);
      return;
    }
    const MIC_TH = LIVE_MIC_RMS_THRESHOLD;
    const TAIL_MS = LIVE_VOICE_TAIL_MS;
    const id = window.setInterval(() => {
      const rms = liveInputRmsRef.current;
      const now = Date.now();
      if (rms > MIC_TH) liveLastVoiceTsRef.current = now;
      const micHot = rms > MIC_TH || now - liveLastVoiceTsRef.current < TAIL_MS;
      setLiveUserMicHot(micHot);
    }, 48);
    return () => window.clearInterval(id);
  }, [isVoiceMode, isVoiceConnecting]);

  /** Live「我说」面板 / 复刻点按麦克风：声纹与字幕同源节奏（复刻用独立麦克风流 + Analyser） */
  useEffect(() => {
    const liveUserPanelVisible =
      allowLiveVoice &&
      (isVoiceMode || isVoiceConnecting || liveVoiceHandoff) &&
      !isSpeaking &&
      (liveStreamUser.trim().length > 0 || liveUserMicHot);

    const cloneUserPanelVisible =
      conversationMode === 'text_clone' && isCloneDictating && !isSpeaking;

    const userPanelVisible = liveUserPanelVisible || cloneUserPanelVisible;

    if (!userPanelVisible) {
      if (voiceprintRafRef.current != null) {
        cancelAnimationFrame(voiceprintRafRef.current);
        voiceprintRafRef.current = null;
      }
      return;
    }

    let cancelled = false;
    const W = VOICEPRINT_CANVAS_W;
    const H = VOICEPRINT_CANVAS_H;
    /** 复刻听写时降频画声纹（仍用同一套 RMS）；减轻与录音/转写争用主线程 */
    const CLONE_VOICERPRINT_MIN_INTERVAL_MS = 52;
    let cloneVoiceprintLastDrawMs = 0;

    const boot = () => {
      if (cancelled) return;
      const canvas = voiceprintCanvasRef.current;
      if (!canvas) {
        voiceprintRafRef.current = requestAnimationFrame(boot);
        return;
      }
      const useCloneMic = conversationModeRef.current === 'text_clone' && isCloneDictating;
      const analyser = useCloneMic ? cloneUserMicAnalyserRef.current : liveVoiceAnalyserRef.current;
      if (!analyser) {
        voiceprintRafRef.current = requestAnimationFrame(boot);
        return;
      }

      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        /** 移动端 WebKit 偶发首帧拿不到 2d（内存或未就绪），勿静默 return 否则永不进入 loop */
        voiceprintRafRef.current = requestAnimationFrame(boot);
        return;
      }
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, W, H);

      const binCount = analyser.frequencyBinCount;
      const fftN = analyser.fftSize;
      if (!binCount || !fftN) {
        voiceprintRafRef.current = requestAnimationFrame(boot);
        return;
      }
      const freqBuf = new Uint8Array(binCount);
      const tdBuf = new Float32Array(fftN);

      const loop = () => {
        if (cancelled) return;
        let scheduledReboot = false;
        try {
          const cv = voiceprintCanvasRef.current;
          if (!cv || cv.width !== W || cv.height !== H) {
            voiceprintRafRef.current = requestAnimationFrame(loop);
            return;
          }
          const c = cv.getContext('2d');
          if (!c) {
            voiceprintRafRef.current = requestAnimationFrame(loop);
            return;
          }
          const cloneMic =
            conversationModeRef.current === 'text_clone' && isCloneDictating;
          const an = cloneMic ? cloneUserMicAnalyserRef.current : liveVoiceAnalyserRef.current;
          if (!an) {
            voiceprintRafRef.current = requestAnimationFrame(loop);
            return;
          }
          if (an.frequencyBinCount !== freqBuf.length || an.fftSize !== tdBuf.length) {
            voiceprintRafRef.current = requestAnimationFrame(boot);
            scheduledReboot = true;
            return;
          }
          if (cloneMic) {
            const tNow = performance.now();
            if (tNow - cloneVoiceprintLastDrawMs < CLONE_VOICERPRINT_MIN_INTERVAL_MS) {
              voiceprintRafRef.current = requestAnimationFrame(loop);
              return;
            }
            cloneVoiceprintLastDrawMs = tNow;
          }
          an.getByteFrequencyData(freqBuf);
          let rawE: number;
          if (cloneMic) {
            an.getFloatTimeDomainData(tdBuf);
            let sumSq = 0;
            for (let i = 0; i < tdBuf.length; i++) sumSq += tdBuf[i] * tdBuf[i];
            const rms = Math.sqrt(sumSq / Math.max(1, tdBuf.length));
            rawE = Math.min(1, rms * 34);
          } else {
            rawE = Math.min(1, liveInputRmsRef.current * 34);
          }
          voiceprintEnergySmoothRef.current += (rawE - voiceprintEnergySmoothRef.current) * 0.26;
          drawRibbonVoiceprint(c, W, H, freqBuf, voiceprintEnergySmoothRef.current, performance.now());
        } catch (e) {
          console.warn('[Voiceprint user]', e);
        }
        if (!scheduledReboot) voiceprintRafRef.current = requestAnimationFrame(loop);
      };

      voiceprintRafRef.current = requestAnimationFrame(loop);
    };

    voiceprintRafRef.current = requestAnimationFrame(boot);

    return () => {
      cancelled = true;
      if (voiceprintRafRef.current != null) {
        cancelAnimationFrame(voiceprintRafRef.current);
        voiceprintRafRef.current = null;
      }
    };
  }, [
    allowLiveVoice,
    isVoiceMode,
    isVoiceConnecting,
    liveVoiceHandoff,
    isSpeaking,
    liveStreamUser,
    liveUserMicHot,
    conversationMode,
    isCloneDictating,
    cloneUserMicPrimed,
  ]);

  /** AI 字幕框声纹：Live（Analyser）；复刻 MiniMax（Analyser）；系统 TTS 无 Analyser 时用合成频谱 */
  useEffect(() => {
    /** 勿用 liveStreamAi：字幕先出时声纹会跟着动，与扬声器不同步；仅在实际播音 isSpeaking 时驱动 */
    const liveAiPanelVoiceprintVisible =
      allowLiveVoice &&
      (isVoiceMode || isVoiceConnecting || liveVoiceHandoff) &&
      isSpeaking;

    /** 复刻模式不再绘制 AI 声纹 canvas（简化链路，避免 AudioContext/Analyser 复杂度） */
    const cloneAiPanelVoiceprintVisible = false;

    const aiPanelVoiceprintVisible =
      liveAiPanelVoiceprintVisible || cloneAiPanelVoiceprintVisible;

    if (!aiPanelVoiceprintVisible) {
      if (voiceprintAiRafRef.current != null) {
        cancelAnimationFrame(voiceprintAiRafRef.current);
        voiceprintAiRafRef.current = null;
      }
      return;
    }

    let cancelled = false;
    const W = VOICEPRINT_CANVAS_W;
    const H = VOICEPRINT_CANVAS_H;

    const boot = () => {
      if (cancelled) return;
      const canvas = voiceprintAiCanvasRef.current;
      if (!canvas) {
        voiceprintAiRafRef.current = requestAnimationFrame(boot);
        return;
      }
      const analyser = liveAiVoiceAnalyserRef.current;
      if (liveAiPanelVoiceprintVisible && !cloneAiPanelVoiceprintVisible && !analyser) {
        voiceprintAiRafRef.current = requestAnimationFrame(boot);
        return;
      }

      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        voiceprintAiRafRef.current = requestAnimationFrame(boot);
        return;
      }
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, W, H);

      const loop = () => {
        if (cancelled) return;
        try {
          const cv = voiceprintAiCanvasRef.current;
          if (!cv || cv.width !== W || cv.height !== H) {
            voiceprintAiRafRef.current = requestAnimationFrame(loop);
            return;
          }
          const c = cv.getContext('2d');
          if (!c) {
            voiceprintAiRafRef.current = requestAnimationFrame(loop);
            return;
          }

          const an = liveAiVoiceAnalyserRef.current;
          const t = performance.now();
          let freqBuf: Uint8Array;
          let rawE: number;

          if (an) {
            freqBuf = new Uint8Array(an.frequencyBinCount);
            const tdBuf = new Float32Array(an.fftSize);
            an.getByteFrequencyData(freqBuf);
            an.getFloatTimeDomainData(tdBuf);
            let sumSq = 0;
            for (let i = 0; i < tdBuf.length; i++) sumSq += tdBuf[i] * tdBuf[i];
            const rms = Math.sqrt(sumSq / Math.max(1, tdBuf.length));
            rawE = Math.min(1, rms * 36);
          } else if (
            conversationMode === 'text_clone' &&
            (isSpeaking || cloneAwaitingTtsStart)
          ) {
            freqBuf = new Uint8Array(128);
            for (let i = 0; i < freqBuf.length; i++) {
              freqBuf[i] = Math.min(
                255,
                (Math.sin(t / 200 + i * 0.08) * 0.5 + 0.5) * 200 + Math.random() * 55,
              );
            }
            rawE = Math.min(1, 0.42 + Math.sin(t / 160) * 0.12);
          } else {
            voiceprintAiRafRef.current = requestAnimationFrame(loop);
            return;
          }

          voiceprintAiEnergySmoothRef.current += (rawE - voiceprintAiEnergySmoothRef.current) * 0.26;
          drawRibbonVoiceprint(c, W, H, freqBuf, voiceprintAiEnergySmoothRef.current, t, 'rose');
          voiceprintAiRafRef.current = requestAnimationFrame(loop);
        } catch (e) {
          console.warn('[Voiceprint AI]', e);
          voiceprintAiRafRef.current = requestAnimationFrame(loop);
        }
      };

      voiceprintAiRafRef.current = requestAnimationFrame(loop);
    };

    voiceprintAiRafRef.current = requestAnimationFrame(boot);

    return () => {
      cancelled = true;
      if (voiceprintAiRafRef.current != null) {
        cancelAnimationFrame(voiceprintAiRafRef.current);
        voiceprintAiRafRef.current = null;
      }
    };
  }, [
    allowLiveVoice,
    isVoiceMode,
    isVoiceConnecting,
    liveVoiceHandoff,
    liveStreamAi,
    isSpeaking,
    conversationMode,
    cloneAwaitingTtsStart,
  ]);

  /** Live 只用面板字幕；切换模式时清计时器 */
  useEffect(() => {
    if (conversationMode !== 'live') return;
    clearFloatingDisplayTimers();
  }, [conversationMode, clearFloatingDisplayTimers]);

  /** 关闭对话层时退出语音；不在组件 unmount 里 stopVoiceMode，避免 React Strict Mode 假卸载打断 Live 连接 */
  useEffect(() => {
    if (!isOpen && (isVoiceMode || isVoiceConnecting || liveVoiceHandoff)) {
      stopVoiceMode();
    }
  }, [isOpen, isVoiceMode, isVoiceConnecting, liveVoiceHandoff]);

  // Cleanup：仅清计时器与复刻 MediaRecorder（语音会话由 isOpen / 用户按键管理）
  useEffect(() => {
    return () => {
      clearFloatingDisplayTimers();
      if (cloneUserDictationHoldTimerRef.current != null) {
        window.clearTimeout(cloneUserDictationHoldTimerRef.current);
        cloneUserDictationHoldTimerRef.current = null;
      }
      cloneDictationEpochRef.current += 1;
      cloneHoldCleanupStopRef.current = true;
      const mr = cloneMediaRecorderRef.current;
      cloneMediaRecorderRef.current = null;
      cloneRecordChunksRef.current = [];
      if (mr && mr.state !== 'inactive') {
        try {
          mr.stop();
        } catch {
          /* ignore */
        }
      }
      stopCloneUserMicCapture();
    };
  }, [clearFloatingDisplayTimers, stopCloneUserMicCapture]);

  /** Live：语音会话中隐藏左侧气泡区；复刻：始终不展示气泡历史（与 Live 同源字幕面板） */
  const hideTextBubbleChatUi =
    conversationMode === 'text_clone' ||
    (allowLiveVoice && (isVoiceMode || isVoiceConnecting || liveVoiceHandoff));

  const aiDisplayName =
    (typeof aiName === 'string' ? aiName : String(aiName ?? '')).trim() ||
    (language === 'zh' ? '潜意识回响' : 'ECHO OF MIND');

  /** 仅流式/即时态：不用历史一轮回填，避免字幕框常驻 */
  const liveUserCaptionMain = liveStreamUser.trim();
  const liveUserCaptionAlt = liveStreamUser.trim() ? liveStreamUserAlt : '';
  const liveAiCaptionMain = liveStreamAi.trim();
  const liveAiCaptionAlt = liveStreamAi.trim() ? liveStreamAiZh : '';

  const liveVoiceChromeOpen = allowLiveVoice && (isVoiceMode || isVoiceConnecting || liveVoiceHandoff);
  const showLiveUserCaptionPanel =
    liveVoiceChromeOpen && !isSpeaking && (liveUserCaptionMain.length > 0 || liveUserMicHot);
  const showLiveAiCaptionPanel = liveVoiceChromeOpen && (liveAiCaptionMain.length > 0 || isSpeaking);
  const showLiveCaptionChrome = showLiveUserCaptionPanel || showLiveAiCaptionPanel;

  const overlayToolbarIconClass =
    'flex h-12 w-12 shrink-0 items-center justify-center rounded-full border border-zinc-800 bg-zinc-900/60 text-zinc-400 shadow-md backdrop-blur-md transition-colors hover:bg-zinc-800/80 hover:text-zinc-200 touch-manipulation active:scale-[0.96]';

  const showOverlayToolbar = onDissolveReset != null && onOpenSavePreview != null;

  const overlayResetBtn = showOverlayToolbar ? (
    <button
      type="button"
      onClick={onDissolveReset}
      title={language === 'zh' ? '消散重置画面' : 'Dissolve & reset'}
      aria-label={language === 'zh' ? '消散重置' : 'Dissolve reset'}
      className={overlayToolbarIconClass}
    >
      <RefreshCw className="h-[18px] w-[18px]" strokeWidth={1.5} />
    </button>
  ) : null;

  const overlaySaveBtn = showOverlayToolbar ? (
    <button
      type="button"
      onClick={onOpenSavePreview}
      title={language === 'zh' ? '保存对话到回忆册' : 'Save chat to album'}
      aria-label={language === 'zh' ? '保存对话' : 'Save chat'}
      className={overlayToolbarIconClass}
    >
      <Save className="h-[18px] w-[18px]" strokeWidth={1.5} />
    </button>
  ) : null;

  const liveVoiceChromeActive = isVoiceMode || isVoiceConnecting || liveVoiceHandoff;
  const liveMicBarHint = (
    <p className="pointer-events-none text-center text-[10px] font-medium uppercase tracking-[0.18em] text-zinc-600">
      {language === 'zh' ? 'LIVE 模式 · 点按麦克风' : 'Live mode · tap mic'}
    </p>
  );
  const liveMicTapButton = (
    <button
      type="button"
      onClick={toggleVoiceMode}
      className={`flex h-16 min-h-[64px] w-16 min-w-[64px] shrink-0 items-center justify-center rounded-full border border-zinc-800 bg-zinc-900/60 text-zinc-500 shadow-lg backdrop-blur-md transition-colors hover:text-zinc-200 touch-manipulation active:bg-zinc-800/50 md:h-[4.25rem] md:min-h-[68px] md:w-[4.25rem] md:min-w-[68px] ${liveVoiceChromeActive ? 'border-rose-900/50 text-rose-500' : ''}`}
      aria-label={language === 'zh' ? '语音对话' : 'Voice'}
    >
      {isVoiceConnecting || liveVoiceHandoff ? (
        <Loader2 strokeWidth={1.5} className="h-7 w-7 animate-spin md:h-8 md:w-8" />
      ) : (
        <Mic strokeWidth={1.5} className="h-7 w-7 md:h-8 md:w-8" />
      )}
    </button>
  );

  if (!isOpen) return null;

  return (
    <>
    {currentImageDataUrl && albumSessions.length > 0 && (
      <div
        className="pointer-events-auto fixed left-[max(0.45rem,env(safe-area-inset-left))] top-[max(20dvh,calc(env(safe-area-inset-top)+5.5rem))] z-[38] flex max-h-[min(40dvh,18rem)] w-9 flex-col items-center gap-2 overflow-y-auto overscroll-contain py-1 [-ms-overflow-style:none] [scrollbar-width:none] md:left-[max(1rem,env(safe-area-inset-left))] md:gap-2.5 [&::-webkit-scrollbar]:hidden"
        role="list"
        aria-label={language === 'zh' ? '回忆册：已镌刻的对话' : 'Memory album: saved conversations'}
      >
        {albumSessions.slice(0, 28).map((s, idx) => (
          <button
            key={s.id}
            type="button"
            title={safeSessionTheme(s.theme)}
            onClick={() => {
              setActiveSessionId(s.id);
              setShowSavePreview(false);
            }}
            className={`relative z-0 h-2 w-2 shrink-0 rounded-full border transition-transform duration-300 hover:z-[1] hover:scale-[1.45] active:scale-95 md:h-2.5 md:w-2.5 ${
              activeSessionId === s.id
                ? 'border-rose-400/80 bg-rose-200/40 shadow-[0_0_14px_rgba(251,113,133,0.5)]'
                : 'border-zinc-500/45 bg-zinc-200/20 shadow-[0_0_10px_rgba(228,228,231,0.22)] hover:border-zinc-400/70 hover:bg-zinc-100/35'
            } ${idx % 5 === 0 ? 'motion-safe:animate-pulse' : ''}`}
            aria-label={safeSessionTheme(s.theme)}
          />
        ))}
      </div>
    )}
    <div
      className={`pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-end pb-[calc(env(safe-area-inset-bottom)+1rem)] transition-opacity duration-700 max-md:pb-[calc(env(safe-area-inset-bottom)+1.1rem)] md:pb-[calc(env(safe-area-inset-bottom)+1.25rem)] ${isOpen ? 'opacity-100' : 'opacity-0'}`}
    >
      {/* AI 状态：尽量不占竖屏空间 */}
      <div className="pointer-events-none absolute left-1/2 top-[max(8dvh,calc(env(safe-area-inset-top)+3.25rem))] z-10 flex -translate-x-1/2 items-center gap-2 md:top-[12%] md:gap-5">
        <div className={`relative flex h-9 w-9 items-center justify-center transition-all duration-700 md:h-9 md:w-9 ${isAIActive ? 'scale-110' : 'opacity-60'}`}>
          {/* Subtle backdrop shadow for visibility */}
          <div className="absolute inset-0 bg-black/40 blur-xl rounded-full -z-10" />
          
          {/* 外层旋转轨道 */}
          <div className={`absolute inset-0 rounded-full border-2 border-dashed transition-colors duration-700 ${
            isAIActive ? 'border-zinc-400/40 animate-[spin_5s_linear_infinite]' : 'border-zinc-800'
          }`} />
          
          {/* 中层静止环 */}
          <div className={`absolute inset-1.5 rounded-full border border-zinc-800/50`} />

          {/* 中心核心 */}
          <div className={`w-2.5 h-2.5 rounded-full transition-all duration-700 ${
            isAIActive 
              ? 'bg-zinc-200 shadow-[0_0_15px_rgba(255,255,255,0.6)]' 
              : 'bg-zinc-700'
          }`} />
          
          {/* 动态光圈扩散 / 说话脉冲：Live 改在字幕区上方展示，避免 AI 出声前顶部先闪 */}
          {!allowLiveVoice && (isTyping || isVoiceConnecting) && (
            <div className="absolute inset-0 rounded-full border-2 border-zinc-400/30 animate-ping" />
          )}
          {!allowLiveVoice && isSpeaking && (
            <div className="absolute -inset-3 rounded-full border border-zinc-400/20 animate-pulse" />
          )}
        </div>
        
        <span
          className={`max-w-[min(14rem,62vw)] truncate text-[11px] font-medium uppercase tracking-[0.2em] drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)] transition-all duration-700 whitespace-nowrap md:max-w-none md:text-base md:tracking-[0.3em] ${
            isAIActive ? 'text-zinc-200 opacity-100' : 'text-zinc-500 opacity-80'
          }`}
        >
          {aiDisplayName}
        </span>
      </div>

      {/* 仅输入区与气泡列表接收触摸；有字幕时消散/保存贴在字幕区左右；复刻输入条见下方 fixed 贴底 */}
      <div
        className={`pointer-events-none flex w-full flex-col gap-2 px-3 sm:px-4 ${
          (conversationMode === 'text_clone' && !allowLiveVoice) || (allowLiveVoice && showOverlayToolbar)
            ? 'pb-[calc(3.75rem+env(safe-area-inset-bottom))]'
            : ''
        }`}
      >
        <div className="mx-auto flex w-full max-w-[min(22rem,calc(100vw-1.5rem))] flex-col gap-2 md:gap-2.5">
          <div className="flex w-full flex-col gap-2">

        {!hideTextBubbleChatUi && (
        <>
        {/* Body */}
        <div
          className="no-scrollbar pointer-events-auto relative max-h-[16dvh] touch-pan-y overflow-y-auto p-1 sm:max-h-[36vh] md:max-h-[44vh]"
          style={{
            maskImage: 'linear-gradient(to bottom, transparent, black 8%, black 92%, transparent)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 8%, black 92%, transparent)',
          }}
        >
             <div className="flex flex-col gap-4 py-8 md:gap-6 md:py-12">
                {pendingUserBubble != null &&
                  !(
                    conversationMode === 'text_clone' &&
                    (cloneUserPostDictationHold || cloneVoiceRoundSuppressPendingBubbleRef.current)
                  ) && (
                  <div className="flex flex-col items-end animate-in fade-in duration-300">
                    <div className="max-w-[90%] rounded-2xl rounded-br-md border border-zinc-700/50 bg-zinc-800/60 px-3 py-2.5 text-[13px] leading-snug tracking-wide text-zinc-200 backdrop-blur-md md:max-w-[85%] md:rounded-3xl md:p-4 md:text-sm md:leading-relaxed">
                      {pendingUserBubble.imageUrl ? (
                        <img
                          src={pendingUserBubble.imageUrl}
                          alt=""
                          className="mb-2 max-h-[min(40dvh,200px)] w-full rounded-xl object-contain object-right"
                        />
                      ) : null}
                      {pendingUserBubble.text}
                    </div>
                    {isTyping && conversationMode !== 'text_clone' && (
                      <div className="mt-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 md:mt-2 md:gap-2 md:text-[10px]">
                        <Loader2 className="h-3 w-3 shrink-0 animate-spin md:h-3 md:w-3" strokeWidth={2} />
                        <span>{language === 'zh' ? '等待回响…' : 'Waiting…'}</span>
                      </div>
                    )}
                  </div>
                )}
                <div ref={endOfMessagesRef} />
             </div>
        </div>
        </>
        )}

        {/* Input Area */}
        <div className="pointer-events-auto transition-all duration-500">
           {allowLiveVoice && (isVoiceMode || isVoiceConnecting || liveVoiceHandoff) && (
             <div className="mb-1.5 flex w-full flex-col gap-1.5">
             <div className="flex min-h-[38px] items-center justify-between gap-2 rounded-full border border-zinc-800 bg-zinc-900/60 px-4 py-2 backdrop-blur-md md:min-h-[48px] md:px-6 md:py-3">
                <div className="flex min-w-0 flex-1 flex-col gap-1">
                <div className="flex min-w-0 flex-wrap items-center gap-2 md:gap-3">
                   <div
                     className={`h-2 w-2 shrink-0 rounded-full md:h-2 ${
                       isSpeaking ? 'animate-pulse bg-zinc-200' : liveVoiceHandoff && !isVoiceMode && !isVoiceConnecting ? 'animate-pulse bg-amber-400/90' : 'bg-zinc-500'
                     }`}
                   />
                   <span className="truncate font-serif text-[10px] font-medium italic tracking-wider text-zinc-400 md:text-xs md:tracking-widest">
                     {isVoiceConnecting
                       ? language === 'zh'
                         ? '连接中...'
                         : 'Connecting...'
                       : liveVoiceHandoff && !isVoiceMode
                         ? language === 'zh'
                           ? '切换音色中…'
                           : 'Switching voice…'
                         : isSpeaking
                           ? language === 'zh'
                             ? '回响中...'
                             : 'Echoing...'
                           : language === 'zh'
                             ? '倾听中...'
                             : 'Listening...'}
                   </span>
                   <label className="flex min-w-0 max-w-full items-center gap-1">
                     <span className="sr-only">{language === 'zh' ? '选择语音音色' : 'Choose voice'}</span>
                     <select
                       value={getLiveVoiceSelectControlValue()}
                       onChange={(e) => {
                         const v = e.target.value;
                         if (v === '__DEFAULT__') {
                           try {
                             sessionStorage.removeItem(LIVE_VOICE_SESSION_STORAGE_KEY);
                           } catch {
                             /* ignore */
                           }
                         } else {
                           try {
                             sessionStorage.setItem(LIVE_VOICE_SESSION_STORAGE_KEY, v);
                           } catch {
                             /* ignore */
                           }
                         }
                         setLiveVoiceRevision((n) => n + 1);
                         if (isVoiceMode || isVoiceConnecting || liveVoiceHandoff) {
                           if (liveVoiceSwitchHintTimerRef.current != null) {
                             window.clearTimeout(liveVoiceSwitchHintTimerRef.current);
                             liveVoiceSwitchHintTimerRef.current = null;
                           }
                           setLiveVoiceSwitchHint('');
                           setLiveVoiceHandoff(true);
                           stopVoiceMode({ preserveLiveVoiceHandoff: true });
                           queueMicrotask(() => {
                             void startVoiceMode();
                           });
                         }
                       }}
                       className="max-w-[min(72vw,14rem)] truncate rounded-md border border-zinc-700 bg-zinc-950/90 px-1.5 py-1 font-serif text-[9px] italic text-zinc-300 md:max-w-[16rem] md:text-[10px]"
                       title={
                         language === 'zh'
                           ? '选择实时语音使用的播报音色。'
                           : 'Choose the voice used for live replies.'
                       }
                     >
                       <option value="__DEFAULT__">
                         {language === 'zh'
                           ? `默认（${getGeminiLiveSpeechVoiceNameFromEnvOnly()}）`
                           : `Default (${getGeminiLiveSpeechVoiceNameFromEnvOnly()})`}
                       </option>
                       {liveVoiceOptionsForSelect().map((name) => (
                         <option key={name} value={name}>
                           {name}
                         </option>
                       ))}
                     </select>
                   </label>
                   <span
                     className="max-w-[min(40vw,7rem)] truncate font-serif text-[9px] italic normal-case tracking-normal text-zinc-600 md:max-w-[10rem] md:text-[10px]"
                     title={language === 'zh' ? '当前播报音色' : 'Current voice'}
                   >
                     {resolveGeminiLiveSpeechVoiceName()}
                   </span>
                </div>
                {liveVoiceSwitchHint ? (
                  <p className="w-full pl-1 font-serif text-[10px] italic leading-snug text-amber-200/90 md:text-[11px]">{liveVoiceSwitchHint}</p>
                ) : null}
                </div>
                <button
                  type="button"
                  onClick={stopVoiceMode}
                  className="flex min-h-[40px] min-w-[40px] shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:text-zinc-300 touch-manipulation active:bg-zinc-800/50 md:min-h-[44px] md:min-w-[44px]"
                  aria-label={language === 'zh' ? '停止语音' : 'Stop voice'}
                >
                  <MicOff className="h-[18px] w-[18px] md:h-[18px] md:w-[18px]" strokeWidth={1.5} />
                </button>
             </div>
             {showLiveCaptionChrome ? (
               <div
                 className="pointer-events-auto flex w-full flex-col gap-2 font-serif italic"
                 aria-live="polite"
                 aria-relevant="additions text"
               >
                 {isSpeaking ? (
                   <div
                     className="relative flex h-5 w-full shrink-0 justify-center motion-reduce:hidden"
                     aria-hidden
                   >
                     <span className="pointer-events-none absolute top-1/2 h-9 w-9 -translate-y-1/2 rounded-full border border-rose-400/25 animate-ping" />
                   </div>
                 ) : null}
                 {showLiveUserCaptionPanel ? (
                   <div className="animate-in fade-in slide-in-from-bottom-1 flex min-h-[5rem] flex-col overflow-hidden rounded-xl border border-zinc-800/90 bg-zinc-950/90 shadow-inner backdrop-blur-md duration-300 md:min-h-[5.75rem]">
                     <div className="shrink-0 border-b border-zinc-800/60 px-2.5 py-1 text-[9px] font-medium uppercase tracking-wider text-zinc-500 not-italic">
                       {language === 'zh' ? '我说' : 'You'}
                     </div>
                     <div className="shrink-0 px-2 pb-1 pt-1" aria-hidden>
                       <canvas
                         ref={voiceprintCanvasRef}
                         className="mx-auto block h-[52px] w-full max-w-full rounded-md bg-black ring-1 ring-cyan-950/40"
                         width={VOICEPRINT_CANVAS_W}
                         height={VOICEPRINT_CANVAS_H}
                       />
                     </div>
                     <div className="flex min-h-0 flex-1 flex-col justify-center overflow-hidden px-2.5 py-2 text-right">
                       {liveUserCaptionMain ? (
                         <p className="line-clamp-6 break-words text-[11px] leading-snug text-zinc-200/95 not-italic md:text-[12px]">
                           {liveUserCaptionMain}
                         </p>
                       ) : liveUserMicHot ? (
                         <p className="text-[10px] text-zinc-600 not-italic md:text-[11px]">
                           {language === 'zh' ? '正在聆听…' : 'Listening…'}
                         </p>
                       ) : null}
                       {liveUserCaptionAlt ? (
                         <p className="mt-1 line-clamp-3 break-words text-[10px] leading-snug text-zinc-500 not-italic md:text-[11px]">
                           {liveUserCaptionAlt}
                         </p>
                       ) : null}
                     </div>
                   </div>
                 ) : null}
                 {showLiveAiCaptionPanel ? (
                   <div
                     className={`animate-in fade-in slide-in-from-bottom-1 flex min-h-[5rem] flex-col overflow-hidden rounded-xl border bg-zinc-950/90 shadow-inner backdrop-blur-md duration-300 md:min-h-[5.75rem] ${
                       isSpeaking ? 'border-rose-500/35 shadow-[0_0_20px_rgba(244,63,94,0.08)]' : 'border-zinc-800/90'
                     }`}
                   >
                     <div className="shrink-0 border-b border-zinc-800/60 px-2.5 py-1 text-[9px] font-medium uppercase tracking-wider text-zinc-500 not-italic">
                       <span className="line-clamp-1 normal-case tracking-normal">{aiDisplayName}</span>
                     </div>
                     {isSpeaking ? (
                       <div className="shrink-0 px-2 pb-1 pt-1" aria-hidden>
                         <canvas
                           ref={voiceprintAiCanvasRef}
                           className="mx-auto block h-[52px] w-full max-w-full rounded-md bg-black ring-1 ring-rose-950/45"
                           width={VOICEPRINT_CANVAS_W}
                           height={VOICEPRINT_CANVAS_H}
                         />
                       </div>
                     ) : null}
                     <div className="flex min-h-0 flex-1 flex-col justify-center overflow-hidden px-2.5 py-2 text-left">
                       {liveAiCaptionMain ? (
                         <p className="line-clamp-6 break-words font-serif text-[12px] italic leading-relaxed tracking-normal text-zinc-50/95 md:text-[13px]">
                           {liveAiCaptionMain}
                         </p>
                       ) : isSpeaking ? (
                         <p className="font-serif text-[12px] italic text-zinc-500 md:text-[13px]">
                           …
                         </p>
                       ) : null}
                       {liveAiCaptionAlt ? (
                         <p className="mt-1 line-clamp-3 break-words font-serif text-[11px] italic leading-relaxed tracking-normal text-zinc-400 md:text-[12px]">
                           {liveAiCaptionAlt}
                         </p>
                       ) : null}
                     </div>
                   </div>
                 ) : null}
               </div>
             ) : null}
             </div>
           )}
          {conversationMode === 'text_clone' && (isCloneDictating || cloneLiveCaptionUserLine || showCloneAiCaptionPanel) ? (
                <div
                  className="pointer-events-auto mb-1.5 w-full font-serif italic"
                  aria-live="polite"
                  aria-relevant="additions text"
                >
                  {isCloneDictating || cloneLiveCaptionUserLine ? (
                    <p className="mb-1 text-center text-[10px] tracking-wider text-zinc-500 not-italic animate-pulse">
                      {cloneUserSubtitleDisplay || (language === 'zh' ? '正在录音…' : 'Recording…')}
                    </p>
                  ) : null}
                  {showCloneAiCaptionPanel ? (
                  <div
                    className="animate-in fade-in slide-in-from-bottom-1 flex min-h-[3rem] max-h-[min(52vh,22rem)] flex-col overflow-hidden rounded-xl border border-zinc-800/90 bg-zinc-950/90 shadow-inner backdrop-blur-md duration-300"
                  >
                    <div className="shrink-0 border-b border-zinc-800/60 px-2.5 py-1 text-[9px] font-medium uppercase tracking-wider text-zinc-500 not-italic">
                      <span className="line-clamp-1 normal-case tracking-normal">{aiDisplayName}</span>
                      {isTyping && cloneWaitingForModel ? (
                        <span className="ml-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-zinc-500 align-middle" />
                      ) : isSpeaking || cloneAwaitingTtsStart ? (
                        <span className="ml-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-rose-400/70 align-middle" />
                      ) : null}
                    </div>
                    <div
                      ref={cloneAiCaptionScrollRef}
                      className="no-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-2.5 py-2 text-left [overflow-anchor:none]"
                    >
                      {cloneAiCaptionMain ? (
                        <p className="break-words font-serif text-[12px] italic leading-relaxed tracking-normal text-zinc-50/95 md:text-[13px]">
                          {cloneAiCaptionMain}
                        </p>
                      ) : cloneTtsNeedsUserGesture ? (
                        <button
                          type="button"
                          className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-950/50 px-4 py-1.5 font-serif text-[11px] italic text-rose-200/90 transition-colors active:bg-rose-900/60"
                          onClick={() => {
                            const a = pendingUserGestureAudioRef.current;
                            if (a) {
                              pendingUserGestureAudioRef.current = null;
                              setCloneTtsNeedsUserGesture(false);
                              a.play().catch(() => {});
                            }
                          }}
                        >
                          {language === 'zh' ? '点击收听' : 'Tap to listen'}
                        </button>
                      ) : cloneAwaitingTtsStart ? (
                        <p className="font-serif text-[12px] italic text-zinc-500 md:text-[13px]">
                          {language === 'zh' ? '正在接通回响…' : 'Preparing voice…'}
                        </p>
                      ) : isSpeaking ? (
                        <p className="font-serif text-[12px] italic text-zinc-500 md:text-[13px]">
                          …
                        </p>
                      ) : isTyping && cloneWaitingForModel ? (
                        <p className="font-serif text-[12px] italic text-zinc-500 md:text-[13px]">
                          {language === 'zh' ? '等待回响…' : 'Waiting…'}
                        </p>
                      ) : null}
                    </div>
                  </div>
                  ) : null}
                </div>
          ) : null}
          {allowLiveVoice ? (
            !showOverlayToolbar ? (
              <div className="flex w-full flex-col items-center gap-1.5 py-1.5">
                <div className="relative flex w-full flex-col items-center justify-center gap-1">
                  {liveMicBarHint}
                  {liveMicTapButton}
                </div>
              </div>
            ) : null
          ) : (
            <div className="relative flex w-full flex-col items-center gap-2 md:gap-2.5">
              <button
                type="button"
                onClick={onCloneMicTap}
                disabled={isTyping}
                title={language === 'zh' ? '点按开始录音，再点完成发送' : 'Tap to start, tap again to send'}
                aria-pressed={isCloneDictating}
                aria-label={
                  language === 'zh' ? '点按开始语音输入，再点一次完成并发送' : 'Tap to start voice input, tap again to send'
                }
                className={`select-none flex h-12 min-h-[48px] w-12 min-w-[48px] shrink-0 touch-manipulation items-center justify-center rounded-full border border-zinc-800 bg-zinc-900/60 text-zinc-400 shadow-lg backdrop-blur-md transition-colors hover:bg-zinc-800/80 hover:text-zinc-200 active:bg-zinc-800/80 md:h-14 md:min-h-[56px] md:w-14 md:min-w-[56px] ${
                  isCloneDictating
                    ? 'border-emerald-500/45 text-emerald-400 shadow-[0_0_16px_rgba(52,211,153,0.12)] hover:bg-zinc-900/60'
                    : 'disabled:opacity-30'
                }`}
              >
                <Mic strokeWidth={1.5} className="h-[22px] w-[22px] md:h-6 md:w-6" />
              </button>
            </div>
          )}
        </div>
          </div>
        </div>

      </div>
    </div>

      {isOpen && conversationMode === 'text_clone' && !allowLiveVoice && !settingsChromeOpen
        ? createPortal(
            <div className="pointer-events-none fixed inset-x-0 bottom-0 z-[45] flex justify-center px-3 pb-[max(0.35rem,env(safe-area-inset-bottom))] pt-1 sm:px-4">
              <div className="pointer-events-auto flex w-full max-w-[min(22rem,calc(100vw-1.5rem))] flex-row items-center gap-1.5 md:gap-2">
                {showOverlayToolbar ? overlayResetBtn : null}
                <div className="relative flex min-h-12 min-w-0 flex-1 items-center overflow-hidden rounded-full border border-zinc-800 bg-zinc-900/60 shadow-lg backdrop-blur-md transition-colors focus-within:border-zinc-600 md:min-h-[48px] md:shadow-xl">
                  <textarea
                    ref={cloneBottomTextareaRef}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    readOnly={isCloneDictating}
                    enterKeyHint="send"
                    inputMode="text"
                    autoComplete="off"
                    autoCorrect="on"
                    placeholder={language === 'zh' ? '输入…' : 'Message…'}
                    className="max-h-24 min-h-[42px] min-w-0 flex-1 resize-none bg-transparent py-2 pl-3 pr-1 text-[15px] leading-snug text-zinc-200 placeholder:text-zinc-600 focus:outline-none md:max-h-28 md:min-h-[48px] md:py-3 md:pl-4 md:pr-2 md:text-sm md:leading-normal"
                    rows={1}
                  />
                  <button
                    type="button"
                    onClick={() => void sendMessage()}
                    disabled={isTyping || !inputText.trim()}
                    className="mr-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800/60 hover:text-zinc-200 disabled:opacity-30 touch-manipulation active:bg-zinc-800/80 md:mr-1.5 md:h-10 md:w-10"
                    aria-label={language === 'zh' ? '发送' : 'Send'}
                  >
                    <Send strokeWidth={1.5} className="h-5 w-5 md:h-[22px] md:w-[22px]" />
                  </button>
                </div>
                {showOverlayToolbar ? overlaySaveBtn : null}
              </div>
            </div>,
            getAppPortalNode(),
          )
        : null}

      {isOpen && allowLiveVoice && showOverlayToolbar && !settingsChromeOpen
        ? createPortal(
            <div className="pointer-events-none fixed inset-x-0 bottom-0 z-[45] flex justify-center px-3 pb-[max(0.35rem,env(safe-area-inset-bottom))] pt-1 sm:px-4">
              <div
                className={`pointer-events-auto flex w-full max-w-[min(22rem,calc(100vw-1.5rem))] items-end ${
                  liveVoiceChromeActive
                    ? 'flex-row justify-between gap-3 px-0.5'
                    : 'flex-row items-end justify-center gap-5 md:gap-7'
                }`}
              >
                {overlayResetBtn}
                {!liveVoiceChromeActive ? (
                  <div className="flex flex-col items-center gap-1 pb-0.5">
                    {liveMicBarHint}
                    {liveMicTapButton}
                  </div>
                ) : null}
                {overlaySaveBtn}
              </div>
            </div>,
            getAppPortalNode(),
          )
        : null}

      {showHistoryModal &&
        createPortal(
          <div className="fixed inset-0 z-[9999] flex flex-col bg-[#030303] pointer-events-auto pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_50%_120%,rgba(39,39,42,0.55)_0%,transparent_55%)]" />
            <div className="pointer-events-none absolute inset-0 opacity-[0.04] bg-[url('https://www.transparenttextures.com/patterns/stardust.png')]" />
            <button
              type="button"
              className="absolute inset-0 z-0 cursor-default"
              aria-label={language === 'zh' ? '关闭' : 'Close'}
              onClick={() => setShowHistoryModal(false)}
            />
            <div
              role="dialog"
              aria-modal="true"
              aria-label={language === 'zh' ? '历史对话' : 'Chat history'}
              className="relative z-[1] flex min-h-0 flex-1 flex-col"
              style={{ perspective: '1400px' }}
            >
              <div className="relative flex shrink-0 items-center justify-between px-5 pt-4 pb-2 md:px-8 md:pt-6">
                <span className="text-[11px] font-medium uppercase tracking-[0.32em] text-zinc-500">
                  {language === 'zh' ? '回忆长廊' : 'Memory hall'}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded-full border border-zinc-800/90 bg-zinc-950/80 px-4 py-2 text-[11px] font-medium uppercase tracking-wider text-zinc-300 backdrop-blur-md transition-colors hover:border-zinc-600 hover:text-zinc-100 touch-manipulation"
                    onClick={() => {
                      createNewSession();
                      setShowHistoryModal(false);
                    }}
                  >
                    {language === 'zh' ? '+ 新对话' : '+ New'}
                  </button>
                  <button
                    type="button"
                    className="rounded-full border border-zinc-800/90 px-3 py-2 text-[11px] text-zinc-500 transition-colors hover:bg-zinc-900/80 hover:text-zinc-200 touch-manipulation"
                    onClick={() => setShowHistoryModal(false)}
                  >
                    {language === 'zh' ? '关闭' : 'Close'}
                  </button>
                </div>
              </div>

              <div
                className="relative flex min-h-0 flex-1 flex-col items-center justify-center overflow-hidden"
                onPointerDown={(e) => {
                  if (e.pointerType === 'mouse' && e.button !== 0) return;
                  historySwipeX0.current = e.clientX;
                }}
                onPointerUp={(e) => {
                  const x0 = historySwipeX0.current;
                  historySwipeX0.current = null;
                  if (x0 == null || sortedHistorySessions.length < 2) return;
                  const d = e.clientX - x0;
                  if (d > 56) setHistorySlideIdx((i) => Math.max(0, i - 1));
                  else if (d < -56) setHistorySlideIdx((i) => Math.min(sortedHistorySessions.length - 1, i + 1));
                }}
                onPointerCancel={() => {
                  historySwipeX0.current = null;
                }}
              >
                {sortedHistorySessions.length === 0 ? (
                  <p className="relative z-[1] px-6 text-center text-[14px] text-zinc-500">
                    {language === 'zh' ? '暂无会话记录' : 'No sessions yet'}
                  </p>
                ) : (
                  <>
                    <div
                      className="relative flex min-h-0 w-full flex-1 flex-col items-center justify-center overflow-hidden"
                      style={{ perspective: '1200px' }}
                    >
                      <div
                        className="relative flex min-h-0 w-full flex-1 items-center justify-center overflow-visible pb-2"
                        style={{ perspective: 'inherit' }}
                      >
                        <div
                          className="flex will-change-transform items-center gap-[30px] transition-[transform] duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
                          style={{
                            transform: `translateX(calc(50vw - ${HISTORY_CARD_W / 2 + historySlideIdx * (HISTORY_CARD_W + HISTORY_CARD_GAP)}px))`,
                          }}
                        >
                          {sortedHistorySessions.map((s, i) => {
                            const d = i - historySlideIdx;
                            const abs = Math.abs(d);
                            const rotY = d * -11;
                            const scale = Math.max(0.82, 1 - abs * 0.065);
                            const z = -abs * 42;
                            const opacity = abs > 2 ? 0.38 : 1;
                            return (
                              <div
                                key={s.id}
                                className="relative shrink-0"
                                style={{
                                  width: HISTORY_CARD_W,
                                  transform: `rotateY(${rotY}deg) translateZ(${z}px) scale(${scale})`,
                                  opacity,
                                  zIndex: 50 - abs,
                                  transformStyle: 'preserve-3d',
                                  transition: 'transform 0.5s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.45s ease',
                                }}
                              >
                                <button
                                  type="button"
                                  className={`group relative flex h-[min(52dvh,400px)] w-full flex-col overflow-hidden rounded-[1.35rem] border text-left shadow-[0_28px_90px_-24px_rgba(0,0,0,0.95)] touch-manipulation ${
                                    activeSessionId === s.id
                                      ? 'border-zinc-400/35 ring-1 ring-zinc-500/25'
                                      : 'border-zinc-800/90 hover:border-zinc-600/80'
                                  }`}
                                  onClick={() => {
                                    setActiveSessionId(s.id);
                                    setShowHistoryModal(false);
                                  }}
                                >
                                  {s.thumbnailDataUrl ? (
                                    <img
                                      src={s.thumbnailDataUrl}
                                      alt=""
                                      className="absolute inset-0 h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.04]"
                                    />
                                  ) : (
                                    <div className="absolute inset-0 bg-gradient-to-br from-zinc-900 via-[#0a0a0c] to-black">
                                      {[...Array(14)].map((_, k) => (
                                        <span
                                          key={k}
                                          className="absolute rounded-full bg-white/15 shadow-[0_0_10px_rgba(255,255,255,0.12)] motion-safe:animate-pulse"
                                          style={{
                                            width: 2 + (k % 3),
                                            height: 2 + (k % 3),
                                            left: `${8 + (k * 37) % 88}%`,
                                            top: `${12 + (k * 29) % 80}%`,
                                            animationDelay: `${k * 0.15}s`,
                                          }}
                                        />
                                      ))}
                                    </div>
                                  )}
                                  <div className="absolute inset-0 bg-gradient-to-t from-black via-black/55 to-black/10" />
                                  <div className="relative mt-auto flex flex-col gap-1.5 p-5 pt-16">
                                    <h3 className="line-clamp-3 font-serif text-[1.05rem] italic leading-snug tracking-wide text-zinc-100 md:text-lg">
                                      {safeSessionTheme(s.theme)}
                                    </h3>
                                    <p className="text-[10px] uppercase tracking-[0.28em] text-zinc-500">
                                      {new Date(s.updatedAt).toLocaleString()}
                                      {s.etchedToAlbum ? (language === 'zh' ? ' · 已镌刻' : ' · Saved') : ''}
                                    </p>
                                  </div>
                                </button>
                                <button
                                  type="button"
                                  className="absolute right-3 top-3 z-[3] flex h-10 w-10 items-center justify-center rounded-full border border-zinc-700/80 bg-black/55 text-zinc-400 backdrop-blur-md transition-colors hover:border-rose-900/50 hover:bg-rose-950/40 hover:text-rose-200 touch-manipulation"
                                  aria-label={language === 'zh' ? '删除' : 'Delete'}
                                  onClick={(ev) => {
                                    ev.stopPropagation();
                                    deleteSessionById(s.id);
                                  }}
                                >
                                  <Trash2 className="h-4 w-4" strokeWidth={1.75} />
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                    <div className="relative z-[4] flex shrink-0 justify-center gap-8 px-6 pb-2 pt-4">
                      <button
                        type="button"
                        className="flex h-11 min-w-[3rem] items-center justify-center rounded-full border border-zinc-700/90 bg-zinc-950/90 px-5 text-lg leading-none text-zinc-300 shadow-lg backdrop-blur-md transition-colors hover:border-zinc-500 hover:text-zinc-50 touch-manipulation disabled:pointer-events-none disabled:opacity-25"
                        disabled={historySlideIdx <= 0}
                        aria-label={language === 'zh' ? '上一张' : 'Previous'}
                        onClick={() => setHistorySlideIdx((i) => Math.max(0, i - 1))}
                      >
                        ‹
                      </button>
                      <button
                        type="button"
                        className="flex h-11 min-w-[3rem] items-center justify-center rounded-full border border-zinc-700/90 bg-zinc-950/90 px-5 text-lg leading-none text-zinc-300 shadow-lg backdrop-blur-md transition-colors hover:border-zinc-500 hover:text-zinc-50 touch-manipulation disabled:pointer-events-none disabled:opacity-25"
                        disabled={historySlideIdx >= sortedHistorySessions.length - 1}
                        aria-label={language === 'zh' ? '下一张' : 'Next'}
                        onClick={() =>
                          setHistorySlideIdx((i) => Math.min(sortedHistorySessions.length - 1, i + 1))
                        }
                      >
                        ›
                      </button>
                    </div>
                    <p className="relative z-[1] shrink-0 px-6 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-1 text-center text-[10px] uppercase tracking-[0.28em] text-zinc-600">
                      {language === 'zh' ? '左右滑动或按箭头翻阅 · 点卡片进入' : 'Swipe or use arrows · tap card to open'}
                    </p>
                  </>
                )}
              </div>
            </div>
          </div>,
          getAppPortalNode(),
        )}

      {/* Save Preview Modal — Portal 到 body，避免被父级 stacking 压在设置按钮下面 */}
      {showSavePreview &&
        activeSession &&
        createPortal(
        <div className="animate-in fade-in zoom-in-95 fixed inset-0 z-[10000] flex items-center justify-center bg-black/85 p-4 backdrop-blur-md duration-500 pointer-events-auto pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]">
          {/* Subtle noise/texture overlay for the whole modal area */}
          <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[url('https://www.transparenttextures.com/patterns/stardust.png')]" />

          <div className="relative flex max-h-[min(85dvh,92svh)] w-full max-w-lg flex-col overflow-hidden rounded-3xl border border-zinc-800/80 bg-zinc-950 shadow-[0_20px_100px_-20px_rgba(0,0,0,1),0_0_40px_rgba(255,255,255,0.02)] md:rounded-[2.5rem]">
            
            {/* Inner Glow/Spotlight Background */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-[2.5rem]">
               <div className="absolute top-[-20%] left-[-20%] w-[140%] h-[140%] bg-[radial-gradient(circle_at_center,rgba(39,39,42,0.4)_0%,transparent_60%)]" />
            </div>

            {/* Header */}
            <div className="relative flex flex-col gap-4 border-b border-zinc-800/40 p-6 md:p-10">
               <div className="flex justify-between items-center w-full">
                  <Dna className="h-6 w-6 text-zinc-700 opacity-50 md:h-5 md:w-5" strokeWidth={1} />
                  <div className="flex gap-4 text-xs font-medium uppercase tracking-[0.35em] text-zinc-600 md:text-[9px] md:tracking-[0.4em]">
                     <span>{new Date(activeSession.updatedAt).toLocaleDateString()}</span>
                     <span>{new Date(activeSession.updatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
               </div>
               <h2 className="mt-2 font-serif text-[1.65rem] italic leading-snug tracking-[0.08em] text-zinc-100 md:text-3xl md:leading-tight md:tracking-[0.1em]">
                 {safeSessionTheme(activeSession.theme)}
               </h2>
            </div>

            {/* Scrollable Content */}
            <div className="no-scrollbar relative flex-1 space-y-8 overflow-y-auto p-6 md:space-y-12 md:p-10">
               {activeSession.messages.map((m, idx) => (
                 <div key={idx} className={`relative flex flex-col ${m.role === 'user' ? 'items-end pl-12' : 'items-start pr-12'}`}>
                    {/* Role Label */}
                    <div className={`mb-3 text-[11px] font-bold uppercase tracking-[0.28em] md:text-[9px] md:tracking-[0.3em] ${m.role === 'user' ? 'text-zinc-700' : m.kind === 'closing_note' ? 'text-rose-400/70' : 'text-zinc-500'}`}>
                      {m.kind === 'closing_note'
                        ? language === 'zh'
                          ? '· 留给你的话 ·'
                          : '· A note for you ·'
                        : m.role === 'user'
                          ? language === 'zh'
                            ? '· 我 ·'
                            : '· SELF ·'
                          : aiName || (language === 'zh' ? '· 潜意识 ·' : '· SUB ·')}
                    </div>

                    {/* Message Body */}
                    <div className={`relative px-0 py-0 ${m.role === 'user' ? 'text-right' : `text-left pl-4 border-l ${m.kind === 'closing_note' ? 'border-rose-900/50' : 'border-zinc-800'}`}`}>
                       <p className={`text-[17px] leading-relaxed italic md:text-base ${
                         m.role === 'user' 
                           ? 'font-sans text-zinc-500 md:text-base' 
                           : m.kind === 'closing_note'
                             ? 'font-serif text-lg tracking-wide text-rose-100/90 md:text-lg'
                             : 'font-serif text-lg tracking-wide text-zinc-200 md:text-lg'
                       }`}>
                         {safeMsgText(m.text)}
                       </p>
                    </div>
                 </div>
               ))}
               
               {/* Decorative Bottom Mark */}
               <div className="pt-8 flex justify-center opacity-10">
                  <div className="w-12 h-px bg-zinc-500" />
                  <div className="mx-4 w-1 h-1 rounded-full bg-zinc-500" />
                  <div className="w-12 h-px bg-zinc-500" />
               </div>
            </div>

            {/* Actions */}
            <div className="relative flex gap-3 border-t border-zinc-900/50 bg-zinc-950/80 p-5 backdrop-blur-sm md:gap-5 md:p-10">
               <button 
                 type="button"
                 onClick={() => setShowSavePreview(false)}
                 className="min-h-[52px] flex-1 rounded-2xl border border-zinc-900 py-4 text-xs font-bold uppercase tracking-[0.28em] text-zinc-600 transition-all hover:bg-zinc-900 hover:text-zinc-400 touch-manipulation md:min-h-[48px] md:py-5 md:text-[10px] md:tracking-[0.3em]"
               >
                 {language === 'zh' ? '放弃' : 'DISCARD'}
               </button>
               <button 
                 type="button"
                 disabled={etchAdviceBusy}
                 onClick={async () => {
                    if (!activeSession || etchAdviceBusy) return;
                    const withoutClosing = activeSession.messages.filter((m) => m.kind !== 'closing_note');
                    let nextMessages: ChatMessage[] = [...withoutClosing];
                    if (conversationMode === 'text_clone' && getGeminiApiKey()) {
                      setEtchAdviceBusy(true);
                      const ac = new AbortController();
                      const tid = window.setTimeout(() => ac.abort(), CHAT_SEND_TIMEOUT_MS);
                      try {
                        const raw = await generateCloneClosingAdvice(withoutClosing, ac.signal);
                        const cleaned = sanitizeModelReplyText(raw || '');
                        if (cleaned.trim()) {
                          nextMessages = [
                            ...withoutClosing,
                            { role: 'model', text: cleaned.trim(), kind: 'closing_note' },
                          ];
                        }
                      } catch (e) {
                        console.error('Clone closing advice failed', e);
                      } finally {
                        window.clearTimeout(tid);
                        setEtchAdviceBusy(false);
                      }
                    }
                    setShowSavePreview(false);
                    const etched: ChatSession = {
                      ...activeSession,
                      messages: nextMessages,
                      etchedToAlbum: true,
                      etchedAt: Date.now(),
                      updatedAt: Date.now(),
                    };
                    const next = sessions.map((s) => (s.id === etched.id ? etched : s));
                    setSessions(next);
                    saveSessions(next);
                 }}
                 className="min-h-[52px] flex-1 rounded-2xl border border-zinc-800 bg-zinc-100/5 py-4 text-xs font-bold uppercase tracking-[0.28em] text-zinc-200 shadow-xl shadow-black/20 transition-all hover:border-zinc-700 hover:bg-zinc-100/10 touch-manipulation enabled:cursor-pointer disabled:cursor-wait disabled:opacity-60 md:min-h-[48px] md:py-5 md:text-[10px] md:tracking-[0.3em]"
               >
                 {etchAdviceBusy ? (
                   <span className="inline-flex items-center justify-center gap-2">
                     <Loader2 className="h-4 w-4 shrink-0 animate-spin" strokeWidth={2} />
                     {language === 'zh' ? '写入回响…' : 'Writing…'}
                   </span>
                 ) : language === 'zh' ? (
                   '镌刻记忆'
                 ) : (
                   'ETCH MEMORY'
                 )}
               </button>
            </div>
          </div>
        </div>,
        getAppPortalNode(),
        )}
    {voiceBlockingMessage &&
      createPortal(
        <div className="fixed inset-0 z-[10001] flex items-center justify-center bg-black/75 p-6 backdrop-blur-md pointer-events-auto pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]">
          <div
            role="dialog"
            aria-modal="true"
            className="max-h-[min(70dvh,26rem)] w-full max-w-sm overflow-y-auto rounded-2xl border border-zinc-700/80 bg-zinc-950 px-5 py-6 shadow-2xl"
          >
            <p className="text-left text-[14px] leading-relaxed text-zinc-300">{voiceBlockingMessage}</p>
            <div className="mt-6 flex flex-col gap-2">
              {voiceBlockingRetryRef.current ? (
                <button
                  type="button"
                  className="min-h-[44px] w-full rounded-xl border border-rose-600/60 bg-rose-950/60 py-3 text-sm font-medium text-rose-100 touch-manipulation active:bg-rose-900/60"
                  onClick={() => {
                    const fn = voiceBlockingRetryRef.current;
                    voiceBlockingRetryRef.current = null;
                    setVoiceBlockingMessage(null);
                    fn?.();
                  }}
                >
                  {language === 'zh' ? '重试' : 'Retry'}
                </button>
              ) : null}
              <button
                type="button"
                className="min-h-[44px] w-full rounded-xl border border-zinc-600 bg-zinc-900 py-3 text-sm font-medium text-zinc-100 touch-manipulation active:bg-zinc-800"
                onClick={() => { voiceBlockingRetryRef.current = null; setVoiceBlockingMessage(null); }}
              >
                {language === 'zh' ? '确定' : 'OK'}
              </button>
            </div>
          </div>
        </div>,
        getAppPortalNode(),
      )}
    </>
  );
});

ChatOverlay.displayName = 'ChatOverlay';

export default ChatOverlay;
