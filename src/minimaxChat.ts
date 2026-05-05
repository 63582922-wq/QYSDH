/// <reference types="vite/client" />

/**
 * 克隆模式专用：MiniMax 对话（与语音复刻/T2A 的 MINIMAX_API_KEY 可分离）。
 * - 纯文本：OpenAI 兼容 `POST /v1/chat/completions`
 * - 含图片：原生 `POST /v1/text/chatcompletion_v2`（兼容路径常不处理 image_url）
 * @see https://platform.minimax.io/docs/api-reference/text-openai-api
 */

import { normalizeMinimaxEnvValue } from './minimaxEnv';
import { getMinimaxApiKey, withMinimaxGroupQuery } from './minimaxTts';

export type MinimaxOpenAiChatMessage =
  | { role: 'system' | 'assistant'; content: string }
  | { role: 'user'; content: string }
  | {
      role: 'user';
      content: Array<
        { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
      >;
    };

/** 优先 MINIMAX_CHAT_API_KEY；留空则与语音链路共用 MINIMAX_API_KEY（一把钥匙即可）。 */
export function getMinimaxChatApiKey(): string {
  const chatOnly = normalizeMinimaxEnvValue(
    typeof process !== 'undefined' && typeof process.env.MINIMAX_CHAT_API_KEY === 'string'
      ? process.env.MINIMAX_CHAT_API_KEY
      : undefined,
  );
  if (chatOnly) return chatOnly;
  return getMinimaxApiKey();
}

/** 构建期注入：聊天代理是否与语音共用 `/minimax-api` */
function minimaxChatUsesSeparateDevProxy(): boolean {
  return (
    typeof process !== 'undefined' &&
    String(process.env.MINIMAX_CHAT_SEPARATE_PROXY || '').trim() === '1'
  );
}

export function resolveMinimaxChatApiRoot(): string {
  const chatBase =
    typeof process !== 'undefined' && typeof process.env.MINIMAX_CHAT_API_BASE === 'string'
      ? normalizeMinimaxEnvValue(process.env.MINIMAX_CHAT_API_BASE)
      : '';
  const voiceBase =
    typeof process !== 'undefined' && typeof process.env.MINIMAX_API_BASE === 'string'
      ? normalizeMinimaxEnvValue(process.env.MINIMAX_API_BASE)
      : '';

  if (import.meta.env.DEV) {
    return minimaxChatUsesSeparateDevProxy() ? '/minimax-chat-api' : '/minimax-api';
  }

  const root = (chatBase || voiceBase || 'https://api.minimax.io').replace(/\/$/, '');
  return root;
}

export function getMinimaxChatModel(): string {
  const m =
    typeof process !== 'undefined' && process.env.MINIMAX_CHAT_MODEL?.trim()
      ? process.env.MINIMAX_CHAT_MODEL.trim()
      : 'MiniMax-M2.7-highspeed';
  return m;
}

/** 带图时用原生 chatcompletion_v2；官方示例为 MiniMax-Text-01（与纯文本推理模型可不同）。 */
export function getMinimaxChatVisionModel(): string {
  const m =
    typeof process !== 'undefined' && process.env.MINIMAX_CHAT_VISION_MODEL?.trim()
      ? process.env.MINIMAX_CHAT_VISION_MODEL.trim()
      : 'MiniMax-Text-01';
  return m;
}

function minimaxMessageHasImage(m: MinimaxOpenAiChatMessage): boolean {
  if (m.role !== 'user') return false;
  if (typeof m.content === 'string') return false;
  return m.content.some((p) => p.type === 'image_url');
}

function minimaxMessagesHaveAnyImage(messages: MinimaxOpenAiChatMessage[]): boolean {
  return messages.some(minimaxMessageHasImage);
}

type MinimaxV2ChatJson = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
      reasoning_content?: string;
    };
  }>;
  error?: { message?: string };
  base_resp?: { status_msg?: string; status_code?: number };
};

function extractChoiceContent(raw: MinimaxV2ChatJson['choices']): string {
  const r = raw?.[0]?.message?.content;
  if (typeof r === 'string') return trimReply(r);
  if (Array.isArray(r)) {
    const parts = r
      .map((p) => {
        if (p && typeof p === 'object' && 'text' in p) return String((p as { text?: string }).text ?? '');
        return '';
      })
      .join('');
    return trimReply(parts);
  }
  return '';
}

async function minimaxNativeChatCompletionV2(params: {
  model: string;
  messages: MinimaxOpenAiChatMessage[];
  signal?: AbortSignal;
  temperature?: number;
  topP?: number;
  maxCompletionTokens?: number;
}): Promise<string> {
  const apiKey = getMinimaxChatApiKey();
  if (!apiKey) throw new Error('MINIMAX_API_KEY / MINIMAX_CHAT_API_KEY is not configured');

  const root = resolveMinimaxChatApiRoot();
  const url = withMinimaxGroupQuery(`${root}/v1/text/chatcompletion_v2`);

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: params.model,
      messages: params.messages,
      temperature: params.temperature ?? 0.75,
      top_p: params.topP ?? 0.8,
      max_completion_tokens: params.maxCompletionTokens ?? 2048,
    }),
    signal: params.signal,
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`MiniMax chatcompletion_v2 HTTP ${res.status}: ${t.slice(0, 400)}`);
  }

  const json = (await res.json()) as MinimaxV2ChatJson;

  if (json.error?.message) throw new Error(json.error.message);
  if (
    json.base_resp != null &&
    json.base_resp.status_code !== undefined &&
    json.base_resp.status_code !== 0
  ) {
    throw new Error(json.base_resp.status_msg || `MiniMax status_code=${json.base_resp.status_code}`);
  }

  const text = extractChoiceContent(json.choices);
  if (text) return text;
  throw new Error('MiniMax chatcompletion_v2 response missing choices[0].message.content');
}

function trimReply(s: string): string {
  let t = typeof s === 'string' ? s : String(s ?? '');
  t = t.replace(/\u0000/g, '').trim();
  return t;
}

export async function minimaxOpenAiChatCompletions(params: {
  messages: MinimaxOpenAiChatMessage[];
  signal?: AbortSignal;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
}): Promise<string> {
  const apiKey = getMinimaxChatApiKey();
  if (!apiKey) throw new Error('MINIMAX_API_KEY / MINIMAX_CHAT_API_KEY is not configured');

  if (minimaxMessagesHaveAnyImage(params.messages)) {
    return minimaxNativeChatCompletionV2({
      model: getMinimaxChatVisionModel(),
      messages: params.messages,
      signal: params.signal,
      temperature: params.temperature,
      topP: params.topP,
      maxCompletionTokens: params.maxTokens,
    });
  }

  const root = resolveMinimaxChatApiRoot();
  const url = withMinimaxGroupQuery(`${root}/v1/chat/completions`);
  const model = getMinimaxChatModel();

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: params.messages,
      temperature: params.temperature ?? 0.75,
      top_p: params.topP ?? 0.8,
      max_tokens: params.maxTokens ?? 2048,
    }),
    signal: params.signal,
  });

  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`MiniMax chat HTTP ${res.status}: ${t.slice(0, 400)}`);
  }

  const json = (await res.json()) as MinimaxV2ChatJson;

  if (json.error?.message) throw new Error(json.error.message);
  if (json.base_resp != null && json.base_resp.status_code !== undefined && json.base_resp.status_code !== 0) {
    throw new Error(json.base_resp.status_msg || `MiniMax status_code=${json.base_resp.status_code}`);
  }

  const text = extractChoiceContent(json.choices);
  if (text) return text;
  throw new Error('MiniMax chat response missing choices[0].message.content');
}
