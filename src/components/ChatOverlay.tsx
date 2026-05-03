import React, { useState, useEffect, useRef, useMemo, useCallback, forwardRef, useImperativeHandle } from 'react';
import { createPortal } from 'react-dom';
import { Send, Loader2, Mic, MicOff, Dna } from 'lucide-react';
import { GoogleGenAI, Type, LiveServerMessage, Modality } from '@google/genai';

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  isInitialImage?: boolean;
}

export interface ChatSession {
  id: string;
  theme: string;
  messages: ChatMessage[];
  updatedAt: number;
}

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
}

export interface ChatOverlayHandle {
  openSavePreview: () => void;
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

if (!process.env.GEMINI_API_KEY) {
  console.error('CRITICAL: GEMINI_API_KEY is not defined in environment!');
}

const TEXT_MODEL_CANDIDATES = ['gemini-flash-latest', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash'];

/** 移动端网络慢时避免永远卡在「等待回复」 */
const CHAT_SEND_TIMEOUT_MS = 70000;

/** 过长回复写入 state / localStorage 时截断，避免撑爆内存 */
const MAX_REPLY_CHARS = 8000;
/** 悬浮气泡内最长展示字符，超出用省略（全文仍在会话里） */
const MAX_FLOAT_PREVIEW_CHARS = 900;
const MAX_TTS_CHARS = 4000;

/** AI 悬浮句：仅 CSS 渐显 / 渐隐；无用户新消息时固定时长后消隐 */
const FLOAT_FADE_MS = 500;
/** 自浮层展示完成后起算：用户若一直不发下一条，则消隐 */
const FLOAT_IDLE_HIDE_MS = 15000;

const isModelNotFoundError = (error: any) => {
  const message = String(error?.message || '').toLowerCase();
  return message.includes('not found') || message.includes('is not supported');
};

/** Short user-facing text; avoids dumping huge JSON (e.g. quota / 429). */
/** 避免 localStorage 损坏或异步完成后解析抛错导致整页白屏 */
function parseStoredSessions(raw: string | null): ChatSession[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function sanitizeModelReplyText(text: string): string {
  let s = typeof text === 'string' ? text : String(text ?? '');
  s = s.replace(/\u0000/g, '');
  if (s.length > MAX_REPLY_CHARS) {
    return `${s.slice(0, MAX_REPLY_CHARS)}\n…`;
  }
  return s;
}

function extractModelReplyText(response: unknown, fallback: string): string {
  try {
    const t = (response as { text?: string })?.text;
    if (typeof t === 'string' && t.trim()) return sanitizeModelReplyText(t.trim());
  } catch {
    /* ignore */
  }
  return fallback;
}

function formatGeminiUserMessage(error: unknown, lang: 'zh' | 'en'): string {
  const raw = String((error as any)?.message ?? error ?? '');
  const lower = raw.toLowerCase();
  if (raw === 'CLIENT_TIMEOUT' || lower === 'client_timeout') {
    return lang === 'zh' ? '请求超时，请检查网络后重试。' : 'Request timed out. Check your network and retry.';
  }
  if (
    raw.includes('429') ||
    lower.includes('resource_exhausted') ||
    lower.includes('quota') ||
    lower.includes('exceeded your current quota') ||
    lower.includes('rate limit')
  ) {
    return lang === 'zh'
      ? '已达到 Gemini 免费额度或频率上限，请稍后再试；或在 Google AI Studio 查看用量 / 升级计费。'
      : 'Gemini free-tier quota or rate limit reached. Wait and retry, check usage in AI Studio, or upgrade billing.';
  }
  if (raw.length > 280) {
    return lang === 'zh' ? '请求失败（详情请打开浏览器控制台查看）' : 'Request failed (see browser console for details).';
  }
  return lang === 'zh' ? `请求失败：${raw}` : `Request failed: ${raw}`;
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
  },
  ref,
) {
  type AIStatus = 'connected' | 'failed';
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [aiStatus, setAiStatus] = useState<AIStatus>(() =>
    process.env.GEMINI_API_KEY ? 'connected' : 'failed'
  );
  const [aiStatusText, setAiStatusText] = useState(() =>
    !process.env.GEMINI_API_KEY
      ? language === 'zh'
        ? '未检测到 GEMINI_API_KEY'
        : 'Missing GEMINI_API_KEY'
      : language === 'zh'
        ? '已就绪 · 发送消息开始对话'
        : 'Ready · send a message to start'
  );
  const [activeTextModel, setActiveTextModel] = useState(TEXT_MODEL_CANDIDATES[0]);
  const [floatingAiText, setFloatingAiText] = useState('');
  const [displayedFloatingText, setDisplayedFloatingText] = useState('');
  const [showFloatingAiText, setShowFloatingAiText] = useState(false);
  const [showSavePreview, setShowSavePreview] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);
  const lastShownModelKeyRef = useRef('');
  const lastSpokenMessageIdRef = useRef<string>(''); // NEW: Track exactly which message was spoken
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
  const onSpeechValueRef = useRef(onSpeechValue);
  useEffect(() => {
    onSpeechValueRef.current = onSpeechValue;
  }, [onSpeechValue]);

  // Real-time Voice State
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [isVoiceConnecting, setIsVoiceConnecting] = useState(false);
  const liveSessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  /** 增益为 0，避免麦克风直连扬声器产生啸叫，同时满足 ScriptProcessor 需接入图的约束 */
  const processorMuteRef = useRef<GainNode | null>(null);
  /** Live onopen 后为 true；超时判断不能用 isTyping/isVoiceMode 闭包 */
  const voiceLiveReadyRef = useRef(false);
  const nextPlayTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const lastChatErrorRef = useRef<unknown>(null);

  const isAIActive = isTyping || isSpeaking || isVoiceConnecting;

  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;
  const activeSessionRef = useRef(activeSession);
  activeSessionRef.current = activeSession;

  useImperativeHandle(
    ref,
    () => ({
      openSavePreview: () => {
        if (activeSessionRef.current) setShowSavePreview(true);
      },
    }),
    [],
  );

  /** 仅当「新增了一条 model 消息」时变化；用户追加 user 消息不会改变（避免误触发朗读/打字）。 */
  const latestModelTurnKey = useMemo(() => {
    if (!activeSession?.messages?.length) return '';
    for (let i = activeSession.messages.length - 1; i >= 0; i -= 1) {
      if (activeSession.messages[i].role === 'model') {
        const m = activeSession.messages[i];
        return `${activeSession.id}-model-${i}-${m.text}`;
      }
    }
    return '';
  }, [activeSession?.id, activeSession?.messages]);

  /** 只显示「当前这一轮」尚未被 AI 回应的用户句，避免手机上用户气泡堆满屏 */
  const pendingUserEcho = useMemo(() => {
    const msgs = activeSession?.messages;
    if (!msgs?.length) return null;
    let lastUserIdx = -1;
    let lastUserText = '';
    for (let i = msgs.length - 1; i >= 0; i -= 1) {
      if (msgs[i].role === 'user') {
        lastUserIdx = i;
        lastUserText = msgs[i].text;
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
    return waitingForThisUserTurn ? lastUserText : null;
  }, [activeSession?.messages]);

  // Auto-start session when an image is loaded
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

  // Refresh status strip when language toggles (no extra API calls — saves free-tier quota).
  useEffect(() => {
    if (!process.env.GEMINI_API_KEY) {
      setAiStatus('failed');
      setAiStatusText(language === 'zh' ? '未检测到 GEMINI_API_KEY' : 'Missing GEMINI_API_KEY');
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

  // 用户发出新一句（等待 AI）时收起浮层并清掉自动消隐计时，避免与上一轮 15s 计时打架
  useEffect(() => {
    const msgs = activeSession?.messages;
    if (!msgs?.length) return;
    if (msgs[msgs.length - 1].role !== 'user') return;
    clearFloatingDisplayTimers();
    setShowFloatingAiText(false);
    setFloatingAiText('');
    setDisplayedFloatingText('');
  }, [activeSession?.messages, clearFloatingDisplayTimers]);

  // AI 悬浮回复：新回复立即覆盖文案；展示满 FLOAT_IDLE_HIDE_MS 后渐隐（无用户插话时）
  // 语音/TTS：必须用「最后一条 model 消息的下标」做 key。若用 messages.length，
  // 用户每发一条消息长度就变，最新仍是旧 AI 回复，会误判为新消息 → 从头再朗读一遍（复述感）。
  useEffect(() => {
    let voicesListener: (() => void) | null = null;
    let voicesFallbackId: number | null = null;

    const cleanupSpeechVoicesWait = () => {
      if (voicesListener) {
        window.speechSynthesis.removeEventListener('voiceschanged', voicesListener);
        voicesListener = null;
      }
      if (voicesFallbackId !== null) {
        window.clearTimeout(voicesFallbackId);
        voicesFallbackId = null;
      }
    };

    if (!latestModelTurnKey || !activeSession?.messages?.length) return () => {};

    let latestModelMessage: ChatMessage | null = null;
    let latestModelIndex = -1;
    for (let i = activeSession.messages.length - 1; i >= 0; i -= 1) {
      if (activeSession.messages[i].role === 'model') {
        latestModelMessage = activeSession.messages[i];
        latestModelIndex = i;
        break;
      }
    }
    if (!latestModelMessage?.text || latestModelIndex < 0) return () => {};

    const messageKey = latestModelTurnKey;
    if (messageKey === lastShownModelKeyRef.current) return () => {};
    lastShownModelKeyRef.current = messageKey;

    const speakNewMessage = (text: string, messageId: string) => {
      if (!isAutoSpeak) return;
      if (typeof window.speechSynthesis === 'undefined') return;
      if (lastSpokenMessageIdRef.current === messageId) return;
      lastSpokenMessageIdRef.current = messageId;
      window.speechSynthesis.cancel();

      const speakSlice = text.length > MAX_TTS_CHARS ? `${text.slice(0, MAX_TTS_CHARS)}…` : text;

      const applyVoiceAndSpeak = () => {
        try {
          const utterance = new SpeechSynthesisUtterance(speakSlice);
          const voices = window.speechSynthesis.getVoices();
          const preferredVoice =
            voices.find((v) => (v.name.includes('Google') || v.name.includes('Premium')) && v.lang.includes('zh')) ||
            voices.find((v) => v.lang.includes('zh'));
          if (preferredVoice) utterance.voice = preferredVoice;
          utterance.lang = language === 'zh' ? 'zh-CN' : 'en-US';
          utterance.rate = 0.85;
          utterance.pitch = 0.9;
          window.speechSynthesis.speak(utterance);
        } catch (e) {
          console.warn('speechSynthesis failed', e);
        }
      };

      const voices = window.speechSynthesis.getVoices();
      if (voices.length === 0) {
        voicesListener = () => {
          cleanupSpeechVoicesWait();
          applyVoiceAndSpeak();
        };
        window.speechSynthesis.addEventListener('voiceschanged', voicesListener);
        voicesFallbackId = window.setTimeout(() => {
          cleanupSpeechVoicesWait();
          applyVoiceAndSpeak();
        }, 600);
      } else {
        applyVoiceAndSpeak();
      }
    };

    const fullText = sanitizeModelReplyText(latestModelMessage.text);
    const preview =
      fullText.length <= MAX_FLOAT_PREVIEW_CHARS
        ? fullText
        : `${fullText.slice(0, MAX_FLOAT_PREVIEW_CHARS)}…`;

    let fadeMs = FLOAT_FADE_MS;
    let idleHideMs = FLOAT_IDLE_HIDE_MS;
    try {
      if (typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        fadeMs = 120;
        idleHideMs = Math.min(idleHideMs, 10000);
      }
    } catch {
      /* ignore */
    }

    setFloatingAiText(fullText);
    setDisplayedFloatingText(preview);
    setShowFloatingAiText(true);
    onSpeechValueRef.current?.(0.38);

    speakNewMessage(fullText, messageKey);

    floatIdleHideTimerRef.current = window.setTimeout(() => {
      floatIdleHideTimerRef.current = null;
      setShowFloatingAiText(false);
      onSpeechValueRef.current?.(0);
      floatAfterFadeTimerRef.current = window.setTimeout(() => {
        floatAfterFadeTimerRef.current = null;
        setFloatingAiText('');
        setDisplayedFloatingText('');
      }, fadeMs);
    }, idleHideMs);

    return () => {
      clearFloatingDisplayTimers();
      cleanupSpeechVoicesWait();
      window.speechSynthesis.cancel();
    };
  }, [latestModelTurnKey, activeSession, isAutoSpeak, language, clearFloatingDisplayTimers]);

  const generateWithFallback = async (contents: any[], config?: any) => {
    let lastError: any = null;
    for (const model of TEXT_MODEL_CANDIDATES) {
      try {
        const result = await ai.models.generateContent({
          model,
          contents,
          config,
        });
        setActiveTextModel(model);
        return result;
      } catch (e) {
        lastError = e;
        if (!isModelNotFoundError(e)) {
          throw e;
        }
      }
    }
    throw lastError || new Error('No available text model');
  };

  const saveSessions = (newSessions: ChatSession[]) => {
    setSessions(newSessions);
    try {
      localStorage.setItem('subconscious_sessions', JSON.stringify(newSessions));
    } catch (e) {
      console.error('Failed to persist sessions (quota / private mode)', e);
      try {
        const lighter = newSessions.map((s) => ({
          ...s,
          messages: Array.isArray(s.messages) ? s.messages.slice(-25) : [],
        }));
        localStorage.setItem('subconscious_sessions', JSON.stringify(lighter));
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
  };

  const getSystemInstruction = () => {
    const langInstructions = language === 'zh' 
      ? '使用中文。'
      : 'Use English.';
    
    return `${langInstructions}
你是一个温和、有洞察力的朋友。用户会向你分享图片和心事，请你以平等的、自然的口吻进行交流。

对话准则：
1. 自然对话：像真正的朋友一样说话。不要使用“作为AI”、“这张图像显示了”这类生硬的词汇。
2. 真实回应：根据用户的情绪给出自然的回应。你可以表达你的感受，但不要原样复述用户的话。
3. 引导提问：在回复的结尾，提出一个有启发性的问题，引导用户进一步探索自己的想法或感受。
4. 简洁明了：保持回复在2-3句话以内。
5. 禁止废话：不要道歉，不要表现得像个机器人助手。`;
  };

  const generateTheme = async (messages: ChatMessage[]) => {
    if (messages.length < 2) return language === 'zh' ? '新的探索' : 'New Exploration';
    
    try {
      const chatLog = messages.map(m => `${m.role}: ${m.text}`).join('\n');
      const response = await generateWithFallback(
        [{ text: `Based on this conversation, generate a very short (2-4 words) poetic theme/title.\n\nConversation:\n${chatLog}` }],
        {
          systemInstruction: `Return ONLY the short title. The title must be in ${language === 'zh' ? 'Chinese' : 'English'}. Be poetic and mysterious.`,
        }
      );
      return response.text?.trim() || (language === 'zh' ? '潜意识回响' : 'Subconscious Echoes');
    } catch (e) {
      console.error('Failed to generate theme', e);
      return language === 'zh' ? '潜意识片段' : 'Subconscious Fragment';
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim() && !currentImageDataUrl) return;
    
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
    
    const userMessage: ChatMessage = { role: 'user', text: inputText.trim() || (language === 'zh' ? '[分享了一张图像]' : '[Shared an image]') };
    
    // Check if this is the very first message and there is an image to attach
    const isFirstMessageWithImage = session.messages.length === 0 && currentImageDataUrl;
    if (isFirstMessageWithImage) {
      userMessage.isInitialImage = true;
    }

    session.messages = [...session.messages, userMessage];
    session.updatedAt = Date.now();
    currentSessions[sessionIndex] = session;
    saveSessions(currentSessions);
    
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);

    try {
      // Build History
      const history = session.messages.slice(0, -1).map(m => ({
        role: m.role,
        parts: [{ text: m.text }]
      }));

      let contents: any[] = [{ text: currentInput || (language === 'zh' ? '我上传了一张图像，看到它你有什么感觉？' : 'I uploaded an image, what do you feel about it?') }];
      
      if (isFirstMessageWithImage && currentImageDataUrl) {
        // Extract base64 and mime type
        const match = currentImageDataUrl.match(/^data:(image\/[a-zA-Z+]+);base64,(.+)$/);
        if (match) {
           contents.unshift({
             inlineData: {
               mimeType: match[1],
               data: match[2],
             }
           });
        }
      }

      let response: any = null;
      let lastModelError: any = null;
      for (const model of TEXT_MODEL_CANDIDATES) {
        try {
          const chat = ai.chats.create({
            model,
            config: {
              systemInstruction: getSystemInstruction(),
              temperature: 0.75, // Lowered for more coherent, friendly conversation
              topP: 0.8,
              topK: 40,
            },
            history: history.length > 0 ? history : undefined,
          });
          response = await Promise.race([
            chat.sendMessage({ message: contents }),
            new Promise<never>((_, reject) => {
              window.setTimeout(() => reject(new Error('CLIENT_TIMEOUT')), CHAT_SEND_TIMEOUT_MS);
            }),
          ]);
          setActiveTextModel(model);
          break;
        } catch (e) {
          lastModelError = e;
          if (!isModelNotFoundError(e)) {
            throw e;
          }
        }
      }
      if (!response) {
        throw lastModelError || new Error('No available chat model');
      }
      
      const aiMessage: ChatMessage = {
        role: 'model',
        text: extractModelReplyText(response, '...'),
      };

      // update session again（主题生成另起异步，避免阻塞 isTyping → 顶部一直转圈）
      const fromStorage = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
      let latestSessions = fromStorage.length > 0 ? fromStorage : [...currentSessions];
      let latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
      if (latestIndex < 0) {
        latestSessions = [...currentSessions];
        latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
      }

      if (latestIndex >= 0) {
        const updatedSession = { ...latestSessions[latestIndex] };
        updatedSession.messages = [...(updatedSession.messages || []), aiMessage];
        updatedSession.updatedAt = Date.now();

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
      
    } catch (error) {
      console.error('Chat error:', error);
      lastChatErrorRef.current = error;
      setAiStatus('failed');
      setAiStatusText(formatGeminiUserMessage(error, language));
      const errSessions = parseStoredSessions(localStorage.getItem('subconscious_sessions'));
      const latestSessions = errSessions.length > 0 ? errSessions : [...currentSessions];
      const latestIndex = latestSessions.findIndex((s: ChatSession) => s.id === currentId);
      if (latestIndex >= 0) {
        const updatedSession = {
          ...latestSessions[latestIndex],
          messages: [
            ...(latestSessions[latestIndex].messages || []),
            { role: 'model' as const, text: formatGeminiUserMessage(error, language) },
          ],
          updatedAt: Date.now(),
        };
        latestSessions[latestIndex] = updatedSession;
        saveSessions(latestSessions);
      }
    } finally {
      setIsTyping(false);
    }
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
    // Send actual audio energy to particles (multiplied for visual effect)
    if (onSpeechValue) onSpeechValue(rms * 12);
    
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    
    const playTime = Math.max(audioCtx.currentTime, nextPlayTimeRef.current);
    source.start(playTime);
    nextPlayTimeRef.current = playTime + audioBuffer.duration;

    audioSourcesRef.current.push(source);
    
    setIsSpeaking(true);
    source.onended = () => {
       audioSourcesRef.current = audioSourcesRef.current.filter(s => s !== source);
       if (audioSourcesRef.current.length === 0) {
           setIsSpeaking(false);
           if (onSpeechValue) onSpeechValue(0); // Reset vibration
       }
    };
  };

  const stopVoiceMode = () => {
    voiceLiveReadyRef.current = false;
    setIsVoiceMode(false);
    setIsVoiceConnecting(false);
    setIsSpeaking(false);

    if (processorMuteRef.current) {
      try {
        processorMuteRef.current.disconnect();
      } catch {
        /* ignore */
      }
      processorMuteRef.current = null;
    }
    if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
    }
    if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(t => t.stop());
        mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
        audioSourcesRef.current.forEach(s => s.stop());
        audioSourcesRef.current = [];
        audioContextRef.current.close();
        audioContextRef.current = null;
    }
    if (liveSessionRef.current) {
         try {
             liveSessionRef.current.close(); // Not officially documented but might be cancel or something
         } catch(e) {}
         liveSessionRef.current = null;
    }
  };

  const startVoiceMode = async () => {
     voiceLiveReadyRef.current = false;
     setIsVoiceConnecting(true);

     const timeoutMsg =
       language === 'zh'
         ? '语音连接超时，请检查网络与 API；若用手机访问 http://局域网地址，请先改用 HTTPS，否则麦克风会被系统禁用。'
         : 'Voice timed out. Try HTTPS if you opened this page over HTTP on LAN.';

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
       setVoiceBlockingMessage(
         language === 'zh'
           ? '请在需要「安全网页」的环境使用语音（HTTPS 或本机 localhost）。由于手机打开 http://192.168… 等局域网地址时，浏览器通常会锁定麦克风权限，语音模式无法工作。请改用 HTTPS 部署本页面，或继续使用下方文字对话。'
           : 'Voice needs a secure context (HTTPS or localhost). Mic access is blocked on plain HTTP (e.g. http://192.168.x.x) on many mobile browsers. Deploy over HTTPS or use text chat below.'
       );
       return;
     }

     if (!navigator.mediaDevices?.getUserMedia) {
       window.clearTimeout(timeoutId);
       setIsVoiceConnecting(false);
       setVoiceBlockingMessage(
         language === 'zh' ? '当前浏览器不支持麦克风（无 getUserMedia）。' : 'This browser does not support the microphone API.'
       );
       return;
     }

     try {
         const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
         audioContextRef.current = audioContext;
         await audioContext.resume();

         let stream: MediaStream;
         try {
           stream = await navigator.mediaDevices.getUserMedia({
             audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
           });
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
                 ? `麦克风错误：${name || micErr}`
                 : `Microphone error: ${name || micErr}`
           );
           return;
         }
         mediaStreamRef.current = stream;

         const source = audioContext.createMediaStreamSource(stream);
         const processor = audioContext.createScriptProcessor(4096, 1, 1);
         processorRef.current = processor;

         const muteOut = audioContext.createGain();
         muteOut.gain.value = 0;
         processorMuteRef.current = muteOut;

         source.connect(processor);
         processor.connect(muteOut);
         muteOut.connect(audioContext.destination);

        // Start connecting to Live API
        const sessionPromise = ai.live.connect({
            model: "gemini-2.0-flash", // Use the GA/Stable preview model
            callbacks: {
              onopen: async () => {
                  try {
                    await audioContext.resume();
                  } catch {
                    /* ignore */
                  }
                  voiceLiveReadyRef.current = true;
                  window.clearTimeout(timeoutId);
                  setIsVoiceConnecting(false);
                  setIsVoiceMode(true);
                  console.log('Live voice connection established');
               },
               onmessage: async (message: LiveServerMessage) => {
                   if (message.serverContent?.interrupted) {
                       audioSourcesRef.current.forEach(s => s.stop());
                       audioSourcesRef.current = [];
                       nextPlayTimeRef.current = audioContextRef.current?.currentTime || 0;
                       setIsSpeaking(false);
                   }
                   
                   const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                   if (base64Audio) {
                       handleAudioOutput(base64Audio);
                   }
               },
               onclose: (e: any) => {
                  console.log("Live voice connection closed", e);
                  stopVoiceMode();
               },
               onerror: (e: any) => {
                  console.error("Live voice connection error", e);
                  setVoiceBlockingMessage(
                    language === 'zh' ? '语音连接错误，请检查网络或 API 权限' : 'Voice connection error. Check network or API permissions.'
                  );
                  stopVoiceMode();
               }
            },
             config: {
               responseModalities: [Modality.AUDIO],
               speechConfig: {
                 voiceConfig: { prebuiltVoiceConfig: { voiceName: "Puck" } },
               },
               systemInstruction: getSystemInstruction(),
             }
         });
         
         const session = await sessionPromise;
         liveSessionRef.current = session;

         processor.onaudioprocess = (e) => {
           const inputData = e.inputBuffer.getChannelData(0);
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

           session.sendRealtimeInput({
             audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' },
           });
         };

         if (currentImageDataUrl) {
           const match = currentImageDataUrl.match(/^data:(image\/[a-zA-Z+]+);base64,(.+)$/);
           if (match) {
             session.sendRealtimeInput({
               video: { data: match[2], mimeType: match[1] },
             });
           }
         }

     } catch(e) {
         window.clearTimeout(timeoutId);
         console.error('Failed to start voice mode', e);
         stopVoiceMode();
         setVoiceBlockingMessage(
           language === 'zh'
             ? '无法启动语音：请检查网络、API Key，以及是否使用 HTTPS / localhost。详情见浏览器控制台。'
             : 'Could not start voice. Check network, API key, and HTTPS/localhost. See console for details.'
         );
     }
  };

  const toggleVoiceMode = () => {
     if (isVoiceMode || isVoiceConnecting) {
         stopVoiceMode();
     } else {
         startVoiceMode();
     }
  };

  // Cleanup
  useEffect(() => {
     return () => {
        clearFloatingDisplayTimers();
         stopVoiceMode();
     };
  }, [clearFloatingDisplayTimers]);

  if (!isOpen) return null;

  return (
    <>
    <div
      className={`pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-end pb-[calc(env(safe-area-inset-bottom)+4.75rem)] transition-opacity duration-700 max-md:pb-[calc(env(safe-area-inset-bottom)+5rem)] md:pb-[calc(env(safe-area-inset-bottom)+6.25rem)] ${isOpen ? 'opacity-100' : 'opacity-0'}`}
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
          
          {/* 动态光圈扩散 (Ping) */}
          {(isTyping || isVoiceConnecting) && (
            <div className="absolute inset-0 rounded-full border-2 border-zinc-400/30 animate-ping" />
          )}
          
          {/* 说话脉冲 (Pulse) */}
          {isSpeaking && (
            <div className="absolute -inset-3 rounded-full border border-zinc-400/20 animate-pulse" />
          )}
        </div>
        
        <span
          className={`max-w-[min(14rem,62vw)] truncate text-[11px] font-medium uppercase tracking-[0.2em] drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)] transition-all duration-700 whitespace-nowrap md:max-w-none md:text-base md:tracking-[0.3em] ${
            isAIActive ? 'text-zinc-200 opacity-100' : 'text-zinc-500 opacity-80'
          }`}
        >
          {aiName || (language === 'zh' ? '潜意识回响' : 'ECHO OF MIND')}
        </span>
      </div>

      {/* 仅输入区与气泡列表接收触摸，避免整块底栏挡住底栏胶囊键 */}
      <div className="pointer-events-none flex w-full max-w-xl flex-col gap-2 px-3 sm:px-4">

        {/* Body */}
        <div
          className="no-scrollbar pointer-events-auto relative max-h-[16dvh] touch-pan-y overflow-y-auto p-1 sm:max-h-[36vh] md:max-h-[44vh]"
          style={{
            maskImage: 'linear-gradient(to bottom, transparent, black 8%, black 92%, transparent)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 8%, black 92%, transparent)',
          }}
        >
             <div className="flex flex-col gap-4 py-8 md:gap-6 md:py-12">
                {pendingUserEcho != null && (
                  <div className="flex flex-col items-end animate-in fade-in duration-300">
                    <div className="max-w-[90%] rounded-2xl rounded-br-md border border-zinc-700/50 bg-zinc-800/60 px-3 py-2.5 text-[13px] leading-snug tracking-wide text-zinc-200 backdrop-blur-md md:max-w-[85%] md:rounded-3xl md:p-4 md:text-sm md:leading-relaxed">
                      {pendingUserEcho}
                    </div>
                    {isTyping && (
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

        {/* Input Area */}
        <div className="pointer-events-auto transition-all duration-500">
           {isVoiceMode && (
             <div className="mb-1.5 flex min-h-[38px] items-center justify-between rounded-full border border-zinc-800 bg-zinc-900/60 px-4 py-2 backdrop-blur-md md:mb-2 md:min-h-[48px] md:px-6 md:py-3">
                <div className="flex min-w-0 items-center gap-2 md:gap-3">
                   <div className={`h-2 w-2 shrink-0 rounded-full md:h-2 ${isSpeaking ? 'animate-pulse bg-zinc-200' : 'bg-zinc-500'}`} />
                   <span className="truncate text-[10px] font-medium uppercase tracking-wider text-zinc-400 md:text-xs md:tracking-widest">
                     {isVoiceConnecting ? (language === 'zh' ? '连接中...' : 'Connecting...') : (isSpeaking ? (language === 'zh' ? '回响中...' : 'Echoing...') : (language === 'zh' ? '倾听中...' : 'Listening...'))}
                   </span>
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
           )}
          <div className="relative mx-auto flex w-full max-w-md items-center">
             <div className="relative flex min-h-[42px] flex-1 items-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/60 shadow-lg backdrop-blur-md transition-colors focus-within:border-zinc-600 md:min-h-[48px] md:rounded-2xl md:shadow-xl">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  enterKeyHint="send"
                  inputMode="text"
                  autoComplete="off"
                  autoCorrect="on"
                  placeholder={isVoiceMode ? (language === 'zh' ? '说话或打字…' : 'Speak or type…') : (language === 'zh' ? '对话…' : 'Message…')}
                  className="max-h-24 min-h-[42px] w-full flex-1 resize-none bg-transparent px-3 py-2 text-[15px] leading-snug text-zinc-200 placeholder:text-zinc-600 focus:outline-none md:max-h-28 md:min-h-[48px] md:px-4 md:py-3 md:text-sm md:leading-normal"
                  rows={1}
                />
                <button
                  type="button"
                  onClick={toggleVoiceMode}
                  className={`flex h-10 min-h-[40px] w-10 min-w-[40px] shrink-0 items-center justify-center text-zinc-500 transition-colors hover:text-zinc-200 touch-manipulation active:bg-zinc-800/50 md:h-11 md:min-h-[44px] md:w-11 md:min-w-[44px] ${isVoiceMode || isVoiceConnecting ? 'text-rose-500' : ''}`}
                  aria-label={language === 'zh' ? '语音对话' : 'Voice'}
                >
                  {isVoiceConnecting ? <Loader2 strokeWidth={1.5} className="h-[18px] w-[18px] animate-spin md:h-5 md:w-5" /> : <Mic strokeWidth={1.5} className="h-[18px] w-[18px] md:h-5 md:w-5" />}
                </button>
                <button
                  type="button"
                  onClick={() => void sendMessage()}
                  disabled={isTyping || (!inputText.trim() && (!currentImageDataUrl || (activeSession?.messages?.length || 0) > 0))}
                  className="flex h-10 min-h-[40px] w-10 min-w-[40px] shrink-0 items-center justify-center text-zinc-500 transition-colors hover:text-zinc-200 disabled:opacity-30 touch-manipulation active:bg-zinc-800/50 md:h-11 md:min-h-[44px] md:w-11 md:min-w-[44px]"
                  aria-label={language === 'zh' ? '发送' : 'Send'}
                >
                  <Send strokeWidth={1.5} className="h-5 w-5 md:h-[22px] md:w-[22px]" />
                </button>
             </div>
          </div>
        </div>

      </div>

      {floatingAiText && (
        <div
          className={`pointer-events-none absolute left-1/2 flex w-[min(92vw,28rem)] max-w-md -translate-x-1/2 flex-col items-center gap-2 transition-[opacity,transform] duration-500 ease-out motion-reduce:duration-150 bottom-[calc(env(safe-area-inset-bottom)+9.25rem)] md:bottom-[calc(env(safe-area-inset-bottom)+11.5rem)] ${
            showFloatingAiText ? 'opacity-100 translate-y-0' : 'pointer-events-none opacity-0 translate-y-1'
          }`}
        >
          <div className="font-ai-reply rounded-xl border border-zinc-700/50 bg-zinc-900/60 px-3 py-2.5 text-center italic text-[15px] leading-snug tracking-wide text-zinc-200 shadow-xl backdrop-blur-xl md:rounded-2xl md:px-5 md:py-3.5 md:text-[1.0625rem] md:leading-relaxed">
            {displayedFloatingText}
          </div>
        </div>
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
                 {activeSession.theme}
               </h2>
            </div>

            {/* Scrollable Content */}
            <div className="no-scrollbar relative flex-1 space-y-8 overflow-y-auto p-6 md:space-y-12 md:p-10">
               {activeSession.messages.map((m, idx) => (
                 <div key={idx} className={`relative flex flex-col ${m.role === 'user' ? 'items-end pl-12' : 'items-start pr-12'}`}>
                    {/* Role Label */}
                    <div className={`mb-3 text-[11px] font-bold uppercase tracking-[0.28em] md:text-[9px] md:tracking-[0.3em] ${m.role === 'user' ? 'text-zinc-700' : 'text-zinc-500'}`}>
                      {m.role === 'user' ? (language === 'zh' ? '· 我 ·' : '· SELF ·') : (aiName || (language === 'zh' ? '· 潜意识 ·' : '· SUB ·'))}
                    </div>

                    {/* Message Body */}
                    <div className={`relative px-0 py-0 ${m.role === 'user' ? 'text-right' : 'text-left pl-4 border-l border-zinc-800'}`}>
                       <p className={`text-[17px] leading-relaxed italic md:text-base ${
                         m.role === 'user' 
                           ? 'font-sans text-zinc-500 md:text-base' 
                           : 'font-serif text-lg tracking-wide text-zinc-200 md:text-lg'
                       }`}>
                         {m.text}
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
                 onClick={() => {
                    setShowSavePreview(false);
                    // Minimal success toast could be added here
                 }}
                 className="min-h-[52px] flex-1 rounded-2xl border border-zinc-800 bg-zinc-100/5 py-4 text-xs font-bold uppercase tracking-[0.28em] text-zinc-200 shadow-xl shadow-black/20 transition-all hover:border-zinc-700 hover:bg-zinc-100/10 touch-manipulation md:min-h-[48px] md:py-5 md:text-[10px] md:tracking-[0.3em]"
               >
                 {language === 'zh' ? '镌刻记忆' : 'ETCH MEMORY'}
               </button>
            </div>
          </div>
        </div>,
        document.body,
        )}
    </div>
    {voiceBlockingMessage &&
      createPortal(
        <div className="fixed inset-0 z-[10001] flex items-center justify-center bg-black/75 p-6 backdrop-blur-md pointer-events-auto pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]">
          <div
            role="dialog"
            aria-modal="true"
            className="max-h-[min(70dvh,26rem)] w-full max-w-sm overflow-y-auto rounded-2xl border border-zinc-700/80 bg-zinc-950 px-5 py-6 shadow-2xl"
          >
            <p className="text-left text-[14px] leading-relaxed text-zinc-300">{voiceBlockingMessage}</p>
            <button
              type="button"
              className="mt-6 min-h-[44px] w-full rounded-xl border border-zinc-600 bg-zinc-900 py-3 text-sm font-medium text-zinc-100 touch-manipulation active:bg-zinc-800"
              onClick={() => setVoiceBlockingMessage(null)}
            >
              {language === 'zh' ? '确定' : 'OK'}
            </button>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
});

ChatOverlay.displayName = 'ChatOverlay';

export default ChatOverlay;
