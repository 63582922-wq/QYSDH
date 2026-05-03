/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 客户端可靠读取：Gemini Live 的 prebuilt voiceName（与 .env 中变量同名，需 VITE_ 前缀） */
  readonly VITE_GEMINI_LIVE_SPEECH_VOICE_NAME?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
