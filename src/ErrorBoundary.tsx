import React, { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

const ERROR_LOG_KEY = 'subconscious_last_render_error';

/** 防止单次渲染抛错导致整页黑屏 / 白屏（常见于移动端 WebView） */
export class ErrorBoundary extends Component<Props, State> {
  declare readonly props: Readonly<Props>;
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('App render error:', error, info.componentStack);
    try {
      localStorage.setItem(ERROR_LOG_KEY, JSON.stringify({
        message: error?.message ?? String(error),
        stack: error?.stack?.slice(0, 2000) ?? '',
        componentStack: info.componentStack?.slice(0, 2000) ?? '',
        ts: Date.now(),
      }));
    } catch { /* ignore */ }
  }

  render() {
    if (this.state.hasError) {
      const errMsg = this.state.error?.message ?? '';
      return (
        <div
          className="flex min-h-dvh flex-col items-center justify-center gap-6 bg-[#050505] px-6 text-center text-zinc-300"
          style={{ fontFamily: 'system-ui, sans-serif' }}
        >
          <p className="max-w-sm text-sm leading-relaxed text-zinc-400">
            页面渲染出错。请点击下方重试。
          </p>
          {errMsg ? (
            <p className="max-w-sm break-all text-[11px] leading-relaxed text-zinc-600">
              {errMsg.slice(0, 300)}
            </p>
          ) : null}
          <button
            type="button"
            className="rounded-full border border-zinc-700 bg-zinc-900 px-8 py-3 text-xs uppercase tracking-widest text-zinc-300 active:bg-zinc-800"
            onClick={() => window.location.reload()}
          >
            重新加载
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
