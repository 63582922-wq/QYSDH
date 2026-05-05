/**
 * Gemini Live：把麦克风 mono Float32 帧推到主线程做 PCM16 → base64 → sendRealtimeInput。
 * 放在 public/，以便 audioWorklet.addModule 用同源 URL 加载（避免 ScriptProcessor 弃用告警）。
 */
class LiveVoicePcmProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) return true;
    const copy = new Float32Array(ch0.length);
    copy.set(ch0);
    this.port.postMessage({ samples: copy }, [copy.buffer]);
    return true;
  }
}

registerProcessor('live-voice-pcm-processor', LiveVoicePcmProcessor);
