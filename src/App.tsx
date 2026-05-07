/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Settings2, X, Dna, Languages, FileAudio, Loader2, History, Mic } from 'lucide-react';
import ChatOverlay, { type ChatOverlayHandle, type ConversationMode } from './components/ChatOverlay';
import {
  ensureMinimaxVoicePersistenceLoaded,
  getAudioFileDurationSec,
  getMinimaxApiKey,
  MINIMAX_CLONE_MAX_DURATION_SEC,
  MINIMAX_CLONE_MIN_DURATION_SEC,
  MINIMAX_CLONED_VOICE_STORAGE_KEY,
  MINIMAX_VOICE_PROFILES_KEY,
  newClonedVoiceId,
  readStoredMinimaxClonedVoiceId,
  readVoiceProfiles,
  removeVoiceProfile,
  requestMinimaxVoiceClone,
  setActiveVoiceProfileId,
  uploadMinimaxVoiceCloneSource,
  upsertVoiceProfile,
  validateCloneAudioFile,
} from './minimaxVoiceClone';

/** 尽早从 session 镜像恢复 localStorage，避免首屏 state 读到空 */
ensureMinimaxVoicePersistenceLoaded();


// --- Particle Engine ---
class Particle {
  x: number;
  y: number;
  originX: number;
  originY: number;
  color: string;
  size: number;
  vx: number;
  vy: number;
  ease: number;
  friction: number;
  baseBrightness: number;
  baseAlpha: number;
  edgeFactor: number; // 0 for core, 1 for far edge
  drift: number;

  constructor(x: number, y: number, color: string, size: number, brightness: number, alpha: number, edgeFactor: number = 0) {
    // 内部紧致、外缘更涣散：边缘初始散开更大
    const spawnSpread = 28 + edgeFactor * 168;
    this.x = x + (Math.random() - 0.5) * spawnSpread;
    this.y = y + (Math.random() - 0.5) * spawnSpread;
    this.originX = x;
    this.originY = y;
    this.color = color;
    this.size = size;
    this.vx = 0;
    this.vy = 0;
    const coreEaseBoost = 1 + (1 - edgeFactor) * 0.45;
    const edgeLag = 1 - edgeFactor * 0.56;
    this.ease = (0.005 + Math.random() * 0.03) * edgeLag * coreEaseBoost;
    this.friction = 0.8 + Math.random() * 0.15; 
    this.baseBrightness = brightness;
    this.baseAlpha = alpha * (1 - edgeFactor * 0.9); // 外缘更虚，轮廓更易消散
    this.edgeFactor = edgeFactor;
    this.drift = Math.random() * Math.PI * 2;
  }

  update(
    pointer: { x: number; y: number; radiusSq: number; isActive: boolean }, 
    time: number, 
    particleSize: number,
    rotation3D: number,
    audioData?: { force: number; bass: number },
    canvasCenter?: { x: number; y: number }
  ) {
    this.size = particleSize * (1 + this.edgeFactor * 0.72); // 外缘略放大，更易呈雾化涣散

    const audioScale = audioData ? audioData.bass / 255 : 0;
    const audioTremor = audioData ? (audioData.force / 255) * 10 : 0;

    // Apply speech/text-typing vibration specifically to edges
    let speechVibeX = 0;
    let speechVibeY = 0;
    if (audioData && this.edgeFactor > 0.45) {
       // High intensity vibration localized on edges
       const intensity = (this.edgeFactor - 0.45) * 4 * audioData.force;
       speechVibeX = (Math.random() - 0.5) * intensity;
       speechVibeY = (Math.random() - 0.5) * intensity;
    }

    // 3D rotation with perspective (around vertical axis)
    let rotatedX = this.originX + speechVibeX;
    let rotatedY = this.originY + speechVibeY;
    if (canvasCenter) {
      const dx = this.originX - canvasCenter.x;
      const dy = this.originY - canvasCenter.y;
      const dz = 0;
      const cosY = Math.cos(rotation3D);
      const sinY = Math.sin(rotation3D);
      const x1 = dx * cosY + dz * sinY;
      const z1 = -dx * sinY + dz * cosY;
      const perspective = 950;
      const scale = perspective / (perspective + z1);
      rotatedX = canvasCenter.x + x1 * scale;
      rotatedY = canvasCenter.y + dy * scale;
    }

    // Breathing：位移「呼吸」略加强，整体更有生命感
    const breathMul = 0.36 + this.edgeFactor * 1.22;
    const breathX =
      Math.sin(time * 0.00085 + this.originY * 0.01) * (1 + audioTremor) * breathMul;
    const breathY =
      Math.cos(time * 0.00085 + this.originX * 0.01) * (1 + audioTremor) * breathMul;

    // 散射：中高 edge 显著加大 + 二次项，外缘更涣散
    const ef = this.edgeFactor;
    const scatterIntensity =
      (0.35 + ef * 11.5 + ef * ef * 5.8) * (1 + audioScale * 1.65);
    const scatterX = Math.sin(time * 0.001 + this.drift + this.originX * 0.005) * scatterIntensity;
    const scatterY = Math.cos(time * 0.001 + this.drift + this.originY * 0.005) * scatterIntensity;

    // 外缘切向飘移，减弱「硬边」感
    let tangentX = 0;
    let tangentY = 0;
    if (canvasCenter && ef > 0.25) {
      const rdx = this.originX - canvasCenter.x;
      const rdy = this.originY - canvasCenter.y;
      const rd = Math.sqrt(rdx * rdx + rdy * rdy) || 1;
      const swirl = (ef - 0.25) * 4.2 * Math.sin(time * 0.0011 + this.drift * 2.1);
      tangentX = (-rdy / rd) * swirl;
      tangentY = (rdx / rd) * swirl;
    }

    let targetX = rotatedX + breathX + scatterX + tangentX;
    let targetY = rotatedY + breathY + scatterY + tangentY;
    let pulseSize = audioScale * particleSize * 3;

    // 略向画面主体中心拉拢 interior，增强中间致密感（边缘仍保持外扩）
    if (canvasCenter && this.edgeFactor < 0.4) {
      const inward = (0.4 - this.edgeFactor) / 0.4;
      const dx = canvasCenter.x - this.originX;
      const dy = canvasCenter.y - this.originY;
      targetX += dx * inward * 0.014;
      targetY += dy * inward * 0.014;
    }

    // Audio explosion effect for loud bass
    if (audioData && canvasCenter && audioData.bass > 160) {
       const explosionForce = (audioData.bass - 160) / 95; // 0 to 1
       const dx = this.originX - canvasCenter.x;
       const dy = this.originY - canvasCenter.y;
       const dist = Math.sqrt(dx*dx + dy*dy) || 1;
       
       // push target outwards
       targetX += (dx / dist) * explosionForce * 40;
       targetY += (dy / dist) * explosionForce * 40;
    }

    if (pointer.isActive) {
      const dx = targetX - pointer.x;
      const dy = targetY - pointer.y;
      const distanceSq = dx * dx + dy * dy;

      if (distanceSq < pointer.radiusSq) {
        const radius = Math.sqrt(pointer.radiusSq);
        const distance = Math.sqrt(distanceSq);
        
        // Use a smoothstep function for a fluid, soft bulge effect
        const falloff = Math.max(0, 1 - distance / radius);
        const smoothFactor = falloff * falloff * (3 - 2 * falloff);
        
        // Push outward gently
        const shiftAmount = smoothFactor * 0.6; 
        
        targetX += dx * shiftAmount;
        targetY += dy * shiftAmount;

        // Subtle and smooth size increase
        pulseSize += smoothFactor * (particleSize * 2.2);
      }
    }

    // Edge particles get a persistent outward pull,
    // similar to the pointer repulsion feeling.
    if (canvasCenter && this.edgeFactor > 0.18) {
      const outDx = targetX - canvasCenter.x;
      const outDy = targetY - canvasCenter.y;
      const outDist = Math.sqrt(outDx * outDx + outDy * outDy) || 1;
      const efOut = this.edgeFactor;
      const outwardStrength = 0.06 + efOut * 0.28 + efOut * efOut * 0.22;
      targetX += (outDx / outDist) * outwardStrength;
      targetY += (outDy / outDist) * outwardStrength;
    }

    this.size += pulseSize;

    // Integrate physics
    this.x += (this.vx *= this.friction) + (targetX - this.x) * this.ease;
    this.y += (this.vy *= this.friction) + (targetY - this.y) * this.ease;
  }

  draw(ctx: CanvasRenderingContext2D, colorMode: string, customHex: string, time: number) {
    if (colorMode === 'original') {
      ctx.fillStyle = this.color;
    } else if (colorMode === 'ghost') {
      ctx.fillStyle = `rgba(255, 255, 255, ${this.baseAlpha})`;
    } else if (colorMode === 'blues') {
      ctx.fillStyle = `hsla(210, 100%, ${(this.baseBrightness / 255) * 100}%, ${this.baseAlpha})`;
    } else if (colorMode === 'greens') {
      ctx.fillStyle = `hsla(150, 100%, ${(this.baseBrightness / 255) * 100}%, ${this.baseAlpha})`;
    } else if (colorMode === 'purples') {
      ctx.fillStyle = `hsla(270, 100%, ${(this.baseBrightness / 255) * 100}%, ${this.baseAlpha})`;
    } else if (colorMode === 'custom') {
      ctx.fillStyle = customHex;
    } else {
      ctx.fillStyle = this.color;
    }
    // 半径随时间略胀缩（每粒相位不同），叠加位移呼吸更明显
    const sizeBreath =
      1 +
      0.1 * Math.sin(time * 0.0014 + this.drift + this.originX * 0.012 + this.originY * 0.011) +
      0.04 * this.edgeFactor * Math.sin(time * 0.0011 + this.drift * 2);
    const r = Math.max(0.35, this.size * 0.5 * sizeBreath);
    const cx = this.x + r;
    const cy = this.y + r;
    // 方形微粒（非圆点），略旋转打破网格感
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(this.drift + time * 0.00022 * (0.55 + this.edgeFactor * 0.35));
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.restore();
  }
}

/** 四边外缘粒子索引：用于与画布边界拉线形成网状感 */
type EdgeMeshBuckets = { left: number[]; right: number[]; top: number[]; bottom: number[] };

function emptyEdgeMesh(): EdgeMeshBuckets {
  return { left: [], right: [], top: [], bottom: [] };
}

function subsampleSortedIndices(indices: number[], max: number): number[] {
  if (indices.length <= max) return indices;
  const step = Math.ceil(indices.length / max);
  const out: number[] = [];
  for (let u = 0; u < indices.length; u += step) out.push(indices[u]);
  return out;
}

function rebuildEdgeMeshBuckets(particles: Particle[], cw: number, ch: number): EdgeMeshBuckets {
  const buckets = emptyEdgeMesh();
  if (!particles.length || cw < 80 || ch < 80) return buckets;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    if (p.originX < minX) minX = p.originX;
    if (p.originX > maxX) maxX = p.originX;
    if (p.originY < minY) minY = p.originY;
    if (p.originY > maxY) maxY = p.originY;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const leftTh = minX + spanX * 0.33;
  const rightTh = maxX - spanX * 0.33;
  const topTh = minY + spanY * 0.28;
  const botTh = maxY - spanY * 0.28;
  const eMin = 0.34;

  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    if (p.edgeFactor < eMin) continue;
    if (p.originX <= leftTh) buckets.left.push(i);
    if (p.originX >= rightTh) buckets.right.push(i);
    if (p.originY <= topTh) buckets.top.push(i);
    if (p.originY >= botTh) buckets.bottom.push(i);
  }

  buckets.left.sort((a, b) => particles[a].originY - particles[b].originY);
  buckets.right.sort((a, b) => particles[a].originY - particles[b].originY);
  buckets.top.sort((a, b) => particles[a].originX - particles[b].originX);
  buckets.bottom.sort((a, b) => particles[a].originX - particles[b].originX);

  const cap = 96;
  buckets.left = subsampleSortedIndices(buckets.left, cap);
  buckets.right = subsampleSortedIndices(buckets.right, cap);
  buckets.top = subsampleSortedIndices(buckets.top, cap);
  buckets.bottom = subsampleSortedIndices(buckets.bottom, cap);

  return buckets;
}

function drawEdgeMesh(
  ctx: CanvasRenderingContext2D,
  particles: Particle[],
  mesh: EdgeMeshBuckets,
  cw: number,
  ch: number,
  time: number,
) {
  if (!particles.length) return;

  // 明显一些的「呼吸灯」：透明度 + 线宽随 time 起伏（原先 sin 系数小，几乎看不出）
  const w = 0.5 + 0.5 * Math.sin(time * 0.002);
  const chainA = 0.085 + 0.1 * w;
  const spokeA = 0.05 + 0.09 * w;
  const crossA = 0.03 + 0.08 * w;
  const lwMain = 0.68 + 0.38 * w;
  const lwCross = 0.52 + 0.32 * w;
  const lwSpoke = 0.6 + 0.36 * w;

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const strokeChain = (indices: number[]) => {
    if (indices.length < 2) return;
    ctx.strokeStyle = `rgba(210, 206, 228, ${chainA})`;
    ctx.lineWidth = lwMain;
    for (let k = 0; k < indices.length - 1; k++) {
      const pa = particles[indices[k]];
      const pb = particles[indices[k + 1]];
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
    ctx.lineWidth = lwCross;
    ctx.strokeStyle = `rgba(188, 184, 208, ${crossA})`;
    for (let k = 0; k < indices.length - 2; k++) {
      const pa = particles[indices[k]];
      const pb = particles[indices[k + 2]];
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
  };

  strokeChain(mesh.left);
  strokeChain(mesh.right);
  strokeChain(mesh.top);
  strokeChain(mesh.bottom);

  ctx.lineWidth = lwSpoke;
  ctx.strokeStyle = `rgba(200, 196, 220, ${spokeA})`;

  for (let k = 0; k < mesh.left.length; k += 2) {
    const p = particles[mesh.left[k]];
    ctx.beginPath();
    ctx.moveTo(0, p.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }
  for (let k = 0; k < mesh.right.length; k += 2) {
    const p = particles[mesh.right[k]];
    ctx.beginPath();
    ctx.moveTo(cw - 1, p.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }
  for (let k = 0; k < mesh.top.length; k += 2) {
    const p = particles[mesh.top[k]];
    ctx.beginPath();
    ctx.moveTo(p.x, 0);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }
  for (let k = 0; k < mesh.bottom.length; k += 2) {
    const p = particles[mesh.bottom[k]];
    ctx.beginPath();
    ctx.moveTo(p.x, ch - 1);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }

  ctx.restore();
}

// --- App Component ---
export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const modelCenterRef = useRef<{ x: number; y: number } | null>(null);
  const chatOverlayRef = useRef<ChatOverlayHandle | null>(null);
  const edgeMeshIndexRef = useRef<EdgeMeshBuckets>(emptyEdgeMesh());

  // App State
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [isHoveringSettings, setIsHoveringSettings] = useState(false);
  const [language, setLanguage] = useState<'zh' | 'en'>('zh');
  const [isAudioReactive, setIsAudioReactive] = useState(false);
  const [isAutoSpeak, setIsAutoSpeak] = useState(() => localStorage.getItem('subconscious_auto_speak') !== 'false');

  /** 上传图像并选定：Live 语音 ↔ 文字+复刻音色，两条链路互斥 */
  const [conversationMode, setConversationMode] = useState<ConversationMode | null>(null);
  const [voiceProfileTick, setVoiceProfileTick] = useState(0);

  const cloneAudioInputRef = useRef<HTMLInputElement>(null);
  const cloneJobAbortRef = useRef<AbortController | null>(null);
  const [clonedMinimaxVoiceId, setClonedMinimaxVoiceId] = useState<string | null>(() => readStoredMinimaxClonedVoiceId());
  const [cloneVoiceBusy, setCloneVoiceBusy] = useState(false);
  const [cloneVoiceHint, setCloneVoiceHint] = useState('');

  useEffect(() => {
    return () => {
      cloneJobAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (
        e.key === null ||
        e.key === MINIMAX_CLONED_VOICE_STORAGE_KEY ||
        e.key === MINIMAX_VOICE_PROFILES_KEY
      ) {
        setClonedMinimaxVoiceId(readStoredMinimaxClonedVoiceId());
        setVoiceProfileTick((n) => n + 1);
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  /** 与 localStorage 一致；若仅有「当前选中 id」也会并入列表，避免界面空白 */
  const voiceListForUi = useMemo(() => {
    const base = readVoiceProfiles();
    const active = clonedMinimaxVoiceId?.trim();
    if (active && !base.some((p) => p.voiceId === active)) {
      return [{ voiceId: active, createdAt: Date.now() }, ...base];
    }
    return base;
  }, [voiceProfileTick, clonedMinimaxVoiceId]);

  const clearClonedMinimaxVoice = useCallback(() => {
    cloneJobAbortRef.current?.abort();
    cloneJobAbortRef.current = null;
    setActiveVoiceProfileId(null);
    setClonedMinimaxVoiceId(null);
    setVoiceProfileTick((n) => n + 1);
    setCloneVoiceHint(language === 'zh' ? '已改用内置默认音色。' : 'Using the built-in default voice.');
  }, [language]);

  const handleCloneAudioSelected = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file) return;
      if (!getMinimaxApiKey()) {
        setCloneVoiceHint(
          language === 'zh' ? '当前无法使用声音复刻，请稍后再试或联系管理员。' : 'Voice cloning is unavailable right now.',
        );
        return;
      }

      const bad = validateCloneAudioFile(file, language);
      if (bad) {
        setCloneVoiceHint(bad);
        return;
      }

      cloneJobAbortRef.current?.abort();
      const ac = new AbortController();
      cloneJobAbortRef.current = ac;

      void (async () => {
        setCloneVoiceBusy(true);
        setCloneVoiceHint(language === 'zh' ? '校验时长…' : 'Checking duration…');
        try {
          const dur = await getAudioFileDurationSec(file);
          if (dur < MINIMAX_CLONE_MIN_DURATION_SEC) {
            setCloneVoiceHint(
              language === 'zh'
                ? `录音至少 ${MINIMAX_CLONE_MIN_DURATION_SEC} 秒（当前约 ${Math.round(dur)} 秒）。`
                : `Need at least ${MINIMAX_CLONE_MIN_DURATION_SEC}s (about ${Math.round(dur)}s now).`,
            );
            return;
          }
          if (dur > MINIMAX_CLONE_MAX_DURATION_SEC) {
            setCloneVoiceHint(
              language === 'zh'
                ? `录音最长 ${MINIMAX_CLONE_MAX_DURATION_SEC / 60} 分钟。`
                : `Max length is ${MINIMAX_CLONE_MAX_DURATION_SEC / 60} minutes.`,
            );
            return;
          }
          if (ac.signal.aborted) return;

          setCloneVoiceHint(language === 'zh' ? '上传中…' : 'Uploading…');
          const fileId = await uploadMinimaxVoiceCloneSource(file, ac.signal);
          if (ac.signal.aborted) return;

          setCloneVoiceHint(language === 'zh' ? '复刻中…' : 'Cloning…');
          const voiceId = newClonedVoiceId();
          await requestMinimaxVoiceClone({
            sourceFileId: fileId,
            voiceId,
            signal: ac.signal,
          });
          if (ac.signal.aborted) return;

          upsertVoiceProfile(voiceId);
          setClonedMinimaxVoiceId(voiceId);
          setVoiceProfileTick((n) => n + 1);
          setCloneVoiceHint(
            language === 'zh'
              ? '复刻成功，已保存。开启「朗读回复」后，将用你的音色播报回复。'
              : 'Voice saved. Turn on read responses to hear replies in your voice.',
          );
        } catch (err) {
          if (ac.signal.aborted) return;
          console.warn('Voice clone failed', err);
          setCloneVoiceHint(
            language === 'zh' ? '复刻失败，请检查录音与网络后重试。' : 'Clone failed. Check your audio and network.',
          );
        } finally {
          if (cloneJobAbortRef.current === ac) cloneJobAbortRef.current = null;
          setCloneVoiceBusy(false);
        }
      })();
    },
    [language],
  );
  
  // Audio Config
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioDataRef = useRef<Uint8Array | null>(null);
  
  // Particle Config
  const [particleSize, setParticleSize] = useState<number>(2);
  const [particleGap, setParticleGap] = useState<number>(4);
  const [colorMode, setColorMode] = useState<string>('original'); // 'original', 'ghost', 'blues', 'greens', 'purples', 'custom'
  const [customColor, setCustomColor] = useState<string>('#4ade80'); 
  const [threshold, setThreshold] = useState<number>(240); // To filter out bright backgrounds
  const [aiName, setAiName] = useState(() => localStorage.getItem('subconscious_ai_name') || '');
  const speechIntensityRef = useRef(0);
  const handleSpeechValue = useCallback((val: number) => {
    speechIntensityRef.current = val;
  }, []);

  // Interaction State (Mutated raw for performance in RAF loop)
  const pointerRef = useRef({ x: -1000, y: -1000, radiusSq: 20000, isActive: false });

  // 1. Process uploaded image and extract pixels
  const parseImagePixels = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Responsive scaling: 横向图尽量用满画布；竖图保留适中留白
    const isLandscape = img.width > img.height;
    const padW = isLandscape ? 0.02 : 0.15;
    const padH = isLandscape ? 0.055 : 0.15; // 横向时底栏区域略留一点
    const PADDING_W = canvas.width * padW;
    const PADDING_H = canvas.height * padH;
    const maxWidth = canvas.width - PADDING_W * 2;
    const maxHeight = canvas.height - PADDING_H * 2;

    const scaleCap = isLandscape ? 1 : 0.95;
    const scale = Math.min(maxWidth / img.width, maxHeight / img.height, scaleCap);
    const drawWidth = Math.floor(img.width * scale);
    const drawHeight = Math.floor(img.height * scale);
    const offsetX = Math.floor((canvas.width - drawWidth) / 2);
    const offsetY = Math.floor((canvas.height - drawHeight) / 2);
    const centerX = offsetX + drawWidth / 2;
    const centerY = offsetY + drawHeight / 2;
    modelCenterRef.current = { x: centerX, y: centerY };

    // Use an offscreen canvas to get unadulterated pixel data
    const offscreen = document.createElement('canvas');
    offscreen.width = drawWidth;
    offscreen.height = drawHeight;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    offCtx.drawImage(img, 0, 0, drawWidth, drawHeight);
    const imgData = offCtx.getImageData(0, 0, drawWidth, drawHeight).data;

    const particles: Particle[] = [];
    
    // Step through pixels based on gap.
    for (let y = 0; y < drawHeight; y += particleGap) {
      for (let x = 0; x < drawWidth; x += particleGap) {
        const index = (y * drawWidth + x) * 4;
        const r = imgData[index];
        const g = imgData[index + 1];
        const b = imgData[index + 2];
        const a = imgData[index + 3];

        const brightness = (r + g + b) / 3;
        
        // 1. Soft Threshold for Shape Outline
        const brightnessDiff = threshold - brightness;
        const softThresholdFactor = Math.max(0, Math.min(1, brightnessDiff / 30)); 
        
        if (softThresholdFactor <= 0 || a <= 10) continue;

        let isIncluded = true;
        let edgeFactor = 1 - softThresholdFactor;

        // --- Irregular soft edges for the 4 rectangular sides ---
        const borderDistX = Math.min(x, drawWidth - x);
        const borderDistY = Math.min(y, drawHeight - y);
        const minBorderDist = Math.min(borderDistX, borderDistY);
        // Build stable pseudo-noise from coordinates to avoid regular borders
        const noise =
          Math.sin(x * 0.045) * 0.5 +
          Math.cos(y * 0.038) * 0.35 +
          Math.sin((x + y) * 0.021) * 0.25;
        const sideBias = Math.abs(borderDistX - borderDistY) < 18 ? 1.2 : 1;
        // Build chunky organic bumps to make contour feel neuron-like.
        const lobeNoise =
          Math.sin(x * 0.009) * 14 +
          Math.cos(y * 0.011) * 12 +
          Math.sin((x + y) * 0.006) * 16 +
          Math.cos((x - y) * 0.007) * 10;
        const microNoise =
          Math.sin(x * 0.041 + y * 0.013) * 5 +
          Math.cos(y * 0.037 - x * 0.015) * 4;
        const borderFadeRange = (52 + noise * 24 + lobeNoise * 1.2 + microNoise * 1.1) * sideBias;

        if (minBorderDist < borderFadeRange) {
          const borderFade = Math.max(0, Math.min(1, minBorderDist / borderFadeRange));
          edgeFactor = Math.max(edgeFactor, 1 - borderFade);
          // Irregular dropout probability to break straight edge feeling
          const dropoutJitter =
            0.68 +
            ((noise + 1) * 0.5) * 0.4 +
            Math.max(0, Math.sin((x + y) * 0.028)) * 0.28;
          if (Math.random() > borderFade * dropoutJitter * 0.88) isIncluded = false;
        }
        // ----------------------------------------------------

        if (isIncluded) {
          // 3. Random Dropout at Edges for "Natural Decay"
          if (edgeFactor > 0.3) {
            if (Math.random() < edgeFactor * 0.32) isIncluded = false;
          }
        }

        if (isIncluded) {
          // 4. Stochastic Sampling (Jitter)：核心区抖动更小，网格更「密实」
          const jitterScale = 0.2 + edgeFactor * 1.72;
          const jitterX = (Math.random() - 0.5) * particleGap * 1.5 * jitterScale;
          const jitterY = (Math.random() - 0.5) * particleGap * 1.5 * jitterScale;
          
          const finalX = x + offsetX + jitterX;
          const finalY = y + offsetY + jitterY;
          const color = `rgba(${r},${g},${b},${a / 255})`;
          
          particles.push(new Particle(finalX, finalY, color, particleSize, brightness, a / 255, edgeFactor));

          // Keep contour clean; do not spawn extra halo dust particles.
        }
      }
    }

    particlesRef.current = particles;
    const cw = canvas?.width ?? 0;
    const ch = canvas?.height ?? 0;
    edgeMeshIndexRef.current = rebuildEdgeMeshBuckets(particles, cw, ch);
  }, [particleGap, particleSize, threshold]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;

      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      // Re-parse purely to recenter/re-scale image if it exists
      if (imageRef.current) {
        parseImagePixels();
      }
    };

    handleResize(); // Initial sizing
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [parseImagePixels]);

  // Handle Audio Context Lifecycle
  useEffect(() => {
    let stream: MediaStream | null = null;
    
    if (isAudioReactive) {
      navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => {
        stream = s;
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        analyserRef.current = analyser;
        audioDataRef.current = new Uint8Array(analyser.frequencyBinCount);
      }).catch(e => {
        console.error("Audio error", e);
        setIsAudioReactive(false);
      });
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
      analyserRef.current = null;
      audioDataRef.current = null;
    };
  }, [isAudioReactive]);

  // Main Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    const render = () => {
      // 拖尾略淡一点，粒子明暗/半径呼吸更容易被看见
      ctx.fillStyle = 'rgba(5, 5, 5, 0.34)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Create a radial gradient vignette to hide the canvas edges
      const gradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) * 0.7
      );
      gradient.addColorStop(0, 'rgba(5, 5, 5, 0)');
      gradient.addColorStop(0.6, 'rgba(5, 5, 5, 0)');
      gradient.addColorStop(1, 'rgba(5, 5, 5, 1)');
      
      let audioForce = 0;
      let bass = 0;
      if (isAudioReactive && analyserRef.current && audioDataRef.current) {
         analyserRef.current.getByteFrequencyData(audioDataRef.current);
         let sum = 0;
         let bassSum = 0;
         for (let i = 0; i < audioDataRef.current.length; i++) {
             sum += audioDataRef.current[i];
             if (i < 10) bassSum += audioDataRef.current[i];
         }
         audioForce = sum / audioDataRef.current.length;
         bass = bassSum / 10;
      }
      
      const canvasCenter = { x: canvas.width / 2, y: canvas.height / 2 };
      const rotationCenter = modelCenterRef.current ?? canvasCenter;
      // Left-right yaw around image center.
      // Slightly faster: one full sway cycle per 45 seconds.
      const rotation3D = Math.sin((performance.now() * 2 * Math.PI) / 45000) * (Math.PI / 10);
      // 整层粒子明暗「呼吸」（略加大摆幅与频率，拖尾下仍可见）
      const ambientBreath = 0.8 + 0.2 * Math.sin(time * 0.00195);
      ctx.globalAlpha = ambientBreath;
      for (let i = 0; i < particlesRef.current.length; i++) {
        const p = particlesRef.current[i];
        p.update(
          pointerRef.current,
          time,
          particleSize,
          rotation3D,
          { force: Math.max(audioForce, speechIntensityRef.current * 80), bass: Math.max(bass, speechIntensityRef.current * 200) },
          rotationCenter,
        );
        p.draw(ctx, colorMode, customColor, time);
      }
      ctx.globalAlpha = 1;

      drawEdgeMesh(ctx, particlesRef.current, edgeMeshIndexRef.current, canvas.width, canvas.height, time);

      // Apply the vignette after drawing particles to soften the edges
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      time += 16;
      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationRef.current);
  }, [particleSize, colorMode, customColor, isAudioReactive]);

  // Input Handling
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      setImageSrc(result);
      setConversationMode(null);
      
      const img = new Image();
      img.src = result;
      img.onload = () => {
        imageRef.current = img;
        parseImagePixels();
      };
    };
    reader.readAsDataURL(file);
  };

  const handleClearImage = () => {
    setImageSrc(null);
    setConversationMode(null);
    imageRef.current = null;
    particlesRef.current = [];
    edgeMeshIndexRef.current = emptyEdgeMesh();
  };

  const handleDissolveReset = () => {
    uploadInputRef.current?.click();
  };

  // Interaction Events Setup
  const updatePointer = (clientX: number, clientY: number, isActive: boolean) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    pointerRef.current.x = clientX - rect.left;
    pointerRef.current.y = clientY - rect.top;
    pointerRef.current.isActive = isActive;
  };

  const onPointerMove = (e: React.PointerEvent) => {
    updatePointer(e.clientX, e.clientY, true);
  };
  const onPointerLeave = () => {
    pointerRef.current.isActive = false;
  };
  const onPointerDown = (e: React.PointerEvent) => {
    pointerRef.current.radiusSq = 40000; 
    updatePointer(e.clientX, e.clientY, true);
  };
  const onPointerUp = () => {
    pointerRef.current.radiusSq = 20000; 
    pointerRef.current.isActive = false;
  };

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-0 min-h-dvh h-dvh w-full bg-[#050505] text-zinc-300 overflow-hidden font-sans select-none touch-auto"
    >
      {/*
        层级（移动端优先）：
        z-0  画布交互
        z-10 首次上传引导；z-42 首页语言键叠在其上
        z-30 对话层（大面积 pointer-events-none，见 ChatOverlay）
        z-42 顶栏语言/设置（始终可点）；消散重置 / 保存对话在对话条右侧
        z-49 设置抽屉遮罩（仅窄屏）
        z-50 设置面板
      */}
      {/* Canvas Layer */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-0 h-full w-full cursor-crosshair touch-none"
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
      />

      {/* Shared uploader input (used by initial upload and reset) */}
      <input
        ref={uploadInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleImageUpload}
      />

      {/* Upload Screen Overlay (Hidden once an image is active) */}
      {!imageSrc && (
        <>
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-6 pt-[calc(env(safe-area-inset-top)+1.5rem)] pb-[calc(env(safe-area-inset-bottom)+1.5rem)] bg-[#050505]/80 backdrop-blur-sm pointer-events-none transition-opacity">
            <div className="flex w-full max-w-md flex-col items-center gap-6 opacity-70 pointer-events-auto md:gap-8">
              <Dna strokeWidth={1} className="h-14 w-14 text-zinc-500 animate-pulse md:h-16 md:w-16" />
              <div className="space-y-3 text-center md:space-y-5">
                <h1 className="text-xl font-light tracking-[0.28em] text-zinc-200 md:text-2xl md:tracking-widest">
                  {language === 'zh' ? '潜意识形态' : 'SUBCONSCIOUS FORMS'}
                </h1>
                <p className="text-[13px] font-normal leading-relaxed text-zinc-500 md:text-sm md:font-medium md:uppercase md:tracking-wide">
                  {language === 'zh' ? '上传一段记忆以显化粒子。' : 'Upload a memory to manifest particles.'}
                  <br />
                  {language === 'zh' ? '深色背景图像会产生最佳的氛围。' : 'Dark background images yield the best aura.'}
                </p>
              </div>
              <button
                type="button"
                onClick={() => uploadInputRef.current?.click()}
                className="group relative min-h-[44px] overflow-hidden rounded-full border border-zinc-800 bg-zinc-900/50 px-8 py-2.5 text-xs font-medium tracking-[0.22em] text-zinc-400 transition-all hover:bg-zinc-800 hover:text-zinc-200 md:min-h-[48px] md:px-10 md:py-3 md:text-sm md:tracking-widest touch-manipulation active:scale-[0.98]"
              >
                <span>{language === 'zh' ? '选择图像' : 'SELECT IMAGE'}</span>
              </button>
            </div>
          </div>
          {/* 首页顶栏：中英文切换（与有图时同款） */}
          <div className="fixed right-[max(0.5rem,env(safe-area-inset-right))] top-[max(0.5rem,env(safe-area-inset-top))] z-[42] flex items-center gap-1 pointer-events-auto touch-manipulation md:gap-2">
            <button
              type="button"
              onClick={() => setLanguage((l) => (l === 'zh' ? 'en' : 'zh'))}
              className="flex min-h-[40px] min-w-[40px] items-center justify-center gap-1.5 rounded-lg px-2 text-zinc-500 opacity-90 transition-colors hover:text-zinc-200 active:bg-zinc-900/80 md:min-h-[44px] md:min-w-[44px] md:gap-2 md:rounded-xl md:px-3"
              aria-label={language === 'zh' ? '切换到英文' : 'Switch to Chinese'}
            >
              <Languages className="h-[18px] w-[18px] md:h-5 md:w-5" strokeWidth={1.5} />
              <span className="text-[11px] font-medium uppercase tracking-wider md:text-xs md:tracking-widest">{language}</span>
            </button>
          </div>
        </>
      )}

      {/* z-30 对话 UI — 低于顶栏，避免挡住按钮 */}
      <ChatOverlay
        ref={chatOverlayRef}
        currentImageDataUrl={imageSrc}
        isOpen={!!imageSrc && conversationMode !== null}
        onClose={() => {}}
        language={language}
        onToggleLanguage={() => setLanguage((l) => (l === 'zh' ? 'en' : 'zh'))}
        aiName={aiName}
        onSpeechValue={handleSpeechValue}
        isAutoSpeak={isAutoSpeak}
        setIsAutoSpeak={setIsAutoSpeak}
        conversationMode={(conversationMode ?? 'live') as ConversationMode}
        settingsChromeOpen={showSettings}
        onDissolveReset={handleDissolveReset}
        onOpenSavePreview={() => chatOverlayRef.current?.openSavePreview()}
      />

      {imageSrc && conversationMode === null && (
        <div className="fixed inset-0 z-[44] flex flex-col items-center justify-center bg-[#050505]/88 px-6 backdrop-blur-md pointer-events-auto pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]">
          <p className="mb-8 max-w-sm text-center text-[13px] leading-relaxed text-zinc-400 md:text-sm">
            {language === 'zh' ? '选择对话方式。' : 'Choose a mode.'}
          </p>
          <div className="flex w-full max-w-md flex-col gap-4 md:flex-row md:gap-5">
            <button
              type="button"
              onClick={() => setConversationMode('live')}
              className="flex min-h-[52px] flex-1 flex-col items-center justify-center gap-2 rounded-2xl border border-zinc-700 bg-zinc-900/70 px-6 py-4 text-zinc-200 shadow-lg transition-colors hover:border-zinc-500 hover:bg-zinc-800/80 touch-manipulation active:scale-[0.99]"
            >
              <Mic className="h-6 w-6 text-zinc-400" strokeWidth={1.5} />
              <span className="text-[11px] font-medium uppercase tracking-[0.22em]">
                {language === 'zh' ? 'LIVE 模式' : 'Live mode'}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setConversationMode('text_clone')}
              className="flex min-h-[52px] flex-1 flex-col items-center justify-center gap-2 rounded-2xl border border-zinc-700 bg-zinc-900/70 px-6 py-4 text-zinc-200 shadow-lg transition-colors hover:border-zinc-500 hover:bg-zinc-800/80 touch-manipulation active:scale-[0.99]"
            >
              <FileAudio className="h-6 w-6 text-zinc-400" strokeWidth={1.5} />
              <span className="text-[11px] font-medium uppercase tracking-[0.22em]">
                {language === 'zh' ? '声音克隆' : 'Voice clone'}
              </span>
            </button>
          </div>
        </div>
      )}

      {imageSrc && (
        <>
          {/* z-42 顶栏：历史 + 语言 + 设置 */}
          <div className="fixed right-[max(0.5rem,env(safe-area-inset-right))] top-[max(0.5rem,env(safe-area-inset-top))] z-[42] flex items-center gap-1 pointer-events-auto touch-manipulation md:gap-2">
            {conversationMode !== null && (
              <button
                type="button"
                onClick={() => chatOverlayRef.current?.openHistory()}
                className="flex min-h-[40px] min-w-[40px] items-center justify-center rounded-lg px-2 text-zinc-500 opacity-90 transition-colors hover:text-zinc-200 active:bg-zinc-900/80 md:min-h-[44px] md:min-w-[44px] md:rounded-xl md:px-3"
                title={language === 'zh' ? '历史对话' : 'Chat history'}
                aria-label={language === 'zh' ? '历史对话' : 'Chat history'}
              >
                <History className="h-[18px] w-[18px] md:h-5 md:w-5" strokeWidth={1.5} />
              </button>
            )}
            <button
              type="button"
              onClick={() => setLanguage((l) => (l === 'zh' ? 'en' : 'zh'))}
              className="flex min-h-[40px] min-w-[40px] items-center justify-center gap-1.5 rounded-lg px-2 text-zinc-500 opacity-90 transition-colors hover:text-zinc-200 active:bg-zinc-900/80 md:min-h-[44px] md:min-w-[44px] md:gap-2 md:rounded-xl md:px-3"
              aria-label={language === 'zh' ? '切换到英文' : 'Switch to Chinese'}
            >
              <Languages className="h-[18px] w-[18px] md:h-5 md:w-5" strokeWidth={1.5} />
              <span className="text-[11px] font-medium uppercase tracking-wider md:text-xs md:tracking-widest">{language}</span>
            </button>

            <button
              type="button"
              onClick={() => setShowSettings((v) => !v)}
              className={`flex min-h-[40px] min-w-[40px] items-center justify-center rounded-lg border transition-all duration-300 md:min-h-[44px] md:min-w-[44px] md:rounded-xl ${
                showSettings || isHoveringSettings
                  ? 'text-zinc-200 bg-zinc-900 border-zinc-700 shadow-lg'
                  : 'text-zinc-500 border-transparent opacity-90 active:bg-zinc-900/80'
              }`}
              onMouseEnter={() => setIsHoveringSettings(true)}
              onMouseLeave={() => setIsHoveringSettings(false)}
              title={language === 'zh' ? '设置' : 'Settings'}
              aria-expanded={showSettings}
            >
              {showSettings ? <X strokeWidth={1.5} className="h-[18px] w-[18px] md:h-[22px] md:w-[22px]" /> : <Settings2 strokeWidth={1.5} className="h-[18px] w-[18px] md:h-[22px] md:w-[22px]" />}
            </button>
          </div>

          {/* 窄屏设置遮罩 */}
          {showSettings && (
            <button
              type="button"
              aria-label={language === 'zh' ? '关闭设置' : 'Close settings'}
              className="fixed inset-0 z-[99] bg-black/55 backdrop-blur-[2px] md:hidden touch-manipulation"
              onClick={() => setShowSettings(false)}
            />
          )}

          {/* 设置：手机底部抽屉 / md+ 右上角 */}
          <div
            className={`fixed z-[100] bg-[#0a0a0a]/95 backdrop-blur-xl border-zinc-900 shadow-2xl transition-all duration-300 overflow-y-auto overscroll-contain touch-manipulation
              max-md:inset-x-0 max-md:bottom-0 max-md:top-auto max-md:rounded-t-[1.75rem] max-md:border-t max-md:border-x-0 max-md:border-b-0 max-md:max-h-[min(88dvh,100svh)] max-md:pb-[calc(env(safe-area-inset-bottom)+0.75rem)] max-md:pt-4 max-md:px-5
              md:absolute md:left-auto md:right-[max(1rem,env(safe-area-inset-right))] md:top-[calc(env(safe-area-inset-top)+3.75rem)] md:bottom-auto md:w-[min(18rem,calc(100vw-2rem))] md:max-h-[min(72dvh,calc(100svh-6rem))] md:rounded-2xl md:border md:p-6
              ${showSettings ? 'opacity-100 pointer-events-auto translate-y-0 md:scale-100' : 'opacity-0 pointer-events-none max-md:translate-y-full md:-translate-y-2 md:scale-95'}
            `}
          >
        <div className="flex flex-col gap-5 md:gap-8">
          
          {conversationMode !== null ? (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2.5 md:py-3">
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                {language === 'zh' ? '当前模式' : 'Mode'}
              </div>
              <p className="mt-1 text-[12px] leading-snug text-zinc-300">
                {conversationMode === 'live'
                  ? language === 'zh'
                    ? '实时语音'
                    : 'Live voice'
                  : language === 'zh'
                    ? '声音克隆'
                    : 'Voice clone'}
              </p>
            </div>
          ) : null}

          <div className="space-y-2 md:space-y-4">
            <div className="text-xs font-medium uppercase tracking-widest text-zinc-500">
              {language === 'zh' ? 'AI 角色名称' : 'AI Character Name'}
            </div>
            <input 
              type="text"
              placeholder={language === 'zh' ? '输入名称...' : 'Enter name...'}
              value={aiName}
              onChange={(e) => {
                setAiName(e.target.value);
                localStorage.setItem('subconscious_ai_name', e.target.value);
              }}
              className="min-h-[42px] w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 transition-colors focus:border-zinc-600 focus:outline-none md:min-h-0 md:px-4 md:py-2 md:text-sm"
            />
          </div>

          <div className={`space-y-2 md:space-y-4 ${conversationMode === 'live' ? 'opacity-45' : ''}`}>
            <div className="flex items-center justify-between text-xs font-medium uppercase tracking-widest text-zinc-500">
              <span>{language === 'zh' ? '朗读回复' : 'READ RESPONSES'}</span>
              <button 
                type="button"
                disabled={conversationMode === 'live'}
                onClick={() => {
                  const newVal = !isAutoSpeak;
                  setIsAutoSpeak(newVal);
                  localStorage.setItem('subconscious_auto_speak', String(newVal));
                }}
                className={`relative h-7 w-12 shrink-0 rounded-full transition-colors md:h-5 md:w-10 ${isAutoSpeak ? 'bg-rose-600' : 'bg-zinc-800'} ${conversationMode === 'live' ? 'cursor-not-allowed' : ''}`}
              >
                <span
                  className={`absolute left-0.5 top-0.5 block h-6 w-6 rounded-full bg-white shadow transition-transform duration-200 ease-out md:left-1 md:top-1 md:h-3 md:w-3 ${isAutoSpeak ? 'translate-x-5 md:translate-x-5' : 'translate-x-0'}`}
                />
              </button>
            </div>
          </div>

          <div className="space-y-2 border-t border-zinc-900 pt-4 md:space-y-3 md:pt-4">
              <div className="text-xs font-medium uppercase tracking-widest text-zinc-500">
                {language === 'zh' ? '声音克隆' : 'Voice clone'}
              </div>
              <input
                ref={cloneAudioInputRef}
                type="file"
                accept=".mp3,.m4a,.wav,.aac,audio/mpeg,audio/wav,audio/wave,audio/mp4,audio/x-m4a,audio/m4a,audio/aac,audio/quicktime,audio/*"
                className="hidden"
                onChange={handleCloneAudioSelected}
              />
              {voiceListForUi.length > 0 ? (
                <div className="space-y-1.5">
                  <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                    {language === 'zh' ? '已保存音色（可切换）' : 'Saved voices'}
                  </div>
                  <select
                    title={language === 'zh' ? '朗读使用的音色' : 'Voice used for read-aloud'}
                    className="min-h-[40px] w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-2 text-[11px] text-zinc-200 md:min-h-0"
                    value={clonedMinimaxVoiceId ?? ''}
                    onChange={(e) => {
                      const v = e.target.value;
                      setActiveVoiceProfileId(v || null);
                      setClonedMinimaxVoiceId(readStoredMinimaxClonedVoiceId());
                      setVoiceProfileTick((n) => n + 1);
                    }}
                  >
                    <option value="">
                      {language === 'zh' ? '内置默认音色' : 'Built-in default'}
                    </option>
                    {voiceListForUi.map((p) => (
                      <option key={p.voiceId} value={p.voiceId}>
                        {(p.label || p.voiceId).length > 40
                          ? `${(p.label || p.voiceId).slice(0, 38)}…`
                          : p.label || p.voiceId}
                      </option>
                    ))}
                  </select>
                  <div className="flex flex-wrap gap-2">
                    {clonedMinimaxVoiceId ? (
                      <button
                        type="button"
                        onClick={() => {
                          removeVoiceProfile(clonedMinimaxVoiceId);
                          setClonedMinimaxVoiceId(readStoredMinimaxClonedVoiceId());
                          setVoiceProfileTick((n) => n + 1);
                          setCloneVoiceHint(
                            language === 'zh' ? '已从列表移除。' : 'Removed from list.',
                          );
                        }}
                        className="inline-flex min-h-[36px] items-center gap-1 rounded-lg border border-zinc-800 px-2 py-1.5 text-[10px] text-zinc-500 transition-colors hover:border-zinc-600 hover:text-zinc-300"
                      >
                        <X className="h-3 w-3" strokeWidth={2} />
                        {language === 'zh' ? '删除所选档案' : 'Delete selected'}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      onClick={clearClonedMinimaxVoice}
                      className="inline-flex min-h-[36px] items-center gap-1 rounded-lg border border-zinc-800 px-2 py-1.5 text-[10px] text-zinc-500 transition-colors hover:border-zinc-600 hover:text-zinc-300"
                    >
                      {language === 'zh' ? '改用默认音色' : 'Use default voice'}
                    </button>
                  </div>
                </div>
              ) : (
                <p className="text-[11px] leading-snug text-zinc-600 md:text-[10px]">
                  {language === 'zh'
                    ? '暂无已保存的复刻音色。请始终在同一浏览器、同一站点地址下使用；无痕模式或清除站点数据会导致记录丢失。'
                    : 'No saved voices yet. Use the same browser and site address; private mode or clearing site data may erase saved voices.'}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={cloneVoiceBusy || !getMinimaxApiKey()}
                  onClick={() => cloneAudioInputRef.current?.click()}
                  className="inline-flex min-h-[40px] items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900/80 px-2.5 py-2 text-[10px] font-medium uppercase tracking-wider text-zinc-300 transition-colors hover:border-zinc-500 hover:text-zinc-100 disabled:opacity-40 md:min-h-0 md:py-1.5 md:text-[11px]"
                >
                  {cloneVoiceBusy ? (
                    <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" strokeWidth={2} />
                  ) : (
                    <FileAudio className="h-3.5 w-3.5 shrink-0" strokeWidth={2} />
                  )}
                  {language === 'zh' ? '上传录音复刻音色' : 'Upload audio to clone'}
                </button>
              </div>
              {cloneVoiceHint ? (
                <p className="text-[11px] leading-snug text-zinc-400 md:text-[10px]">{cloneVoiceHint}</p>
              ) : (
                <p className="text-[11px] leading-snug text-zinc-600 md:text-[10px]">
                  {language === 'zh'
                    ? '需 mp3 / m4a / wav，时长 10 秒～5 分钟，最大 20MB。'
                    : 'mp3 / m4a / wav, 10s–5min, max 20MB.'}
                </p>
              )}
            </div>

          <div className="space-y-2 md:space-y-4">
            <div className="flex items-center justify-between text-xs font-medium uppercase tracking-widest text-zinc-500">
              <span>{language === 'zh' ? '粒子大小' : 'Particle Size'}</span>
              <span className="tabular-nums">{particleSize}</span>
            </div>
            <input 
              title="Particle Size"
              type="range" 
              min="1" max="10" step="0.5" 
              value={particleSize} 
              onChange={(e) => setParticleSize(parseFloat(e.target.value))}
              className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-zinc-800 accent-zinc-400 md:h-1"
            />
          </div>

          <div className="space-y-2 md:space-y-4">
            <div className="flex items-center justify-between text-xs font-medium uppercase tracking-widest text-zinc-500">
              <span>{language === 'zh' ? '密度间隙' : 'Density Gap'}</span>
              <span className="tabular-nums">{particleGap}</span>
            </div>
            <input 
              title="Particle Density Gap"
              type="range" 
              min="2" max="12" step="1" 
              value={particleGap} 
              onChange={(e) => {
                setParticleGap(parseInt(e.target.value));
                if(imageRef.current) parseImagePixels(); // requires reparsing
              }}
              className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-zinc-800 accent-zinc-400 md:h-1"
            />
          </div>

          <div className="space-y-2 md:space-y-4">
            <div className="flex items-center justify-between text-xs font-medium uppercase tracking-widest text-zinc-500">
              <span>{language === 'zh' ? '光线容差' : 'Light Tolerance'}</span>
              <span className="tabular-nums">{threshold}</span>
            </div>
            <p className="text-xs leading-snug text-zinc-600 md:text-[10px] md:leading-tight">
              {language === 'zh' ? '较低的值会擦除较亮的背景。' : 'Lower values erase lighter backgrounds.'}
            </p>
            <input 
              title="Light Threshold Minimum filter"
              type="range" 
              min="10" max="255" step="5" 
              value={threshold} 
              onChange={(e) => {
                setThreshold(parseInt(e.target.value));
                if(imageRef.current) parseImagePixels(); // requires reparsing
              }}
              className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-zinc-800 accent-zinc-400 md:h-1"
            />
          </div>

          <div className="space-y-2 border-t border-zinc-900 pt-3 md:space-y-4 md:pt-2">
             <div className="flex items-center justify-between text-xs font-medium uppercase tracking-widest text-zinc-500">
               <span>{language === 'zh' ? '显化色彩' : 'Manifestation Color'}</span>
             </div>
             <div className="grid grid-cols-3 gap-2 md:gap-2">
                <button 
                  type="button"
                  onClick={() => setColorMode('original')}
                  className={`min-h-[38px] rounded-lg border py-2 text-[11px] font-medium tracking-wider transition-colors md:min-h-0 md:py-2 md:text-[10px] ${colorMode === 'original' ? 'border-zinc-500 text-zinc-300 bg-zinc-800' : 'border-zinc-800 text-zinc-600 hover:border-zinc-700'}`}
                >
                  {language === 'zh' ? '原始' : 'ORIGINAL'}
                </button>
                <button 
                  type="button"
                  onClick={() => setColorMode('ghost')}
                  className={`min-h-[38px] rounded-lg border py-2 text-[11px] font-medium tracking-wider transition-colors md:min-h-0 md:rounded md:py-2 md:text-[10px] ${colorMode === 'ghost' ? 'border-zinc-500 text-zinc-300 bg-zinc-800' : 'border-zinc-800 text-zinc-600 hover:border-zinc-700'}`}
                >
                  {language === 'zh' ? '幽灵' : 'GHOST'}
                </button>
                <button 
                  type="button"
                  onClick={() => setColorMode('blues')}
                  className={`min-h-[38px] rounded-lg border py-2 text-[11px] font-medium tracking-wider transition-colors md:min-h-0 md:rounded md:py-2 md:text-[10px] ${colorMode === 'blues' ? 'border-blue-500/50 text-blue-400 bg-blue-900/20' : 'border-zinc-800 text-zinc-600 hover:border-blue-900/50 hover:text-blue-500'}`}
                >
                  {language === 'zh' ? '蓝调' : 'BLUES'}
                </button>
                <button 
                  type="button"
                  onClick={() => setColorMode('greens')}
                  className={`min-h-[38px] rounded-lg border py-2 text-[11px] font-medium tracking-wider transition-colors md:min-h-0 md:rounded md:py-2 md:text-[10px] ${colorMode === 'greens' ? 'border-green-500/50 text-green-400 bg-emerald-900/20' : 'border-zinc-800 text-zinc-600 hover:border-green-900/50 hover:text-green-500'}`}
                >
                  {language === 'zh' ? '绿调' : 'GREENS'}
                </button>
                <button 
                  type="button"
                  onClick={() => setColorMode('purples')}
                  className={`min-h-[38px] rounded-lg border py-2 text-[11px] font-medium tracking-wider transition-colors md:min-h-0 md:rounded md:py-2 md:text-[10px] ${colorMode === 'purples' ? 'border-purple-500/50 text-purple-400 bg-purple-900/20' : 'border-zinc-800 text-zinc-600 hover:border-purple-900/50 hover:text-purple-500'}`}
                >
                  {language === 'zh' ? '紫调' : 'PURPLES'}
                </button>
                <div className="relative group">
                  <button 
                    type="button"
                    onClick={() => setColorMode('custom')}
                    className={`h-full min-h-[38px] w-full rounded-lg border py-2 text-[11px] font-medium tracking-wider transition-colors md:min-h-0 md:rounded md:py-2 md:text-[10px] ${colorMode === 'custom' ? 'border-zinc-500 text-zinc-300 bg-zinc-800' : 'border-zinc-800 text-zinc-600 hover:border-zinc-700'}`}
                  >
                    {language === 'zh' ? '自定义' : 'CUSTOM'}
                  </button>
                  <input 
                    type="color" 
                    value={customColor} 
                    onChange={(e) => {
                      setColorMode('custom');
                      setCustomColor(e.target.value);
                    }}
                    className="absolute inset-0 opacity-0 w-full h-full cursor-pointer z-10"
                  />
                </div>
             </div>
          </div>

        </div>
      </div>
        </>
      )}
    </div>
  );
}

