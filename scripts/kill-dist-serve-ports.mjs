/**
 * 释放 `serve-render-dist` / `vite preview` 常用端口，避免 EADDRINUSE。
 * 仅 macOS/Linux（lsof）；Windows 下跳过不报错。
 */
import { execSync } from 'node:child_process';

const ports = [4173, 4174];
if (process.platform === 'win32') process.exit(0);

for (const p of ports) {
  try {
    const out = execSync(`lsof -ti tcp:${p}`, { encoding: 'utf8' }).trim();
    for (const pid of out.split(/\s+/).filter(Boolean)) {
      try {
        process.kill(Number(pid), 'SIGKILL');
      } catch {
        /* ignore */
      }
    }
  } catch {
    /* no listener */
  }
}
