/**
 * 释放本机常见 Vite 端口（改 .env 后 `npm run dev:restart` 会先执行此脚本）。
 * 仅 macOS/Linux（lsof）；Windows 下跳过不报错。
 */
import { execSync } from 'node:child_process';

const ports = [3000, 3001, 3002, 3003, 3004, 3005, 3006];
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
