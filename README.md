<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/7c2d2cc1-e438-425e-a0ac-4e31bb067b09

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Optional: set `MINIMAX_API_KEY` (and optionally `MINIMAX_VOICE_ID`, `MINIMAX_TTS_MODEL`) so AI replies are spoken via MiniMax TTS instead of the browser voice. Local `npm run dev` proxies `/minimax-api` to `api.minimax.io` to avoid CORS.
4. Run the app:
   `npm run dev`
