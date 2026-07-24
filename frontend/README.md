## Frontend

This folder contains the Next.js 16 chat interface for the Dual-Engine Banking Chatbot.

## Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Configuration

The frontend talks to the FastAPI backend through `NEXT_PUBLIC_API_URL`.

Copy [`.env.example`](.env.example) to `.env.local` if you need to override the default backend URL.

## Production build

```bash
npm run build
npm run start
```

## Notes

The frontend is designed to run alongside the root backend API. For the fastest local setup, use `docker compose up --build` from the repository root.
