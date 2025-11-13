# Crypto Trading Bot - Frontend

React + TypeScript frontend for the cryptocurrency trading bot.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Material-UI (MUI)** - Component library
- **React Router** - Routing
- **React Query** - Data fetching and caching
- **Zustand** - State management
- **Recharts** - Charts and visualizations
- **Axios** - HTTP client

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Project Structure

```
src/
├── components/       # Reusable components
│   └── Layout.tsx   # Main layout with sidebar
├── pages/           # Page components
│   ├── Dashboard.tsx
│   ├── Trading.tsx
│   ├── Strategies.tsx
│   ├── Backtesting.tsx
│   ├── Settings.tsx
│   └── Logs.tsx
├── services/        # API and WebSocket services
│   └── api.ts
├── hooks/           # Custom React hooks
│   └── useWebSocket.ts
├── types/           # TypeScript type definitions
│   └── index.ts
├── App.tsx          # Main app component with routing
└── main.tsx         # Entry point
```

## Features

- **Dashboard**: Real-time overview of bot performance
- **Trading**: Live positions, orders, and trade history
- **Strategies**: Manage and configure trading strategies
- **Backtesting**: Run and view backtest results
- **Settings**: Configure bot parameters
- **Logs**: View system logs and monitoring data

## API Integration

The frontend communicates with the FastAPI backend via:
- **REST API**: For standard operations
- **WebSocket**: For real-time updates

API base URL is configured via environment variable `VITE_API_URL` (defaults to `http://localhost:8000/api`)

WebSocket URL is configured via `VITE_WS_URL` (defaults to `ws://localhost:8000/api/ws/live`)

## Development

```bash
# Run dev server with hot reload
npm run dev

# Type checking
npm run tsc

# Lint
npm run lint

# Format code
npm run format
```

## Environment Variables

Create a `.env.local` file:

```
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/api/ws/live
```

## License

MIT
