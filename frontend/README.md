# Crypto Trading Bot Dashboard

Modern web dashboard for monitoring and controlling the cryptocurrency trading bot.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Query** - Data fetching and state management
- **Recharts** - Charts and data visualization
- **React Router** - Navigation
- **Axios** - HTTP client

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Configure the API URL in `.env` if needed (default: http://localhost:8000/api)

### Development

Start the development server:
```bash
npm run dev
```

The dashboard will be available at http://localhost:3000

### Build for Production

Build the optimized production bundle:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Features

- **Real-time Monitoring** - Live bot status and performance metrics
- **Position Tracking** - Monitor open positions and unrealized P&L
- **Trade History** - View all executed trades
- **Strategy Management** - Configure and switch trading strategies
- **Bot Control** - Start, stop, and restart the bot from the UI
- **Performance Charts** - Visualize trading performance over time

## Project Structure

```
frontend/
├── src/
│   ├── api/           # API client and endpoints
│   ├── components/    # Reusable React components
│   ├── pages/         # Page components
│   ├── types/         # TypeScript type definitions
│   ├── App.tsx        # Main app component
│   ├── main.tsx       # App entry point
│   └── index.css      # Global styles
├── public/            # Static assets
├── index.html         # HTML template
├── package.json       # Dependencies and scripts
├── vite.config.ts     # Vite configuration
├── tailwind.config.js # Tailwind CSS configuration
└── tsconfig.json      # TypeScript configuration
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The dashboard connects to the trading bot API. Make sure the backend API is running before starting the frontend.

Default API URL: http://localhost:8000/api

## License

MIT
