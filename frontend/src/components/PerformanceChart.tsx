import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

// Placeholder data - in real implementation, this would come from API
const data = [
  { time: '00:00', equity: 10000 },
  { time: '04:00', equity: 10150 },
  { time: '08:00', equity: 10100 },
  { time: '12:00', equity: 10300 },
  { time: '16:00', equity: 10450 },
  { time: '20:00', equity: 10500 },
]

export default function PerformanceChart() {
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="#0ea5e9"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
