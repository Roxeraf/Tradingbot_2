import { Outlet, NavLink } from 'react-router-dom'
import { Home, TrendingUp, Settings, Activity } from 'lucide-react'
import BotControlPanel from './BotControlPanel'

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Trades', href: '/trades', icon: TrendingUp },
  { name: 'Strategies', href: '/strategies', icon: Activity },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 w-64 bg-white shadow-lg">
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center px-6 border-b">
            <h1 className="text-xl font-bold text-primary-600">
              Crypto Trading Bot
            </h1>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                end={item.href === '/'}
                className={({ isActive }) =>
                  `flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    isActive
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`
                }
              >
                <item.icon className="mr-3 h-5 w-5" />
                {item.name}
              </NavLink>
            ))}
          </nav>

          {/* Bot Control Panel */}
          <div className="border-t p-4">
            <BotControlPanel />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="py-6 px-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
