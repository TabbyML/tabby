import { Metadata } from 'next'

import { Header } from '@/components/header'

import Sidebar from './components/sidebar'

export const metadata: Metadata = {
  title: {
    default: 'Home',
    template: `Tabby - %s`
  }
}

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: DashboardLayoutProps) {
  return (
    <main className="flex flex-1">
      <Sidebar />
      <div className="flex-1 flex flex-col min-h-full">
        <Header />
        <div className="p-4 flex-1">{children}</div>
      </div>
    </main>
  )
}
