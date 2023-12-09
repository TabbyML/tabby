import { Metadata } from 'next'

import { Header } from '@/components/header'

import Sidebar from './components/sidebar'

export const metadata: Metadata = {
  title: {
    default: 'Dashboard',
    template: `Tabby - %s`
  },
}

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: DashboardLayoutProps) {
  return (
    <>
      <Header />
      <main className="flex flex-1 flex-col bg-muted/50">
        <Sidebar className="flex-1">{children}</Sidebar>
      </main>
    </>
  )
}
