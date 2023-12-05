import { Metadata } from 'next'
import Sidebar from './components/sidebar'
import { Header } from '@/components/header'

export const metadata: Metadata = {
  title: 'Dashboard'
}

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: DashboardLayoutProps) {
  return <>
    <Header />
    <main className="bg-muted/50 flex flex-1 flex-col">
      <Sidebar className="flex-1">{children}</Sidebar>
    </main>
  </>
}
