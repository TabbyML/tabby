import { Metadata } from 'next'
import Sidebar from './components/sidebar'

export const metadata: Metadata = {
  title: 'Dashboard'
}

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: DashboardLayoutProps) {
  return <Sidebar className="flex-1">{children}</Sidebar>
}
