import { Metadata } from 'next'

import { Header } from '@/components/header'

import Sidebar from './components/sidebar'
import { ScrollArea } from '@/components/ui/scroll-area'

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
    <main className="flex w-full h-full">
      <Sidebar />
      <ScrollArea className="h-[100vh] flex-1 flex flex-col">
        <Header />
        <div className="flex-1 p-4 lg:p-10">{children}</div>
      </ScrollArea>
    </main>
  )
}
