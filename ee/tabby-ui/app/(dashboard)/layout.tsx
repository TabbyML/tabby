import { Metadata } from 'next'

import { ScrollArea } from '@/components/ui/scroll-area'
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
    <>
      <main className="flex flex-1 flex-col">
        <Sidebar className="flex-1">
          <ScrollArea className="max-h-[100vh]">
            <Header />
            <div className="p-4">{children}</div>
          </ScrollArea>
        </Sidebar>
      </main>
    </>
  )
}
