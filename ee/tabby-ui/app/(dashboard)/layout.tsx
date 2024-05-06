import { Metadata } from 'next'

import Sidebar from './components/sidebar'
import MainContent from './components/main-content'

export const metadata: Metadata = {
  title: {
    default: 'Home',
    template: `Tabby - %s`
  }
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex flex-1">
      <Sidebar />
      <MainContent>{children}</MainContent>
    </main>
  )
}
