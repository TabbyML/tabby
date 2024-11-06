import { Metadata } from 'next'

import { LicenseBanner } from '@/components/license-banner'

import MainContent from './components/main-content'
import Sidebar from './components/sidebar'

export const metadata: Metadata = {
  title: {
    default: 'Home',
    template: `Tabby - %s`
  }
}

export default function RootLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <LicenseBanner />
      <main className="flex flex-1">
        <Sidebar />
        <MainContent>{children}</MainContent>
      </main>
    </>
  )
}
