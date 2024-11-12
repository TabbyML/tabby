import { Metadata } from 'next'

import { SidebarProvider } from '@/components/ui/sidebar'
import { LicenseBanner } from '@/components/license-banner'

import MainContent from './components/main-content'
import AppSidebar from './components/sidebar'

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
      <SidebarProvider className="relative">
        <AppSidebar />
        <MainContent>{children}</MainContent>
      </SidebarProvider>
    </>
  )
}
