import { Metadata } from 'next'

import { LicenseBanner } from '@/components/license-banner'

import MainContent from './components/main-content'
import AppSidebar from './components/sidebar'
import { SidebarProvider } from '@/components/ui/sidebar'

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
      <SidebarProvider className='flex-1'>
        <AppSidebar />
        <MainContent>{children}</MainContent>
      </SidebarProvider>
    </>
  )
}
