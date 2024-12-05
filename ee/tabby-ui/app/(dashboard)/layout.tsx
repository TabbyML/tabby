import { Metadata } from 'next'

import { LicenseBanner } from '@/components/license-banner'

import Layout from './components/dashboard-layout'

export const metadata: Metadata = {
  title: {
    default: 'Dashboard',
    template: `Tabby - %s`
  }
}

export default function DashboardLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <LicenseBanner />
      <Layout>{children}</Layout>
    </>
  )
}
