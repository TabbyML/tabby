import { Metadata } from 'next'

import RepositoryTabsHeader from './components/tabs-header'

export const metadata: Metadata = {
  title: 'Providers'
}

export default function ProvidersLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <RepositoryTabsHeader />
      <div className="mt-8">{children}</div>
    </>
  )
}
