import { Metadata } from 'next'

import { ScrollArea } from '@/components/ui/scroll-area'

import ProviderNavBar from './components/tabs-header'

export const metadata: Metadata = {
  title: 'Context Providers'
}

export default function ProvidersLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <div className="-m-4 flex lg:-m-10">
      <ProviderNavBar className="w-[250px] pl-4 pt-4" />
      <ScrollArea className="flex-1 p-4">{children}</ScrollArea>
    </div>
  )
}
