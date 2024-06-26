import { Metadata } from 'next'

import { ScrollArea } from '@/components/ui/scroll-area'

import ProviderNavBar from './components/nav-bar'

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
      <ProviderNavBar className="w-[200px] pl-4 pt-4 lg:w-[250px]" />
      <ScrollArea className="flex-1">
        <div className="p-4 lg:p-10">{children}</div>
      </ScrollArea>
    </div>
  )
}
