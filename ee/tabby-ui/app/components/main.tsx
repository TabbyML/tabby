'use client'

import { usePathname } from 'next/navigation'

import { cn } from '@/lib/utils'
import { DemoBanner } from '@/components/demo-banner'

export default function Main({ children }: { children: React.ReactNode }) {
  const pathName = usePathname()
  return (
    <div
      className={cn('flex min-h-screen flex-col', {
        'bg-background': pathName !== '/chat',
        'bg-transparent': pathName === '/chat'
      })}
    >
      <DemoBanner />
      {children}
    </div>
  )
}
