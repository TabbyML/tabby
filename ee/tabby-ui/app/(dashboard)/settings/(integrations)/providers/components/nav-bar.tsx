'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cva } from 'class-variance-authority'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'

import { PROVIDER_KIND_METAS } from '../constants'

const linkVariants = cva(
  'flex items-center gap-1 rounded-lg px-3 py-2 transition-all hover:bg-accent',
  {
    variants: {
      state: {
        selected: 'bg-accent',
        'not-selected': ''
      }
    },
    defaultVariants: {
      state: 'not-selected'
    }
  }
)

interface SidebarButtonProps {
  href: string
  children: React.ReactNode
}

export default function NavBar({ className }: { className?: string }) {
  const [isShowDemoBanner] = useShowDemoBanner()

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT} - 4rem)` }
    : { height: 'calc(100vh - 4rem)' }

  return (
    <div
      className={cn(
        'sticky top-16 space-y-1 overflow-y-auto border-r pr-4 text-sm font-medium',
        className
      )}
      style={style}
    >
      <SidebarButton href="/settings/providers/git">Git</SidebarButton>
      {PROVIDER_KIND_METAS.map(provider => {
        return (
          <SidebarButton
            href={`/settings/providers/${provider.name}`}
            key={provider.name}
          >
            {provider.meta.displayName}
          </SidebarButton>
        )
      })}
      <SidebarButton href="/settings/providers/doc">
        Developer Docs
        <Badge
          variant="outline"
          className="h-3.5 border-secondary-foreground/60 px-1.5 text-[10px] text-secondary-foreground/60"
        >
          Beta
        </Badge>
      </SidebarButton>
    </div>
  )
}

function SidebarButton({ href, children }: SidebarButtonProps) {
  const pathname = usePathname()
  const isSelected = React.useMemo(() => {
    const docPathname = '/settings/providers/doc'
    if (pathname?.startsWith(docPathname)) {
      return href.startsWith(docPathname)
    }

    const matcher = pathname.match(/^(\/settings\/providers\/[\w-]+)/)?.[1]
    return matcher === href
  }, [pathname, href])

  const state = isSelected ? 'selected' : 'not-selected'
  return (
    <Link className={linkVariants({ state })} href={href}>
      {children}
    </Link>
  )
}
