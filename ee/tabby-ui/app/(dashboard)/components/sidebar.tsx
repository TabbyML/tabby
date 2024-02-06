'use client'

import React from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import { cva } from 'class-variance-authority'

import { useSession } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from '@/components/ui/collapsible'
import {
  IconChevronRight,
  IconGear,
  IconHome,
  IconLightingBolt,
  IconNetwork
} from '@/components/ui/icons'

export interface SidebarProps {
  children: React.ReactNode
  className?: string
}

export default function Sidebar({ children, className }: SidebarProps) {
  const { data: session } = useSession()
  const isAdmin = session?.isAdmin || false
  return (
    <div
      className={cn('grid overflow-hidden md:grid-cols-[280px_1fr]', className)}
    >
      <div className="hidden border-r md:block">
        <div className="flex h-full flex-col gap-2">
          <div className="h-[12px]"></div>
          <div className="flex-1">
            <nav className="grid items-start gap-2 px-4 text-sm font-medium">
              <Link href="/" className="flex justify-center pb-4">
                <Image
                  src={logoUrl}
                  alt="logo"
                  width={128}
                  className="dark:hidden"
                />
                <Image
                  src={logoDarkUrl}
                  alt="logo"
                  width={96}
                  className="hidden dark:block"
                />
              </Link>
              <SidebarButton href="/">
                <IconHome /> Home
              </SidebarButton>
              {isAdmin && (
                <>
                  <SidebarButton href="/cluster">
                    <IconNetwork /> Cluster Information
                  </SidebarButton>
                  <SidebarCollapsible
                    title={
                      <>
                        <IconGear />
                        Settings
                      </>
                    }
                  >
                    <SidebarButton href="/settings/general">
                      General
                    </SidebarButton>
                    <SidebarButton href="/settings/team">Members</SidebarButton>
                  </SidebarCollapsible>
                  <SidebarCollapsible
                    title={
                      <>
                        <IconLightingBolt />
                        Integrations
                      </>
                    }
                  >
                    <SidebarButton href="/settings/repository">
                      Git Repositories
                    </SidebarButton>
                    <SidebarButton href="/settings/sso">SSO</SidebarButton>
                    <SidebarButton href="/settings/mail">
                      Mail Delivery
                    </SidebarButton>
                  </SidebarCollapsible>
                </>
              )}
            </nav>
          </div>
        </div>
      </div>
      <div className="flex flex-1 flex-col overflow-auto">{children}</div>
    </div>
  )
}

interface SidebarButtonProps {
  href: string
  children: React.ReactNode
}

const linkVariants = cva(
  'flex items-center gap-3 rounded-lg px-3 py-2 transition-all hover:bg-accent',
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

function SidebarButton({ href, children }: SidebarButtonProps) {
  const pathname = usePathname()
  const isSelected = React.useMemo(() => {
    return href === '/' ? href === pathname : pathname?.startsWith(href)
  }, [pathname, href])

  const state = isSelected ? 'selected' : 'not-selected'
  return (
    <Link className={linkVariants({ state })} href={href}>
      {children}
    </Link>
  )
}

interface SidebarCollapsibleProps {
  title: React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
}

function SidebarCollapsible({
  title,
  children,
  defaultOpen = true
}: SidebarCollapsibleProps) {
  return (
    <Collapsible
      defaultOpen={defaultOpen}
      className="[&_svg.ml-auto]:data-[state=open]:rotate-90"
    >
      <CollapsibleTrigger className="w-full">
        <span className={linkVariants()}>
          {title}
          <IconChevronRight className="ml-auto" />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent className="ml-7 flex flex-col gap-1 data-[state=open]:py-1">
        {children}
      </CollapsibleContent>
    </Collapsible>
  )
}
