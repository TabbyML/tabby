'use client'

import React from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import { cva } from 'class-variance-authority'

import { useMe } from '@/lib/hooks/use-me'
import { cn } from '@/lib/utils'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from '@/components/ui/collapsible'
import {
  IconBookOpenText,
  IconChevronRight,
  IconGear,
  IconLightingBolt,
  IconUser
} from '@/components/ui/icons'

export interface SidebarProps {
  children?: React.ReactNode
  className?: string
}

export default function Sidebar({ children, className }: SidebarProps) {
  const [{ data }] = useMe()
  const isAdmin = data?.me.isAdmin
  return (
    <div
      className={cn('grid overflow-hidden md:grid-cols-[280px_1fr]', className)}
    >
      <div className="fixed inset-y-0 left-0 hidden w-[280px] border-r pt-4 md:block">
        <nav className="flex h-full flex-col overflow-hidden text-sm font-medium">
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
          <div className="flex-1 overflow-y-auto">
            <div className="flex flex-col gap-2 px-4 pb-4">
              <SidebarButton href="/profile">
                <IconUser /> Profile
              </SidebarButton>
              {isAdmin && (
                <>
                  <SidebarCollapsible
                    title={
                      <>
                        <IconBookOpenText /> Information
                      </>
                    }
                  >
                    <SidebarButton href="/system">System</SidebarButton>
                    <SidebarButton href="/jobs">Jobs</SidebarButton>
                    <SidebarButton href="/reports">Reports</SidebarButton>
                    <SidebarButton href="/activities">Activities</SidebarButton>
                  </SidebarCollapsible>
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
                    <SidebarButton href="/settings/subscription">
                      Subscription
                    </SidebarButton>
                  </SidebarCollapsible>
                  <SidebarCollapsible
                    title={
                      <>
                        <IconLightingBolt />
                        Integrations
                      </>
                    }
                  >
                    <SidebarButton href="/settings/repository/git">
                      Repository Providers
                    </SidebarButton>
                    <SidebarButton href="/settings/sso">SSO</SidebarButton>
                    <SidebarButton href="/settings/mail">
                      Mail Delivery
                    </SidebarButton>
                  </SidebarCollapsible>
                </>
              )}
            </div>
          </div>
        </nav>
      </div>
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
    if (href === '/') return href === pathname
    if (href.startsWith('/settings/repository')) {
      return pathname.startsWith('/settings/repository/')
    }

    return shouldPathnameHighlight(pathname, href)
  }, [pathname, href])

  const state = isSelected ? 'selected' : 'not-selected'
  return (
    <Link className={linkVariants({ state })} href={href}>
      {children}
    </Link>
  )
}

function shouldPathnameHighlight(
  currentPathname: string,
  pathToHighlight: string
) {
  const regex = new RegExp(`^${escapeRegExp(pathToHighlight)}(/|\\?|$)`)
  return regex.test(currentPathname)
}

function escapeRegExp(string: String) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

interface SidebarCollapsibleProps {
  title: React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
}

function SidebarCollapsible({ title, children }: SidebarCollapsibleProps) {
  return (
    <Collapsible
      defaultOpen={true}
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
