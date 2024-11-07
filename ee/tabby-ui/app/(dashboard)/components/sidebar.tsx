'use client'

import React from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import { cva } from 'class-variance-authority'
import { escapeRegExp } from 'lodash-es'

import { useMe } from '@/lib/hooks/use-me'
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
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { useShowLicenseBanner } from '@/components/license-banner'
import LoadingWrapper from '@/components/loading-wrapper'
import { Sidebar, SidebarContent, SidebarHeader } from '@/components/ui/sidebar'

export interface SidebarProps {
  children?: React.ReactNode
  className?: string
}

type MenuLeaf = {
  title: string
  href: string
  allowUser?: boolean
}

type Menu =
  | {
    title: string
    icon: React.ReactNode
    allowUser?: boolean
    children: MenuLeaf[]
  }
  | {
    title: string
    href: string
    icon: React.ReactNode
    allowUser?: boolean
    children?: never
  }

const menus: Menu[] = [
  {
    title: 'Profile',
    icon: <IconUser />,
    href: '/profile',
    allowUser: true
  },
  {
    title: 'Information',
    icon: <IconBookOpenText />,
    children: [
      {
        title: 'System',
        href: '/system'
      },
      {
        title: 'Jobs',
        href: '/jobs'
      },
      {
        title: 'Reports',
        href: '/reports'
      },
      {
        title: 'Activities',
        href: '/activities'
      }
    ]
  },
  {
    title: 'Settings',
    icon: <IconGear />,
    allowUser: true,
    children: [
      {
        title: 'General',
        href: '/settings/general'
      },
      {
        title: 'Users & Groups',
        href: '/settings/team',
        allowUser: true
      },
      {
        title: 'Subscription',
        href: '/settings/subscription'
      }
    ]
  },
  {
    title: 'Integrations',
    icon: <IconLightingBolt />,
    children: [
      {
        title: 'Context Providers',
        href: '/settings/providers/git'
      },
      {
        title: 'SSO',
        href: '/settings/sso'
      },
      {
        title: 'Mail Delivery',
        href: '/settings/mail'
      }
    ]
  }
]

export default function AppSidebar({ children, className }: SidebarProps) {
  const [{ data, fetching: fetchingMe }] = useMe()
  const isAdmin = data?.me.isAdmin
  const [isShowDemoBanner] = useShowDemoBanner()
  const [isShowLicenseBanner] = useShowLicenseBanner()
  const showBanner = isShowDemoBanner || isShowLicenseBanner
  const style = showBanner
    ? {
      height: `calc(100vh - ${isShowDemoBanner ? BANNER_HEIGHT : '0rem'} - ${isShowLicenseBanner ? BANNER_HEIGHT : '0rem'
        })`,
      top: isShowDemoBanner ? BANNER_HEIGHT : 0
    }
    : { height: '100vh' }

  return (
    <Sidebar style={style} collapsible="icon">
      <SidebarHeader>
        <Link href="/" className="flex justify-center py-4">
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
      </SidebarHeader>
      <SidebarContent>
        <div className='text-sm font-medium'>
          <div className="flex flex-col gap-2 px-4 pb-4">
            <LoadingWrapper loading={fetchingMe}>
              {menus.map((menu, index) => {
                if (menu.allowUser || isAdmin) {
                  if (menu.children) {
                    return (
                      <SidebarCollapsible
                        key={index}
                        title={
                          <>
                            {menu.icon} {menu.title}
                          </>
                        }
                      >
                        {menu.children.map((child, childIndex) => {
                          if (child.allowUser || isAdmin) {
                            return (
                              <SidebarButton
                                key={childIndex}
                                href={child.href}
                              >
                                {child.title}
                              </SidebarButton>
                            )
                          }
                          return null
                        })}
                      </SidebarCollapsible>
                    )
                  } else {
                    return (
                      <SidebarButton key={index} href={menu.href}>
                        {menu.icon} {menu.title}
                      </SidebarButton>
                    )
                  }
                }
                return null
              })}
            </LoadingWrapper>
          </div>
        </div>
      </SidebarContent>
    </Sidebar>
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
    if (href.startsWith('/settings/providers')) {
      return pathname.startsWith('/settings/providers/')
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
