'use client'

import React, { FunctionComponent } from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import tabbyLogo from '@/assets/tabby.png'
import { HoverCardPortal } from '@radix-ui/react-hover-card'

import { useMe } from '@/lib/hooks/use-me'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from '@/components/ui/collapsible'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import {
  IconBookOpenText,
  IconChevronRight,
  IconGear,
  IconLightingBolt,
  IconUser
} from '@/components/ui/icons'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarHeader,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  useSidebar
} from '@/components/ui/sidebar'
import LoadingWrapper from '@/components/loading-wrapper'

export interface SidebarProps {
  children?: React.ReactNode
  className?: string
}

type SubMenu = {
  title: string
  href: string
  allowUser?: boolean
}

type Menu =
  | {
      title: string
      icon: FunctionComponent
      allowUser?: boolean
      items: SubMenu[]
    }
  | {
      title: string
      href: string
      icon: FunctionComponent
      allowUser?: boolean
      items?: never
    }

const menus: Menu[] = [
  {
    title: 'Profile',
    icon: IconUser,
    href: '/profile',
    allowUser: true
  },
  {
    title: 'Information',
    icon: IconBookOpenText,
    items: [
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
    icon: IconGear,
    allowUser: true,
    items: [
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
    icon: IconLightingBolt,
    items: [
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

export default function AppSidebar() {
  const pathname = usePathname()
  const [{ data, fetching: fetchingMe }] = useMe()
  const isAdmin = data?.me.isAdmin
  const { isMobile, state } = useSidebar()

  return (
    <Sidebar
      style={{
        position: 'absolute',
        top: 0,
        bottom: 0
      }}
      collapsible="icon"
    >
      <SidebarHeader>
        <Link
          href="/"
          className="flex h-[3.375rem] items-center justify-center py-2"
        >
          <>
            <Image
              src={tabbyLogo}
              width={32}
              alt="logo"
              className="hidden group-data-[collapsible=icon]:block"
            />
            <div className="w-[128px] group-data-[collapsible=icon]:hidden">
              <Image
                src={logoUrl}
                alt="logo"
                className="dark:hidden"
                width={128}
              />
              <Image
                src={logoDarkUrl}
                alt="logo"
                width={96}
                className="hidden dark:block"
              />
            </div>
          </>
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup className="list-none space-y-2 text-sm font-medium leading-normal">
          <LoadingWrapper loading={fetchingMe}>
            {menus.map(menu => {
              if (isAdmin || menu.allowUser) {
                if (menu.items) {
                  return (
                    <Collapsible
                      defaultOpen
                      asChild
                      className="group/collapsible"
                      key={`collapsible_${menu.title}`}
                    >
                      <SidebarMenuItem>
                        <HoverCard openDelay={200} closeDelay={200}>
                          <HoverCardTrigger asChild>
                            <CollapsibleTrigger asChild>
                              <SidebarMenuButton key={menu.title}>
                                {!!menu.icon && <menu.icon />}
                                <span>{menu.title}</span>
                                <IconChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                              </SidebarMenuButton>
                            </CollapsibleTrigger>
                          </HoverCardTrigger>
                          <HoverCardPortal>
                            <HoverCardContent
                              align="start"
                              side="right"
                              sideOffset={4}
                              hidden={state !== 'collapsed' || isMobile}
                              className="w-[theme(space.48)] py-2"
                            >
                              <div key={menu.title}>
                                <div className="mb-2 ml-2 mt-1 text-sm font-medium text-muted-foreground">
                                  {menu.title}
                                </div>
                                <div className="space-y-1">
                                  {menu.items.map(item => {
                                    if (isAdmin || item.allowUser) {
                                      return (
                                        <SidebarMenuButton
                                          key={item.title}
                                          asChild
                                          isActive={pathname.startsWith(
                                            item.href
                                          )}
                                        >
                                          <Link href={item.href}>
                                            <span>{item.title}</span>
                                          </Link>
                                        </SidebarMenuButton>
                                      )
                                    } else {
                                      return null
                                    }
                                  })}
                                </div>
                              </div>
                            </HoverCardContent>
                          </HoverCardPortal>
                        </HoverCard>
                        <CollapsibleContent>
                          <SidebarMenuSub>
                            {menu.items.map(item => {
                              if (isAdmin || item.allowUser) {
                                return (
                                  <SidebarMenuSubItem key={item.title}>
                                    <SidebarMenuSubButton
                                      asChild
                                      isActive={pathname.startsWith(item.href)}
                                    >
                                      <Link href={item.href}>
                                        <span>{item.title}</span>
                                      </Link>
                                    </SidebarMenuSubButton>
                                  </SidebarMenuSubItem>
                                )
                              }
                            })}
                          </SidebarMenuSub>
                        </CollapsibleContent>
                      </SidebarMenuItem>
                    </Collapsible>
                  )
                } else {
                  return (
                    <SidebarMenuItem key={menu.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={pathname.startsWith(menu.href)}
                        tooltip={{
                          children: (
                            <span className="text-sm font-medium text-muted-foreground">
                              {menu.title}
                            </span>
                          )
                        }}
                      >
                        <Link href={menu.href}>
                          {!!menu.icon && <menu.icon />}
                          <span>{menu.title}</span>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )
                }
              }
              return null
            })}
          </LoadingWrapper>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}
