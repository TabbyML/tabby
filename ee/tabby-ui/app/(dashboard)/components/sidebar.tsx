'use client'

import React, { FunctionComponent } from 'react'
import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import tabbyLogo from '@/assets/tabby.png'

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
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarHeader,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem
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
          className="flex justify-center items-center py-2 h-[3.375rem]"
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
        <SidebarGroup className="text-sm font-medium leading-normal list-none space-y-2">
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
                        <CollapsibleTrigger asChild>
                          <SidebarMenuButton
                            key={menu.title}
                            tooltip={{
                              children: (
                                <div
                                  className="w-[theme(space.48)]"
                                  key={menu.title}
                                >
                                  <div className="mt-1 mb-2 text-sm ml-2 text-muted-foreground font-medium">
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
                              ),
                              align: 'start',
                              side: 'right'
                            }}
                          >
                            {!!menu.icon && <menu.icon />}
                            <span>{menu.title}</span>
                            <IconChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                          </SidebarMenuButton>
                        </CollapsibleTrigger>
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
                        tooltip={menu.title}
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
