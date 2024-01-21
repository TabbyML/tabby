'use client'

import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import { cva } from 'class-variance-authority'

import { useSession } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import { IconHome, IconNetwork, IconUsers } from '@/components/ui/icons'

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Button } from '@/components/ui/button'

export interface SidebarProps {
  children: React.ReactNode
  className?: string
}

export default function Sidebar({ children, className }: SidebarProps) {
  const { data: session } = useSession()
  const isAdmin = session?.isAdmin || false
  return (
    <div
      className={cn('grid overflow-hidden lg:grid-cols-[280px_1fr]', className)}
    >
      <div className="hidden border-r lg:block">
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
                  <Collapsible>
                    <CollapsibleTrigger className='w-full'>
                      <span className={linkVariants()}><IconUsers /> Workspace</span>
                    </CollapsibleTrigger>
                    <CollapsibleContent className='pt-2'>
                      <SidebarButton href="/team">
                        <span className='w-4' /> Team Management
                      </SidebarButton>
                    </CollapsibleContent>
                  </Collapsible>
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
  const state = pathname == href ? 'selected' : 'not-selected'
  return (
    <Link className={linkVariants({ state })} href={href}>
      {children}
    </Link>
  )
}
