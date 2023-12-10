'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cva } from 'class-variance-authority'

import { useSession } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import { IconHome, IconNetwork, IconUsers } from '@/components/ui/icons'
import UserPanel from '@/components/user-panel'

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
      <div className="hidden border-r bg-zinc-100/40 dark:bg-zinc-800/40 lg:block">
        <div className="flex h-full flex-col gap-2">
          <div className="h-[12px]"></div>
          <div className="flex-1">
            <nav className="grid items-start gap-4 px-4 text-sm font-medium">
              <SidebarButton href="/">
                <IconHome /> Home
              </SidebarButton>
              {isAdmin && (
                <>
                  <SidebarButton href="/cluster">
                    <IconNetwork /> Cluster Information
                  </SidebarButton>
                  <SidebarButton href="/team">
                    <IconUsers /> Team Management
                  </SidebarButton>
                </>
              )}
            </nav>
          </div>

          <div className="mt-auto">
            <UserPanel />
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
  'flex items-center gap-3 rounded-lg px-3 py-2 text-zinc-900 transition-all hover:text-zinc-900 dark:text-zinc-50 dark:hover:text-zinc-50',
  {
    variants: {
      state: {
        selected: 'bg-zinc-200 dark:bg-zinc-800',
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
