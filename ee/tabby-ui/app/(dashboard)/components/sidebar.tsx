'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cva } from 'class-variance-authority'

import { cn } from '@/lib/utils'
import UserPanel from '@/components/user-panel'

export interface SidebarProps {
  children: React.ReactNode
  className?: string
}

export default function Sidebar({ children, className }: SidebarProps) {
  return (
    <div
      className={cn('grid overflow-hidden lg:grid-cols-[280px_1fr]', className)}
    >
      <div className="hidden border-r bg-zinc-100/40 dark:bg-zinc-800/40 lg:block">
        <div className="flex flex-col gap-2 h-full">
          <div className="h-[12px]"></div>
          <div className="flex-1">
            <nav className="grid items-start gap-4 px-4 text-sm font-medium">
              <SidebarButton href="/">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className=" h-4 w-4"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                  <polyline points="9 22 9 12 15 12 15 22" />
                </svg>
                Home
              </SidebarButton>
              <SidebarButton href="/swagger">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className=" h-4 w-4"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                  <path d="M6 8h2" />
                  <path d="M6 12h2" />
                  <path d="M16 8h2" />
                  <path d="M16 12h2" />
                </svg>
                Swagger
              </SidebarButton>
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
