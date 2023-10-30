'use client'

import { cn } from '@/lib/utils'
import { VariantProps, cva } from 'class-variance-authority'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

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
        <div className="flex flex-col gap-2">
          <div className="h-[12px]"></div>
          <div className="flex-1">
            <nav className="grid items-start gap-4 px-4 text-sm font-medium">
              <SidebarButton href="/dashboard/home">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className=" h-4 w-4"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                  <polyline points="9 22 9 12 15 12 15 22" />
                </svg>
                Home
              </SidebarButton>
              <SidebarButton href="/dashboard/runners">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className=" h-4 w-4"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z" />
                  <path d="m6.08 9.5-3.5 1.6a1 1 0 0 0 0 1.81l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9a1 1 0 0 0 0-1.83l-3.5-1.59" />
                  <path d="m6.08 14.5-3.5 1.6a1 1 0 0 0 0 1.81l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9a1 1 0 0 0 0-1.83l-3.5-1.59" />
                </svg>
                Runners
              </SidebarButton>
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
