'use client'

import * as React from 'react'
import Link from 'next/link'
import { compare } from 'compare-versions'
import { has } from 'lodash-es'

import { WorkerKind } from '@/lib/gql/generates/graphql'
import { useHealth } from '@/lib/hooks/use-health'
import { ReleaseInfo, useLatestRelease } from '@/lib/hooks/use-latest-release'
import { useWorkers } from '@/lib/hooks/use-workers'
import { useAuthenticatedSession } from '@/lib/tabby/auth'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconGitHub, IconNotice } from '@/components/ui/icons'

import { ThemeToggle } from './theme-toggle'

export function Header() {
  // Ensure login status.
  useAuthenticatedSession()

  const { data } = useHealth()
  const workers = useWorkers()
  const isChatEnabled = has(workers, WorkerKind.Chat)
  const version = data?.version?.git_describe
  const { data: latestRelease } = useLatestRelease()
  const newVersionAvailable = isNewVersionAvailable(version, latestRelease)

  return (
    <header className="sticky top-0 z-50 flex h-16 w-full shrink-0 items-center justify-between border-b px-4 backdrop-blur-xl">
      <div className="flex items-center">
        <Link href="/">
          <span className="hidden select-none px-2 font-logo font-semibold sm:inline-block">
            Tabby
          </span>
        </Link>
        <HeaderLink href="/api">API</HeaderLink>
        {isChatEnabled && (
          <HeaderLink href="/playground">Playground</HeaderLink>
        )}
      </div>
      <div className="flex items-center justify-end space-x-2">
        <ThemeToggle />
        {newVersionAvailable && (
          <a
            target="_blank"
            href="https://github.com/TabbyML/tabby/releases/latest"
            rel="noopener noreferrer"
            className={buttonVariants({ variant: 'ghost' })}
          >
            <IconNotice className="text-yellow-600 dark:text-yellow-400" />
            <span className="ml-2 hidden md:flex">
              New version ({latestRelease?.name}) available
            </span>
          </a>
        )}
        <a
          target="_blank"
          href="https://github.com/TabbyML/tabby"
          rel="noopener noreferrer"
          className={cn(buttonVariants({ variant: 'outline' }))}
        >
          <IconGitHub />
          <span className="ml-2 hidden md:flex">GitHub</span>
        </a>
      </div>
    </header>
  )
}

function isNewVersionAvailable(version?: string, latestRelease?: ReleaseInfo) {
  try {
    return version && latestRelease && compare(latestRelease.name, version, '>')
  } catch (err) {
    // Handle invalid semver
    console.warn(err)
    return true
  }
}

const HeaderLink = ({
  children,
  href
}: {
  children: React.ReactNode
  href: string
}) => (
  <Link
    href={href}
    className={cn(buttonVariants({ variant: 'link' }), 'text-foreground')}
  >
    {children}
  </Link>
)
