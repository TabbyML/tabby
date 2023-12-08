'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconGitHub, IconNotice } from '@/components/ui/icons'
import Link from 'next/link'
import { useHealth } from '@/lib/hooks/use-health'
import { ReleaseInfo, useLatestRelease } from '@/lib/hooks/use-latest-release'
import { compare } from 'compare-versions'
import { useWorkers } from '@/lib/hooks/use-workers'
import { WorkerKind } from '@/lib/gql/generates/graphql'
import { has } from 'lodash-es'
import { ThemeToggle } from './theme-toggle'

export function Header() {
  const { data } = useHealth()
  const workers = useWorkers(data)
  const isChatEnabled = has(workers, WorkerKind.Chat)
  const version = data?.version?.git_describe
  const { data: latestRelease } = useLatestRelease()
  const newVersionAvailable = isNewVersionAvailable(version, latestRelease)

  return (
    <header className="from-background/10 via-background/50 to-background/80 sticky top-0 z-50 flex h-16 w-full shrink-0 items-center justify-between border-b bg-gradient-to-b px-4 backdrop-blur-xl">
      <div className="flex items-center">
        <ThemeToggle />
        <Link href="/" className={cn(buttonVariants({ variant: 'link' }))}>
          Dashboard
        </Link>
        {isChatEnabled && (
          <Link
            href="/playground"
            className={cn(buttonVariants({ variant: 'link' }))}
          >
            Playground
          </Link>
        )}
      </div>
      <div className="flex items-center justify-end space-x-2">
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
