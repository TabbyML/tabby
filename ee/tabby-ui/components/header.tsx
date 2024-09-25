'use client'

import * as React from 'react'
import { compare } from 'compare-versions'

import { useHealth } from '@/lib/hooks/use-health'
import { ReleaseInfo, useLatestRelease } from '@/lib/hooks/use-latest-release'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconNotice } from '@/components/ui/icons'

import { ClientOnly } from './client-only'
import { ThemeToggle } from './theme-toggle'
import { MyAvatar } from './user-avatar'
import UserPanel from './user-panel'

export function Header() {
  const { data } = useHealth()
  const version = data?.version?.git_describe
  const { data: latestRelease } = useLatestRelease()
  const newVersionAvailable = isNewVersionAvailable(version, latestRelease)

  return (
    <header className="sticky top-0 z-50 flex h-16 w-full shrink-0 items-center justify-between border-b px-4 backdrop-blur-xl lg:px-10">
      <div className="flex items-center">
        {newVersionAvailable && (
          <a
            target="_blank"
            href="https://github.com/TabbyML/tabby/releases/latest"
            rel="noopener noreferrer"
            className={cn('!pl-0', buttonVariants({ variant: 'ghost' }))}
          >
            <IconNotice className="text-yellow-600 dark:text-yellow-400" />
            <span className="ml-2 hidden md:flex">
              New version ({latestRelease?.name}) available
            </span>
          </a>
        )}
      </div>
      <div className="flex items-center justify-center gap-6">
        <ClientOnly>
          <ThemeToggle />
        </ClientOnly>
        <UserPanel>
          <MyAvatar className="h-10 w-10 border" />
        </UserPanel>
      </div>
    </header>
  )
}

function isNewVersionAvailable(version?: string, latestRelease?: ReleaseInfo) {
  try {
    return version && latestRelease && compare(latestRelease.name, version, '>')
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(err)

    // Handle invalid semver
    return true
  }
}
