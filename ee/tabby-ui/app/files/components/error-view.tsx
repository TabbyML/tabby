import React, { ReactComponentElement } from 'react'
import Link from 'next/link'

import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconArrowRight, IconFileSearch } from '@/components/ui/icons'

import { CodeBrowserError } from './utils'

interface ErrorViewProps extends React.HTMLAttributes<HTMLDivElement> {
  error: Error | undefined
}

export const ErrorView: React.FC<ErrorViewProps> = ({ className, error }) => {
  let errorComponent: ReactComponentElement<any, any> = <NotFoundError />

  switch (error?.message) {
    case CodeBrowserError.REPOSITORY_NOT_FOUND:
      errorComponent = <RepositoryNotFoundError />
      break
    case CodeBrowserError.REPOSITORY_SYNC_FAILED:
      errorComponent = <RepositorySyncError />
      break
    case CodeBrowserError.INVALID_URL:
      errorComponent = <InvalidUrlError />
      break
    case CodeBrowserError.FAILED_TO_FETCH:
      errorComponent = <FailToFetchError />
      break
  }

  return (
    <div
      className={cn('flex min-h-[30vh] items-center justify-center', className)}
    >
      <div className="flex flex-col items-center gap-4">{errorComponent}</div>
    </div>
  )
}

function RepositoryNotFoundError() {
  return (
    <>
      <div className="flex items-center gap-2">
        <IconFileSearch className="h-6 w-6" />
        <div className="text-xl font-semibold">Repository not found</div>
      </div>
      <Link href="/files" className={cn(buttonVariants(), 'gap-2')}>
        <span>Back to repositories</span>
        <IconArrowRight />
      </Link>
    </>
  )
}

function NotFoundError() {
  return (
    <>
      <div className="flex items-center gap-2">
        <IconFileSearch className="h-6 w-6" />
        <div className="text-xl font-semibold">Not found</div>
      </div>
      <Link href="/files" className={cn(buttonVariants(), 'gap-2')}>
        <span>Back to repositories</span>
        <IconArrowRight />
      </Link>
    </>
  )
}

function RepositorySyncError() {
  return (
    <>
      <div className="flex items-center gap-2">
        <IconFileSearch className="h-6 w-6" />
        <div className="text-xl font-semibold">
          Repository is not cloned properly
        </div>
      </div>
      <div>
        The cloning of the repository has failed. Please verify your settings or
        attempt to retry the job.
      </div>
      <Link
        href="/settings/providers/git"
        className={cn(buttonVariants(), 'gap-2')}
      >
        <span>Providers Configuration</span>
        <IconArrowRight />
      </Link>
    </>
  )
}

function InvalidUrlError() {
  return (
    <>
      <div className="flex items-center gap-2">
        <IconFileSearch className="h-6 w-6" />
        <div className="text-xl font-semibold">Invalid URL</div>
      </div>
      <Link href="/files" className={cn(buttonVariants(), 'gap-2')}>
        <span>Back to repositories</span>
        <IconArrowRight />
      </Link>
    </>
  )
}

function FailToFetchError() {
  return (
    <>
      <div className="flex items-center gap-2">
        <IconFileSearch className="h-6 w-6" />
        <div className="text-xl font-semibold">Failed to fetch</div>
      </div>
    </>
  )
}
