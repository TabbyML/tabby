import React, { ReactComponentElement } from 'react'
import Link from 'next/link'

import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconArrowRight, IconFileSearch } from '@/components/ui/icons'

import { Errors } from './utils'

interface ErrorViewProps extends React.HTMLAttributes<HTMLDivElement> {
  error: Error | undefined
}

export const ErrorView: React.FC<ErrorViewProps> = ({ className, error }) => {
  const isEmptyRepository = error?.message === Errors.EMPTY_REPOSITORY

  let errorMessge = 'Not found'
  if (isEmptyRepository) {
    errorMessge = 'Empty repository'
  }

  let errorComponent: ReactComponentElement<any, any> = <NotFoundError />

  switch (error?.message) {
    case Errors.EMPTY_REPOSITORY:
      errorComponent = <RepositoryNotFoundError />
      break
    case Errors.REPOSITORY_SYNC_FAILED:
      errorComponent = <RepositorySyncError />
      break
    case Errors.INVALID_URL:
      errorComponent = <InvalidUrlError />
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
        <div className="text-xl font-semibold">Synchronization failed</div>
      </div>
      <div>
        Repository synchronization has failed. Please verify your repository
        connection settings or attempt to manually initiate a sync task.
      </div>
      <Link
        href="/settings/repository/git"
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
