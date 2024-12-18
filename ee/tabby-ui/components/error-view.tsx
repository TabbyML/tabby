'use client'

import { ReactNode } from 'react'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

export function ErrorView({
  title,
  description,
  className,
  children,
  statusText,
  hideChildren
}: {
  title?: string
  description?: string
  className?: string
  hideChildren?: boolean
  children?: ReactNode
  statusText?: string
}) {
  let defaultTitle = 'Something went wrong'
  let defaultDescription = 'Oops! Please try again later.'
  let displayTitle = ''
  let displayDescription = description || defaultDescription

  switch (statusText) {
    case 'Too Many Requests':
      displayTitle = 'Too Many Requests'
      break
    case 'Bad Request':
      displayTitle = 'Bad Request'
      break
    default:
      displayTitle = title || defaultTitle
  }

  return (
    <div className={cn('mx-auto mt-8 max-w-md text-center', className)}>
      <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
        {displayTitle}
      </h1>
      {!!displayDescription && (
        <p className="mt-4 text-muted-foreground">{displayDescription}</p>
      )}
      {!hideChildren && (
        <div>
          {children ? (
            children
          ) : (
            <Button
              className={cn('mt-6')}
              onClick={e => window.location.reload()}
            >
              Refresh
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
