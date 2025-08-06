'use client'

import React from 'react'
import useSWRImmutable from 'swr/immutable'
import fetcher from '@/lib/tabby/fetcher'
import { cn } from '@/lib/utils'

const NOT_FOUND_ERROR = 'not_found'
interface BrandingLogoProps extends React.HTMLAttributes<HTMLImageElement> {
  customLogoUrl: string
  defaultLogoUrl: string
  alt?: string
  width?: number
  classNames?: {
    customLogo?: string
    defaultLogo?: string
  }
}

export const BrandingLogo = ({
  customLogoUrl,
  defaultLogoUrl,
  alt = 'logo',
  className,
  classNames,
  ...props
}: BrandingLogoProps) => {

  const {
    data: avatarImageSrc,
    isLoading,
  } = useSWRImmutable(customLogoUrl, (url: string) => {
    if (!customLogoUrl) return undefined

    return fetcher(url, {
      responseFormatter: async response => {
        const blob = await response.blob()
        const buffer = Buffer.from(await blob.arrayBuffer())
        return `data:image/png;base64,${buffer.toString('base64')}`
      },
      errorHandler: response => {
        if (response.status === 404) throw new Error(NOT_FOUND_ERROR)
        return undefined
      }
    })
  })

  if (isLoading) {
    // placeholder
    return <div {...props} className={className} />
  }

  return <img
    src={avatarImageSrc ?? defaultLogoUrl}
    alt={alt}
    {...props}
    className={cn(
      className,
      avatarImageSrc ? classNames?.customLogo : classNames?.defaultLogo
    )}
  />
}
