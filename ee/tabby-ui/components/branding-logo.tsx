'use client'

import React from 'react'
import useSWRImmutable from 'swr/immutable'
import fetcher from '@/lib/tabby/fetcher'
import { cn } from '@/lib/utils'
import { mutate } from 'swr'

const NOT_FOUND_ERROR = 'not_found'
let hasCustomLogo = true

export const mutateBrandingLogo = (url: string) => {
  hasCustomLogo = true
  mutate(url)
}

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
    error
  } = useSWRImmutable(customLogoUrl, (url: string) => {
    if (!customLogoUrl || !hasCustomLogo) return undefined

    return fetcher(url, {
      responseFormatter: async response => {
        const blob = await response.blob()
        const buffer = Buffer.from(await blob.arrayBuffer())
        hasCustomLogo = true
        return `data:image/png;base64,${buffer.toString('base64')}`
      },
      errorHandler: response => {
        if (response.status === 404) throw new Error(NOT_FOUND_ERROR)
        return undefined
      }
    })
  })

  if (error?.message === NOT_FOUND_ERROR) {
    hasCustomLogo = false
  }

  // todo fix height mutate
  if (isLoading) {
    // placeholder
    return <div {...props} className={className} />
  }

  return <img
    src={avatarImageSrc ?? defaultLogoUrl}
    alt={alt}
    className={cn(
      className,
      avatarImageSrc ? classNames?.customLogo : classNames?.defaultLogo
    )}
    {...props}
  />
}
