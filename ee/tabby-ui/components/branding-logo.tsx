'use client'

import React from 'react'
import { useQuery } from 'urql'

import { brandingSettingQuery } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'

interface BrandingLogoProps extends React.HTMLAttributes<HTMLImageElement> {
  defaultLogoUrl: string
  alt?: string
  width?: number
  classNames?: {
    customLogo?: string
    defaultLogo?: string
  }
}

export const BrandingLogo = ({
  defaultLogoUrl,
  alt = 'logo',
  className,
  classNames,
  ...props
}: BrandingLogoProps) => {
  const [{ data }] = useQuery({ query: brandingSettingQuery })

  // todo fix loading skeleton height
  // if (isLoading) {
  //   // placeholder
  //   return <div {...props} className={className} />
  // }
  const logoSrc = data?.brandingSetting?.brandingLogo

  return (
    <img
      src={logoSrc ?? defaultLogoUrl}
      alt={alt}
      className={cn(
        className,
        logoSrc ? classNames?.customLogo : classNames?.defaultLogo
      )}
      {...props}
    />
  )
}

export const BrandingIcon = ({
  defaultLogoUrl,
  alt = 'logo',
  className,
  classNames,
  ...props
}: BrandingLogoProps) => {
  const [{ data }] = useQuery({ query: brandingSettingQuery })

  // todo fix loading skeleton height
  // if (isLoading) {
  //   // placeholder
  //   return <div {...props} className={className} />
  // }
  const logoSrc = data?.brandingSetting?.brandingIcon

  return (
    <img
      src={logoSrc ?? defaultLogoUrl}
      alt={alt}
      className={cn(
        className,
        logoSrc ? classNames?.customLogo : classNames?.defaultLogo
      )}
      {...props}
    />
  )
}
