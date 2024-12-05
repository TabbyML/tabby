'use client'

import React, { ReactNode } from 'react'
import NiceAvatar, { genConfig } from 'react-nice-avatar'
import { mutate } from 'swr'
import useSWRImmutable from 'swr/immutable'

import { useMe } from '@/lib/hooks/use-me'
import fetcher from '@/lib/tabby/fetcher'
import { cn } from '@/lib/utils'
import {
  Avatar as AvatarComponent,
  AvatarFallback,
  AvatarImage
} from '@/components/ui/avatar'
import { Skeleton } from '@/components/ui/skeleton'

const NOT_FOUND_ERROR = 'not_found'
const failedAvatarUserIds: Set<string> = new Set()

export const mutateAvatar = (userId: string) => {
  failedAvatarUserIds.delete(userId)
  mutate(`/avatar/${userId}`)
}

interface UserAvatarProps {
  className?: string
  fallback?: string | ReactNode
  user:
    | {
        id: string
        email: string
      }
    | undefined
}
export function UserAvatar({ user, className, fallback }: UserAvatarProps) {
  const userId = user?.id
  const avatarUrl = userId ? `/avatar/${userId}` : null

  const {
    data: avatarImageSrc,
    isLoading,
    error
  } = useSWRImmutable(avatarUrl, (url: string) => {
    if (!userId || failedAvatarUserIds.has(userId)) return undefined

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

  const avatarConfigFromEmail = React.useMemo(() => {
    if (!user?.email) return undefined
    return genConfig(user.email)
  }, [user?.email])

  if (isLoading) {
    return <Skeleton className={cn('h-16 w-16 rounded-full', className)} />
  }

  if (error?.message === NOT_FOUND_ERROR && userId) {
    failedAvatarUserIds.add(userId)
  }

  if (!avatarImageSrc && !avatarConfigFromEmail && fallback) return fallback
  if (!avatarImageSrc && avatarConfigFromEmail) {
    return (
      <NiceAvatar
        className={cn('h-16 w-16', className)}
        {...avatarConfigFromEmail}
      />
    )
  }

  return (
    <AvatarComponent className={cn('h-16 w-16', className)}>
      <AvatarImage
        src={avatarImageSrc}
        alt={user?.email}
        className="object-cover"
      />
      <AvatarFallback>{user?.email?.substring(0, 2)}</AvatarFallback>
    </AvatarComponent>
  )
}

interface MyAvatarProps extends Omit<UserAvatarProps, 'user'> {}
export function MyAvatar(props: MyAvatarProps) {
  const [{ data }] = useMe()

  return <UserAvatar user={data?.me} {...props} />
}
