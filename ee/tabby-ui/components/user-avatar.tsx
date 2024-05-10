'use client'

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
let shouldFetchAvatar = true

export function UserAvatar({ className }: { className?: string }) {
  const [{ data }] = useMe()
  const userId = data?.me.id

  const avatarUrl = (userId && `/avatar/${data.me.id}`) || null
  const {
    data: avatarImageSrc,
    isLoading,
    error
  } = useSWRImmutable(avatarUrl, (url: string) => {
    if (!shouldFetchAvatar) return undefined
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

  if (!userId) return null

  if (isLoading) {
    return <Skeleton className={cn('h-16 w-16 rounded-full', className)} />
  }

  if (error?.message === NOT_FOUND_ERROR) {
    shouldFetchAvatar = false
  }

  if (!avatarImageSrc) {
    const config = genConfig(data.me.email)
    return <NiceAvatar className={cn('h-16 w-16', className)} {...config} />
  }

  return (
    <AvatarComponent className={cn('h-16 w-16', className)}>
      <AvatarImage
        src={avatarImageSrc}
        alt={data.me.email}
        className="object-cover"
      />
      <AvatarFallback>{data.me?.email.substring(0, 2)}</AvatarFallback>
    </AvatarComponent>
  )
}

export const mutateAvatar = (userId: string) => {
  shouldFetchAvatar = true
  mutate(`/avatar/${userId}`)
}
