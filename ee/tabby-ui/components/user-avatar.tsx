'use client'

import NiceAvatar, { genConfig } from 'react-nice-avatar'
import { mutate } from 'swr';
import useSWRImmutable from 'swr/immutable'

import { useMe } from '@/lib/hooks/use-me'
import fetcher from '@/lib/tabby/fetcher'
import { cn } from '@/lib/utils';

import {
  Avatar as AvatarComponent,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar"
import { Skeleton } from '@/components/ui/skeleton'

export function UserAvatar ({
  className
}: {
  className?: string;
}) {
  const [{ data }] = useMe()
  const avatarUrl = !data?.me?.email ? null : `/avatar/${data.me.id}`
  const {
    data: avatarImageSrc,
    isLoading
  } = useSWRImmutable(avatarUrl, (url: string) => {
    return fetcher(url, {
      responseFormatter: async response => {
        if (!response.ok) return undefined
        const blob = await response.blob()
        const buffer = Buffer.from(await blob.arrayBuffer())
        return `data:image/png;base64,${buffer.toString('base64')}`
      }
    })
  })

  if (!data?.me?.email) return null

  if (isLoading) {
    return (
      <Skeleton className={cn('h-16 w-16 rounded-full', className)} />
    )
  }
  
  if (!avatarImageSrc) {
    const config = genConfig(data.me.email)
    return (
      <NiceAvatar className={cn("h-16 w-16", className)} {...config} />
    )
  }

  return (
    <AvatarComponent className={cn("h-16 w-16", className)}>
      <AvatarImage src={avatarImageSrc} alt={data.me.email} className="object-cover" />
      <AvatarFallback>{data.me?.email.substring(0, 2)}</AvatarFallback>
    </AvatarComponent>
  )
}

export const mutateAvatar = (userId: string) => {
  mutate(`/avatar/${userId}`)
}