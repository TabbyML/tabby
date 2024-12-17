'use client'

import { HTMLAttributes, useMemo } from 'react'
import { TabsContent } from '@radix-ui/react-tabs'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { NotificationsQuery } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { notificationsQuery } from '@/lib/tabby/query'
import { ArrayElementType } from '@/lib/types'
import { cn } from '@/lib/utils'

import LoadingWrapper from './loading-wrapper'
import { ListSkeleton } from './skeleton'
import { Button } from './ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger
} from './ui/dropdown-menu'
import { IconArrowRight, IconBell, IconCheck } from './ui/icons'
import { Separator } from './ui/separator'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'

interface Props extends HTMLAttributes<HTMLDivElement> {}

const markNotificationsReadMutation = graphql(/* GraphQL */ `
  mutation markNotificationsRead($notificationId: ID) {
    markNotificationsRead(notificationId: $notificationId)
  }
`)

export function NotificationBox({ className, ...rest }: Props) {
  const [{ data, fetching }] = useQuery({
    query: notificationsQuery
  })

  const notifications = useMemo(() => {
    return data?.notifications.slice().reverse()
  }, [data?.notifications])

  const unreadNotifications = useMemo(() => {
    return notifications?.filter(o => !o.read) ?? []
  }, [notifications])
  const hasUnreadNotification = unreadNotifications.length > 0

  const markNotificationsRead = useMutation(markNotificationsReadMutation)
  const onClickMarkAllRead = () => {
    markNotificationsRead()
  }

  return (
    <div className={cn(className)} {...rest}>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="relative">
            <IconBell />
            {hasUnreadNotification && (
              <div className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-red-400"></div>
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="bottom"
          align="end"
          className="flex w-96 flex-col overflow-hidden p-0"
          style={{ maxHeight: '60vh' }}
        >
          <div className="flex items-center justify-between px-4 py-2">
            <div className="text-sm font-medium">Nofitications</div>
            <Button
              size="sm"
              className="h-6 py-1 text-xs"
              onClick={onClickMarkAllRead}
              disabled={!hasUnreadNotification}
            >
              Mark all as read
            </Button>
          </div>
          <Separator />
          <Tabs
            className="relative my-2 flex-1 overflow-y-auto px-4"
            defaultValue="unread"
          >
            <TabsList className="sticky top-0 z-10 grid w-full grid-cols-2">
              <TabsTrigger value="unread">Unread</TabsTrigger>
              <TabsTrigger value="all">All</TabsTrigger>
            </TabsList>
            <TabsContent value="unread" className="mt-4">
              <LoadingWrapper loading={fetching} fallback={<ListSkeleton />}>
                <NotificationList
                  type="unread"
                  notifications={unreadNotifications}
                />
              </LoadingWrapper>
            </TabsContent>
            <TabsContent value="all" className="mt-4">
              <LoadingWrapper loading={fetching} fallback={<ListSkeleton />}>
                <NotificationList type="all" notifications={notifications} />
              </LoadingWrapper>
            </TabsContent>
          </Tabs>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

function NotificationList({
  notifications,
  type
}: {
  notifications: NotificationsQuery['notifications'] | undefined
  type: 'unread' | 'all'
}) {
  const len = notifications?.length ?? 0

  if (!len) {
    return (
      <div className="my-4 text-center text-sm text-muted-foreground">
        {type === 'unread' ? 'No unread notifications' : 'No notifications'}
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {notifications?.map((item, index) => {
        return (
          <div key={item.id}>
            <NotificationItem data={item} />
            <Separator
              className={cn('my-3', {
                hidden: index === len - 1
              })}
            />
          </div>
        )
      })}
    </div>
  )
}

interface NotificationItemProps extends HTMLAttributes<HTMLDivElement> {
  data: ArrayElementType<NotificationsQuery['notifications']>
}

function NotificationItem({ data }: NotificationItemProps) {
  const { type, title, content } = resolveNotification(data.content)

  const markNotificationsRead = useMutation(markNotificationsReadMutation)

  const onClickMarkRead = () => {
    markNotificationsRead({
      notificationId: data.id
    })
  }

  const onAction = () => {
    onClickMarkRead()

    if (type === 'license_will_expire') {
      return window.open('/settings/subscription')
    }
  }

  return (
    <div className="group space-y-1.5">
      <div className="space-y-1.5" onClick={onAction}>
        <div className="flex cursor-pointer items-center gap-1.5 overflow-hidden text-sm font-medium">
          {!data.read && (
            <span className="h-2 w-2 shrink-0 rounded-full bg-red-400"></span>
          )}
          <span className="flex-1 truncate group-hover:opacity-70">
            {title}
          </span>
        </div>
        <div className="cursor-pointer whitespace-pre-wrap break-words text-sm text-muted-foreground group-hover:opacity-70">
          {content}
        </div>
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="text-muted-foreground">
          {formatNotificationTime(data.createdAt)}
        </span>
        <div className="flex items-center gap-1.5">
          <Button
            variant="link"
            className="flex h-auto items-center gap-0.5 p-1 text-xs text-muted-foreground"
            onClick={onAction}
          >
            <IconArrowRight className="h-3 w-3" />
            Detail
          </Button>
          {!data.read && (
            <Button
              variant="link"
              className="flex h-auto items-center gap-0.5 p-1 text-xs text-muted-foreground"
              onClick={onClickMarkRead}
            >
              <IconCheck className="h-3 w-3" />
              Mark as read
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

function resolveNotification(content: string) {
  // use first line as title
  const title = content.split('\n')[0]
  const _content = content.split('\n').slice(1).join('\n')

  if (content.startsWith('Your license will expire')) {
    return {
      type: 'license_will_expire',
      title,
      content: _content
    }
  }

  return {
    type: '',
    title,
    content: _content
  }
}

// Nov 21, 2022, 7:03 AM
// Nov 21, 7:03 AM
function formatNotificationTime(time: string) {
  const targetTime = moment(time)

  if (targetTime.isBefore(moment().subtract(1, 'year'))) {
    const timeText = targetTime.format('MMM D, YYYY, h:mm A')
    return timeText
  }

  if (targetTime.isBefore(moment().subtract(1, 'month'))) {
    const timeText = targetTime.format('MMM D, hh:mm A')
    return `${timeText}`
  }

  return `${targetTime.fromNow()}`
}
