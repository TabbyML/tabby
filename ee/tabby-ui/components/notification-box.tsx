'use client'

import { HTMLAttributes, useMemo } from 'react'
import { TabsContent } from '@radix-ui/react-tabs'
import { AnimatePresence, motion } from 'framer-motion'
import moment from 'moment'
import useSWR from 'swr'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { NotificationsQuery } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { notificationsQuery } from '@/lib/tabby/query'
import { ArrayElementType } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconBell, IconCheck } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import LoadingWrapper from '@/components/loading-wrapper'
import { MemoizedReactMarkdown } from '@/components/markdown'
import { ListSkeleton } from '@/components/skeleton'

interface Props extends HTMLAttributes<HTMLDivElement> {}

const markNotificationsReadMutation = graphql(/* GraphQL */ `
  mutation markNotificationsRead($notificationId: ID) {
    markNotificationsRead(notificationId: $notificationId)
  }
`)

export function NotificationBox({ className, ...rest }: Props) {
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: notificationsQuery
  })

  useSWR('refresh_notifications', () => reexecuteQuery(), {
    revalidateOnFocus: true,
    revalidateOnReconnect: true,
    revalidateOnMount: false,
    refreshInterval: 1000 * 60 * 10 // 10 mins
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
            className="relative my-2 flex-1 overflow-y-auto px-5"
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
      <AnimatePresence>
        {notifications?.map((item, index) => {
          return (
            <motion.div layout key={item.id}>
              <NotificationItem data={item} />
              <Separator
                className={cn('my-3', {
                  hidden: index === len - 1
                })}
              />
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}

interface NotificationItemProps extends HTMLAttributes<HTMLDivElement> {
  data: ArrayElementType<NotificationsQuery['notifications']>
}

function NotificationItem({ data }: NotificationItemProps) {
  const markNotificationsRead = useMutation(markNotificationsReadMutation)

  const onClickMarkRead = () => {
    markNotificationsRead({
      notificationId: data.id
    })
  }

  return (
    <div className="space-y-1.5">
      <MemoizedReactMarkdown
        className={cn(
          'prose max-w-none break-words text-sm dark:prose-invert prose-p:my-1 prose-p:leading-relaxed',
          { 'unread-notification': !data.read }
        )}
        components={{
          a: props => (
            <a
              {...props}
              onClick={e => {
                onClickMarkRead()
                props.onClick?.(e)
              }}
            />
          )
        }}
      >
        {data.content}
      </MemoizedReactMarkdown>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="text-muted-foreground">
          {formatNotificationTime(data.createdAt)}
        </span>
        <div className="flex items-center gap-1.5">
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

  return {
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
