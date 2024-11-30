'use client'

import { HTMLAttributes } from 'react'
import Link from 'next/link'
import { TabsContent } from '@radix-ui/react-tabs'
import moment from 'moment'

import { cn } from '@/lib/utils'

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

const notifications_unread = [
  {
    title: 'License expired',
    content:
      'Your enterprise license has expired, with validity until October 1, 2024. Please contact us to renew your subscription.',
    date: '2024-11-25T08:50:12.395Z',
    read: false
  },
  {
    title: 'Third party integration job failed',
    content:
      'The job for integrating with github repository `tabby` failed. Please check the logs or configuration for more details.',
    date: '2024-10-25T08:50:12.395Z',
    read: false
  },
  {
    title: 'Indexing job failed',
    content:
      'The job for indexing the repository `tabby` failed. Please check the logs for more details.',
    date: '2024-09-23T08:50:12.395Z',
    read: false
  },
  {
    title: 'License about to expired',
    content:
      'Your enterprise license is about to expire on October 1, 2023. Please contact us to renew your subscription.',
    date: '2023-09-20T08:50:12.395Z',
    read: false
  }
]

const notifications_all = [
  {
    title: 'Third party integration job failed',
    content:
      'The job for integrating with github repository `tabby` failed. Please check the logs or configuration for more details.',
    date: '2024-10-25T08:50:12.395Z',
    read: true
  },
  {
    title: 'License expired',
    content:
      'Your enterprise license has expired, with validity until October 1, 2024. Please contact us to renew your subscription.',
    date: '2024-11-25T08:50:12.395Z',
    read: false
  },
  {
    title: 'Third party integration job failed',
    content:
      'The job for integrating with github repository `tabby` failed. Please check the logs or configuration for more details.',
    date: '2024-10-25T08:50:12.395Z',
    read: false
  },
  {
    title: 'Indexing job failed',
    content:
      'The job for indexing the repository `tabby` failed. Please check the logs for more details.',
    date: '2024-09-23T08:50:12.395Z',
    read: false
  },
  {
    title: 'License about to expired',
    content:
      'Your enterprise license is about to expire on October 1, 2023. Please contact us to renew your subscription.',
    date: '2023-09-20T08:50:12.395Z',
    read: false
  }
]

export function NotificationBox({ className, ...rest }: Props) {
  const hasUnreadNotificatiosn = true

  return (
    <div className={cn(className)} {...rest}>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="relative">
            <IconBell />
            {hasUnreadNotificatiosn && (
              <div className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-red-400"></div>
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="bottom"
          align="end"
          className="flex w-96 flex-col overflow-hidden p-0"
          style={{ maxHeight: 'calc(100vh - 6rem)' }}
        >
          <div className="flex items-center justify-between px-4 py-2">
            <div className="text-sm font-medium">Nofitications</div>
            <Button size="sm" className="h-6 py-1 text-xs">
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
            <TabsContent value="unread">
              <NotificationList type="unread" />
            </TabsContent>
            <TabsContent value="all">
              <NotificationList type="all" />
            </TabsContent>
          </Tabs>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

function NotificationList({ type }: { type: 'unread' | 'all' }) {
  const list = type === 'unread' ? notifications_unread : notifications_all
  const len = list.length
  return (
    <div className="mt-4 space-y-2">
      {list.map((item, index) => {
        return (
          <div key={`${type}_${index}`}>
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
  data: {
    read: boolean
    title: string
    date: string
    content: string
  }
}

function NotificationItem({ data }: NotificationItemProps) {
  return (
    <div className="group space-y-1.5">
      <div className="flex cursor-pointer items-center gap-1.5 overflow-hidden text-sm font-medium">
        {!data.read && (
          <span className="h-2 w-2 shrink-0 rounded-full bg-red-400"></span>
        )}
        <span className="flex-1 truncate group-hover:opacity-70">
          {data.title}
        </span>
      </div>
      <div className="line-clamp-3 cursor-pointer text-sm text-muted-foreground group-hover:opacity-70">
        {data.content}
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="text-muted-foreground">
          {formatNotificationTime(data.date)}
        </span>
        <div className="flex items-center gap-1.5">
          <Link
            href="/"
            className="flex items-center gap-0.5 p-1 font-medium underline-offset-4 hover:underline"
          >
            <IconArrowRight className="h-3 w-3" />
            Detail
          </Link>
          {!data.read && (
            <Button
              variant="link"
              className="flex h-auto items-center gap-0.5 p-1 text-xs text-muted-foreground"
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
