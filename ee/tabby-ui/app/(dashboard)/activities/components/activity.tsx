'use client'

import React from 'react'
import moment from 'moment'
import { useTheme } from 'next-themes'
import { DateRange } from 'react-day-picker'
import ReactJson from 'react-json-view'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { EventKind, ListUserEventsQuery } from '@/lib/gql/generates/graphql'
import { useAllMembers } from '@/lib/hooks/use-all-members'
import { getLanguageColor, getLanguageDisplayName } from '@/lib/language-utils'
import { QueryVariables } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import {
  IconChevronLeft,
  IconChevronRight,
  IconFileSearch
} from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import DateRangePicker from '@/components/date-range-picker'
import LoadingWrapper from '@/components/loading-wrapper'

const DEFAULT_DATE_RANGE = '-24h'

export const listUserEvents = graphql(/* GraphQL */ `
  query ListUserEvents(
    $after: String
    $before: String
    $first: Int
    $last: Int
    $start: DateTimeUtc!
    $end: DateTimeUtc!
  ) {
    userEvents(
      after: $after
      before: $before
      first: $first
      last: $last
      start: $start
      end: $end
    ) {
      edges {
        node {
          id
          userId
          createdAt
          kind
          payload
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

export default function Activity() {
  const defaultFromDate = moment().add(parseInt(DEFAULT_DATE_RANGE, 10), 'day')
  const defaultToDate = moment()

  const [dateRange, setDateRange] = React.useState<DateRange>({
    from: defaultFromDate.toDate(),
    to: defaultToDate.toDate()
  })
  const [userEvents, setUserEvents] =
    React.useState<ListUserEventsQuery['userEvents']>()
  const [page, setPage] = React.useState(1)

  const [queryVariables, setQueryVariables] = React.useState<
    Omit<QueryVariables<typeof listUserEvents>, 'start' | 'end'>
  >({
    last: DEFAULT_PAGE_SIZE
  })

  const [{ data, error, fetching }] = useQuery({
    query: listUserEvents,
    variables: {
      start: moment(dateRange.from!).utc().format(),
      end: dateRange.to
        ? moment(dateRange.to).utc().format()
        : moment(dateRange.from!).utc().format(),
      ...queryVariables
    }
  })

  React.useEffect(() => {
    if (data?.userEvents.edges.length) {
      setUserEvents(data.userEvents)
    }
  }, [data])

  React.useEffect(() => {
    if (error?.message) {
      toast.error(error.message)
    }
  }, [error])

  const updateDateRange = (range: DateRange) => {
    setDateRange(range)
    setPage(1)
    setQueryVariables({ last: DEFAULT_PAGE_SIZE })
  }

  return (
    <LoadingWrapper loading={fetching}>
      <div className="flex w-full flex-col">
        <div className="flex flex-col sm:gap-4 sm:py-4 sm:pl-14">
          <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0">
            <div className="ml-auto flex items-center gap-2">
              <DateRangePicker
                options={[
                  { label: 'Last 24 hours', value: '-24h' },
                  { label: 'Last 7 days', value: '-7d' },
                  { label: 'Last 14 days', value: '-14d' }
                ]}
                defaultValue={DEFAULT_DATE_RANGE}
                onSelect={updateDateRange}
                hasToday
                hasYesterday
              />
            </div>

            <Card x-chunk="dashboard-06-chunk-0" className="bg-transparent">
              {(!data?.userEvents.edges ||
                data?.userEvents.edges.length === 0) && (
                <CardContent className="flex flex-col items-center py-40 text-sm">
                  <IconFileSearch className="mb-2 h-10 w-10" />
                  <p className="font-semibold">
                    No data available for the chosen dates
                  </p>
                  <p className="text-muted-foreground">
                    Please try a different date range
                  </p>
                </CardContent>
              )}

              {data?.userEvents.edges && data?.userEvents.edges.length > 0 && (
                <>
                  <CardContent className="w-[calc(100vw-4rem)] overflow-x-auto pb-0 md:w-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="md:w-[25%]">Event</TableHead>
                          <TableHead className="md:w-[25%]">People</TableHead>
                          <TableHead className="md:w-[25%]">Time</TableHead>
                          <TableHead className="md:w-[25%]">Language</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {userEvents?.edges
                          .sort(
                            (a, b) =>
                              new Date(b.node.createdAt).getTime() -
                              new Date(a.node.createdAt).getTime()
                          )
                          .map(userEvent => (
                            <ActivityRow
                              key={userEvent.cursor}
                              activity={userEvent.node}
                            />
                          ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </>
              )}
            </Card>

            {(data?.userEvents.pageInfo?.hasNextPage ||
              data?.userEvents.pageInfo?.hasPreviousPage) && (
              <div className="flex justify-end">
                <div className="flex w-[100px] items-center justify-center text-sm font-medium">
                  Page {page}
                </div>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    className="h-8 w-8 p-0"
                    disabled={!data?.userEvents.pageInfo?.hasNextPage}
                    onClick={e => {
                      setQueryVariables({
                        first: DEFAULT_PAGE_SIZE,
                        after: data?.userEvents.pageInfo?.endCursor
                      })
                      setPage(page - 1)
                    }}
                  >
                    <IconChevronLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    className="h-8 w-8 p-0"
                    disabled={!data?.userEvents.pageInfo?.hasPreviousPage}
                    onClick={e => {
                      setQueryVariables({
                        last: DEFAULT_PAGE_SIZE,
                        before: data?.userEvents.pageInfo?.startCursor
                      })
                      setPage(page + 1)
                    }}
                  >
                    <IconChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </main>
        </div>
      </div>
    </LoadingWrapper>
  )
}

function ActivityRow({
  activity
}: {
  activity: ListUserEventsQuery['userEvents']['edges'][0]['node']
}) {
  const { theme } = useTheme()
  const [members] = useAllMembers()
  const [isCollapse, setIsCollapse] = React.useState(false)

  const payloadJson = JSON.parse(activity.payload) as {
    [key: string]: { language?: string }
  }
  let language = payloadJson[activity.kind.toLocaleLowerCase()]?.language
  if (language?.startsWith('typescript')) language = 'typescript'
  const languageColor =
    (language && getLanguageColor(language)) ||
    (theme === 'dark' ? '#ffffff' : '#000000')

  let tooltip = ''
  switch (activity.kind) {
    case EventKind.Completion: {
      tooltip = 'Code completion supplied'
      break
    }

    case EventKind.Dismiss: {
      tooltip = 'Code completion viewed but not used'
      break
    }
    case EventKind.Select: {
      tooltip = 'Code completion accepted and inserted'
      break
    }
    case EventKind.View: {
      tooltip = 'Code completion shown in editor'
      break
    }
  }
  return (
    <>
      <TableRow
        key={`${activity.id}}-1`}
        className="cursor-pointer"
        onClick={() => setIsCollapse(!isCollapse)}
      >
        <TableCell className="font-medium">
          <Tooltip>
            <TooltipTrigger>{activity.kind}</TooltipTrigger>
            <TooltipContent>
              <p>{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TableCell>
        <TableCell>
          {members.find(user => user.id === activity.userId)?.email ||
            activity.userId}
        </TableCell>
        <TableCell>
          {moment(activity.createdAt).isBefore(moment().subtract(1, 'days'))
            ? moment(activity.createdAt).format('YYYY-MM-DD HH:mm')
            : moment(activity.createdAt).fromNow()}
        </TableCell>
        <TableCell>
          <div className="flex items-center text-xs">
            <div
              className="mr-1.5 h-2 w-2 rounded-full"
              style={{ backgroundColor: languageColor }}
            />
            {getLanguageDisplayName(language)}
          </div>
        </TableCell>
      </TableRow>

      {isCollapse && (
        <TableRow key={`${activity.id}-2`} className="w-full bg-muted/30">
          <TableCell className="font-medium" colSpan={4}>
            <ReactJson
              src={payloadJson}
              name={false}
              collapseStringsAfterLength={50}
              theme={theme === 'dark' ? 'tomorrow' : 'rjv-default'}
              style={theme === 'dark' ? { background: 'transparent' } : {}}
            />
          </TableCell>
        </TableRow>
      )}
    </>
  )
}
