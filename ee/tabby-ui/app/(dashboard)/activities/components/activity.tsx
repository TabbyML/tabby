'use client'

import React from 'react'
import moment, { unitOfTime } from 'moment'
import { useTheme } from 'next-themes'
import { DateRange } from 'react-day-picker'
import ReactJson from 'react-json-view'
import { useQuery } from 'urql'
import { toast } from 'sonner'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { QueryVariables } from '@/lib/tabby/gql'
import { graphql } from '@/lib/gql/generates'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Card, CardContent } from '@/components/ui/card'
import LoadingWrapper from '@/components/loading-wrapper'
import {
  IconCheck,
  IconChevronLeft,
  IconChevronRight
} from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
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

import { getLanguageColor, getLanguageDisplayName } from '@/lib/language-utils'

import type { ListUserEventsQuery } from '@/lib/gql/generates/graphql'

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

const mockData = [
  {
    type: 'Completion',
    user: 'acme@tabbyml.com',
    date: moment().subtract(1, 'second').toDate(),
    language: 'Rust',
    id: 1
  },
  {
    type: 'Select',
    user: 'bob@tabbyml.com',
    date: moment().subtract(10, 'second').toDate(),
    language: 'Typescript',
    id: 2
  },
  {
    type: 'Completion',
    user: 'bob@tabbyml.com',
    date: moment().subtract(10, 'hour').toDate(),
    language: 'Typescript',
    id: 3
  },
  {
    type: 'View',
    user: 'james@tabbyml.com',
    date: moment().subtract(25, 'hour').toDate(),
    language: 'Python',
    id: 4
  },
  {
    type: 'Select',
    user: 'kevin@tabbyml.com',
    date: moment().subtract(30, 'hour').toDate(),
    language: 'Rust',
    id: 5
  }
]

const demoJson = {
  ts: 1712150088030,
  event: {
    completion: {
      completion_id: 'cmpl-db05ba2b-4e31-475e-8ac3-381b83c47177',
      language: 'typescriptreact',
      prompt:
        "<fim_prefix>// Path: clients/tabby-agent/src/JsonLineServer.ts\n// type CancellationResponse = [\n//   id: number, // Matched request id\n//   data: boolean | null,\n// ];\n//\n// Path: clients/tabby-agent/src/utils.ts\n// function isBlank(input: string) {\n//   return input.trim().length === 0;\n// }\n//\n// Path: clients/tabby-agent/src/utils.ts\n// function isTimeoutError(error: any) {\n//   return (\n//     (error instanceof Error && error.name === \"TimeoutError\") ||\n//     (error instanceof HttpError && [408, 499].includes(error.status))\n//   );\n// }\n    name: '7 Jan',\n    IntelliJ: 34.9,\n    VSCode: 43\n  }\n]\n\nexport function Analytic() {\n  const DEAFULT_RANGE = 14\n\n  const endDate = moment().add(1, 'day').startOf('day').utc().format();\n  const starDate = moment().subtract(DEAFULT_RANGE, 'day').startOf('day').utc().format();\n\n  const [{ data, fetching }] = useQuery({\n    query: queryDailyStatsInPastYear\n  })\n  console.log(data)\n  console.log('endDate: ', endDate, \"starDate: \", starDate)\n  \n  // todo query\n  if (fetching) {<fim_suffix>}\n  return (\n    <div>\n      <AnalyticHeader />\n      <AnalyticSummary />\n      <CompletionsChartSection />\n      <div className=\"flex gap-x-5\">\n        <div className=\"flex-1\">\n          <AcceptanceChartSection />\n        </div>\n        <div style={{ flex: 3 }}>\n          <ActivityChartSection />\n        </div>\n      </div>\n\n    </div>\n  )\n}\n\nfunction AnalyticHeader() {\n<fim_middle>",
      segments: {
        prefix:
          "    name: '7 Jan',\n    IntelliJ: 34.9,\n    VSCode: 43\n  }\n]\n\nexport function Analytic() {\n  const DEAFULT_RANGE = 14\n\n  const endDate = moment().add(1, 'day').startOf('day').utc().format();\n  const starDate = moment().subtract(DEAFULT_RANGE, 'day').startOf('day').utc().format();\n\n  const [{ data, fetching }] = useQuery({\n    query: queryDailyStatsInPastYear\n  })\n  console.log(data)\n  console.log('endDate: ', endDate, \"starDate: \", starDate)\n  \n  // todo query\n  if (fetching) {",
        suffix:
          '}\n  return (\n    <div>\n      <AnalyticHeader />\n      <AnalyticSummary />\n      <CompletionsChartSection />\n      <div className="flex gap-x-5">\n        <div className="flex-1">\n          <AcceptanceChartSection />\n        </div>\n        <div style={{ flex: 3 }}>\n          <ActivityChartSection />\n        </div>\n      </div>\n\n    </div>\n  )\n}\n\nfunction AnalyticHeader() {\n'
      },
      choices: [
        {
          index: 0,
          text: '\n    return <div>Loading...</div>\n  }\n\n  if (data.length === 0) {\n    return <div>No data</div>\n  '
        }
      ]
    }
  }
}

enum DATE_OPTIONS {
  'TODAY' = 'today',
  'YESTERDAY' = 'yesterday',
  'LAST24HOURS' = '-24h',
  'LAST7DAYS' = '-7d',
  'LAST14DAYS' = '-14d',
  'CUSTOM_DATE' = 'custom_date',
  'CUSTOM_RANGE' = 'custom_range'
}

export default function Activity() {
  const [queryVariables, setQueryVariables] = React.useState<
    QueryVariables<typeof listUserEvents>
  >({
    last: DEFAULT_PAGE_SIZE,
    start: moment()
      .subtract(1, 'day')
      .utc()
      .format(),
    end: moment().utc().format()
  })
  const [{ data, error, fetching }] = useQuery({
    query: listUserEvents,
    variables: queryVariables
  })
  const [userEvents, setUserEvents] = React.useState<ListUserEventsQuery['userEvents']>()

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


  const [dateRange, setDateRange] = React.useState<DateRange>({
    from: moment()
      .add(
        parseInt(DATE_OPTIONS.LAST24HOURS, 10),
        DATE_OPTIONS.LAST24HOURS[DATE_OPTIONS.LAST24HOURS.length - 1] as 'h'
      )
      .toDate(),
    to: moment().toDate()
  })
  const [showDateFilter, setShowDateFilter] = React.useState(false)
  const [selectDateFilter, setSelectDateFilter] = React.useState<DATE_OPTIONS>(
    DATE_OPTIONS.LAST24HOURS
  )
  const [showDateRangerPicker, setShowDateRangerPicker] = React.useState(false)
  const [calendarDateRange, setCalendarDateRange] = React.useState<
    DateRange | undefined
  >({
    from: moment()
      .add(
        parseInt(DATE_OPTIONS.LAST24HOURS, 10),
        DATE_OPTIONS.LAST24HOURS[DATE_OPTIONS.LAST24HOURS.length - 1] as 'h'
      )
      .toDate(),
    to: moment().toDate()
  })
  const [showDateUntilNowPicker, setShowDateUntilNowPicker] = React.useState(false)
  const [dateUntilNow, setDateUntilNow] = React.useState<Date | undefined>(
    moment().toDate()
  )

  const onDateFilterChange = (value: DATE_OPTIONS) => {
    switch (value) {
      case DATE_OPTIONS.TODAY: {
        setDateRange({
          from: moment().startOf('day').toDate(),
          to: moment().toDate()
        })
        break
      }
      case DATE_OPTIONS.YESTERDAY: {
        setDateRange({
          from: moment().subtract(1, 'd').startOf('day').toDate(),
          to: moment().subtract(1, 'd').endOf('day').toDate()
        })
        break
      }
      default: {
        // TODO: noticed the unit for the shared component
        const unit = value[value.length - 1]
        const number = parseInt(value, 10)
        setDateRange({
          from: moment()
            .add(number, unit as unitOfTime.DurationConstructor)
            .startOf('day')
            .toDate(),
          to: moment().toDate()
        })
      }
    }
    setSelectDateFilter(value)
  }

  const onDateFilterOpenChange = (open: boolean) => {
    if (!open && !showDateRangerPicker && !showDateUntilNowPicker) {
      setShowDateFilter(false)
    }
  }

  const applyDateRange = () => {
    setShowDateFilter(false)
    setShowDateRangerPicker(false)
    setSelectDateFilter(DATE_OPTIONS.CUSTOM_RANGE)
    setDateRange(calendarDateRange!)
  }

  const applyDateUntilNow = () => {
    setShowDateFilter(false)
    setShowDateUntilNowPicker(false)
    setSelectDateFilter(DATE_OPTIONS.CUSTOM_DATE)
    setDateRange({
      from: moment(dateUntilNow).startOf('day').toDate(),
      to: moment().toDate()
    })
  }


  console.log('userEvents', userEvents)
  return (
    <LoadingWrapper loading={fetching}>
      <div className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col sm:gap-4 sm:py-4 sm:pl-14">
        <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0">
          <div className="ml-auto flex items-center gap-2">
            {/* TODO: make it as a shared component being used in the reports page as well */}
            <div className="relative">
              <Select
                value={selectDateFilter}
                onValueChange={onDateFilterChange}
                open={showDateFilter}
                onOpenChange={onDateFilterOpenChange}
              >
                <SelectTrigger
                  className="w-[240px]"
                  onClick={() => setShowDateFilter(!showDateFilter)}
                >
                  <SelectValue placeholder="Date range" />
                </SelectTrigger>
                <SelectContent align="end">
                  <SelectItem value={DATE_OPTIONS.TODAY}>Today</SelectItem>
                  <SelectItem value={DATE_OPTIONS.YESTERDAY}>
                    Yesterday
                  </SelectItem>
                  <SelectItem value={DATE_OPTIONS.LAST24HOURS}>
                    Last 24 hours
                  </SelectItem>
                  <SelectItem value={DATE_OPTIONS.LAST7DAYS}>
                    Last 7 days
                  </SelectItem>
                  <SelectItem value={DATE_OPTIONS.LAST14DAYS}>
                    Last 14 days
                  </SelectItem>
                  <SelectItem
                    value={DATE_OPTIONS.CUSTOM_DATE}
                    className="hidden"
                  >
                    {moment(dateRange?.from).format('ll')} - Now
                  </SelectItem>
                  <SelectItem
                    value={DATE_OPTIONS.CUSTOM_RANGE}
                    className="hidden"
                  >
                    {moment(dateRange?.from).format('ll')}
                    {dateRange?.to
                      ? ` - ${moment(dateRange.to).format('ll')}`
                      : ''}
                  </SelectItem>
                  <SelectSeparator />
                  <div
                    className="relative cursor-default py-1.5 pl-8 text-sm hover:bg-accent hover:text-accent-foreground"
                    onClick={() => setShowDateUntilNowPicker(true)}
                  >
                    {selectDateFilter === DATE_OPTIONS.CUSTOM_DATE && (
                      <div className="absolute inset-y-0 left-2 flex items-center">
                        <IconCheck />
                      </div>
                    )}
                    Custom date until now
                  </div>
                  <div
                    className="relative cursor-default py-1.5 pl-8 text-sm hover:bg-accent hover:text-accent-foreground"
                    onClick={() => setShowDateRangerPicker(true)}
                  >
                    {selectDateFilter === DATE_OPTIONS.CUSTOM_RANGE && (
                      <div className="absolute inset-y-0 left-2 flex items-center">
                        <IconCheck />
                      </div>
                    )}
                    Custom date range
                  </div>
                </SelectContent>
              </Select>

              <Card
                className={cn('absolute right-0 mt-1', {
                  'opacity-0 z-0 pointer-events-none': !showDateUntilNowPicker,
                  'opacity-100 pointer-events-auto': showDateUntilNowPicker
                })}
                style={(showDateUntilNowPicker && { zIndex: 99 }) || {}}
              >
                <CardContent className="w-auto pb-0">
                  <Calendar
                    initialFocus
                    mode="single"
                    selected={dateUntilNow}
                    onSelect={setDateUntilNow}
                    disabled={(date: Date) => date > new Date()}
                  />
                  <Separator />
                  <div className="flex items-center justify-end gap-x-3 py-4">
                    <Button
                      variant="ghost"
                      onClick={() => setShowDateUntilNowPicker(false)}
                    >
                      Cancel
                    </Button>
                    <Button onClick={applyDateUntilNow}>Apply</Button>
                  </div>
                </CardContent>
              </Card>

              <Card
                className={cn('absolute right-0 mt-1', {
                  'opacity-0 z-0 pointer-events-none': !showDateRangerPicker,
                  'opacity-100 pointer-events-auto': showDateRangerPicker
                })}
                style={(showDateRangerPicker && { zIndex: 99 }) || {}}
              >
                <CardContent className="w-auto pb-0">
                  <Calendar
                    initialFocus
                    mode="range"
                    defaultMonth={moment(calendarDateRange?.from)
                      .subtract(1, 'month')
                      .toDate()}
                    selected={calendarDateRange}
                    onSelect={setCalendarDateRange}
                    numberOfMonths={2}
                    disabled={(date: Date) => date > new Date()}
                  />
                  <Separator />
                  <div className="flex items-center justify-end gap-x-3 py-4">
                    <Button
                      variant="ghost"
                      onClick={() => setShowDateRangerPicker(false)}
                    >
                      Cancel
                    </Button>
                    <Button onClick={applyDateRange}>Apply</Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <Card x-chunk="dashboard-06-chunk-0" className="bg-transparent">
            <CardContent className="pb-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[25%]">Event</TableHead>
                    <TableHead className="w-[25%]">People</TableHead>
                    <TableHead className="w-[25%]">Time</TableHead>
                    <TableHead className="w-[25%]">Language</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {userEvents?.edges.map(userEvent => (
                    <ActivityRow key={userEvent.cursor} activity={userEvent.node} />
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <div className="flex w-[100px] items-center justify-center text-sm font-medium">
              Page 1
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                className="h-8 w-8 p-0"
                onClick={() => {}}
                disabled={true}
              >
                <IconChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                className="h-8 w-8 p-0"
                onClick={() => {}}
                disabled={false}
              >
                <IconChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
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
  const [isCollapse, setIsCollapse] = React.useState(false)
  const { theme } = useTheme()
  const payloadJson = JSON.parse(activity.payload) as { [key: string]: { language?: string } }

  let language = payloadJson[activity.kind.toLocaleLowerCase()]?.language
  if (language?.startsWith('typescript')) language = 'typescript'
  const languageColor = language && getLanguageColor(language) || (theme === 'dark'? '#ffffff' : '#000000')
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
              <p>Code completion showed</p>
            </TooltipContent>
          </Tooltip>
        </TableCell>
        <TableCell>{activity.userId}</TableCell>
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
