'use client'

import { useState } from 'react'
import chroma from 'chroma-js'
import moment from 'moment'
import { useTheme } from 'next-themes'
import { DateRange } from 'react-day-picker'
import SyntaxHighlighter from 'react-syntax-highlighter'
import {
  tomorrow,
  tomorrowNightEighties
} from 'react-syntax-highlighter/dist/esm/styles/hljs'

import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import DatePickerWithRange from '@/components/ui/date-range-picker'
import { IconChevronLeft, IconChevronRight } from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

// TODO
import languageColors from '../../../(home)/language-colors.json'

const getLanguageColorMap = (): Record<string, string> => {
  return Object.entries(languageColors).reduce((acc, cur) => {
    const [lan, color] = cur
    return { ...acc, [lan.toLocaleLowerCase()]: color }
  }, {})
}

export default function Activity() {
  const [dateRange, setDateRange] = useState<DateRange>({
    from: moment().subtract(1, 'day').toDate(),
    to: moment().toDate()
  })
  const [showDateSelector, setShowDateSelector] = useState(false)
  const { theme } = useTheme()
  const data = [
    {
      type: 'completion',
      user: 'acme@tabbyml.com',
      date: moment().subtract(1, 'second').toDate(),
      language: 'rust'
    },
    {
      type: 'select',
      user: 'bob@tabbyml.com',
      date: moment().subtract(10, 'second').toDate(),
      language: 'typescript'
    },
    {
      type: 'completion',
      user: 'bob@tabbyml.com',
      date: moment().subtract(10, 'hour').toDate(),
      language: 'typescript'
    },
    {
      type: 'views',
      user: 'james@tabbyml.com',
      date: moment().subtract(25, 'hour').toDate(),
      language: 'python'
    },
    {
      type: 'select',
      user: 'kevin@tabbyml.com',
      date: moment().subtract(30, 'hour').toDate(),
      language: 'rust'
    }
  ]
  const colorMap = getLanguageColorMap()
  const demoCode = `pub async fn setting() -> Json<ServerSetting> {
      let config = ServerSetting {
          disable_client_side_telemetry: false,
      };
      Json(config)
  }`

  const onDateOpenChange = (
    isOpen: boolean,
    dateRange: DateRange | undefined
  ) => {
    if (!isOpen) {
      if (dateRange) {
        setDateRange(dateRange)
      }
    }
  }

  const onDateRangeFilterChange = (value: string) => {
    if (value === 'custom') {
      setShowDateSelector(true)
    } else {
      if (showDateSelector) setShowDateSelector(false)
    }
  }

  return (
    <div className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col sm:gap-4 sm:py-4 sm:pl-14">
        <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0">
          <div className="ml-auto flex items-center gap-2">
            {showDateSelector && (
              <DatePickerWithRange
                buttonClassName="h-9"
                contentAlign="end"
                dateRange={dateRange}
                onOpenChange={onDateOpenChange}
              />
            )}
            <Select
              defaultValue="past1hour"
              onValueChange={onDateRangeFilterChange}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Date range" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="past1hour">Past 1 hour</SelectItem>
                <SelectItem value="past24hour">Past 24 hours</SelectItem>
                <SelectItem value="past3days">Past 3 days</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Card x-chunk="dashboard-06-chunk-0" className="bg-transparent">
            <CardContent className="pb-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Event</TableHead>
                    <TableHead>People</TableHead>
                    <TableHead>Time</TableHead>
                    <TableHead>Language</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.map((item, idx) => {
                    const color = colorMap[item.language]
                    const bgColor = chroma(color).alpha(0.8).css()
                    return (
                      <TableRow key={idx}>
                        <TableCell className="font-medium">
                          <Popover>
                            <PopoverTrigger className="transition-colors hover:text-primary">
                              {item.type}
                            </PopoverTrigger>
                            <PopoverContent className="p-4" align="start">
                              <div>
                                <h4 className="mb-1 text-sm leading-none text-secondary-foreground">
                                  Event:
                                </h4>
                                <p className="mb-3">Code completion showed</p>
                                <h4 className="mb-1 text-sm  leading-none text-secondary-foreground">
                                  Prompt:
                                </h4>
                                <SyntaxHighlighter
                                  class="my-2 rounded"
                                  language="rust"
                                  style={
                                    theme === 'dark'
                                      ? tomorrowNightEighties
                                      : tomorrow
                                  }
                                >
                                  {demoCode}
                                </SyntaxHighlighter>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </TableCell>
                        <TableCell>{item.user}</TableCell>
                        <TableCell>
                          {moment(item.date).isBefore(
                            moment().subtract(1, 'days')
                          )
                            ? moment(item.date).format('YYYY-MM-DD HH:mm')
                            : moment(item.date).fromNow()}
                        </TableCell>
                        <TableCell>
                          <p
                            className="inline-block rounded-full px-2 py-1 text-xs font-bold leading-none text-white"
                            style={{ backgroundColor: bgColor }}
                          >
                            {item.language}
                          </p>
                        </TableCell>
                      </TableRow>
                    )
                  })}
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
                <span className="sr-only">Go to previous page</span>
                <IconChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                className="h-8 w-8 p-0"
                onClick={() => {}}
                disabled={false}
              >
                <span className="sr-only">Go to next page</span>
                <IconChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
