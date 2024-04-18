'use client'

import { useState } from 'react'
import moment from 'moment'
import { useTheme } from 'next-themes'
import SyntaxHighlighter from 'react-syntax-highlighter'
import { tomorrowNightEighties } from 'react-syntax-highlighter/dist/esm/styles/hljs'

import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { IconChevronLeft, IconChevronRight } from '@/components/ui/icons'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

// TODO
import languageColors from '../../../(home)/language-colors.json'

const getLanguageColorMap = (): Record<string, string> => {
  return Object.entries(languageColors).reduce((acc, cur) => {
    const [lan, color] = cur
    return { ...acc, [lan.toLocaleLowerCase()]: color }
  }, {})
}

const data = [
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

export default function Activity() {
  return (
    <div className="flex min-h-screen w-full flex-col">
      <div className="flex flex-col sm:gap-4 sm:py-4 sm:pl-14">
        <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0">
          <div className="ml-auto flex items-center gap-2">
            <Select defaultValue="past1hour">
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Date range" />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectItem value="today">Today</SelectItem>
                <SelectItem value="yesterday">Yesterday</SelectItem>
                <SelectItem value="past1hour">Last 1 hour</SelectItem>
                <SelectItem value="past24hour">Last 24 hours</SelectItem>
                <SelectItem value="past3days">Last 3 days</SelectItem>
                <SelectItem value="custom">Custom date until now</SelectItem>
                <SelectItem value="custom">Custom date range</SelectItem>
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
                  {data.map(item => (
                    <ActivityRow key={item.id} activity={item} />
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
  )
}

function ActivityRow({ activity }: { activity: (typeof data)[0] }) {
  const [isCollapse, setIsCollapse] = useState(false)
  const { theme } = useTheme()
  const colorMap = getLanguageColorMap()
  const color = colorMap[activity.language.toLocaleLowerCase()]
  return (
    <>
      <TableRow
        key={`${activity.id}}-1`}
        className="cursor-pointer"
        onClick={() => setIsCollapse(!isCollapse)}
      >
        <TableCell className="font-medium">
          <Tooltip>
            <TooltipTrigger>{activity.type}</TooltipTrigger>
            <TooltipContent>
              <p>Code completion showed</p>
            </TooltipContent>
          </Tooltip>
        </TableCell>
        <TableCell>{activity.user}</TableCell>
        <TableCell>
          {moment(activity.date).isBefore(moment().subtract(1, 'days'))
            ? moment(activity.date).format('YYYY-MM-DD HH:mm')
            : moment(activity.date).fromNow()}
        </TableCell>
        <TableCell>
          <div className="flex items-center text-xs">
            <div
              className="mr-1.5 h-2 w-2 rounded-full"
              style={{ backgroundColor: color }}
            />
            {activity.language}
          </div>
        </TableCell>
      </TableRow>

      {isCollapse && (
        <TableRow key={`${activity.id}-2`}>
          <TableCell className="font-medium" colSpan={4}>
            <ScrollArea className="w-full whitespace-nowrap rounded-md">
              <div className="flex flex-col gap-y-3">
                <div>
                  <h4 className="scroll-m-20 text-xl font-semibold tracking-tight">
                    Completion Id
                  </h4>
                  <p>{demoJson.event.completion.completion_id}</p>
                </div>

                <div>
                  <h4 className="scroll-m-20 text-xl font-semibold tracking-tight">
                    Language
                  </h4>
                  <p>{demoJson.event.completion.language}</p>
                </div>

                <div>
                  <h4 className="scroll-m-20 text-xl font-semibold tracking-tight">
                    prompt
                  </h4>
                  <SyntaxHighlighter
                    language={
                      demoJson.event.completion.language.startsWith(
                        'typescript'
                      )
                        ? 'typescript'
                        : demoJson.event.completion.language
                    }
                    style={tomorrowNightEighties}
                    wrapLongLines
                  >
                    {demoJson.event.completion.prompt}
                  </SyntaxHighlighter>
                </div>

                <div>
                  <h4 className="scroll-m-20 text-xl font-semibold tracking-tight">
                    segments
                  </h4>
                  <p>prefix</p>
                  <SyntaxHighlighter
                    language={
                      demoJson.event.completion.language.startsWith(
                        'typescript'
                      )
                        ? 'typescript'
                        : demoJson.event.completion.language
                    }
                    style={tomorrowNightEighties}
                    wrapLongLines
                  >
                    {demoJson.event.completion.segments.prefix}
                  </SyntaxHighlighter>

                  <p>suffix</p>
                  <SyntaxHighlighter
                    language={
                      demoJson.event.completion.language.startsWith(
                        'typescript'
                      )
                        ? 'typescript'
                        : demoJson.event.completion.language
                    }
                    style={tomorrowNightEighties}
                    wrapLongLines
                  >
                    {demoJson.event.completion.segments.suffix}
                  </SyntaxHighlighter>
                </div>

                <div>
                  <h4 className="scroll-m-20 text-xl font-semibold tracking-tight">
                    choices
                  </h4>
                  <SyntaxHighlighter
                    language="json"
                    style={tomorrowNightEighties}
                    wrapLongLines
                  >
                    {JSON.stringify(demoJson.event.completion.choices, null, 2)}
                  </SyntaxHighlighter>
                </div>
              </div>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}
