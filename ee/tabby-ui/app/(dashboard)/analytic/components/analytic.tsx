'use client'

import moment from 'moment'

import {
  Bar,
  BarChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  PieChart, Pie, Cell
} from 'recharts'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import DatePickerWithRange from '@/components/date-range-picker'
import ActivityCalendar from '@/components/activity-calendar'

const data = [
  {
    name: '1 Jan',
    value: 4000
  },
  {
    name: '2 Jan',
    value: 3000
  },
  {
    name: '3 Jan',
    value: 2000
  },
  {
    name: '4 Jan',
    value: 2780
  },
  {
    name: '5 Jan',
    value: 1890
  },
  {
    name: '6 Jan',
    value: 2390
  },
  {
    name: '7 Jan',
    value: 3490
  },
  {
    name: '8 Jan',
    value: 4000
  },
  {
    name: '9 Jan',
    value: 3000
  },
  {
    name: '10 Jan',
    value: 2000
  },
  {
    name: '11 Jan',
    value: 2780
  },
  {
    name: '12 Jan',
    value: 1890
  },
  {
    name: '13 Jan',
    value: 2390
  },
  {
    name: '14 Jan',
    value: 3490
  },
  {
    name: '15 Jan',
    value: 490
  }
]

const data_acceptance = [
  {
    name: '1 Jan',
    IntelliJ: 40,
    VSCode: 24
  },
  {
    name: '2 Jan',
    IntelliJ: 30,
    VSCode: 13
  },
  {
    name: '3 Jan',
    IntelliJ: 20,
    VSCode: 98
  },
  {
    name: '4 Jan',
    IntelliJ: 27,
    VSCode: 39
  },
  {
    name: '5 Jan',
    IntelliJ: 18,
    VSCode: 48
  },
  {
    name: '6 Jan',
    IntelliJ: 23.9,
    VSCode: 38
  },
  {
    name: '7 Jan',
    IntelliJ: 34.9,
    VSCode: 43
  }
]

export function Analytic() {
  // todo query

  return (
    <div>
      <AnalyticHeader />
      <AnalyticSummary />
      <CompletionsChartSection />
      <div className="flex gap-x-5">
        <div className="flex-1">
          <AcceptanceChartSection />
        </div>
        <div style={{ flex: 3 }}>
          <ActivityChartSection />
        </div>
      </div>
     
    </div>
  )
}

function AnalyticHeader() {
  return (
    <div className="mb-6 flex items-center justify-between">
      <div>
        <h1 className="mb-1.5 scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
          Analytics
        </h1>
        <p className="text-muted-foreground">Overview of code completion usage</p>
      </div>

      <div className="flex space-x-4">
        <Select defaultValue='all'>
          <SelectTrigger className="w-[180px]" >
            <div className="flex w-full items-center truncate ">
              <span className="mr-1.5 hidden text-muted-foreground sm:inline-block">
                Member:
              </span>
              <div className="overflow-hidden text-ellipsis">
                <SelectValue />
              </div>
            </div>
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="jueliang">Jueliang</SelectItem>
              <SelectItem value="wayne">Wayne</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
        
        <Select defaultValue='all'>
          <SelectTrigger className="w-[180px]" >
            <div className="flex w-full items-center truncate">
              <span className="mr-1.5 hidden text-muted-foreground sm:inline-block">
                Language:
              </span>
              <div className="overflow-hidden text-ellipsis">
                <SelectValue />
              </div>
            </div>
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="python">Python</SelectItem>
              <SelectItem value="rust">Rust</SelectItem>
              <SelectItem value="javascript">Javascript</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>

        <DatePickerWithRange
          buttonClassName="h-full"
          contentAlign="end" />
      </div>
    </div>
  )
}

function AnalyticSummary() {
  return (
    <div className="mb-5 flex items-center space-x-4">
      <div className="w-60 space-y-0.5 rounded-lg border bg-primary-foreground/30 p-4">
        <p className="text-sm text-muted-foreground">Total completions</p>
        <p className="text-3xl font-bold">
          6,579
        </p>
      </div>

      <div className="w-60 space-y-0.5 rounded-lg border bg-primary-foreground/30 p-4">
        <p className="text-sm text-muted-foreground">Minutes saved / completion</p>
        <p className=" text-3xl font-bold">2</p>
      </div>

      <div className="w-60 space-y-0.5 rounded-lg border bg-primary-foreground/30 p-4">
        <p className="text-sm text-muted-foreground">Hours saved in total</p>
        <p className=" text-3xl font-bold">100</p>
      </div>
    </div>
  )
}

function CompletionsChartSection() {
  return (
    <div className="mb-5 rounded-lg border bg-primary-foreground/30 p-4">
      <h1 className="mb-5 text-xl font-bold">Completions</h1>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          width={500}
          height={300}
          data={data}
          margin={{
            top: 5,
            right: 20,
            left: 20,
            bottom: 5
          }}
        >
          <Bar dataKey="value" fill="#8884d8" />
          <XAxis dataKey="name" fontSize={12} />
          <YAxis fontSize={12} />
          <Tooltip cursor={{ fill: 'hsl(var(--accent))' }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function AcceptanceChartSection() {
  const data = [
    { name: 'Accept', value: 512},
    { name: 'Pending', value: 1013},
  ];
  
  const COLORS = ['#8884d8', '#b9b7e2'];
  
  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percent,
    name
  }: {
    cx: number;
    cy: number;
    midAngle: number;
    innerRadius: number;
    outerRadius: number;
    percent: number;
    name: string;
  }) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.4;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
  
    if (name.toLocaleLowerCase() === 'accept') {
      return (
        <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={15}>
          {`${(percent * 100).toFixed(0)}%`}
        </text>
      );
    }
    return
  };

  return (
    <div className="rounded-lg border bg-primary-foreground/30 p-4">
      <h1 className="text-xl font-bold">Acceptance</h1>
      <p className="mt-0.5 text-xs text-muted-foreground">Jan 1, 2024 - Jan 15, 2024</p>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart width={700} height={400}>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderCustomizedLabel}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

function ActivityChartSection () {
  const data = new Array(365).fill("").map((_, idx) => ({
    date: moment().subtract(idx, 'days').format('YYYY-MM-DD'),
    count: Math.round(Math.random() * 20),
    level: Math.floor(Math.random() * 5)
  }))

  return (
    <div className="flex h-full flex-col rounded-lg border bg-primary-foreground/30 p-4">
      <h1 className="text-xl font-bold">Activity</h1>
      <p className="mt-0.5 text-xs text-muted-foreground">5944 completions in the last year</p>
      <div className="flex flex-1 items-center justify-center">
        <ActivityCalendar data={data} />
      </div>
    </div>
  );
}
