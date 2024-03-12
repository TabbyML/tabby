'use client'

import React from 'react'
import {
  Bar,
  BarChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Label
} from 'recharts'

import { Separator } from '@/components/ui/separator'

const data = [
  {
    name: '1 Jan',
    IntelliJ: 4000,
    VSCode: 2400,
    amt: 2400
  },
  {
    name: '2 Jan',
    IntelliJ: 3000,
    VSCode: 1398,
    amt: 2210
  },
  {
    name: '3 Jan',
    IntelliJ: 2000,
    VSCode: 9800,
    amt: 2290
  },
  {
    name: '4 Jan',
    IntelliJ: 2780,
    VSCode: 3908,
    amt: 2000
  },
  {
    name: '5 Jan',
    IntelliJ: 1890,
    VSCode: 4800,
    amt: 2181
  },
  {
    name: '6 Jan',
    IntelliJ: 2390,
    VSCode: 3800,
    amt: 3500
  },
  {
    name: '7 Jan',
    IntelliJ: 3490,
    VSCode: 4300,
    amt: 2100
  }
]

const data_acceptance = [
  {
    name: '1 Jan',
    IntelliJ: 4000,
    VSCode: 2400,
    amt: 2400
  },
  {
    name: '2 Jan',
    IntelliJ: 3000,
    VSCode: 1398,
    amt: 2210
  },
  {
    name: '3 Jan',
    IntelliJ: 2000,
    VSCode: 9800,
    amt: 2290
  },
  {
    name: '4 Jan',
    IntelliJ: 2780,
    VSCode: 3908,
    amt: 2000
  },
  {
    name: '5 Jan',
    IntelliJ: 1890,
    VSCode: 4800,
    amt: 2181
  },
  {
    name: '6 Jan',
    IntelliJ: 2390,
    VSCode: 3800,
    amt: 3500
  },
  {
    name: '7 Jan',
    IntelliJ: 3490,
    VSCode: 4300,
    amt: 2100
  }
]

export function Analytic() {
  // todo query

  return (
    <div className="space-y-10">
      <AnalyticSummary />
      <CompletionsByDay />
      <CompletionAcceptanceRate />
    </div>
  )
}

function AnalyticSummary() {
  return (
    <div className="flex items-center space-x-8 text-sm">
      <div>
        <div className="text-2xl text-[#70DAE8] text-center font-bold">666</div>
        <div>Total accepted completions</div>
      </div>
      <Separator orientation="vertical" className="h-14" />
      <div>
        <div className="text-2xl text-[#FF4F3A] text-center font-bold">2</div>
        <div>Minutes saved per completion</div>
      </div>
      <Separator orientation="vertical" className="h-14" />
      <div>
        <div className="text-2xl text-[#A110FE] text-center font-bold">10</div>
        <div>Hours saved by completions</div>
      </div>
    </div>
  )
}

function CompletionsByDay() {
  return (
    <div className="space-y-4">
      <h1 className="font-bold text-xl">Tabby completions by day</h1>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          width={500}
          height={300}
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5
          }}
        >
          <XAxis dataKey="name" fontSize={12} />
          <YAxis fontSize={12}>
            <Label
              value="Completion events"
              position="insideLeft"
              angle={-90}
              style={{ textAnchor: 'middle' }}
            />
          </YAxis>
          <Tooltip />
          <Legend />
          <Bar
            dataKey="VSCode"
            fill="#8884d8"
            radius={[2, 2, 0, 0]}
          />
          <Bar
            dataKey="IntelliJ"
            fill="#82ca9d"
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function CompletionAcceptanceRate() {
  return (
    <div className="space-y-4">
      <h1 className="font-bold text-xl">Completion acceptance rate</h1>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart
          width={500}
          height={300}
          data={data_acceptance}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5
          }}
        >
          <XAxis dataKey="name" fontSize={12} />
          <YAxis fontSize={12}>
            <Label
              value="Rate"
              position="insideLeft"
              angle={-90}
              style={{ textAnchor: 'middle' }}
            />
          </YAxis>
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="VSCode"
            stroke="#8884d8"
            activeDot={{ r: 8 }}
          />
          <Line type="monotone" dataKey="IntelliJ" stroke="#82ca9d" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
