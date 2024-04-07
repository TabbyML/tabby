import { eachDayOfInterval } from 'date-fns'
import moment from 'moment'

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card, CardContent } from '@/components/ui/card'

import type { DateRange } from "react-day-picker"
import type { DailyStats } from '../types/stats'

function BarTooltip ({ active, payload, label }: {
	active?: boolean;
	label?: string
	payload?: {
		name: string;
		value: number;
	}[]
}) {
	if (active && payload && payload.length) {
		const completion = payload[0].value
		if (!completion) return null;
		return (
			<Card>
				<CardContent className="px-4 py-2 text-sm">
					<p>Completions: <b>{completion}</b></p>
					<p className="text-muted-foreground">{label}</p>
				</CardContent>
			</Card>
		);
	}

	return null;
};

export function AnalyticDailyCompletion ({
	dailyStats,
	dateRange,
}: {
	dailyStats: DailyStats[] | undefined;
	dateRange: DateRange;
}) {
	const from = dateRange.from || new Date()
	const to = dateRange.to || from

	const dailyCompletionMap: Record<string, number> = dailyStats?.reduce((acc, cur) => {
		const date = moment(cur.start).format('YYYY-MM-DD')
		return { ...acc, [date]: cur.completions }
	}, {}) || {}

	const daysBetweenRange = eachDayOfInterval({
		start: from,
		end: to
	});

	const chartData = daysBetweenRange.map(date => {
		const completionKey = moment(date).format('YYYY-MM-DD')
		const value = dailyCompletionMap[completionKey] || 0
		return {
			name: moment(date).format('D MMM'),
			value
		}
	})
	return (
		<div className="rounded-lg border bg-primary-foreground/30 p-4">
			<h1 className="mb-5 text-xl font-bold">Completions</h1>
			<ResponsiveContainer width="100%" height={350}>
				<BarChart
					width={500}
					height={300}
					data={chartData}
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
					<Tooltip
						cursor={{ fill: 'transparent' }}
						content={<BarTooltip />} />
				</BarChart>
			</ResponsiveContainer>
		</div>
	)
}