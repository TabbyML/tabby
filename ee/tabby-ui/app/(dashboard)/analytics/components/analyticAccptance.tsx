
'use client'

import moment from "moment";

import type { DateRange } from "react-day-picker"
import type { DailyStats } from '../types/stats'

import {
  Legend,
  ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts'

export function AnlyticAcceptance ({
	dailyStats,
	dateRange,
}: {
	dailyStats: DailyStats[] | undefined;
	dateRange: DateRange;
}) {
	const from = dateRange.from || new Date()
	const to = dateRange.to || from

	let totalAccpet = 0
	let totalCompletions = 0
	dailyStats?.forEach(stats => {
		totalCompletions += stats.completions
		totalAccpet += stats.selects
	})
	const totalPending = totalCompletions - totalAccpet

	const data = [
		{ name: 'Accept', value: totalAccpet},
		{ name: 'Pending', value: totalPending},
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
		const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
		const x = cx + radius * Math.cos(-midAngle * RADIAN);
		const y = cy + radius * Math.sin(-midAngle * RADIAN);

		if (name.toLocaleLowerCase() === 'accept') {
			return (
				<text
					x={x}
					y={y}
					fill="white"
					textAnchor={x > cx ? 'start' : 'end'}
					dominantBaseline="central"
					fontSize={12}>
					{`${(percent * 100).toFixed(0)}%`}
				</text>
			);
		}
		return
	};

	return (
		<div className="rounded-lg border bg-primary-foreground/30 p-4">
			<h1 className="text-xl font-bold">Acceptance</h1>
			<p className="mt-0.5 text-xs text-muted-foreground">
				{moment(from).format('D MMM, YYYY')} - {moment(to).format('D MMM, YYYY')}
			</p>
			<ResponsiveContainer width="100%" height={210}>
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
					{totalCompletions !== 0 && <Legend wrapperStyle={{ fontSize: '12px' }} />}
				</PieChart>
			</ResponsiveContainer>
		</div>
	)
}