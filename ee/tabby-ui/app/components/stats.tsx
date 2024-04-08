import moment from 'moment'

import { useMe } from '@/lib/hooks/use-me'
import ActivityCalendar from '@/components/activity-calendar'

export default function Stats() {
  const [{ data }] = useMe()

  if (!data?.me?.email) return null

  // TODO: mock activity data
  const activies = new Array(365).fill('').map((_, idx) => ({
    date: moment().subtract(idx, 'days').format('YYYY-MM-DD'),
    count: Math.round(Math.random() * 20),
    level: Math.floor(Math.random() * 5)
  }))

  return (
    <div className="flex flex-col gap-y-8">
      <div className="flex justify-between gap-x-6">
        <div className="flex-1 rounded-xl bg-primary-foreground/50 p-5">
          <p className="text-sm text-muted-foreground">
            Completion In Last 7 Days
          </p>
          <p className="mt-1 text-3xl font-bold">211</p>
        </div>

        <div className="flex-1 rounded-xl bg-primary-foreground/50 p-5">
          <p className="text-sm text-muted-foreground">Completion In Total</p>
          <p className="mt-1 text-3xl font-bold">1530</p>
        </div>

        <div className="flex-1 rounded-xl bg-primary-foreground/50 p-5">
          <p className="text-sm text-muted-foreground">Streak</p>
          <p className="mt-1 text-3xl font-bold">3</p>
        </div>
      </div>

      <div>
        <p className="mb-2 text-sm text-secondary-foreground">
          <b>1200</b> contributions in the last year
        </p>
        <div className="flex items-end justify-center rounded-xl bg-primary-foreground/50 py-5">
          <ActivityCalendar data={activies} />
        </div>
      </div>

      <div>
        <p className="mb-2 text-sm text-secondary-foreground">
          My top coding languages
        </p>
        <div className="flex flex-col gap-y-5 rounded-xl bg-primary-foreground/50 px-11 py-6">
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">Rust</p>
            <div className="flex-1">
              <div className="h-2 w-[80%] rounded-full bg-yellow-600 dark:bg-yellow-600" />
            </div>
          </div>
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">Python</p>
            <div className="flex-1">
              <div className="h-2 w-[30%] rounded-full bg-blue-500 dark:bg-blue-300" />
            </div>
          </div>
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">Javascript</p>
            <div className="flex-1">
              <div className="h-2 w-[2%] rounded-full bg-yellow-400 dark:bg-yellow-300" />
            </div>
          </div>
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">CSS</p>
            <div className="flex-1">
              <div className="h-2 w-[2%] rounded-full bg-red-500 dark:bg-red-400" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
