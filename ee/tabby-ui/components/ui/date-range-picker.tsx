'use client'

import * as React from 'react'
import { CalendarIcon } from '@radix-ui/react-icons'
import { addDays, format } from 'date-fns'
import moment from 'moment'
import { DateRange } from 'react-day-picker'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'

export default function DatePickerWithRange({
  dateRange,
  className,
  buttonClassName,
  contentAlign,
  onOpenChange
}: React.HTMLAttributes<HTMLDivElement> & {
  dateRange?: DateRange
  buttonClassName?: string
  contentAlign?: 'start' | 'end' | 'center'
  onOpenChange?: (isOpen: boolean, date: DateRange | undefined) => void
  onSelectDateRange?: (date: DateRange | undefined) => void
}) {
  const [date, setDate] = React.useState<DateRange | undefined>({
    from: dateRange?.from || new Date(2022, 0, 20),
    to: dateRange?.to || addDays(new Date(2022, 0, 20), 20)
  })

  const toggleOpen = (isOpen: boolean) => {
    if (onOpenChange) onOpenChange(isOpen, date)
  }

  return (
    <div className={cn('grid gap-2', className)}>
      <Popover onOpenChange={toggleOpen}>
        <PopoverTrigger asChild>
          <Button
            id="date"
            variant={'outline'}
            className={cn(
              'w-[270px] justify-start text-left font-normal ',
              !date && 'text-muted-foreground',
              buttonClassName
            )}
          >
            <CalendarIcon className="size-4 mr-2" />
            {date?.from ? (
              date.to ? (
                <>
                  {format(date.from, 'LLL dd, y')} -{' '}
                  {format(date.to, 'LLL dd, y')}
                </>
              ) : (
                format(date.from, 'LLL dd, y')
              )
            ) : (
              <span>Pick a date</span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align={contentAlign}>
          <Calendar
            initialFocus
            mode="range"
            defaultMonth={moment(date?.from).subtract(1, 'month').toDate()}
            selected={date}
            onSelect={setDate}
            numberOfMonths={2}
            disabled={(date: Date) => date > new Date()}
          />
        </PopoverContent>
      </Popover>
    </div>
  )
}
