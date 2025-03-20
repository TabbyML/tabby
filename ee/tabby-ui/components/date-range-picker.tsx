'use client'

import React from 'react'
import moment, { unitOfTime } from 'moment'
import { DateRange } from 'react-day-picker'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Card, CardContent } from '@/components/ui/card'
import { IconCheck } from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'

enum DATE_OPTIONS {
  'TODAY' = 'today',
  'YESTERDAY' = 'yesterday',
  'CUSTOM_DATE' = 'custom_date',
  'CUSTOM_RANGE' = 'custom_range'
}

const parseDateValue = (value: string) => {
  return {
    number: parseInt(value, 10) || -1,
    unit: value[value.length - 1] || 'h'
  }
}

export default function DateRangePicker({
  options,
  onSelect,
  defaultValue,
  hasToday,
  hasYesterday,
  className
}: {
  options: { label: string; value: string }[]
  onSelect?: (range: DateRange) => void
  defaultValue?: string
  hasToday?: boolean
  hasYesterday?: boolean
  className?: string
}) {
  defaultValue = defaultValue || options[0].value
  const defaultDate = parseDateValue(defaultValue)
  const [dateRange, setDateRange] = React.useState<DateRange>({
    from: moment()
      .add(
        defaultDate.number,
        defaultDate.unit as unitOfTime.DurationConstructor
      )
      .toDate(),
    to: moment().toDate()
  })
  const [showDateFilter, setShowDateFilter] = React.useState(false)
  const [selectDateFilter, setSelectDateFilter] = React.useState(defaultValue)
  const [showDateRangerPicker, setShowDateRangerPicker] = React.useState(false)
  const [calendarDateRange, setCalendarDateRange] = React.useState<
    DateRange | undefined
  >({
    from: moment()
      .add(
        defaultDate.number,
        defaultDate.unit as unitOfTime.DurationConstructor
      )
      .toDate(),
    to: moment().toDate()
  })
  const [showDateUntilNowPicker, setShowDateUntilNowPicker] =
    React.useState(false)
  const [dateUntilNow, setDateUntilNow] = React.useState<Date | undefined>(
    moment().toDate()
  )

  const onDateFilterChange = (value: DATE_OPTIONS | string) => {
    switch (value) {
      case DATE_OPTIONS.TODAY: {
        updateDateRange({
          from: moment().startOf('day').toDate(),
          to: moment().toDate()
        })
        break
      }
      case DATE_OPTIONS.YESTERDAY: {
        updateDateRange({
          from: moment().subtract(1, 'd').startOf('day').toDate(),
          to: moment().subtract(1, 'd').endOf('day').toDate()
        })
        break
      }
      default: {
        const { unit, number } = parseDateValue(value)
        let from = moment().add(number, unit as unitOfTime.DurationConstructor)
        if (!['h', 's', 'ms'].includes(unit)) from = from.startOf('day')
        updateDateRange({
          from: from.toDate(),
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
    updateDateRange({
      from: calendarDateRange?.from,
      to: moment(calendarDateRange?.to).endOf('day').toDate()
    })
  }

  const applyDateUntilNow = () => {
    setShowDateFilter(false)
    setShowDateUntilNowPicker(false)
    setSelectDateFilter(DATE_OPTIONS.CUSTOM_DATE)
    updateDateRange({
      from: moment(dateUntilNow).startOf('day').toDate(),
      to: moment().toDate()
    })
  }

  const updateDateRange = (range: DateRange) => {
    if (onSelect) onSelect(range)
    setDateRange(range)
  }

  return (
    <div className="relative">
      <Select
        value={selectDateFilter}
        onValueChange={onDateFilterChange}
        open={showDateFilter}
        onOpenChange={onDateFilterOpenChange}
      >
        <SelectTrigger
          className={cn('w-[240px]', className)}
          onClick={() => setShowDateFilter(!showDateFilter)}
        >
          <SelectValue placeholder="Date range" />
        </SelectTrigger>
        <SelectContent align="end">
          {hasToday && (
            <SelectItem value={DATE_OPTIONS.TODAY}>Today</SelectItem>
          )}
          {hasYesterday && (
            <SelectItem value={DATE_OPTIONS.YESTERDAY}>Yesterday</SelectItem>
          )}
          {options.map(option => (
            <SelectItem value={option.value} key={option.value}>
              {option.label}
            </SelectItem>
          ))}
          <SelectItem value={DATE_OPTIONS.CUSTOM_DATE} className="hidden">
            {moment(dateRange?.from).format('ll')} - Now
          </SelectItem>
          <SelectItem value={DATE_OPTIONS.CUSTOM_RANGE} className="hidden">
            {moment(dateRange?.from).format('ll')}
            {dateRange?.to ? ` - ${moment(dateRange.to).format('ll')}` : ''}
          </SelectItem>
          <SelectSeparator />
          <div
            className="relative cursor-default py-1.5 pl-8 text-sm hover:bg-accent hover:text-accent-foreground"
            onClick={() => {
              setShowDateFilter(false)
              setShowDateUntilNowPicker(true)
            }}
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
            onClick={() => {
              setShowDateFilter(false)
              setShowDateRangerPicker(true)
            }}
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
  )
}
