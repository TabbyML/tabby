'use client'

import { useState } from 'react'
import { useSearchParams } from 'next/navigation'

import { useAllMembers } from '@/lib/hooks/use-all-members'
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { useYearlyStats } from '@/lib/hooks/use-statistics'
import { IconUsers } from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { SubHeader } from '@/components/sub-header'

import { AnnualActivity } from './annual-activity'
import { ChatDailyActivity } from './chat-daily-activity'
import { CompletionDailyActivity } from './completion-daily-activity'
import { KEY_SELECT_ALL } from './constants'

export function Report() {
  const searchParams = useSearchParams()
  const [members, fetchingMembers] = useAllMembers()
  const isDemoMode = useIsDemoMode()
  const [selectedMember, setSelectedMember] = useState(KEY_SELECT_ALL)
  const sample = isDemoMode || searchParams.get('sample') === 'true'

  const {
    dailyData,
    fetching: fetchingYearlyStats,
    totalCount
  } = useYearlyStats({
    selectedMember,
    sample
  })

  return (
    <div className="- w-[calc(100vw-2rem)] md:w-auto 2xl:mx-auto 2xl:max-w-5xl">
      <div className="sticky top-16 z-10 -mt-4 flex flex-col items-center justify-between gap-y-2 border-b bg-background py-4 lg:flex-row lg:items-end lg:gap-y-0">
        <SubHeader className="mb-0">
          Statistics around Tabby IDE / Extensions
        </SubHeader>

        <LoadingWrapper
          loading={fetchingMembers}
          fallback={<Skeleton className="h-6 w-32" />}
        >
          <Select
            defaultValue={KEY_SELECT_ALL}
            onValueChange={setSelectedMember}
          >
            <SelectTrigger className="h-auto w-auto border-none py-0 shadow-none">
              <div className="flex h-6 items-center">
                <IconUsers className="mr-[0.45rem]" />
                <div className="w-[190px] overflow-hidden text-ellipsis text-left">
                  <SelectValue />
                </div>
              </div>
            </SelectTrigger>
            <SelectContent align="end">
              <SelectGroup>
                <SelectItem value={KEY_SELECT_ALL}>All members</SelectItem>
                {members.map(member => (
                  <SelectItem value={member.id} key={member.id}>
                    {member.name || member.email}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </LoadingWrapper>
      </div>

      {/* Yearly */}
      <LoadingWrapper
        loading={fetchingYearlyStats}
        fallback={<Skeleton className="mb-8 h-48" />}
      >
        <div className="mb-8 pt-4">
          <h1 className="mb-2 text-center text-xl font-semibold md:text-start">
            Activity
          </h1>
          <AnnualActivity totalCount={totalCount} dailyData={dailyData} />
        </div>
      </LoadingWrapper>

      {/* completions */}
      <CompletionDailyActivity
        sample={sample}
        selectedMember={selectedMember}
      />

      {/* chats */}
      <ChatDailyActivity sample={sample} selectedMember={selectedMember} />
    </div>
  )
}
