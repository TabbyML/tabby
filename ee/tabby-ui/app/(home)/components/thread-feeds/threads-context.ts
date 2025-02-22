import { createContext } from 'react'

import { ContextSource } from '@/lib/gql/generates/graphql'
import { Member } from '@/lib/hooks/use-all-members'

export type ThreadFeedsContextValue = {
  allUsers: Member[] | undefined
  fetchingUsers: boolean
  sources: ContextSource[] | undefined
  fetchingSources: boolean
  onNavigateToThread: () => void
}

export const ThreadFeedsContext = createContext<ThreadFeedsContextValue>(
  {} as ThreadFeedsContextValue
)
