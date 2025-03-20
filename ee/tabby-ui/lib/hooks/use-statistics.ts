import { useMemo } from 'react'
import { eachDayOfInterval } from 'date-fns'
import { compact, sum } from 'lodash-es'
import moment from 'moment'
import { DateRange } from 'react-day-picker'
import seedrandom from 'seedrandom'
import { useQuery } from 'urql'

import { graphql } from '../gql/generates'
import {
  ChatDailyStatsInPastYearQuery,
  ChatDailyStatsQuery,
  DailyStatsInPastYearQuery,
  DailyStatsQuery,
  Language
} from '../gql/generates/graphql'
import { ArrayElementType } from '../types'

const KEY_SELECT_ALL = 'all'

const dailyStatsInPastYearQuery = graphql(/* GraphQL */ `
  query DailyStatsInPastYear($users: [ID!]) {
    dailyStatsInPastYear(users: $users) {
      __typename
      start
      end
      completions
      selects
      views
    }
  }
`)

const chatDailyStatsQuery = graphql(/* GraphQL */ `
  query ChatDailyStats($start: DateTime!, $end: DateTime!, $users: [ID!]) {
    chatDailyStats(start: $start, end: $end, users: $users) {
      start
      end
      chats
    }
  }
`)

const chatDailyStatsInPastYearQuery = graphql(/* GraphQL */ `
  query chatDailyStatsInPastYear($users: [ID!]) {
    chatDailyStatsInPastYear(users: $users) {
      __typename
      start
      end
      chats
    }
  }
`)

const dailyStatsQuery = graphql(/* GraphQL */ `
  query DailyStats(
    $start: DateTime!
    $end: DateTime!
    $users: [ID!]
    $languages: [Language!]
  ) {
    dailyStats(start: $start, end: $end, users: $users, languages: $languages) {
      start
      end
      completions
      selects
      views
      language
    }
  }
`)

export function useChatDailyStats({
  dateRange,
  selectedMember,
  sample
}: {
  dateRange: DateRange
  selectedMember: string | undefined
  sample?: boolean
}) {
  const from = dateRange.from || new Date()
  const to = dateRange.to || from
  // Query chat stats of selected date range
  const [{ data: chatDailyStatsData, fetching: fetchingChatDailyStats }] =
    useQuery({
      query: chatDailyStatsQuery,
      variables: {
        start: moment(dateRange.from).startOf('day').utc().format(),
        end: moment(dateRange.to).endOf('day').utc().format(),
        users:
          selectedMember === KEY_SELECT_ALL
            ? undefined
            : compact([selectedMember])
      },
      pause: !selectedMember
    })

  const chatDailyStats: ChatDailyStatsQuery['chatDailyStats'] | undefined =
    useMemo(() => {
      if (sample) {
        const daysBetweenRange = eachDayOfInterval({
          start: from,
          end: to
        })
        return daysBetweenRange.map(date => {
          const rng = seedrandom(
            moment(date).format('YYYY-MM-DD') + selectedMember + 'chats'
          )
          const chats = Math.ceil(Math.ceil(rng() * 50))
          return {
            start: moment(date).utc().format(),
            end: moment(date).add(1, 'day').utc().format(),
            chats
          }
        })
      } else {
        return chatDailyStatsData?.chatDailyStats
      }
    }, [chatDailyStatsData, sample])

  const totalCount = sum(chatDailyStats?.map(stats => stats.chats))

  const dailyChatMap: Record<string, number> = {}

  chatDailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyChatMap[date] = dailyChatMap[date] || 0
    dailyChatMap[date] += stats.chats
  }, {})

  const daysBetweenRange = eachDayOfInterval({
    start: dateRange.from!,
    end: dateRange.to!
  })

  const chatChartData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const chats = dailyChatMap[dateKey] || 0
    return {
      name: moment(date).format('MMMM D'),
      chats
    }
  })

  return {
    chatDailyStats,
    totalCount,
    chatChartData,
    fetchingChatDailyStats
  }
}

export function useCompletionDailyStats({
  dateRange,
  selectedMember,
  sample,
  languages
}: {
  dateRange: DateRange
  selectedMember: string | undefined
  sample?: boolean
  languages?: Language[]
}) {
  const [{ data: dailyStatsData, fetching: fetchingCompletionDailyStats }] =
    useQuery({
      query: dailyStatsQuery,
      variables: {
        start: moment(dateRange.from).startOf('day').utc().format(),
        end: moment(dateRange.to).endOf('day').utc().format(),
        users:
          selectedMember === KEY_SELECT_ALL
            ? undefined
            : compact([selectedMember]),
        languages
      }
    })

  const completionDailyStats: DailyStatsQuery['dailyStats'] | undefined =
    useMemo(() => {
      let dailyStats: DailyStatsQuery['dailyStats'] | undefined
      if (sample) {
        const daysBetweenRange = eachDayOfInterval({
          start: dateRange.from!,
          end: dateRange.to || dateRange.from!
        })
        dailyStats = daysBetweenRange.map(date => {
          const _languages = [
            Language.Typescript,
            Language.Python,
            Language.Rust
          ]
          const rng = seedrandom(
            moment(date).format('YYYY-MM-DD') + selectedMember + _languages
          )
          const selects = Math.ceil(rng() * 20)
          const completions = Math.ceil(selects / 0.35)
          return {
            start: moment(date).utc().format(),
            end: moment(date).add(1, 'day').utc().format(),
            completions,
            selects,
            views: completions,
            language: _languages[selects % _languages.length]
          }
        })
      } else {
        dailyStats = dailyStatsData?.dailyStats
      }
      dailyStats = dailyStats?.filter(stats => {
        if (!languages?.length) return true
        return languages.includes(stats.language)
      })

      return dailyStats
    }, [sample, languages, dailyStatsData])

  const from = dateRange.from || new Date()
  const to = dateRange.to || from

  const dailyViewMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}

  completionDailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyViewMap[date] = dailyViewMap[date] || 0
    dailySelectMap[date] = dailySelectMap[date] || 0

    dailyViewMap[date] += stats.views
    dailySelectMap[date] += stats.selects
  }, {})

  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  const completionChartData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const views = dailyViewMap[dateKey] || 0
    const selects = dailySelectMap[dateKey] || 0
    const pendings = views - selects
    return {
      name: moment(date).format('MMMM D'),
      views,
      selects,
      pendings
    }
  })

  return {
    completionChartData,
    completionDailyStats,
    fetchingCompletionDailyStats
  }
}

export function useYearlyStats({
  selectedMember,
  sample
}: {
  selectedMember: string | undefined
  sample?: boolean
}) {
  // query chat yearly stats
  const [{ data: chatYearlyStatsData, fetching: fetchingChatYearlyStats }] =
    useQuery({
      query: chatDailyStatsInPastYearQuery,
      variables: {
        users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
      },
      pause: !selectedMember
    })

  // Query yearly stats
  const [
    { data: completionYearlyStatsData, fetching: fetchingCompletionYearlyStats }
  ] = useQuery({
    query: dailyStatsInPastYearQuery,
    variables: {
      users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
    },
    pause: !selectedMember
  })
  let yearlyStats:
    | Array<
        | ArrayElementType<DailyStatsInPastYearQuery['dailyStatsInPastYear']>
        | ArrayElementType<
            ChatDailyStatsInPastYearQuery['chatDailyStatsInPastYear']
          >
      >
    | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: moment().toDate(),
      end: moment().subtract(365, 'days').toDate()
    })
    yearlyStats = daysBetweenRange.map(date => {
      const rng = seedrandom(moment(date).format('YYYY-MM-DD') + selectedMember)
      const selects = Math.ceil(rng() * 20)
      const completions = selects + Math.floor(rng() * 10)
      return {
        __typename: 'CompletionStats',
        start: moment(date).format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        end: moment(date).add(1, 'day').format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        completions,
        selects,
        views: completions
      }
    })
  } else {
    yearlyStats = [
      ...(completionYearlyStatsData?.dailyStatsInPastYear || []),
      ...(chatYearlyStatsData?.chatDailyStatsInPastYear || [])
    ]
  }

  let lastYearActivities = 0
  const dailyViewMap: Record<string, number> =
    yearlyStats?.reduce((acc, cur) => {
      const date = moment.utc(cur.start).format('YYYY-MM-DD')
      if (cur.__typename === 'CompletionStats') {
        lastYearActivities += cur.views
        lastYearActivities += cur.selects
        return {
          ...acc,
          [date]: (acc[date] ?? 0) + cur.views
        }
      } else if (cur.__typename === 'ChatCompletionStats') {
        lastYearActivities += cur.chats
        return {
          ...acc,
          [date]: (acc[date] ?? 0) + cur.chats
        }
      } else {
        return acc
      }
    }, {} as Record<string, number>) || {}

  const data = new Array(365)
    .fill('')
    .map((_, idx) => {
      const date = moment().subtract(idx, 'days').format('YYYY-MM-DD')
      const count = dailyViewMap[date] || 0
      const level = Math.min(4, Math.ceil(count / 5))
      return {
        date: date,
        count,
        level
      }
    })
    .reverse()

  return {
    yearlyStats,
    dailyData: data,
    totalCount: lastYearActivities,
    fetching: fetchingChatYearlyStats || fetchingCompletionYearlyStats
  }
}
