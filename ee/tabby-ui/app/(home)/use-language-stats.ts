import { useEffect, useState } from 'react'
import { sum } from 'lodash-es'
import moment from 'moment'
import { useQuery } from 'urql'

import { Language } from '@/lib/gql/generates/graphql'
import { QueryVariables } from '@/lib/tabby/gql'
import { queryDailyStats } from '@/lib/tabby/query'

import { type LanguageStats } from './components/summary'

// Find auto-completion stats of each language
export function useLanguageStats({
  start,
  end,
  users
}: {
  start: Date
  end: Date
  users?: string
}) {
  const languages = Object.values(Language)
  const [lanIdx, setLanIdx] = useState(0)
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof queryDailyStats>
  >({
    start: moment(start).utc().format(),
    end: moment(end).utc().format(),
    users,
    languages: languages[0]
  })
  const [languageStats, setLanguageStats] = useState<LanguageStats>(
    {} as LanguageStats
  )

  const [{ data, fetching }] = useQuery({
    query: queryDailyStats,
    variables: queryVariables
  })

  useEffect(() => {
    if (lanIdx >= languages.length) return
    if (!fetching && data?.dailyStats) {
      const language = languages[lanIdx]
      const newLanguageStats = { ...languageStats }
      newLanguageStats[language] = newLanguageStats[language] || {
        selects: 0,
        completions: 0,
        name: Object.keys(Language)[lanIdx]
      }
      newLanguageStats[language].selects += sum(
        data.dailyStats.map(stats => stats.selects)
      )
      newLanguageStats[language].completions += sum(
        data.dailyStats.map(stats => stats.completions)
      )

      const newLanIdx = lanIdx + 1
      setLanguageStats(newLanguageStats)
      setLanIdx(newLanIdx)
      if (newLanIdx < languages.length) {
        setQueryVariables({
          start: moment(start).utc().format(),
          end: moment(end).utc().format(),
          users,
          languages: languages[newLanIdx]
        })
      }
    }
  }, [queryVariables, lanIdx, fetching])

  return [languageStats]
}
