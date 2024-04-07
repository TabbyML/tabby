import { graphql } from '@/lib/gql/generates'

export const queryDailyStatsInPastYear = graphql(/* GraphQL */ `
  query DailyStatsInPastYear($users: [ID!]) {
    dailyStatsInPastYear(users: $users) {
      start
      end
      completions
      selects
    }
  }
`)

export const queryDailyStats = graphql(/* GraphQL */ `
  query DailyStats(
    $start: DateTimeUtc!
    $end: DateTimeUtc!
    $users: [ID!]
    $languages: [Language!]
  ) {
    dailyStats(start: $start, end: $end, users: $users, languages: $languages) {
      start
      end
      completions
      selects
    }
  }
`)
