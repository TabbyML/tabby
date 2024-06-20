import { graphql } from '@/lib/gql/generates'

export const triggerJobRunMutation = graphql(/* GraphQL */ `
  mutation triggerJobRun($command: String!) {
    triggerJobRun(command: $command)
  }
`)
