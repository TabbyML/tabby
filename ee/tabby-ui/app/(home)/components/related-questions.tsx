import { useMemo } from 'react'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { ContextInfoQuery } from '@/lib/gql/generates/graphql'

const readRepositoryRelatedQuestionsQuery = graphql(/* GraphQL */ `
  query readRepositoryRelatedQuestions($sourceId: String!) {
    readRepositoryRelatedQuestions(sourceId: $sourceId)
  }
`)

interface RelatedQuestionsProps {
  sourceId: string | undefined
  contextInfo: ContextInfoQuery['contextInfo'] | undefined
  onClickQuestion: (question: string, sourceId: string) => void
}

export function RelatedQuestions({
  sourceId,
  contextInfo,
  onClickQuestion
}: RelatedQuestionsProps) {
  const [
    {
      data: repositoryRelatedQuestionsData,
      operation: repositoryRelatedQuestionsOperation
    }
  ] = useQuery({
    query: readRepositoryRelatedQuestionsQuery,
    variables: {
      sourceId: sourceId as string
    },
    pause: !sourceId
  })

  const sourceIdForQuestions =
    repositoryRelatedQuestionsOperation?.variables?.sourceId
  const repoForQuestions = useMemo(() => {
    return contextInfo?.sources.find(
      source => source.sourceId === sourceIdForQuestions
    )
  }, [contextInfo, sourceIdForQuestions])
  const repositoryRelatedQuestions =
    repositoryRelatedQuestionsData?.readRepositoryRelatedQuestions

  const onClickRelatedQuestion = (question: string) => {
    onClickQuestion(question, sourceIdForQuestions as string)
  }

  if (!repositoryRelatedQuestions || !sourceIdForQuestions) return null

  return (
    <div className="flex flex-wrap gap-2 mb-3 mt-5 text-sm justify-center align-middle">
      {repositoryRelatedQuestions.map((x, idx) => {
        return (
          <div
            key={`${x}_${idx}`}
            className="cursor-pointer px-4 py-2 bg-muted hover:bg-muted/70 rounded-lg truncate transition-opacity"
            onClick={e => onClickRelatedQuestion(x)}
          >
            {repoForQuestions && (
              <span>{formatRepoName(repoForQuestions.sourceName)}:&nbsp;</span>
            )}
            <span>{x}</span>
          </div>
        )
      })}
    </div>
  )
}

function formatRepoName(name: string) {
  return name.split('/').pop()
}
