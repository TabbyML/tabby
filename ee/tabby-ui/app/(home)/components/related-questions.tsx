import { AnimatePresence, motion } from 'framer-motion'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

const readRepositoryRelatedQuestionsQuery = graphql(/* GraphQL */ `
  query readRepositoryRelatedQuestions($sourceId: String!) {
    readRepositoryRelatedQuestions(sourceId: $sourceId)
  }
`)

interface RelatedQuestionsProps {
  sourceId: string | undefined
  onClickQuestion: (question: string, sourceId: string) => void
}

export function RelatedQuestions({
  sourceId,
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

  const sourceIdInOperation =
    repositoryRelatedQuestionsOperation?.variables?.sourceId
  const repositoryRelatedQuestions =
    repositoryRelatedQuestionsData?.readRepositoryRelatedQuestions

  const onClickRelatedQuestion = (question: string) => {
    onClickQuestion(question, sourceIdInOperation as string)
  }

  const showList =
    !!repositoryRelatedQuestions &&
    !!sourceIdInOperation &&
    sourceIdInOperation === sourceId

  return (
    <AnimatePresence>
      {showList ? (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{
            duration: 0.2,
            ease: 'easeOut',
            delay: 0.1
          }}
        >
          <div className="flex flex-wrap justify-center gap-2 pb-3 pt-5 align-middle text-xs">
            {repositoryRelatedQuestions.map((x, idx) => {
              return (
                <div
                  key={`${x}_${idx}`}
                  className="cursor-pointer truncate rounded-lg bg-muted px-4 py-2 transition-opacity hover:bg-muted/70"
                  onClick={e => onClickRelatedQuestion(x)}
                >
                  <span>{x}</span>
                </div>
              )
            })}
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}
