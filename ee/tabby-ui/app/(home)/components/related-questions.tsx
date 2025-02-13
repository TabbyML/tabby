import Autoplay from 'embla-carousel-autoplay'
import { AnimatePresence, motion } from 'framer-motion'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious
} from '@/components/ui/carousel'

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
          <Carousel
            orientation="vertical"
            opts={{
              align: 'start',
              loop: true
            }}
            plugins={[
              Autoplay({
                delay: 5000,
                stopOnInteraction: true,
                stopOnMouseEnter: true
              })
            ]}
            className="group mt-3 w-full"
          >
            <CarouselContent className="h-14 pt-1.5">
              {repositoryRelatedQuestions.map((x, index) => (
                <CarouselItem key={`${x}_${index}`}>
                  <div className="mx-auto flex max-w-[80%] justify-center overflow-hidden p-1 text-sm">
                    <span
                      className="cursor-pointer truncate rounded-lg bg-muted px-4 py-1 transition-opacity hover:bg-muted/70"
                      onClick={e => onClickRelatedQuestion(x)}
                    >
                      {x}
                    </span>
                  </div>
                </CarouselItem>
              ))}
            </CarouselContent>
            <CarouselPrevious className="absolute left-auto right-10 top-1 z-10 h-6 w-6 opacity-0 group-hover:opacity-100" />
            <CarouselNext className="absolute left-auto right-2 top-1 z-10 h-6 w-6 opacity-0 group-hover:opacity-100" />
          </Carousel>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}
