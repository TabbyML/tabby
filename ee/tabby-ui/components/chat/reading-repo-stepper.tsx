import { Maybe, MessageAttachmentCodeFileList, ThreadAssistantMessageReadingCode } from "@/lib/gql/generates/graphql"
import { AttachmentDocItem, RelevantCodeContext } from "@/lib/types"
import { useContext, useMemo } from "react"
import { ChatContext } from "./chat-context"
import { Accordion, AccordionItem, AccordionTrigger } from "../ui/accordion"
import { SourceIcon } from "../source-icon"
import { IconCode } from "../ui/icons"

interface ReadingRepoStepperProps {
  supportsOpenInEditor?: boolean
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  isReadingDocs: boolean | undefined
  readingCode: ThreadAssistantMessageReadingCode | undefined
  codeSourceId: Maybe<string>
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  codeFileList?: Maybe<MessageAttachmentCodeFileList>
  docs?: Maybe<AttachmentDocItem[]> | undefined
}

export function ReadingRepoStepper({
  codeSourceId,
  clientCodeContexts,
  serverCodeContexts,
  docs
}: ReadingRepoStepperProps) {

  const { repos } = useContext(ChatContext)
  const totalContextLength =
    (clientCodeContexts?.length || 0) +
    serverCodeContexts.length +
    (docs?.length || 0)

  const targetRepo = useMemo(() => {
    return repos?.find(x => x.sourceId === codeSourceId)
  }, [repos, codeSourceId])

  return (
    <Accordion collapsible type='single'>
      <AccordionItem value="readingRepo">
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex flex-1 items-center justify-between pr-2">
            <div className="flex flex-1 items-center gap-2">
              <IconCode className="mr-2 h-5 w-5 shrink-0" />
              <span>Look into</span>
              {!!targetRepo && (
                <div className="inline-flex cursor-pointer items-center gap-0.5 font-medium">
                  <SourceIcon
                    kind={targetRepo.sourceKind}
                    className="h-3.5 w-3.5 shrink-0"
                  />
                  <span className="truncate">{targetRepo.sourceName}</span>
                </div>
              )}
            </div>
            <div>
              {totalContextLength ? (
                <div className="text-sm text-muted-foreground">
                  {totalContextLength} sources
                </div>
              ) : null}
            </div>
          </div>
        </AccordionTrigger>
      </AccordionItem>
    </Accordion>
  )
}

