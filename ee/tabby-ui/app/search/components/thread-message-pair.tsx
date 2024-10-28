import {
  createContext,
  HTMLAttributes,
  useContext,
  useMemo,
  useState
} from 'react'
import { isEmpty } from 'lodash-es'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'

import { AssistantMessageSection } from './assistant-message-section'
import { ConversationMessage, SearchContext } from './search'
import { UserMessageSection } from './user-message-section'

interface ThreadMessagePairProps extends HTMLAttributes<HTMLDivElement> {
  userMessage: ConversationMessage
  assistantMessage: ConversationMessage
  isGenerating: boolean
  isDeletable?: boolean
  isLastMessagePair: boolean
}

type ThreadMessagePairContextValue = {
  isEditing: boolean
  onToggleEditMode: (v: boolean) => void
  draftUserMessage: ConversationMessage
  draftAssistantMessage: ConversationMessage
  setDraftUserMessage: React.Dispatch<React.SetStateAction<ConversationMessage>>
  setDraftAssistantMessage: React.Dispatch<
    React.SetStateAction<ConversationMessage>
  >
}

export const ThreadMessagePairContext =
  createContext<ThreadMessagePairContextValue>(
    {} as ThreadMessagePairContextValue
  )

export function ThreadMessagePair({
  userMessage,
  assistantMessage,
  isGenerating,
  isLastMessagePair,
  isDeletable
}: ThreadMessagePairProps) {
  const { onUpdateMessagePair } = useContext(SearchContext)
  const [isEditing, setIsEditing] = useState(false)
  // message for edit
  const [draftUserMessage, setDraftUserMessage] =
    useState<ConversationMessage>(userMessage)
  const [draftAssistantMessage, setDraftAssistantMessage] =
    useState<ConversationMessage>(assistantMessage)

  const onToggleEditMode = (edit: boolean) => {
    if (edit) {
      setDraftUserMessage(userMessage)
      setDraftAssistantMessage(assistantMessage)
    }

    setIsEditing(edit)
  }

  const onSubmit = async () => {
    const errorMessage = await onUpdateMessagePair({
      assistantMessage: draftAssistantMessage,
      userMessage: draftUserMessage
    })

    if (!errorMessage) {
      setIsEditing(false)
    }
  }

  const disabled = useMemo(() => {
    if (!isEditing) return true

    return (
      isEmpty(draftUserMessage.content.trim()) ||
      isEmpty(draftAssistantMessage.content.trim())
    )
  }, [draftUserMessage, draftAssistantMessage])

  return (
    <ThreadMessagePairContext.Provider
      value={{
        isEditing,
        onToggleEditMode,
        draftUserMessage,
        setDraftUserMessage,
        draftAssistantMessage,
        setDraftAssistantMessage
      }}
    >
      {/* <div className={cn({
        'px-4 py-2 bg-white rounded-lg': isEditing
      })}> */}
      <div className="pb-2 pt-8">
        <UserMessageSection message={userMessage} isEditing={isEditing} />
      </div>
      <div className={cn('pb-8 pt-2')}>
        <AssistantMessageSection
          answer={assistantMessage}
          isLastAssistantMessage={isLastMessagePair}
          showRelatedQuestion={isLastMessagePair}
          isLoading={isGenerating && isLastMessagePair}
          isDeletable={isDeletable}
        />
      </div>
      {/* </div> */}
      {isEditing && (
        <div className="-mt-2 mb-4 flex items-center justify-end gap-2 px-2">
          <Button
            variant="outline"
            onClick={e => {
              setIsEditing(false)
            }}
            className="min-w-[2rem]"
          >
            Cancel
          </Button>
          <Button
            disabled={disabled}
            onClick={e => {
              onSubmit()
            }}
          >
            Update messages
          </Button>
        </div>
      )}
      {!isLastMessagePair && <Separator />}
    </ThreadMessagePairContext.Provider>
  )
}
