import { Button } from '@/components/ui/button'
import { IconArrowRight } from '@/components/ui/icons'

const exampleMessages = [
  {
    heading: 'Convert list of string to numbers',
    message: `How to convert a list of string to numbers in python`
  },
  {
    heading: 'How to parse email address',
    message: 'How to parse email address with regex'
  }
]

export function EmptyScreen({
  setInput,
  chatMaxWidthClass,
  welcomeMessage
}: {
  setInput: (v: string) => void
  chatMaxWidthClass: string
  welcomeMessage?: string
}) {
  const welcomeMsg = welcomeMessage || 'Welcome'
  return (
    <div className={`mx-auto px-4 ${chatMaxWidthClass}`}>
      <div className="rounded-lg border bg-background p-8">
        <h1 className="mb-2 text-lg font-semibold">{welcomeMsg}</h1>
        <p className="leading-normal text-muted-foreground">
          You can start a conversation here or try the following examples:
        </p>
        <div className="mt-4 flex flex-col items-start space-y-2">
          {exampleMessages.map((message, index) => (
            <Button
              key={index}
              variant="link"
              className="h-auto p-0 text-base"
              onClick={() => setInput(message.message)}
            >
              <IconArrowRight className="mr-2 text-muted-foreground" />
              <p className="text-left">{message.heading}</p>
            </Button>
          ))}
        </div>
      </div>
    </div>
  )
}
