import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { ChatCommand } from 'tabby-chat-panel/index'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconChevronUpDown } from '@/components/ui/icons'

import { emitter } from '../../lib/event-emitter'

interface ActionBarWidgetProps extends React.HTMLAttributes<HTMLDivElement> {
  text: string
  language?: string
  path: string
  lineFrom: number
  lineTo: number
  gitUrl: string
}

export const ActionBarWidget: React.FC<ActionBarWidgetProps> = ({
  className,
  text,
  language,
  path,
  lineFrom,
  lineTo,
  gitUrl,
  ...props
}) => {
  const onTriggerCommand = (command: ChatCommand) => {
    emitter.emit('quick_action_command', command)
  }

  return (
    <div
      className={cn(
        'mt-2 flex items-center gap-2 rounded-md border bg-background px-2 py-1',
        className
      )}
      {...props}
    >
      <Image src={tabbyLogo} width={32} alt="logo" />
      <Button
        size="sm"
        variant="outline"
        onClick={e => onTriggerCommand('explain')}
      >
        Explain
      </Button>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button size="sm" variant="outline">
            Generate
            <IconChevronUpDown className="ml-1" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem
            className="cursor-pointer"
            onSelect={() => onTriggerCommand('generate-tests')}
          >
            Unit Test
          </DropdownMenuItem>
          <DropdownMenuItem
            className="cursor-pointer"
            onSelect={() => onTriggerCommand('generate-docs')}
          >
            Documentation
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
