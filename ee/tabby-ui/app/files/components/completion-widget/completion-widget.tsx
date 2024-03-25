import { cn } from "@/lib/utils"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Button } from "@/components/ui/button"
import { IconChevronUpDown } from "@/components/ui/icons"

interface CompletionWidgetProps extends React.HTMLAttributes<HTMLDivElement> { }

export const CompletionWidget: React.FC<CompletionWidgetProps> = ({ className, ...props }) => {
  return (
    <div className={cn('flex items-center gap-2 px-2 py-1 rounded-md bg-background border rounded-md', className)} {...props}>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button size='sm' variant='outline'>
            Explain
            <IconChevronUpDown />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align='start'
        >
          <DropdownMenuItem
            className="cursor-pointer"
          >
            Detailed
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button size='sm' variant='outline'>
            Generate
            <IconChevronUpDown />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align='start'
        >
          <DropdownMenuItem
            className="cursor-pointer"
          >
            A unit test
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}