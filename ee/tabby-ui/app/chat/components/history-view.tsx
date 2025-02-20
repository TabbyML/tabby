import { Button } from '@/components/ui/button'

interface HistoryViewProps {
  onClose: () => void
  onNavigate: (threadId: string) => void
}

export function HistoryView({ onClose, onNavigate }: HistoryViewProps) {
  return (
    <div className="fixed inset-0 z-10 px-[16px] pt-4 md:pt-10">
      <div className="flex items-center justify-between">
        <span className="text-lg font-semibold">History</span>
        <Button size="sm" onClick={onClose}>
          Done
        </Button>
      </div>
    </div>
  )
}
