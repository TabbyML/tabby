import { cn } from '@/lib/utils'
import { CardContent, CardTitle } from '@/components/ui/card'

interface ProfileCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
  description?: string
}

const ProfileCard: React.FC<ProfileCardProps> = ({
  title,
  description,
  className,
  children,
  ...props
}) => {
  return (
    <div
      className={cn('flex flex-col gap-8 border p-6 rounded-lg', className)}
      {...props}
    >
      <div>
        <CardTitle>{title}</CardTitle>
        {description && (
          <div className="text-sm text-muted-foreground mt-2">
            {description}
          </div>
        )}
      </div>
      <CardContent className="p-0">{children}</CardContent>
    </div>
  )
}

export { ProfileCard }
