import { cn } from '@/lib/utils'
import { CardContent, CardTitle } from '@/components/ui/card'

interface ProfileCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
}

const ProfileCard: React.FC<ProfileCardProps> = ({
  title,
  className,
  children,
  ...props
}) => {
  return (
    <div className={cn('border p-2 rounded-lg', className)} {...props}>
      <CardTitle>{title}</CardTitle>
      <CardContent className="p-0">{children}</CardContent>
    </div>
  )
}

export { ProfileCard }
