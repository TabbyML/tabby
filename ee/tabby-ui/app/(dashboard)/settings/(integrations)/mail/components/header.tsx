import { SubHeader } from '@/components/sub-header'

export const MailDeliveryHeader = ({ className }: { className?: string }) => {
  return (
    <SubHeader className={className}>
      Configuring SMTP information will enable users to receive database reports
      via email, such as slow query weekly reports.
    </SubHeader>
  )
}
