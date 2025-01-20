import { SSOType } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'

interface SSOTypeRadioProps {
  value: SSOType
  onChange?: (value: SSOType) => void
  className?: string
  readonly?: boolean
}

export function SSOTypeRadio({
  value,
  onChange,
  className,
  readonly
}: SSOTypeRadioProps) {
  return (
    <div className={cn('space-y-2', className)}>
      <Label>Type</Label>
      <RadioGroup
        value={value}
        onValueChange={v => onChange?.(v as SSOType)}
        className="flex gap-8"
        orientation="horizontal"
        disabled={readonly}
      >
        <div className="flex items-center">
          <RadioGroupItem value="oauth" id="r_oauth" />
          <Label
            className="flex cursor-pointer items-center gap-2 pl-2"
            htmlFor="r_oauth"
          >
            OAuth 2.0
          </Label>
        </div>
        <div className="flex items-center">
          <RadioGroupItem value="ldap" id="r_ldap" />
          <Label
            className="flex cursor-pointer items-center gap-2 pl-2"
            htmlFor="r_ldap"
          >
            LDAP
          </Label>
        </div>
      </RadioGroup>
    </div>
  )
}
