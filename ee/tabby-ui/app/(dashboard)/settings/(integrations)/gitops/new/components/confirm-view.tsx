'use client'

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card'
import { IconGitHub } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'

import { BasicInfoFormValues } from '../../components/basic-info-form'
import { OAuthApplicationFormValues } from '../../components/oauth-application-form'

interface ConfirmViewProps {
  data: BasicInfoFormValues & OAuthApplicationFormValues
}

export default function ConfirmView({ data }: ConfirmViewProps) {
  return (
    <div className="mx-auto max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle>Confirm the info</CardTitle>
          <CardDescription>
            After creation, this Git provider can be chosen under Gitops
          </CardDescription>
        </CardHeader>
        <Separator />
        <CardContent className="pb-2">
          <InfoItem title="Type">
            <div className="flex items-center gap-2">
              <IconGitHub className="h-6 w-6" />
              GitHub.com
            </div>
          </InfoItem>
          <Separator />
          <InfoItem title="Display name">{data.displayName}</InfoItem>
          <Separator />
          <InfoItem title="Application ID">{data.applicationId}</InfoItem>
          <Separator />
          <InfoItem title="Application secret">
            {data.applicationSecret}
          </InfoItem>
        </CardContent>
      </Card>
    </div>
  )
}

function InfoItem({
  title,
  children
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-6 py-2 text-sm">
      <div className="w-[30%] text-right text-muted-foreground">{title}</div>
      <div>{children}</div>
    </div>
  )
}
