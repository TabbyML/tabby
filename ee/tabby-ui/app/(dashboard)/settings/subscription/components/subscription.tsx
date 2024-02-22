'use client'

import { SubHeader } from '@/components/sub-header'

import { LicenseForm } from './license-form'
import { LicenseTable } from './license-table'

export default function Subscription() {
  return (
    <div className="p-4">
      <SubHeader>
        You can upload your Tabby license to unlock enterprise features.
      </SubHeader>
      <div className="flex flex-col gap-8">
        <div className="grid font-bold lg:grid-cols-3">
          <div>
            <div className="text-muted-foreground">Current plan</div>
            <div className="text-3xl">Team</div>
          </div>
          <div>
            <div className="text-muted-foreground">
              Assigned / Total Seats
            </div>
            <div className="text-3xl">2 / 999</div>
          </div>
          <div>
            <div className="text-muted-foreground">Expires at</div>
            <div className="text-3xl">09/20/2222</div>
          </div>
        </div>
        <LicenseForm />
        <LicenseTable />
      </div>
    </div>
  )
}
