'use client'

import { ReactNode } from 'react'

import { IconCheck } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

export const LicenseTable = () => {
  return (
    <Table className="border text-center">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[40%]"></TableHead>
          {PLANS.map(({ name, pricing, limit }, i) => (
            <TableHead className="w-[20%] text-center" key={i}>
              <h1 className="py-4 text-2xl font-bold">{name}</h1>
              <p className="text-center font-semibold">{pricing}</p>
              <p className="pb-2 pt-1">{limit}</p>
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {FEATURES.map(({ name, features }, i) => (
          <FeatureList key={i} name={name} features={features} />
        ))}
      </TableBody>
    </Table>
  )
}

const FeatureList = ({
  name,
  features
}: {
  name: String
  features: Feature[]
}) => {
  return (
    <>
      <TableRow>
        <TableCell
          colSpan={4}
          className="bg-accent text-left text-accent-foreground"
        >
          {name}
        </TableCell>
      </TableRow>
      {features.map(({ name, community, team, enterprise }, i) => (
        <TableRow key={i}>
          <TableCell className="text-left">{name}</TableCell>
          <TableCell className="font-semibold">{community}</TableCell>
          <TableCell className="font-semibold">{team}</TableCell>
          <TableCell className="font-semibold text-primary">
            {enterprise}
          </TableCell>
        </TableRow>
      ))}
    </>
  )
}

interface Plan {
  name: ReactNode | String
  pricing: ReactNode | String
  limit: ReactNode | String
}

const PLANS: Plan[] = [
  {
    name: 'Community',
    pricing: '$0 per user/month',
    limit: 'Up to 5 users, single node'
  },
  {
    name: 'Team',
    pricing: '$19 per user/month',
    limit: 'Up to 30 users, up to 2 nodes'
  },
  {
    name: 'Enterprise',
    pricing: 'Contact Us',
    limit: 'Customized, billed annually'
  }
]

interface Feature {
  name: ReactNode | String
  community: ReactNode | String
  team: ReactNode | String
  enterprise: ReactNode | String
}

interface FeatureGroup {
  name: String
  features: Feature[]
}

const checked = <IconCheck className="mx-auto" />
const dashed = '–'

const FEATURES: FeatureGroup[] = [
  {
    name: 'Features',
    features: [
      {
        name: 'User count',
        community: 'Up to 5',
        team: 'Up to 30',
        enterprise: 'Unlimited'
      },
      {
        name: 'Node count',
        community: dashed,
        team: 'Up to 2',
        enterprise: 'Unlimited'
      },
      {
        name: 'Secure Access',
        community: checked,
        team: checked,
        enterprise: checked
      },
      {
        name: 'Toggle IDE / Extensions telemetry',
        community: dashed,
        team: dashed,
        enterprise: checked
      },
      {
        name: 'Authentication Domain',
        community: dashed,
        team: dashed,
        enterprise: checked
      },
      {
        name: 'Single Sign-On (SSO)',
        community: dashed,
        team: dashed,
        enterprise: checked
      }
    ]
  },
  {
    name: 'Bespoke',
    features: [
      {
        name: 'Support',
        community: 'Community',
        team: 'Email',
        enterprise: 'Dedicated Slack channel'
      },
      {
        name: 'Roadmap prioritization',
        community: dashed,
        team: dashed,
        enterprise: checked
      }
    ]
  }
]
