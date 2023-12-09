'use client'

import { useEffect, useState } from 'react'

import { graphql } from '@/lib/gql/generates'
import { useHealth } from '@/lib/hooks/use-health'
import { useAuthenticatedGraphQLQuery } from '@/lib/tabby/gql'
import {
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { CopyButton } from '@/components/copy-button'
import SlackDialog from '@/components/slack-dialog'

export default function Home() {
  return (
    <div className="p-4 lg:p-16">
      <MainPanel />
      <SlackDialog />
    </div>
  )
}

const meQuery = graphql(/* GraphQL */ `
  query MeQuery {
    me {
      authToken
    }
  }
`)

function MainPanel() {
  const { data: healthInfo } = useHealth()
  const { data } = useAuthenticatedGraphQLQuery(meQuery)
  const [origin, setOrigin] = useState('')
  useEffect(() => {
    setOrigin(new URL(window.location.href).origin)
  }, [])

  if (!healthInfo || !data) return

  return (
    <div>
      <CardHeader>
        <CardTitle>Getting Started</CardTitle>
      </CardHeader>
      <CardContent className="flex max-w-[420px] flex-col gap-2">
        <span className="flex items-center justify-between gap-2">
          <span>Endpoint URL</span>
          <span className="flex items-center">
            <Input value={origin} />
            <CopyButton value={origin} />
          </span>
        </span>

        <span className="flex items-center justify-between gap-2">
          <span>Token</span>
          <span className="flex items-center">
            <Input value={data.me.authToken} />
            <CopyButton value={data.me.authToken} />
          </span>
        </span>
      </CardContent>
      <CardFooter>
        <span>
          Use informations above for IDE extensions / plugins configuration, see{' '}
          <a
            className="underline"
            target="_blank"
            href="https://tabby.tabbyml.com/docs/extensions/configurations#server"
          >
            documentation website
          </a>{' '}
          for details
        </span>
      </CardFooter>
    </div>
  )
}
