import type { NextApiRequest } from 'next'
import type { NextFetchEvent, NextRequest } from 'next/server'

export const initAnalytics = ({
  request,
  event
}: {
  request: NextRequest | NextApiRequest | Request
  event?: NextFetchEvent
}) => {
  const endpoint = process.env.VERCEL_URL

  return {
    track: async (eventName: string, data?: any) => {
      try {
        if (!endpoint && process.env.NODE_ENV === 'development') {
          console.log(
            `[Vercel Web Analytics] Track "${eventName}"` +
              (data ? ` with data ${JSON.stringify(data || {})}` : '')
          )
          return
        }

        const headers: { [key: string]: string } = {}
        Object.entries(request.headers).map(([key, value]) => {
          headers[key] = value
        })

        const body = {
          o: headers.referer,
          ts: new Date().getTime(),
          r: '',
          en: eventName,
          ed: data
        }

        const promise = fetch(
          `https://${process.env.VERCEL_URL}/_vercel/insights/event`,
          {
            headers: {
              'content-type': 'application/json',
              'user-agent': headers['user-agent'] as string,
              'x-forwarded-for': headers['x-forwarded-for'] as string,
              'x-va-server': '1'
            },
            body: JSON.stringify(body),
            method: 'POST'
          }
        )

        if (event) {
          event.waitUntil(promise)
        }
        {
          await promise
        }
      } catch (err) {
        console.error(err)
      }
    }
  }
}
