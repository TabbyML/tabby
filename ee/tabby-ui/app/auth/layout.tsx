import { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Authentication',
  description: 'Authentication forms built using the components.'
}

export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <div className="flex flex-col items-center justify-center flex-1">
        {children}
    </div>
  )
}
