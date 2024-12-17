import { useState } from 'react'
import Image from 'next/image'
import defaultFavicon from '@/assets/default-favicon.png'

import { cn } from '@/lib/utils'

export function SiteFavicon({
  hostname,
  className
}: {
  hostname: string
  className?: string
}) {
  const [isLoaded, setIsLoaded] = useState(false)

  const handleImageLoad = () => {
    setIsLoaded(true)
  }

  return (
    <div className="relative h-3.5 w-3.5 shrink-0">
      <Image
        src={defaultFavicon}
        alt={hostname}
        width={14}
        height={14}
        className={cn(
          'absolute left-0 top-0 z-0 h-3.5 w-3.5 rounded-full leading-none',
          className
        )}
      />
      <Image
        src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=${hostname}`}
        alt={hostname}
        width={14}
        height={14}
        className={cn(
          'relative z-10 h-3.5 w-3.5 rounded-full bg-card leading-none',
          className,
          {
            'opacity-0': !isLoaded
          }
        )}
        onLoad={handleImageLoad}
      />
    </div>
  )
}
