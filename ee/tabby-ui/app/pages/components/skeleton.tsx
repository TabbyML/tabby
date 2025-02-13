import { Skeleton } from '@/components/ui/skeleton'

export function PageSkeleton() {
  return (
    <div className="mx-auto mt-8 grid grid-cols-4 gap-2 px-4 pb-32 lg:max-w-5xl lg:px-0">
      <div className="col-span-3 space-y-12">
        <div className="w-full">
          <Skeleton className="mb-6 h-6 w-[50%]" />
          <AuthorSkeleton />
          <SectionContentSkeleton />
        </div>
        <SectionsSkeleton />
      </div>
    </div>
  )
}

export function SectionsSkeleton() {
  const sectionsCount = 3
  return (
    <div className="w-full space-y-12">
      {Array.from({ length: sectionsCount }).map((x, idx) => (
        <div className="w-full" key={idx}>
          <SectionTitleSkeleton />
          <SectionContentSkeleton />
        </div>
      ))}
    </div>
  )
}

export function SectionTitleSkeleton() {
  return (
    <div className="mb-6 w-full">
      <Skeleton className="w-[50%]" />
    </div>
  )
}

export function SectionContentSkeleton() {
  return (
    <div className="w-full space-y-3">
      <Skeleton className="w-[80%]" />
      <Skeleton className="w-[30%]" />
      <Skeleton className="w-[80%]" />
      <Skeleton className="w-[30%]" />
    </div>
  )
}

export function AuthorSkeleton() {
  return (
    <div className="my-4 flex w-full items-center gap-4">
      <Skeleton className="h-6 w-6 rounded-full" />
      <Skeleton className="w-20" />
    </div>
  )
}
