import React from 'react'

import { cn } from '@/lib/utils'

import { IconCheck, IconClose } from '../ui/icons'

import './steps.css'

type StepStatus = 'current' | 'finish' | 'error' | 'loading' | 'wait'

type SourceStepItem = {
  title: string
  description?: string
  icon?: React.ReactNode
  disabled?: boolean
  status?: StepStatus
}

type ComputedStepItem = SourceStepItem & {
  status: StepStatus
}

interface UseStepsReturn {
  currentStep: number
  steps: ComputedStepItem[]
  setStep: (step: number) => void
  // status for currentStep
  status?: StepStatus
}

interface StepsContextValue extends UseStepsReturn {}

interface StepsProps extends UseStepsReturn {
  children: React.ReactNode
  className?: string
}

const StepsContext = React.createContext<StepsContextValue>(
  {} as StepsContextValue
)

const StepsContextProvider: React.FC<StepsProps> = ({
  children,
  className,
  ...props
}) => {
  return (
    <StepsContext.Provider value={props}>
      <div className={cn(className)}>{children}</div>
    </StepsContext.Provider>
  )
}

const Steps = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, children, ...props }, ref) => {
  return (
    <div ref={ref} className={cn('flex gap-2', className)} {...props}>
      {children}
    </div>
  )
})
Steps.displayName = 'Steps'

interface StepItemProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'>,
    ComputedStepItem {
  index: number
}

const StepItem = React.forwardRef<HTMLDivElement, StepItemProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn('flex-1 steps-item', className)}
        data-state={props.status}
        {...props}
      >
        <div className="flex gap-2 overflow-x-hidden">
          <StepIcon status={props.status} index={props.index} />
          <div className="flex flex-col gap-1">
            <div className="leading-8 steps-item-title pr-4">{props.title}</div>
            <div>{props.description}</div>
          </div>
        </div>
      </div>
    )
  }
)
StepItem.displayName = 'StepItem'

const StepIcon: React.FC<
  Pick<ComputedStepItem, 'icon' | 'status'> & { index: number }
> = ({ icon, index, status }) => {
  const getIcon = () => {
    if (icon) {
      return icon
    }
    if (status === 'finish') {
      return <IconCheck />
    }
    if (status === 'error') {
      return <IconClose />
    }
    return index + 1
  }

  return (
    <div
      className={cn(
        'flex h-8 w-8 items-center justify-center rounded-full border',
        {
          'text-primary border-primary': status === 'finish',
          'bg-primary border-primary text-primary-foreground':
            status === 'current',
          'text-muted-foreground': status === 'wait'
        }
      )}
    >
      {getIcon()}
    </div>
  )
}

interface UseStepsOptions {
  items: SourceStepItem[]
  initialStep?: number
}

function useSteps(options: UseStepsOptions): UseStepsReturn {
  const { items } = options
  const [currentStep, setCurrentStep] = React.useState(options.initialStep || 0)

  const computedSteps = React.useMemo(() => {
    return items.map((item, index) => {
      const isPrevStep = index < currentStep
      const isCurrentStep = index === currentStep
      const computedStatusFromIndex = isPrevStep
        ? 'finish'
        : isCurrentStep
        ? 'current'
        : 'wait'
      return {
        ...item,
        status: item.status || computedStatusFromIndex
      }
    })
  }, [items, currentStep])

  return {
    currentStep,
    setStep: setCurrentStep,
    steps: computedSteps
  }
}

export type { ComputedStepItem }

export { StepsContextProvider, Steps, StepItem, useSteps }
