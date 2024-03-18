import { cn } from "@/lib/utils";

export default function AdminRegisterStep ({
  step,
  currentStep,
  children
}: {
  step: number;
  currentStep: number;
  children: React.ReactNode
}) {
  return (
    <div id={`step-${step}`} className={cn('border-l border-foreground py-8 pl-12', {
      'step-mask': step !== currentStep,
      'remote': Math.abs(currentStep - step) > 1
    })}>
      {children}
    </div>
  )
}