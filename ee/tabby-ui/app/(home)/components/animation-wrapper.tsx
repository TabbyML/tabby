import { CSSProperties } from 'react'
import { motion, Transition, UseInViewOptions, Variants } from 'framer-motion'

const cardTransition: Transition = {
  ease: 'easeOut',
  duration: 0.5
}

function getCardVariants(delay?: number): Variants {
  return {
    initial: {
      opacity: 0,
      y: 24,
      transition: cardTransition
    },
    onscreen: {
      opacity: 1,
      y: 0,
      transition: {
        ...cardTransition,
        delay
      }
    }
  }
}

interface AnimationWrapperProps {
  viewport?: UseInViewOptions
  children: React.ReactNode
  style?: CSSProperties
  className?: string
  delay?: number
}

export function AnimationWrapper({
  viewport,
  children,
  className,
  style,
  delay
}: AnimationWrapperProps) {
  return (
    <motion.div
      initial="initial"
      whileInView="onscreen"
      viewport={viewport}
      // onViewportEnter={handleEnterViewport}
      // onViewportLeave={handleLeaveViewport}
      style={style}
      className={className}
    >
      <motion.div variants={getCardVariants(delay)}>{children}</motion.div>
    </motion.div>
  )
}
