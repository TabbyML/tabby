import { CSSProperties, useEffect, useRef } from 'react'
import {
  motion,
  Transition,
  useAnimationControls,
  useInView,
  UseInViewOptions,
  Variants
} from 'framer-motion'

const cardTransition: Transition = {
  ease: 'easeOut',
  duration: 0.5
}

function getCardVariants(delay?: number): Variants {
  return {
    initial: {
      y: 24,
      opacity: 0
    },
    offscreen: {
      opacity: 0,
      transition: cardTransition,
      transitionEnd: {
        y: 24
      }
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
  const controls = useAnimationControls()
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, viewport)

  const handleLeaveViewport = () => {
    controls.start('offscreen')
  }

  const handleEnterViewport = () => {
    controls.start('onscreen')
  }

  useEffect(() => {
    // FIXME(jueliang) call to much times
    if (inView) {
      handleEnterViewport()
    } else {
      handleLeaveViewport()
    }
  }, [controls, inView])

  useEffect(() => {
    return () => {
      controls.stop()
    }
  }, [])

  return (
    <motion.div
      ref={ref}
      animate={controls}
      initial="initial"
      // whileInView='onscreen'
      // viewport={viewport}
      // onViewportEnter={handleEnterViewport}
      // onViewportLeave={handleLeaveViewport}
      style={style}
      className={className}
    >
      <motion.div variants={getCardVariants(delay)}>{children}</motion.div>
    </motion.div>
  )
}
