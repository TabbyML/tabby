import { CSSProperties } from 'react'
import { motion, useAnimationControls, UseInViewOptions } from 'framer-motion'

import { getCardVariants } from '../constants'

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

  const handleLeaveViewport = () => {
    controls.start('offscreen')
  }

  const handleEnterViewport = () => {
    controls.set('initial')
    controls.start('onscreen')
  }

  return (
    <motion.div
      animate={controls}
      initial="initial"
      viewport={viewport}
      onViewportEnter={handleEnterViewport}
      onViewportLeave={handleLeaveViewport}
      style={style}
      className={className}
    >
      <motion.div variants={getCardVariants(delay)}>{children}</motion.div>
    </motion.div>
  )
}
