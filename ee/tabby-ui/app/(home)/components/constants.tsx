import { Transition, Variants } from 'framer-motion'

const cardTransition: Transition = {
  ease: 'easeOut',
  duration: 0.5
}

export const cardVariants: Variants = {
  initial: {
    opacity: 0,
    y: 24
  },
  hidden: {
    opacity: 0,
    transition: {
      ease: 'easeOut',
      duration: 0.1
    },
    transitionEnd: {
      y: 24
    }
  },
  onscreen: {
    opacity: 1,
    y: 0,
    // transition: {
    //   ...cardTransition,
    //   staggerChildren: 0.05
    // }
    transition: cardTransition
  }
}
