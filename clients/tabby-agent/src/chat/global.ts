export let mutexAbortController: AbortController | undefined = undefined;

// reset global mutexAbortController to undefined
export const resetMutexAbortController = () => {
  mutexAbortController = undefined;
};

// initialize global mutexAbortController
export const initMutexAbortController = () => {
  if (!mutexAbortController) {
    mutexAbortController = new AbortController();
  }
  return mutexAbortController;
};
