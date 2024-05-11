function createClient(createFn, target) {
  return createFn(target, {
    callable: ["init", "sendMessage"]
  });
}
function createServer(createFn, api, target) {
  const opts = {
    expose: {
      init: api.init,
      sendMessage: api.sendMessage
    }
  };
  if (target)
    return createFn(target, opts);
  const createFnWithoutTarget = createFn;
  return createFnWithoutTarget(opts);
}

export { createClient, createServer };
